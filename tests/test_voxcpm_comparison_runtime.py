"""CPU-only tests of the isolated adapter's gates and streaming serialization."""
from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


DIRECTORY = Path(__file__).resolve().parents[1] / "benchmarks/architecture_compare"
spec = importlib.util.spec_from_file_location("benchmark", DIRECTORY / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
sys.modules["benchmark"] = benchmark
spec.loader.exec_module(benchmark)
spec = importlib.util.spec_from_file_location("voxcpm_comparison_runtime", DIRECTORY / "voxcpm_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_plan_and_unapproved_execution_never_probe_gpu(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(runtime, "gpu_snapshot", lambda: pytest.fail("GPU accessed"))
    assert runtime.main([]) == 0
    assert json.loads(capsys.readouterr().out)["gpu_touched"] is False
    window = tmp_path / "window.json"
    window.write_text(json.dumps({"owner_approved": False}))
    with pytest.raises(ValueError, match="owner-approved"):
        runtime.main(["--execute", "--window-file", str(window), "--source-dir", "missing",
                      "--model-dir", "missing", "--reference-wav", "missing",
                      "--reference-text-file", "missing", "--output-dir", str(tmp_path / "out")])


def test_quantization_preserves_bounds_and_refuses_invalid_audio():
    values = np.array([-1, -.5, 0, .5, 1], dtype=np.float32)
    pcm = runtime.pcm16_bytes(values, np)
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [-32767, -16384, 0, 16384, 32767]
    for invalid in [np.array([np.nan]), np.array([np.inf]), np.array([1.01]), np.zeros((1, 4))]:
        with pytest.raises(ValueError):
            runtime.pcm16_bytes(invalid, np)


@pytest.mark.parametrize("alternate", ["audiovae.safetensors", "pytorch_model.bin"])
def test_unpinned_decoder_precedence_is_rejected_before_model_load(monkeypatch, tmp_path, alternate):
    source, model = tmp_path / "src", tmp_path / "model"
    source.mkdir()
    model.mkdir()
    (model / alternate).write_bytes(b"unpinned")
    monkeypatch.setattr(runtime.subprocess, "run", lambda args, **kwargs:
                        SimpleNamespace(stdout=runtime.SOURCE_COMMIT if "rev-parse" in args else ""))
    with pytest.raises(ValueError, match="alternate unpinned"):
        runtime.validate_local_files(SimpleNamespace(source_dir=source, model_dir=model))


def test_compile_flag_cannot_misreport_skipped_backend():
    with pytest.raises(SystemExit) as error:
        runtime.main(["--optimize"])
    assert error.value.code == 2


def make_runtime(tmp_path, model):
    cuda = SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
                           max_memory_allocated=lambda: 1048576, max_memory_reserved=lambda: 2097152)
    torch = SimpleNamespace(cuda=cuda, inference_mode=nullcontext)
    args = SimpleNamespace(max_text_chars=1000, cfg_value=2, timesteps=10, max_len=512,
                           seed=1, max_audio_seconds=60, save_float=True)
    instance = runtime.Runtime(args, {}, tmp_path, model, Path("reference.wav"), "Referência.", torch, np)
    instance.guard = lambda **kwargs: None
    handler = SimpleNamespace(wfile=io.BytesIO(), send_response=lambda *args: None,
                              send_header=lambda *args: None, end_headers=lambda: None)
    return instance, handler


def test_native_chunks_are_flushed_incrementally_and_record_seed_and_float_output(tmp_path):
    state = {}
    instance, handler = make_runtime(tmp_path, None)

    class Model:
        def generate_streaming(self, **kwargs):
            state.update(kwargs)
            yield np.full(480, .25, dtype=np.float32)
            assert len(handler.wfile.getvalue()) > 960  # first chunk already serialized
            assert not handler.wfile.getvalue().endswith(b"0\r\n\r\n")
            yield np.full(480, -.25, dtype=np.float32)

    instance.model = Model()
    instance.synthesize({"input": "Teste.", "stream": True, "response_format": "pcm", "seed": 72}, handler)
    assert state["seed"] == 72
    assert state["normalize"] is False and state["denoise"] is False
    assert handler.wfile.getvalue().endswith(b"0\r\n\r\n")
    metadata = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert metadata["effective_seed"] == 72
    assert metadata["audio_seconds"] == .02
    assert "not verified" in metadata["semantic_completion"]
    assert np.load(next(tmp_path.glob("*.npy"))).shape == (960,)


def test_failed_generation_never_writes_successful_terminator(tmp_path):
    class BrokenModel:
        def generate_streaming(self, **kwargs):
            yield np.zeros(480, dtype=np.float32)
            raise RuntimeError("simulated generation failure")

    instance, handler = make_runtime(tmp_path, BrokenModel())
    with pytest.raises(RuntimeError, match="simulated"):
        instance.synthesize({"input": "Teste.", "stream": True, "response_format": "pcm"}, handler)
    assert not handler.wfile.getvalue().endswith(b"0\r\n\r\n")
    assert not list(tmp_path.glob("*.json"))
