"""CPU-only safety, audio and compatibility tests; no models/downloads required."""
import importlib.util
import builtins
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "benchmarks/architecture_compare"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("f5_probe", DIRECTORY / "f5_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_plan_and_cpu_guard_never_access_cuda(monkeypatch, capsys):
    monkeypatch.setattr(probe, "gpu_snapshot", lambda: pytest.fail("GPU accessed"))
    assert probe.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert not plan["gpu_touched"] and not plan["neural_streaming"]
    probe.Guard(SimpleNamespace(device="cpu"))()


def test_unapproved_cuda_is_rejected_before_local_files_or_models(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "gpu_snapshot", lambda: pytest.fail("GPU accessed"))
    monkeypatch.setattr(probe, "validate_files", lambda _: pytest.fail("local files accessed"))
    window = tmp_path / "window.json"
    window.write_text(json.dumps({"owner_approved": False}))
    with pytest.raises(ValueError, match="owner-approved"):
        probe.execute(SimpleNamespace(device="cuda", window_file=window, candidate="f5-ptbr-vocos"))


def test_no_window_is_not_implicit_authorization():
    with pytest.raises(ValueError, match="owner-approved"):
        probe.execute(SimpleNamespace(device="cuda", window_file=None))


def test_only_uniform_raw_or_ema_checkpoints_are_adapted():
    raw = {"transformer.x": object(), "mel_spec.mel_stft.mel_scale.fb": object()}
    state, ignored = probe.ema_state(raw)
    assert state == {"transformer.x": raw["transformer.x"]}
    assert ignored == ["mel_spec.mel_stft.mel_scale.fb"]
    ema = {"ema_model.transformer.x": raw["transformer.x"], "step": 4}
    state, ignored = probe.ema_state(ema)
    assert state == {"transformer.x": raw["transformer.x"]} and ignored == ["step"]
    with pytest.raises(ValueError, match="mixed"):
        probe.ema_state({"ema_model.transformer.x": 1, "transformer.y": 2})


def test_duration_refuses_truncation_and_text_normalization_keeps_accents():
    assert probe.normalize_text("  AMANHA\u0303, JOÃO! ", "lowercase") == "amanhã, joão!"
    assert probe.normalize_text("R$ 1.247,83 API", "verbatim") == "R$ 1.247,83 API"
    with pytest.raises(ValueError, match="do not truncate"):
        probe.duration_frames(1000, "curto", "uma frase muito longa" * 50, 30)
    with pytest.raises(ValueError, match="required"):
        probe.duration_frames(1000, "", "fala", 30)


def test_full_utterance_pcm_metrics_do_not_call_silent_prefix_useful():
    values = np.concatenate([np.zeros(2400), np.full(2400, .2)]).astype(np.float32)
    metrics = probe.useful_metrics(values, .5, np)
    assert metrics["first_pcm_ms"] == 500
    assert metrics["first_useful_pcm_available_ms"] == 500
    assert metrics["onset_audio_offset_ms"] == 100
    assert metrics["estimated_first_audible_playback_ms"] == 600
    silent = probe.useful_metrics(np.zeros(4800), .5, np)
    assert silent["first_useful_pcm_available_ms"] is None
    invalid = probe.useful_metrics(np.array([1.1]), .5, np)
    assert "pcm_metrics_unavailable" in invalid


def test_wave_outputs_preserve_float_overrange_and_refuse_overwrite(tmp_path):
    sf = pytest.importorskip("soundfile")
    values = np.array([0., .12, 1.2, -.4], dtype=np.float32)
    metadata = probe.save_wave(tmp_path, "native.wav", values, sf, np)
    decoded, rate = sf.read(tmp_path / "native.wav", dtype="float32")
    np.testing.assert_array_equal(decoded, values)
    assert rate == 24000 and metadata["samples_outside_unit_range"] == 1
    assert (tmp_path / "native.wav").stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        probe.save_wave(tmp_path, "native.wav", values, sf, np)


def test_integrity_failure_stops_before_model_load(monkeypatch, tmp_path):
    monkeypatch.setattr(probe.subprocess, "run", lambda command, **_: SimpleNamespace(
        stdout=probe.SOURCE_COMMIT if "rev-parse" in command else ""))
    source = tmp_path / "source"
    source.mkdir()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"vocab_sha256": "0" * 64}))
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(" \na\n")
    with pytest.raises(ValueError, match="SHA256: vocab"):
        probe.validate_files(SimpleNamespace(source_dir=source, receipt=receipt,
                                            vocab=vocab, vocoder_dir=tmp_path))


def test_import_failure_is_preserved_in_private_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "validate_files", lambda _: (tmp_path, tmp_path / "ref.wav", "fala", {}))
    original_import = builtins.__import__
    def fail_soundfile(name, *args, **kwargs):
        if name == "soundfile":
            raise ImportError("dependency unavailable")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fail_soundfile)
    for name in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "WANDB_MODE"):
        monkeypatch.setenv(name, "")
    output = tmp_path / "out"
    with pytest.raises(ImportError, match="dependency unavailable"):
        probe.execute(SimpleNamespace(device="cpu", candidate="f5", action="roundtrip", output_dir=output))
    result = json.loads((output / "results.json").read_text())
    assert result["success"] is False and result["failure_stage"] == "imports"
    assert result["failures"][0]["type"] == "ImportError"
    assert (output / "results.json").stat().st_mode & 0o777 == 0o600
