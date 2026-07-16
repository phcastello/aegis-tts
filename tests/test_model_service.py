from __future__ import annotations

import asyncio
from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading
import wave

import numpy as np
import pytest

import app.model_service as model_module
from app.config import AppSettings, MODEL_ID
from app.metrics import GpuMemory
from app.model_service import (
    GenerationTimeoutError,
    GpuOutOfMemoryError,
    ModelService,
    QueueTimeoutError,
    retain_samples,
)
from app.reference_repository import ReferenceRepository
from app.schemas import SynthesisRequest
from tests.conftest import ReferenceLayout


class FakeMetrics:
    def __init__(self, *, cuda_available: bool = True) -> None:
        self.available = cuda_available
        self.empty_cache_calls = 0
        self.reset_peak_calls = 0

    def cuda_available(self) -> bool:
        return self.available

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def reset_peak(self) -> None:
        self.reset_peak_calls += 1

    def memory(self) -> GpuMemory:
        return GpuMemory(allocated_mib=128.0, reserved_mib=256.0, peak_allocated_mib=384.0)

    def gpu_info(self) -> dict[str, object]:
        return {"cuda_available": self.available, "process": self.memory().as_dict()}


class FakeQwenModel:
    def __init__(self, *, gate: threading.Event | None = None, raises: Exception | None = None) -> None:
        self.gate = gate
        self.raises = raises
        self.prompt_calls: list[dict[str, object]] = []
        self.generation_calls: list[dict[str, object]] = []
        self.started_generation = threading.Event()
        self.finished_generation = threading.Event()
        self._active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def create_voice_clone_prompt(self, **kwargs: object) -> dict[str, object]:
        self.prompt_calls.append(kwargs)
        return {"cached_prompt": len(self.prompt_calls), "audio": kwargs["ref_audio"]}

    def generate_voice_clone(self, **kwargs: object) -> tuple[list[np.ndarray], int]:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        self.started_generation.set()
        try:
            if self.gate is not None:
                if not self.gate.wait(timeout=5):
                    raise RuntimeError("test generation gate did not open")
            if self.raises is not None:
                raise self.raises
            self.generation_calls.append(kwargs)
            return [np.full(2400, 0.1, dtype=np.float32)], 24_000
        finally:
            with self._lock:
                self._active -= 1
            self.finished_generation.set()


def _make_settings(
    layout: ReferenceLayout,
    *,
    model_id: str = MODEL_ID,
    queue_timeout: float = 1.0,
    generation_timeout: float = 5.0,
    save_samples: bool = False,
    max_samples: int = 100,
    cache_implementation: str | None = None,
) -> AppSettings:
    model_dir = layout.root / "models" / model_id.rsplit("/", maxsplit=1)[-1]
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"test-only-placeholder")
    return AppSettings(
        model_id=model_id,
        voices_config=layout.config_path,
        voices_dir=layout.root / "voices",
        model_dir=model_dir,
        samples_dir=layout.root / "samples",
        cache_dir=layout.root / "cache",
        queue_timeout_seconds=queue_timeout,
        generation_timeout_seconds=generation_timeout,
        save_samples=save_samples,
        max_samples=max_samples,
        cache_implementation=cache_implementation,
    )


def _request(voice: str = "giseli-10s") -> SynthesisRequest:
    return SynthesisRequest(
        text="Olá, Pedro. Este é um teste local.",
        voice=voice,
        language="pt-BR",
        seed=42,
        generation_parameters={
            "temperature": 0.8,
            "top_p": 0.9,
            "do_sample": True,
            "max_new_tokens": 256,
        },
    )


async def _started(service: ModelService) -> None:
    await service.start()
    assert service.health_snapshot()["status"] == "ready"


def _service(
    layout: ReferenceLayout,
    monkeypatch: pytest.MonkeyPatch,
    model: FakeQwenModel,
    **settings_kwargs: object,
) -> tuple[ModelService, FakeMetrics, list[tuple[tuple[object, ...], dict[str, object]]]]:
    # The unit suite must not make queue assertions depend on PyTorch's cold
    # import. Production imports it during model startup, before requests.
    monkeypatch.setattr(model_module, "_seed_everything", lambda seed: None)
    settings = _make_settings(layout, **settings_kwargs)
    factory_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def factory(*args: object, **kwargs: object) -> FakeQwenModel:
        factory_calls.append((args, kwargs))
        return model

    # The suite must not import PyTorch or need a CUDA device merely to test
    # prompt caching and queue behavior.
    bf16 = object()
    fake_torch = SimpleNamespace(bfloat16=bf16, float16=object())
    monkeypatch.setattr(model_module, "_select_torch_dtype", lambda requested: (fake_torch, bf16))
    service = ModelService(settings, ReferenceRepository(settings), model_factory=factory)
    metrics = FakeMetrics()
    service._metrics = metrics  # injection point intentionally limited to tests
    return service, metrics, factory_calls


@pytest.mark.asyncio
async def test_model_loads_once_and_caches_full_clone_prompts(
    reference_layout: ReferenceLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeQwenModel()
    service, _metrics, factory_calls = _service(reference_layout, monkeypatch, model)
    try:
        await _started(service)
        await service.start()

        assert len(factory_calls) == 1
        assert len(model.prompt_calls) == 2
        assert set(service.clone_prompts) == {"giseli-10s", "giseli-34s"}
        assert all(call["x_vector_only_mode"] is False for call in model.prompt_calls)
        assert all(call["ref_text"] for call in model.prompt_calls)
        assert factory_calls[0][1]["local_files_only"] is True
        assert factory_calls[0][1]["device_map"] == "cuda:0"
        assert factory_calls[0][1]["attn_implementation"] == "sdpa"

        result = await service.synthesize(_request())
        assert result.wav_bytes.startswith(b"RIFF")
        with wave.open(BytesIO(result.wav_bytes), "rb") as audio:
            assert audio.getnchannels() == 1
            assert audio.getsampwidth() == 2
            assert audio.getframerate() == 24_000
        assert model.generation_calls[0]["language"] == "Portuguese"
        assert model.generation_calls[0]["voice_clone_prompt"] is service.clone_prompts["giseli-10s"]
        assert model.generation_calls[0]["non_streaming_mode"] is True

        default_result = await service.synthesize(SynthesisRequest(text="Use a voz padrão."))
        assert default_result.headers["X-Aegis-TTS-Voice"] == "giseli-10s"
        assert len(model.prompt_calls) == 2
        assert not Path(service.settings.samples_dir).exists()
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_0_6b_model_identity_flows_to_health_info_and_response_headers(
    reference_layout: ReferenceLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    model = FakeQwenModel()
    service, _metrics, _calls = _service(reference_layout, monkeypatch, model, model_id=model_id)
    try:
        await _started(service)
        result = await service.synthesize(_request("giseli-34s"))

        assert service.health_snapshot()["model"] == model_id
        assert service.info()["model"] == model_id
        assert result.headers["X-Aegis-TTS-Model"] == model_id
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_static_cache_is_forwarded_only_when_explicitly_enabled(
    reference_layout: ReferenceLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeQwenModel()
    service, _metrics, _factory_calls = _service(
        reference_layout, monkeypatch, model, cache_implementation="static"
    )
    try:
        await _started(service)
        await service.synthesize(_request())

        assert model.generation_calls[0]["use_cache"] is True
        assert model.generation_calls[0]["cache_implementation"] == "static"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_missing_model_never_invokes_a_factory_or_downloads(
    reference_layout: ReferenceLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppSettings(
        voices_config=reference_layout.config_path,
        voices_dir=reference_layout.root / "voices",
        model_dir=reference_layout.root / "models" / "absent",
    )
    calls = 0

    def factory(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return object()

    service = ModelService(settings, ReferenceRepository(settings), model_factory=factory)
    service._metrics = FakeMetrics()
    try:
        await service.start()
        health = service.health_snapshot()
        assert calls == 0
        assert health["status"] == "error"
        assert health["model_loaded"] is False
        assert any("modelo local ausente" in error for error in health["errors"])
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_only_one_generation_runs_at_a_time_and_second_waits_in_queue(
    reference_layout: ReferenceLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = threading.Event()
    model = FakeQwenModel(gate=gate)
    service, _metrics, _calls = _service(reference_layout, monkeypatch, model, queue_timeout=2.0, generation_timeout=3.0)
    try:
        await _started(service)
        first = asyncio.create_task(service.synthesize(_request()))
        assert await asyncio.to_thread(model.started_generation.wait, 1.0)
        second = asyncio.create_task(service.synthesize(_request("giseli-34s")))
        await asyncio.sleep(0.05)
        assert model.max_active == 1
        assert len(model.generation_calls) == 0
        assert service._queue_waiters >= 1

        gate.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert model.max_active == 1
        assert len(model.generation_calls) == 2
        assert float(second_result.headers["X-Aegis-TTS-Queue-Wait-Seconds"]) > 0
        assert first_result.headers["X-Aegis-TTS-Voice"] == "giseli-10s"
    finally:
        gate.set()
        await service.stop()


@pytest.mark.asyncio
async def test_queue_timeout_does_not_start_a_second_cuda_generation(
    reference_layout: ReferenceLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = threading.Event()
    model = FakeQwenModel(gate=gate)
    service, _metrics, _calls = _service(reference_layout, monkeypatch, model, queue_timeout=0.05, generation_timeout=3.0)
    try:
        await _started(service)
        first = asyncio.create_task(service.synthesize(_request()))
        assert await asyncio.to_thread(model.started_generation.wait, 1.0)
        with pytest.raises(QueueTimeoutError):
            await service.synthesize(_request("giseli-34s"))
        assert model.max_active == 1
        gate.set()
        await first
    finally:
        gate.set()
        await service.stop()


@pytest.mark.asyncio
async def test_generation_timeout_keeps_gpu_lock_until_sync_work_really_finishes(
    reference_layout: ReferenceLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = threading.Event()
    model = FakeQwenModel(gate=gate)
    service, _metrics, _calls = _service(reference_layout, monkeypatch, model, queue_timeout=0.05, generation_timeout=0.05)
    try:
        await _started(service)
        with pytest.raises(GenerationTimeoutError):
            await service.synthesize(_request())
        assert service._inference_lock.locked()
        with pytest.raises(QueueTimeoutError):
            await service.synthesize(_request("giseli-34s"))
        assert model.max_active == 1

        gate.set()
        assert await asyncio.to_thread(model.finished_generation.wait, 1.0)
        for _ in range(20):
            if not service._inference_lock.locked():
                break
            await asyncio.sleep(0.01)
        assert not service._inference_lock.locked()
    finally:
        gate.set()
        await service.stop()


@pytest.mark.asyncio
async def test_simulated_cuda_oom_clears_unused_cache_and_preserves_service(
    reference_layout: ReferenceLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeQwenModel(raises=RuntimeError("simulated cuda oom"))
    service, metrics, _calls = _service(reference_layout, monkeypatch, model)
    monkeypatch.setattr(model_module, "_is_cuda_oom", lambda error: True)
    try:
        await _started(service)
        with pytest.raises(GpuOutOfMemoryError, match="VRAM insuficiente"):
            await service.synthesize(_request())
        assert metrics.empty_cache_calls == 1
        assert service.health_snapshot()["model_loaded"] is True
        assert service._inference_lock.locked() is False
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_samples_are_opt_in_bounded_and_do_not_store_plaintext(
    reference_layout: ReferenceLayout,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeQwenModel()
    service, _metrics, _calls = _service(
        reference_layout,
        monkeypatch,
        model,
        save_samples=True,
        max_samples=1,
    )
    try:
        await _started(service)
        await service.synthesize(_request())
        await service.synthesize(_request("giseli-34s"))

        samples = Path(service.settings.samples_dir)
        wavs = list(samples.glob("*.wav"))
        metadata_files = list(samples.glob("*.json"))
        assert len(wavs) == len(metadata_files) == 1
        metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
        assert metadata["voice"] == "giseli-34s"
        assert "text" not in metadata
        assert len(metadata["text_sha256"]) == 64
    finally:
        await service.stop()


def test_sample_retention_removes_only_oldest_root_level_sample_pairs(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    samples.mkdir()
    created: list[Path] = []
    for index in range(4):
        wav = samples / f"sample-{index}.wav"
        metadata = wav.with_suffix(".json")
        wav.write_bytes(b"RIFFtest")
        metadata.write_text("{}", encoding="utf-8")
        # Stable mtimes make oldest-first behavior deterministic on all filesystems.
        import os

        os.utime(wav, ns=(index + 1, index + 1))
        created.append(wav)
    nested = samples / "smoke" / "keep.wav"
    nested.parent.mkdir()
    nested.write_bytes(b"RIFFkeep")
    note = samples / "keep.txt"
    note.write_text("never a generated WAV", encoding="utf-8")

    retain_samples(samples, max_samples=2)

    assert not created[0].exists()
    assert not created[0].with_suffix(".json").exists()
    assert not created[1].exists()
    assert created[2].exists()
    assert created[3].exists()
    assert nested.exists()
    assert note.exists()


def test_bfloat16_falls_back_explicitly_to_float16_when_cuda_lacks_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bfloat16 = object()
    float16 = object()
    fake_torch = SimpleNamespace(
        bfloat16=bfloat16,
        float16=float16,
        cuda=SimpleNamespace(is_bf16_supported=lambda: False),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    torch, dtype = model_module._select_torch_dtype("bfloat16")

    assert torch is fake_torch
    assert dtype is float16


def test_qwen_loader_forces_offline_mode_even_when_called_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeQwen3TTSModel:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs: object) -> object:
            calls.append((model_path, kwargs))
            return object()

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setitem(
        __import__("sys").modules,
        "qwen_tts",
        SimpleNamespace(Qwen3TTSModel=FakeQwen3TTSModel),
    )

    model_module._load_qwen_model("/models/Qwen3-TTS-12Hz-1.7B-Base", local_files_only=True)

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert calls == [
        (
            "/models/Qwen3-TTS-12Hz-1.7B-Base",
            {"local_files_only": True},
        )
    ]
