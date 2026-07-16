from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import AppSettings, MODEL_ID
from app.main import create_app
from app.model_service import (
    GenerationTimeoutError,
    GpuOutOfMemoryError,
    QueueTimeoutError,
    ServiceNotReadyError,
    UnknownVoiceError,
    VoiceUnavailableError,
)
from tests.helpers import wav_bytes


REQUIRED_AUDIO_HEADERS = {
    "X-Aegis-TTS-Model",
    "X-Aegis-TTS-Voice",
    "X-Aegis-TTS-Seed",
    "X-Aegis-TTS-Generation-Seconds",
    "X-Aegis-TTS-Audio-Duration-Seconds",
    "X-Aegis-TTS-RTF",
    "X-Aegis-TTS-Queue-Wait-Seconds",
    "X-Aegis-TTS-Sample-Rate",
    "X-Aegis-TTS-Dtype",
    "X-Aegis-TTS-VRAM-Before-MiB",
    "X-Aegis-TTS-VRAM-After-MiB",
    "X-Aegis-TTS-VRAM-Peak-MiB",
}


@dataclass
class _FakeResult:
    wav_bytes: bytes
    headers: dict[str, str]


class FakeService:
    """A no-model service double accepted by ``create_app``."""

    def __init__(
        self,
        *,
        ready: bool = True,
        failure: Exception | None = None,
        model_id: str = MODEL_ID,
    ) -> None:
        self.is_ready = ready
        self.failure = failure
        self.model_id = model_id
        self.requests: list[Any] = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "status": "ready" if self.is_ready else "loading_model",
            "model": self.model_id,
            "model_loaded": self.is_ready,
            "device": "cuda:0",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "voices": {"giseli-10s": "ready" if self.is_ready else "pending"},
            "gpu": {"cuda_available": self.is_ready},
            "load_seconds": 1.25 if self.is_ready else None,
            "prompt_seconds": {"giseli-10s": 2.3} if self.is_ready else {},
            "errors": [],
        }

    def info(self) -> dict[str, Any]:
        return {
            "service": "aegis-tts",
            "model": self.model_id,
            "device": "cuda:0",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "max_text_length": 1000,
            "default_voice": "giseli-10s",
        }

    def voices_response(self) -> dict[str, Any]:
        return {
            "default_voice": "giseli-10s",
            "voices": [
                {
                    "id": "giseli-10s",
                    "display_name": "Giseli — referência de 10 segundos",
                    "ready": self.is_ready,
                    "reference_duration_seconds": 10.2,
                    "reference_sample_rate": 32000,
                }
            ],
        }

    async def synthesize(self, request: Any) -> _FakeResult:
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return _FakeResult(
            wav_bytes=wav_bytes(),
            headers={
                "X-Aegis-TTS-Model": self.model_id,
                "X-Aegis-TTS-Voice": request.voice or "giseli-10s",
                "X-Aegis-TTS-Seed": str(request.seed if request.seed is not None else 42),
                "X-Aegis-TTS-Generation-Seconds": "0.12",
                "X-Aegis-TTS-Audio-Duration-Seconds": "0.10",
                "X-Aegis-TTS-RTF": "1.2",
                "X-Aegis-TTS-Queue-Wait-Seconds": "0.0",
                "X-Aegis-TTS-Sample-Rate": "24000",
                "X-Aegis-TTS-Dtype": "bfloat16",
                "X-Aegis-TTS-VRAM-Before-MiB": "100.0",
                "X-Aegis-TTS-VRAM-After-MiB": "110.0",
                "X-Aegis-TTS-VRAM-Peak-MiB": "120.0",
            },
        )


def _client(
    service: FakeService,
    *,
    api_key: str | None = None,
    max_text_length: int = 1000,
    model_id: str = MODEL_ID,
) -> TestClient:
    return TestClient(
        create_app(
            AppSettings(model_id=model_id, api_key=api_key, max_text_length=max_text_length),
            service,
        )
    )


def _valid_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": "Olá, Pedro. O serviço de voz está funcionando corretamente.",
        "voice": "giseli-10s",
        "language": "pt-BR",
        "seed": 42,
        "generation_parameters": {
            "temperature": 0.8,
            "top_p": 0.9,
            "do_sample": True,
            "max_new_tokens": 1024,
        },
    }
    payload.update(overrides)
    return payload


def test_liveness_is_always_public_while_readiness_tracks_the_service() -> None:
    service = FakeService(ready=False)
    with _client(service) as client:
        assert client.get("/health/live").json() == {"status": "live"}
        health = client.get("/health")
        ready = client.get("/health/ready")

    assert health.status_code == 200
    assert health.json()["status"] == "loading_model"
    assert ready.status_code == 503
    assert service.started and service.stopped


def test_ready_health_info_and_voice_metadata_do_not_expose_references() -> None:
    with _client(FakeService()) as client:
        ready = client.get("/health/ready")
        info = client.get("/v1/info")
        voices = client.get("/v1/voices")

    assert ready.status_code == 200
    assert ready.json()["model"] == MODEL_ID
    assert info.status_code == 200
    assert info.json()["default_voice"] == "giseli-10s"
    assert voices.status_code == 200
    first_voice = voices.json()["voices"][0]
    assert set(first_voice) == {
        "id",
        "display_name",
        "ready",
        "reference_duration_seconds",
        "reference_sample_rate",
    }
    serialized = str(voices.json()).lower()
    assert "transcript" not in serialized
    assert "audio_path" not in serialized
    assert "reference.wav" not in serialized


def test_synthesis_returns_a_pcm_wav_and_all_metrics_headers() -> None:
    service = FakeService()
    with _client(service) as client:
        response = client.post("/v1/synthesize", json=_valid_payload())

    assert response.status_code == 200
    assert response.headers["content-type"].split(";", 1)[0] == "audio/wav"
    assert response.content.startswith(b"RIFF")
    assert all(response.headers.get(name) is not None for name in REQUIRED_AUDIO_HEADERS)
    assert response.headers["X-Aegis-TTS-Voice"] == "giseli-10s"
    assert service.requests[0].language == "Portuguese"
    assert service.requests[0].generation_parameters.max_new_tokens == 1024


def test_info_health_and_headers_report_the_effectively_loaded_0_6b_model() -> None:
    model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    service = FakeService(model_id=model_id)
    with _client(service, model_id=model_id) as client:
        ready = client.get("/health/ready")
        info = client.get("/v1/info")
        response = client.post("/v1/synthesize", json=_valid_payload())

    assert ready.json()["model"] == model_id
    assert info.json()["model"] == model_id
    assert response.headers["X-Aegis-TTS-Model"] == model_id


def test_synthesis_requires_key_only_when_one_is_configured() -> None:
    service = FakeService()
    with _client(service, api_key="local-secret") as client:
        assert client.get("/health").status_code == 200
        missing = client.post("/v1/synthesize", json=_valid_payload())
        wrong = client.post("/v1/synthesize", json=_valid_payload(), headers={"X-API-Key": "wrong"})
        allowed = client.post("/v1/synthesize", json=_valid_payload(), headers={"X-API-Key": "local-secret"})

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert allowed.status_code == 200
    assert "local-secret" not in missing.text


def test_invalid_payload_and_unknown_parameters_are_rejected_before_inference() -> None:
    service = FakeService()
    with _client(service) as client:
        invalid_voice = client.post("/v1/synthesize", json=_valid_payload(voice="../invalid"))
        unknown_parameter = client.post(
            "/v1/synthesize",
            json=_valid_payload(generation_parameters={"temperature": 0.8, "speed": 1.2}),
        )

    assert invalid_voice.status_code == 422
    assert unknown_parameter.status_code == 422
    assert service.requests == []


def test_deployment_can_set_a_stricter_text_limit_than_the_public_1000_character_ceiling() -> None:
    service = FakeService()
    with _client(service, max_text_length=5) as client:
        response = client.post("/v1/synthesize", json=_valid_payload(text="123456"))

    assert response.status_code == 422
    assert "5 caracteres" in response.json()["detail"]
    assert service.requests == []


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (UnknownVoiceError("voice id desconhecido"), 422),
        (VoiceUnavailableError("prompt não está pronto"), 503),
        (ServiceNotReadyError("modelo ainda carrega"), 503),
        (GpuOutOfMemoryError("VRAM insuficiente"), 503),
        (QueueTimeoutError("fila excedeu o limite"), 504),
        (GenerationTimeoutError("geração excedeu o limite"), 504),
    ],
)
def test_service_failures_are_mapped_to_stable_http_statuses(
    failure: Exception,
    expected_status: int,
) -> None:
    with _client(FakeService(failure=failure)) as client:
        response = client.post("/v1/synthesize", json=_valid_payload())

    assert response.status_code == expected_status
    assert "traceback" not in response.text.lower()


def test_index_is_dependency_free_and_docs_stay_available() -> None:
    with _client(FakeService()) as client:
        page = client.get("/")
        docs = client.get("/docs")

    assert page.status_code == 200
    assert "Aegis TTS" in page.text
    assert docs.status_code == 200
