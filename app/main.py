"""HTTP surface for the independent, local Aegis TTS service."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
from typing import Annotated, Any, AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import AppSettings, load_settings
from .model_service import (
    GenerationTimeoutError,
    GpuOutOfMemoryError,
    ModelService,
    QueueTimeoutError,
    ServiceError,
    ServiceNotReadyError,
    UnknownVoiceError,
    VoiceUnavailableError,
)
from .reference_repository import ReferenceRepository
from .schemas import HealthResponse, InfoResponse, SynthesisRequest, VoicesResponse
from .security import require_api_key
from .ui import render_ui


LOGGER = logging.getLogger("aegis_tts.api")


def create_app(
    settings: AppSettings | None = None,
    service: ModelService | Any | None = None,
) -> FastAPI:
    """Create an API app, allowing a fake service in GPU-free unit tests."""

    active_settings = settings or load_settings()
    logging.basicConfig(
        level=getattr(logging, active_settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    active_service = service or ModelService(
        active_settings,
        ReferenceRepository(active_settings),
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.service = active_service
        application.state.settings = active_settings
        startup_task: asyncio.Task[Any] | None = None
        try:
            background = getattr(active_service, "start_background", None)
            if callable(background):
                value = background()
                if isinstance(value, asyncio.Task):
                    startup_task = value
            else:
                start = getattr(active_service, "start", None)
                if callable(start):
                    value = start()
                    if hasattr(value, "__await__"):
                        startup_task = asyncio.create_task(value)
            yield
        finally:
            stop = getattr(active_service, "stop", None)
            if callable(stop):
                value = stop()
                if hasattr(value, "__await__"):
                    await value
            elif startup_task and not startup_task.done():
                # ModelService.stop() is the normal cancellation-safe path. A
                # test double without it should still not leak a background task.
                startup_task.cancel()
                await asyncio.gather(startup_task, return_exceptions=True)

    app = FastAPI(
        title="Aegis TTS",
        version="1.0.0",
        description="API local para síntese por clone de voz usando Qwen3-TTS Base.",
        lifespan=lifespan,
    )
    app.state.service = active_service
    app.state.settings = active_settings

    if active_settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(active_settings.allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
        )

    def current_service() -> Any:
        return app.state.service

    async def authorize(
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> None:
        require_api_key(active_settings, x_api_key)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return render_ui()

    @app.get("/health/live")
    async def health_live() -> dict[str, str]:
        # A liveness probe must stay successful even while CUDA/model startup is
        # pending or degraded, otherwise Docker loses useful diagnostics.
        return {"status": "live"}

    @app.get("/health", response_model=HealthResponse)
    async def health(service_instance: Any = Depends(current_service)) -> dict[str, Any]:
        return _health_snapshot(service_instance, active_settings)

    @app.get("/health/ready", response_model=HealthResponse)
    async def health_ready(service_instance: Any = Depends(current_service)) -> Response:
        snapshot = _health_snapshot(service_instance, active_settings)
        code = status.HTTP_200_OK if _is_ready(service_instance, snapshot) else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(status_code=code, content=snapshot)

    @app.get("/v1/info", response_model=InfoResponse)
    async def info(service_instance: Any = Depends(current_service)) -> dict[str, Any]:
        method = getattr(service_instance, "info", None)
        if callable(method):
            return _as_jsonable(method())
        snapshot = _health_snapshot(service_instance, active_settings)
        return {
            "service": "aegis-tts",
            "model": snapshot["model"],
            "device": snapshot["device"],
            "dtype": snapshot["dtype"],
            "attention": snapshot["attention"],
            "cache_implementation": snapshot.get("cache_implementation"),
            "max_text_length": active_settings.max_text_length,
            "default_voice": getattr(service_instance, "default_voice_id", None),
        }

    @app.get("/v1/voices", response_model=VoicesResponse)
    async def voices(service_instance: Any = Depends(current_service)) -> dict[str, Any]:
        method = getattr(service_instance, "voices_response", None)
        if not callable(method):
            raise HTTPException(status_code=503, detail="Serviço de vozes ainda não inicializado.")
        return _as_jsonable(method())

    @app.post(
        "/v1/synthesize",
        responses={
            401: {"description": "API key ausente ou inválida."},
            422: {"description": "Entrada ou voz inválida."},
            503: {"description": "Modelo/voz indisponível ou sem VRAM."},
            504: {"description": "Timeout de fila ou geração; a GPU continua protegida."},
        },
    )
    async def synthesize(
        request: SynthesisRequest,
        _authorized: None = Depends(authorize),
        service_instance: Any = Depends(current_service),
    ) -> Response:
        # The schema enforces the public hard ceiling of 1,000 characters;
        # deployments may choose a stricter operational ceiling through env.
        if len(request.text) > active_settings.max_text_length:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"O texto excede o limite configurado de {active_settings.max_text_length} caracteres.",
            )
        try:
            result = await service_instance.synthesize(request)
        except UnknownVoiceError as exc:
            raise HTTPException(status_code=422, detail=_public_detail(exc, "Voz inválida.")) from exc
        except (ServiceNotReadyError, VoiceUnavailableError, GpuOutOfMemoryError) as exc:
            raise HTTPException(status_code=503, detail=_public_detail(exc, "Serviço de voz indisponível.")) from exc
        except (QueueTimeoutError, GenerationTimeoutError) as exc:
            raise HTTPException(status_code=504, detail=_public_detail(exc, "Tempo limite da síntese excedido.")) from exc
        except ServiceError as exc:
            raise HTTPException(status_code=503, detail=_public_detail(exc, "Falha ao gerar áudio.")) from exc
        except Exception as exc:  # pragma: no cover - defensive production path
            LOGGER.exception("erro inesperado de síntese", extra={"voice": request.voice})
            raise HTTPException(status_code=503, detail="Falha ao gerar áudio; tente novamente.") from exc

        try:
            audio, headers = _result_parts(result)
        except ServiceError as exc:
            # A malformed third-party model output is an availability failure,
            # not an unhandled API exception (and must not expose internals).
            raise HTTPException(
                status_code=503,
                detail=_public_detail(exc, "O modelo não retornou áudio válido."),
            ) from exc
        return Response(content=audio, media_type="audio/wav", headers=headers)

    return app


def _health_snapshot(service: Any, settings: AppSettings) -> dict[str, Any]:
    method = getattr(service, "health_snapshot", None)
    if not callable(method):
        return {
            "status": "error",
            "model": settings.model_id,
            "model_loaded": False,
            "device": "cuda:0",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "cache_implementation": None,
            "voices": {},
            "gpu": {},
            "load_seconds": None,
            "errors": ["serviço de modelo indisponível"],
        }
    return _as_jsonable(method())


def _is_ready(service: Any, snapshot: dict[str, Any]) -> bool:
    value = getattr(service, "is_ready", None)
    if callable(value):
        return bool(value())
    if value is not None:
        return bool(value)
    return snapshot.get("status") == "ready"


def _as_jsonable(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        raise TypeError("service response must be a dict or Pydantic model")
    return value


def _result_parts(result: Any) -> tuple[bytes, dict[str, str]]:
    """Normalize real and fake synthesis results without adding unsafe headers."""

    audio = getattr(result, "wav_bytes", None)
    if audio is None:
        audio = getattr(result, "audio", None)
    if audio is None and isinstance(result, tuple):
        audio = result[0]
    if not isinstance(audio, (bytes, bytearray)) or not bytes(audio).startswith(b"RIFF"):
        raise ServiceError("o modelo não retornou um WAV válido")

    headers_value = getattr(result, "headers", None)
    if headers_value is None and isinstance(result, tuple) and len(result) > 1:
        headers_value = result[1]
    headers = {str(key): str(value) for key, value in (headers_value or {}).items() if value is not None}
    required = {
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
    for name in required:
        headers.setdefault(name, "n/a")
    return bytes(audio), headers


def _public_detail(error: Exception, fallback: str) -> str:
    message = str(error).strip().replace("\n", " ")
    return message[:300] if message else fallback


app = create_app()
