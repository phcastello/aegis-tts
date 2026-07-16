"""Small vLLM-Omni gateway that supplies Aegis' approved ICL reference.

The vLLM endpoint requires a Base reference in every request.  The gateway
keeps that detail server-side for ordinary callers while preserving support for
an explicit, caller-provided reference.
"""

from __future__ import annotations

import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse


BACKEND_URL = os.environ.get("AEGIS_VLLM_BACKEND_URL", "http://127.0.0.1:8091")
DEFAULT_AUDIO_PATH = Path(os.environ.get("AEGIS_DEFAULT_REFERENCE_AUDIO", "/defaults/reference.wav"))
DEFAULT_TEXT_PATH = Path(os.environ.get("AEGIS_DEFAULT_REFERENCE_TEXT", "/defaults/reference.txt"))
DEFAULT_MODEL = os.environ.get("AEGIS_DEFAULT_MODEL", "/models/Qwen3-TTS-12Hz-0.6B-Base")


def _load_default_reference() -> tuple[str, str]:
    """Load once at startup; reference data is never logged or returned."""
    try:
        audio = DEFAULT_AUDIO_PATH.read_bytes()
        text = DEFAULT_TEXT_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"default reference is unavailable: {exc}") from exc
    if not audio or not text:
        raise RuntimeError("default reference audio or transcript is empty")
    return "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii"), text


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.default_audio, app.state.default_text = _load_default_reference()
    app.state.client = httpx.AsyncClient(base_url=BACKEND_URL, timeout=httpx.Timeout(900.0, connect=5.0))
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(title="Aegis TTS vLLM gateway", docs_url=None, redoc_url=None, lifespan=lifespan)


def _apply_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    request = dict(payload)
    if not request.get("ref_audio"):
        request["ref_audio"] = app.state.default_audio
        # The transcript must correspond to the audio, so it is paired with the
        # server default even if a caller supplied an unrelated ref_text.
        request["ref_text"] = app.state.default_text
    elif not request.get("ref_text"):
        raise HTTPException(status_code=422, detail="ref_text is required when ref_audio is supplied")

    # vLLM-Omni treats ``voice`` as a request for a server-side speaker cache.
    # It is incompatible with raw ICL ref_audio and can make the Stage 0 engine
    # fail.  The reference itself is the voice selection in Base mode.
    request.pop("voice", None)

    request.setdefault("model", DEFAULT_MODEL)
    request.setdefault("task_type", "Base")
    request.setdefault("x_vector_only_mode", False)
    request.setdefault("language", "Portuguese")
    request.setdefault("response_format", "wav")
    return request


def _response_headers(upstream: httpx.Response) -> dict[str, str]:
    allowed = ("content-type", "content-disposition")
    return {name: upstream.headers[name] for name in allowed if name in upstream.headers}


@app.get("/health")
async def health() -> Response:
    try:
        upstream = await app.state.client.get("/health")
    except httpx.HTTPError:
        return Response(status_code=503)
    return Response(content=upstream.content, status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))


@app.get("/v1/models")
async def models() -> Response:
    try:
        upstream = await app.state.client.get("/v1/models")
    except httpx.HTTPError:
        return Response(status_code=503)
    return Response(content=upstream.content, status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))


@app.post("/v1/audio/speech")
async def speech(payload: dict[str, Any]) -> Response:
    request = _apply_defaults(payload)
    try:
        upstream = await app.state.client.send(
            app.state.client.build_request("POST", "/v1/audio/speech", json=request), stream=True
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="vLLM backend unavailable") from exc

    headers = _response_headers(upstream)
    if request.get("stream"):
        async def stream_body():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(stream_body(), status_code=upstream.status_code, headers=headers)

    try:
        body = await upstream.aread()
    finally:
        await upstream.aclose()
    return Response(content=body, status_code=upstream.status_code, headers=headers)
