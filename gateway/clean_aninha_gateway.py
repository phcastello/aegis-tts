"""Minimal diagnostic proxy: Qwen3-TTS Base + private Aninha ICL only.

Unlike the production gateway, this process applies no Q00 override, no WORLD,
and no mastering.  It deliberately forwards ``stream`` and
``non_streaming_mode`` exactly as sent by the diagnostic caller.
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
REFERENCE_AUDIO = Path(os.environ.get("AEGIS_DEFAULT_REFERENCE_AUDIO", "/protected/reference/Aninha_AegisVoice4.wav"))
REFERENCE_TEXT = Path(os.environ.get("AEGIS_DEFAULT_REFERENCE_TEXT", "/protected/reference/Aninha_AegisVoice4.txt"))
MODEL = os.environ.get("AEGIS_DEFAULT_MODEL", "/models/Qwen3-TTS-12Hz-0.6B-Base")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        audio = REFERENCE_AUDIO.read_bytes()
        text = REFERENCE_TEXT.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError("clean diagnostic reference is unavailable") from exc
    if not audio or not text:
        raise RuntimeError("clean diagnostic reference is empty")
    app.state.ref_audio = "data:audio/wav;base64," + base64.b64encode(audio).decode("ascii")
    app.state.ref_text = text
    app.state.client = httpx.AsyncClient(base_url=BACKEND_URL, timeout=httpx.Timeout(900, connect=5))
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(title="Aegis clean Qwen/vLLM diagnostic", docs_url=None, redoc_url=None, lifespan=lifespan)


def _request(payload: dict[str, Any]) -> dict[str, Any]:
    request = dict(payload)
    # This diagnostic has one private reference.  It accepts no public profile
    # selection but otherwise leaves decoding and streaming controls untouched.
    request["ref_audio"] = app.state.ref_audio
    request["ref_text"] = app.state.ref_text
    request.pop("voice", None)
    request.setdefault("model", MODEL)
    request.setdefault("task_type", "Base")
    request.setdefault("x_vector_only_mode", False)
    request.setdefault("language", "Portuguese")
    request.setdefault("response_format", "wav")
    return request


def _headers(upstream: httpx.Response) -> dict[str, str]:
    headers = {name: upstream.headers[name] for name in ("content-type", "content-disposition") if name in upstream.headers}
    headers["X-Aegis-Diagnostic"] = "clean-qwen-vllm-aninha"
    return headers


@app.get("/health")
async def health() -> Response:
    try:
        upstream = await app.state.client.get("/health")
    except httpx.HTTPError:
        return Response(status_code=503)
    return Response(status_code=upstream.status_code)


@app.post("/v1/audio/speech")
async def speech(payload: dict[str, Any]) -> Response:
    request = _request(payload)
    try:
        upstream = await app.state.client.send(app.state.client.build_request("POST", "/v1/audio/speech", json=request), stream=True)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="vLLM backend unavailable") from exc
    headers = _headers(upstream)
    if request.get("stream"):
        async def body():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(body(), status_code=upstream.status_code, headers=headers)
    try:
        content = await upstream.aread()
    finally:
        await upstream.aclose()
    return Response(content=content, status_code=upstream.status_code, headers=headers)
