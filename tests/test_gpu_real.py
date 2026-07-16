"""Opt-in integration test against a running GPU-backed Aegis TTS service.

This module is intentionally skipped by default.  It neither downloads a
checkpoint nor starts a model process; an operator must explicitly supply a
ready local service and opt in with ``AEGIS_TTS_RUN_GPU_TESTS=1``.
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


RUN_GPU_TESTS = os.environ.get("AEGIS_TTS_RUN_GPU_TESTS") == "1"
BASE_URL = os.environ.get("AEGIS_TTS_GPU_TEST_URL", "http://127.0.0.1:8001").rstrip("/")


def _request(path: str, *, method: str = "GET", payload: dict | None = None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if body else {}
    api_key = os.environ.get("AEGIS_TTS_GPU_TEST_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    request = Request(f"{BASE_URL}{path}", data=body, headers=headers, method=method)
    return urlopen(request, timeout=120)


@pytest.mark.gpu
@pytest.mark.skipif(not RUN_GPU_TESTS, reason="set AEGIS_TTS_RUN_GPU_TESTS=1 to run the real GPU test")
def test_running_gpu_service_returns_a_wav_without_downloading_anything() -> None:
    """Exercise a previously started service with its already-imported voice."""

    try:
        with _request("/health/ready") as response:
            assert response.status == 200
        with _request("/v1/voices") as response:
            voices = json.load(response)
    except HTTPError as error:  # Keep diagnostics concise if the service is not ready.
        pytest.fail(f"GPU service was not ready: HTTP {error.code}")

    default_voice = voices["default_voice"]
    with _request(
        "/v1/synthesize",
        method="POST",
        payload={
            "text": "Olá. Este é um teste opcional da Aegis.",
            "voice": default_voice,
            "language": "pt-BR",
            "seed": 42,
            "generation_parameters": {
                "temperature": 0.8,
                "top_p": 0.9,
                "do_sample": True,
                "max_new_tokens": 256,
            },
        },
    ) as response:
        audio = response.read()
        assert response.status == 200
        assert response.headers.get_content_type() == "audio/wav"
        assert audio.startswith(b"RIFF")
        assert response.headers.get("X-Aegis-TTS-RTF") is not None

