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
    """Exercise the production gateway without exposing a voice asset."""

    try:
        with _request("/ready") as response:
            assert response.status == 200
    except HTTPError as error:  # Keep diagnostics concise if the service is not ready.
        pytest.fail(f"GPU service was not ready: HTTP {error.code}")

    with _request(
        "/v1/audio/speech",
        method="POST",
        payload={
            "input": "Olá. Este é um teste opcional da Aegis.",
            "response_format": "wav",
        },
    ) as response:
        audio = response.read()
        assert response.status == 200
        assert response.headers.get_content_type() == "audio/wav"
        assert audio.startswith(b"RIFF")
        assert response.headers.get("X-Aegis-Voice-Profile") == "AegisVoicev1.0-Q00-W04-P07-M02"
