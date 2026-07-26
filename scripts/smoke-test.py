#!/usr/bin/env python3
"""Smoke an already-running Aegis TTS gateway without persisting audio.

The command covers the public health endpoints, OpenAI WAV, native WAV,
native PCM streaming, SSE and explicit native cancellation.  It uses only the
standard library and never sends a reference asset or stores generated speech.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(base_url: str, path: str, *, method: str = "GET", payload: dict | None = None, timeout: int = 180):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    return urlopen(Request(f"{base_url}{path}", data=data, headers=headers, method=method), timeout=timeout)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def wav(response, label: str) -> str:
    body = response.read()
    require(response.headers.get_content_type() == "audio/wav", f"{label}: content-type inesperado")
    require(body.startswith(b"RIFF") and len(body) > 44, f"{label}: WAV vazio ou inválido ({len(body)} bytes)")
    return response.headers["X-Aegis-Request-Id"]


def events(base_url: str, request_id: str) -> str:
    with request(base_url, f"/v1/aegis/speech/{request_id}/events", timeout=30) as response:
        data = response.read().decode("utf-8")
    require("event: accepted" in data and "event: completed" in data, "SSE não contém accepted/completed")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    for path in ("/health", "/ready", "/metrics", "/v1/aegis/queue"):
        with request(base_url, path, timeout=30) as response:
            require(response.status == 200, f"{path}: HTTP {response.status}")
            response.read()
        print(f"ok {path}")

    with request(base_url, "/v1/audio/speech", method="POST", payload={"input": "Esta é uma frase um pouco mais longa.", "stream": False, "response_format": "wav"}) as response:
        wav(response, "OpenAI WAV")
    print("ok OpenAI WAV")

    native = {"text": "A porta está aberta.", "priority": 50, "interrupt_policy": "enqueue", "stream": False, "response_format": "wav"}
    with request(base_url, "/v1/aegis/speech", method="POST", payload=native) as response:
        request_id = wav(response, "Aegis WAV")
    events(base_url, request_id)
    print("ok Aegis WAV + SSE")

    with request(base_url, "/v1/aegis/speech", method="POST", payload={"text": "A GPU está em sessenta e dois por cento.", "stream": True, "response_format": "pcm"}) as response:
        pcm = response.read()
        require(response.headers.get_content_type() == "application/octet-stream", "PCM: content-type inesperado")
        require(len(pcm) > 0 and len(pcm) % 2 == 0, f"PCM vazio ou desalinhado ({len(pcm)} bytes)")
    print("ok Aegis PCM streaming")

    cancelling = {"text": "Esta é uma solicitação longa. " * 30, "stream": True, "response_format": "pcm"}
    response = request(base_url, "/v1/aegis/speech", method="POST", payload=cancelling)
    cancel_id = response.headers["X-Aegis-Request-Id"]
    with request(base_url, f"/v1/aegis/speech/{cancel_id}", method="DELETE", timeout=30) as cancelled:
        require(cancelled.status == 200, "cancelamento não aceito")
        cancelled.read()
    response.close()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with request(base_url, f"/v1/aegis/speech/{cancel_id}", timeout=30) as status:
            state = json.loads(status.read())
        if state["status"] == "cancelled":
            break
        time.sleep(0.2)
    require(state["status"] == "cancelled", f"cancelamento não concluiu: {state['status']}")
    print("ok cancelamento explícito")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HTTPError, URLError, KeyError, RuntimeError) as error:
        print(f"smoke falhou: {error}", file=sys.stderr)
        raise SystemExit(1)
