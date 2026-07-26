"""Static checks for the production default-reference gateway."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gateway_injects_the_private_default_without_logging_its_contents() -> None:
    gateway = (ROOT / "gateway" / "default_reference_gateway.py").read_text(encoding="utf-8")

    assert "request[\"ref_audio\"] = app.state.default_audio" in gateway
    assert "request[\"ref_text\"] = app.state.default_text" in gateway
    assert "request.pop(\"voice\", None)" in gateway
    assert "print(" not in gateway
    assert "logging." not in gateway


def test_gateway_includes_a_same_origin_pcm_streaming_player() -> None:
    gateway = (ROOT / "gateway" / "default_reference_gateway.py").read_text(encoding="utf-8")

    assert '@app.get("/", include_in_schema=False)' in gateway
    assert "new AudioContext" in gateway
    assert "fetch('/v1/audio/speech'" in gateway
    assert "response_format: 'pcm'" in gateway
    assert "stream: true" in gateway


def test_api_documentation_explains_default_and_custom_references() -> None:
    documentation = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")

    assert "giselle_AegisVoice3.wav" in documentation
    assert "POST /v1/audio/speech" in documentation
    assert "ref_audio" in documentation
    assert "ref_text" in documentation
