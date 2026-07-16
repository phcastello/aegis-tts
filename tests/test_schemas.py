from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import GenerationParameters, SynthesisRequest, VoiceConfig, VoicesConfig


@pytest.mark.parametrize("language", ["pt-BR", "Portuguese"])
def test_portuguese_aliases_are_normalized(language: str) -> None:
    request = SynthesisRequest(text="Olá, Pedro.", voice="giseli-10s", language=language)

    assert request.language == "Portuguese"


def test_request_defaults_to_portuguese_and_rejects_overlong_text() -> None:
    request = SynthesisRequest(text="Olá", voice="giseli-10s")
    assert request.language == "Portuguese"
    assert request.seed == 41

    with pytest.raises(ValidationError):
        SynthesisRequest(text="x" * 1001, voice="giseli-10s")


@pytest.mark.parametrize("voice_id", ["../giseli", "giseli/10s", "giseli 10s", "giseli.10s", ""])
def test_voice_ids_have_a_strict_safe_shape(voice_id: str) -> None:
    with pytest.raises(ValidationError):
        SynthesisRequest(text="Olá", voice=voice_id)


def test_generation_parameters_are_allowlisted_and_bounded() -> None:
    params = GenerationParameters(temperature=0.8, top_p=0.9, do_sample=True, max_new_tokens=1024)
    assert params.temperature == 0.8
    assert params.max_new_tokens == 1024
    assert GenerationParameters().model_dump(exclude_none=True) == {}

    with pytest.raises(ValidationError):
        GenerationParameters(speed=1.5)
    with pytest.raises(ValidationError):
        GenerationParameters(top_p=1.1)
    with pytest.raises(ValidationError):
        GenerationParameters(max_new_tokens=0)


def test_request_rejects_unknown_top_level_and_generation_fields() -> None:
    with pytest.raises(ValidationError):
        SynthesisRequest(text="Olá", voice="giseli-10s", unexpected=True)
    with pytest.raises(ValidationError):
        SynthesisRequest(
            text="Olá",
            voice="giseli-10s",
            generation_parameters={"temperature": 0.7, "speed": 1.2},
        )


def test_voice_configuration_rejects_path_traversal() -> None:
    with pytest.raises(ValidationError):
        VoiceConfig(
            id="giseli-10s",
            display_name="Giseli",
            audio_path="/voices/../outside.wav",
            transcript_path="/voices/giseli-10s/reference.txt",
            enabled=True,
        )
    with pytest.raises(ValidationError):
        VoiceConfig(
            id="giseli-10s",
            display_name="Giseli",
            audio_path="/voices/giseli-10s/reference.wav",
            transcript_path="../reference.txt",
            enabled=True,
        )


def test_default_voice_must_point_to_a_configured_voice() -> None:
    voice = VoiceConfig(
        id="giseli-10s",
        display_name="Giseli",
        audio_path="/voices/giseli-10s/reference.wav",
        transcript_path="/voices/giseli-10s/reference.txt",
        enabled=True,
    )
    config = VoicesConfig(default_voice="giseli-10s", voices=[voice])
    assert config.default_voice == "giseli-10s"

    with pytest.raises(ValidationError):
        VoicesConfig(default_voice="missing", voices=[voice])
