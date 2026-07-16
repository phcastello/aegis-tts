"""Strict, public data contracts for Aegis TTS.

Reference fields exist only for loading local configuration.  Response models
deliberately omit both reference paths and transcriptions.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


VOICE_ID_PATTERN = r"^[a-z0-9_-]+$"
_VOICE_ID_RE = re.compile(VOICE_ID_PATTERN)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _validate_voice_id(value: str) -> str:
    if not _VOICE_ID_RE.fullmatch(value):
        raise ValueError("voice id must contain only lowercase a-z, 0-9, - or _")
    return value


def _validate_local_path(value: str) -> str:
    # This is a lexical first line of defence.  ReferenceRepository resolves
    # the path under its configured root and checks symlink traversal too.
    normalized = value.replace("\\", "/")
    if not normalized or "\x00" in normalized:
        raise ValueError("reference path must not be empty")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        # A leading slash produces an empty first segment and is allowed below.
        if normalized.startswith("/") and all(
            part not in {".", ".."} for part in normalized.split("/")[1:]
        ):
            return value
        raise ValueError("reference path must not contain traversal segments")
    return value


class GenerationParameters(StrictModel):
    """The entire allowlist forwarded to ``generate_voice_clone``."""

    # Omitted fields preserve the checkpoint's native generate_config/defaults.
    temperature: Annotated[float | None, Field(default=None, gt=0.0, le=2.0)]
    top_p: Annotated[float | None, Field(default=None, gt=0.0, le=1.0)]
    do_sample: bool | None = None
    max_new_tokens: Annotated[int | None, Field(default=None, ge=1, le=4096)]


class SynthesisRequest(StrictModel):
    text: Annotated[str, Field(min_length=1, max_length=1000)]
    voice: str | None = None
    language: str = "Portuguese"
    # A stable default makes the LAN endpoint and manual UI reproducible while
    # callers can still explicitly choose a different seed or null behavior.
    seed: Annotated[int | None, Field(ge=0, le=2_147_483_647)] = 41
    generation_parameters: GenerationParameters = Field(default_factory=GenerationParameters)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator("voice")
    @classmethod
    def safe_voice_id(cls, value: str | None) -> str | None:
        return None if value is None else _validate_voice_id(value)

    @field_validator("language")
    @classmethod
    def portuguese_only(cls, value: str) -> str:
        normalized = value.strip().lower().replace("_", "-")
        if normalized in {"pt-br", "portuguese"}:
            return "Portuguese"
        raise ValueError("language must be pt-BR or Portuguese")


class VoiceConfig(StrictModel):
    id: str
    display_name: Annotated[str, Field(min_length=1, max_length=200)]
    audio_path: str
    transcript_path: str
    enabled: bool = True

    @field_validator("id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        return _validate_voice_id(value)

    @field_validator("audio_path", "transcript_path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _validate_local_path(value)


class VoicesConfig(StrictModel):
    default_voice: str
    voices: list[VoiceConfig] = Field(min_length=1)

    @field_validator("default_voice")
    @classmethod
    def safe_default_id(cls, value: str) -> str:
        return _validate_voice_id(value)

    @model_validator(mode="after")
    def validate_voice_set(self) -> "VoicesConfig":
        ids = [voice.id for voice in self.voices]
        if len(set(ids)) != len(ids):
            raise ValueError("voice ids must be unique")
        if self.default_voice not in ids:
            raise ValueError("default_voice must point to a configured voice")
        return self


class VoiceMetadata(StrictModel):
    id: str
    display_name: str
    ready: bool
    reference_duration_seconds: float | None = None
    reference_sample_rate: int | None = None


class VoicesResponse(StrictModel):
    default_voice: str
    voices: list[VoiceMetadata]


class HealthResponse(StrictModel):
    status: str
    model: str
    model_loaded: bool
    device: str
    dtype: str
    attention: str
    cache_implementation: str | None = None
    voices: dict[str, str]
    gpu: dict[str, Any]
    load_seconds: float | None = None
    prompt_seconds: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class InfoResponse(StrictModel):
    service: str = "aegis-tts"
    model: str
    device: str
    dtype: str
    attention: str
    cache_implementation: str | None = None
    max_text_length: int
    default_voice: str | None = None
