"""Environment-backed configuration for the local Aegis TTS service.

This module intentionally does not use ``pydantic-settings``.  Keeping the
small environment parser here makes the runtime image smaller and, more
importantly, makes tests able to construct settings with temporary paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# These are intentionally an exact allowlist rather than a Hub namespace
# prefix. The service must never turn an environment variable into an
# arbitrary model download or local model path.
SUPPORTED_MODELS = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "/models/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": "/models/Qwen3-TTS-12Hz-0.6B-Base",
}
SUPPORTED_MODEL_IDS = frozenset(SUPPORTED_MODELS)

# Kept as the production defaults for backwards-compatible direct imports.
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_DIR = Path(SUPPORTED_MODELS[MODEL_ID])


class ConfigurationError(ValueError):
    """Raised when an environment value would make the service unsafe."""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean")


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    try:
        value = default if raw is None or raw == "" else int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = os.getenv(name)
    try:
        value = default if raw is None or raw == "" else float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


def _env_optional_choice(name: str, allowed: set[str]) -> str | None:
    """Read an optional enum-like setting, treating an empty value as off."""

    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ConfigurationError(f"{name} must be one of: {choices}")
    return value


@dataclass(frozen=True, slots=True)
class AppSettings:
    """All service configuration, with paths expressed inside the container."""

    host: str = "0.0.0.0"
    port: int = 8001
    model_id: str = MODEL_ID
    # ``None`` means the safe allowlisted directory for ``model_id``. Unit
    # tests may provide a temporary checkpoint explicitly without opening an
    # environment-controlled model-path override in production.
    model_dir: Path | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    voices_config: Path = Path("/config/voices.json")
    voices_dir: Path = Path("/voices")
    samples_dir: Path = Path("/samples")
    cache_dir: Path = Path("/cache")
    api_key: str | None = None
    max_text_length: int = 1000
    queue_timeout_seconds: float = 240.0
    generation_timeout_seconds: float = 180.0
    save_samples: bool = False
    max_samples: int = 100
    profile_generation: bool = False
    cache_implementation: str | None = None
    log_level: str = "INFO"
    allowed_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.model_id not in SUPPORTED_MODEL_IDS:
            raise ConfigurationError(
                "QWEN_MODEL_ID must be one of: " + ", ".join(sorted(SUPPORTED_MODEL_IDS))
            )
        if self.model_dir is None:
            object.__setattr__(self, "model_dir", Path(SUPPORTED_MODELS[self.model_id]))
        if self.dtype.lower() not in {"bfloat16", "float16"}:
            raise ConfigurationError("QWEN_DTYPE must be bfloat16 or float16")
        if self.attention_implementation != "sdpa":
            raise ConfigurationError("QWEN_ATTENTION_IMPLEMENTATION must be sdpa")
        if not self.device.startswith("cuda:"):
            raise ConfigurationError("QWEN_DEVICE must select a CUDA device")
        if not (1 <= self.port <= 65535):
            raise ConfigurationError("AEGIS_TTS_PORT must be between 1 and 65535")
        if not (1 <= self.max_text_length <= 1000):
            raise ConfigurationError("AEGIS_TTS_MAX_TEXT_LENGTH must be between 1 and 1000")
        if self.queue_timeout_seconds <= 0 or self.generation_timeout_seconds <= 0:
            raise ConfigurationError("TTS timeouts must be greater than zero")
        if self.max_samples < 0:
            raise ConfigurationError("AEGIS_TTS_MAX_SAMPLES cannot be negative")
        if self.cache_implementation not in {None, "static"}:
            raise ConfigurationError("AEGIS_TTS_CACHE_IMPLEMENTATION must be 'static' or empty")
        if "*" in self.allowed_origins:
            raise ConfigurationError("AEGIS_TTS_ALLOWED_ORIGINS must not contain *")

    @property
    def api_key_required(self) -> bool:
        return bool(self.api_key)


def load_settings() -> AppSettings:
    """Load settings from the process environment without logging secrets."""

    # AEGIS_TTS_HOST is the container setting.  BIND_ADDRESS is retained as a
    # useful direct-run fallback used by the documented .env file.
    host = os.getenv("AEGIS_TTS_HOST") or os.getenv("AEGIS_TTS_BIND_ADDRESS") or "0.0.0.0"
    dtype = os.getenv("QWEN_DTYPE", "bfloat16").strip().lower()
    origins = tuple(
        item.strip()
        for item in os.getenv("AEGIS_TTS_ALLOWED_ORIGINS", "").split(",")
        if item.strip()
    )
    api_key = os.getenv("AEGIS_TTS_API_KEY") or None
    return AppSettings(
        host=host,
        port=_env_int("AEGIS_TTS_PORT", 8001, minimum=1),
        # The local directory is derived in AppSettings from the exact model
        # allowlist. QWEN_MODEL_DIR is deliberately not read, so an
        # environment cannot select a checkpoint independently of the model ID.
        model_id=os.getenv("QWEN_MODEL_ID", MODEL_ID).strip(),
        device=os.getenv("QWEN_DEVICE", "cuda:0").strip(),
        dtype=dtype,
        attention_implementation=os.getenv("QWEN_ATTENTION_IMPLEMENTATION", "sdpa").strip(),
        voices_config=Path(os.getenv("AEGIS_TTS_VOICES_CONFIG", "/config/voices.json")),
        voices_dir=Path(os.getenv("AEGIS_TTS_VOICES_DIR", "/voices")),
        samples_dir=Path(os.getenv("AEGIS_TTS_SAMPLES_DIR", "/samples")),
        cache_dir=Path(os.getenv("AEGIS_TTS_CACHE_DIR", "/cache")),
        api_key=api_key,
        max_text_length=_env_int("AEGIS_TTS_MAX_TEXT_LENGTH", 1000, minimum=1),
        queue_timeout_seconds=_env_float("AEGIS_TTS_QUEUE_TIMEOUT_SECONDS", 240.0, minimum=0.001),
        generation_timeout_seconds=_env_float("AEGIS_TTS_GENERATION_TIMEOUT_SECONDS", 180.0, minimum=0.001),
        save_samples=_env_bool("AEGIS_TTS_SAVE_SAMPLES", False),
        max_samples=_env_int("AEGIS_TTS_MAX_SAMPLES", 100, minimum=0),
        profile_generation=_env_bool("AEGIS_TTS_PROFILE_GENERATION", False),
        cache_implementation=_env_optional_choice("AEGIS_TTS_CACHE_IMPLEMENTATION", {"static"}),
        log_level=os.getenv("AEGIS_TTS_LOG_LEVEL", "INFO").strip().upper(),
        allowed_origins=origins,
    )
