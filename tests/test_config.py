from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (
    AppSettings,
    ConfigurationError,
    MODEL_ID,
    SUPPORTED_MODEL_IDS,
    SUPPORTED_MODELS,
    load_settings,
)


def test_only_the_allowlisted_base_models_and_cuda_sdpa_settings_are_accepted() -> None:
    settings = AppSettings()
    assert settings.model_id == MODEL_ID
    assert settings.model_dir == Path(SUPPORTED_MODELS[MODEL_ID])
    assert settings.device == "cuda:0"
    assert settings.attention_implementation == "sdpa"

    assert SUPPORTED_MODEL_IDS == {
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    }
    for model_id, model_dir in SUPPORTED_MODELS.items():
        assert AppSettings(model_id=model_id).model_dir == Path(model_dir)

    with pytest.raises(ConfigurationError):
        AppSettings(model_id="Qwen/Qwen3-TTS-12Hz-9B-Base")
    with pytest.raises(ConfigurationError):
        AppSettings(dtype="float32")
    with pytest.raises(ConfigurationError):
        AppSettings(device="cpu")
    with pytest.raises(ConfigurationError):
        AppSettings(attention_implementation="flash_attention_2")
    with pytest.raises(ConfigurationError):
        AppSettings(allowed_origins=("*",))


def test_environment_loads_explicit_operational_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_TTS_PORT", "8001")
    monkeypatch.setenv("AEGIS_TTS_MAX_TEXT_LENGTH", "1000")
    monkeypatch.setenv("AEGIS_TTS_QUEUE_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("AEGIS_TTS_SAVE_SAMPLES", "true")
    monkeypatch.setenv("AEGIS_TTS_MAX_SAMPLES", "7")
    monkeypatch.setenv("AEGIS_TTS_PROFILE_GENERATION", "true")
    monkeypatch.setenv("AEGIS_TTS_CACHE_IMPLEMENTATION", "static")
    monkeypatch.setenv("AEGIS_TTS_API_KEY", "test-key")

    settings = load_settings()

    assert settings.port == 8001
    assert settings.max_text_length == 1000
    assert settings.queue_timeout_seconds == 12.5
    assert settings.save_samples is True
    assert settings.max_samples == 7
    assert settings.profile_generation is True
    assert settings.cache_implementation == "static"
    assert settings.api_key_required is True


def test_environment_can_select_only_an_allowlisted_model_and_its_mapped_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    monkeypatch.setenv("QWEN_MODEL_ID", model_id)
    # A path override cannot point startup at a different local checkpoint.
    monkeypatch.setenv("QWEN_MODEL_DIR", "/models/not-approved")

    settings = load_settings()

    assert settings.model_id == model_id
    assert settings.model_dir == Path(SUPPORTED_MODELS[model_id])

    monkeypatch.setenv("QWEN_MODEL_ID", "not/a-qwen-model")
    with pytest.raises(ConfigurationError, match="QWEN_MODEL_ID"):
        load_settings()


def test_cache_implementation_rejects_unknown_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AEGIS_TTS_CACHE_IMPLEMENTATION", "dynamic")

    with pytest.raises(ConfigurationError, match="AEGIS_TTS_CACHE_IMPLEMENTATION"):
        load_settings()
