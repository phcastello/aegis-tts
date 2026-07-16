from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import AppSettings
from app.reference_repository import (
    ReferenceNotFoundError,
    ReferenceRepository,
    ReferenceValidationError,
)
from tests.conftest import ReferenceLayout
from tests.helpers import write_pcm_wav


def _settings(layout: ReferenceLayout) -> AppSettings:
    return AppSettings(
        voices_config=layout.config_path,
        voices_dir=layout.root / "voices",
        model_dir=layout.root / "models" / "Qwen3-TTS-12Hz-1.7B-Base",
        samples_dir=layout.root / "samples",
        cache_dir=layout.root / "cache",
    )


def test_valid_reference_is_cached_and_exposes_only_safe_metadata(
    reference_layout: ReferenceLayout,
) -> None:
    repository = ReferenceRepository(_settings(reference_layout))

    reference = repository.get_reference("giseli-10s")
    again = repository.get_reference("giseli-10s")

    assert reference is again
    assert reference.config.id == "giseli-10s"
    assert reference.metadata.sample_rate == 32_000
    assert reference.metadata.channels == 1
    assert reference.metadata.bits_per_sample == 16
    assert reference.metadata.codec == "PCM_16"
    assert reference.metadata.duration_seconds == 4.0
    assert reference.metadata.size_bytes and reference.metadata.size_bytes > 44
    assert len(reference.metadata.audio_sha256 or "") == 64
    assert len(reference.metadata.transcript_sha256 or "") == 64
    assert reference.metadata.public() == {
        "reference_duration_seconds": 4.0,
        "reference_sample_rate": 32_000,
    }
    assert "transcript" not in reference.metadata.public()
    assert repository.status_by_voice()["giseli-10s"] == "validated"


def test_default_voice_and_enabled_reference_listing(reference_layout: ReferenceLayout) -> None:
    repository = ReferenceRepository(_settings(reference_layout))

    assert repository.default_voice_id == "giseli-10s"
    references = repository.list_references()

    assert set(references) == {"giseli-10s", "giseli-34s"}
    assert repository.status_by_voice() == {
        "giseli-10s": "validated",
        "giseli-34s": "validated",
    }


def test_missing_reference_and_unknown_voice_are_reported_without_paths(
    reference_layout: ReferenceLayout,
) -> None:
    reference_layout.short_audio.unlink()
    repository = ReferenceRepository(_settings(reference_layout))

    with pytest.raises(ReferenceValidationError, match="arquivo de referência ausente"):
        repository.get_reference("giseli-10s")
    with pytest.raises(ReferenceNotFoundError, match="voice id desconhecido"):
        repository.get_reference("not-a-voice")


def test_empty_or_missing_transcript_is_rejected(reference_layout: ReferenceLayout) -> None:
    reference_layout.short_transcript.write_text("\n \t", encoding="utf-8")
    repository = ReferenceRepository(_settings(reference_layout))

    with pytest.raises(ReferenceValidationError, match="transcrição de referência vazia"):
        repository.get_reference("giseli-10s")

    reference_layout.short_transcript.unlink()
    repository = ReferenceRepository(_settings(reference_layout))
    with pytest.raises(ReferenceValidationError, match="transcrição de referência ausente"):
        repository.get_reference("giseli-10s")


def test_symlink_path_traversal_cannot_escape_the_private_voices_root(
    reference_layout: ReferenceLayout,
) -> None:
    outside = write_pcm_wav(reference_layout.root / "outside.wav", duration_seconds=4.0)
    reference_layout.short_audio.unlink()
    reference_layout.short_audio.symlink_to(outside)
    repository = ReferenceRepository(_settings(reference_layout))

    with pytest.raises(ReferenceValidationError, match="fora do diretório permitido"):
        repository.get_reference("giseli-10s")


def test_invalid_config_path_traversal_is_rejected_before_file_access(reference_layout: ReferenceLayout) -> None:
    raw = json.loads(reference_layout.config_path.read_text(encoding="utf-8"))
    raw["voices"][0]["audio_path"] = "/voices/../outside.wav"
    reference_layout.config_path.write_text(json.dumps(raw), encoding="utf-8")
    repository = ReferenceRepository(_settings(reference_layout))

    with pytest.raises(ReferenceValidationError, match="configuração de vozes não passou"):
        repository.load_config()

