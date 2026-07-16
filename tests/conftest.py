from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.helpers import write_pcm_wav, write_voices_config


@dataclass(frozen=True)
class ReferenceLayout:
    root: Path
    config_path: Path
    short_audio: Path
    short_transcript: Path
    long_audio: Path
    long_transcript: Path


@pytest.fixture
def reference_layout(tmp_path: Path) -> ReferenceLayout:
    """A pair of small, valid local reference assets for unit tests only."""

    voices = tmp_path / "voices"
    short_audio = write_pcm_wav(voices / "giseli-10s" / "reference.wav", duration_seconds=4.0)
    short_transcript = short_audio.with_suffix(".txt")
    short_transcript.write_text("Esta é uma transcrição de teste exata.", encoding="utf-8")
    long_audio = write_pcm_wav(voices / "giseli-34s" / "reference.wav", duration_seconds=5.0)
    long_transcript = long_audio.with_suffix(".txt")
    long_transcript.write_text("Esta é outra transcrição de teste exata.", encoding="utf-8")
    config_path = write_voices_config(
        tmp_path / "config" / "voices.json",
        short_audio=short_audio,
        short_text=short_transcript,
        long_audio=long_audio,
        long_text=long_transcript,
    )
    return ReferenceLayout(
        root=tmp_path,
        config_path=config_path,
        short_audio=short_audio,
        short_transcript=short_transcript,
        long_audio=long_audio,
        long_transcript=long_transcript,
    )

