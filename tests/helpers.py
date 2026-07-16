"""Small local fixtures shared by tests.

They deliberately use only the Python standard library so tests never need a
real Qwen checkpoint, an NVIDIA driver, or a network connection.
"""

from __future__ import annotations

from array import array
from math import pi, sin
from pathlib import Path
import json
import wave


def write_pcm_wav(
    path: Path,
    *,
    duration_seconds: float = 4.0,
    sample_rate: int = 32_000,
    channels: int = 1,
    amplitude: float = 0.2,
    frequency_hz: float = 440.0,
) -> Path:
    """Create a valid short PCM16 WAV fixture without audio dependencies."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(1, round(duration_seconds * sample_rate))
    samples = array("h")
    for index in range(frames):
        value = int(32_767 * amplitude * sin(2 * pi * frequency_hz * index / sample_rate))
        samples.extend([value] * channels)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())
    return path


def wav_bytes(*, duration_seconds: float = 0.1, sample_rate: int = 24_000) -> bytes:
    """Return a small valid PCM16 mono WAV payload for HTTP endpoint tests."""

    # Do not leave a shared file behind: construct bytes in memory through wave.
    from io import BytesIO

    output = BytesIO()
    frames = max(1, round(duration_seconds * sample_rate))
    samples = array("h", [0] * frames)
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return output.getvalue()


def write_voices_config(
    path: Path,
    *,
    short_audio: Path,
    short_text: Path,
    long_audio: Path | None = None,
    long_text: Path | None = None,
    default_voice: str = "giseli-10s",
) -> Path:
    """Write the public configuration shape using explicit local test paths."""

    voices = [
        {
            "id": "giseli-10s",
            "display_name": "Giseli — referência de 10 segundos",
            "audio_path": str(short_audio),
            "transcript_path": str(short_text),
            "enabled": True,
        }
    ]
    if long_audio is not None and long_text is not None:
        voices.append(
            {
                "id": "giseli-34s",
                "display_name": "Giseli — referência de 34 segundos",
                "audio_path": str(long_audio),
                "transcript_path": str(long_text),
                "enabled": True,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"default_voice": default_voice, "voices": voices}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
