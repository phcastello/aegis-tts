from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def validate_audio(wave: np.ndarray, sample_rate: int, expected_samples: int | None = None) -> None:
    if wave.ndim != 1 or not len(wave): raise ValueError("audio must be a non-empty mono waveform")
    if not np.isfinite(wave).all(): raise ValueError("audio contains NaN or infinity")
    if sample_rate < 8000: raise ValueError("unexpected sample rate")
    if np.max(np.abs(wave)) > 1.5: raise ValueError("extreme clipping")
    if np.mean(wave * wave) < 1e-10: raise ValueError("near-zero energy")
    if expected_samples is not None and len(wave) != expected_samples: raise ValueError("duration is not sample-accurate")
