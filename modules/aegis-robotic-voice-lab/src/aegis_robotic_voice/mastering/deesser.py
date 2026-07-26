"""Deterministic detector/gain de-esser for resynthesized audio only."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


def _smooth_detector(level, sample_rate, attack_ms, release_ms):
    attack = np.exp(-1.0 / max(sample_rate * attack_ms / 1000.0, 1.0))
    release = np.exp(-1.0 / max(sample_rate * release_ms / 1000.0, 1.0))
    result = np.empty_like(level)
    previous = 0.0
    for index, value in enumerate(level):
        coefficient = attack if value > previous else release
        previous = coefficient * previous + (1.0 - coefficient) * value
        result[index] = previous
    return result


def deess(wave, sample_rate, *, enabled=None, deesser_enabled=True, center_hz=None, deesser_center_hz=6500.0, bandwidth_hz=2800.0, threshold_dbfs=-32.0, max_reduction_db=2.0, attack_ms=2.0, release_ms=45.0, **_):
    """Reduce only over-threshold high-band energy and preserve duration."""
    # A variant-level ``enabled`` is deliberate and takes precedence over the
    # chain default, enabling spectral-only comparison candidates.
    if enabled is None:
        enabled = deesser_enabled
    x = np.asarray(wave, dtype=float)
    if not enabled or max_reduction_db <= 0 or not len(x):
        return x.copy()
    centre = float(deesser_center_hz if center_hz is None else center_hz)
    low, high = centre - bandwidth_hz / 2.0, centre + bandwidth_hz / 2.0
    if not (0 < low < high < sample_rate / 2):
        raise ValueError("de-esser band must be below Nyquist")
    if attack_ms <= 0 or release_ms <= 0:
        raise ValueError("de-esser attack/release must be positive")
    band = sosfilt(butter(2, [low, high], fs=sample_rate, btype="bandpass", output="sos"), x)
    envelope = _smooth_detector(np.abs(band), sample_rate, attack_ms, release_ms)
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    excess_db = np.maximum(0.0, 20.0 * np.log10(np.maximum(envelope, 1e-12) / threshold))
    reduction_db = np.minimum(float(max_reduction_db), 0.75 * excess_db)
    gain = 10.0 ** (-reduction_db / 20.0)
    # Keep the complementary signal intact.  This is a real band de-esser,
    # not a broadband compressor triggered by sibilants.
    return x - band * (1.0 - gain)
