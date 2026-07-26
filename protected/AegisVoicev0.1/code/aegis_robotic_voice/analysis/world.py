from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class AcousticRepresentation:
    sample_rate: int; frame_period_ms: float; duration_samples: int; f0_hz: np.ndarray; voiced: np.ndarray
    spectral_envelope: np.ndarray; aperiodicity: np.ndarray | None; frame_energy: np.ndarray; timestamps: np.ndarray
    metadata: dict = field(default_factory=dict)

def analyze_world(waveform: np.ndarray, sample_rate: int, frame_period_ms=5.0, method="dio_stonemask") -> AcousticRepresentation:
    import pyworld
    x = np.asarray(waveform, dtype=np.float64)
    if method == "harvest": f0, ts = pyworld.harvest(x, sample_rate, frame_period=frame_period_ms)
    elif method == "dio_stonemask":
        raw, ts = pyworld.dio(x, sample_rate, frame_period=frame_period_ms); f0 = pyworld.stonemask(x, raw, ts, sample_rate)
    else: raise ValueError("method must be dio_stonemask or harvest")
    sp = pyworld.cheaptrick(x, f0, ts, sample_rate)
    ap = pyworld.d4c(x, f0, ts, sample_rate)
    n = len(f0); hop = max(1, round(sample_rate * frame_period_ms / 1000)); energy = np.array([np.mean(x[i*hop:min(len(x),(i+1)*hop)]**2) for i in range(n)])
    return AcousticRepresentation(sample_rate, frame_period_ms, len(x), f0, f0 > 0, sp, ap, energy, ts, {"f0_method":method, "aperiodicity_min":float(ap.min()), "aperiodicity_max":float(ap.max())})
