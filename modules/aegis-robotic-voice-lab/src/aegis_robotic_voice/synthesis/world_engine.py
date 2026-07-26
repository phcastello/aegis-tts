from __future__ import annotations
import numpy as np
from ..analysis.world import AcousticRepresentation
def synthesize_world(representation: AcousticRepresentation, *, f0_hz=None, spectral_envelope=None, aperiodicity=None, seed=0, duration_samples=None):
    """Strict WORLD synthesis: deliberately accepts no source waveform."""
    import pyworld
    f0 = representation.f0_hz if f0_hz is None else f0_hz; sp = representation.spectral_envelope if spectral_envelope is None else spectral_envelope; ap = representation.aperiodicity if aperiodicity is None else aperiodicity
    wav=pyworld.synthesize(np.asarray(f0,float), np.asarray(sp,float), np.asarray(ap,float), representation.sample_rate, representation.frame_period_ms)
    n=representation.duration_samples if duration_samples is None else duration_samples
    return np.pad(wav[:n], (0,max(0,n-len(wav))))
