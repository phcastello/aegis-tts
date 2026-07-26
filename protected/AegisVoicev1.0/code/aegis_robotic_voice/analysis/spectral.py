from __future__ import annotations
import numpy as np
def validate_envelope(envelope):
    if envelope.ndim != 2 or not np.isfinite(envelope).all() or (envelope <= 0).any(): raise ValueError("invalid spectral envelope")
