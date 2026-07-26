from __future__ import annotations
import numpy as np
def transform_aperiodicity(ap, voiced, *, aperiodicity_voiced_scale=1.0, floor=0.0, ceiling=1.0, **_):
    # PyWorld D4C reports aperiodicity as a ratio [0, 1].
    out=np.asarray(ap,float).copy(); out[np.asarray(voiced)] *= aperiodicity_voiced_scale
    return np.clip(out, floor, ceiling)
