from __future__ import annotations
import numpy as np
from scipy.ndimage import median_filter
from ..analysis.f0 import voiced_regions

def transform_f0(f0, voiced, *, modulation_factor=1.0, range_compression=1.0, quantization="none", quantization_step_cents=100, **_):
    """Transforms each voiced island independently: no interpolation across silence."""
    out=np.asarray(f0, dtype=float).copy()
    for start,end in voiced_regions(np.asarray(voiced, dtype=bool)):
        x=out[start:end]; cents=1200*np.log2(x/55.0); trend=median_filter(cents, size=min(len(cents)//2*2+1, 9), mode="nearest")
        y=trend + modulation_factor*(cents-trend); mid=np.median(y); y=mid + range_compression*(y-mid)
        if quantization == "fixed_cents": y=np.round(y/quantization_step_cents)*quantization_step_cents
        elif quantization == "chromatic": y=np.round(y/100)*100
        elif quantization == "custom_steps": y=np.round(y/quantization_step_cents)*quantization_step_cents
        elif quantization != "none": raise ValueError("unsupported quantization")
        out[start:end]=55.0*2**(y/1200)
    out[~np.asarray(voiced, dtype=bool)] = 0
    return out
