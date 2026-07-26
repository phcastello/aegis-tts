from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt
from .deesser import deess

def master(wave, sample_rate, high_pass_hz=60, limiter_true_peak_db=-1.0, **kwargs):
    """Conservative post-resynthesis chain; it has no source or dry signal."""
    y=np.asarray(wave,float); y=sosfilt(butter(2,high_pass_hz,fs=sample_rate,btype="highpass",output="sos"),y)
    y=deess(y, sample_rate, **kwargs)
    peak=10**(limiter_true_peak_db/20); return np.clip(y,-peak,peak)
