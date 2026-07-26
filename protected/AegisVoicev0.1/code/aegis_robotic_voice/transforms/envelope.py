from __future__ import annotations
import numpy as np
def transform_envelope(sp, *, formant_scale=1.0, frequency_warp=0.0, spectral_tilt_db_octave=0.0, envelope_contrast=1.0, **_):
    n=sp.shape[1]; pos=np.arange(n); src=np.clip(pos / max(formant_scale + frequency_warp, .1),0,n-1)
    out=np.array([np.interp(src,pos,row) for row in sp]); log=np.log(np.maximum(out,1e-12)); log=(log-log.mean(1,keepdims=True))*envelope_contrast+log.mean(1,keepdims=True)
    log += (spectral_tilt_db_octave/20*np.log(10))*np.log2(np.maximum(pos,1)/max(n/32,1))[None,:]
    return np.exp(log)
