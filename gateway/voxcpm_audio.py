"""VoxCPM2 native FLOAT48k -> PCM24k streaming, independent of training.

HalfRateFIR is copied unchanged from the locally validated native probe.
No model imports, gain, limiter, clipping, or per-chunk resampler reset.
"""
from __future__ import annotations
import hashlib

PROBE_SOURCE_SHA256 = "c0b6edf891bf44f215219875d2e29331da97e18ca8b518890022f6e38b76accb"

class HalfRateFIR:
    """Bounded-history causal FIR, aligned to scipy resample_poly Kaiser10.

    A 41-tap filter delays20 native samples (10 outputs). Hold these outputs
    until future input is known; flush20 zeros at a normal EOF and trim exactly
    to ceil(input_samples/2). This compensates delay without introducing audio
    padding. Native chunk timestamps include the lookahead wait. upfirdn on
    phase-aligned history uses the same summation/order as the batch reference.
    """
    native_rate=48000
    final_rate=24000
    lookahead_native_samples=20

    def __init__(self):
        import numpy as np
        from scipy.signal import firwin
        self.h=firwin(41,.5,window=('kaiser',10.)).astype(np.float32)
        self.history=np.empty(0,dtype=np.float32);self.input_samples=0
        self.causal_outputs=0;self.output_samples=0;self.closed=False

    def _feed(self,values):
        import numpy as np
        from scipy.signal import upfirdn
        values=np.asarray(values,dtype=np.float32)
        if values.ndim!=1 or not np.isfinite(values).all():raise ValueError('Finite mono float samples required')
        if not values.size:return np.empty(0,dtype=np.float32)
        old=self.input_samples;start=old-len(self.history)
        if start%2:raise AssertionError('FIR history lost decimation phase')
        joined=np.concatenate((self.history,values))
        end=old+len(values);causal_count=(end+1)//2
        causal=upfirdn(self.h,joined,up=1,down=2)
        first=max(self.causal_outputs,10)
        result=causal[first-start//2:causal_count-start//2].copy()
        self.input_samples=end;self.causal_outputs=causal_count
        # Preserve all40 FIR predecessors plus one sample when phase requires.
        next_start=max(0,end-40);next_start-=next_start%2
        self.history=joined[next_start-start:].copy()
        self.output_samples+=len(result)
        return result

    def push(self,values):
        if self.closed:raise RuntimeError('Resampler already closed')
        return self._feed(values)

    def flush(self):
        import numpy as np
        if self.closed:raise RuntimeError('Resampler already closed')
        native=self.input_samples;expected=(native+1)//2
        result=self._feed(np.zeros(20,dtype=np.float32)) if native else np.empty(0,dtype=np.float32)
        self.closed=True;self.history=np.empty(0,dtype=np.float32)
        if self.output_samples!=expected:raise AssertionError('FIR final length differs from batch reference')
        return result

    def cancel(self):
        import numpy as np
        self.closed=True;self.history=np.empty(0,dtype=np.float32)

    def recipe(self):
        return dict(native_rate=48000,final_rate=24000,taps=41,window=['kaiser',10.],
                    filter_sha256=hashlib.sha256(self.h.tobytes()).hexdigest(),
                    phase='global input sample0',lookahead_native_samples=20,
                    lookahead_ms=20/48,normal_eof='zero extension20samples; compensate10output delay; ceil(N/2)',
                    cancel='discard pending FIR state; no synthetic tail or normal flush',
                    reference="scipy.signal.resample_poly(float32,1,2,window=('kaiser',10.))")


def native_numpy(chunk):
    import numpy as np
    if hasattr(chunk, 'detach'): chunk = chunk.detach().float().cpu().numpy()
    values = np.asarray(chunk, dtype=np.float32)
    if values.ndim == 2 and values.shape[0] == 1: values = values[0]
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Native model must yield finite mono batch1 samples')
    return values


def pcm16_bytes(values):
    import numpy as np
    values = np.asarray(values)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(np.abs(values) > 1):
        raise ValueError('PCM out of range/nonfinite; no clipping is permitted')
    return np.rint(values*32767).astype('<i2').tobytes()
