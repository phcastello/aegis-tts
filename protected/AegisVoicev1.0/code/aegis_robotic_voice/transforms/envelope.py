from __future__ import annotations
import numpy as np


def transform_envelope(sp, *, formant_scale=1.0, frequency_warp=0.0, spectral_tilt_db_octave=0.0, envelope_contrast=1.0, **_):
    n=sp.shape[1]; pos=np.arange(n); src=np.clip(pos / max(formant_scale + frequency_warp, .1),0,n-1)
    out=np.array([np.interp(src,pos,row) for row in sp]); log=np.log(np.maximum(out,1e-12)); log=(log-log.mean(1,keepdims=True))*envelope_contrast+log.mean(1,keepdims=True)
    log += (spectral_tilt_db_octave/20*np.log(10))*np.log2(np.maximum(pos,1)/max(n/32,1))[None,:]
    return np.exp(log)


def attenuate_sibilance_envelope(spectral_envelope, sample_rate, voiced, *, cutoff_hz=5200.0, shelf_db=-2.0, transition_frames=2, shelf_width_hz=None):
    """Apply a high shelf only to unvoiced and voicing-edge WORLD frames.

    WORLD's envelope is a power spectrum, so the dB value is converted to a
    power ratio. Sustained voiced material, notably vowels, is not changed.
    """
    sp = np.asarray(spectral_envelope, dtype=float)
    voice = np.asarray(voiced, dtype=bool)
    if sp.ndim != 2 or len(voice) != sp.shape[0]:
        raise ValueError("spectral envelope and voiced frames must agree")
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2 or shelf_db > 0:
        raise ValueError("invalid sibilance shelf")
    target = ~voice.copy()
    for offset in range(1, max(0, int(transition_frames)) + 1):
        target[offset:] |= voice[offset:] != voice[:-offset]
        target[:-offset] |= voice[:-offset] != voice[offset:]
    frequency = np.linspace(0.0, sample_rate / 2, sp.shape[1])
    width_hz = cutoff_hz if shelf_width_hz is None else float(shelf_width_hz)
    if width_hz <= 0:
        raise ValueError("sibilance shelf width must be positive")
    end_hz = min(sample_rate / 2, cutoff_hz + width_hz)
    position = np.clip((frequency - cutoff_hz) / max(end_hz - cutoff_hz, 1.0), 0.0, 1.0)
    shelf = 0.5 - 0.5 * np.cos(np.pi * position)
    power_gain = 10.0 ** ((shelf_db * shelf) / 10.0)
    out = sp.copy()
    out[target] *= power_gain[None, :]
    return np.maximum(out, 1e-12)
