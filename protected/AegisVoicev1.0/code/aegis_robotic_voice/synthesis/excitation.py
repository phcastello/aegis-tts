from __future__ import annotations
import numpy as np
def harmonic_excitation(f0_hz, sample_rate, samples, rolloff_db_octave=-8.0, phase=0.0):
    """Band-limited, phase-continuous excitation; no residual is accepted."""
    t=np.arange(samples)/sample_rate; hz=max(float(f0_hz),1.0); count=max(1,int((sample_rate/2-1)//hz)); out=np.zeros(samples)
    for k in range(1,count+1): out += 10**((rolloff_db_octave*np.log2(k))/20)*np.sin(2*np.pi*k*hz*t+phase)
    return out/(np.max(np.abs(out))+1e-12)
def bandlimited_impulse_train(f0_hz, sample_rate, samples, phase=0.0):
    """Impulse train synthesized by a finite Nyquist-safe Fourier series."""
    return harmonic_excitation(f0_hz, sample_rate, samples, rolloff_db_octave=0.0, phase=phase)
def synthetic_glottal_pulse(f0_hz, sample_rate, samples, phase=0.0):
    """A smooth asymmetric pulse represented with band-limited harmonics."""
    t=np.arange(samples)/sample_rate; hz=max(float(f0_hz),1.0); count=max(1,int((sample_rate/2-1)//hz)); out=np.zeros(samples)
    for k in range(1,count+1): out += (1/k**1.5)*np.sin(2*np.pi*k*hz*t+phase)
    return out/(np.max(np.abs(out))+1e-12)
def deterministic_colored_noise(samples, seed, color=0.85):
    rng=np.random.default_rng(seed); white=rng.standard_normal(samples); return np.convolve(white,[1,color],mode="same")/(1+abs(color))
