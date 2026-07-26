from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.signal import lfilter
from .excitation import harmonic_excitation, bandlimited_impulse_train, synthetic_glottal_pulse, deterministic_colored_noise

@dataclass
class LPCRepresentation:
    sample_rate:int; frame_length:int; hop_length:int; duration_samples:int; coefficients:np.ndarray; gains:np.ndarray; f0_hz:np.ndarray; voiced:np.ndarray; noise_envelopes:np.ndarray; frame_energy:np.ndarray; metadata:dict=field(default_factory=dict)

def stabilize_poles(a):
    roots=np.roots(a); roots=np.where(np.abs(roots)>=.995, .994*roots/np.maximum(np.abs(roots),1e-12), roots); return np.real_if_close(np.poly(roots)).real

def analyze_lpc(waveform, sample_rate, f0_hz, voiced, frame_ms=25, hop_ms=5, lpc_order=None, preemphasis=.97):
    """Frame LPC analysis. The waveform ends here and is never passed to synthesis."""
    x=np.asarray(waveform,float); frame=max(16,round(sample_rate*frame_ms/1000)); hop=max(1,round(sample_rate*hop_ms/1000)); order=lpc_order or min(40,max(12,2+sample_rate//1000))
    coeffs=[]; gains=[]; energies=[]; frames=(len(x)+hop-1)//hop
    for i in range(frames):
        y=x[i*hop:i*hop+frame]; y=np.pad(y,(0,max(0,frame-len(y))))*np.hanning(frame); y=np.r_[y[0],y[1:]-preemphasis*y[:-1]]
        r=np.correlate(y,y,"full")[frame-1:frame+order]; R=np.empty((order,order))
        for a in range(order): R[a]=r[np.abs(a-np.arange(order))]
        try: a=np.r_[1.,np.linalg.solve(R+np.eye(order)*1e-6,-r[1:])]
        except np.linalg.LinAlgError: a=np.r_[1.,np.zeros(order)]
        coeffs.append(stabilize_poles(a)); gains.append(max(1e-6,np.sqrt(max(r[0]+np.dot(a[1:],r[1:]),1e-12)))); energies.append(np.mean(y*y))
    f=np.pad(np.asarray(f0_hz,float),(0,max(0,frames-len(f0_hz))))[:frames]; v=np.pad(np.asarray(voiced,bool),(0,max(0,frames-len(voiced))))[:frames]
    return LPCRepresentation(sample_rate,frame,hop,len(x),np.asarray(coeffs),np.asarray(gains),f,v,np.zeros((frames,order)),np.asarray(energies),{"lpc_order":order,"preemphasis":preemphasis})

def synthesize_lpc(representation: LPCRepresentation, *, source="harmonic_oscillator_bank", seed=0, duration_samples=None, voiced_noise_mix=.02, harmonic_rolloff_db_octave=-8, **_):
    """Strict source-filter resynthesis from LPC coefficients + synthetic excitation only."""
    n=representation.duration_samples if duration_samples is None else duration_samples; out=np.zeros(n); phase=0.0; corrected=0
    for i,a in enumerate(representation.coefficients):
        start=i*representation.hop_length; end=min(n,start+representation.hop_length)
        if start>=n: break
        a=stabilize_poles(a); corrected += int(not np.allclose(a,representation.coefficients[i])); m=end-start; f0=float(representation.f0_hz[i]) if i<len(representation.f0_hz) else 0
        if f0 > 0:
            sources={"harmonic_oscillator_bank":lambda:harmonic_excitation(f0,representation.sample_rate,m,harmonic_rolloff_db_octave,phase),"bandlimited_impulse_train":lambda:bandlimited_impulse_train(f0,representation.sample_rate,m,phase),"synthetic_glottal_pulse":lambda:synthetic_glottal_pulse(f0,representation.sample_rate,m,phase)}
            if source not in sources: raise ValueError(f"unsupported synthetic source: {source}")
            exc=sources[source]()
        else: exc=deterministic_colored_noise(m,seed+i)
        phase=(phase+2*np.pi*f0*m/representation.sample_rate)%(2*np.pi)
        if f0>0: exc=(1-voiced_noise_mix)*exc+voiced_noise_mix*deterministic_colored_noise(m,seed+i)
        out[start:end]+=lfilter([1],a,exc)*max(float(representation.gains[i]),1e-6)
    representation.metadata["poles_corrected"] = corrected
    return out
