#!/usr/bin/env python3
"""Offline, audio-only reverse-engineering laboratory for the GLaDOS reference.

The module intentionally does not import or modify the production TTS.  All
external programs receive argv arrays and dotenv files are parsed as data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "experiments/glados-filter-lab"
DEFAULT_CONFIG = ROOT / "config/glados-reverse-engineering.yaml"
EFFECT_CONFIG = ROOT / "config/glados-effect-references.yaml"
EFFECT_FEATURE_SCHEMA_VERSION = 2
DEFAULTS = {
    "AEGIS_GLADOS_CLEAN_WAV": str(LAB / "reference/clean.wav"),
    "AEGIS_GLADOS_TARGET_WAV": str(LAB / "reference/target-melodyne.wav"),
    "AEGIS_GLADOS_MELODYNE_PROJECT": str(LAB / "reference/target-melodyne.mpd"),
    "AEGIS_GLADOS_RE_OUTPUT_DIR": str(LAB / "reverse-engineering-results"),
    "AEGIS_GLADOS_RE_REPORT_DIR": str(ROOT / "reports/glados-lab/reverse-engineering"),
    "AEGIS_GLADOS_RE_SAMPLE_RATE": "24000", "AEGIS_GLADOS_RE_CHANNELS": "1",
    "AEGIS_GLADOS_RE_WORKERS": "1", "AEGIS_GLADOS_RE_LOG_LEVEL": "INFO",
    "AEGIS_GLADOS_RE_ENABLE_GRAILLON": "1", "AEGIS_GLADOS_RE_ENABLE_PRAAT": "1",
    "AEGIS_GLADOS_RE_ENABLE_WORLD": "1", "AEGIS_GLADOS_RE_ENABLE_RUBBERBAND": "1",
    "AEGIS_GLADOS_RE_MAX_CANDIDATES": "80", "AEGIS_GLADOS_RE_KEEP_INTERMEDIATE": "1",
    "AEGIS_GLADOS_RE_NORMALIZE_LISTENING_FILES": "1",
    "AEGIS_GLADOS_RE_FFMPEG_BIN": "ffmpeg", "AEGIS_GLADOS_RE_FFPROBE_BIN": "ffprobe",
    "AEGIS_GRAILLON_PLUGIN_URI": "https://www.auburnsounds.com/products/Graillon.html40733133#mono",
    "AEGIS_GRAILLON_BUNDLE": str(LAB / "plugins/lv2/Auburn Sounds Graillon 3.lv2"),
    "AEGIS_GRAILLON_LV2_PATH": f"{LAB}/plugins/lv2:/usr/lib/lv2:/usr/local/lib/lv2:{Path.home()}/.lv2",
    "AEGIS_GRAILLON_JACK_SERVER": "aegis-glados-re", "AEGIS_GRAILLON_SAMPLE_RATE": "24000",
    "AEGIS_GRAILLON_BLOCK_SIZE": "256", "AEGIS_GRAILLON_PERIODS": "3",
    "AEGIS_GRAILLON_REALTIME": "0", "AEGIS_GRAILLON_JACKD_BIN": "jackd",
    "AEGIS_GRAILLON_JALV_BIN": "jalv", "AEGIS_GRAILLON_JACK_LSP_BIN": "jack_lsp",
    "AEGIS_GRAILLON_JACK_CONNECT_BIN": "jack_connect",
}
BOOL_KEYS = {k for k in DEFAULTS if "ENABLE_" in k or k.endswith(("INTERMEDIATE", "FILES"))}
INT_KEYS = {"AEGIS_GLADOS_RE_SAMPLE_RATE", "AEGIS_GLADOS_RE_CHANNELS", "AEGIS_GLADOS_RE_WORKERS",
            "AEGIS_GLADOS_RE_MAX_CANDIDATES", "AEGIS_GRAILLON_SAMPLE_RATE", "AEGIS_GRAILLON_BLOCK_SIZE",
            "AEGIS_GRAILLON_PERIODS"}
SCALES = {"chromatic": tuple(range(12)), "c_major": (0, 2, 4, 5, 7, 9, 11),
          "c_minor": (0, 2, 3, 5, 7, 8, 10), "minor_pentatonic": (0, 3, 5, 7, 10)}


class LabError(RuntimeError):
    pass


def run(argv: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    if not argv or any("\x00" in str(x) for x in argv):
        raise LabError("invalid subprocess argv")
    return subprocess.run([str(x) for x in argv], text=True, capture_output=True, check=False, env=env)


def parse_bool(value: str) -> bool:
    if value.lower() in {"1", "true", "yes", "on"}: return True
    if value.lower() in {"0", "false", "no", "off"}: return False
    raise LabError(f"invalid boolean: {value!r}")


def load_dotenv(path: Path, allowed: set[str] | None = None) -> dict[str, str]:
    """Parse KEY=value only; never evaluate quotes, expansions, or commands."""
    result: dict[str, str] = {}
    if not path.is_file(): return result
    for line_no, raw in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"): continue
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=([^\x00\r\n]*)", line)
        if not match: raise LabError(f"{path}:{line_no}: invalid dotenv syntax")
        key, value = match.groups()
        if allowed is not None and key not in allowed: continue
        if value.startswith(("'", '"')) or "$" in value or "`" in value:
            raise LabError(f"{path}:{line_no}: shell syntax is forbidden")
        result[key] = value
    return result


def environment(cli: argparse.Namespace | None = None) -> dict[str, str]:
    # defaults <- .env.graillon <- .env.reverse-engineering <- process <- CLI
    data = dict(DEFAULTS)
    data.update(load_dotenv(LAB / ".env.graillon", set(DEFAULTS)))
    data.update(load_dotenv(LAB / ".env.reverse-engineering", set(DEFAULTS)))
    data.update({k: os.environ[k] for k in DEFAULTS if k in os.environ})
    for attr, key in (("clean", "AEGIS_GLADOS_CLEAN_WAV"), ("target", "AEGIS_GLADOS_TARGET_WAV"),
                      ("project", "AEGIS_GLADOS_MELODYNE_PROJECT"), ("output_dir", "AEGIS_GLADOS_RE_OUTPUT_DIR")):
        value = getattr(cli, attr, None) if cli else None
        if value: data[key] = value
    for key in BOOL_KEYS: parse_bool(data[key])
    for key in INT_KEYS:
        try: value = int(data[key])
        except ValueError as exc: raise LabError(f"{key} must be an integer") from exc
        if value <= 0: raise LabError(f"{key} must be positive")
    if int(data["AEGIS_GLADOS_RE_CHANNELS"]) != 1: raise LabError("laboratory output must be mono")
    return data


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml
    value = yaml.safe_load(path.read_text(encoding="utf8"))
    if not isinstance(value, dict): raise LabError("YAML root must be a mapping")
    validate_yaml(value)
    return value


def validate_yaml(value: dict[str, Any]) -> None:
    engines = value.get("engines", [])
    ids = [x.get("id") for x in engines]
    if not ids or any(not isinstance(x, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", x) for x in ids):
        raise LabError("invalid engine id")
    if len(ids) != len(set(ids)): raise LabError("engine ids must be unique")
    weights = value.get("scoring", {})
    required = {"pitch", "spectral_envelope", "dynamics", "timing", "artifacts_penalty"}
    if set(weights) != required or any(not isinstance(weights[x], (int, float)) or weights[x] < 0 for x in required):
        raise LabError("invalid scoring weights")


def paths(env: dict[str, str]) -> dict[str, Path]:
    out = Path(env["AEGIS_GLADOS_RE_OUTPUT_DIR"])
    return {"clean": Path(env["AEGIS_GLADOS_CLEAN_WAV"]), "target": Path(env["AEGIS_GLADOS_TARGET_WAV"]),
            "project": Path(env["AEGIS_GLADOS_MELODYNE_PROJECT"]), "out": out,
            "work": out / "work", "analysis": out / "analysis", "raw": out / "candidates/raw",
            "normalized": out / "candidates/normalized", "diagnostics": out / "candidates/diagnostics",
            "report": Path(env["AEGIS_GLADOS_RE_REPORT_DIR"])}


def ensure_dirs(p: dict[str, Path]) -> None:
    for key in ("out", "work", "analysis", "raw", "normalized", "diagnostics", "report"):
        p[key].mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def probe(path: Path, env: dict[str, str]) -> dict[str, Any]:
    proc = run([env["AEGIS_GLADOS_RE_FFPROBE_BIN"], "-v", "error", "-show_entries",
                "format=duration,size,bit_rate:stream=codec_name,sample_fmt,sample_rate,channels,bits_per_sample",
                "-of", "json", str(path)])
    if proc.returncode: raise LabError(f"invalid WAV {path}: {proc.stderr.strip()}")
    payload = json.loads(proc.stdout); streams = payload.get("streams", [])
    if len(streams) != 1 or not streams[0].get("sample_rate"): raise LabError(f"{path}: one audio stream required")
    result = {**streams[0], **payload["format"], "path": str(path), "sha256": sha256(path)}
    result["sample_rate"] = int(result["sample_rate"]); result["channels"] = int(result["channels"])
    result["duration"] = float(result["duration"]); result["size"] = int(result["size"])
    return result


def load_audio(path: Path, sr: int | None = None) -> tuple[Any, int]:
    import soundfile as sf
    import scipy.signal
    data, source_sr = sf.read(path, dtype="float64", always_2d=True)
    mono = data.mean(axis=1)
    if sr and source_sr != sr:
        divisor = math.gcd(source_sr, sr)
        mono = scipy.signal.resample_poly(mono, sr // divisor, source_sr // divisor, window=("kaiser", 8.6))
        source_sr = sr
    return mono, source_sr


def write_audio(path: Path, audio: Any, sr: int, subtype: str = "FLOAT") -> None:
    import numpy as np
    import soundfile as sf
    value = np.asarray(audio, dtype=np.float64).reshape(-1)
    if not len(value) or not np.all(np.isfinite(value)): raise LabError("invalid audio output")
    path.parent.mkdir(parents=True, exist_ok=True); sf.write(path, value, sr, subtype=subtype)


def active_bounds(audio: Any, sr: int) -> tuple[int, int]:
    import numpy as np
    frame = max(1, int(.02 * sr)); kernel = np.ones(frame) / frame
    env = np.sqrt(np.convolve(np.asarray(audio) ** 2, kernel, mode="same"))
    threshold = max(10 ** (-55 / 20), float(env.max()) * 0.003)
    indices = np.flatnonzero(env > threshold)
    return (int(indices[0]), int(indices[-1] + 1)) if len(indices) else (0, len(audio))


def align_audio(clean: Any, target: Any, sr: int) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np
    import scipy.signal
    cb, ce = active_bounds(clean, sr); tb, te = active_bounds(target, sr)
    hop = max(1, sr // 200)
    def envelope(x: Any) -> Any:
        n = len(x) // hop; x = np.asarray(x[:n * hop]).reshape(n, hop)
        return np.sqrt(np.mean(x * x, axis=1))
    a, b = envelope(clean), envelope(target)
    corr = scipy.signal.correlate(b - b.mean(), a - a.mean(), mode="full", method="fft")
    lag_frames = int(np.argmax(corr) - (len(a) - 1)); lag = lag_frames * hop
    # Positive lag means target starts later. Preserve useful duration and record trims.
    c_start, t_start = max(0, -lag), max(0, lag)
    length = min(len(clean) - c_start, len(target) - t_start)
    aligned_c, aligned_t = clean[c_start:c_start + length], target[t_start:t_start + length]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return aligned_c, aligned_t, {"lag_samples": lag, "lag_ms": lag * 1000 / sr,
        "clean_active_start_ms": cb * 1000 / sr, "clean_active_end_ms": ce * 1000 / sr,
        "target_active_start_ms": tb * 1000 / sr, "target_active_end_ms": te * 1000 / sr,
        "clean_trim_start_samples": c_start, "target_trim_start_samples": t_start,
        "aligned_samples": length, "aligned_duration_seconds": length / sr,
        "envelope_correlation_peak_normalized": float(corr.max() / denom),
        "duration_change_recorded": bool(c_start or t_start or len(clean) != len(target))}


def prepare(env: dict[str, str]) -> dict[str, Any]:
    p = paths(env); ensure_dirs(p); sr = int(env["AEGIS_GLADOS_RE_SAMPLE_RATE"])
    source = {"clean": probe(p["clean"], env), "target": probe(p["target"], env)}
    clean, _ = load_audio(p["clean"], sr); target, _ = load_audio(p["target"], sr)
    write_audio(p["work"] / "clean-analysis.wav", clean, sr)
    write_audio(p["work"] / "target-analysis.wav", target, sr)
    ac, at, alignment = align_audio(clean, target, sr)
    write_audio(p["work"] / "clean-aligned.wav", ac, sr); write_audio(p["work"] / "target-aligned.wav", at, sr)
    ref = p["out"] / "reference"; ref.mkdir(exist_ok=True)
    write_audio(ref / "clean-normalized.wav", clean, sr, "PCM_24")
    write_audio(ref / "target-normalized.wav", target, sr, "PCM_24")
    payload = {"source": source, "analysis_sample_rate": sr,
        "resampling": {k: {"from_hz": source[k]["sample_rate"], "to_hz": sr,
                            "method": "scipy.signal.resample_poly Kaiser beta=8.6",
                            "performed": source[k]["sample_rate"] != sr} for k in source}, "alignment": alignment}
    (p["analysis"] / "input-properties.json").write_text(json.dumps(payload, indent=2), encoding="utf8")
    return payload


def pitch_track(audio: Any, sr: int, floor: float = 70, ceiling: float = 500, frame_ms: float = 10) -> dict[str, Any]:
    import numpy as np
    import parselmouth
    sound = parselmouth.Sound(np.asarray(audio), sampling_frequency=sr)
    pitch = sound.to_pitch_ac(time_step=frame_ms / 1000, pitch_floor=floor, pitch_ceiling=ceiling)
    f0 = pitch.selected_array["frequency"].astype(float)
    times = pitch.xs(); voiced = f0 > 0
    midi = np.full_like(f0, np.nan); midi[voiced] = 69 + 12 * np.log2(f0[voiced] / 440)
    cents_grid = np.full_like(f0, np.nan); cents_grid[voiced] = (midi[voiced] - np.round(midi[voiced])) * 100
    return {"time": times, "f0": f0, "voiced": voiced, "midi": midi, "distance_cents": cents_grid}


def regions(mask: Any) -> list[tuple[int, int]]:
    import numpy as np
    indices = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(a), int(b)) for a, b in indices.reshape(-1, 2)]


def pitch_summary(track: dict[str, Any], frame_ms: float = 10) -> dict[str, Any]:
    import numpy as np
    f0, midi, voiced = track["f0"], track["midi"], track["voiced"]
    values = midi[voiced]; rounded = np.round(values) if len(values) else np.array([])
    jumps = np.diff(rounded); changed = jumps[jumps != 0]
    distribution = {str(n): float(np.mean((rounded.astype(int) % 12) == n)) for n in range(12)} if len(values) else {}
    slow = scipy_smooth(values, 21) if len(values) else values
    residual = values - slow if len(values) else values
    tonal = [((b-a) * frame_ms) for a, b in regions(voiced)]
    return {"voiced_fraction": float(np.mean(voiced)), "median_f0_hz": float(np.median(f0[voiced])) if voiced.any() else None,
        "mean_abs_distance_to_chromatic_cents": float(np.mean(np.abs(track["distance_cents"][voiced]))) if voiced.any() else None,
        "note_class_distribution": distribution, "tonal_region_count": len(tonal),
        "tonal_region_median_ms": float(np.median(tonal)) if tonal else 0, "jump_count": int(len(changed)),
        "median_jump_semitones": float(np.median(np.abs(changed))) if len(changed) else 0,
        "note_change_rate_hz": float(len(changed) / max(.001, len(f0) * frame_ms / 1000)),
        "pitch_drift_std_semitones": float(np.std(slow)) if len(values) else None,
        "pitch_modulation_std_cents": float(np.std(residual) * 100) if len(values) else None}


def scipy_smooth(values: Any, width: int) -> Any:
    import numpy as np
    if len(values) < 3: return np.asarray(values)
    width = min(width, len(values) if len(values) % 2 else len(values)-1); width = max(1, width)
    return np.convolve(values, np.ones(width)/width, mode="same")


def audio_features(audio: Any, sr: int, track: dict[str, Any] | None = None) -> dict[str, Any]:
    import librosa
    import numpy as np
    x = np.asarray(audio); hop = max(1, sr // 100); n_fft = 2048
    s = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop)); power = s*s + 1e-12
    centroid = librosa.feature.spectral_centroid(S=s, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=s, sr=sr, roll_percent=.85)[0]
    flatness = librosa.feature.spectral_flatness(S=s)[0]
    rms_frames = librosa.feature.rms(y=x, frame_length=2048, hop_length=hop)[0]
    peak = float(np.max(np.abs(x))); rms = float(np.sqrt(np.mean(x*x))); crest = peak/max(rms,1e-12)
    freqs = np.fft.rfftfreq(n_fft, 1/sr); mean_spec = np.mean(power, axis=1)
    valid = (freqs >= 100) & (freqs <= min(10000, sr/2))
    tilt = float(np.polyfit(np.log10(freqs[valid]), 10*np.log10(mean_spec[valid]), 1)[0])
    bands = {}
    for low, high in ((0,250),(250,500),(500,1000),(1000,2000),(2000,4000),(4000,8000),(8000,12000)):
        use = (freqs >= low) & (freqs < min(high, sr/2+1)); bands[f"{low}-{high}"] = float(mean_spec[use].sum()/mean_spec.sum()) if use.any() else 0
    harmonic, percussive = librosa.effects.hpss(x); hnr = 10*math.log10((float(np.mean(harmonic**2))+1e-12)/(float(np.mean(percussive**2))+1e-12))
    return {"rms": rms, "rms_dbfs": 20*math.log10(max(rms,1e-12)), "peak": peak,
        "peak_dbfs": 20*math.log10(max(peak,1e-12)), "crest_factor": crest,
        "centroid_hz": float(np.mean(centroid)), "rolloff_hz": float(np.mean(rolloff)),
        "spectral_flatness": float(np.mean(flatness)), "spectral_tilt_db_per_decade": tilt,
        "harmonic_to_noise_db_proxy": hnr, "band_energy_fraction": bands,
        "dynamic_range_rms_db": float(20*np.log10(max(np.percentile(rms_frames,95),1e-12)/max(np.percentile(rms_frames,10),1e-12))),
        "silence_fraction_below_-50db": float(np.mean(np.abs(x) < 10**(-50/20))),
        "dc_offset": float(np.mean(x)), "spectral_envelope_db": (10*np.log10(mean_spec)).tolist()}


def formants(audio: Any, sr: int, frame_ms: int = 20) -> dict[str, Any]:
    import numpy as np
    import parselmouth
    from parselmouth.praat import call
    sound = parselmouth.Sound(np.asarray(audio), sampling_frequency=sr)
    obj = call(sound, "To Formant (burg)", .01, 5, 5500, .025, 50)
    result: dict[str, Any] = {}
    times = np.arange(.05, sound.duration-.05, frame_ms/1000)
    for index in (1,2,3):
        vals=[]
        for t in times:
            value=call(obj,"Get value at time",index,float(t),"Hertz","Linear")
            if value and np.isfinite(value): vals.append(float(value))
        result[f"F{index}_median_hz"] = float(np.median(vals)) if vals else None
        result[f"F{index}_confidence"] = min(1.0, len(vals)/max(1,len(times)))
    return result


def analyze_pair(env: dict[str, str]) -> dict[str, Any]:
    import numpy as np
    p=paths(env); metadata=prepare(env); sr=int(env["AEGIS_GLADOS_RE_SAMPLE_RATE"])
    clean,_=load_audio(p["work"]/"clean-aligned.wav"); target,_=load_audio(p["work"]/"target-aligned.wav")
    tc=pitch_track(clean,sr); tt=pitch_track(target,sr)
    fc=audio_features(clean,sr,tc); ft=audio_features(target,sr,tt)
    pc=pitch_summary(tc); pt=pitch_summary(tt)
    lc=loudness_metrics(p["work"]/"clean-aligned.wav");lt=loudness_metrics(p["work"]/"target-aligned.wav")
    result={"input":metadata,"pitch":{"clean":pc,"target":pt},"timbre":{"clean":{**fc,**formants(clean,sr)},"target":{**ft,**formants(target,sr)}},
            "dynamics":{"clean":{**{k:fc[k] for k in ("rms_dbfs","peak_dbfs","crest_factor","dynamic_range_rms_db","silence_fraction_below_-50db")},**lc},
                        "target":{**{k:ft[k] for k in ("rms_dbfs","peak_dbfs","crest_factor","dynamic_range_rms_db","silence_fraction_below_-50db")},**lt}},
            "timing":metadata["alignment"]}
    (p["analysis"] / "clean-vs-target.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    parse_mpd(p["project"],p["analysis"])
    make_analysis_plots(p["analysis"],clean,target,sr,tc,tt,fc,ft)
    return result


def make_analysis_plots(out: Path, clean: Any, target: Any, sr: int, tc: dict[str,Any], tt: dict[str,Any], fc:dict[str,Any], ft:dict[str,Any]) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    def save(name:str,draw:Any)->None:
        fig,ax=plt.subplots(figsize=(10,3)); draw(ax); fig.tight_layout(); fig.savefig(out/name,dpi=120); plt.close(fig)
    save("pitch-clean.png",lambda ax:(ax.plot(tc["time"],np.where(tc["voiced"],tc["f0"],np.nan)),ax.set(ylabel="F0 Hz",title="Clean pitch")))
    save("pitch-target.png",lambda ax:(ax.plot(tt["time"],np.where(tt["voiced"],tt["f0"],np.nan),color="tab:orange"),ax.set(ylabel="F0 Hz",title="Target pitch")))
    save("pitch-overlay.png",lambda ax:(ax.plot(tc["time"],np.where(tc["voiced"],tc["f0"],np.nan),label="clean"),ax.plot(tt["time"],np.where(tt["voiced"],tt["f0"],np.nan),label="target"),ax.legend()))
    save("pitch-distance-to-grid.png",lambda ax:(ax.hist(tc["distance_cents"][tc["voiced"]],bins=50,alpha=.5,label="clean"),ax.hist(tt["distance_cents"][tt["voiced"]],bins=50,alpha=.5,label="target"),ax.legend()))
    freq=np.linspace(0,sr/2,len(fc["spectral_envelope_db"]))
    save("spectral-envelope-comparison.png",lambda ax:(ax.plot(freq,fc["spectral_envelope_db"],label="clean"),ax.plot(freq,ft["spectral_envelope_db"],label="target"),ax.set_xlim(0,8000),ax.legend()))
    def rms_curve(x:Any)->Any:
        n=sr//50; return 20*np.log10(np.maximum(1e-8,np.sqrt(np.convolve(np.asarray(x)**2,np.ones(n)/n,"same"))))
    save("loudness-comparison.png",lambda ax:(ax.plot(np.arange(len(clean))/sr,rms_curve(clean),label="clean"),ax.plot(np.arange(len(target))/sr,rms_curve(target),label="target"),ax.set_ylim(-80,0),ax.legend()))
    fclean=formants(clean,sr); ftarget=formants(target,sr)
    save("formants-comparison.png",lambda ax:ax.bar(np.arange(3)-.2,[fclean[f"F{i}_median_hz"] or 0 for i in (1,2,3)],.4,label="clean") or None)
    # Replace the terse bar plot with both series.
    fig,ax=plt.subplots(figsize=(7,3)); x=np.arange(3); ax.bar(x-.2,[fclean[f"F{i}_median_hz"] or 0 for i in (1,2,3)],.4,label="clean"); ax.bar(x+.2,[ftarget[f"F{i}_median_hz"] or 0 for i in (1,2,3)],.4,label="target"); ax.set_xticks(x,["F1","F2","F3"]);ax.legend();fig.tight_layout();fig.savefig(out/"formants-comparison.png",dpi=120);plt.close(fig)
    save("waveform-alignment.png",lambda ax:(ax.plot(np.arange(len(clean))/sr,clean,alpha=.5,label="clean"),ax.plot(np.arange(len(target))/sr,target,alpha=.5,label="target"),ax.legend()))


def parse_mpd(source: Path, out: Path) -> dict[str, Any]:
    summary={"path":str(source),"exists":source.is_file(),"format":"unknown","size":0,"sha256":None,"chunks":[],"strings":[],"confidence":"low"}
    parameters={"interpreted":[],"raw_fields":[],"warning":"No undocumented binary field is assigned Melodyne semantics without evidence."}
    notes:list[dict[str,Any]]=[]
    if source.is_file():
        copy=out/"melodyne-project-copy.mpd"; shutil.copy2(source,copy); data=copy.read_bytes()
        summary.update(size=len(data),sha256=hashlib.sha256(data).hexdigest(),magic=data[:16].hex(),format="Celemony GN binary container" if data.startswith(b"GNBCFA") else "unknown binary")
        for match in re.finditer(rb"[ -~]{5,}",data):
            text=match.group().decode("ascii","replace"); item={"offset":match.start(),"type":"ascii","raw":text,"confidence":"high"}; summary["strings"].append(item)
            if len(parameters["raw_fields"])<500: parameters["raw_fields"].append(item)
        for match in re.finditer(rb"x[\x01\x5e\x9c\xda]",data):
            try:
                raw=zlib.decompress(data[match.start():]); found=[(m.start(),m.group().decode("utf8","replace")) for m in re.finditer(rb"[ -~]{5,}",raw)];strings=[x[1] for x in found]
                summary["chunks"].append({"offset":match.start(),"type":"zlib","decompressed_size":len(raw),"strings":strings[:2000],"string_count":len(strings),"confidence":"medium"})
                for offset,text in found[:2000]:
                    if len(parameters["raw_fields"])<2500:parameters["raw_fields"].append({"name":text,"type":"decompressed_ascii","value_raw":text,"location":{"container_offset":match.start(),"decompressed_offset":offset},"confidence":"low"})
            except zlib.error: pass
        summary["confidence"]="medium" if summary["chunks"] else "low"
        all_strings=[s for chunk in summary["chunks"] for s in chunk["strings"]]
        if "creatorVersion" in all_strings and "5.4.2" in all_strings:
            summary["melodyne_version"]={"value":"5.4.2","confidence":"medium","evidence":"creatorVersion and value occur in the same decompressed metadata chunk"}
    (out/"melodyne-project-summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")
    (out/"melodyne-parameters.json").write_text(json.dumps(parameters,indent=2,ensure_ascii=False),encoding="utf8")
    with (out/"melodyne-notes.csv").open("w",newline="",encoding="utf8") as f: csv.DictWriter(f,["start","end","pitch","confidence","raw_location"]).writeheader()
    return summary


def nearest_allowed(midi: Any, scale: str) -> Any:
    import numpy as np
    allowed=SCALES.get(scale)
    if allowed is None: raise LabError(f"unknown scale: {scale}")
    x=np.asarray(midi,dtype=float); base=np.floor(x/12)*12
    candidates=np.stack([base+note+octave for note in allowed for octave in (-12,0,12)],axis=-1)
    return np.take_along_axis(candidates,np.argmin(np.abs(candidates-x[...,None]),axis=-1)[...,None],axis=-1)[...,0]


def modify_track(track: dict[str,Any], config:dict[str,Any]) -> Any:
    import numpy as np
    f0=track["f0"].copy(); voiced=track["voiced"]; midi=track["midi"].copy(); scale=config.get("quantization","chromatic")
    quant=nearest_allowed(midi[voiced],scale); correction=float(config.get("correction_amount",1)); result=midi[voiced]+correction*(quant-midi[voiced])
    flatten=float(config.get("flatten_amount",1))
    if flatten:
        for a,b in regions(voiced):
            idx=np.flatnonzero(voiced)[0:0]  # shape-safe placeholder
            local=np.arange(a,b); vals=midi[local]; center=nearest_allowed(np.array([np.median(vals)]),scale)[0]
            result_indices=np.cumsum(voiced)-1; result[result_indices[local]]=result[result_indices[local]]*(1-flatten)+center*flatten
    f0[voiced]=440*2**((result-69)/12); f0[~voiced]=0
    transition=float(config.get("transition_ms",0))
    if transition>0:
        width=max(1,int(transition/10)); full=midi.copy();full[voiced]=69+12*np.log2(f0[voiced]/440)
        smooth=scipy_smooth(np.nan_to_num(full,nan=0),2*width+1);f0[voiced]=440*2**((smooth[voiced]-69)/12)
    return f0


@dataclass
class RenderResult:
    success: bool
    output: str | None
    elapsed_seconds: float
    error: str | None = None
    details: dict[str,Any] | None = None


class AudioTransformEngine(ABC):
    id="base"
    @abstractmethod
    def preflight(self)->dict[str,Any]: ...
    @abstractmethod
    def render(self,source:Path,config:dict[str,Any],target:Path)->RenderResult: ...
    def describe(self)->dict[str,Any]: return {"id":self.id,"type":type(self).__name__}


class PassthroughEngine(AudioTransformEngine):
    id="passthrough"
    def preflight(self): return {"available":True}
    def render(self,source,config,target):
        start=time.monotonic(); audio,sr=load_audio(source);write_audio(target,audio,sr);return RenderResult(True,str(target),time.monotonic()-start)


class PraatEngine(AudioTransformEngine):
    id="praat"
    def preflight(self):
        try: import parselmouth; return {"available":True,"version":parselmouth.__version__}
        except ImportError as exc:return {"available":False,"reason":str(exc)}
    def render(self,source,config,target):
        start=time.monotonic()
        try:
            import numpy as np, parselmouth
            from parselmouth.praat import call
            audio,sr=load_audio(source);sound=parselmouth.Sound(audio,sampling_frequency=sr)
            manipulation=call(sound,"To Manipulation",.01,70,500);tier=call(manipulation,"Extract pitch tier")
            track=pitch_track(audio,sr); modified=modify_track(track,config); call(tier,"Remove points between",0,sound.duration)
            for t,value in zip(track["time"],modified):
                if value>0:call(tier,"Add point",float(t),float(value))
            call([tier,manipulation],"Replace pitch tier");resynth=call(manipulation,"Get resynthesis (overlap-add)")
            value=np.asarray(resynth.values[0],dtype=float); value=fit_duration(value,len(audio));write_audio(target,value,sr)
            return RenderResult(True,str(target),time.monotonic()-start,details={"voiced_frames":int(np.sum(modified>0))})
        except Exception as exc:return RenderResult(False,None,time.monotonic()-start,str(exc))


class WorldEngine(AudioTransformEngine):
    id="world"
    def preflight(self):
        try: import pyworld; return {"available":True,"version":getattr(pyworld,"__version__","unknown")}
        except ImportError as exc:return {"available":False,"reason":str(exc)}
    def render(self,source,config,target):
        start=time.monotonic()
        try:
            import numpy as np,pyworld
            audio,sr=load_audio(source); x=np.asarray(audio,dtype=np.float64)
            f0,t=pyworld.harvest(x,sr,f0_floor=70,f0_ceil=500,frame_period=10);f0=pyworld.stonemask(x,f0,t,sr)
            sp=pyworld.cheaptrick(x,f0,t,sr);ap=pyworld.d4c(x,f0,t,sr)
            midi=np.full_like(f0,np.nan);v=f0>0;midi[v]=69+12*np.log2(f0[v]/440)
            mf=modify_track({"f0":f0,"voiced":v,"midi":midi,"time":t},config)
            shift=float(config.get("formant_shift",0));
            if shift: sp=warp_spectrum(sp,2**(shift/12))
            y=pyworld.synthesize(mf,sp,ap,sr,frame_period=10);y=fit_duration(y,len(x))
            wet=float(config.get("wet",1.0));wet=max(0.0,min(1.0,wet));y=(1-wet)*x+wet*y
            write_audio(target,y,sr)
            return RenderResult(True,str(target),time.monotonic()-start,details={"spectral_envelope_preserve":config.get("spectral_envelope_preserve",1.0)})
        except Exception as exc:return RenderResult(False,None,time.monotonic()-start,str(exc))


def warp_spectrum(sp:Any,factor:float)->Any:
    import numpy as np
    grid=np.arange(sp.shape[1]);source=np.clip(grid/factor,0,sp.shape[1]-1)
    return np.stack([np.interp(source,grid,row) for row in sp])


def fit_duration(audio:Any,n:int)->Any:
    import numpy as np
    value=np.asarray(audio);return np.pad(value[:n],(0,max(0,n-len(value))))


class RubberBandEngine(AudioTransformEngine):
    id="rubberband"
    def __init__(self,ffmpeg="ffmpeg"):self.ffmpeg=ffmpeg
    def preflight(self):
        p=run([self.ffmpeg,"-hide_banner","-filters"]);return {"available":p.returncode==0 and "rubberband" in p.stdout,"implementation":"FFmpeg librubberband filter"}
    def render(self,source,config,target):
        start=time.monotonic();semitones=float(config.get("pitch_shift_semitones",0));pitch=2**(semitones/12)
        filt=f"rubberband=pitch={pitch:.10f}:formant={config.get('formant','preserved')}:transients=smooth"
        proc=run([self.ffmpeg,"-y","-v","error","-i",str(source),"-af",filt,"-ac","1",str(target)])
        return RenderResult(proc.returncode==0,str(target) if proc.returncode==0 else None,time.monotonic()-start,proc.stderr.strip() or None)


class GraillonEngine(AudioTransformEngine):
    id="graillon_lv2"
    def __init__(self,env:dict[str,str]):self.env=env
    def _config(self):
        sys.path.insert(0,str(ROOT/"tools"));import graillon_lab
        values=dict(graillon_lab.DEFAULTS)
        mapping={k:v for k,v in self.env.items() if k in values};values.update(mapping)
        values["AEGIS_GRAILLON_JACK_SERVER"]=self.env["AEGIS_GRAILLON_JACK_SERVER"]
        return graillon_lab,graillon_lab.validate_config(values)
    def preflight(self):
        try:
            lab,c=self._config();checks=lab.preflight(c);return {"available":all(v for k,v in checks.items() if k not in {"input_wav"}),"checks":checks}
        except Exception as exc:return {"available":False,"reason":str(exc)}
    def render(self,source,config,target):
        start=time.monotonic()
        try:
            lab,c=self._config();inv=lab.discover(c);sm=lab.semantic_map();scale=SCALES[config.get("quantization","chromatic")]
            controls={"correction_enabled":1,"correction_amount":config.get("correction_amount",1),"wet_mix":config.get("wet",1),"dry_mix":1-config.get("wet",1),"midi_enabled":0,"fx_enabled":1,"pitch_shift":.5,"formant_enabled":1 if config.get("formant_enabled",False) else 0,"formant_amount":config.get("formant_amount",1),"formant_shift":.5+float(config.get("formant_shift",0))/24,"smooth":config.get("smooth",0),"inertia":config.get("inertia",0),"snap_min":config.get("snap_min",0),"snap_max":config.get("snap_max",1),"output_gain":.5}
            for n,name in enumerate(("c","c_sharp","d","d_sharp","e","f","f_sharp","g","g_sharp","a","a_sharp","b")):controls["allow_"+name]=1 if n in scale else 0
            # Mark this as an explicit controlled preset.  Otherwise the old
            # G-series compatibility path recenters p51 and disables p63 after
            # these optimizer controls have been applied.
            pre={"id":"RE","name":"reverse engineering","description":"audio-only",
                 "base_state":"official-hard-tune","round2":True,"controls":controls}
            with tempfile.TemporaryDirectory(prefix="aegis-re-graillon-") as td:
                out=Path(td);(out/"raw").mkdir();(out/"normalized").mkdir();rdir=out/"run";rdir.mkdir()
                record,_=lab.render_one(c,pre,inv,out,source,rdir)
                source_audio,source_sr=load_audio(source);rendered,_=load_audio(out/"raw"/record["raw_file"],source_sr)
                write_audio(target,fit_duration(rendered,len(source_audio)),source_sr)
            return RenderResult(True,str(target),time.monotonic()-start,details={"controls":controls})
        except Exception as exc:return RenderResult(False,None,time.monotonic()-start,str(exc))


class ChainEngine(AudioTransformEngine):
    id="chain"
    def __init__(self,registry:dict[str,AudioTransformEngine]):self.registry=registry
    def preflight(self):return {"available":True,"note":"availability is checked per stage"}
    def render(self,source,config,target):
        start=time.monotonic();stages=config.get("stages",[])
        if not stages:return RenderResult(False,None,0,"empty chain")
        with tempfile.TemporaryDirectory(prefix="aegis-re-chain-") as td:
            current=source;details=[]
            for index,stage in enumerate(stages):
                engine=self.registry.get(stage["engine"])
                if not engine:return RenderResult(False,None,time.monotonic()-start,f"unknown stage {stage['engine']}")
                stage_target=target if index==len(stages)-1 else Path(td)/f"stage-{index}.wav"
                result=engine.render(current,stage.get("parameters",{}),stage_target);details.append({"stage":stage,"result":asdict(result)})
                if not result.success:return RenderResult(False,None,time.monotonic()-start,f"stage {index}: {result.error}",{"stages":details})
                current=stage_target
        return RenderResult(True,str(target),time.monotonic()-start,details={"stages":details})


def engines(env:dict[str,str])->dict[str,AudioTransformEngine]:
    registry={"passthrough":PassthroughEngine(),"praat":PraatEngine(),"world":WorldEngine(),
              "rubberband":RubberBandEngine(env["AEGIS_GLADOS_RE_FFMPEG_BIN"]),"graillon_lv2":GraillonEngine(env)}
    registry["chain"]=ChainEngine(registry);return registry


def candidate_specs(config:dict[str,Any])->list[dict[str,Any]]:
    specs=[]
    for item in config["engines"]:
        if item.get("enabled",True):specs.append({"id":item["id"],"engine":item["type"],"parameters":dict(item.get("parameters",{})),"round":"A"})
    aggressive={"correction_amount":1.0,"flatten_amount":1.0,"preserve_drift":0.0,"preserve_modulation":0.0}
    for engine in ("praat","world"):
        for scale in SCALES:
            for transition in (0,25):
                params={**aggressive,"quantization":scale,"transition_ms":transition}
                if engine=="world":params.update(spectral_envelope_preserve=1.0,aperiodicity_preserve=1.0,formant_shift=0.0)
                specs.append({"id":f"{engine}-{scale.replace('_','-')}-t{transition}","engine":engine,"parameters":params,"round":"A"})
        if engine=="world":
            for shift in (-1.0,1.0):specs.append({"id":f"{engine}-chromatic-formant-{str(shift).replace('-','m').replace('.','p')}","engine":engine,"parameters":{**aggressive,"quantization":"chromatic","transition_ms":0,"formant_shift":shift,"spectral_envelope_preserve":1.0},"round":"B"})
    for scale in SCALES:
        specs.append({"id":f"graillon-{scale.replace('_','-')}","engine":"graillon_lv2","parameters":{"quantization":scale,"correction_amount":1.0,"smooth":0,"inertia":0,"wet":1,"snap_min":0,"snap_max":1},"round":"A"})
    specs += [{"id":"rubberband-minus-1","engine":"rubberband","parameters":{"pitch_shift_semitones":-1,"formant":"preserved"},"round":"A"},
              {"id":"rubberband-plus-1","engine":"rubberband","parameters":{"pitch_shift_semitones":1,"formant":"preserved"},"round":"A"}]
    hard={**aggressive,"quantization":"chromatic","transition_ms":0}
    specs += [
      {"id":"chain-praat-rubberband","engine":"chain","round":"C","parameters":{"stages":[{"engine":"praat","parameters":hard},{"engine":"rubberband","parameters":{"pitch_shift_semitones":.25,"formant":"preserved"}}]}},
      {"id":"chain-world-rubberband","engine":"chain","round":"C","parameters":{"stages":[{"engine":"world","parameters":{**hard,"spectral_envelope_preserve":1.0}},{"engine":"rubberband","parameters":{"pitch_shift_semitones":.25,"formant":"preserved"}}]}},
      {"id":"chain-praat-graillon-light","engine":"chain","round":"C","parameters":{"stages":[{"engine":"praat","parameters":hard},{"engine":"graillon_lv2","parameters":{"quantization":"chromatic","correction_amount":.3,"smooth":0,"inertia":0,"wet":.35}}]}},
      {"id":"chain-world-graillon-light","engine":"chain","round":"C","parameters":{"stages":[{"engine":"world","parameters":{**hard,"spectral_envelope_preserve":1.0}},{"engine":"graillon_lv2","parameters":{"quantization":"chromatic","correction_amount":.3,"smooth":0,"inertia":0,"wet":.35}}]}},
    ]
    seen=set();return [x for x in specs if not (x["id"] in seen or seen.add(x["id"]))]


def diagnostics(audio:Any,sr:int,reference_len:int)->dict[str,Any]:
    import numpy as np
    x=np.asarray(audio);diff=np.diff(x);peak=float(np.max(np.abs(x)));rms=float(np.sqrt(np.mean(x*x)))
    return {"clipping":bool(np.sum(np.abs(x)>=.999)>3),"empty":bool(rms<1e-6),"dc_offset":float(np.mean(x)),
            "click_count":int(np.sum(np.abs(diff)>max(.5,8*np.std(diff)))),"gap_fraction":float(np.mean(np.abs(x)<1e-6)),
            "duration_error_ms":(len(x)-reference_len)*1000/sr,"valid_duration":bool(len(x)>0),
            "ultrasonic_energy_fraction":0.0 if sr<=24000 else None,"peak":peak}


def compare_audio(candidate:Path,target:Path,sr:int,weights:dict[str,float])->dict[str,Any]:
    import librosa
    import numpy as np
    a,_=load_audio(candidate,sr);b,_=load_audio(target,sr);n=min(len(a),len(b));a,b=a[:n],b[:n]
    pa,pb=pitch_track(a,sr),pitch_track(b,sr);m=min(len(pa["f0"]),len(pb["f0"]));both=pa["voiced"][:m]&pb["voiced"][:m]
    cents=np.abs(1200*np.log2(np.maximum(pa["f0"][:m][both],1e-6)/np.maximum(pb["f0"][:m][both],1e-6))) if both.any() else np.array([1200.])
    vuv=float(np.mean(pa["voiced"][:m]==pb["voiced"][:m]));qa=np.round(pa["midi"][:m][both]);qb=np.round(pb["midi"][:m][both])
    pitch_error=float(np.median(np.minimum(cents,1200)));grid_error=float(np.mean(np.abs(qa-qb))) if len(qa) else 12
    sa=np.abs(librosa.stft(a,n_fft=1024,hop_length=240))+1e-8;sb=np.abs(librosa.stft(b,n_fft=1024,hop_length=240))+1e-8;k=min(sa.shape[1],sb.shape[1]);logdist=float(np.sqrt(np.mean((20*np.log10(sa[:,:k])-20*np.log10(sb[:,:k]))**2)))
    mca=librosa.feature.mfcc(y=a,sr=sr,n_mfcc=13);mcb=librosa.feature.mfcc(y=b,sr=sr,n_mfcc=13);k=min(mca.shape[1],mcb.shape[1]);mcd=float(np.mean(np.linalg.norm(mca[1:,:k]-mcb[1:,:k],axis=0)))
    rms_a=float(np.sqrt(np.mean(a*a)));rms_b=float(np.sqrt(np.mean(b*b)));dynamics=abs(20*math.log10(max(rms_a,1e-9)/max(rms_b,1e-9)))
    window=max(1,int(.02*sr));kernel=np.ones(window)/window
    ea=np.sqrt(np.convolve(a*a,kernel,"valid"))[::max(1,sr//200)];eb=np.sqrt(np.convolve(b*b,kernel,"valid"))[::max(1,sr//200)];corr=float(np.corrcoef(ea,eb)[0,1]) if len(ea)>2 else 0
    loud_a=loudness_metrics(candidate);loud_b=loudness_metrics(target);lufs_difference=abs(loud_a.get("integrated_lufs",-70)-loud_b.get("integrated_lufs",-70))
    diag=diagnostics(a,sr,len(b)); components={"pitch":min(1,pitch_error/600)*.75+min(1,grid_error/6)*.15+(1-vuv)*.10,
        "spectral_envelope":min(1,logdist/40)*.7+min(1,mcd/300)*.3,"dynamics":min(1,(dynamics+lufs_difference)/24),"timing":min(1,abs(len(a)-len(b))/sr)+max(0,1-corr),
        "artifacts_penalty":min(1,(1 if diag["clipping"] else 0)+(1 if diag["empty"] else 0)+diag["click_count"]/100)}
    score=sum(float(weights[k])*components[k] for k in weights)
    return {"composite_score":float(score),"components":components,"pitch":{"median_f0_error_cents":pitch_error,"quantized_note_error_semitones":grid_error,"voiced_unvoiced_agreement":vuv},
            "timbre":{"log_spectral_distance_db":logdist,"mel_cepstral_distance_proxy":mcd},"dynamics":{"rms_difference_db":dynamics,"lufs_difference":lufs_difference,"candidate_lufs":loud_a.get("integrated_lufs"),"target_lufs":loud_b.get("integrated_lufs")},"timing":{"envelope_correlation":corr,"duration_difference_ms":(len(a)-len(b))*1000/sr},"penalties":diag}


def loudness_metrics(path:Path)->dict[str,float]:
    proc=run(["ffmpeg","-hide_banner","-nostats","-i",str(path),"-af","ebur128=peak=true","-f","null","-"])
    matches=re.findall(r"\bI:\s*(-?[0-9.]+) LUFS",proc.stderr);peaks=re.findall(r"Peak:\s*(-?[0-9.]+) dBFS",proc.stderr)
    return {"integrated_lufs":float(matches[-1]) if matches else -70.0,"true_peak_dbfs":float(peaks[-1]) if peaks else -70.0}


def normalize_listening(source:Path,target:Path,env:dict[str,str])->None:
    # loudnorm is a two-pass-capable EBU R128 filter; here each comparison file
    # receives the same requested target and is never fed back into scoring.
    proc=run([env["AEGIS_GLADOS_RE_FFMPEG_BIN"],"-y","-v","error","-i",str(source),"-af","loudnorm=I=-16:TP=-1:LRA=11","-ac","1",str(target)])
    if proc.returncode:raise LabError(proc.stderr.strip())


def search(env:dict[str,str],config:dict[str,Any])->dict[str,Any]:
    p=paths(env);prepare(env);source=p["work"]/"clean-analysis.wav";target=p["work"]/"target-analysis.wav";sr=int(env["AEGIS_GLADOS_RE_SAMPLE_RATE"])
    registry=engines(env);specs=candidate_specs(config)[:min(int(env["AEGIS_GLADOS_RE_MAX_CANDIDATES"]),int(config.get("search",{}).get("max_candidates",80)))]
    prior={}
    if (p["out"]/"manifest.json").is_file():
        try:prior={x["id"]:x for x in json.loads((p["out"]/"manifest.json").read_text())["candidates"]}
        except (KeyError,json.JSONDecodeError):prior={}
    records=[]
    for i,spec in enumerate(specs,1):
        print(f"[{i}/{len(specs)}] {spec['id']} ({spec['engine']})",flush=True);raw=p["raw"]/f"{spec['id']}.wav";engine=registry.get(spec["engine"])
        cached=prior.get(spec["id"]);cache_valid=bool(cached and cached.get("parameters")==spec["parameters"] and cached.get("render",{}).get("success") and raw.is_file())
        if cache_valid:result=RenderResult(True,str(raw),0,details={"reused_from_previous_identical_search":True})
        elif engine is None:result=RenderResult(False,None,0,"unknown engine")
        elif not engine.preflight().get("available"):result=RenderResult(False,None,0,engine.preflight().get("reason","engine unavailable"))
        else:result=engine.render(source,spec["parameters"],raw)
        record={**spec,"render":asdict(result),"raw_file":str(raw.relative_to(p["out"])) if result.success else None}
        if result.success:
            try:
                audio,out_sr=load_audio(raw);record["validation"]={"sample_rate":out_sr,"channels":1,"duration":len(audio)/out_sr,**diagnostics(audio,out_sr,len(load_audio(target)[0]))}
                record["metrics"]=compare_audio(raw,target,sr,config["scoring"]);norm=p["normalized"]/f"{spec['id']}.wav";normalize_listening(raw,norm,env);record["normalized_file"]=str(norm.relative_to(p["out"]))
            except Exception as exc:record["render"]["success"]=False;record["render"]["error"]=f"post-processing: {exc}"
        records.append(record)
    success=[x for x in records if x["render"]["success"]];success.sort(key=lambda x:x["metrics"]["composite_score"])
    manifest={"created_at":datetime.now(timezone.utc).isoformat(),"source":str(source),"target":str(target),"candidate_count":len(records),"success_count":len(success),"engines":{k:v.preflight() for k,v in registry.items()},"candidates":records}
    (p["out"]/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    write_manifest_csv(p["out"]/"manifest.csv",records)
    rankings={"composite":[rank_row(x) for x in success],"pitch":[rank_row(x) for x in sorted(success,key=lambda x:x["metrics"]["components"]["pitch"])],"timbre":[rank_row(x) for x in sorted(success,key=lambda x:x["metrics"]["components"]["spectral_envelope"])]}
    (p["out"]/"rankings.json").write_text(json.dumps(rankings,indent=2),encoding="utf8");write_listening(p,records,env)
    return {"candidate_count":len(records),"success_count":len(success),"failed_count":len(records)-len(success),"top":rankings["composite"][:5]}


def rank_row(x:dict[str,Any])->dict[str,Any]:return {"id":x["id"],"engine":x["engine"],"parameters":x["parameters"],"score":x["metrics"]["composite_score"],"components":x["metrics"]["components"]}


def write_manifest_csv(path:Path,records:list[dict[str,Any]])->None:
    fields=["id","engine","round","success","score","pitch_score","timbre_score","clipping","error","parameters"]
    with path.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        for x in records:w.writerow({"id":x["id"],"engine":x["engine"],"round":x["round"],"success":x["render"]["success"],"score":x.get("metrics",{}).get("composite_score"),"pitch_score":x.get("metrics",{}).get("components",{}).get("pitch"),"timbre_score":x.get("metrics",{}).get("components",{}).get("spectral_envelope"),"clipping":x.get("validation",{}).get("clipping"),"error":x["render"].get("error"),"parameters":json.dumps(x["parameters"],sort_keys=True)})


def write_listening(p:dict[str,Path],records:list[dict[str,Any]],env:dict[str,str])->None:
    refnorm=p["normalized"]/"target-reference.wav";cleannorm=p["normalized"]/"clean-reference.wav";normalize_listening(p["work"]/"target-analysis.wav",refnorm,env);normalize_listening(p["work"]/"clean-analysis.wav",cleannorm,env)
    ok=[x for x in records if x["render"]["success"]];order=sorted(ok,key=lambda x:hashlib.sha256(("blind:"+x["id"]).encode()).hexdigest());blind={f"B{i:03d}":x["id"] for i,x in enumerate(order,1)};(p["out"]/"blind-map.json").write_text(json.dumps(blind,indent=2),encoding="utf8")
    criteria=["melodyne_similarity","glados_similarity","pitch","timbre","intelligibility","interesting_artifacts","bad_artifacts"]
    def page(blind_mode:bool)->str:
        seq=order if blind_mode else ok;cards=[]
        for i,x in enumerate(seq,1):
            label=f"B{i:03d}" if blind_mode else x["id"];details="" if blind_mode else f"<pre>{html.escape(json.dumps({'engine':x['engine'],'parameters':x['parameters'],'score':x['metrics']['composite_score']},indent=2))}</pre><details><summary>gráficos de referência</summary><img width='480' src='analysis/pitch-overlay.png'><img width='480' src='analysis/spectral-envelope-comparison.png'></details>"
            ratings=''.join(f'<label>{c} <input type="number" min="1" max="5" data-key="{c}"></label> ' for c in criteria)
            cards.append(f'<article data-id="{label}"><h2>{label}</h2><audio controls src="{x["normalized_file"]}"></audio>{details}<p>{ratings}</p><textarea placeholder="notas humanas"></textarea></article>')
        script="""<button onclick='exportNotes()'>Exportar notas JSON</button><script>function exportNotes(){let r={};document.querySelectorAll('article').forEach(a=>{r[a.dataset.id]={notes:a.querySelector('textarea').value};a.querySelectorAll('input').forEach(i=>r[a.dataset.id][i.dataset.key]=Number(i.value)||null)});let b=new Blob([JSON.stringify(r,null,2)],{type:'application/json'}),u=URL.createObjectURL(b),x=document.createElement('a');x.href=u;x.download='glados-listening-notes.json';x.click()}</script>"""
        return '<!doctype html><meta charset="utf-8"><title>GLaDOS listening</title><style>body{font:15px system-ui;max-width:1100px;margin:2rem}article{border-top:1px solid #aaa;padding:1rem}audio{display:block;margin:.5rem 0}label{white-space:nowrap}</style><h1>Comparação '+('cega' if blind_mode else 'identificada')+'</h1><p>Clean <audio controls src="candidates/normalized/clean-reference.wav"></audio> Target Melodyne <audio controls src="candidates/normalized/target-reference.wav"></audio></p>'+''.join(cards)+script
    (p["out"]/"listening.html").write_text(page(False),encoding="utf8");(p["out"]/"listening-blind.html").write_text(page(True),encoding="utf8")


def compare_candidate(env:dict[str,str],candidate_id:str)->dict[str,Any]:
    p=paths(env);manifest=json.loads((p["out"]/"manifest.json").read_text());x=next((x for x in manifest["candidates"] if x["id"]==candidate_id),None)
    if not x or not x["render"]["success"]:raise LabError("candidate not found or failed")
    config=read_yaml(DEFAULT_CONFIG);metrics=compare_audio(p["out"]/x["raw_file"],p["work"]/"target-analysis.wav",int(env["AEGIS_GLADOS_RE_SAMPLE_RATE"]),config["scoring"])
    out=p["diagnostics"]/candidate_id;out.mkdir(exist_ok=True);(out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf8")
    make_candidate_plots(p["out"]/x["raw_file"],p["work"]/"target-analysis.wav",out,int(env["AEGIS_GLADOS_RE_SAMPLE_RATE"]))
    return metrics


def make_candidate_plots(candidate:Path,target:Path,out:Path,sr:int)->None:
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    a,_=load_audio(candidate,sr);b,_=load_audio(target,sr);pa=pitch_track(a,sr);pb=pitch_track(b,sr)
    fig,ax=plt.subplots(figsize=(10,3));ax.plot(pa["time"],np.where(pa["voiced"],pa["f0"],np.nan),label="candidate");ax.plot(pb["time"],np.where(pb["voiced"],pb["f0"],np.nan),label="target");ax.legend();fig.tight_layout();fig.savefig(out/"pitch-overlay.png",dpi=120);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3));n=min(len(a),len(b));t=np.arange(n)/sr;ax.plot(t,a[:n],alpha=.5,label="candidate");ax.plot(t,b[:n],alpha=.5,label="target");ax.legend();fig.tight_layout();fig.savefig(out/"waveform-overlay.png",dpi=120);plt.close(fig)


def normalized_reference_score(candidate:Any, reference:Any, baseline:Any, epsilon:float=1e-9)->float:
    """Distance ratio where reference=0 and unchanged baseline=1."""
    import numpy as np
    c=np.asarray(candidate,dtype=float);r=np.asarray(reference,dtype=float);b=np.asarray(baseline,dtype=float)
    denominator=float(np.linalg.norm(b-r))
    if denominator<epsilon:return 0.0 if np.linalg.norm(c-r)<epsilon else float("inf")
    return float(np.linalg.norm(c-r)/denominator)


def effect_feature_vectors(audio:Any,sr:int)->dict[str,Any]:
    """Orthogonal-ish feature spaces for the three Melodyne controls.

    They deliberately remain separate: pitch centering measures grid/location,
    modulation measures frame derivatives and local residuals, while formants
    use vocal-tract/spectral features and never reward pitch accuracy.
    """
    import numpy as np
    import scipy.fft
    import scipy.signal
    track=pitch_track(audio,sr);v=track["voiced"];midi=track["midi"];values=midi[v]
    rounded=np.round(values);distribution=np.array([np.mean((rounded.astype(int)%12)==n) for n in range(12)]) if len(values) else np.zeros(12)
    center=np.r_[np.nanmean(np.abs(track["distance_cents"][v]))/50,np.nanmedian(values)/12,distribution]
    derivatives=[];residuals=[]
    for a,b in regions(v):
        local=midi[a:b]
        if len(local)<3:continue
        derivatives.extend(np.diff(local))
        width=min(21,len(local) if len(local)%2 else len(local)-1);trend=scipy_smooth(local,max(1,width));residuals.extend(local-trend)
    d=np.asarray(derivatives) if derivatives else np.zeros(1);res=np.asarray(residuals) if residuals else np.zeros(1)
    modulation=np.array([np.mean(np.abs(d)),np.std(d),np.percentile(np.abs(d),95),np.std(res),np.mean(np.abs(res)),pitch_summary(track)["note_change_rate_hz"]/20])
    # Compact spectral statistics keep optimization scoring fast.  The full
    # analysis path still computes HPSS/MFCC plots, but doing that for every
    # factorial context would dominate trial time without improving ranking.
    x=np.asarray(audio,dtype=float);freqs,_,stft=scipy.signal.stft(x,fs=sr,nperseg=1024,noverlap=784,boundary=None)
    power=np.mean(np.abs(stft)**2,axis=1)+1e-12;total=float(np.sum(power));power=power/total
    centroid=float(np.sum(freqs*power))
    cumulative=np.cumsum(power);rolloff=float(freqs[min(len(freqs)-1,int(np.searchsorted(cumulative,.85*cumulative[-1])))])
    valid=(freqs>=100)&(freqs<=min(10000,sr/2));tilt=float(np.polyfit(np.log10(freqs[valid]),10*np.log10(power[valid]),1)[0])
    flatness=float(np.exp(np.mean(np.log(power)))/np.mean(power))
    # Exclude c0: it represents global spectral energy and previously made
    # harmless renderer gain differences dominate the formant objective.
    log_shape=np.log(power)-np.mean(np.log(power));cepstral=scipy.fft.dct(log_shape,norm="ortho")[1:13]/100
    fm=formants(audio,sr)
    formant=np.r_[[fm[f"F{i}_median_hz"]/5000 for i in (1,2,3)],centroid/12000,rolloff/12000,tilt/20,flatness,cepstral]
    return {"pitch_centralization":center.tolist(),"pitch_modulation":modulation.tolist(),"formant_shift":formant.tolist()}


def analyze_effect_references(env:dict[str,str],config_path:Path=EFFECT_CONFIG)->dict[str,Any]:
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    config=read_yaml_unvalidated(config_path);root=ROOT/config["root"];p=paths(env);out=p["analysis"]/"effects";out.mkdir(parents=True,exist_ok=True);sr=int(env["AEGIS_GLADOS_RE_SAMPLE_RATE"])
    refs=config["references"];missing=[];features={};properties={};tracks={}
    for name,item in refs.items():
        path=root/item["file"]
        if not path.is_file():missing.append({"combination":name,"path":str(path)});continue
        audio,_=load_audio(path,sr);features[name]=effect_feature_vectors(audio,sr);track=pitch_track(audio,sr);tracks[name]=track
        ps=pitch_summary(track);fm=formants(audio,sr);properties[name]={"file":str(path),"effects":item["effects"],"sha256":sha256(path),"duration_seconds":len(audio)/sr,
            "pitch":{"distance_to_grid_cents":ps["mean_abs_distance_to_chromatic_cents"],"note_change_rate_hz":ps["note_change_rate_hz"],"median_f0_hz":ps["median_f0_hz"]},"formants":fm,
            "confirmed_by_operator":bool(item.get("confirmed_by_operator",False))}
    required={"none","C","M","F","CM","FM","CF","CFM"}
    complete=required.issubset(features)
    scores={}
    singleton={"pitch_centralization":"C","pitch_modulation":"M","formant_shift":"F"}
    for combo,vectors in features.items():
        scores[combo]={effect:normalized_reference_score(vectors[effect],features[ref][effect],features["none"][effect]) for effect,ref in singleton.items()}
        scores[combo]["against_full_target"]={effect:normalized_reference_score(vectors[effect],features["CFM"][effect],features["none"][effect]) for effect in singleton}
    # Feature-domain factorial interaction: zero means additive independent effects.
    interactions={}
    if complete:
        for effect in singleton:
            x={k:np.asarray(features[k][effect]) for k in required}
            scale=np.linalg.norm(x["CFM"]-x["none"])+1e-9
            interactions[effect]={"CM":float(np.linalg.norm((x["CM"]-x["none"])-(x["C"]-x["none"])-(x["M"]-x["none"]))/scale),
              "CF":float(np.linalg.norm((x["CF"]-x["none"])-(x["C"]-x["none"])-(x["F"]-x["none"]))/scale),
              "FM":float(np.linalg.norm((x["FM"]-x["none"])-(x["F"]-x["none"])-(x["M"]-x["none"]))/scale),
              "CFM":float(np.linalg.norm((x["CFM"]-x["none"])-(x["C"]-x["none"])-(x["F"]-x["none"])-(x["M"]-x["none"]))/scale)}
    distance_matrix={}
    if complete and config.get("scoring"):
        import glados_optimization
        distance_matrix=glados_optimization.informative_matrix(features,config)
    payload={"feature_schema_version":EFFECT_FEATURE_SCHEMA_VERSION,
        "scale":{"perfect_reference":0,"unchanged_passthrough":1,"worse_than_passthrough":">1","clipped":False},"complete_factorial":complete,"missing":missing,"properties":properties,"feature_vectors":features,"normalized_scores":scores,"factorial_interactions":interactions,"informative_distance_matrix":distance_matrix}
    (out/"effect-reference-analysis.json").write_text(json.dumps(payload,indent=2),encoding="utf8")
    with (out/"effect-reference-scores.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,["combination","effects","centralization_score","modulation_score","formant_score"]);w.writeheader()
        for combo in refs:
            if combo in scores:w.writerow({"combination":combo,"effects":"+".join(refs[combo]["effects"]) or "none","centralization_score":scores[combo]["pitch_centralization"],"modulation_score":scores[combo]["pitch_modulation"],"formant_score":scores[combo]["formant_shift"]})
    labels=[x for x in refs if x in scores];x=np.arange(len(labels));fig,ax=plt.subplots(figsize=(11,4));width=.25
    for index,effect in enumerate(singleton):ax.bar(x+(index-1)*width,[scores[k][effect] for k in labels],width,label=effect)
    ax.axhline(1,color="black",ls="--",lw=1,label="passthrough=1");ax.set_xticks(x,labels);ax.set_ylabel("normalized distance to isolated reference");ax.legend();fig.tight_layout();fig.savefig(out/"effect-scores.png",dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    for name in labels:
        tr=tracks[name];ax.plot(tr["time"],np.where(tr["voiced"],tr["f0"],np.nan),label=name,alpha=.75)
    ax.set_ylabel("F0 Hz");ax.legend(ncol=4);fig.tight_layout();fig.savefig(out/"factorial-pitch-overlay.png",dpi=130);plt.close(fig)
    return payload


def compare_effect_candidate(env:dict[str,str],candidate_id:str)->dict[str,Any]:
    p=paths(env);effect_file=p["analysis"]/"effects/effect-reference-analysis.json"
    if not effect_file.is_file():analyze_effect_references(env)
    analysis=json.loads(effect_file.read_text());manifest=json.loads((p["out"]/"manifest.json").read_text())
    item=next((x for x in manifest["candidates"] if x["id"]==candidate_id and x["render"]["success"]),None)
    if not item:raise LabError("candidate not found or failed")
    audio,sr=load_audio(p["out"]/item["raw_file"],int(env["AEGIS_GLADOS_RE_SAMPLE_RATE"]));vectors=effect_feature_vectors(audio,sr)
    baseline=analysis["feature_vectors"]["none"];target=analysis["feature_vectors"]["CFM"]
    scores={effect:normalized_reference_score(vectors[effect],target[effect],baseline[effect]) for effect in ("pitch_centralization","pitch_modulation","formant_shift")}
    payload={"candidate":candidate_id,"scale":analysis["scale"],"scores":scores,"mean_three_effects":sum(scores.values())/3,
             "interpretation":"0 matches the full Melodyne target; 1 equals unchanged clean; values above 1 are worse than passthrough"}
    out=p["diagnostics"]/candidate_id;out.mkdir(exist_ok=True);(out/"effect-scores.json").write_text(json.dumps(payload,indent=2),encoding="utf8")
    return payload


def read_yaml_unvalidated(path:Path)->dict[str,Any]:
    import yaml
    value=yaml.safe_load(path.read_text(encoding="utf8"))
    if not isinstance(value,dict):raise LabError("effect YAML root must be a mapping")
    return value


def render_candidate(env:dict[str,str],config:dict[str,Any],candidate_id:str)->dict[str,Any]:
    p=paths(env);prepare(env);spec=next((x for x in candidate_specs(config) if x["id"]==candidate_id),None)
    if not spec:raise LabError("candidate id not found")
    engine=engines(env)[spec["engine"]];target=p["raw"]/f"{candidate_id}.wav";result=engine.render(p["work"]/"clean-analysis.wav",spec["parameters"],target)
    return {"candidate":spec,"render":asdict(result)}


def generate_report(env:dict[str,str])->Path:
    p=paths(env);analysis=json.loads((p["analysis"]/"clean-vs-target.json").read_text());manifest=json.loads((p["out"]/"manifest.json").read_text());rank=json.loads((p["out"]/"rankings.json").read_text());mpd=json.loads((p["analysis"]/"melodyne-project-summary.json").read_text())
    failed=[x for x in manifest["candidates"] if not x["render"]["success"]];clipped=[x["id"] for x in manifest["candidates"] if x.get("validation",{}).get("clipping")];near_target=[x["id"] for x in manifest["candidates"] if x.get("metrics",{}).get("composite_score",1)<.02];near_original=[x["id"] for x in manifest["candidates"] if x["engine"]=="passthrough"]
    def table(rows:list[dict[str,Any]])->str:
        return "\n".join(["| ID | engine | score | pitch | timbre |","|---|---|---:|---:|---:|"]+[f"| {x['id']} | {x['engine']} | {x['score']:.5f} | {x['components']['pitch']:.5f} | {x['components']['spectral_envelope']:.5f} |" for x in rows[:10]])
    lines=["# GLaDOS reverse-engineering laboratory","","> Ranking automático auxilia a triagem; nenhum vencedor definitivo é declarado sem avaliação humana.","","## 1. Arquivos de entrada","","```json",json.dumps(analysis["input"]["source"],indent=2),"```","","## 2. Alinhamento","","```json",json.dumps(analysis["timing"],indent=2),"```","","## 3. Clean versus target","","```json",json.dumps({"pitch":analysis["pitch"],"dynamics":analysis["dynamics"],"timbre_summary":{k:{q:v for q,v in val.items() if q!='spectral_envelope_db'} for k,val in analysis["timbre"].items()}},indent=2),"```","","## 4. Projeto MPD","",f"Formato: {mpd['format']}; versão: {mpd.get('melodyne_version',{}).get('value','não interpretada')} ({mpd.get('melodyne_version',{}).get('confidence','n/a')}); chunks zlib: {len(mpd['chunks'])}; strings: {len(mpd['strings'])}; confiança geral: {mpd['confidence']}.","","## 5–6. Motores disponíveis/indisponíveis","","```json",json.dumps(manifest["engines"],indent=2),"```","","## 7. Configurações testadas","",f"{manifest['candidate_count']} candidatos; {manifest['success_count']} renderizados.","","```json",json.dumps([{"id":x["id"],"engine":x["engine"],"parameters":x["parameters"]} for x in manifest["candidates"]],indent=2),"```","","## 8. Ranking por pitch","",table(rank["pitch"]),"","## 9. Ranking por timbre","",table(rank["timbre"]),"","## 10. Ranking composto","",table(rank["composite"]),"","## 11. Clipping","",", ".join(clipped) or "Nenhum.","","## 12. Falhas","",json.dumps([{"id":x["id"],"error":x["render"]["error"]} for x in failed],indent=2),"","## 13. Quase idênticos ao original/target","",f"Original: {', '.join(near_original) or 'nenhum'}. Target (limiar composto 0,02): {', '.join(near_target) or 'nenhum'}.","","## 14. Mais próximos do target","",", ".join(x["id"] for x in rank["composite"][:5]),"","## 15. Diferenças ainda não reproduzidas","","A avaliação objetiva separa pitch, envelope, dinâmica, timing e penalidades; diferenças perceptuais de transientes, consoantes e artefatos semelhantes ao Melodyne ainda exigem a escuta cega.","","## 16. Páginas de escuta","",f"- `{p['out']/ 'listening.html'}`",f"- `{p['out']/ 'listening-blind.html'}`",""]
    target=p["report"]/"summary.md";target.write_text("\n".join(lines),encoding="utf8");return target


def command_preflight(env:dict[str,str])->dict[str,Any]:
    p=paths(env);checks={"not_root":os.geteuid()!=0,"clean_wav":p["clean"].is_file(),"target_wav":p["target"].is_file(),"melodyne_project":p["project"].is_file(),"config":DEFAULT_CONFIG.is_file()}
    for binary in (env["AEGIS_GLADOS_RE_FFMPEG_BIN"],env["AEGIS_GLADOS_RE_FFPROBE_BIN"]):checks[f"binary:{binary}"]=shutil.which(binary) is not None
    try:config=read_yaml(DEFAULT_CONFIG);checks["yaml_valid"]=True
    except Exception as exc:checks["yaml_valid"]=False;config={};checks["yaml_error"]=str(exc)
    available={k:v.preflight() for k,v in engines(env).items()};payload={"checks":checks,"engines":available,"effective_paths":{k:str(v) for k,v in p.items()},"precedence":"CLI > process environment > .env.reverse-engineering > .env.graillon > defaults"}
    return payload


def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument("--clean");ap.add_argument("--target");ap.add_argument("--project");ap.add_argument("--output-dir");sub=ap.add_subparsers(dest="command",required=True)
    sub.add_parser("preflight");sub.add_parser("analyze");e=sub.add_parser("analyze-effects");e.add_argument("--config",default=str(EFFECT_CONFIG));s=sub.add_parser("search");s.add_argument("--config",default=str(DEFAULT_CONFIG));sub.add_parser("report");r=sub.add_parser("render");r.add_argument("--candidate",required=True);c=sub.add_parser("compare");c.add_argument("--candidate",required=True);ce=sub.add_parser("compare-effects");ce.add_argument("--candidate",required=True)
    o=sub.add_parser("optimize");o.add_argument("--config",default=str(EFFECT_CONFIG));o.add_argument("--trials",type=int,required=True);o.add_argument("--max-total-trials",type=int,default=1000);o.add_argument("--resume",action="store_true")
    for name in ("optimization-status","optimization-report","rebuild-dashboard"):
        parser=sub.add_parser(name);parser.add_argument("--config",default=str(EFFECT_CONFIG))
    archive=sub.add_parser("optimization-archive-invalid");archive.add_argument("--reason",required=True)
    args=ap.parse_args();env=environment(args)
    try:
        if args.command=="preflight":payload=command_preflight(env)
        elif args.command=="analyze":payload=analyze_pair(env)
        elif args.command=="analyze-effects":payload=analyze_effect_references(env,Path(args.config))
        elif args.command=="search":payload=search(env,read_yaml(Path(args.config)))
        elif args.command=="optimize":
            import glados_optimization
            payload=glados_optimization.optimize(env,Path(args.config),trials=args.trials,max_total_trials=args.max_total_trials,resume=args.resume,lab=sys.modules[__name__])
        elif args.command=="optimization-status":
            import glados_optimization
            payload=glados_optimization.status(env,Path(args.config),lab=sys.modules[__name__])
        elif args.command=="optimization-report":
            import glados_optimization
            payload=glados_optimization.report(env,Path(args.config))
        elif args.command=="rebuild-dashboard":
            import glados_optimization
            payload={"dashboard":str(glados_optimization.rebuild_dashboard(env,glados_optimization.load_config(Path(args.config))))}
        elif args.command=="optimization-archive-invalid":
            import glados_optimization
            payload=glados_optimization.archive_invalid_optimization(env,args.reason)
        elif args.command=="report":payload={"report":str(generate_report(env))}
        elif args.command=="render":payload=render_candidate(env,read_yaml(DEFAULT_CONFIG),args.candidate)
        elif args.command=="compare":payload=compare_candidate(env,args.candidate)
        else:payload=compare_effect_candidate(env,args.candidate)
        print(json.dumps(payload,indent=2,ensure_ascii=False));return 0
    except (LabError,OSError,ValueError,json.JSONDecodeError) as exc:
        print(f"glados_reverse_engineer: {exc}",file=sys.stderr);return 2


if __name__=="__main__":raise SystemExit(main())
