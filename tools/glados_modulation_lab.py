#!/usr/bin/env python3
"""Factorial, target-safe laboratory for Melodyne Pitch Modulation (effect M).

Targets are loaded only by preparation/evaluation code.  Engine.render has the
deliberately narrow source/output/parameters interface required for inference on
new audio, making source-target morphing impossible through the public API.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import inspect
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "experiments/glados-filter-lab"
DEFAULT_CONFIG = ROOT / "config/glados-modulation-search.yaml"
DEFAULT_DATASET_CONFIG = ROOT / "config/glados-factorial-dataset.yaml"
STATE_IDS = ("clean", "C", "M", "F", "CM", "CF", "FM", "CFM")
CONTEXTS = (("clean-to-M", "clean", "M"), ("C-to-CM", "C", "CM"),
            ("F-to-FM", "F", "FM"), ("CF-to-CFM", "CF", "CFM"))
TRANSFORM_VERSION = "modulation-v1"
ENGINE_VERSION = {"world": "world-modulation-v1", "praat": "praat-modulation-v1"}

DEFAULTS = {
    **{f"AEGIS_GLADOS_AUDIO_{state.upper()}": str(LAB / f"reference/{state}.wav") for state in STATE_IDS},
    "AEGIS_GLADOS_MOD_CONFIG": str(DEFAULT_CONFIG),
    "AEGIS_GLADOS_MOD_DATASET_CONFIG": str(DEFAULT_DATASET_CONFIG),
    "AEGIS_GLADOS_MOD_OUTPUT_DIR": str(LAB / "modulation-results"),
    "AEGIS_GLADOS_MOD_REPORT_DIR": str(ROOT / "reports/glados-lab/modulation"),
    "AEGIS_GLADOS_MOD_SAMPLE_RATE": "24000", "AEGIS_GLADOS_MOD_CHANNELS": "1",
    "AEGIS_GLADOS_MOD_FRAME_PERIOD_MS": "5.0", "AEGIS_GLADOS_MOD_ENABLE_WORLD": "1",
    "AEGIS_GLADOS_MOD_ENABLE_PRAAT": "1", "AEGIS_GLADOS_MOD_ENABLE_GRAILLON_BASELINE": "0",
    "AEGIS_GLADOS_MOD_WORKERS": "1", "AEGIS_GLADOS_MOD_BATCH_TRIALS": "200",
    "AEGIS_GLADOS_MOD_MAX_TOTAL_TRIALS": "1000", "AEGIS_GLADOS_MOD_SEED": "40733133",
    "AEGIS_GLADOS_MOD_KEEP_INTERMEDIATE": "1", "AEGIS_GLADOS_MOD_NORMALIZE_LISTENING_FILES": "1",
    "AEGIS_GLADOS_MOD_LOG_LEVEL": "INFO", "AEGIS_GLADOS_MOD_FFMPEG_BIN": "ffmpeg",
    "AEGIS_GLADOS_MOD_FFPROBE_BIN": "ffprobe",
}
BOOL_KEYS = {x for x in DEFAULTS if "ENABLE_" in x or x.endswith(("INTERMEDIATE", "FILES"))}
INT_KEYS = {"AEGIS_GLADOS_MOD_SAMPLE_RATE", "AEGIS_GLADOS_MOD_CHANNELS", "AEGIS_GLADOS_MOD_WORKERS",
            "AEGIS_GLADOS_MOD_BATCH_TRIALS", "AEGIS_GLADOS_MOD_MAX_TOTAL_TRIALS", "AEGIS_GLADOS_MOD_SEED"}


class LabError(RuntimeError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_bool(value: str) -> bool:
    if value.lower() in {"1", "true", "yes", "on"}: return True
    if value.lower() in {"0", "false", "no", "off"}: return False
    raise LabError(f"invalid boolean: {value!r}")


def load_dotenv(path: Path, allowed: set[str] | None = None) -> dict[str, str]:
    """Read dotenv as inert data: no source, interpolation, quoting or commands."""
    result: dict[str, str] = {}
    if not path.is_file(): return result
    for number, raw in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"): continue
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=([^\x00\r\n]*)", line)
        if not match: raise LabError(f"{path}:{number}: invalid dotenv line; expected KEY=value")
        key, value = match.groups()
        if key not in (allowed or set(DEFAULTS)): raise LabError(f"{path}:{number}: unknown key {key}")
        if value.startswith(("'", '"')) or "$" in value or "`" in value or "\\" in value:
            raise LabError(f"{path}:{number}: shell syntax/expansion is forbidden")
        result[key] = value
    return result


def environment(cli: argparse.Namespace | None = None) -> dict[str, str]:
    # defaults <- legacy dotenv <- modulation dotenv <- process <- CLI
    data = dict(DEFAULTS)
    data.update(load_dotenv(LAB / ".env.reverse-engineering", set(DEFAULTS)))
    data.update(load_dotenv(LAB / ".env.modulation", set(DEFAULTS)))
    data.update({key: os.environ[key] for key in DEFAULTS if key in os.environ})
    if cli:
        for attr, key in (("config", "AEGIS_GLADOS_MOD_CONFIG"), ("dataset_config", "AEGIS_GLADOS_MOD_DATASET_CONFIG"),
                          ("output_dir", "AEGIS_GLADOS_MOD_OUTPUT_DIR"), ("report_dir", "AEGIS_GLADOS_MOD_REPORT_DIR")):
            value = getattr(cli, attr, None)
            if value: data[key] = value
    for key in BOOL_KEYS: parse_bool(data[key])
    for key in INT_KEYS:
        try: int(data[key])
        except ValueError as exc: raise LabError(f"{key} must be an integer") from exc
    try: float(data["AEGIS_GLADOS_MOD_FRAME_PERIOD_MS"])
    except ValueError as exc: raise LabError("AEGIS_GLADOS_MOD_FRAME_PERIOD_MS must be numeric") from exc
    if int(data["AEGIS_GLADOS_MOD_CHANNELS"]) != 1: raise LabError("modulation laboratory requires mono output")
    return data


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml
    if not path.is_file(): raise LabError(f"configuration does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf8"))
    if not isinstance(value, dict): raise LabError(f"{path}: YAML root must be a mapping")
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf8") as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"); handle.flush(); os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def output_paths(env: dict[str, str]) -> dict[str, Path]:
    out = Path(env["AEGIS_GLADOS_MOD_OUTPUT_DIR"]); report = Path(env["AEGIS_GLADOS_MOD_REPORT_DIR"])
    return {"out": out, "work": out / "work", "analysis": out / "analysis", "baselines": out / "baselines",
            "raw": out / "candidates/raw", "normalized": out / "candidates/normalized",
            "diagnostics": out / "candidates/diagnostics", "cache": out / "cache", "report": report}


def ensure_dirs(paths: dict[str, Path]) -> None:
    for path in paths.values(): path.mkdir(parents=True, exist_ok=True)


def load_audio(path: Path, sample_rate: int | None = None) -> tuple[Any, int]:
    import numpy as np
    import scipy.signal
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float64", always_2d=True)
    if not len(audio): raise LabError(f"empty audio: {path}")
    mono = np.mean(audio, axis=1, dtype=np.float64)
    if sample_rate and sr != sample_rate:
        divisor = math.gcd(int(sr), int(sample_rate))
        mono = scipy.signal.resample_poly(mono, sample_rate // divisor, sr // divisor, window=("kaiser", 8.6))
        sr = sample_rate
    return np.ascontiguousarray(mono, dtype=np.float64), int(sr)


def write_audio(path: Path, audio: Any, sample_rate: int, subtype: str = "FLOAT") -> None:
    import numpy as np
    import soundfile as sf
    value = np.asarray(audio, dtype=np.float64).reshape(-1)
    if not len(value) or not np.all(np.isfinite(value)): raise LabError("empty, NaN or Inf audio output")
    path.parent.mkdir(parents=True, exist_ok=True); sf.write(path, value, sample_rate, subtype=subtype)


def probe(path: Path) -> dict[str, Any]:
    import soundfile as sf
    if not path.is_file(): raise LabError(f"audio does not exist: {path}")
    try: info = sf.info(path); audio, _ = load_audio(path)
    except Exception as exc: raise LabError(f"invalid audio {path}: {exc}") from exc
    if info.frames <= 0 or not len(audio): raise LabError(f"audio contains no frames: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path), "sample_rate": info.samplerate,
            "channels": info.channels, "frames": info.frames, "duration": info.duration, "format": info.format}


def load_dataset(env: dict[str, str], config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or Path(env["AEGIS_GLADOS_MOD_DATASET_CONFIG"]); config = read_yaml(path)
    states = config.get("dataset", {})
    if tuple(states) != STATE_IDS: raise LabError(f"dataset must define states in canonical order: {STATE_IDS}")
    resolved: dict[str, Path] = {}; reverse: dict[Path, str] = {}
    for state in STATE_IDS:
        entry = states[state]
        if not isinstance(entry, dict) or set(entry) != {"path_env"}: raise LabError(f"dataset.{state} must contain only path_env")
        key = entry["path_env"]
        if key not in DEFAULTS or not key.startswith("AEGIS_GLADOS_AUDIO_"): raise LabError(f"dataset.{state}: unknown path_env {key}")
        audio = Path(env[key]).resolve()
        if audio in reverse and reverse[audio] != state: raise LabError(f"same file mapped to incompatible states {reverse[audio]} and {state}: {audio}")
        reverse[audio] = state; resolved[state] = audio
    contexts = config.get("effects", {}).get("modulation", {}).get("contexts", [])
    actual = tuple((x.get("id"), x.get("source"), x.get("target")) for x in contexts)
    if actual != CONTEXTS: raise LabError(f"modulation contexts must be exactly {CONTEXTS}")
    return {"config": str(path), "states": resolved, "contexts": contexts}


def active_bounds(audio: Any, sr: int) -> tuple[int, int]:
    import numpy as np
    frame = max(1, int(.02 * sr)); env = np.sqrt(np.convolve(np.asarray(audio) ** 2, np.ones(frame) / frame, mode="same"))
    threshold = max(10 ** (-55 / 20), float(env.max(initial=0)) * .003); indices = np.flatnonzero(env > threshold)
    return (int(indices[0]), int(indices[-1] + 1)) if len(indices) else (0, len(audio))


def align_audio(source: Any, target: Any, sr: int) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np
    import scipy.signal
    hop = max(1, sr // 200)
    def envelope(value: Any) -> Any:
        n = len(value) // hop
        if not n: return np.zeros(1)
        x = np.asarray(value[:n * hop]).reshape(n, hop); return np.sqrt(np.mean(x * x, axis=1))
    left, right = envelope(source), envelope(target)
    corr = scipy.signal.correlate(right - right.mean(), left - left.mean(), mode="full", method="fft")
    lag_frames = int(np.argmax(corr) - (len(left) - 1)); lag = lag_frames * hop
    source_start, target_start = max(0, -lag), max(0, lag)
    length = min(len(source) - source_start, len(target) - target_start)
    if length <= 0: raise LabError("alignment produced no overlapping audio")
    sb, se = active_bounds(source, sr); tb, te = active_bounds(target, sr)
    denom = float(np.linalg.norm(left - left.mean()) * np.linalg.norm(right - right.mean())) or 1.0
    details = {"lag_samples": lag, "lag_ms": lag * 1000 / sr, "source_trim_samples": source_start,
               "target_trim_samples": target_start, "source_duration_seconds": len(source) / sr,
               "target_duration_seconds": len(target) / sr, "duration_difference_ms": (len(target)-len(source))*1000/sr,
               "aligned_samples": length, "aligned_duration_seconds": length / sr,
               "envelope_correlation": float(np.max(corr) / denom),
               "source_active_ms": [sb*1000/sr, se*1000/sr], "target_active_ms": [tb*1000/sr, te*1000/sr],
               "method": "global energy-envelope offset only; no time warping"}
    return source[source_start:source_start+length], target[target_start:target_start+length], details


def preflight(env: dict[str, str], *, prepare_audio: bool = True) -> dict[str, Any]:
    paths = output_paths(env); ensure_dirs(paths); dataset = load_dataset(env)
    properties, hashes = {}, {}
    for state, path in dataset["states"].items():
        properties[state] = probe(path); hashes.setdefault(properties[state]["sha256"], []).append(state)
    conflicts = [states for states in hashes.values() if len(states) > 1]
    if conflicts: raise LabError(f"identical audio mapped to incompatible states: {conflicts}")
    duration_span = max(x["duration"] for x in properties.values()) - min(x["duration"] for x in properties.values())
    if duration_span > .25: raise LabError(f"dataset durations are incompatible (span {duration_span:.3f}s)")
    engines = {"world": WorldModulationEngine(env).preflight(), "praat": PraatModulationEngine(env).preflight()}
    pairs = []
    if prepare_audio:
        sr = int(env["AEGIS_GLADOS_MOD_SAMPLE_RATE"])
        for context in dataset["contexts"]:
            cid, source_id, target_id = context["id"], context["source"], context["target"]
            source, _ = load_audio(dataset["states"][source_id], sr); target, _ = load_audio(dataset["states"][target_id], sr)
            directory = paths["work"] / cid; directory.mkdir(parents=True, exist_ok=True)
            write_audio(directory / "source-analysis.wav", source, sr); write_audio(directory / "target-analysis.wav", target, sr)
            aligned_source, aligned_target, alignment = align_audio(source, target, sr)
            write_audio(directory / "source-aligned.wav", aligned_source, sr)
            write_audio(directory / "target-aligned.wav", aligned_target, sr)
            write_json(directory / "alignment.json", alignment)
            pairs.append({"id": cid, "source": source_id, "target": target_id, "alignment": alignment,
                          "same_speech_proxy": alignment["envelope_correlation"] > .30})
    payload = {"ready": all(x["available"] for name, x in engines.items() if parse_bool(env[f"AEGIS_GLADOS_MOD_ENABLE_{name.upper()}"])),
               "dataset_config": dataset["config"], "states": properties, "pairs": pairs,
               "duration_span_seconds": duration_span, "normalization": {"sample_rate": int(env["AEGIS_GLADOS_MOD_SAMPLE_RATE"]), "channels": 1},
               "engines": engines, "render_signature": str(inspect.signature(ModulationRemovalEngine.render)),
               "target_safe": "target" not in inspect.signature(ModulationRemovalEngine.render).parameters}
    write_json(paths["analysis"] / "preflight.json", payload); return payload


def hz_to_cents(f0: Any, reference: float = 1.0) -> Any:
    import numpy as np
    values = np.asarray(f0, dtype=np.float64); result = np.full(values.shape, np.nan)
    mask = np.isfinite(values) & (values > 0); result[mask] = 1200 * np.log2(values[mask] / reference); return result


def cents_to_hz(cents: Any, reference: float = 1.0) -> Any:
    import numpy as np
    values = np.asarray(cents, dtype=np.float64); result = np.zeros(values.shape)
    mask = np.isfinite(values); result[mask] = reference * np.power(2.0, values[mask] / 1200); return result


def contiguous_regions(voiced: Any, cents: Any | None = None, *, split_jump_cents: float = 300,
                       minimum_frames: int = 1) -> list[tuple[int, int]]:
    import numpy as np
    mask = np.asarray(voiced, dtype=bool); pitch = np.asarray(cents) if cents is not None else None
    result: list[tuple[int, int]] = []; start: int | None = None
    for index, active in enumerate(mask):
        split = bool(active and start is not None and pitch is not None and index > 0 and
                     np.isfinite(pitch[index]) and np.isfinite(pitch[index-1]) and abs(pitch[index]-pitch[index-1]) > split_jump_cents)
        if (not active or split) and start is not None:
            if index - start >= minimum_frames: result.append((start, index))
            start = index if split else None
        elif active and start is None: start = index
    if start is not None and len(mask)-start >= minimum_frames: result.append((start, len(mask)))
    return result


def _odd_window(window_ms: float, frame_ms: float, length: int, minimum: int = 3) -> int:
    size = max(minimum, int(round(window_ms / frame_ms))); size += 1 - size % 2
    largest = length if length % 2 else length - 1
    return max(1, min(size, largest))


def estimate_trend(values: Any, method: str, *, frame_ms: float, window_ms: float,
                   cutoff_hz: float = 2.0, savgol_order: int = 3, spline_smoothing: float = 1.0) -> Any:
    import numpy as np
    import scipy.interpolate
    import scipy.ndimage
    import scipy.signal
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 3: return x.copy()
    window = _odd_window(window_ms, frame_ms, len(x))
    if method == "moving_median": return scipy.ndimage.median_filter(x, size=window, mode="nearest")
    if method == "moving_mean": return scipy.ndimage.uniform_filter1d(x, size=window, mode="nearest")
    if method == "savgol":
        order = min(int(savgol_order), window - 1)
        return scipy.signal.savgol_filter(x, window, order, mode="interp") if window > order else x.copy()
    if method in {"butterworth_lowpass", "median_then_lowpass"}:
        initial = scipy.ndimage.median_filter(x, size=max(1, min(window, 9)), mode="nearest") if method.startswith("median") else x
        nyquist = .5 / (frame_ms / 1000); normalized = min(.999, max(.0001, cutoff_hz / nyquist))
        sos = scipy.signal.butter(3, normalized, btype="low", output="sos")
        pad = 3 * (2 * len(sos) + 1)
        return scipy.signal.sosfiltfilt(sos, initial, padlen=min(len(x)-1, pad)) if len(x) > 3 else initial.copy()
    if method == "robust_spline":
        grid = np.arange(len(x), dtype=float); weights = np.ones(len(x)); fitted = x.copy()
        smooth = max(0.0, float(spline_smoothing)) * len(x)
        for _ in range(3):
            spline = scipy.interpolate.UnivariateSpline(grid, x, w=weights, s=smooth, k=min(3,len(x)-1))
            fitted = spline(grid); residual = x-fitted; scale = 1.4826*np.median(np.abs(residual-np.median(residual))) + 1e-9
            weights = 1 / np.maximum(1, np.abs(residual)/(2.5*scale))
        return fitted
    raise LabError(f"unknown trend method: {method}")


def boundary_mix(length: int, parameters: dict[str, Any], frame_ms: float) -> Any:
    import numpy as np
    weights = np.ones(length, dtype=float)
    attack = min(length, int(round(float(parameters.get("preserve_attack_ms", 0)) / frame_ms)))
    release = min(length, int(round(float(parameters.get("preserve_release_ms", 0)) / frame_ms)))
    transition = int(round(float(parameters.get("transition_ms", 0)) / frame_ms)); curve = parameters.get("transition_curve", "step")
    weights[:attack] = 0; weights[max(0, length-release):] = 0
    def ramp(n: int) -> Any:
        r = np.linspace(0, 1, n+2)[1:-1]
        return .5-.5*np.cos(np.pi*r) if curve == "cosine" else r if curve == "linear" else np.ones(n)
    if transition > 0:
        n = min(transition, max(0, length-attack)); weights[attack:attack+n] = np.minimum(weights[attack:attack+n], ramp(n))
        n = min(transition, max(0, length-release)); begin = length-release-n
        weights[begin:begin+n] = np.minimum(weights[begin:begin+n], ramp(n)[::-1])
    return weights


def transform_f0(f0: Any, parameters: dict[str, Any], frame_ms: float) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    original = np.asarray(f0, dtype=np.float64); output = original.copy(); voiced = np.isfinite(original) & (original > 0)
    cents = hz_to_cents(original); minimum_frames = max(1, int(round(float(parameters.get("minimum_region_ms", 15))/frame_ms)))
    all_regions = contiguous_regions(voiced, cents, split_jump_cents=float(parameters.get("split_jump_cents", 300)), minimum_frames=1)
    regions = [region for region in all_regions if region[1]-region[0]>=minimum_frames or parameters.get("merge_short_regions",False)]
    factor = float(parameters.get("modulation_factor", 0.0))
    if not -.25 <= factor <= 1.0: raise LabError("modulation_factor outside safe range")
    residual_rms = []
    gap_frames=max(0,int(round(float(parameters.get("max_unvoiced_gap_ms",0))/frame_ms)))
    for region_index,(start, stop) in enumerate(regions):
        segment = cents[start:stop]
        trend = estimate_trend(segment, parameters.get("trend_method", "moving_median"), frame_ms=frame_ms,
                               window_ms=float(parameters.get("smoothing_window_ms", 180)),
                               cutoff_hz=float(parameters.get("lowpass_cutoff_hz", 2.0)),
                               savgol_order=int(parameters.get("savgol_order", 3)),
                               spline_smoothing=float(parameters.get("spline_smoothing", 1.0)))
        transformed = trend + factor * (segment-trend)
        boundary_parameters=dict(parameters)
        # A short tracking gap never enters either filter, but may be treated
        # as an internal tracking break rather than a fresh attack/release.
        if region_index and start-regions[region_index-1][1] <= gap_frames: boundary_parameters["preserve_attack_ms"]=0
        if region_index+1<len(regions) and regions[region_index+1][0]-stop <= gap_frames: boundary_parameters["preserve_release_ms"]=0
        mix = boundary_mix(stop-start, boundary_parameters, frame_ms)
        result = segment + mix*(transformed-segment); hz = cents_to_hz(result)
        valid = np.isfinite(hz) & (hz >= 40) & (hz <= 1200)
        if not np.all(valid): raise LabError("transformation produced invalid F0")
        output[start:stop] = hz; residual_rms.append(float(np.sqrt(np.mean((segment-trend)**2))))
    output[~voiced] = 0.0
    return output, {"regions": len(regions), "voiced_frames": int(voiced.sum()),
                    "residual_rms_cents": statistics.mean(residual_rms) if residual_rms else 0.0}


class ModulationRemovalEngine(ABC):
    id = "base"
    @abstractmethod
    def preflight(self) -> dict[str, Any]: ...
    @abstractmethod
    def render(self, source_path: Path, output_path: Path, parameters: dict[str, Any]) -> dict[str, Any]: ...
    @abstractmethod
    def parameter_space(self) -> dict[str, Any]: ...
    def describe(self) -> dict[str, Any]:
        return {"id": self.id, "class": type(self).__name__, "version": ENGINE_VERSION.get(self.id, TRANSFORM_VERSION),
                "target_input": False, "changes": ["F0"], "preserves": ["duration", "spectral envelope", "aperiodicity"]}


class WorldModulationEngine(ModulationRemovalEngine):
    id = "world"
    def __init__(self, env: dict[str, str]): self.env = env
    def preflight(self) -> dict[str, Any]:
        try:
            import pyworld
            return {"available": True, "version": getattr(pyworld, "__version__", "unknown"), "extractors": ["dio_stonemask", "harvest"]}
        except ImportError as exc: return {"available": False, "reason": str(exc)}
    def parameter_space(self) -> dict[str, Any]: return read_yaml(Path(self.env["AEGIS_GLADOS_MOD_CONFIG"]))["optimization"]["search_space"]
    def render(self, source_path: Path, output_path: Path, parameters: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        import pyworld
        started = time.monotonic(); audio, sr = load_audio(Path(source_path), int(self.env["AEGIS_GLADOS_MOD_SAMPLE_RATE"])); frame = float(self.env["AEGIS_GLADOS_MOD_FRAME_PERIOD_MS"])
        extractor = parameters.get("f0_extractor", "dio_stonemask")
        if extractor == "dio_stonemask":
            f0, times = pyworld.dio(audio, sr, f0_floor=60, f0_ceil=600, frame_period=frame); f0 = pyworld.stonemask(audio, f0, times, sr)
        elif extractor == "harvest": f0, times = pyworld.harvest(audio, sr, f0_floor=60, f0_ceil=600, frame_period=frame)
        else: raise LabError(f"unknown WORLD extractor: {extractor}")
        spectrum = pyworld.cheaptrick(audio, f0, times, sr); aperiodicity = pyworld.d4c(audio, f0, times, sr)
        if parameters.get("passthrough_f0", False): changed, details = f0.copy(), {"regions": 0, "voiced_frames": int(np.sum(f0 > 0))}
        else: changed, details = transform_f0(f0, parameters, frame)
        rendered = pyworld.synthesize(np.ascontiguousarray(changed), spectrum, aperiodicity, sr, frame_period=frame)
        rendered = np.pad(rendered[:len(audio)], (0, max(0, len(audio)-len(rendered))))
        write_audio(Path(output_path), rendered, sr)
        return {"success": True, "output": str(output_path), "engine": self.id, "extractor": extractor,
                "duration_seconds": time.monotonic()-started, **details}


class PraatModulationEngine(ModulationRemovalEngine):
    id = "praat"
    def __init__(self, env: dict[str, str]): self.env = env
    def preflight(self) -> dict[str, Any]:
        try:
            import parselmouth
            return {"available": True, "version": parselmouth.__version__, "synthesis": "PSOLA overlap-add"}
        except ImportError as exc: return {"available": False, "reason": str(exc)}
    def parameter_space(self) -> dict[str, Any]: return read_yaml(Path(self.env["AEGIS_GLADOS_MOD_CONFIG"]))["optimization"]["search_space"]
    def render(self, source_path: Path, output_path: Path, parameters: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        import parselmouth
        from parselmouth.praat import call
        started = time.monotonic(); audio, sr = load_audio(Path(source_path), int(self.env["AEGIS_GLADOS_MOD_SAMPLE_RATE"])); frame = float(self.env["AEGIS_GLADOS_MOD_FRAME_PERIOD_MS"])
        sound = parselmouth.Sound(audio, sampling_frequency=sr); manipulation = call(sound, "To Manipulation", frame/1000, 60, 600)
        tier = call(manipulation, "Extract pitch tier")
        tracking=fixed_pitch_track(audio,sr,frame);times=tracking["time"];f0=tracking["f0"]
        if not parameters.get("passthrough_f0", False): changed, details = transform_f0(f0, parameters, frame)
        else: changed, details = f0, {"regions": 0, "voiced_frames": len(f0)}
        call(tier, "Remove points between", 0, sound.duration)
        for when, value in zip(times, changed):
            if value > 0: call(tier, "Add point", float(when), float(value))
        call([tier, manipulation], "Replace pitch tier"); result = call(manipulation, "Get resynthesis (overlap-add)")
        rendered = np.asarray(result.values[0], dtype=np.float64); rendered = np.pad(rendered[:len(audio)], (0, max(0, len(audio)-len(rendered))))
        write_audio(Path(output_path), rendered, sr)
        return {"success": True, "output": str(output_path), "engine": self.id,
                "duration_seconds": time.monotonic()-started, **details}


def engine_registry(env: dict[str, str]) -> dict[str, ModulationRemovalEngine]:
    return {"world": WorldModulationEngine(env), "praat": PraatModulationEngine(env)}


def fixed_pitch_track(audio: Any, sr: int, frame_ms: float = 5.0) -> dict[str, Any]:
    import numpy as np
    import parselmouth
    sound=parselmouth.Sound(np.asarray(audio,dtype=np.float64),sampling_frequency=sr)
    pitch=sound.to_pitch_ac(time_step=frame_ms/1000,pitch_floor=60,pitch_ceiling=600)
    f0=np.asarray(pitch.selected_array["frequency"],dtype=np.float64);times=np.asarray(pitch.xs(),dtype=np.float64)
    voiced = f0 > 0; return {"f0": f0, "time": times, "voiced": voiced, "cents": hz_to_cents(f0), "extractor": "Praat autocorrelation fixed evaluator"}


def frame_energy(audio: Any, sr: int, frame_ms: float, count: int) -> Any:
    import numpy as np
    hop = max(1, int(round(sr*frame_ms/1000))); result = np.zeros(count)
    for i in range(count):
        piece = np.asarray(audio[i*hop:min(len(audio), (i+1)*hop)])
        result[i] = np.sqrt(np.mean(piece*piece)) if len(piece) else 0
    return result


def fixed_features(audio: Any, sr: int, frame_ms: float = 5.0) -> dict[str, Any]:
    import numpy as np
    import scipy.signal
    track = fixed_pitch_track(audio, sr, frame_ms); cents, voiced = track["cents"], track["voiced"]
    residuals, trends = np.full_like(cents, np.nan), np.full_like(cents, np.nan)
    for start, stop in contiguous_regions(voiced, cents, split_jump_cents=300, minimum_frames=3):
        trend = estimate_trend(cents[start:stop], "savgol", frame_ms=frame_ms, window_ms=350, savgol_order=3)
        trends[start:stop] = trend; residuals[start:stop] = cents[start:stop]-trend
    rv = residuals[np.isfinite(residuals)]; tv = trends[np.isfinite(trends)]
    if len(rv) > 2:
        frequencies, power = scipy.signal.periodogram(rv, fs=1000/frame_ms); bands = [float(np.sum(power[(frequencies>=a)&(frequencies<b)])) for a,b in ((0,.5),(.5,2),(2,8),(8,40),(40,100))]
        derivative = np.diff(rv); autocorrelation = float(np.corrcoef(rv[:-1], rv[1:])[0,1]) if np.std(rv[:-1]) and np.std(rv[1:]) else 0.0
        crossings = float(np.mean(np.signbit(rv[:-1]) != np.signbit(rv[1:])))
        dominant = float(frequencies[1+np.argmax(power[1:])]) if len(power)>1 else 0.0
    else: bands=[0.0]*5; derivative=np.zeros(1); autocorrelation=crossings=dominant=0.0
    # Gain-normalized log spectrum and envelope are intentionally fixed for every trial.
    frequencies, spectrum = scipy.signal.welch(audio, sr, nperseg=min(2048, len(audio)))
    spectrum = 10*np.log10(np.maximum(spectrum, 1e-12)); spectrum -= np.mean(spectrum)
    energy = frame_energy(audio, sr, frame_ms, len(track["f0"])); envelope = scipy.ndimage.gaussian_filter1d(np.abs(audio), max(1, sr//200))
    return {"track": track, "trend": trends, "residual": residuals,
            "modulation": np.asarray([np.sqrt(np.mean(rv*rv)) if len(rv) else 0, np.std(rv) if len(rv) else 0,
                np.percentile(np.abs(rv),90) if len(rv) else 0, autocorrelation, crossings, dominant,
                np.sqrt(np.mean(derivative*derivative)), *bands], dtype=float),
            "drift": np.asarray([np.mean(tv) if len(tv) else 0, np.std(tv) if len(tv) else 0,
                np.percentile(tv,10) if len(tv) else 0, np.percentile(tv,90) if len(tv) else 0], dtype=float),
            "f0": np.asarray([np.nanmedian(cents) if voiced.any() else 0, np.nanpercentile(cents[voiced],90) if voiced.any() else 0,
                np.nanstd(np.diff(cents[voiced])) if voiced.sum()>1 else 0], dtype=float),
            "spectral": spectrum, "voiced": voiced.astype(float), "energy": energy,
            "envelope": envelope[::max(1, sr//200)], "duration": len(audio)/sr}


def acoustic_analysis_summary(audio:Any,sr:int,features:dict[str,Any],frame_ms:float)->dict[str,Any]:
    import numpy as np
    import parselmouth
    import scipy.signal
    track=features["track"];voiced=track["voiced"];cents=track["cents"]
    derivative=[]
    durations=[]
    for start,stop in contiguous_regions(voiced,cents,minimum_frames=1):
        values=cents[start:stop]
        if len(values)>1:derivative.extend(np.diff(values)/(frame_ms/1000))
        durations.append((stop-start)*frame_ms)
    sound=parselmouth.Sound(np.asarray(audio,dtype=float),sampling_frequency=sr)
    harmonicity=sound.to_harmonicity_ac(time_step=frame_ms/1000,minimum_pitch=60)
    harmonic_values=np.asarray(harmonicity.values[0],dtype=float);harmonic_values=harmonic_values[np.isfinite(harmonic_values)&(harmonic_values>-100)]
    _,power=scipy.signal.welch(audio,sr,nperseg=min(2048,len(audio)));flatness=float(np.exp(np.mean(np.log(np.maximum(power,1e-16))))/np.mean(np.maximum(power,1e-16)))
    energy=np.asarray(features["energy"]);envelope=np.asarray(features["envelope"])
    return {"f0_derivative_rms_cents_per_second":float(np.sqrt(np.mean(np.square(derivative)))) if derivative else 0.0,
            "pitch_velocity_median_abs_cents_per_second":float(np.median(np.abs(derivative))) if derivative else 0.0,
            "frame_energy_rms":float(np.sqrt(np.mean(energy*energy))),"frame_energy_std":float(np.std(energy)),
            "harmonicity_mean_db":float(np.mean(harmonic_values)) if len(harmonic_values) else None,
            "harmonicity_median_db":float(np.median(harmonic_values)) if len(harmonic_values) else None,
            "spectral_flatness":flatness,"amplitude_envelope_mean":float(np.mean(envelope)),
            "amplitude_envelope_std":float(np.std(envelope)),"voiced_region_count":len(durations),
            "voiced_region_median_ms":float(np.median(durations)) if durations else 0.0,
            "voiced_region_total_ms":float(np.sum(durations))}


def vector_distance(left: Any, right: Any) -> float:
    import numpy as np
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float); n = min(a.size, b.size)
    if n == 0: return float("inf")
    a, b = a.flat[:n], b.flat[:n]; finite = np.isfinite(a)&np.isfinite(b)
    return float(np.sqrt(np.mean((a[finite]-b[finite])**2))) if finite.any() else float("inf")


def normalized_distance(output: Any, target: Any, source: Any, *, minimum: float = .01, epsilon: float = 1e-8) -> dict[str, Any]:
    baseline = vector_distance(source, target); distance = vector_distance(output, target)
    if not math.isfinite(baseline) or baseline < minimum:
        return {"status": "non_informative", "baseline_distance": baseline, "output_distance": distance, "score": None}
    return {"status": "informative", "baseline_distance": baseline, "output_distance": distance, "score": distance/max(baseline, epsilon)}


def output_constraints(path: Path, expected_samples: int, expected_sr: int, config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    violations = []
    try: info = sf.info(path); audio, sr = load_audio(path)
    except Exception as exc: return {"valid": False, "violations": [f"invalid_wav:{exc}"]}
    if not len(audio): violations.append("empty")
    if not np.all(np.isfinite(audio)): violations.append("nan_or_inf")
    if sr != expected_sr: violations.append("sample_rate")
    if info.channels != 1: violations.append("channels")
    error_ms = abs(len(audio)-expected_samples)*1000/expected_sr
    if error_ms > float(config["scoring"]["maximum_duration_error_ms"]): violations.append("duration")
    clipping = float(np.mean(np.abs(audio) >= .999))
    if clipping > .05: violations.append("extreme_clipping")
    return {"valid": not violations, "violations": violations, "duration_error_ms": error_ms,
            "clipping_fraction": clipping, "peak": float(np.max(np.abs(audio))), "sample_rate": sr, "channels": info.channels}


def compare_features(output: dict[str, Any], target: dict[str, Any], source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    keys = {"modulation_residual": "modulation", "drift_preservation": "drift", "f0_agreement": "f0",
            "spectral_preservation": "spectral", "voiced_unvoiced": "voiced", "timing": "energy"}
    values, raw = {}, {}
    minimum = float(config["scoring"]["minimum_informative_distance"]); epsilon = float(config["scoring"]["denominator_epsilon"])
    for name, feature in keys.items():
        item = normalized_distance(output[feature], target[feature], source[feature], minimum=minimum, epsilon=epsilon)
        raw[name] = item; values[name] = item["score"] if item["score"] is not None else 0.0
    n = min(len(output["track"]["f0"]), len(target["track"]["f0"])); both = output["track"]["voiced"][:n]&target["track"]["voiced"][:n]
    errors = np.abs(output["track"]["cents"][:n][both]-target["track"]["cents"][:n][both]) if both.any() else np.asarray([1200.])
    detailed = {"median_f0_error_cents": float(np.median(errors)), "p90_f0_error_cents": float(np.percentile(errors,90)),
                "voiced_unvoiced_agreement": float(np.mean(output["track"]["voiced"][:n] == target["track"]["voiced"][:n])) if n else 0,
                "fast_residual_rms_cents": float(output["modulation"][0]), "target_fast_residual_rms_cents": float(target["modulation"][0]),
                "drift_std_cents": float(output["drift"][1]), "target_drift_std_cents": float(target["drift"][1]),
                "extractor_disagreement_logged": False}
    weights = config["scoring"]; composite = sum(float(weights[name])*values[name] for name in keys)
    return {"score": float(composite), "normalized": values, "distances": raw, **detailed}


def artifact_penalty(output: dict[str, Any], source: dict[str, Any], target: dict[str, Any], tolerance: float = .001) -> float:
    allowed = max(float(source.get("clipping_fraction",0)), float(target.get("clipping_fraction",0))) + tolerance
    return max(0.0, float(output.get("clipping_fraction",0))-allowed)*100


def canonical_parameters(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def cache_key(source: Path, engine: str, parameters: dict[str, Any], sample_rate: int) -> str:
    payload = "|".join((sha256(source), engine, ENGINE_VERSION[engine], canonical_parameters(parameters), str(sample_rate), TRANSFORM_VERSION))
    return hashlib.sha256(payload.encode("utf8")).hexdigest()


def _jsonable(value: Any) -> Any:
    import numpy as np
    if isinstance(value,np.ndarray): return _jsonable(value.tolist())
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {key:_jsonable(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)): return [_jsonable(item) for item in value]
    return value


def _arrays(value: Any) -> Any:
    import numpy as np
    if isinstance(value,dict): return {key:_arrays(item) for key,item in value.items()}
    if isinstance(value,list): return np.asarray([np.nan if item is None else item for item in value])
    return value


def feature_cached(path: Path, paths: dict[str,Path], sr: int, frame: float) -> dict[str,Any]:
    key=hashlib.sha256(f"praat-fixed-v1|{sha256(path)}|{sr}|{frame}".encode()).hexdigest();directory=paths["cache"]/"features";directory.mkdir(parents=True,exist_ok=True);cache=directory/f"{key}.json"
    if not cache.is_file():
        audio,_=load_audio(path,sr);payload=_jsonable(fixed_features(audio,sr,frame))
        write_json(cache,payload)
    return _arrays(json.loads(cache.read_text(encoding="utf8")))


def render_cached(engine: ModulationRemovalEngine, source: Path, output: Path, parameters: dict[str, Any], paths: dict[str, Path], sr: int) -> tuple[dict[str, Any], bool]:
    key = cache_key(source, engine.id, parameters, sr); cached = paths["cache"] / f"{key}.wav"; metadata = paths["cache"] / f"{key}.json"
    if cached.is_file() and metadata.is_file():
        output.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(cached, output); return json.loads(metadata.read_text()), True
    if engine.id=="praat":
        result=engine.render(source,output,parameters)
        if result.get("success"):
            shutil.copy2(output,cached);write_json(metadata,result)
        return result,False
    # WORLD's native analysis buffers are not always returned to the OS.  One
    # short, target-free worker per WORLD render bounds memory across searches.
    command=[sys.executable,str(Path(__file__).resolve()),"_render-worker","--engine",engine.id,
             "--source",str(source),"--output",str(output),"--parameters-json",canonical_parameters(parameters)]
    process=subprocess.run(command,text=True,capture_output=True,check=False)
    if process.returncode:
        result={"success":False,"error":process.stderr.strip() or process.stdout.strip(),"engine":engine.id}
    else:
        try: result=json.loads(process.stdout)
        except json.JSONDecodeError as exc: result={"success":False,"error":f"invalid worker response: {exc}","engine":engine.id}
    if result.get("success"):
        shutil.copy2(output, cached); write_json(metadata, result)
    return result, False


def analysis(env: dict[str, str]) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.signal
    paths = output_paths(env); manifest = preflight(env); frame = float(env["AEGIS_GLADOS_MOD_FRAME_PERIOD_MS"]); sr = int(env["AEGIS_GLADOS_MOD_SAMPLE_RATE"])
    report_analysis=paths["report"]/"analysis";report_analysis.mkdir(parents=True,exist_ok=True)
    contexts, band_rows, residual_spectra, vuv_rows = {}, [], [], []
    windows = [40,80,120,180,250,350,500]
    for item in manifest["pairs"]:
        cid = item["id"]; directory = paths["work"] / cid; source, _ = load_audio(directory/"source-aligned.wav", sr); target, _ = load_audio(directory/"target-aligned.wav", sr)
        sf, tf = fixed_features(source,sr,frame), fixed_features(target,sr,frame); st, tt = sf["track"], tf["track"]; n=min(len(st["f0"]),len(tt["f0"])); both=st["voiced"][:n]&tt["voiced"][:n]
        source_acoustics=acoustic_analysis_summary(source,sr,sf,frame);target_acoustics=acoustic_analysis_summary(target,sr,tf,frame)
        multiscale={}
        for window in windows:
            values=[]
            for track in (st,tt):
                residual=[]
                for a,b in contiguous_regions(track["voiced"][:n],track["cents"][:n],minimum_frames=3):
                    x=track["cents"][a:b]; residual.extend(x-estimate_trend(x,"savgol",frame_ms=frame,window_ms=window,savgol_order=3))
                values.append(float(np.sqrt(np.mean(np.square(residual)))) if residual else 0)
            multiscale[str(window)]={"source_residual_rms_cents":values[0],"target_residual_rms_cents":values[1],"removed_fraction":1-values[1]/max(values[0],1e-8)}
        sr_res=sf["residual"][np.isfinite(sf["residual"])]; tr_res=tf["residual"][np.isfinite(tf["residual"])]
        for label,value in (("source",sr_res),("target",tr_res)):
            freq,power=scipy.signal.periodogram(value,fs=1000/frame) if len(value)>2 else (np.zeros(1),np.zeros(1)); residual_spectra.append((cid,label,freq,power))
        source_fast=float(sf["modulation"][0]); target_fast=float(tf["modulation"][0]); slow_distance=vector_distance(sf["drift"],tf["drift"])
        spectrum_distance=vector_distance(sf["spectral"],tf["spectral"]); vuv=float(np.mean(st["voiced"][:n]==tt["voiced"][:n]))
        region_stats=[]
        for a,b in contiguous_regions(st["voiced"][:n],st["cents"][:n],minimum_frames=3):
            width=max(1,int(80/frame)); original=st["cents"][a:b]; reference=np.interp(np.arange(a,b),np.flatnonzero(tt["voiced"][:n]),tt["cents"][:n][tt["voiced"][:n]]) if tt["voiced"][:n].any() else original
            region_stats.append({"attack_error_cents":vector_distance(original[:width],reference[:width]),"release_error_cents":vector_distance(original[-width:],reference[-width:]),"duration_ms":(b-a)*frame})
        contexts[cid]={"frames":n,"voiced_unvoiced_agreement":vuv,"source_fast_residual_rms_cents":source_fast,
            "target_fast_residual_rms_cents":target_fast,"fast_modulation_removed_fraction":1-target_fast/max(source_fast,1e-8),
            "slow_drift_distance":slow_distance,"log_spectral_distance":spectrum_distance,"multiscale":multiscale,
            "voiced_region_count":len(region_stats),"attack_mean_error_cents":statistics.mean([x["attack_error_cents"] for x in region_stats]) if region_stats else None,
            "release_mean_error_cents":statistics.mean([x["release_error_cents"] for x in region_stats]) if region_stats else None,
            "target_has_spectral_changes_beyond_f0":spectrum_distance>.5,"source_acoustics":source_acoustics,
            "target_acoustics":target_acoustics,"energy_distance":vector_distance(sf["energy"],tf["energy"]),
            "amplitude_envelope_distance":vector_distance(sf["envelope"],tf["envelope"])}
        band_rows.append((cid,sf["modulation"][-5:],tf["modulation"][-5:])); vuv_rows.append((cid,vuv))
        fig,ax=plt.subplots(figsize=(12,4));ax.plot(st["time"][:n],st["cents"][:n],label="source",alpha=.7);ax.plot(tt["time"][:n],tt["cents"][:n],label="target Melodyne",alpha=.7);ax.set(title=cid,ylabel="pitch (cents, ref 1 Hz)",xlabel="seconds");ax.legend();fig.tight_layout();fig.savefig(paths["report"]/"analysis"/f"f0-overlay-{cid}.png",dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for cid,label,freq,power in residual_spectra: ax.semilogy(freq,power+1e-12,label=f"{cid} {label}",alpha=.7)
    ax.set(xlim=(0,30),xlabel="temporal frequency (Hz)",ylabel="residual power",title="F0 residual spectrum");ax.legend(fontsize=6);fig.tight_layout();fig.savefig(report_analysis/"f0-residual-spectrum.png",dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(5); width=.09
    for index,(cid,s,t) in enumerate(band_rows): ax.bar(x+(index*2)*width,s,width,label=f"{cid} source");ax.bar(x+(index*2+1)*width,t,width,label=f"{cid} target")
    ax.set_xticks(x+.35);ax.set_xticklabels(["0-.5",".5-2","2-8","8-40","40-100"]);ax.set(title="Temporal modulation-band energy",xlabel="Hz");ax.legend(fontsize=6);fig.tight_layout();fig.savefig(report_analysis/"modulation-band-comparison.png",dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));ax.bar([x[0] for x in vuv_rows],[x[1] for x in vuv_rows]);ax.set_ylim(0,1);ax.tick_params(axis='x',rotation=20);ax.set(title="Voiced/unvoiced agreement",ylabel="agreement");fig.tight_layout();fig.savefig(report_analysis/"voiced-unvoiced-comparison.png",dpi=130);plt.close(fig)
    payload={"created_at":utcnow(),"fixed_evaluator":"Praat autocorrelation + fixed 350 ms Savitzky-Golay trend","windows_ms":windows,"contexts":contexts,
             "summary":{"largest_temporal_differences":"See multiscale removal fractions and residual spectrum plot.","slow_trend_preservation":"Reported as slow_drift_distance per context.","boundary_treatment":"Attack/release error is reported separately.","unvoiced_similarity":"Reported as voiced_unvoiced_agreement.","spectral_changes":"Reported as gain-normalized log_spectral_distance; this is diagnostic, not proof of causality."}}
    write_json(report_analysis/"modulation-analysis.json",payload);write_json(paths["analysis"]/"modulation-analysis.json",payload);return payload


BASELINES = {
    "B00": {"engine":"passthrough","name":"passthrough","parameters":{}},
    "B01": {"engine":"world","name":"WORLD unchanged F0","parameters":{"passthrough_f0":True,"f0_extractor":"dio_stonemask"}},
    "B02": {"engine":"world","name":"strong robust moving median","parameters":{"trend_method":"moving_median","smoothing_window_ms":350,"modulation_factor":0.0}},
    "B03": {"engine":"world","name":"Savitzky-Golay","parameters":{"trend_method":"savgol","smoothing_window_ms":250,"savgol_order":3,"modulation_factor":0.0}},
    "B04": {"engine":"world","name":"Butterworth low-pass","parameters":{"trend_method":"butterworth_lowpass","lowpass_cutoff_hz":2.0,"smoothing_window_ms":250,"modulation_factor":0.0}},
    "B05": {"engine":"world","name":"robust spline","parameters":{"trend_method":"robust_spline","spline_smoothing":1.0,"smoothing_window_ms":250,"modulation_factor":0.0}},
}


def prepared_contexts(env: dict[str,str]) -> list[dict[str,Any]]:
    paths=output_paths(env);dataset=load_dataset(env);result=[]
    for context in dataset["contexts"]:
        directory=paths["work"]/context["id"]
        if not (directory/"source-aligned.wav").is_file(): preflight(env); break
    for context in dataset["contexts"]:
        directory=paths["work"]/context["id"]
        result.append({**context,"source_path":directory/"source-aligned.wav","target_path":directory/"target-aligned.wav"})
    return result


def evaluate_candidate(env: dict[str,str], config: dict[str,Any], candidate_id: str, engine_name: str,
                       parameters: dict[str,Any], *, destination_root: Path | None=None) -> dict[str,Any]:
    import numpy as np
    paths=output_paths(env);engine=engine_registry(env).get(engine_name);root=destination_root or paths["raw"]/candidate_id;root.mkdir(parents=True,exist_ok=True)
    context_records={};scores=[];penalties=[];cache_hits=0;started=time.monotonic();sr=int(env["AEGIS_GLADOS_MOD_SAMPLE_RATE"])
    for context in prepared_contexts(env):
        cid=context["id"];output=root/f"{cid}.wav";source_audio,_=load_audio(context["source_path"],sr);target_audio,_=load_audio(context["target_path"],sr)
        try:
            if engine_name=="passthrough": write_audio(output,source_audio,sr);render={"success":True,"engine":"passthrough","output":str(output)};hit=False
            else:
                if engine is None: raise LabError(f"unknown engine: {engine_name}")
                render,hit=render_cached(engine,context["source_path"],output,parameters,paths,sr)
            cache_hits+=int(hit);constraints=output_constraints(output,len(source_audio),sr,config)
            if not constraints["valid"]: raise LabError("invalid output: "+", ".join(constraints["violations"]))
            output_audio,_=load_audio(output,sr);n=min(len(source_audio),len(target_audio),len(output_audio));source_audio,target_audio,output_audio=source_audio[:n],target_audio[:n],output_audio[:n]
            frame=float(env["AEGIS_GLADOS_MOD_FRAME_PERIOD_MS"]);sf=feature_cached(context["source_path"],paths,sr,frame);tf=feature_cached(context["target_path"],paths,sr,frame);of=feature_cached(output,paths,sr,frame)
            metrics=compare_features(of,tf,sf,config);source_diag=output_constraints(context["source_path"],len(source_audio),sr,config);target_diag=output_constraints(context["target_path"],len(target_audio),sr,config)
            penalty=artifact_penalty(constraints,source_diag,target_diag);score=metrics["score"];scores.append(score);penalties.append(penalty)
            context_records[cid]={"score":score,"metrics":metrics,"output":str(output),"constraints":constraints,"render":render,"cache_hit":hit}
        except Exception as exc:
            context_records[cid]={"score":1e6,"metrics":{},"output":str(output),"constraints":{"valid":False,"violations":[str(exc)]},"render":{"success":False,"error":str(exc)},"cache_hit":False};scores.append(1e6)
    worst=max(scores);median=statistics.median(scores);penalty=sum(penalties);effect=.5*worst+.5*median+penalty
    return {"candidate":candidate_id,"engine":engine_name,"parameters":parameters,"contexts":context_records,
            "worst_context":worst,"median_context":median,"mean_context":statistics.mean(scores),"std_context":statistics.pstdev(scores),
            "context_variance":statistics.pvariance(scores),"effect_score":effect,"artifacts_penalty":penalty,
            "status":"complete" if all(x["constraints"]["valid"] for x in context_records.values()) else "invalid",
            "cache_hits":cache_hits,"duration_seconds":time.monotonic()-started}


def baseline(env: dict[str,str]) -> dict[str,Any]:
    config=read_yaml(Path(env["AEGIS_GLADOS_MOD_CONFIG"]));paths=output_paths(env);preflight(env);records=[]
    for bid,spec in BASELINES.items():
        record=evaluate_candidate(env,config,bid,spec["engine"],spec["parameters"],destination_root=paths["baselines"]/bid);record["name"]=spec["name"];records.append(record)
    write_json(paths["baselines"]/"baselines.json",records);return {"baselines":records}


def suggest_parameters(trial: Any, config: dict[str,Any]) -> tuple[str,dict[str,Any]]:
    # Praat is secondary but remains represented; all shared parameters use one candidate configuration.
    engine=trial.suggest_categorical("engine",["world","praat"])
    method=trial.suggest_categorical("trend_method",config["optimization"]["search_space"]["trend_method"]["choices"])
    p={"trend_method":method,"modulation_factor":trial.suggest_float("modulation_factor",-.25,.5),
       "smoothing_window_ms":trial.suggest_float("smoothing_window_ms",20,600,log=True),
       "split_jump_cents":trial.suggest_float("split_jump_cents",30,300),
       "minimum_region_ms":trial.suggest_float("minimum_region_ms",15,250),
       "max_unvoiced_gap_ms":trial.suggest_float("max_unvoiced_gap_ms",0,120),"split_on_unvoiced":True,
       "merge_short_regions":trial.suggest_categorical("merge_short_regions",[False,True]),
       "preserve_attack_ms":trial.suggest_float("preserve_attack_ms",0,150),
       "preserve_release_ms":trial.suggest_float("preserve_release_ms",0,150),
       "transition_ms":trial.suggest_float("transition_ms",0,120),
       "transition_curve":trial.suggest_categorical("transition_curve",["step","linear","cosine"])}
    if method in {"butterworth_lowpass","median_then_lowpass"}: p["lowpass_cutoff_hz"]=trial.suggest_float("lowpass_cutoff_hz",.2,8,log=True)
    if method=="savgol": p["savgol_order"]=trial.suggest_categorical("savgol_order",[2,3,4])
    if method=="robust_spline": p["spline_smoothing"]=trial.suggest_float("spline_smoothing",.001,100,log=True)
    if engine=="world": p["f0_extractor"]=trial.suggest_categorical("f0_extractor",["dio_stonemask","harvest"])
    return engine,p


def read_trials(path: Path) -> list[dict[str,Any]]:
    if not path.is_file(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def export_trials(paths:dict[str,Path],records:list[dict[str,Any]]) -> None:
    fields=["trial","checkpoint","engine","effect_score","worst_context","median_context","mean_context","std_context","status","cache_hits","duration_seconds","parameters"]
    target=paths["out"]/"trials.csv";target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",newline="",encoding="utf8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in records: writer.writerow({key:json.dumps(row.get(key),sort_keys=True) if key=="parameters" else row.get(key) for key in fields})


def optimize(env:dict[str,str],trials:int,max_total_trials:int,resume:bool=False,study_suffix:str|None=None) -> dict[str,Any]:
    import optuna
    if trials<=0 or max_total_trials<=0: raise LabError("--trials and --max-total-trials must be positive")
    config=read_yaml(Path(env["AEGIS_GLADOS_MOD_CONFIG"]));paths=output_paths(env);preflight(env)
    suffix=f"-{study_suffix}" if study_suffix else "";name=f"melodyne-modulation-replication{suffix}";db=paths["out"]/"studies.sqlite3"
    storage=f"sqlite:///{db}";existing_names=[]
    if db.is_file(): existing_names=[x.study_name for x in optuna.study.get_all_study_summaries(storage)]
    exists=name in existing_names
    if exists and not resume: raise LabError(f"study {name} already exists; --resume is required")
    if resume and not exists: raise LabError(f"cannot --resume: study {name} does not exist")
    records=read_trials(paths["out"]/"trials.jsonl");existing=sum(1 for x in records if x.get("study")==name and x.get("status") in {"complete","invalid"})
    additional=min(trials,max_total_trials-existing)
    if additional<=0: raise LabError(f"maximum total trials reached ({max_total_trials})")
    sampler=optuna.samplers.TPESampler(seed=int(env["AEGIS_GLADOS_MOD_SEED"])+existing,n_startup_trials=int(config["optimization"]["n_startup_trials"]),
        multivariate=bool(config["optimization"]["multivariate"]),group=bool(config["optimization"]["group"]),
        constraints_func=lambda trial:trial.user_attrs.get("constraints",[0.0]))
    study=optuna.create_study(study_name=name,storage=storage,direction="minimize",sampler=sampler,load_if_exists=resume)
    if not exists:
        for factor in (-.25,0.0,.10,.25,.50):
            study.enqueue_trial({"engine":"world","trend_method":"moving_median","modulation_factor":factor,
                "smoothing_window_ms":180.0,"split_jump_cents":300.0,"minimum_region_ms":15.0,
                "max_unvoiced_gap_ms":0.0,"merge_short_regions":False,"preserve_attack_ms":0.0,
                "preserve_release_ms":0.0,"transition_ms":0.0,"transition_curve":"step","f0_extractor":"dio_stonemask"})
    history_path=paths["out"]/"checkpoint-history.json";history=json.loads(history_path.read_text()) if history_path.is_file() else [];checkpoint=1+sum(1 for x in history if x.get("study")==name)
    build_dashboard(env)
    def objective(trial:Any)->float:
        engine,parameters=suggest_parameters(trial,config);candidate=f"candidate-{trial.number:06d}{suffix}";record=evaluate_candidate(env,config,candidate,engine,parameters)
        record.update({"trial":trial.number,"checkpoint":checkpoint,"study":name});append_jsonl(paths["out"]/"trials.jsonl",record)
        trial.set_user_attr("status",record["status"]);trial.set_user_attr("worst_context",record["worst_context"]);trial.set_user_attr("constraints",[0.0 if record["status"]=="complete" else 1.0])
        if (existing+trial.number+1)%int(config["visualization"]["update_every_trials"])==0: build_dashboard(env)
        return float(record["effect_score"])
    try: study.optimize(objective,n_trials=additional,n_jobs=1)
    finally:
        current=read_trials(paths["out"]/"trials.jsonl");export_trials(paths,current);build_dashboard(env)
    current=read_trials(paths["out"]/"trials.jsonl");mine=[x for x in current if x.get("study")==name]
    summary={"study":name,"checkpoint":checkpoint,"additional_trials":additional,"total_trials":len(mine),"max_total_trials":max_total_trials,
             "best_trial":min(mine,key=lambda x:x["effect_score"])["trial"],"best_score":min(x["effect_score"] for x in mine),"completed_at":utcnow()}
    history.append(summary);write_json(history_path,history);select_best(paths,current);build_dashboard(env);write_listening(env);return summary


def select_best(paths:dict[str,Path],records:list[dict[str,Any]]) -> list[dict[str,Any]]:
    valid=[x for x in records if x.get("status")=="complete"]
    if not valid: write_json(paths["out"]/"best-candidates.json",[]);return []
    chosen=[]
    def add(row:dict[str,Any]):
        if row not in chosen:chosen.append(row)
    add(min(valid,key=lambda x:x["effect_score"]));add(min(valid,key=lambda x:x["worst_context"]))
    for engine in ("world","praat"):
        rows=[x for x in valid if x["engine"]==engine]
        if rows:add(min(rows,key=lambda x:x["effect_score"]))
    add(min(valid,key=lambda x:x["parameters"].get("modulation_factor",1)))
    add(min(valid,key=lambda x:statistics.mean([c["metrics"].get("normalized",{}).get("spectral_preservation",1e6) for c in x["contexts"].values()])))
    for row in sorted(valid,key=lambda x:x["effect_score"])[:10]:
        add(row)
        if len(chosen)>=10:break
    compact=[{"candidate":x["candidate"],"trial":x["trial"],"engine":x["engine"],"effect_score":x["effect_score"],"worst_context":x["worst_context"],"status":x["status"],"parameters":x["parameters"],"contexts":x["contexts"]} for x in chosen[:12]]
    write_json(paths["out"]/"best-candidates.json",compact);return compact


def build_dashboard(env:dict[str,str]) -> Path:
    paths=output_paths(env);records=read_trials(paths["out"]/"trials.jsonl");best=[];running=float("inf")
    for row in records:running=min(running,float(row["effect_score"]));best.append(running)
    methods={};engines={}
    for row in records:
        methods[row["parameters"].get("trend_method","-")]=methods.get(row["parameters"].get("trend_method","-"),0)+1;engines[row["engine"]]=engines.get(row["engine"],0)+1
    top=sorted(records,key=lambda x:x["effect_score"])[:20]
    rows="".join(f"<tr><td>{r.get('trial')}</td><td>{html.escape(r['engine'])}</td><td>{r['effect_score']:.4f}</td><td>{r['worst_context']:.4f}</td><td>{html.escape(str(r['parameters']))}</td><td>"+" ".join(f"<a href='{html.escape(os.path.relpath(c['output'],paths['out']))}'>{cid}</a>" for cid,c in r['contexts'].items())+"</td></tr>" for r in top)
    context_series={cid:[r["contexts"][cid]["score"] for r in records] for cid,_,_ in CONTEXTS}
    residual=[statistics.mean([c["metrics"].get("normalized",{}).get("modulation_residual",0) for c in r["contexts"].values()]) for r in records]
    drift=[statistics.mean([c["metrics"].get("normalized",{}).get("drift_preservation",0) for c in r["contexts"].values()]) for r in records]
    numeric=sorted({key for row in records for key,value in row["parameters"].items() if isinstance(value,(int,float)) and not isinstance(value,bool)})
    importance={}
    for key in numeric:
        pairs=[(float(r["parameters"][key]),float(r["effect_score"])) for r in records if key in r["parameters"] and math.isfinite(float(r["effect_score"]))]
        if len(pairs)>2:
            left,right=zip(*pairs);lm,rm=statistics.mean(left),statistics.mean(right);den=math.sqrt(sum((x-lm)**2 for x in left)*sum((y-rm)**2 for y in right));importance[key]=abs(sum((x-lm)*(y-rm) for x,y in pairs)/den) if den else 0
    baseline_path=paths["baselines"]/"baselines.json";baseline_rows=json.loads(baseline_path.read_text()) if baseline_path.is_file() else []
    parallel=[{"score":r["effect_score"],**{key:r["parameters"].get(key) for key in numeric}} for r in top]
    payload={"scores":[r["effect_score"] for r in records],"best":best,"worst":[r["worst_context"] for r in records],"median":[r["median_context"] for r in records],"residual":residual,"drift":drift,"contexts":context_series,"methods":methods,"engines":engines,"importance":importance,"parallel":parallel}
    baseline_html="".join(f"<tr><td>{x['candidate']}</td><td>{x['name']}</td><td>{x['effect_score']:.4f}</td></tr>" for x in baseline_rows)
    document=f"""<!doctype html><meta charset=utf-8><title>GLaDOS M dashboard</title><style>body{{font:14px system-ui;max-width:1200px;margin:2rem auto}}canvas{{border:1px solid #ccc;width:100%;height:260px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:.4rem;text-align:left}}pre{{white-space:pre-wrap}}</style><h1>Pitch Modulation laboratory</h1><p>Trials: {len(records)} · offline dashboard · passthrough anchor ≈ 1, target = 0.</p><p>Linhas: trial, best-so-far, pior contexto, mediana, residual rápido, drift e os quatro contextos.</p><canvas id=c width=1100 height=260></canvas><h2>Baselines</h2><table><tr><th>ID</th><th>Descrição</th><th>Score</th></tr>{baseline_html}</table><h2>Distribuição de métodos e motores</h2><pre>{html.escape(json.dumps({'methods':methods,'engines':engines},indent=2))}</pre><h2>Importância dos parâmetros</h2><p>Proxy determinístico: correlação absoluta parâmetro-score; não é importância causal.</p><pre>{html.escape(json.dumps(dict(sorted(importance.items(),key=lambda x:-x[1])),indent=2))}</pre><h2>Coordenadas paralelas — dados dos top 20</h2><pre>{html.escape(json.dumps(parallel,indent=2))}</pre><h2>Melhores candidatos</h2><table><tr><th>trial</th><th>engine</th><th>score</th><th>worst</th><th>parameters</th><th>WAVs</th></tr>{rows}</table><script>const d={json.dumps(payload)};let c=document.querySelector('#c'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);let all=[d.scores,d.best,d.worst,d.median,d.residual,d.drift,...Object.values(d.contexts)].flat(),mx=Math.max(1,...all.filter(Number.isFinite)),colors=['#888','#0a5','#d33','#36c','#b50','#707','#099','#f68','#a65','#4aa'];[d.scores,d.best,d.worst,d.median,d.residual,d.drift,...Object.values(d.contexts)].forEach((s,j)=>{{x.beginPath();x.strokeStyle=colors[j%colors.length];s.forEach((v,i)=>{{let px=10+i*(1080/Math.max(1,s.length-1)),py=250-v/mx*235;i?x.lineTo(px,py):x.moveTo(px,py)}});x.stroke()}});</script>"""
    target=paths["out"]/"optimization-dashboard.html";atomic_text(target,document);return target


def write_listening(env:dict[str,str]) -> dict[str,str]:
    paths=output_paths(env);best=json.loads((paths["out"]/"best-candidates.json").read_text()) if (paths["out"]/"best-candidates.json").is_file() else []
    contexts=prepared_contexts(env);blind_map={};identified=[];blind=[]
    criteria=["semelhança com o efeito de modulação do Melodyne","caráter GLaDOS","rigidez tonal","preservação do movimento lento","inteligibilidade","qualidade das vogais","qualidade das consoantes","artefatos interessantes","artefatos ruins"]
    for candidate in best:
        sections=[];blind_sections=[]
        for context in contexts:
            cid=context["id"]
            def listening_copy(source_path:Path,destination:Path)->Path:
                import numpy as np
                audio,sr=load_audio(source_path);peak=float(np.max(np.abs(audio))) or 1.0;write_audio(destination,audio*(10**(-1/20))/peak,sr,"PCM_24");return destination
            if parse_bool(env["AEGIS_GLADOS_MOD_NORMALIZE_LISTENING_FILES"]):
                refdir=paths["normalized"]/"references";canddir=paths["normalized"]/candidate["candidate"]
                source_path=listening_copy(context["source_path"],refdir/f"{cid}-source.wav");target_path=listening_copy(context["target_path"],refdir/f"{cid}-target.wav");output_path=listening_copy(Path(candidate["contexts"][cid]["output"]),canddir/f"{cid}.wav")
            else:source_path,target_path,output_path=context["source_path"],context["target_path"],Path(candidate["contexts"][cid]["output"])
            source=os.path.relpath(source_path,paths["out"]);target=os.path.relpath(target_path,paths["out"]);output=os.path.relpath(output_path,paths["out"])
            token=hashlib.sha256(f"{candidate['candidate']}:{cid}".encode()).digest()[0]%2;refs=[target,output] if token==0 else [output,target];blind_map[f"{candidate['candidate']}:{cid}"]={"A":"target" if token==0 else "candidate","B":"candidate" if token==0 else "target"}
            sections.append(f"<h3>{cid}</h3>source<audio controls src='{html.escape(source)}'></audio> target Melodyne<audio controls src='{html.escape(target)}'></audio> candidate<audio controls src='{html.escape(output)}'></audio>")
            blind_sections.append(f"<h3>{cid}</h3>source<audio controls src='{html.escape(source)}'></audio> referência A<audio controls src='{html.escape(refs[0])}'></audio> resultado B<audio controls src='{html.escape(refs[1])}'></audio>")
        identified.append(f"<section><h2>{candidate['candidate']}</h2>{''.join(sections)}</section>");blind.append(f"<section><h2>Amostra {len(blind)+1}</h2>{''.join(blind_sections)}</section>")
    form="".join(f"<label>{x} <input type=range min=1 max=5 value=3 data-key='{html.escape(x)}'></label><br>" for x in criteria)+"<textarea id=notes placeholder='observações'></textarea><button onclick='save()'>Salvar/exportar JSON</button><pre id=result></pre>"
    script="<script>function save(){let d={};document.querySelectorAll('[data-key]').forEach(x=>d[x.dataset.key]=+x.value);d.observacoes=notes.value;localStorage.setItem('aegis-glados-modulation-review',JSON.stringify(d));result.textContent=JSON.stringify(d,null,2);let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([result.textContent],{type:'application/json'}));a.download='modulation-listening-review.json';a.click()}</script>"
    style="<meta charset=utf-8><style>body{font:14px system-ui;max-width:1200px;margin:auto}audio{display:block;margin:.3rem 0 1rem}section{border-top:1px solid #ccc}</style>"
    atomic_text(paths["out"]/"listening.html",f"<!doctype html>{style}<h1>Escuta identificada — módulo M</h1>{''.join(identified)}{form}{script}")
    atomic_text(paths["out"]/"listening-blind.html",f"<!doctype html>{style}<h1>Escuta cega — módulo M</h1>{''.join(blind)}{form}{script}");write_json(paths["out"]/"blind-map.json",blind_map)
    return {"identified":str(paths["out"]/"listening.html"),"blind":str(paths["out"]/"listening-blind.html"),"map":str(paths["out"]/"blind-map.json")}


def render_command(env:dict[str,str],candidate_id:str) -> dict[str,Any]:
    paths=output_paths(env);records=read_trials(paths["out"]/"trials.jsonl");matches=[x for x in records if x["candidate"]==candidate_id]
    if not matches: raise LabError(f"unknown candidate: {candidate_id}")
    row=matches[-1];return evaluate_candidate(env,read_yaml(Path(env["AEGIS_GLADOS_MOD_CONFIG"])),candidate_id,row["engine"],row["parameters"])


def compare_command(env:dict[str,str],candidate_id:str) -> dict[str,Any]:
    paths=output_paths(env);records=read_trials(paths["out"]/"trials.jsonl");matches=[x for x in records if x["candidate"]==candidate_id]
    if not matches: raise LabError(f"unknown candidate: {candidate_id}")
    return matches[-1]


def report(env:dict[str,str]) -> dict[str,Any]:
    paths=output_paths(env);manifest=json.loads((paths["analysis"]/"preflight.json").read_text()) if (paths["analysis"]/"preflight.json").is_file() else preflight(env)
    analyses=json.loads((paths["analysis"]/"modulation-analysis.json").read_text()) if (paths["analysis"]/"modulation-analysis.json").is_file() else {}
    baselines=json.loads((paths["baselines"]/"baselines.json").read_text()) if (paths["baselines"]/"baselines.json").is_file() else []
    trials=read_trials(paths["out"]/"trials.jsonl");main_trials=[x for x in trials if x.get("study")=="melodyne-modulation-replication"];smoke_trials=[x for x in trials if x.get("study")=="melodyne-modulation-replication-smoke"];best=select_best(paths,trials);links=write_listening(env);dashboard=build_dashboard(env)
    def table_rows(rows:list[dict[str,Any]])->str:
        return "\n".join(f"| {x.get('candidate',x.get('name'))} | {x['engine']} | {x['effect_score']:.4f} | {x['worst_context']:.4f} | {x['status']} |" for x in rows)
    state_rows="\n".join(f"| {state} | `{info['path']}` | {info['sample_rate']} | {info['duration']:.6f} |" for state,info in manifest["states"].items())
    invalid=[x for x in trials if x.get("status")!="complete"];near=[x for x in trials if .95<=x.get("effect_score",99)<=1.05]
    document=f"""# Laboratório GLaDOS — Pitch Modulation (M)

Gerado em {utcnow()}.

## Arquivos e mapeamento

| Estado | Arquivo | Hz | Duração (s) |
|---|---|---:|---:|
{state_rows}

O mapeamento explícito confirma `Voice_FormantAndPitchCentralization.wav → CF`. Os quatro pares são clean→M, C→CM, F→FM e CF→CFM; apenas offset global foi removido, sem time warping.

## Validação e análise

Todos os arquivos existem, contêm áudio mono e têm durações compatíveis (span {manifest['duration_span_seconds']:.6f} s). Sample rates são normalizados para 24 kHz. Detalhes completos: `{paths['report']/ 'analysis/modulation-analysis.json'}`.

```json
{json.dumps(analyses.get('contexts',{}),indent=2,ensure_ascii=False)}
```

## Baselines e candidatos

| Candidato | Motor | Score | Pior contexto | Estado |
|---|---|---:|---:|---|
{table_rows(baselines if isinstance(baselines,list) else baselines.get('baselines',[]))}
{table_rows(best)}

Trials registrados: {len(main_trials)} no estudo principal e {len(smoke_trials)} no smoke separado. Saídas inválidas: {len(invalid)}. Houve um trial principal interrompido antes da persistência e marcado `FAIL` somente no Optuna. Candidatos quase passthrough: {len(near)}.

Motores disponíveis: {', '.join(k for k,v in manifest['engines'].items() if v['available']) or 'nenhum'}. Motores indisponíveis: {', '.join(k for k,v in manifest['engines'].items() if not v['available']) or 'nenhum'}.

Os parâmetros pesquisados incluem fator de modulação, método/janela de tendência, segmentação acústica, ataques/finais e transições, com parâmetros condicionais por método e por motor. Métricas de residual rápido, drift, F0, espectro, V/UV e timing são persistidas separadamente em cada contexto de cada trial. O pior contexto e os componentes completos dos melhores candidatos estão na tabela/JSON acima e em `best-candidates.json`.

## Limitações e conclusão

Diferenças ainda não reproduzidas devem ser decididas pela comparação auditiva; métricas espectrais não provam causalidade e WORLD/Praat introduzem assinaturas de ressíntese diferentes. A conclusão permanece **inconclusivo** até avaliação humana. Não declaramos automaticamente replicação.

- Dashboard: `{dashboard}`
- Escuta identificada: `{links['identified']}`
- Escuta cega: `{links['blind']}`
"""
    target=paths["report"]/"summary.md";atomic_text(target,document);return {"report":str(target),"dashboard":str(dashboard),**links,"trials":len(trials),"invalid":len(invalid),"conclusion":"inconclusivo (depende de avaliação humana)"}


def parser() -> argparse.ArgumentParser:
    result=argparse.ArgumentParser(description=__doc__);result.add_argument("--config");result.add_argument("--dataset-config");result.add_argument("--output-dir");result.add_argument("--report-dir")
    commands=result.add_subparsers(dest="command",required=True)
    for name in ("preflight","analyze","baseline","report"):commands.add_parser(name)
    opt=commands.add_parser("optimize");opt.add_argument("--trials",type=int);opt.add_argument("--max-total-trials",type=int);opt.add_argument("--resume",action="store_true");opt.add_argument("--study-suffix")
    for name in ("render","compare"):sub=commands.add_parser(name);sub.add_argument("--candidate",required=True)
    worker=commands.add_parser("_render-worker",help=argparse.SUPPRESS);worker.add_argument("--engine",choices=("world","praat"),required=True);worker.add_argument("--source",required=True);worker.add_argument("--output",required=True);worker.add_argument("--parameters-json",required=True)
    feature=commands.add_parser("_features-worker",help=argparse.SUPPRESS);feature.add_argument("--source",required=True)
    return result


def main(argv:list[str]|None=None) -> int:
    args=parser().parse_args(argv);env=environment(args)
    try:
        if args.command=="_render-worker":value=engine_registry(env)[args.engine].render(Path(args.source),Path(args.output),json.loads(args.parameters_json))
        elif args.command=="_features-worker":
            audio,sr=load_audio(Path(args.source),int(env["AEGIS_GLADOS_MOD_SAMPLE_RATE"]));value=_jsonable(fixed_features(audio,sr,float(env["AEGIS_GLADOS_MOD_FRAME_PERIOD_MS"])))
        elif args.command=="preflight":value=preflight(env)
        elif args.command=="analyze":value=analysis(env)
        elif args.command=="baseline":value=baseline(env)
        elif args.command=="optimize":value=optimize(env,args.trials or int(env["AEGIS_GLADOS_MOD_BATCH_TRIALS"]),args.max_total_trials or int(env["AEGIS_GLADOS_MOD_MAX_TOTAL_TRIALS"]),args.resume,args.study_suffix)
        elif args.command=="render":value=render_command(env,args.candidate)
        elif args.command=="compare":value=compare_command(env,args.candidate)
        else:value=report(env)
        print(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));return 0
    except (LabError,ValueError) as exc:
        print(f"error: {exc}",file=sys.stderr);return 2


if __name__=="__main__":raise SystemExit(main())
