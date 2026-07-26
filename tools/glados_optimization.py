"""Checkpointed Optuna search for the offline GLaDOS effects laboratory.

The optimizer is deliberately independent from production TTS.  This module
contains the persistence/scoring policy; rendering is delegated to
``glados_reverse_engineer`` so tests can replace it with a deterministic lab.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

EFFECTS = ("pitch_centralization", "pitch_modulation", "formant_shift")
CONSTRAINT_COUNT = 6
RENDER_FAILURE_CONSTRAINTS = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
SCORING_SCHEMA_VERSION = 2
RENDER_SCHEMA_VERSION = 2
FEATURE_NAMES = {
    "pitch_centralization": "pitch",
    "pitch_modulation": "modulation",
    "formant_shift": "formant",
}
EFFECT_PAIRS = {
    "pitch_centralization": (("none", "C"), ("M", "CM"), ("F", "CF"), ("FM", "CFM")),
    "pitch_modulation": (("none", "M"), ("C", "CM"), ("F", "FM"), ("CF", "CFM")),
    "formant_shift": (("none", "F"), ("C", "CF"), ("M", "FM"), ("CM", "CFM")),
}
STUDY_NAMES = (*EFFECTS, "chains")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def vector_distance(left: Any, right: Any) -> float:
    import numpy as np
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("inf")
    return float(np.linalg.norm(a - b))


def normalized_distance(output: Any, target: Any, source: Any, *, minimum: float,
                        epsilon: float = 1e-8) -> dict[str, Any]:
    """Normalize a distance while explicitly excluding weak denominators."""
    baseline = vector_distance(source, target)
    distance = vector_distance(output, target)
    if not math.isfinite(baseline) or baseline < minimum:
        return {"status": "non_informative", "baseline_distance": baseline,
                "output_distance": distance, "score": None}
    # epsilon is only a numerical guard; it never decides informativeness.
    score = distance / max(baseline, epsilon)
    return {"status": "informative", "baseline_distance": baseline,
            "output_distance": distance, "score": float(score)}


def aggregate_effect_contexts(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(x["score"]) for x in contexts if x.get("status") == "informative"]
    if not scores:
        return {"status": "non_informative", "score": None, "informative_contexts": 0}
    return {"status": "informative", "score": .5 * max(scores) + .5 * statistics.median(scores),
            "maximum": max(scores), "median": statistics.median(scores),
            "informative_contexts": len(scores)}


def final_chain_score(effect_scores: dict[str, float], artifacts_penalty: float = 0.0) -> float:
    values = [float(effect_scores[x]) for x in EFFECTS]
    return .8 * max(values) + .2 * statistics.mean(values) + float(artifacts_penalty)


def artifact_excess(output: dict[str, float], source: dict[str, float], target: dict[str, float],
                    tolerances: dict[str, float], weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Return raw excesses and a dimensionless, explicitly weighted penalty.

    Raw diagnostic units are intentionally preserved in ``excess``.  They must
    not be summed directly: one click is not comparable to one unit of clipping
    fraction.  The optional weights perform that unit conversion.
    """
    weights = weights or {name: 1.0 for name in tolerances}
    details, total = {}, 0.0
    for name, tolerance in tolerances.items():
        allowed = max(float(source.get(name, 0)), float(target.get(name, 0))) + float(tolerance)
        excess = max(0.0, float(output.get(name, 0)) - allowed)
        weight = float(weights.get(name, 0.0))
        penalty = excess * weight
        details[name] = {"allowed": allowed, "output": float(output.get(name, 0)),
                         "excess": excess, "weight": weight, "penalty": penalty}
        total += penalty
    return {"total": total, "details": details}


def chain_record_score(record: dict[str, Any]) -> float:
    """Canonical promotion score used by reports, validation and stability."""
    stored = record.get("minimax")
    if stored is not None and math.isfinite(float(stored)):
        return float(stored)
    objectives = record.get("objectives", [])
    if len(objectives) != len(EFFECTS):
        return float("inf")
    penalty = float(record.get("artifacts", {}).get("total", 0.0))
    return final_chain_score(dict(zip(EFFECTS, objectives)), penalty)


def technical_constraints(diag: dict[str, Any], *, expected_duration: float | None = None) -> list[float]:
    """Positive values are violations, in Optuna's constraints convention."""
    duration = float(diag.get("duration", 0) or 0)
    duration_error = abs(duration - expected_duration) / max(expected_duration, 1e-8) if expected_duration else 0
    return [
        1.0 if diag.get("empty", duration <= 0) else 0.0,
        1.0 if diag.get("nan", False) else 0.0,
        max(0.0, duration_error - .05),
        1.0 if diag.get("tracking_failed", False) else 0.0,
        1.0 if diag.get("clipping_fraction", 0) > .05 else 0.0,
        1.0 if diag.get("invalid_wav", False) else 0.0,
    ]


def merge_constraints(rows:list[list[float]])->list[float]:
    """Worst violation per fixed technical constraint across contexts."""
    if not rows:return list(RENDER_FAILURE_CONSTRAINTS)
    if any(len(row)!=CONSTRAINT_COUNT for row in rows):
        raise ValueError(f"every trial must have exactly {CONSTRAINT_COUNT} constraints")
    return [max(float(row[index]) for row in rows) for index in range(CONSTRAINT_COUNT)]


def informative_matrix(features: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    scoring = config["scoring"]
    minimums = scoring["minimum_informative_distance"]
    matrix, counts = {}, {}
    for effect in EFFECTS:
        name = FEATURE_NAMES[effect]
        rows = []
        for source, target in EFFECT_PAIRS[effect]:
            distance = vector_distance(features[source][effect], features[target][effect])
            rows.append({"source": source, "target": target, "distance": distance,
                         "minimum": float(minimums[name]),
                         "status": "informative" if distance >= float(minimums[name]) else "non_informative"})
        matrix[effect] = rows
        counts[effect] = sum(x["status"] == "informative" for x in rows)
    return {"effects": matrix, "informative_context_counts": counts,
            "ready": all(counts[x] >= 2 for x in EFFECTS)}


def allocation(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Deterministic largest-remainder allocation (200 => 40/40/40/80)."""
    if total <= 0 or any(float(weights.get(x, 0)) < 0 for x in STUDY_NAMES):
        raise ValueError("trial allocation must be positive")
    norm = sum(float(weights[x]) for x in STUDY_NAMES)
    if norm <= 0:
        raise ValueError("trial allocation weights sum to zero")
    exact = {x: total * float(weights[x]) / norm for x in STUDY_NAMES}
    result = {x: math.floor(exact[x]) for x in STUDY_NAMES}
    order = sorted(STUDY_NAMES, key=lambda x: (-(exact[x] - result[x]), STUDY_NAMES.index(x)))
    for name in order[:total - sum(result.values())]:
        result[name] += 1
    return result


def hypervolume_3d(points: list[list[float]], reference: list[float]) -> float:
    """Exact dominated volume for a small 3-D minimization front."""
    clipped = [tuple(min(float(p[i]), float(reference[i])) for i in range(3)) for p in points
               if all(float(p[i]) < float(reference[i]) for i in range(3))]
    if not clipped:
        return 0.0
    xs = sorted({p[0] for p in clipped} | {float(reference[0])})
    volume = 0.0
    for lo, hi in zip(xs, xs[1:]):
        active = [p for p in clipped if p[0] <= lo]
        ys = sorted({p[1] for p in active} | {float(reference[1])})
        area = 0.0
        for ylo, yhi in zip(ys, ys[1:]):
            zs = [p[2] for p in active if p[1] <= ylo]
            if zs:
                area += (yhi - ylo) * (float(reference[2]) - min(zs))
        volume += (hi - lo) * area
    return float(volume)


def pareto_front(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feasible = [x for x in records if not any(float(v) > 0 for v in x.get("constraints", []))
                and len(x.get("objectives", [])) == 3]
    result = []
    for item in feasible:
        p = item["objectives"]
        dominated = any(all(q <= v for q, v in zip(other["objectives"], p)) and
                        any(q < v for q, v in zip(other["objectives"], p))
                        for other in feasible if other is not item)
        if not dominated:
            result.append(item)
    return result


def nsga_survivors(records:list[dict[str,Any]],size:int)->list[dict[str,Any]]:
    feasible=[x for x in records if len(x.get("objectives",[]))==3 and not any(float(v)>0 for v in x.get("constraints",[]))]
    remaining=list(feasible);selected=[]
    while remaining and len(selected)<size:
        front=pareto_front(remaining)
        if len(selected)+len(front)<=size: selected.extend(front)
        else:
            crowd={id(x):0.0 for x in front}
            for axis in range(3):
                ordered=sorted(front,key=lambda x:x["objectives"][axis]);crowd[id(ordered[0])]=crowd[id(ordered[-1])]=float("inf")
                span=ordered[-1]["objectives"][axis]-ordered[0]["objectives"][axis]
                if span:
                    for index in range(1,len(ordered)-1):
                        crowd[id(ordered[index])]+=(ordered[index+1]["objectives"][axis]-ordered[index-1]["objectives"][axis])/span
            selected.extend(sorted(front,key=lambda x:crowd[id(x)],reverse=True)[:size-len(selected)])
        front_ids={id(x) for x in front};remaining=[x for x in remaining if id(x) not in front_ids]
    if len(selected)<size:
        infeasible=sorted((x for x in records if len(x.get("objectives",[]))==3 and x not in selected),key=lambda x:sum(max(0,float(v)) for v in x.get("constraints",[])))
        selected.extend(infeasible[:size-len(selected)])
    return selected


def _lab_module(lab: Any | None) -> Any:
    if lab is not None:
        return lab
    import glados_reverse_engineer
    return glados_reverse_engineer


def optimization_dir(env: dict[str, str]) -> Path:
    return Path(env["AEGIS_GLADOS_RE_OUTPUT_DIR"]) / "optimization"


def load_config(path: Path) -> dict[str, Any]:
    import yaml
    value = yaml.safe_load(path.read_text(encoding="utf8"))
    for section in ("references", "scoring", "optimization", "validation"):
        if section not in value:
            raise ValueError(f"optimization config missing {section}")
    if int(value["optimization"]["chains"]["population_size"]) <= 1:
        raise ValueError("population_size must be greater than one")
    configured_schema = int(value["scoring"].get("schema_version", 1))
    if configured_schema != SCORING_SCHEMA_VERSION:
        raise ValueError(f"scoring.schema_version must be {SCORING_SCHEMA_VERSION}, got {configured_schema}")
    return value


def validation_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    section = config["validation"]
    directory = Path(section.get("directory", "AegisVoicemods/validation"))
    if not directory.is_absolute():
        # config lives in this repository; validation paths are intentionally fixed outside optimizer output.
        directory = Path(__file__).resolve().parents[1] / directory
    return directory / section["clean_file"], directory / section["target_file"]


def validation_preflight(config: dict[str, Any], lab: Any) -> dict[str, Any]:
    clean, target = validation_paths(config)
    section=config["validation"];directory=clean.parent
    transcript=directory/section["transcript_file"] if section.get("transcript_file") else None
    result = {"clean": str(clean), "target": str(target), "transcript":str(transcript) if transcript else None,
              "present": clean.is_file() and target.is_file()}
    if not result["present"]:
        result["ready"] = not config["validation"].get("required_before_optimization", True)
        result["reason"] = "required external validation pair is missing"
        return result
    try:
        a, sr_a = lab.load_audio(clean); b, sr_b = lab.load_audio(target)
        durations = [len(a) / sr_a, len(b) / sr_b]
        result.update(durations_seconds=durations, ready=all(10 <= x <= 20 for x in durations),
                      transcript_present=bool(transcript and transcript.is_file()))
        if not result["ready"]:
            result["reason"] = "validation WAVs must each be 10-20 seconds"
        refs=section.get("references",{})
        if refs:
            paths={name:directory/file for name,file in refs.items()};missing=[str(path) for path in paths.values() if not path.is_file()]
            factorial={"complete":not missing,"missing":missing,"files":{name:str(path) for name,path in paths.items()}}
            if not missing:
                sr=24000;features={}
                for name,path in paths.items():
                    audio,_=lab.load_audio(path,sr);features[name]=lab.effect_feature_vectors(audio,sr)
                factorial["informative_distance_matrix"]=informative_matrix(features,config)
            result["factorial_references"]=factorial
    except Exception as exc:
        result.update(ready=False, reason=f"invalid validation pair: {exc}")
    return result


def optimization_preflight(env: dict[str, str], config: dict[str, Any], *, lab: Any | None = None) -> dict[str, Any]:
    lab = _lab_module(lab)
    analysis = effect_analysis(env,config,lab)
    matrix = informative_matrix(analysis["feature_vectors"], config) if analysis.get("complete_factorial") else {
        "ready": False, "effects": {}, "informative_context_counts": {}}
    validation = validation_preflight(config, lab)
    blockers = []
    if not analysis.get("complete_factorial"):
        blockers.append("factorial reference set is incomplete")
    if not matrix["ready"]:
        blockers.append("each effect requires at least two informative contexts")
    if config["validation"].get("enabled", True) and not validation["ready"]:
        blockers.append(validation.get("reason", "external validation is not ready"))
    return {"ready": not blockers, "blockers": blockers, "distance_matrix": matrix,
            "validation": validation, "analysis_file": str(optimization_dir(env).parent / "analysis/effects/effect-reference-analysis.json")}


def effect_analysis(env:dict[str,str],config:dict[str,Any],lab:Any)->dict[str,Any]:
    path=optimization_dir(env).parent/"analysis/effects/effect-reference-analysis.json"
    if path.is_file():
        try:
            cached=json.loads(path.read_text());root=Path(__file__).resolve().parents[1]/config["root"]
            if (cached.get("feature_schema_version") == SCORING_SCHEMA_VERSION and
                    cached.get("complete_factorial") and
                    all(cached["properties"][name]["sha256"]==lab.sha256(root/item["file"])
                        for name,item in config["references"].items())):
                return cached
        except (KeyError,OSError,json.JSONDecodeError):
            pass
    return lab.analyze_effect_references(env)


def suggest_effect(trial: Any, effect: str, available: set[str] | None = None) -> tuple[str, dict[str, Any]]:
    def choices(wanted:list[str])->list[str]:
        selected=[name for name in wanted if available is None or name in available]
        if not selected:raise ValueError(f"no available engine for {effect}: {wanted}")
        return selected
    if effect == "pitch_centralization":
        engine = trial.suggest_categorical("engine", choices(["praat", "world", "graillon_lv2"]))
        params = {"quantization": trial.suggest_categorical("quantization", ["chromatic", "c_major", "c_minor"]),
                  "correction_amount": trial.suggest_float("correction_amount", .2, 1.0),
                  "smooth": trial.suggest_float("smooth", 0, 1), "inertia": trial.suggest_float("inertia", 0, 1),
                  "wet": trial.suggest_float("wet", .4, 1.0)}
        if engine != "graillon_lv2":
            params.update(flatten_amount=trial.suggest_float("flatten_amount", 0, 1),
                          transition_ms=trial.suggest_int("transition_ms", 0, 60))
    elif effect == "pitch_modulation":
        engine = trial.suggest_categorical("engine", choices(["praat", "world", "graillon_lv2"]))
        params = {"quantization": "chromatic", "correction_amount": trial.suggest_float("correction_amount", 0, 1),
                  "preserve_modulation": trial.suggest_float("preserve_modulation", 0, 1),
                  "flatten_amount": trial.suggest_float("flatten_amount", 0, 1),
                  "smooth": trial.suggest_float("smooth", 0, 1), "inertia": trial.suggest_float("inertia", 0, 1),
                  "wet": trial.suggest_float("wet", .2, 1.0)}
    else:
        engine = trial.suggest_categorical("engine", choices(["world", "graillon_lv2"]))
        shift = trial.suggest_float("formant_shift", -4, 4)
        params = {"formant_shift": shift, "formant_enabled": True, "wet": trial.suggest_float("wet", .4, 1.0),
                  "correction_amount":0.0,"flatten_amount":0.0,"quantization":"chromatic"}
    return engine, params


def suggest_chain(trial: Any, available: set[str] | None = None) -> tuple[str, dict[str, Any]]:
    order = trial.suggest_categorical("order", ["CMF", "CFM", "MCF", "MFC", "FCM", "FMC"])
    specs = {}
    for letter, effect in zip("CMF", EFFECTS):
        engine, params = suggest_effect(_PrefixedTrial(trial, letter.lower()), effect,available)
        specs[letter] = {"engine": engine, "parameters": params}
    return "chain", {"order": order, "stages": [specs[x] for x in order]}


class _PrefixedTrial:
    def __init__(self, trial: Any, prefix: str): self.trial, self.prefix = trial, prefix
    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        return self.trial.suggest_categorical(f"{self.prefix}_{name}", choices)
    def suggest_float(self, name: str, low: float, high: float, **kwargs: Any) -> float:
        return self.trial.suggest_float(f"{self.prefix}_{name}", low, high, **kwargs)
    def suggest_int(self, name: str, low: int, high: int, **kwargs: Any) -> int:
        return self.trial.suggest_int(f"{self.prefix}_{name}", low, high, **kwargs)


class OptimizationRunner:
    def __init__(self, env: dict[str, str], config: dict[str, Any], lab: Any):
        self.env, self.config, self.lab = env, config, lab
        self.out = optimization_dir(env); self.out.mkdir(parents=True, exist_ok=True)
        self.audio = self.out / "audio"; self.audio.mkdir(exist_ok=True)
        self.cache = self.out / "cache"; self.cache.mkdir(exist_ok=True)
        self.registry = lab.engines(env)
        self.available_engines={name for name,engine in self.registry.items() if name!="chain" and engine.preflight().get("available")}
        self.sr = int(env["AEGIS_GLADOS_RE_SAMPLE_RATE"])
        self.analysis = effect_analysis(env,config,lab)
        root = Path(__file__).resolve().parents[1] / config["root"]
        self.references = {name: root / item["file"] for name, item in config["references"].items()}
        self.ref_features = self.analysis["feature_vectors"]
        self.ref_diag = {name: self._diagnostics(path) for name, path in self.references.items()}
        self.checkpoint = len(self._history()) + 1

    def _history(self) -> list[dict[str, Any]]:
        path = self.out / "checkpoint-history.json"
        return json.loads(path.read_text()) if path.is_file() else []

    def _diagnostics(self, path: Path) -> dict[str, Any]:
        try:
            audio, sr = self.lab.load_audio(path, self.sr)
            d = self.lab.diagnostics(audio, sr, len(audio)); d["duration"] = len(audio) / sr
            try:
                import numpy as np
                d["nan"] = bool(not np.all(np.isfinite(audio)))
                d["clipping_fraction"] = float(np.mean(np.abs(audio) >= .999))
            except Exception:
                pass
            return d
        except Exception:
            return {"invalid_wav": True, "duration": 0, "empty": True}

    def _cache_key(self, source: Path, engine: str, params: dict[str, Any]) -> str:
        payload = json.dumps({"render_schema_version": RENDER_SCHEMA_VERSION,
                              "source": self.lab.sha256(source), "engine": engine,
                              "parameters": params}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def render(self, source: Path, engine: str, params: dict[str, Any], destination: Path,
               *, use_cache: bool = True) -> tuple[Any, bool]:
        key = self._cache_key(source, engine, params); cached = self.cache / f"{key}.wav"
        if use_cache and cached.is_file():
            shutil.copy2(cached, destination)
            self._fit_render_duration(source,destination)
            return self.lab.RenderResult(True, str(destination), 0, details={"cache_key": key}), True
        instance = self.registry.get(engine)
        if instance is None or not instance.preflight().get("available"):
            return self.lab.RenderResult(False, None, 0, f"engine unavailable: {engine}"), False
        result = instance.render(source, params, destination)
        if result.success and use_cache:
            self._fit_render_duration(source,destination)
            shutil.copy2(destination, cached)
        elif result.success:
            self._fit_render_duration(source,destination)
        return result, False

    def _fit_render_duration(self,source:Path,destination:Path)->None:
        source_audio,source_sr=self.lab.load_audio(source)
        output,_=self.lab.load_audio(destination,source_sr)
        fitted=self.lab.fit_duration(output,len(source_audio))
        self.lab.write_audio(destination,fitted,source_sr)

    def score_output(self, output: Path, source_name: str, target_name: str, effect: str) -> dict[str, Any]:
        audio, _ = self.lab.load_audio(output, self.sr); features = self.lab.effect_feature_vectors(audio, self.sr)
        scoring = self.config["scoring"]
        return normalized_distance(features[effect], self.ref_features[target_name][effect],
                                   self.ref_features[source_name][effect],
                                   minimum=float(scoring["minimum_informative_distance"][FEATURE_NAMES[effect]]),
                                   epsilon=float(scoring["denominator_epsilon"]))

    def evaluate_effect(self, trial: Any, effect: str) -> tuple[float, dict[str, Any]]:
        engine, params = suggest_effect(trial, effect,self.available_engines); contexts=[]; total_elapsed=0.; cache_hits=0; constraint_rows=[]
        trial_dir = self.audio / effect / f"{trial.number:06d}"; trial_dir.mkdir(parents=True, exist_ok=True)
        for source_name, target_name in EFFECT_PAIRS[effect]:
            destination = trial_dir / f"{source_name}-to-{target_name}.wav"
            result, hit = self.render(self.references[source_name], engine, params, destination)
            total_elapsed += result.elapsed_seconds; cache_hits += int(hit)
            if not result.success:
                contexts.append({"source":source_name,"target":target_name,"status":"failed","score":None,"error":result.error})
                constraint_rows.append(list(RENDER_FAILURE_CONSTRAINTS)); continue
            scored = self.score_output(destination, source_name, target_name, effect)
            diag=self._diagnostics(destination)
            contexts.append({"source":source_name,"target":target_name,**scored,"duration_seconds":diag.get("duration")})
            constraint_rows.append(technical_constraints(diag,expected_duration=self.ref_diag[source_name]["duration"]))
        aggregate = aggregate_effect_contexts(contexts)
        constraints=merge_constraints(constraint_rows)
        if aggregate["status"] != "informative":constraints[3]=max(constraints[3],1.0)
        score = float(aggregate.get("score") if aggregate.get("score") is not None else 1e6)
        audio_files=[str(path.relative_to(self.out)) for path in sorted(trial_dir.glob("*.wav"))]
        return score, {"engine":engine,"parameters":params,"contexts":contexts,"aggregate":aggregate,
                       "constraints":constraints,"elapsed_seconds":total_elapsed,"cache_hits":cache_hits,
                       "audio_file":audio_files[0] if audio_files else None,"audio_files":audio_files}

    def evaluate_chain(self, trial: Any, *, source: Path | None = None, target_features: dict[str, Any] | None = None,
                       destination: Path | None = None) -> tuple[tuple[float,float,float], dict[str, Any]]:
        engine, params = suggest_chain(trial,self.available_engines); source = source or self.references["none"]
        destination = destination or self.audio / "chains" / f"{trial.number:06d}.wav"; destination.parent.mkdir(parents=True,exist_ok=True)
        result, hit = self.render(source, engine, params, destination)
        if not result.success:
            return (1e6,1e6,1e6), {"engine":engine,"parameters":params,"constraints":list(RENDER_FAILURE_CONSTRAINTS),"error":result.error,"cache_hit":hit}
        audio, _ = self.lab.load_audio(destination, self.sr); vectors=self.lab.effect_feature_vectors(audio,self.sr)
        targets = target_features or self.ref_features["CFM"]
        scores={}
        for effect in EFFECTS:
            item=normalized_distance(vectors[effect],targets[effect],self.ref_features["none"][effect],
                minimum=float(self.config["scoring"]["minimum_informative_distance"][FEATURE_NAMES[effect]]),
                epsilon=float(self.config["scoring"]["denominator_epsilon"]))
            scores[effect]=float(item["score"] if item["score"] is not None else 1e6)
        diag=self._diagnostics(destination); constraints=technical_constraints(diag,expected_duration=self.ref_diag["none"]["duration"])
        artifact=artifact_excess(diag,self.ref_diag["none"],self.ref_diag["CFM"],
            self.config["scoring"].get("artifact_tolerance",{}),
            self.config["scoring"].get("artifact_penalty_weights",{}))
        return tuple(scores[x] for x in EFFECTS), {"engine":engine,"parameters":params,"scores":scores,
            "minimax":final_chain_score(scores,artifact["total"]),"artifacts":artifact,"constraints":constraints,
            "duration_seconds":diag.get("duration"),"elapsed_seconds":result.elapsed_seconds,"cache_hit":hit,"audio_file":str(destination.relative_to(self.out))}

    def record(self, study_name: str, trial: Any, details: dict[str, Any]) -> None:
        values=list(trial.values or []); constraints=details.get("constraints",[])
        if trial.state.name!="COMPLETE": constraints=list(RENDER_FAILURE_CONSTRAINTS)
        prior=[x for x in read_jsonl(self.out/"trials.jsonl") if x.get("study")==study_name and x.get("feasible")]
        scalar=max(values) if values else float("inf")
        best_so_far=not any(max(x.get("objectives",[float("inf")])) <= scalar for x in prior)
        current={"objectives":values,"constraints":constraints}
        record={"study":study_name,"checkpoint":self.checkpoint,"trial":trial.number,"generation":
                trial.number // int(self.config["optimization"]["chains"]["population_size"]) if study_name=="chains" else None,
                "timestamp":utcnow(),"seed":self.config["optimization"]["tpe"]["seed"],"state":trial.state.name,
                "objectives":values,"constraints":constraints,"feasible":not any(float(x)>0 for x in constraints),
                "best_so_far":best_so_far,"pareto_at_record_time":study_name=="chains" and
                current in pareto_front(prior+[current]),**details}
        append_jsonl(self.out/"trials.jsonl",record)
        export_csv(self.out,read_jsonl(self.out/"trials.jsonl"))
        if study_name=="chains": trial.set_user_attr("constraints",constraints)
        if self.total_trials() % 10 == 0:
            rebuild_dashboard(self.env,self.config)

    def total_trials(self) -> int:
        return sum(1 for _ in read_jsonl(self.out/"trials.jsonl"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file(): return []
    result=[]
    for line in path.read_text(encoding="utf8").splitlines():
        try: result.append(json.loads(line))
        except json.JSONDecodeError: continue
    return result


def audio_content_sha256(lab: Any, path: Path) -> str:
    """Hash decoded samples, excluding nondeterministic WAV container metadata."""
    import numpy as np
    audio, sr = lab.load_audio(path)
    values = np.ascontiguousarray(np.asarray(audio, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(int(sr)).encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def completed_records(records:list[dict[str,Any]])->list[dict[str,Any]]:
    return [row for row in records if row.get("state")=="COMPLETE" and len(row.get("objectives",[])) in (1,3)]


def _sampler(config: dict[str, Any], name: str, *, seed_offset: int = 0) -> Any:
    import optuna
    opt=config["optimization"]
    seed=int(opt["tpe"]["seed"])+int(seed_offset)
    if name=="chains":
        return optuna.samplers.NSGAIISampler(seed=seed,
            population_size=int(opt["chains"]["population_size"]),
            constraints_func=lambda t: t.user_attrs.get("constraints",[]))
    return optuna.samplers.TPESampler(seed=seed,
        n_startup_trials=int(opt["tpe"]["n_startup_trials"]),multivariate=bool(opt["tpe"]["multivariate"]),
        group=bool(opt["tpe"]["group"]),constraints_func=lambda t:t.user_attrs.get("constraints",[]))


def _create_study(config: dict[str, Any], out: Path, name: str, *, resume: bool) -> Any:
    import optuna
    storage=f"sqlite:///{out/'studies.sqlite3'}"
    directions=["minimize"]*3 if name=="chains" else ["minimize"]
    seed_offset=0
    if resume:
        try: seed_offset=len(optuna.load_study(study_name=name,storage=storage).trials)
        except KeyError: seed_offset=0
    effective_seed=int(config["optimization"]["tpe"]["seed"])+seed_offset
    study=optuna.create_study(study_name=name,storage=storage,directions=directions,
                              sampler=_sampler(config,name,seed_offset=seed_offset),
                              load_if_exists=resume)
    expected_schema={"scoring":SCORING_SCHEMA_VERSION,"render":RENDER_SCHEMA_VERSION}
    previous_schema=study.user_attrs.get("pipeline_schema")
    if previous_schema is not None and previous_schema != expected_schema:
        raise ValueError(f"study {name} uses incompatible pipeline schema {previous_schema}; expected {expected_schema}")
    if resume and study.trials and previous_schema is None:
        raise ValueError(f"study {name} predates pipeline schema {expected_schema}; archive it and start a new study")
    study.set_user_attr("pipeline_schema",expected_schema)
    study.set_user_attr("sampler_seed",effective_seed)
    if name=="chains":
        size=int(config["optimization"]["chains"]["population_size"])
        previous=study.user_attrs.get("population_size")
        if previous is not None and int(previous)!=size:
            raise ValueError(f"population_size is immutable ({previous} in study, {size} requested)")
        study.set_user_attr("population_size",size)
    return study


def _trial_manifest(runner: OptimizationRunner, study: Any, name: str, evaluate: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def objective(trial: Any) -> Any:
        started=time.monotonic()
        if name=="chains": values,details=evaluate(trial)
        else: value,details=evaluate(trial);values=value
        details["wall_seconds"]=time.monotonic()-started
        trial.set_user_attr("constraints",details.get("constraints",[]))
        trial.set_user_attr("engine",details.get("engine"));trial.set_user_attr("parameters",details.get("parameters",{}))
        return values
    def callback(study: Any,trial: Any)->None:
        details={"engine":trial.user_attrs.get("engine"),"parameters":trial.user_attrs.get("parameters",{}),
                 "constraints":trial.user_attrs.get("constraints",[])}
        # Full details are persisted by objective wrapper through a temporary user attr.
        details.update(trial.user_attrs.get("record_details",{}));runner.record(name,trial,details)
    def wrapped(trial:Any)->Any:
        if name=="chains": values,details=evaluate(trial)
        else: values,details=evaluate(trial)
        details["sampler_seed"]=study.user_attrs.get("sampler_seed")
        trial.set_user_attr("constraints",details.get("constraints",[]));trial.set_user_attr("record_details",details)
        return values
    wrapped.callback=callback  # type: ignore[attr-defined]
    return wrapped


def optimize(env: dict[str, str], config_path: Path, *, trials: int, max_total_trials: int,
             resume: bool = False, lab: Any | None = None) -> dict[str, Any]:
    if trials <= 0 or max_total_trials <= 0: raise ValueError("--trials and --max-total-trials must be positive")
    lab=_lab_module(lab);config=load_config(config_path);out=optimization_dir(env);db=out/"studies.sqlite3"
    if db.exists() and not resume: raise ValueError("optimization studies already exist; pass --resume")
    if not db.exists() and resume: raise ValueError("cannot --resume before the first checkpoint")
    preflight=optimization_preflight(env,config,lab=lab)
    if not preflight["ready"]: raise ValueError("optimization blocked: "+"; ".join(preflight["blockers"]))
    existing=len(completed_records(read_jsonl(out/"trials.jsonl")));configured=min(int(config["optimization"]["max_total_trials"]),int(max_total_trials))
    additional=min(int(trials),configured-existing)
    if additional<=0: raise ValueError(f"maximum total trials reached ({configured})")
    runner=OptimizationRunner(env,config,lab);counts=allocation(additional,config["optimization"]["allocation"])
    # Make the dashboard discoverable before the first (potentially long,
    # four-context) audio trial finishes. Subsequent refreshes remain every 10
    # completed trials and at checkpoint completion.
    rebuild_dashboard(env,config)
    studies={}
    for name in STUDY_NAMES:
        study=_create_study(config,out,name,resume=resume);studies[name]=study
        evaluate=(lambda t,n=name: runner.evaluate_effect(t,n)) if name!="chains" else runner.evaluate_chain
        wrapped=_trial_manifest(runner,study,name,evaluate)
        # Rendering failures are converted into fixed technical constraints by
        # the objective. Any other exception is a programming/storage error and
        # must stop the checkpoint immediately instead of burning more trials.
        study.optimize(wrapped,n_trials=counts[name],callbacks=[wrapped.callback],n_jobs=1)
    return finish_checkpoint(runner,studies,preflight,counts)


def finish_checkpoint(runner: OptimizationRunner, studies: dict[str, Any], preflight: dict[str, Any], counts: dict[str,int]) -> dict[str,Any]:
    records=read_jsonl(runner.out/"trials.jsonl");chains=[x for x in records if x["study"]=="chains"]
    front=pareto_front(chains);reference=runner.config["optimization"]["pareto"]["reference_point"]
    pareto={"reference_point":reference,"hypervolume":hypervolume_3d([x["objectives"] for x in front],reference),"trials":front}
    write_json(runner.out/"pareto-front.json",pareto)
    populations=[];population_size=int(runner.config["optimization"]["chains"]["population_size"])
    for generation in sorted({x["generation"] for x in chains}):
        individuals=[x for x in chains if x["generation"]==generation]
        survivors=nsga_survivors([x for x in chains if x["generation"]<=generation],population_size)
        item={"generation":generation,"individuals":individuals,"elite_survivors":survivors,
              "lineage_available":False,"parents":None,"crossover":None,"mutations":None}
        populations.append(item)
    atomic_text(runner.out/"populations.jsonl","".join(json.dumps(x,ensure_ascii=False,allow_nan=False)+"\n" for x in populations))
    stability=evaluate_stability(runner,front)
    validation=evaluate_validation(runner,front)
    write_json(runner.out/"validation-results.json",validation)
    history=runner._history();summary={"checkpoint":runner.checkpoint,"created_at":utcnow(),"additional_trials":counts,
        "total_trials":len(records),"hypervolume":pareto["hypervolume"],"validation":validation,"stability":stability,
        "strict_reproducibility":runner.config["optimization"]["reproducibility"]["strict_mode"],
        "suggestion_sequence_may_vary":not runner.config["optimization"]["reproducibility"]["strict_mode"]}
    history.append(summary);write_json(runner.out/"checkpoint-history.json",history)
    export_csv(runner.out,records);rebuild_dashboard(runner.env,runner.config)
    return summary


def evaluate_stability(runner:OptimizationRunner,front:list[dict[str,Any]])->dict[str,Any]:
    repeats=int(runner.config["optimization"]["reproducibility"].get("rerender_best_candidates",3))
    top=sorted(front,key=chain_record_score)[:3]
    results=[]
    for rank,item in enumerate(top,1):
        runs=[]
        for repeat in range(repeats):
            destination=runner.audio/"stability"/f"top-{rank:02d}-run-{repeat+1}.wav";destination.parent.mkdir(parents=True,exist_ok=True)
            rendered,_=runner.render(runner.references["none"],"chain",item.get("parameters") or {},destination,use_cache=False)
            if not rendered.success:
                runs.append({"success":False,"error":rendered.error});continue
            audio,_=runner.lab.load_audio(destination,runner.sr);vectors=runner.lab.effect_feature_vectors(audio,runner.sr);objectives=[]
            for effect in EFFECTS:
                value=normalized_distance(vectors[effect],runner.ref_features["CFM"][effect],runner.ref_features["none"][effect],minimum=float(runner.config["scoring"]["minimum_informative_distance"][FEATURE_NAMES[effect]]),epsilon=float(runner.config["scoring"]["denominator_epsilon"]))
                objectives.append(float(value["score"] if value["score"] is not None else 1e6))
            diag=runner._diagnostics(destination)
            artifact=artifact_excess(diag,runner.ref_diag["none"],runner.ref_diag["CFM"],
                runner.config["scoring"].get("artifact_tolerance",{}),
                runner.config["scoring"].get("artifact_penalty_weights",{}))
            file_hash=runner.lab.sha256(destination);audio_hash=audio_content_sha256(runner.lab,destination)
            runs.append({"success":True,"objectives":objectives,
                "final_score":final_chain_score(dict(zip(EFFECTS,objectives)),artifact["total"]),
                "artifacts":artifact,"sha256":file_hash,"audio_sha256":audio_hash,
                "audio_file":str(destination.relative_to(runner.out))})
        scores=[x["final_score"] for x in runs if x.get("success")]
        results.append({"training_trial":item["trial"],"runs":runs,"median":statistics.median(scores) if scores else None,
                        "variance":statistics.pvariance(scores) if len(scores)>1 else 0.0 if scores else None,
                        "hashes_identical":len({x["audio_sha256"] for x in runs if x.get("success")})<=1,
                        "file_hashes_identical":len({x["sha256"] for x in runs if x.get("success")})<=1})
    payload={"rerenders_per_candidate":repeats,"candidates":results}
    write_json(runner.out/"stability-results.json",payload);return payload


def evaluate_validation(runner: OptimizationRunner, front: list[dict[str,Any]]) -> dict[str,Any]:
    section=runner.config["validation"]
    if not section.get("enabled",True): return {"enabled":False,"results":[]}
    clean,target=validation_paths(runner.config);audio,_=runner.lab.load_audio(target,runner.sr);target_features=runner.lab.effect_feature_vectors(audio,runner.sr)
    top=sorted(front,key=chain_record_score)[:int(section["evaluate_top_k"])]
    results=[]
    # Parameters are persisted independently from Optuna's internal suggestion attrs.
    for rank,item in enumerate(top,1):
        params=item.get("parameters") or {};destination=runner.audio/"validation"/f"top-{rank:02d}.wav";destination.parent.mkdir(parents=True,exist_ok=True)
        result,hit=runner.render(clean,"chain",params,destination)
        if not result.success:
            results.append({"training_trial":item["trial"],"overfit":True,"constraints":[1.0],"error":result.error});continue
        values=[];candidate,_=runner.lab.load_audio(destination,runner.sr);vectors=runner.lab.effect_feature_vectors(candidate,runner.sr)
        clean_audio,_=runner.lab.load_audio(clean,runner.sr);clean_vectors=runner.lab.effect_feature_vectors(clean_audio,runner.sr)
        for effect in EFFECTS:
            n=normalized_distance(vectors[effect],target_features[effect],clean_vectors[effect],minimum=float(runner.config["scoring"]["minimum_informative_distance"][FEATURE_NAMES[effect]]),epsilon=float(runner.config["scoring"]["denominator_epsilon"]))
            values.append(float(n["score"] if n["score"] is not None else 1e6))
        diag=runner._diagnostics(destination);constraints=technical_constraints(diag,expected_duration=len(clean_audio)/runner.sr)
        clean_diag=runner._diagnostics(clean);target_diag=runner._diagnostics(target)
        artifact=artifact_excess(diag,clean_diag,target_diag,
            runner.config["scoring"].get("artifact_tolerance",{}),
            runner.config["scoring"].get("artifact_penalty_weights",{}))
        training=chain_record_score(item);validation=final_chain_score(dict(zip(EFFECTS,values)),artifact["total"])
        degradation=validation-training;overfit=degradation>float(section["reject_if_score_degradation_over"]) or any(x>0 for x in constraints)
        results.append({"training_trial":item["trial"],"training_final_score":training,"validation_objectives":values,
            "validation_final_score":validation,"degradation":degradation,"overfit":overfit,
            "artifacts":artifact,"constraints":constraints,
            "cache_hit":hit,"audio_file":str(destination.relative_to(runner.out)),"participates_in_optimization":False})
    recommended=[x["training_trial"] for x in sorted(results,key=lambda x:x.get("validation_final_score",float("inf"))) if not x.get("overfit")]
    return {"enabled":True,"evaluated_top_k":len(results),"threshold":section["reject_if_score_degradation_over"],"results":results,
            "recommended_training_trials":recommended,"overfit_candidates_remain_in_training_pareto":True}


def export_csv(out:Path,records:list[dict[str,Any]])->None:
    fields=["study","checkpoint","trial","generation","timestamp","seed","state","engine","objectives","constraints","feasible","minimax","elapsed_seconds","cache_hit","audio_file"]
    temporary=out/".trials.csv.tmp"
    with temporary.open("w",newline="",encoding="utf8") as handle:
        writer=csv.DictWriter(handle,fields,extrasaction="ignore");writer.writeheader()
        for row in records:
            value=dict(row);value["objectives"]=json.dumps(value.get("objectives"));value["constraints"]=json.dumps(value.get("constraints"));writer.writerow(value)
    os.replace(temporary,out/"trials.csv")


def rebuild_dashboard(env:dict[str,str],config:dict[str,Any]|None=None)->Path:
    out=optimization_dir(env);records=read_jsonl(out/"trials.jsonl");validation={};stability={}
    if (out/"validation-results.json").is_file(): validation=json.loads((out/"validation-results.json").read_text())
    if (out/"stability-results.json").is_file(): stability=json.loads((out/"stability-results.json").read_text())
    try:
        import plotly.graph_objects as go
        from plotly.offline import get_plotlyjs, plot
        chains=[x for x in records if x.get("study")=="chains" and len(x.get("objectives",[]))==3]
        effects=go.Figure()
        for study,label in zip(EFFECTS,("Centralização","Modulação","Formant")):
            rows=[x for x in records if x.get("study")==study and len(x.get("objectives",[]))==1]
            effects.add_trace(go.Scatter(x=[x["trial"] for x in rows],y=[x["objectives"][0] for x in rows],mode="lines+markers",name=label))
        fig=go.Figure()
        for index,name in enumerate(("C","M","F")):
            fig.add_trace(go.Scatter(x=[x["trial"] for x in chains],y=[x["objectives"][index] for x in chains],mode="lines+markers",name=name))
        minimax=[chain_record_score(x) for x in chains]
        fig.add_trace(go.Scatter(x=[x["trial"] for x in chains],y=minimax,mode="lines",name="minimax"))
        best=[]
        for value in minimax: best.append(min(value,best[-1]) if best else value)
        fig.add_trace(go.Scatter(x=[x["trial"] for x in chains],y=best,mode="lines",name="best-so-far"))
        front=pareto_front(chains)
        pareto=go.Figure(go.Scatter3d(x=[x["objectives"][0] for x in front],y=[x["objectives"][1] for x in front],z=[x["objectives"][2] for x in front],mode="markers",text=[str(x["trial"]) for x in front]))
        parallel=go.Figure(go.Parcoords(dimensions=[dict(label=n,values=[x["objectives"][i] for x in front]) for i,n in enumerate(("C","M","F"))]))
        checkpoint=go.Figure()
        for label,function in (("best",min),("median",statistics.median),("worst",max)):
            groups=sorted({x.get("checkpoint",1) for x in chains});checkpoint.add_trace(go.Scatter(x=groups,y=[function([chain_record_score(x) for x in chains if x.get("checkpoint",1)==group]) for group in groups],name=label))
        generations=go.Figure()
        for axis,label in enumerate(("C","M","F")): generations.add_trace(go.Box(x=[x.get("generation") for x in chains],y=[x["objectives"][axis] for x in chains],name=label))
        reference=(config or {}).get("optimization",{}).get("pareto",{}).get("reference_point",[2,2,2]);groups=sorted({x.get("checkpoint",1) for x in chains})
        hv=go.Figure(go.Scatter(x=groups,y=[hypervolume_3d([x["objectives"] for x in pareto_front([x for x in chains if x.get("checkpoint",1)<=group])],reference) for group in groups],mode="lines+markers",name="hypervolume"))
        engines={};orders={}
        for x in records:
            engines[x.get("engine","unknown")]=engines.get(x.get("engine","unknown"),0)+1
            order=x.get("parameters",{}).get("order")
            if order: orders[order]=orders.get(order,0)+1
        distribution=go.Figure();distribution.add_trace(go.Bar(x=list(engines),y=list(engines.values()),name="engines"));distribution.add_trace(go.Bar(x=list(orders),y=list(orders.values()),name="orders"))
        stable=go.Figure()
        for item in stability.get("candidates",[]): stable.add_trace(go.Box(y=[run["final_score"] for run in item["runs"] if run.get("success")],name=f'trial {item["training_trial"]}'))
        train_validation=go.Figure()
        vr=validation.get("results",[]);train_validation.add_trace(go.Scatter(x=[x.get("training_final_score") for x in vr],y=[x.get("validation_final_score") for x in vr],mode="markers",text=["overfit" if x.get("overfit") else "eligible" for x in vr],name="train vs validation"))
        figures=(("Estudos individuais",effects),("C, M, F, minimax e best-so-far",fig),("Resumo por checkpoint",checkpoint),("População NSGA-II por geração",generations),("Pareto 3D",pareto),("Coordenadas paralelas",parallel),("Hipervolume",hv),("Motores e ordens",distribution),("Estabilidade dos rerenders",stable),("Treino versus validação",train_validation))
        for _,value in figures:value.update_layout(template=None)
        atomic_text(out/"plotly.min.js",get_plotlyjs())
        body="".join(f"<h2>{title}</h2>"+plot(value,include_plotlyjs=False,output_type="div") for title,value in figures)
    except ImportError:
        body="<p>Plotly is not installed; persistent data remains available.</p>"
    links="".join(f'<li><a href="{x.get("audio_file","")}">{x.get("study")} #{x.get("trial")}</a></li>' for x in records if x.get("audio_file"))
    latest="".join(f"<tr><td>{x.get('study')}</td><td>{x.get('trial')}</td><td>{html_escape(str(x.get('objectives')))}</td><td>{html_escape(str(x.get('engine')))}</td></tr>" for x in records[-20:])
    html=("<!doctype html><html><head><meta charset=utf-8><meta http-equiv='refresh' content='30'><title>GLaDOS optimization</title><script src='plotly.min.js'></script>"
          "<style>body{font:15px system-ui;margin:2rem;max-width:1400px}.plotly-graph-div{height:480px!important}table{border-collapse:collapse}td,th{padding:.35rem .6rem;border:1px solid #ccc}pre{white-space:pre-wrap}</style></head><body>"
          f"<h1>GLaDOS optimization dashboard</h1><p><strong>{len(records)} trials persistidos.</strong> Atualização automática a cada 30 s; arquivo gerado em {utcnow()}. Os gráficos de cadeia aparecem depois dos três estudos individuais.</p>"
          "<h2>Trials recentes</h2><table><tr><th>estudo</th><th>trial</th><th>objetivos</th><th>motor</th></tr>"+latest+"</table>"+body+
          "<h2>Training versus validation</h2><pre>"+json.dumps(validation,indent=2,ensure_ascii=False)+"</pre><h2>WAVs</h2><ul>"+links+
          "</ul><p>lineage_available=false unless captured during child generation.</p></body></html>")
    target=out/"optimization-dashboard.html";atomic_text(target,html)
    # A running optimizer may still have an older dashboard function loaded;
    # this stable alias is not overwritten by that process.
    atomic_text(out/"optimization-dashboard-v2.html",html)
    return target


def html_escape(value:str)->str:
    import html
    return html.escape(value)


def status(env:dict[str,str],config_path:Path,*,lab:Any|None=None)->dict[str,Any]:
    config=load_config(config_path);out=optimization_dir(env);records=read_jsonl(out/"trials.jsonl");complete=completed_records(records);maximum=int(config["optimization"]["max_total_trials"])
    try: preflight=optimization_preflight(env,config,lab=lab)
    except Exception as exc: preflight={"ready":False,"blockers":[str(exc)]}
    best={}
    for name in STUDY_NAMES:
        rows=[x for x in complete if x.get("study")==name and x.get("feasible")]
        if rows: best[name]=min(rows,key=lambda x:max(x["objectives"]))["objectives"]
    return {"trials_used":len(complete),"failed_trials":len(records)-len(complete),"trials_remaining":max(0,maximum-len(complete)),"studies":{x:sum(r.get("study")==x for r in complete) for x in STUDY_NAMES},
            "checkpoint":len(json.loads((out/"checkpoint-history.json").read_text())) if (out/"checkpoint-history.json").is_file() else 0,
            "best_scores":best,"validation":preflight.get("validation"),"blocked":not preflight.get("ready",False),"blockers":preflight.get("blockers",[])}


def report(env:dict[str,str],config_path:Path)->dict[str,Any]:
    out=optimization_dir(env);history=json.loads((out/"checkpoint-history.json").read_text()) if (out/"checkpoint-history.json").is_file() else []
    pareto=json.loads((out/"pareto-front.json").read_text()) if (out/"pareto-front.json").is_file() else {}
    validation=json.loads((out/"validation-results.json").read_text()) if (out/"validation-results.json").is_file() else {}
    return {"output_dir":str(out),"checkpoints":history,"pareto":pareto,"validation":validation,"dashboard":str(out/"optimization-dashboard.html")}


def archive_invalid_optimization(env:dict[str,str],reason:str,*,reuse_cache:bool=True)->dict[str,Any]:
    """Archive a broken run without deleting history and seed a clean cache."""
    out=optimization_dir(env)
    if not (out/"studies.sqlite3").is_file():raise ValueError("no optimization study to archive")
    index=1
    while True:
        archived=out.with_name(f"optimization-invalid-checkpoint-{index:03d}")
        if not archived.exists():break
        index+=1
    out.rename(archived);(out/"cache").mkdir(parents=True)
    reused=0
    if reuse_cache:
        for source in (archived/"cache").glob("*.wav"):
            target=out/"cache"/source.name
            try:os.link(source,target)
            except OSError:shutil.copy2(source,target)
            reused+=1
    manifest={"archived_at":utcnow(),"reason":reason,"archived_directory":str(archived),
              "cache_files_reused":reused,"cache_reuse_enabled":reuse_cache,
              "history_preserved":True}
    write_json(archived/"invalid-checkpoint.json",manifest);write_json(out/"recovery-source.json",manifest)
    return manifest
