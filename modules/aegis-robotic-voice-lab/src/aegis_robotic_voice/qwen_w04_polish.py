"""YAML-driven, source-locked polishing of the selected Q00 -> W04 render.

This renderer is intentionally separate from :mod:`qwen_cold`: it neither
imports nor starts vLLM/Qwen.  It reads the already approved Q00 WAVs, derives
WORLD analysis features, and writes only a new polish result tree.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from contextlib import contextmanager

import numpy as np
import yaml
from scipy.signal import butter, sosfilt

from .analysis.world import analyze_world
from .audio import sha256, validate_audio
from .manifests import append_jsonl, read_jsonl
from .mastering.chain import master
from .synthesis.world_engine import synthesize_world
from .timing import Timer
from .transforms.aperiodicity import transform_aperiodicity
from .transforms.envelope import attenuate_sibilance_envelope, transform_envelope
from .transforms.prosody import transform_f0


MODULE_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = MODULE_ROOT / "config" / "corpus.yaml"
PROTECTED_COMPONENTS = ("protected", "AegisVoicev0.1")
W04_LOCK = {
    "modulation_factor": 0.0,
    "range_compression": 0.0,
    "quantization": "fixed_cents",
    "quantization_step_cents": 200,
    "transition_ms": 0,
    "formant_scale": 1.10,
    "aperiodicity_voiced_scale": 0.20,
}
VARIANT_LOCK = {"S01": (-2.0, 2.0), "S02": (-3.5, 3.5), "S03": (-5.0, 5.0)}
# 0 is fully periodic voiced excitation and 1 preserves WORLD's original
# voiced aperiodicity.  The broad range is reserved for explicit diagnostics.
VOICED_APERIODICITY_RANGE = (0.0, 1.0)


def _under_protected(path: Path) -> bool:
    parts = path.resolve(strict=False).parts
    return any(parts[index : index + 2] == PROTECTED_COMPONENTS for index in range(len(parts) - 1))


def _safe_path(value: str | Path, *, label: str, output: bool = False) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{label} must be a non-empty path")
    raw = Path(value)
    if "~" in raw.parts:
        raise ValueError(f"{label} must not use home expansion")
    path = (raw if raw.is_absolute() else MODULE_ROOT / raw).resolve(strict=False)
    if _under_protected(path):
        raise ValueError(f"{label} must never point to protected/AegisVoicev0.1")
    if output:
        try:
            path.relative_to(MODULE_ROOT)
        except ValueError as exc:
            raise ValueError(f"{label} must stay inside the robotic-voice module") from exc
    return path


def _config_fingerprint(document: dict[str, Any]) -> str:
    stable = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _corpus() -> list[dict[str, str]]:
    data = yaml.safe_load(CORPUS_PATH.read_text(encoding="utf-8"))
    sentences = data.get("sentences", [])
    if len(sentences) != 15 or len({item.get("id") for item in sentences}) != 15:
        raise ValueError("the polish renderer requires the fixed 15-sentence corpus")
    return sentences


def _selected_sentences(document: dict[str, Any]) -> list[dict[str, str]]:
    """Return the YAML-selected corpus subset in its declared listening order."""
    corpus = _corpus()
    by_id = {sentence["id"]: sentence for sentence in corpus}
    selection = document.get("selection", {})
    if selection is None:
        selection = {}
    if not isinstance(selection, dict):
        raise ValueError("selection must be a mapping")
    sentence_ids = selection.get("sentence_ids", list(by_id))
    if not isinstance(sentence_ids, list) or not sentence_ids or any(not isinstance(item, str) for item in sentence_ids):
        raise ValueError("selection.sentence_ids must be a non-empty sentence-id list")
    if len(sentence_ids) != len(set(sentence_ids)) or any(item not in by_id for item in sentence_ids):
        raise ValueError("selection.sentence_ids contains an unknown or repeated corpus id")
    return [by_id[sentence_id] for sentence_id in sentence_ids]


def _load_document(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve(strict=False)
    if _under_protected(path):
        raise ValueError("configuration must never be read from protected/AegisVoicev0.1")
    if not path.is_file():
        raise ValueError(f"polish config is unavailable: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("polish config must be a mapping")
    return document, path


def _required_mapping(document: dict[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _validate_document(document: dict[str, Any]) -> dict[str, Any]:
    source = _required_mapping(document, "source")
    baseline = _required_mapping(document, "baseline")
    output = _required_mapping(document, "output")
    source_dir = _safe_path(source.get("directory"), label="source.directory")
    baseline_dir = _safe_path(baseline.get("directory"), label="baseline.directory")
    output_dir = _safe_path(output.get("directory"), label="output.directory", output=True)
    if source.get("profile") != "Q00" or source_dir.as_posix().rstrip("/").endswith("results/qwen-cold/sources/Q00") is False:
        raise ValueError("polish input is fixed to results/qwen-cold/sources/Q00 (profile Q00)")
    baseline_stage = baseline.get("stage", "mastered")
    if baseline_stage not in {"raw", "mastered"}:
        raise ValueError("baseline.stage must be raw or mastered")
    if not baseline_dir.as_posix().rstrip("/").endswith(f"results/qwen-cold/{baseline_stage}/Q00/W04"):
        raise ValueError(f"baseline must be the external Q00 -> W04 {baseline_stage} directory")
    hashes = source.get("expected_sha256")
    sentence_ids = [sentence["id"] for sentence in _corpus()]
    if not isinstance(hashes, dict) or set(hashes) != set(sentence_ids):
        raise ValueError("source.expected_sha256 must pin all 15 Q00 sentence files")
    if any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        raise ValueError("source.expected_sha256 contains an invalid SHA-256")
    world = _required_mapping(document, "world_w04")
    if any(world.get(key) != value for key, value in W04_LOCK.items()):
        raise ValueError("world_w04 must preserve the selected W04 identity exactly")
    if world.get("strict_resynthesis") is not True or world.get("source_waveform_is_audible") is not False:
        raise ValueError("polish must use strict WORLD resynthesis with no dry audio")
    mastering = _required_mapping(document, "mastering")
    if mastering.get("high_pass_hz") != 60 or mastering.get("limiter_true_peak_db") != -1.0:
        raise ValueError("mastering must preserve W04 high-pass and limiter settings")
    variants = _required_mapping(document, "variants")
    if not variants:
        raise ValueError("variants must contain at least one ordered candidate")
    legacy_v02 = document.get("version") == "qwen-w04-sibilance-polish-v0.2"
    if legacy_v02 and list(variants) != ["S01", "S02", "S03"]:
        raise ValueError("v0.2 variants must be rendered sequentially as S01, S02, S03")
    for code, variant in variants.items():
        if not isinstance(variant, dict):
            raise ValueError(f"{code} must be a mapping")
        spectral = _required_mapping(variant, "spectral_sibilance")
        deesser = _required_mapping(variant, "deesser")
        if spectral.get("cutoff_hz", 0) <= 0 or spectral.get("shelf_db", 1) > 0:
            raise ValueError(f"{code} has an invalid spectral sibilance setting")
        if spectral.get("transition_frames", 0) < 0:
            raise ValueError(f"{code} has an invalid transition-frame setting")
        if spectral.get("shelf_width_hz") is not None and spectral["shelf_width_hz"] <= 0:
            raise ValueError(f"{code} has an invalid sibilance shelf width")
        if deesser.get("enabled") not in {True, False}:
            raise ValueError(f"{code} must explicitly enable or disable its de-esser")
        if deesser["enabled"]:
            if deesser.get("max_reduction_db", 0) <= 0 or deesser.get("center_hz", 0) <= 0:
                raise ValueError(f"{code} has an invalid dynamic de-esser setting")
            if deesser.get("attack_ms", 0) <= 0 or deesser.get("release_ms", 0) <= 0:
                raise ValueError("de-esser must preserve positive attack/release")
        voiced_aperiodicity = variant.get("voiced_aperiodicity_scale")
        if voiced_aperiodicity is not None and (
            not isinstance(voiced_aperiodicity, (int, float))
            or not VOICED_APERIODICITY_RANGE[0] <= voiced_aperiodicity <= VOICED_APERIODICITY_RANGE[1]
        ):
            raise ValueError(f"{code} voiced_aperiodicity_scale must stay within the WORLD [0, 1] range")
        if legacy_v02:
            shelf_db, max_reduction_db = VARIANT_LOCK[code]
            if spectral.get("cutoff_hz") != 5200 or spectral.get("shelf_db") != shelf_db:
                raise ValueError(f"{code} has an unapproved v0.2 spectral sibilance setting")
            if deesser.get("enabled") is not True or deesser.get("max_reduction_db") != max_reduction_db or deesser.get("center_hz") != 6500:
                raise ValueError(f"{code} has an unapproved v0.2 dynamic de-esser setting")
    selected = _selected_sentences(document)
    return {"source_dir": source_dir, "baseline_dir": baseline_dir, "baseline_stage": baseline_stage, "output_dir": output_dir, "source_hashes": hashes, "sentences": selected}


def _validate_inputs(layout: dict[str, Any]) -> None:
    for sentence in layout["sentences"]:
        sentence_id = sentence["id"]
        source = layout["source_dir"] / f"{sentence_id}.wav"
        baseline = layout["baseline_dir"] / f"{sentence_id}.wav"
        if not source.is_file():
            raise ValueError(f"missing frozen Q00 source: {source}")
        if sha256(source) != layout["source_hashes"][sentence_id]:
            raise ValueError(f"Q00 source hash mismatch: {source}")
        if not baseline.is_file():
            raise ValueError(f"missing external W04 listening baseline: {baseline}")


def _latest(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        latest[(record.get("variant"), record.get("sentence_id"))] = record
    return latest


def _cache_hit(record: dict[str, Any] | None, raw_path: Path, mastered_path: Path, source_hash: str, fingerprint: str) -> bool:
    if not raw_path.exists() and not mastered_path.exists():
        return False
    if record is None or record.get("status") != "complete":
        raise RuntimeError(f"refusing to overwrite an untracked polish result: {mastered_path}")
    if not raw_path.is_file() or not mastered_path.is_file():
        raise RuntimeError(f"refusing partial polish cache: {mastered_path}")
    valid = (
        record.get("source_sha256") == source_hash
        and record.get("config_sha256") == fingerprint
        and record.get("raw_sha256") == sha256(raw_path)
        and record.get("mastered_sha256") == sha256(mastered_path)
    )
    if not valid:
        raise RuntimeError(f"refusing to overwrite a polish result with a different hash: {mastered_path}")
    return True


def _band_energy(wave: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    high = min(float(high_hz), sample_rate / 2.0 - 1.0)
    if not (0 < low_hz < high):
        return 0.0
    filtered = sosfilt(butter(2, [low_hz, high], fs=sample_rate, btype="bandpass", output="sos"), wave)
    return float(np.mean(filtered * filtered))


def _diagnostics(wave: np.ndarray, sample_rate: int) -> dict[str, float]:
    return {
        "high_band_5200_10000": _band_energy(wave, sample_rate, 5200, 10000),
        "detection_band_5000_8000": _band_energy(wave, sample_rate, 5000, 8000),
        "intelligibility_band_300_4000": _band_energy(wave, sample_rate, 300, 4000),
    }


def _write_reports(output_dir: Path, baseline_dir: Path, baseline_stage: str, records: list[dict[str, Any]], document: dict[str, Any], sentences: list[dict[str, str]]) -> None:
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    latest = _latest(records)
    report = document.get("reporting", {})
    if not isinstance(report, dict):
        raise ValueError("reporting must be a mapping")
    listening_stage = report.get("listening_stage")
    if listening_stage is None:
        listening_stage = "both" if bool(report.get("include_raw_in_listening", True)) else "mastered"
    if listening_stage not in {"raw", "mastered", "both"}:
        raise ValueError("reporting.listening_stage must be raw, mastered, or both")
    title = str(report.get("title", "Q00 → W04 — sibilance comparison"))
    lines = [f"<meta charset=utf-8><title>{html.escape(title)}</title>", f"<h1>{html.escape(title)}</h1>", "<p>Reference is retained for listening only; no automatic winner is selected.</p>"]
    technical_lines = [f"<meta charset=utf-8><title>{html.escape(title)} — technical stages</title>", f"<h1>{html.escape(title)} — raw vs. mastered</h1>"]
    for sentence in sentences:
        sentence_id = sentence["id"]
        lines.append(f"<h2>{html.escape(sentence_id)}</h2><p>{html.escape(sentence['text'])}</p>")
        technical_lines.append(f"<h2>{html.escape(sentence_id)}</h2><p>{html.escape(sentence['text'])}</p>")
        baseline_label = f"Reference Q00 → W04 {baseline_stage}"
        entries = [(baseline_label, baseline_dir / f"{sentence_id}.wav")]
        technical_entries = [(baseline_label, baseline_dir / f"{sentence_id}.wav")]
        for code in document["variants"]:
            record = latest.get((code, sentence_id))
            if record and record.get("status") == "complete":
                if listening_stage in {"raw", "both"}:
                    entries.append((f"{code} raw", Path(record["output_raw_path"])))
                if listening_stage in {"mastered", "both"}:
                    entries.append((f"{code} mastered", Path(record["output_mastered_path"])))
                technical_entries.extend([(f"{code} raw", Path(record["output_raw_path"])), (f"{code} mastered", Path(record["output_mastered_path"]))])
        for label, target in entries:
            relative = os.path.relpath(target, report_dir).replace(os.sep, "/")
            lines.append(f'<p>{html.escape(label)}<br><audio controls src="{html.escape(relative)}"></audio></p>')
        for label, target in technical_entries:
            relative = os.path.relpath(target, report_dir).replace(os.sep, "/")
            technical_lines.append(f'<p>{html.escape(label)}<br><audio controls src="{html.escape(relative)}"></audio></p>')
    (report_dir / "listening.html").write_text("\n".join(lines), encoding="utf-8")
    (report_dir / "technical-stages.html").write_text("\n".join(technical_lines), encoding="utf-8")
    (report_dir / "manifest.json").write_text(json.dumps({"config": document, "candidates": records}, ensure_ascii=False, indent=2), encoding="utf-8")


def _relative_db(value: float, reference: float) -> float:
    return float(10.0 * np.log10(max(value, 1e-20) / max(reference, 1e-20)))


def _world_parameters(document: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    """Keep W04 fixed except for an explicit, bounded voiced-noise experiment."""
    parameters = dict(document["world_w04"])
    override = variant.get("voiced_aperiodicity_scale")
    if override is not None:
        parameters["aperiodicity_voiced_scale"] = override
    return parameters


def _write_metrics(output_dir: Path, baseline_dir: Path, records: list[dict[str, Any]], document: dict[str, Any], sentences: list[dict[str, str]]) -> None:
    latest = [record for record in _latest(records).values() if record.get("status") == "complete"]
    metrics = document.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a mapping")
    diagnostics = set(metrics.get("sentence_ids", [sentence["id"] for sentence in sentences]))
    selected_ids = {sentence["id"] for sentence in sentences}
    if not diagnostics <= selected_ids:
        raise ValueError("metrics.sentence_ids must be selected for rendering")
    import soundfile as sf
    reference_diagnostics: dict[str, dict[str, float]] = {}
    for sentence_id in diagnostics:
        reference_wave, reference_rate = sf.read(baseline_dir / f"{sentence_id}.wav", always_2d=False)
        reference_diagnostics[sentence_id] = _diagnostics(np.asarray(reference_wave, dtype=float), reference_rate)
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for record in latest:
        if record["sentence_id"] in diagnostics:
            by_variant.setdefault(record["variant"], []).append(record)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for code in document["variants"]:
        rows = by_variant.get(code, [])
        summary[code] = {
            "diagnostic_sentences": [row["sentence_id"] for row in rows],
            "reference": {row["sentence_id"]: reference_diagnostics[row["sentence_id"]] for row in rows},
            "raw": {row["sentence_id"]: row["raw_diagnostics"] for row in rows},
            "mastered": {row["sentence_id"]: row["mastered_diagnostics"] for row in rows},
            "mastered_relative_to_reference_db": {
                row["sentence_id"]: {
                    key: _relative_db(value, reference_diagnostics[row["sentence_id"]][key])
                    for key, value in row["mastered_diagnostics"].items()
                }
                for row in rows
            },
            "note": "Comparative diagnostic only; this report does not select a winner.",
        }
        (metrics_dir / f"{code}.json").write_text(json.dumps(summary[code], ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


@contextmanager
def _render_lock(output_dir: Path):
    """Prevent two invocations from racing to write the immutable cache."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (output_dir / ".render.lock").open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another Q00/W04 polish render is already running") from exc
        yield
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def _run_locked(document: dict[str, Any], layout: dict[str, Any]) -> Path:
    output_dir = layout["output_dir"]
    manifest_path = output_dir / "candidates.jsonl"
    records = read_jsonl(manifest_path)
    latest = _latest(records)
    fingerprint = _config_fingerprint(document)
    import soundfile as sf

    for code in document["variants"]:
        variant = document["variants"][code]
        for sentence in layout["sentences"]:
            sentence_id = sentence["id"]
            source_path = layout["source_dir"] / f"{sentence_id}.wav"
            source_hash = layout["source_hashes"][sentence_id]
            raw_path = output_dir / "raw" / code / f"{sentence_id}.wav"
            mastered_path = output_dir / "mastered" / code / f"{sentence_id}.wav"
            if _cache_hit(latest.get((code, sentence_id)), raw_path, mastered_path, source_hash, fingerprint):
                continue
            wave, sample_rate = sf.read(source_path, always_2d=False)
            wave = np.asarray(wave, dtype=float)
            validate_audio(wave, sample_rate)
            started = perf_counter()
            with Timer() as analysis_timer:
                representation = analyze_world(wave, sample_rate, document.get("frame_period_ms", 5.0))
            with Timer() as transform_timer:
                world_parameters = _world_parameters(document, variant)
                f0 = transform_f0(representation.f0_hz, representation.voiced, **world_parameters)
                envelope = transform_envelope(representation.spectral_envelope, **world_parameters)
                envelope = attenuate_sibilance_envelope(envelope, sample_rate, representation.voiced, **variant["spectral_sibilance"])
                aperiodicity = transform_aperiodicity(representation.aperiodicity, representation.voiced, **world_parameters)
            with Timer() as synthesis_timer:
                raw = synthesize_world(representation, f0_hz=f0, spectral_envelope=envelope, aperiodicity=aperiodicity, duration_samples=len(wave))
            with Timer() as mastering_timer:
                final = master(raw, sample_rate, **document["mastering"], **variant["deesser"])
            validate_audio(raw, sample_rate, len(wave))
            validate_audio(final, sample_rate, len(wave))
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            mastered_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(raw_path, raw, sample_rate, subtype="FLOAT")
            sf.write(mastered_path, final, sample_rate, subtype="FLOAT")
            record = {
                "status": "complete", "source_profile": "Q00", "engine": "world", "preset": "W04", "variant": code,
                "sentence_id": sentence_id, "text": sentence["text"], "source_path": str(source_path), "source_sha256": source_hash,
                "output_raw_path": str(raw_path), "output_mastered_path": str(mastered_path), "raw_sha256": sha256(raw_path), "mastered_sha256": sha256(mastered_path),
                "config_sha256": fingerprint, "sample_rate": sample_rate, "input_samples": len(wave), "output_samples": len(final),
                "duration_seconds": len(final) / sample_rate, "parameters": {"world_w04": world_parameters, "spectral_sibilance": variant["spectral_sibilance"], "deesser": variant["deesser"], "mastering": document["mastering"]},
                "analysis_seconds": analysis_timer.seconds, "transform_seconds": transform_timer.seconds, "synthesis_seconds": synthesis_timer.seconds, "mastering_seconds": mastering_timer.seconds, "total_dsp_seconds": perf_counter() - started,
                "raw_diagnostics": _diagnostics(raw, sample_rate), "mastered_diagnostics": _diagnostics(final, sample_rate),
            }
            append_jsonl(manifest_path, record)
            records.append(record)
            latest[(code, sentence_id)] = record
    _write_metrics(output_dir, layout["baseline_dir"], records, document, layout["sentences"])
    _write_reports(output_dir, layout["baseline_dir"], layout["baseline_stage"], records, document, layout["sentences"])
    return output_dir / "reports" / "listening.html"


def run(config_path: str | Path) -> Path:
    """Render the YAML-declared variants in order and return the listening page."""
    document, _ = _load_document(config_path)
    layout = _validate_document(document)
    _validate_inputs(layout)
    with _render_lock(layout["output_dir"]):
        return _run_locked(document, layout)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render a source-locked Q00/W04 polish catalog.")
    parser.add_argument("--config", required=True, help="YAML polish catalog")
    args = parser.parse_args(argv)
    print(run(args.config))


if __name__ == "__main__":
    main()
