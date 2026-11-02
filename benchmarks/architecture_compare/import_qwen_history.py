"""Recompute historical Qwen metrics and package existing WAVs; never infer."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import wave


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def summarize(path: Path) -> dict:
    data = path.read_bytes()
    original = json.loads(data)
    cohorts = {}
    for name in ("final-idle", "final-heldout", "final-reserved"):
        source = original["client_cohorts"][name]
        rows = []
        for row in source:
            if "error" in row or row["audio_ms"] <= 0:
                raise ValueError("historical final cohort contains a failed request")
            rows.append({key: row[key] for key in (
                "sample", "case", "repetition", "first_pcm_ms", "first_voice_available_ms",
                "estimated_first_audible_playback_ms", "eof_ms", "audio_ms",
                "immediate_playback_underruns", "immediate_playback_underrun_ms")})
            rows[-1]["rtf_eof_over_audio"] = row["eof_ms"] / row["audio_ms"]
        metrics = {}
        for key in ("first_pcm_ms", "first_voice_available_ms", "estimated_first_audible_playback_ms",
                    "eof_ms", "rtf_eof_over_audio"):
            values = [row[key] for row in rows]
            metrics[key] = {"min": min(values), "median": statistics.median(values),
                            "max": max(values), "mean": statistics.mean(values)}
        cohorts[name] = {"count": len(rows), "metrics": metrics, "rows": rows,
                         "weighted_rtf": sum(row["eof_ms"] for row in rows) /
                                         sum(row["audio_ms"] for row in rows)}
    return {
        "schema": "aegis.architecture_comparison.historical_baseline.v1",
        "measured_date_utc": "2026-11-01", "recomputed_from": str(path),
        "source_sha256": sha256(data),
        "runtime_source_commit": "1749f5fb29320dced0a16add618989e36954ee7b",
        "sample_rate": 24000, "channels": 1, "pcm_bits": 16,
        "scope": "Qwen 0.6B + production gateway WORLD W04/P07/M02; not raw model",
        "onset_definition": "historical single sample abs(PCM16)>100; digital proxy, not listening",
        "limitations": ["No new inference", "No candidate architecture evaluated",
                         "Historical onset differs from the new windowed onset metric",
                         "Resident GPU snapshot is not peak per-request VRAM"],
        "cohorts": cohorts,
    }


def private_directory(path: Path) -> Path:
    path = path.expanduser().resolve()
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("private output must be outside Git")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ValueError("private directory must be owned by current user and mode 0700")
    return path


def private_write(path: Path, content: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(content)


def package_existing(source: Path, destination: Path) -> dict:
    """Two honest control pairs per text: latency revision and raw->DSP replay."""
    source = source.resolve()
    destination = private_directory(destination)
    original = json.loads((source / "review/AB-index.json").read_text())
    rows = []
    for row in original:
        case = row["case"]
        pairs = [("qwen-runtime-revisions", source / "review" / row["A"],
                  source / "review" / row["B"], False),
                 ("qwen-raw-versus-dsp", source / "raw22" / f"{case}-01-raw.pcm",
                  source / "raw22" / f"{case}-01-stable-onset.wav", True)]
        for category, a, b, same_generation in pairs:
            entry = {"id": f"{category}-{case}", "text": row["text"], "category": category,
                     "same_generation": same_generation,
                     "review_status": "pending_owner_listening",
                     "not_an_architecture_AB": True}
            for label, source_path in (("A", a), ("B", b)):
                content = source_path.read_bytes()
                source_hash = sha256(content)
                if source_path.suffix == ".pcm":
                    if not content or len(content) % 2:
                        raise ValueError("invalid historical PCM16")
                    buffer = io.BytesIO()
                    with wave.open(buffer, "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(24000)
                        wav.writeframes(content)
                    content = buffer.getvalue()
                with wave.open(io.BytesIO(content), "rb") as wav:
                    if wav.getparams()[:3] != (1, 2, 24000) or wav.getnframes() <= 0:
                        raise ValueError("unexpected historical WAV format")
                    duration = wav.getnframes() / wav.getframerate()
                name = f"{entry['id']}-{label}.wav"
                private_write(destination / name, content)
                entry[label] = name
                entry[f"{label}_source_sha256"] = source_hash
                entry[f"{label}_sha256"] = sha256(content)
                entry[f"{label}_seconds"] = duration
            rows.append(entry)
    manifest = {"schema": "aegis.architecture_comparison.historical_controls.v1",
                "scope": "Existing Qwen controls only; no GPT-SoVITS/VoxCPM2 samples exist",
                "new_inference": False, "sample_rate": 24000, "pairs": rows}
    private_write(destination / "controls.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path,
                        default=Path("docs/v0.8.0/host-benchmark-20261101.json"))
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--existing-captures", type=Path)
    parser.add_argument("--private-output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.history)
    if args.summary_output:
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if bool(args.existing_captures) != bool(args.private_output):
        parser.error("--existing-captures and --private-output must be given together")
    if args.existing_captures:
        result = package_existing(args.existing_captures, args.private_output)
        print(json.dumps({"private_pairs": len(result["pairs"]), "new_inference": False}))


if __name__ == "__main__":
    main()
