#!/usr/bin/env python3
"""Private architecture comparison; default is a local plan, never synthesis.

No model is imported, downloaded, started, stopped or trained by this program.
The run command measures an already approved isolated raw PCM16 HTTP runtime.
"""
from __future__ import annotations

import argparse
import array
import datetime as dt
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import statistics
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import wave

HERE = Path(__file__).resolve().parent
UTC = dt.timezone.utc


def utc_now():
    return dt.datetime.now(UTC)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def private_directory(path):
    original = Path(path).expanduser()
    if original.is_symlink():
        raise ValueError("private directory cannot be a symlink")
    directory = original.resolve()
    if any((p / ".git").exists() for p in (directory, *directory.parents)):
        raise ValueError("select a private directory outside any Git checkout")
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = directory.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError("private directory must be owned by this user with mode 0700")
    return directory


def write_private(path, data):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(data)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def wav_bytes(pcm, rate):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return output.getvalue()


def validate_endpoint(endpoint):
    parsed = urlsplit(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1"}:
        raise ValueError("endpoint must use literal HTTP loopback, without DNS")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("endpoint cannot contain credentials, query or fragment")
    if parsed.port is None or not 18000 <= parsed.port <= 65535:
        raise ValueError("use an isolated loopback port >=18000; production ports are forbidden")
    if not parsed.path.startswith("/"):
        raise ValueError("endpoint needs an explicit request path")
    return endpoint


def validate_config(config, execute=False):
    validate_endpoint(config["endpoint"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", config["candidate"]):
        raise ValueError("invalid candidate identifier")
    if config.get("wire_format") != "s16le-mono-raw":
        raise ValueError("only headerless little-endian mono PCM16 is supported")
    if config.get("sample_rate") not in {16000, 22050, 24000, 32000, 44100, 48000}:
        raise ValueError("declare the actual native sample rate")
    if not isinstance(config.get("payload"), dict) or not config.get("text_field"):
        raise ValueError("payload object and top-level text_field are required")
    if execute:
        if config.get("runtime_verified") is not True:
            raise ValueError("runtime contract must be verified before execution")
        if config.get("response_cache_disabled") is not True:
            raise ValueError("disable response caching before measuring")
        if config.get("native_rate_confirmed") is not True:
            raise ValueError("native rate must be confirmed; output resampling is not fidelity")
        revision = config.get("runtime_revision", "")
        if not re.fullmatch(r"(?:[a-f0-9]{40}|sha256:[a-f0-9]{64})", revision):
            raise ValueError("pin runtime_revision to a commit or image digest")
        weights = config.get("weights", [])
        if not weights or any(not re.fullmatch(r"[a-f0-9]{64}", w.get("sha256", "")) for w in weights):
            raise ValueError("record SHA256 of every actual checkpoint and decoder weight")
        if not config.get("allowed_compute_pids") or any(type(p) is not int or p <= 0 for p in config["allowed_compute_pids"]):
            raise ValueError("declare the isolated runtime's CUDA PIDs; production may not share the GPU")
    return config


def parse_utc(value):
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() != dt.timedelta(0):
        raise ValueError("window timestamps must include UTC")
    return result


def validate_window(window, candidate, now=None):
    now = now or utc_now()
    if window.get("owner_approved") is not True or not window.get("authorization_id"):
        raise ValueError("owner-approved operational window required")
    if window.get("candidate") != candidate:
        raise ValueError("window must authorize this exact candidate")
    start, end = parse_utc(window["starts_utc"]), parse_utc(window["ends_utc"])
    if not start <= now < end:
        raise ValueError("outside the authorized GPU window")
    return (end - now).total_seconds()


def pcm_metrics(pcm, chunks, rate, *, frame_ms=10, minimum_frames=2, peak_threshold=100, rms_threshold=30):
    """Digital onset proxy: two full consecutive 10 ms windows, not a header/click.

    Availability is when all samples needed to confirm onset arrived. Playback
    respects odd-byte boundaries and late chunks instead of adding an offset to
    the first network packet. This does not establish intelligibility by itself.
    """
    if len(pcm) % 2:
        raise ValueError("truncated PCM16 sample")
    samples = array.array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    frame = round(rate * frame_ms / 1000)
    consecutive, onset_start, onset_end = 0, None, None
    for start in range(0, len(samples) - frame + 1, frame):
        values = samples[start:start + frame]
        peak = max(abs(v) for v in values)
        rms = math.sqrt(sum(v * v for v in values) / len(values))
        consecutive = consecutive + 1 if peak > peak_threshold and rms > rms_threshold else 0
        if consecutive >= minimum_frames:
            onset_start, onset_end = start - (minimum_frames - 1) * frame, start + frame
            break
    received, playback_end = 0, None
    first_pcm, useful_available, playback_onset = None, None, None
    underruns, underrun_ms = 0, 0.0
    previous_ms = -1.0
    for chunk in chunks:
        arrival, size = chunk["ms"], chunk["bytes"]
        if arrival < previous_ms or size < 0:
            raise ValueError("invalid arrival schedule")
        previous_ms = arrival
        before = received // 2
        received += size
        after = received // 2
        if after == before:
            continue
        if first_pcm is None:
            first_pcm = arrival
        if playback_end is not None and arrival - playback_end > 10:
            underruns += 1
            underrun_ms += arrival - playback_end
        start = max(arrival, playback_end) if playback_end is not None else arrival
        if onset_start is not None and before <= onset_start < after:
            playback_onset = start + (onset_start - before) * 1000 / rate
        if onset_end is not None and useful_available is None and after >= onset_end:
            useful_available = arrival
        playback_end = start + (after - before) * 1000 / rate
    if received != len(pcm):
        raise ValueError("arrival schedule does not consume PCM")
    return {
        "first_pcm_ms": first_pcm,
        "first_useful_pcm_available_ms": useful_available,
        "onset_audio_offset_ms": onset_start * 1000 / rate if onset_start is not None else None,
        "estimated_first_audible_playback_ms": playback_onset,
        "audio_seconds": len(samples) / rate,
        "peak_pcm16": max((abs(v) for v in samples), default=0),
        "clipped_samples": sum(v in {-32768, 32767} for v in samples),
        "immediate_playback_underruns_over_10ms": underruns,
        "immediate_playback_underrun_ms": underrun_ms,
        "onset_proxy": {"frame_ms": frame_ms, "minimum_frames": minimum_frames,
                        "peak_over_pcm16": peak_threshold, "rms_over_pcm16": rms_threshold},
        "playback_assumption": "immediate FIFO playback; excludes device/client delay; digital proxy, not listening",
    }


def gpu_snapshot():
    def query(arguments):
        result = subprocess.run(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
                                text=True, capture_output=True, timeout=3, check=True)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    rows = query(["--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,driver_version"])
    if len(rows) != 1:
        raise ValueError("benchmark expects exactly one visible GPU")
    parts = [part.strip() for part in rows[0].split(",")]
    processes = []
    for row in query(["--query-compute-apps=pid,used_gpu_memory"]):
        pid, memory = [part.strip() for part in row.split(",")]
        processes.append({"pid": int(pid), "used_mib": float(memory)})
    return {"utc": utc_now().isoformat(), "index": parts[0], "uuid": parts[1], "name": parts[2],
            "total_mib": float(parts[3]), "used_mib": float(parts[4]), "free_mib": float(parts[5]),
            "driver": parts[6], "processes": processes}


def exclusive_gpu(snapshot, allowed_pids):
    actual = {p["pid"] for p in snapshot["processes"]}
    if not actual or actual != set(allowed_pids):
        raise ValueError("GPU process set differs from explicitly allowed candidate PIDs; no request sent")
    if "A2000" not in snapshot["name"] or not 11000 <= snapshot["total_mib"] <= 12288:
        raise ValueError("expected RTX A2000 12GB (approximately 11514 MiB visible)")


class GpuSampler:
    def __init__(self, window, config):
        self.window, self.config = window, config
        self.samples, self.error = [], None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                validate_window(self.window, self.config["candidate"])
                sample = gpu_snapshot()
                exclusive_gpu(sample, self.config["allowed_compute_pids"])
                self.samples.append(sample)
            except Exception as exc:
                self.error = type(exc).__name__ + ": " + str(exc)
                return
            self.stop_event.wait(.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=7)

    def result(self):
        return {"samples": self.samples, "sampling_error": self.error,
                "sampled_device_peak_used_mib": max((s["used_mib"] for s in self.samples), default=None),
                "sampled_process_peak_sum_mib": max((sum(p["used_mib"] for p in s["processes"]) for s in self.samples), default=None),
                "limitation": "NVML/nvidia-smi sampled device/process use; not allocator peaks or model-load peak; short spikes may be missed"}


def receive(config, window, payload, sampler):
    import httpx  # Optional dependency is only imported for explicitly executed inference.
    remaining = validate_window(window, config["candidate"])
    deadline = time.monotonic() + min(remaining, config.get("request_timeout_seconds", 120))
    raw, chunks = bytearray(), []
    timings = {}
    timeout = min(remaining, config.get("request_timeout_seconds", 120))
    with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as client:
        started = time.perf_counter()
        with client.stream("POST", config["endpoint"], json=payload,
                           headers={"Accept-Encoding": "identity"}) as response:
            timings["headers_ms"] = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("runtime did not return HTTP 200")
            content_type = response.headers.get("content-type", "").split(";")[0].strip()
            if content_type not in {"application/octet-stream", "audio/pcm", "audio/raw", "audio/L16"}:
                raise ValueError("expected raw PCM response, got " + content_type)
            if content_type == "audio/L16":
                raise ValueError("audio/L16 is big-endian; declare/export s16le as audio/pcm")
            if response.headers.get("content-encoding", "identity") != "identity":
                raise ValueError("compressed HTTP body cannot provide faithful PCM arrival timing")
            # A deadline timer closes a stalled response when the approved window
            # ends; repeated socket reads cannot silently extend the window.
            timer = threading.Timer(max(0, deadline - time.monotonic()), response.close)
            timer.daemon = True
            timer.start()
            try:
                for chunk in response.iter_raw():
                    validate_window(window, config["candidate"])
                    if time.monotonic() >= deadline:
                        raise TimeoutError("request deadline reached")
                    if sampler.error:
                        raise ValueError(sampler.error)
                    if not chunk:
                        continue
                    chunks.append({"ms": (time.perf_counter() - started) * 1000, "bytes": len(chunk)})
                    raw.extend(chunk)
                    if len(raw) > config["sample_rate"] * 2 * config.get("max_audio_seconds", 90):
                        raise ValueError("response exceeds declared maximum audio duration")
            finally:
                timer.cancel()
            validate_window(window, config["candidate"])
            if time.monotonic() >= deadline:
                raise TimeoutError("request deadline reached before valid EOF")
            timings["eof_ms"] = (time.perf_counter() - started) * 1000
    timings.update(first_transport_byte_ms=chunks[0]["ms"] if chunks else None,
                   chunks=chunks)
    if bytes(raw[:4]) in {b"RIFF", b"OggS", b"fLaC"}:
        raise ValueError("runtime sent container bytes instead of raw PCM")
    if not raw:
        raise ValueError("runtime returned no audio")
    timings.update(pcm_metrics(bytes(raw), chunks, config["sample_rate"]))
    timings["rtf_http_complete"] = timings["eof_ms"] / 1000 / timings["audio_seconds"]
    timings["delivery_before_eof_observed"] = (timings["first_useful_pcm_available_ms"] is not None
        and timings["first_useful_pcm_available_ms"] < timings["eof_ms"] - 1)
    timings["streaming_proof_limitation"] = "transport timing cannot prove incremental model decoding; inspect pinned runtime separately"
    return bytes(raw), timings


def run(config, window, directory, cases, repetitions):
    validate_config(config, execute=True)
    validate_window(window, config["candidate"])
    baseline = gpu_snapshot()
    exclusive_gpu(baseline, config["allowed_compute_pids"])
    write_private(directory / "provenance.json", json_bytes({"schema": "aegis.architecture_benchmark.v1",
        "created_utc": utc_now().isoformat(), "config": config, "window": window,
        "corpus_sha256": sha256(json_bytes(cases)), "gpu_before": baseline,
        "measurement": "already loaded isolated runtime; no warmup omitted; first request retained; no cold-load measurement",
        "client_sha256": sha256(Path(__file__).read_bytes())}))
    rows = []
    for repetition in range(1, repetitions + 1):
        for case in cases:
            row = {"candidate": config["candidate"], "case": case["id"], "repetition": repetition,
                   "text_sha256": sha256(case["text"].encode()), "text": case["text"],
                   "input_mode": case.get("input_mode", "verbatim"), "outcome": "failed"}
            name = f"{config['candidate']}-{case['id']}-{repetition:02d}"
            started = time.perf_counter()
            sampler = None
            try:
                validate_window(window, config["candidate"])
                exclusive_gpu(gpu_snapshot(), config["allowed_compute_pids"])
                payload = dict(config["payload"])
                payload[config["text_field"]] = case["text"]
                seeds = config.get("seeds", [40733133, 40733134, 40733135])
                seed = seeds[(repetition - 1) % len(seeds)]
                row["requested_seed"] = seed if config.get("seed_field") else None
                row["seed_control_supported"] = bool(config.get("seed_field"))
                if config.get("seed_field"):
                    payload[config["seed_field"]] = seed
                with GpuSampler(window, config) as sampler:
                    pcm, timings = receive(config, window, payload, sampler)
                row.update(timings)
                if sampler.error:
                    raise ValueError(sampler.error)
                content = wav_bytes(pcm, config["sample_rate"])
                write_private(directory / (name + ".wav"), content)
                row.update(audio_file=name + ".wav", audio_sha256=sha256(content),
                           outcome="ok" if timings["first_useful_pcm_available_ms"] is not None else "no_useful_pcm")
            except Exception as exc:
                row["error"] = type(exc).__name__ + ": " + str(exc)
                row["failed_after_ms"] = (time.perf_counter() - started) * 1000
            if sampler is not None:
                row["gpu"] = sampler.result()
            rows.append(row)
            write_private(directory / (name + ".json"), json_bytes(row))
            print(json.dumps({k: row.get(k) for k in ["candidate", "case", "repetition", "outcome", "first_useful_pcm_available_ms", "rtf_http_complete", "error"]}), flush=True)
            if "error" in row:
                # Preserve failed observation and stop; do not keep hammering a failed runtime.
                write_private(directory / "results.json", json_bytes(rows))
                raise RuntimeError("benchmark stopped after failure; observation retained")
    write_private(directory / "results.json", json_bytes(rows))
    return rows


def build_ab(manifest, directory, seed):
    """Offline only: pair transcripts are declarations, not ASR verification."""
    rng, key, review = random.Random(seed), [], []
    for number, pair in enumerate(manifest["pairs"], 1):
        if not isinstance(pair.get("case"), str) or not isinstance(pair.get("text"), str):
            raise ValueError("each A/B pair requires case and declared shared text")
        files = [("a", Path(pair["a"]).expanduser().resolve()), ("b", Path(pair["b"]).expanduser().resolve())]
        rng.shuffle(files)
        names, source_info = [], []
        for label, (source_label, path) in zip(("A", "B"), files):
            content = path.read_bytes()
            with wave.open(io.BytesIO(content), "rb") as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError("A/B expects mono PCM16 WAV")
                info = {"sample_rate": wav.getframerate(), "frames": wav.getnframes()}
            name = f"pair-{number:03d}-{label}.wav"
            write_private(directory / name, content)
            names.append(name)
            source_info.append({"label": label, "source_label": source_label, "source": str(path),
                                "sha256": sha256(content), **info})
        review.append({"pair": number, "case": pair["case"], "text": pair["text"],
                       "text_sha256": sha256(pair["text"].encode()),
                       "category": pair.get("category", "unspecified_control"),
                       "same_generation": pair.get("same_generation", False),
                       "transcript_evidence": "shared text declared by pair manifest; waveform content not verified",
                       "A": names[0], "B": names[1], "scores": {criterion: None for criterion in
                       ["intelligibility", "brazilian_pronunciation", "numbers_acronyms", "identity", "effects", "sibilants_attacks", "bandwidth", "artifacts", "preference"]}})
        key.append({"pair": number, "category": pair.get("category", "unspecified_control"),
                    "same_generation": pair.get("same_generation", False), "sources": source_info})
    write_private(directory / "answer-key.private.json", json_bytes({"seed": seed, "pairs": key}))
    write_private(directory / "review.json", json_bytes(review))
    cards = []
    for pair in review:
        cards.append(f'<article><h2>Par {pair["pair"]}: {html.escape(pair["case"])}</h2><p>{html.escape(pair["text"])}</p>'
                     + "".join(f'<p>{label} <audio controls preload="none" src="{pair[label]}"></audio></p>' for label in ("A", "B")) + '</article>')
    page = '<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>AegisVoice — A/B privado</title><body><h1>A/B privado</h1><p>Escute em volume fixo. Registre as notas em review.json; a chave fica separada. Taxas nativas preservadas, sem ganho, recorte ou reamostragem.</p>' + "".join(cards) + '</body></html>'
    write_private(directory / "index.html", page.encode())
    return review


def summarize(rows):
    """Offline descriptive statistics; failures stay in every denominator."""
    if not isinstance(rows, list) or any(not isinstance(row, dict) or
            not isinstance(row.get("candidate"), str) or not isinstance(row.get("case"), str) for row in rows):
        raise ValueError("results must be a list of observations with candidate and case")

    def group(selected, candidate, case=None):
        successful = [row for row in selected if row.get("outcome") == "ok" and not row.get("error")]
        failures = len(selected) - len(successful)
        outcomes = {}
        for row in selected:
            outcome = str(row.get("outcome", "missing"))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        result = {"candidate": candidate, "case": case,
                  "scope": "single_case" if case is not None else "mixed_cases_descriptive_only",
                  "observations": len(selected), "successes": len(successful), "failures": failures,
                  "failure_rate": failures / len(selected) if selected else None,
                  "outcomes": outcomes, "metrics": {}}
        for metric in ["first_useful_pcm_available_ms", "eof_ms", "rtf_http_complete"]:
            values = sorted(float(row[metric]) for row in successful
                            if type(row.get(metric)) in {int, float} and math.isfinite(row[metric]) and row[metric] >= 0)
            result["metrics"][metric] = {
                "successful_observations_with_valid_metric": len(values),
                "successful_observations_missing_or_invalid_metric": len(successful) - len(values),
                "p50": statistics.median(values) if values else None,
                "p95_nearest_rank": values[math.ceil(.95 * len(values)) - 1] if values else None,
                "max": values[-1] if values else None,
            }
        return result

    candidates = sorted({row["candidate"] for row in rows})
    by_case = []
    aggregates = []
    for candidate in candidates:
        selected = [row for row in rows if row["candidate"] == candidate]
        aggregates.append(group(selected, candidate))
        for case in sorted({row["case"] for row in selected}):
            by_case.append(group([row for row in selected if row["case"] == case], candidate, case))
    return {"schema": "aegis.architecture_summary.v1", "observations": len(rows),
            "policy": "all observations counted; no outlier filtering; metrics use outcome=ok without error only; failed timings are not substituted",
            "percentiles": "p50=median; p95=nearest rank ceil(0.95*n); small samples have poor tail precision and p95 often equals max",
            "aggregate_limitation": "mixed-case aggregates depend on corpus composition and are not an architecture ranking",
            "by_candidate_case": by_case, "by_candidate_mixed_cases": aggregates}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["plan", "run", "ab", "summarize"], default="plan")
    parser.add_argument("--config", type=Path, default=HERE / "candidate.example.json")
    parser.add_argument("--window", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--corpus", type=Path, default=HERE / "corpus.pt-BR.json")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--results", type=Path, help="existing run results.json for offline summarize")
    parser.add_argument("--blind-seed", type=int, default=40733133)
    args = parser.parse_args(argv)
    try:
        if args.command == "summarize":
            if not args.results:
                raise ValueError("summarize requires --results")
            print(json.dumps(summarize(load_json(args.results)), ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if args.command == "ab":
            if not args.pairs or not args.output:
                raise ValueError("ab requires --pairs and --output")
            build_ab(load_json(args.pairs), private_directory(args.output), args.blind_seed)
            print("Private offline A/B written; no model or network used.")
            return 0
        config = validate_config(load_json(args.config))
        corpus = load_json(args.corpus)
        if not 1 <= args.repetitions <= 20:
            raise ValueError("repetitions must be between 1 and 20")
        for case in corpus:
            if not re.fullmatch(r"[a-z0-9_-]+", case["id"]):
                raise ValueError("invalid corpus case identifier")
        if args.command == "plan":
            print(json.dumps({"mode": "plan_only", "candidate": config["candidate"],
                "requests_if_authorized": len(corpus) * args.repetitions,
                "runtime_verified": config.get("runtime_verified", False),
                "endpoint": config["endpoint"], "sample_rate": config["sample_rate"],
                "corpus_sha256": sha256(json_bytes(corpus)),
                "next": "verify runtime/pins, prepare private config and owner-approved UTC window; run only after production GPU has been released by the operator",
                "network_used": False, "gpu_used": False}, indent=2))
            return 0
        if not args.window or not args.output:
            raise ValueError("run requires --window and --output")
        validate_config(config, execute=True)
        window = load_json(args.window)
        validate_window(window, config["candidate"])
        run(config, window, private_directory(args.output), corpus, args.repetitions)
        return 0
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
