#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import http.client
import json
import math
import statistics
import struct
import time
import os
import wave
from pathlib import Path

BASELINE_SECONDS = 87.25
BASELINE_RTF = 5.2434
BASELINE_DURATION = 16.64


def request_payload(root: Path, streaming: bool) -> dict:
    cfg_path = Path(os.environ.get("VLLM_BENCHMARK_CONFIG", root / "experiments/vllm-omni/benchmark-config.json"))
    cfg = json.loads(cfg_path.read_text())
    audio = (root / cfg["reference_audio"]).read_bytes()
    ref_text = (root / cfg["reference_text"]).read_text().strip()
    return {
        "model": cfg["model"],
        "input": cfg["text"],
        "voice": cfg["voice"],
        "task_type": "Base",
        "ref_audio": "data:audio/wav;base64," + base64.b64encode(audio).decode(),
        "ref_text": ref_text,
        "x_vector_only_mode": False,
        "language": "Portuguese",
        "response_format": "pcm" if streaming else "wav",
        "stream": streaming,
        "stream_format": "audio",
        # Qwen3-TTS Base has a prompt-construction mode distinct from HTTP
        # streaming.  The official non-streaming Base example requires this.
        "non_streaming_mode": not streaming,
        "max_new_tokens": 1024,
        "seed": 41,
    }


def post_speech(root: Path, output: Path, streaming: bool) -> dict:
    body = json.dumps(request_payload(root, streaming), ensure_ascii=False).encode()
    api_port = int(os.environ.get("VLLM_API_PORT", "8004"))
    conn = http.client.HTTPConnection("127.0.0.1", api_port, timeout=600)
    start = time.perf_counter()
    conn.request("POST", "/v1/audio/speech", body, {"Content-Type": "application/json"})
    response = conn.getresponse()
    headers_at = time.perf_counter()
    if response.status != 200:
        error = response.read(8192).decode(errors="replace")
        raise RuntimeError(f"HTTP {response.status}: {error}")
    content_type = response.getheader("Content-Type", "")
    chunks: list[bytes] = []
    chunk_times: list[float] = []
    first_audio_at = None
    while True:
        chunk = response.read(4096)
        if not chunk:
            break
        now = time.perf_counter()
        if first_audio_at is None:
            first_audio_at = now
        chunks.append(chunk)
        chunk_times.append(now - start)
    total = time.perf_counter() - start
    conn.close()
    raw = b"".join(chunks)
    output.parent.mkdir(parents=True, exist_ok=True)
    if streaming:
        write_pcm_wav(output, raw, 24000)
    else:
        output.write_bytes(raw)
    audio_info = validate_wav(output)
    intervals = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
    bytes_per_chunk = [len(c) for c in chunks]
    chunk_audio_seconds = [len(c) / (24000 * 2) for c in chunks]
    sustainable = True
    if streaming and intervals:
        deficits = [interval - duration for interval, duration in zip(intervals, chunk_audio_seconds[:-1])]
        sustainable = sum(deficits) <= 0.25 and max(deficits, default=0) <= 0.25
    return {
        "total_seconds": total,
        "ttfb_seconds": headers_at - start,
        "ttfa_seconds": None if first_audio_at is None else first_audio_at - start,
        "content_type": content_type,
        "audio_duration_seconds": audio_info["duration_seconds"],
        "sample_rate": audio_info["sample_rate"],
        "rtf": total / audio_info["duration_seconds"],
        "size_bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "chunk_count": len(chunks),
        "bytes_per_chunk": bytes_per_chunk,
        "chunk_intervals_seconds": intervals,
        "first_chunk_audio_seconds": chunk_audio_seconds[0] if chunk_audio_seconds else 0,
        "sustainable_playback": sustainable,
    }


def write_pcm_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def validate_wav(path: Path) -> dict:
    data = path.read_bytes()
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")
    with wave.open(str(path), "rb") as wav:
        if wav.getcomptype() != "NONE" or wav.getnframes() <= 0:
            raise ValueError(f"{path} is not readable non-empty PCM")
        duration = wav.getnframes() / wav.getframerate()
        if not 2 <= duration <= 120:
            raise ValueError(f"{path} has implausible duration {duration}")
        return {
            "duration_seconds": duration,
            "sample_rate": wav.getframerate(),
            "frames": wav.getnframes(),
            "channels": wav.getnchannels(),
            "sample_width": wav.getsampwidth(),
        }


def summarize(rows: list[dict], streaming: bool) -> dict:
    result = {
        "runs": rows,
        "generation_seconds": [r["total_seconds"] for r in rows],
        "audio_duration_seconds": [r["audio_duration_seconds"] for r in rows],
        "rtf": [r["rtf"] for r in rows],
        "median_generation_seconds": statistics.median(r["total_seconds"] for r in rows),
        "median_rtf": statistics.median(r["rtf"] for r in rows),
    }
    if streaming:
        result.update(
            total_seconds=[r["total_seconds"] for r in rows],
            ttfa_seconds=[r["ttfa_seconds"] for r in rows],
            median_ttfa_seconds=statistics.median(r["ttfa_seconds"] for r in rows),
            sustainable_playback=all(r["sustainable_playback"] for r in rows),
        )
    return result


def write_summary(report_dir: Path, name: str, summary: dict) -> None:
    (report_dir / f"{name}-summary.json").write_text(json.dumps(summary, indent=2))
    lines = [f"# vLLM-Omni {name} benchmark", "", f"Median RTF: {summary['median_rtf']:.4f}"]
    lines.append(f"Median total time: {summary['median_generation_seconds']:.4f} s")
    if name == "stream":
        lines.append(f"Median TTFA: {summary['median_ttfa_seconds']:.4f} s")
        lines.append(f"Sustainable playback: {summary['sustainable_playback']}")
    lines += ["", "| Run | Total s | Audio s | RTF | SHA-256 |", "|---:|---:|---:|---:|---|"]
    for index, row in enumerate(summary["runs"], 1):
        lines.append(
            f"| {index:02d} | {row['total_seconds']:.4f} | "
            f"{row['audio_duration_seconds']:.4f} | {row['rtf']:.4f} | `{row['sha256']}` |"
        )
    (report_dir / f"{name}-summary.md").write_text("\n".join(lines) + "\n")


def run_all(root: Path) -> None:
    report_dir = Path(os.environ.get("VLLM_REPORT_DIR", root / "reports/vllm"))
    sample_root = Path(os.environ.get("VLLM_SAMPLES_DIR", root / "samples/benchmark-vllm"))
    non_dir = sample_root / "nonstream"
    stream_dir = sample_root / "stream"
    post_speech(root, non_dir / "warmup.wav", False)
    non_rows = [post_speech(root, non_dir / f"run-{i:02d}.wav", False) for i in range(1, 6)]
    write_summary(report_dir, "nonstream", summarize(non_rows, False))
    post_speech(root, stream_dir / "warmup.wav", True)
    stream_rows = [post_speech(root, stream_dir / f"run-{i:02d}.wav", True) for i in range(1, 6)]
    write_summary(report_dir, "stream", summarize(stream_rows, True))


def classify(rtf: float | None) -> str:
    if rtf is None:
        return "not_executed"
    if rtf <= 1:
        return "excellent"
    if rtf <= 2:
        return "very_good"
    if rtf <= 3:
        return "acceptable"
    if rtf < 4.5:
        return "partial_gain"
    return "failed"


def compare(reports: Path, samples: Path, golden: str) -> None:
    non_path = reports / "nonstream-summary.json"
    stream_path = reports / "stream-summary.json"
    non = json.loads(non_path.read_text()) if non_path.exists() else None
    stream = json.loads(stream_path.read_text()) if stream_path.exists() else None
    measured_non = non if non and len(non.get("runs", [])) == 5 and non.get("status") != "not_executed" else None
    measured_stream = (
        stream if stream and len(stream.get("runs", [])) == 5 and stream.get("status") != "not_executed" else None
    )
    compatibility = json.loads((reports / "compatibility.json").read_text())
    metrics_path = reports / "gpu-metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    median_time = measured_non["median_generation_seconds"] if measured_non else None
    median_rtf = measured_non["median_rtf"] if measured_non else None
    reduction_time = None if median_time is None else (BASELINE_SECONDS - median_time) / BASELINE_SECONDS * 100
    reduction_rtf = None if median_rtf is None else (BASELINE_RTF - median_rtf) / BASELINE_RTF * 100
    restored_text = (reports / "production-state-final.txt").read_text() if (reports / "production-state-final.txt").exists() else ""
    restored = "production_service_restored=true" in restored_text
    final = {
        "runtime": "vLLM-Omni",
        "vllm_version": "0.22.0",
        "vllm_omni_version": "0.22.0",
        "container_image": "vllm/vllm-omni:v0.22.0",
        "container_digest": "sha256:8b0cc5438eb27b34cdfd22011b735da6a94835a09e9c56ddf9e8cb300d679919",
        "driver_version": compatibility["host"]["driver_version"],
        "cuda_runtime": "13.0.2",
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "gpu": "NVIDIA RTX A2000 12 GB",
        "gpu_memory_utilization_by_stage": [],
        "seed_requested": 41,
        "seed_supported": True,
        "nonstream": non or {
            "generation_seconds": [], "audio_duration_seconds": [], "rtf": [],
            "median_generation_seconds": 0, "median_rtf": 0,
        },
        "stream": stream or {
            "ttfa_seconds": [], "total_seconds": [], "audio_duration_seconds": [], "rtf": [],
            "median_ttfa_seconds": 0, "median_rtf": 0, "sustainable_playback": False,
        },
        "gpu_metrics": metrics or {
            "sm_average_percent": 0,
            "power_average_watts": 0,
            "vram_peak_mib": 0,
            "clock_average_mhz": 0,
            "container_cpu_average_percent": 0,
        },
        "comparison": {
            "baseline_generation_seconds": BASELINE_SECONDS,
            "baseline_rtf": BASELINE_RTF,
            "time_reduction_percent": reduction_time,
            "rtf_reduction_percent": reduction_rtf,
        },
        "classification": classify(median_rtf),
        "production_service_restored": restored,
    }
    memory_path = reports / "memory-fit.json"
    if memory_path.exists():
        memory = json.loads(memory_path.read_text())
        if memory.get("selected_utilization") is not None:
            u = memory["selected_utilization"]
            final["gpu_memory_utilization_by_stage"] = [u, u]
    (reports / "comparison.json").write_text(json.dumps(final, indent=2))
    if measured_non and measured_stream:
        md = (
            "# Comparison with Transformers eager baseline\n\n"
            f"- Baseline: {BASELINE_SECONDS:.4f} s, RTF {BASELINE_RTF:.4f}, audio {BASELINE_DURATION:.4f} s\n"
            f"- vLLM-Omni median: {median_time:.4f} s, RTF {median_rtf:.4f}\n"
            f"- Time reduction: {reduction_time:.2f}%\n"
            f"- RTF reduction: {reduction_rtf:.2f}%\n"
            f"- Classification: **{classify(median_rtf)}**\n"
            f"- Streaming median TTFA: {measured_stream['median_ttfa_seconds']:.4f} s\n"
            f"- Streaming sustainable: {measured_stream['sustainable_playback']}\n"
            "- Auditory quality: pending human scoring in listening.html\n"
        )
    else:
        blocker = compatibility["probe"]["status"]
        memory = json.loads(memory_path.read_text()) if memory_path.exists() else {}
        md = (
            "# Comparison with Transformers eager baseline\n\n"
            f"Benchmark not executed. Compatibility status: **{blocker}**. "
            f"Memory-fit status: **{memory.get('status', 'not_attempted')}**.\n\n"
            "No RTF reduction is claimed.\n"
        )
    (reports / "comparison.md").write_text(md)
    if non is None:
        empty_non = {
            "status": "not_executed",
            "generation_seconds": [],
            "audio_duration_seconds": [],
            "rtf": [],
            "median_generation_seconds": 0,
            "median_rtf": 0,
            "runs": [],
        }
        non_path.write_text(json.dumps(empty_non, indent=2))
        (reports / "nonstream-summary.md").write_text(
            "# vLLM-Omni nonstream benchmark\n\nStatus: **not executed**. See compatibility.md and memory-fit.md.\n"
        )
    if stream is None:
        empty_stream = {
            "status": "not_executed",
            "ttfa_seconds": [],
            "total_seconds": [],
            "audio_duration_seconds": [],
            "rtf": [],
            "median_ttfa_seconds": 0,
            "median_rtf": 0,
            "sustainable_playback": False,
            "runs": [],
        }
        stream_path.write_text(json.dumps(empty_stream, indent=2))
        (reports / "stream-summary.md").write_text(
            "# vLLM-Omni streaming benchmark\n\nStatus: **not executed**. See compatibility.md and memory-fit.md.\n"
        )
    make_listening_html(reports / "listening.html", samples, golden)


def make_listening_html(output: Path, samples: Path, golden: str) -> None:
    entries = []
    if golden:
        entries.append(("golden eager 1.7B", Path(golden)))
    for mode in ("nonstream", "stream"):
        for i in range(1, 6):
            entries.append((f"vLLM {mode} run {i:02d}", samples / mode / f"run-{i:02d}.wav"))
    players = []
    for label, path in entries:
        if path.exists():
            src = path.resolve().as_uri()
            players.append(f"<section><h2>{html.escape(label)}</h2><audio controls src=\"{html.escape(src)}\"></audio></section>")
        else:
            players.append(f"<section><h2>{html.escape(label)}</h2><p>Arquivo indisponível: {html.escape(str(path))}</p></section>")
    criteria = "Identidade, sotaque brasileiro, pronúncia, naturalidade, ritmo, entonação, omissões, repetições, glitches, clipping, truncamento e volume."
    output.write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>Aegis TTS listening test</title>"
        "<style>body{font:16px sans-serif;max-width:900px;margin:2rem auto}section{padding:1rem;border-bottom:1px solid #ddd}audio{width:100%}</style>"
        f"<h1>Aegis TTS listening test</h1><p>{criteria}</p>{''.join(players)}"
    )


def parse_metrics(reports: Path) -> None:
    dmon = reports / "nvidia-dmon.txt"
    sm, power, clocks, memory = [], [], [], []
    if dmon.exists():
        for line in dmon.read_text(errors="replace").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # nvidia-smi dmon -s pucvmet -o DT columns are date, time, gpu,
            # power, temperatures, SM, memory %, ..., memory clock, graphics
            # clock, ..., framebuffer MiB. Keep positions explicit so date/time
            # do not shift the metric indexes.
            if len(parts) >= 17:
                try:
                    power.append(float(parts[3]))
                    sm.append(float(parts[6]))
                    clocks.append(float(parts[13]))
                    memory.append(float(parts[16]))
                except ValueError:
                    continue
    cpu = []
    stats = reports / "docker-stats.csv"
    if stats.exists():
        for line in stats.read_text().splitlines():
            fields = line.split(",")
            if len(fields) > 1:
                try:
                    cpu.append(float(fields[1].strip().rstrip("%")))
                except ValueError:
                    pass
    result = {
        "sm_average_percent": statistics.fmean(sm) if sm else 0,
        "power_average_watts": statistics.fmean(power) if power else 0,
        "vram_peak_mib": max(memory, default=0),
        "clock_average_mhz": statistics.fmean(clocks) if clocks else 0,
        "container_cpu_average_percent": statistics.fmean(cpu) if cpu else 0,
    }
    (reports / "gpu-metrics.json").write_text(json.dumps(result, indent=2))


def memory_report(source: Path, output: Path) -> None:
    data = json.loads(source.read_text())
    lines = ["# A2000 memory-fit attempt", "", f"Status: **{data['status']}**", ""]
    lines += ["| Stage 0 | Stage 1 | VRAM before MiB | VRAM after load MiB | Startup s | Result |", "|---:|---:|---:|---:|---:|---|"]
    for row in data["attempts"]:
        lines.append(
            f"| {row['stage_0_utilization']:.2f} | {row['stage_1_utilization']:.2f} | "
            f"{row['vram_before_mib']} | {row['vram_after_load_mib']} | "
            f"{row['startup_seconds']:.2f} | {'success' if row['success'] else 'failed'} |"
        )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--root", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    comp = sub.add_parser("compare")
    comp.add_argument("--reports", type=Path, required=True)
    comp.add_argument("--samples", type=Path, required=True)
    comp.add_argument("--golden", default="")
    metrics = sub.add_parser("metrics")
    metrics.add_argument("--reports", type=Path, required=True)
    mem = sub.add_parser("memory-report")
    mem.add_argument("--input", type=Path, required=True)
    mem.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_all(args.root)
    elif args.command == "probe":
        print(json.dumps(post_speech(args.root, args.output, False)))
    elif args.command == "compare":
        compare(args.reports, args.samples, args.golden)
    elif args.command == "metrics":
        parse_metrics(args.reports)
    elif args.command == "memory-report":
        memory_report(args.input, args.output)


if __name__ == "__main__":
    main()
