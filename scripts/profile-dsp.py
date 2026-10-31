#!/usr/bin/env python3
"""Profile WORLD/M02 on private raw captures without loading or calling Qwen.

The optional original pipeline is loaded from an operator-owned file. Each
comparison consumes the same PCM and transport chunks, and reports waveform
equality rather than inferring perceptual approval. No audio is logged.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pyworld

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))
import voice_pipeline


def load_original(path: Path):
    spec = importlib.util.spec_from_file_location("aegis_original_pipeline", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load original pipeline")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def profile_stages(module) -> Iterator[dict]:
    """Wrap stages in this isolated CLI process, restoring every callable."""
    stages: dict[str, list[dict[str, float]]] = {}
    originals = []
    targets = [(pyworld, name) for name in ("dio", "stonemask", "cheaptrick", "d4c", "synthesize")]
    targets += [(module, "_warp_formants"), (module, "_transform_f0"), (module.StatefulM02, "process")]
    for owner, name in targets:
        function = getattr(owner, name)
        originals.append((owner, name, function))
        label = "m02" if owner is module.StatefulM02 else name

        def measured(*args, _function=function, _label=label, **kwargs):
            wall, cpu = time.perf_counter(), time.thread_time()
            try:
                return _function(*args, **kwargs)
            finally:
                stages.setdefault(_label, []).append({
                    "wall_ms": (time.perf_counter() - wall) * 1000,
                    "thread_cpu_ms": (time.thread_time() - cpu) * 1000,
                })

        setattr(owner, name, measured)
    try:
        yield stages
    finally:
        for owner, name, function in originals:
            setattr(owner, name, function)


def render(module, raw: bytes, chunks: list[dict], *, pieces: bool, paced: bool, eof_ms: float):
    processor = module.AcousticStreamProcessor()
    assembler = module.Pcm16Reassembler()
    output, offset = [], 0
    first_ms = None
    started, cpu = time.perf_counter(), time.thread_time()

    def append(blocks):
        nonlocal first_ms
        for block in blocks:
            if block:
                if first_ms is None:
                    first_ms = (time.perf_counter() - started) * 1000
                output.append(block)

    with profile_stages(module) as stages:
        for chunk in chunks:
            if paced:
                delay = chunk["ms"] / 1000 - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
            samples = assembler.feed(raw[offset:offset + chunk["bytes"]])
            offset += chunk["bytes"]
            width = 6000 if pieces else max(1, len(samples))
            for start in range(0, len(samples), width):
                append(processor.push(samples[start:start + width]))
        assembler.finish()
        if paced:
            delay = eof_ms / 1000 - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)
        append(processor.finish())
    if offset != len(raw):
        raise ValueError("transport schedule does not account for raw bytes")
    return b"".join(output), {
        "first_pcm_ms": first_ms,
        "wall_ms": (time.perf_counter() - started) * 1000,
        "thread_cpu_ms": (time.thread_time() - cpu) * 1000,
        "stages": {name: {"calls": len(rows), "wall_ms": sum(r["wall_ms"] for r in rows),
                          "thread_cpu_ms": sum(r["thread_cpu_ms"] for r in rows), "samples": rows}
                   for name, rows in stages.items()},
    }


async def measure_event_loop(module, raw: bytes, chunks: list[dict], *, worker: bool, eof_ms: float):
    """Measure timer progress while this isolated process renders real PCM."""
    gaps = []
    done = asyncio.Event()

    async def timer():
        previous = time.perf_counter()
        while not done.is_set():
            await asyncio.sleep(.010)
            current = time.perf_counter()
            gaps.append((current - previous) * 1000)
            previous = current

    observer = asyncio.create_task(timer())
    await asyncio.sleep(.025)
    try:
        if worker:
            result = await asyncio.to_thread(render, module, raw, chunks, pieces=True, paced=False, eof_ms=eof_ms)
        else:
            result = render(module, raw, chunks, pieces=False, paced=False, eof_ms=eof_ms)
        await asyncio.sleep(.025)
    finally:
        done.set()
        await observer
    result[1]["event_loop_gap_max_ms"] = max(gaps, default=0)
    result[1]["event_loop_timer_observations"] = len(gaps)
    result[1]["worker"] = worker
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-pipeline", type=Path)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--paced", action="store_true")
    parser.add_argument("--event-loop", action="store_true",
                        help="compare original synchronous rendering with candidate worker timer progress")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    if args.event_loop and args.paced:
        parser.error("event-loop probe uses unpaced CPU replay; omit --paced")
    os.umask(0o077)
    files = sorted(args.input_dir.glob("*-raw.pcm"))
    if not files:
        parser.error("no raw captures found")
    original = load_original(args.baseline_pipeline) if args.baseline_pipeline else None
    rows = []
    for raw_path in files:
        name = raw_path.name.removesuffix("-raw.pcm")
        transport_path = args.input_dir / f"{name}-transport.json"
        if not transport_path.exists():
            transport_path = args.input_dir / f"{name}-trace.json"
        transport = json.loads(transport_path.read_text())
        raw = raw_path.read_bytes()
        for repeat in range(args.repetitions):
            baseline = None
            if original:
                if args.event_loop:
                    baseline, row = asyncio.run(measure_event_loop(
                        original, raw, transport["chunks"], worker=False, eof_ms=transport["eof_ms"]))
                else:
                    baseline, row = render(original, raw, transport["chunks"], pieces=False, paced=args.paced,
                                           eof_ms=transport["eof_ms"])
                row.update(case=name, repeat=repeat + 1, implementation="original")
                rows.append(row)
            if args.event_loop:
                pcm, row = asyncio.run(measure_event_loop(
                    voice_pipeline, raw, transport["chunks"], worker=True, eof_ms=transport["eof_ms"]))
            else:
                pcm, row = render(voice_pipeline, raw, transport["chunks"], pieces=True, paced=args.paced,
                                  eof_ms=transport["eof_ms"])
            row.update(case=name, repeat=repeat + 1, implementation="candidate_stable",
                       original_pcm_identical=pcm == baseline if baseline is not None else None)
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "stages"}), flush=True)
    args.output.write_text(json.dumps({"paced": args.paced, "event_loop_probe": args.event_loop,
                                      "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
