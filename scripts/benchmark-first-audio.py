#!/usr/bin/env python3
"""Small opt-in benchmark. Run inside the existing TTS container to capture
synthetic raw PCM once, then replay identical PCM through both DSP variants.
No second model process, protected-asset edits, permanent traffic or API routes.
All audio stays in the explicitly selected private directory outside Git.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gateway'))
from voice_pipeline import AcousticStreamProcessor, Pcm16Reassembler

CORPUS = {
    'time': 'São nove horas e quinze minutos.',
    'math': 'Dezessete mais vinte e cinco são quarenta e dois.',
    'long': 'A água ferve a cem graus Celsius ao nível do mar. Em lugares mais altos, ela ferve a uma temperatura menor, por causa da pressão atmosférica.',
}


def capture(directory: Path) -> None:
    import default_reference_gateway as gateway
    gateway._verify_protected_assets()
    gateway.app.state.default_audio, gateway.app.state.default_text = gateway._load_default_reference()
    with httpx.Client(base_url=gateway.BACKEND_URL, timeout=100) as client:
        for name, text in CORPUS.items():
            payload = gateway._apply_defaults({'input': text, 'stream': True, 'response_format': 'pcm'})
            started = time.perf_counter()
            chunks, raw = [], bytearray()
            with client.stream('POST', '/v1/audio/speech', json=payload) as response:
                response.raise_for_status()
                headers_ms = (time.perf_counter() - started) * 1000
                for chunk in response.iter_bytes():
                    chunks.append({'ms': (time.perf_counter() - started) * 1000, 'bytes': len(chunk)})
                    raw.extend(chunk)
            (directory / f'{name}-raw.pcm').write_bytes(raw)
            (directory / f'{name}-transport.json').write_text(json.dumps({'headers_ms': headers_ms, 'eof_ms': (time.perf_counter() - started) * 1000, 'chunks': chunks}, indent=2))
            print(json.dumps({'captured': name, 'raw_audio_ms': len(raw) / 48, 'first_raw_ms': chunks[0]['ms'] if chunks else None}), flush=True)


def render(raw: bytes, chunks: list[dict], variant: str, *, paced: bool) -> tuple[bytes, dict]:
    processor, reassembler = AcousticStreamProcessor(initial_segment=variant), Pcm16Reassembler()
    output, offset, first, nonsilent = [], 0, None, None
    started, cpu_started = time.perf_counter(), time.process_time()
    def append(blocks: list[bytes]) -> None:
        nonlocal first, nonsilent
        for block in blocks:
            if not block: continue
            elapsed = (time.perf_counter() - started) * 1000
            if first is None: first = elapsed
            if nonsilent is None and np.any(np.abs(np.frombuffer(block, dtype='<i2').astype(np.int32)) > 100): nonsilent = elapsed
            output.append(block)
    for item in chunks:
        if paced:
            delay = item['ms'] / 1000 - (time.perf_counter() - started)
            if delay > 0: time.sleep(delay)
        samples = reassembler.feed(raw[offset:offset + item['bytes']]); offset += item['bytes']
        for begin in range(0, len(samples), 6000): append(processor.push(samples[begin:begin + 6000]))
    reassembler.finish(); append(processor.finish())
    pcm = b''.join(output)
    values = np.frombuffer(pcm, dtype='<i2').astype(np.float64)
    metrics = {'variant': variant, 'first_pcm_ms': first, 'first_nonsilent_ms': nonsilent,
        'dsp_cpu_ms': (time.process_time() - cpu_started) * 1000,
        'wall_ms': (time.perf_counter() - started) * 1000, 'world_ms': processor.metrics.world_seconds * 1000,
        'input_audio_ms': len(raw) / 48, 'output_audio_ms': len(pcm) / 48,
        'first_dsp_input_audio_ms': processor.metrics.first_render_received_samples / 24,
        'peak_pcm16': float(np.max(np.abs(values))) if len(values) else 0,
        'rms_pcm16': float(np.sqrt(np.mean(values * values))) if len(values) else 0,
        'clipped_samples': int(np.sum(np.abs(values) >= 32767)),
        'hard_seams': processor.metrics.hard_seams, 'natural_segments': processor.metrics.natural_segments}
    return pcm, metrics


def replay(directory: Path) -> None:
    rows = []
    for name in CORPUS:
        raw = (directory / f'{name}-raw.pcm').read_bytes()
        transport = json.loads((directory / f'{name}-transport.json').read_text())
        for variant in ['stable', 'initial-quiet-900ms-v1']:
            pcm, row = render(raw, transport['chunks'], variant, paced=True)
            row['case'] = name; rows.append(row)
            (directory / f'{name}-{variant}.pcm').write_bytes(pcm)
            # Repeat with arbitrary odd transport fragments; waveform equality
            # is an integrity assertion, not a substitute for listening.
            rng = np.random.default_rng(40733133); offset = 0; pieces = []
            while offset < len(raw):
                count = min(int(rng.integers(1, 8192)), len(raw) - offset)
                pieces.append({'ms': 0, 'bytes': count}); offset += count
            fragmented, _ = render(raw, pieces, variant, paced=False)
            row['fragmentation_identical'] = fragmented == pcm
            print(json.dumps(row), flush=True)
    (directory / 'dsp-comparison.json').write_text(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    directory = args.output_dir.expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    if (repository / '.git').exists() and (repository == directory or repository in directory.parents):
        parser.error('select a private directory outside the repository')
    os.umask(0o077)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.capture: capture(directory)
    if args.replay: replay(directory)
    if not args.capture and not args.replay: parser.error('select --capture and/or --replay')

if __name__ == '__main__': main()
