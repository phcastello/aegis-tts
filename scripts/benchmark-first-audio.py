#!/usr/bin/env python3
"""Opt-in measurements using an existing TTS service, never a second model.

Capture synthetic raw PCM once per sample, then replay exactly those bytes and
arrival times through both DSP variants. Audio and WAV listening pairs remain
in an explicitly selected private directory outside Git. Automatic comparisons
establish sample integrity, not perceptual equivalence. Audible playback times
are estimates assuming immediate uninterrupted playback, excluding client and
speaker delay; thresholded digital samples are not an acoustic measurement.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import stat
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gateway'))
from voice_pipeline import AcousticStreamProcessor, Pcm16Reassembler

CORPUS = {
    'time': 'São nove horas e quinze minutos.',
    'math': 'Dezessete mais vinte e cinco são quarenta e dois.',
    'negation': 'Não. Primeiro confirme o endereço; depois eu continuo.',
    'numbers': 'A reunião ficou para vinte e três de novembro, às dezesseis horas e quarenta minutos.',
    'consonants': 'Chaves, filtros e três sensores passaram pela verificação.',
    'reserved_short': 'Fique por aqui; verifico os três resultados antes de confirmar.',
    'reserved_long': 'Para sexta-feira, reserve quinze centímetros de margem e confira o nome de cada arquivo, porque a última revisão precisa corresponder exatamente ao relatório aprovado.',
    'long': 'A água ferve a cem graus Celsius ao nível do mar. Em lugares mais altos, ela ferve a uma temperatura menor, por causa da pressão atmosférica.',
}


def private_directory(path: Path) -> Path:
    directory = path.expanduser().resolve()
    if any((parent / '.git').exists() for parent in (directory, *directory.parents)):
        raise ValueError('select a private directory outside any Git checkout')
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = directory.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError('output directory must be owned by this user with mode 0700')
    return directory


def write_private(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as output:
        os.fchmod(output.fileno(), 0o600)
        output.write(content)


def write_json(path: Path, value: object) -> None:
    write_private(path, json.dumps(value, indent=2, ensure_ascii=False).encode())


def write_wav(path: Path, pcm: bytes) -> None:
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    write_private(path, output.getvalue())


def request_id() -> str:
    alphabet = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'
    value = int.from_bytes(int(time.time() * 1000).to_bytes(6, 'big') + os.urandom(10), 'big')
    return ''.join(alphabet[(value >> shift) & 31] for shift in range(125, -1, -5))


def audible_estimate(pcm: bytes, first_pcm_ms: float | None, chunks: list[dict] | None = None) -> dict:
    values = np.frombuffer(pcm, dtype='<i2').astype(np.int32)
    above_threshold = np.abs(values) > 100
    leading = int(np.argmax(above_threshold)) / 24 if np.any(above_threshold) else None
    playback = first_pcm_ms + leading if first_pcm_ms is not None and leading is not None else None
    available, underruns, underrun_ms, end_ms, received = None, 0, 0.0, None, 0
    if chunks is not None:
        # Simulate immediate playback of each complete sample, respecting its
        # actual arrival. first_pcm + audio_offset alone is wrong when a tiny
        # silent codec chunk is followed by a late batch and an underrun.
        playback = None
        voice_sample = int(np.argmax(above_threshold)) if np.any(above_threshold) else None
        for chunk in chunks:
            before = received // 2
            received += chunk['bytes']
            after = received // 2
            if after == before:
                continue
            arrival = chunk['ms']
            if end_ms is not None and arrival - end_ms > 10:
                underruns += 1
                underrun_ms += arrival - end_ms
            start = max(arrival, end_ms) if end_ms is not None else arrival
            if voice_sample is not None and available is None and before <= voice_sample < after:
                available = arrival
                playback = start + (voice_sample - before) / 24
            end_ms = start + (after - before) / 24
        if received != len(pcm):
            raise ValueError('playback transport schedule does not consume PCM')
    return {'leading_nonsilent_audio_ms': leading,
            'first_voice_available_ms': available,
            'estimated_first_audible_playback_ms': playback,
            'immediate_playback_underruns': underruns if chunks is not None else None,
            'immediate_playback_underrun_ms': underrun_ms if chunks is not None else None,
            'audible_playback_estimate_assumption': 'immediate playback respecting chunk arrivals; excludes client and speaker delay'}


def receive(client: httpx.Client, path: str, payload: dict, identifier: str) -> tuple[bytes, dict]:
    started = time.perf_counter()
    raw, chunks, first_nonsilent = bytearray(), [], None
    reassembler = Pcm16Reassembler()
    with client.stream('POST', path, json=payload, headers={'X-Request-Id': identifier}) as response:
        response.raise_for_status()
        headers_ms = (time.perf_counter() - started) * 1000
        allowed_headers = {'content-type', 'x-request-id', 'x-aegis-request-id', 'x-aegis-voice-profile',
                           'x-aegis-audio-format', 'x-aegis-sample-rate', 'x-aegis-channels',
                           'x-aegis-tts-source-commit', 'x-aegis-tts-dsp-variant', 'x-aegis-tts-runtime-variant'}
        safe_headers = {key: value for key, value in response.headers.items() if key.lower() in allowed_headers}
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            elapsed = (time.perf_counter() - started) * 1000
            chunks.append({'ms': elapsed, 'bytes': len(chunk)})
            raw.extend(chunk)
            samples = reassembler.feed(chunk)
            if first_nonsilent is None and np.any(np.abs(samples) > 100 / 32768):
                first_nonsilent = elapsed
    reassembler.finish()
    return bytes(raw), {'request_id': identifier, 'headers': safe_headers,
        'headers_ms': headers_ms, 'first_pcm_ms': chunks[0]['ms'] if chunks else None,
        'first_nonsilent_ms': first_nonsilent, 'eof_ms': (time.perf_counter() - started) * 1000,
        'audio_ms': len(raw) / 48, 'chunks': chunks,
        **audible_estimate(bytes(raw), chunks[0]['ms'] if chunks else None, chunks)}


def samples(cases: list[str], repetitions: int):
    for repetition in range(1, repetitions + 1):
        for case in cases:
            yield f'{case}-{repetition:02d}', case, repetition


def capture(directory: Path, cases: list[str], repetitions: int,
            initial_codec_chunk_frames: int | None = None) -> None:
    import default_reference_gateway as gateway
    gateway._verify_protected_assets()
    gateway.app.state.default_audio, gateway.app.state.default_text = gateway._load_default_reference()
    rows = []
    with httpx.Client(base_url=gateway.BACKEND_URL, timeout=100) as client:
        for name, case, repetition in samples(cases, repetitions):
            payload = gateway._apply_defaults({'input': CORPUS[case], 'stream': True, 'response_format': 'pcm'})
            if initial_codec_chunk_frames is not None:
                payload['initial_codec_chunk_frames'] = initial_codec_chunk_frames
            identifier = request_id()
            observed = time.perf_counter()
            try:
                raw, row = receive(client, '/v1/audio/speech', payload, identifier)
                row.update(sample=name, case=case, repetition=repetition)
                row['initial_codec_chunk_frames_override'] = initial_codec_chunk_frames
                write_private(directory / f'{name}-raw.pcm', raw)
                write_json(directory / f'{name}-transport.json', row)
                print(json.dumps({'captured': name, 'request_id': identifier,
                                  'raw_audio_ms': row['audio_ms'], 'first_raw_ms': row['first_pcm_ms']}), flush=True)
            except (httpx.HTTPError, ValueError) as error:
                row = {'sample': name, 'case': case, 'repetition': repetition,
                       'request_id': identifier, 'error': type(error).__name__,
                       'failed_after_ms': (time.perf_counter() - observed) * 1000}
                print(json.dumps(row), flush=True)
            rows.append(row)
            write_json(directory / 'capture-results.json', rows)
    if any('error' in row for row in rows):
        raise RuntimeError('one or more captures failed; all observations retained')


def benchmark(directory: Path, url: str, cases: list[str], repetitions: int,
              wait_idle_seconds: float = 0) -> None:
    rows = []
    with httpx.Client(base_url=url.rstrip('/'), timeout=120) as client:
        for name, case, repetition in samples(cases, repetitions):
            identifier = request_id()
            row = {'sample': name, 'case': case, 'repetition': repetition, 'request_id': identifier}
            observed = time.perf_counter()
            try:
                queue = client.get('/v1/aegis/queue')
                queue.raise_for_status()
                row['queue_before'] = queue.json()
                if wait_idle_seconds:
                    waiting = time.perf_counter()
                    while row['queue_before']['active_request_id'] or row['queue_before']['queued_requests']:
                        if time.perf_counter() - waiting >= wait_idle_seconds:
                            raise TimeoutError('idle condition not reached; no synthesis sent')
                        time.sleep(.1)
                        queue = client.get('/v1/aegis/queue')
                        queue.raise_for_status()
                        row['queue_before'] = queue.json()
                    row['idle_gate_wait_ms'] = (time.perf_counter() - waiting) * 1000
                raw, timings = receive(client, '/v1/aegis/speech', {
                    'request_id': identifier, 'text': CORPUS[case], 'stream': True,
                    'response_format': 'pcm', 'priority': 50, 'interrupt_policy': 'enqueue'}, identifier)
                row.update(timings)
                write_private(directory / f'{name}-client.pcm', raw)
                events = client.get(f'/v1/aegis/speech/{identifier}/events')
                events.raise_for_status()
                row['events'] = [json.loads(line[6:]) for line in events.text.splitlines() if line.startswith('data: ')]
            except (httpx.HTTPError, ValueError, TimeoutError) as error:
                row['error'] = type(error).__name__
                row['failed_after_ms'] = (time.perf_counter() - observed) * 1000
            rows.append(row)
            write_json(directory / 'client-results.json', rows)
            print(json.dumps({key: value for key, value in row.items() if key not in {'chunks', 'events'}}), flush=True)
    if any('error' in row for row in rows):
        raise RuntimeError('one or more requests failed; all observations retained')


def render(raw: bytes, chunks: list[dict], variant: str, *, paced: bool,
           eof_ms: float | None = None) -> tuple[bytes, dict]:
    processor, reassembler = AcousticStreamProcessor(initial_segment=variant), Pcm16Reassembler()
    output, emissions, offset, first, nonsilent = [], [], 0, None, None
    started, cpu_started = time.perf_counter(), time.process_time()
    processor_offset = processor._started - started

    def wait_until(milliseconds: float) -> None:
        delay = milliseconds / 1000 - (time.perf_counter() - started)
        if paced and delay > 0:
            time.sleep(delay)

    def append(blocks: list[bytes]) -> None:
        nonlocal first, nonsilent
        for block in blocks:
            if not block:
                continue
            elapsed = (time.perf_counter() - started) * 1000
            if first is None:
                first = elapsed
            if nonsilent is None and np.any(np.abs(np.frombuffer(block, dtype='<i2').astype(np.int32)) > 100):
                nonsilent = elapsed
            output.append(block)
            emissions.append({'ms': elapsed, 'bytes': len(block)})

    for item in chunks:
        wait_until(item['ms'])
        size = item['bytes']
        if size < 0 or offset + size > len(raw):
            raise ValueError('transport schedule exceeds raw PCM')
        values = reassembler.feed(raw[offset:offset + size])
        offset += size
        for begin in range(0, len(values), 6000):
            append(processor.push(values[begin:begin + 6000]))
    if offset != len(raw):
        raise ValueError('transport schedule does not consume raw PCM')
    if eof_ms is not None:
        wait_until(eof_ms)
    reassembler.finish()
    append(processor.finish())
    pcm = b''.join(output)
    values = np.frombuffer(pcm, dtype='<i2').astype(np.float64)
    first_start = processor.metrics.first_render_started_seconds
    first_end = processor.metrics.first_render_finished_seconds
    metrics = {'variant': variant, 'first_pcm_ms': first, 'first_nonsilent_ms': nonsilent,
        'dsp_cpu_ms': (time.process_time() - cpu_started) * 1000,
        'wall_ms': (time.perf_counter() - started) * 1000, 'world_ms': processor.metrics.world_seconds * 1000,
        'input_audio_ms': len(raw) / 48, 'output_audio_ms': len(pcm) / 48,
        'first_dsp_input_audio_ms': processor.metrics.first_render_received_samples / 24,
        'first_dsp_start_ms': (processor_offset + first_start) * 1000 if first_start is not None else None,
        'first_dsp_end_ms': (processor_offset + first_end) * 1000 if first_end is not None else None,
        'first_dsp_duration_ms': (first_end - first_start) * 1000 if first_start is not None and first_end is not None else None,
        'peak_pcm16': float(np.max(np.abs(values))) if len(values) else 0,
        'rms_pcm16': float(np.sqrt(np.mean(values * values))) if len(values) else 0,
        'clipped_samples': int(np.sum(np.abs(values) >= 32767)),
        'hard_seams': processor.metrics.hard_seams, 'natural_segments': processor.metrics.natural_segments,
        'chunks': emissions, **audible_estimate(pcm, first, emissions)}
    return pcm, metrics


def load_baseline(path: Path):
    spec = importlib.util.spec_from_file_location('aegis_latency_baseline_pipeline', path)
    if spec is None or spec.loader is None:
        raise ValueError('cannot load baseline pipeline')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def baseline_pcm(module, raw: bytes, chunks: list[dict]) -> bytes:
    processor, reassembler = module.AcousticStreamProcessor(), module.Pcm16Reassembler()
    output, offset = [], 0
    for item in chunks:
        end = offset + item['bytes']
        output.extend(processor.push(reassembler.feed(raw[offset:end])))
        offset = end
    reassembler.finish()
    output.extend(processor.finish())
    return b''.join(output)


def replay(directory: Path, baseline_path: Path | None = None) -> None:
    # Production imports DSP at startup. Exclude the lazy pyworld import from
    # both timed variants; otherwise the first stable replay is unfairly slow.
    import pyworld  # noqa: F401
    rows = []
    baseline = load_baseline(baseline_path) if baseline_path else None
    for path in sorted(directory.glob('*-transport.json')):
        name = path.name.removesuffix('-transport.json')
        raw = (directory / f'{name}-raw.pcm').read_bytes()
        transport = json.loads(path.read_text())
        stable_pcm = None
        for variant in ['stable', 'initial-quiet-900ms-v1']:
            pcm, row = render(raw, transport['chunks'], variant, paced=True, eof_ms=transport.get('eof_ms'))
            row.update(sample=name, case=transport.get('case', name), request_id=transport.get('request_id'))
            if variant == 'stable':
                stable_pcm = pcm
                if baseline:
                    row['baseline_pcm_identical'] = baseline_pcm(baseline, raw, transport['chunks']) == pcm
            row['stable_pcm_identical'] = stable_pcm == pcm
            write_private(directory / f'{name}-{variant}.pcm', pcm)
            write_wav(directory / f'{name}-{variant}.wav', pcm)
            # Refragment only for sample integrity; timing below always uses the
            # original schedule, including EOF for short finalization paths.
            rng = np.random.default_rng(40733133)
            offset, pieces = 0, []
            while offset < len(raw):
                count = min(int(rng.integers(1, 8192)), len(raw) - offset)
                pieces.append({'ms': 0, 'bytes': count})
                offset += count
            fragmented, _ = render(raw, pieces, variant, paced=False)
            row['fragmentation_identical'] = fragmented == pcm
            rows.append(row)
            write_json(directory / 'dsp-comparison.json', rows)
            print(json.dumps(row), flush=True)
    if not rows:
        raise ValueError('no captured raw transport samples found')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--benchmark-url', help='measure native production API; does not capture model raw PCM')
    parser.add_argument('--wait-idle-seconds', type=float, default=0,
                        help='optional read-only queue gate before each request; wait is recorded outside request latency')
    parser.add_argument('--baseline-pipeline', type=Path, help='original deployed voice_pipeline.py for byte equality')
    parser.add_argument('--initial-codec-chunk-frames', type=int,
                        help='experimental capture-only override; verify installed API support and review acoustic differences')
    parser.add_argument('--repetitions', type=int, default=1, choices=range(1, 6))
    parser.add_argument('--case', action='append', choices=CORPUS, dest='cases')
    args = parser.parse_args()
    if not (args.capture or args.replay or args.benchmark_url):
        parser.error('select --capture, --replay and/or --benchmark-url')
    if args.initial_codec_chunk_frames is not None and (not args.capture or args.initial_codec_chunk_frames < 1):
        parser.error('--initial-codec-chunk-frames requires --capture and a positive value')
    if not 0 <= args.wait_idle_seconds <= 60:
        parser.error('--wait-idle-seconds must be between 0 and 60')
    os.umask(0o077)
    try:
        directory = private_directory(args.output_dir)
        cases = args.cases or list(CORPUS)
        if args.benchmark_url:
            benchmark(directory, args.benchmark_url, cases, args.repetitions, args.wait_idle_seconds)
        if args.capture:
            capture(directory, cases, args.repetitions, args.initial_codec_chunk_frames)
        if args.replay:
            replay(directory, args.baseline_pipeline)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'benchmark failed: {error}\n')


if __name__ == '__main__':
    main()
