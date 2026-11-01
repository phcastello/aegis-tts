#!/usr/bin/env python3
"""Private single-master dataset pipeline. No provider/voice/budget defaults.

Requires numpy/scipy only for audio inspection, resampling and separator matching.
All generated files must live outside the repository, even manifests/review files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import urllib.request
import wave

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('provider', 'base_url', 'model', 'voice', 'training_rights_confirmed',
            'voice_rights_confirmed', 'terms_url', 'terms_reviewed_on', 'currency',
            'price_per_1000_characters', 'max_cost', 'max_characters',
            'max_requests', 'max_examples', 'generation_approved', 'approval_reference', 'corpus_review_sha256')
VERSION = 'aegis-single-master-v1'
MAX_AUDIO_WORK_BYTES = 8 * 1024 ** 3


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    hashed = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hashed.update(block)
    return hashed.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def private_directory(path: Path) -> Path:
    path = path.resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError('generated data must be outside the Git checkout')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def json_write(path: Path, value) -> None:
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2).encode())


def corpus_read(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids, groups, text_groups = set(), {}, {}
    for record in records:
        for field in ('id', 'text', 'group', 'split', 'stage'):
            if not record.get(field):
                raise ValueError(f'corpus entry missing {field}')
        if record['id'] in ids or record['split'] not in ('train', 'validation', 'test'):
            raise ValueError('duplicate id or invalid split')
        ids.add(record['id'])
        if groups.setdefault(record['group'], record['split']) != record['split']:
            raise ValueError('group leaks across splits')
        normalized = ' '.join(record['text'].lower().split())
        if normalized in text_groups:
            raise ValueError('duplicate text in corpus')
        text_groups[normalized] = record['group']
    return records


def selected(records: list[dict], stage: str) -> list[dict]:
    return [record for record in records if stage == 'full' or record['stage'] == 'pilot']


def missing_config(config: dict) -> list[str]:
    missing = [field for field in REQUIRED if config.get(field) is None or config.get(field) == '']
    for field in ('training_rights_confirmed', 'voice_rights_confirmed', 'generation_approved'):
        if config.get(field) is not True and field not in missing:
            missing.append(field)
    return missing


def validate_config(config: dict) -> None:
    missing = missing_config(config)
    if missing:
        raise ValueError('missing explicit dataset configuration: ' + ', '.join(missing))
    if config['provider'] != 'openai-compatible-http':
        raise ValueError('supported provider: openai-compatible-http; choose it explicitly')
    if not config['base_url'].startswith(('http://', 'https://')):
        raise ValueError('base_url must be an explicit HTTP(S) speech endpoint')
    if not config['terms_url'].startswith('https://'):
        raise ValueError('terms_url must identify the reviewed provider/voice training terms')
    for key in ('price_per_1000_characters', 'max_cost'):
        if isinstance(config[key], bool) or not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'{key} must be nonnegative')
    for key in ('max_characters', 'max_requests', 'max_examples'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 1 <= config.get('concurrency', 1) <= 4 or not 0 <= config.get('retries', 2) <= 4:
        raise ValueError('concurrency must be 1..4 and retries 0..4')
    if config.get('api_key_env') and not os.environ.get(config['api_key_env']):
        raise ValueError('missing credential environment variable: ' + config['api_key_env'])


def audio_read(path_or_bytes) -> tuple[np.ndarray, int]:
    source = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else str(path_or_bytes)
    with wave.open(source, 'rb') as audio:
        if audio.getcomptype() != 'NONE' or audio.getsampwidth() != 2:
            raise ValueError('expected uncompressed PCM16 WAV; export PCM16 from DAW')
        channels, rate = audio.getnchannels(), audio.getframerate()
        frames = audio.getnframes()
        # Bound peak decode/inspection arrays before allocation (not the on-disk WAV size).
        estimated_work = frames * channels * 8 * 3
        if estimated_work > MAX_AUDIO_WORK_BYTES:
            raise ValueError('WAV exceeds 8GiB decoded working-set limit; export mono 24kHz or process windows')
        available = next((int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')), None)
        if available is not None and estimated_work + 1024 ** 3 > available:
            raise ValueError('insufficient available host RAM for bounded WAV inspection')
        data = audio.readframes(frames)
        if len(data) != frames * channels * 2:
            raise ValueError('truncated WAV data')
        values = np.frombuffer(data, dtype='<i2').astype(np.float64) / 32768
        if channels != 1:
            raise ValueError('expected mono WAV; stereo must be explicitly converted before import')
    if not 8000 <= rate <= 192000 or not len(values):
        raise ValueError('invalid sample rate or empty WAV')
    return values, rate


def wav_bytes(values: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(np.rint(np.clip(values, -1, 32767 / 32768) * 32768).astype('<i2').tobytes())
    return buf.getvalue()


def resample(values: np.ndarray, old: int, new: int) -> np.ndarray:
    if old == new:
        return values
    divisor = math.gcd(old, new)
    return resample_poly(values, new // divisor, old // divisor)


def audio_stats(values: np.ndarray, rate: int) -> dict:
    return {'duration': len(values) / rate, 'samples': len(values), 'sample_rate': rate,
            'peak': float(np.max(np.abs(values), initial=0)),
            'rms': float(np.sqrt(np.mean(values * values))) if len(values) else 0,
            'clipped_samples': int(np.sum(np.abs(values) >= 32767 / 32768))}


def request_identity(record: dict, config: dict) -> tuple[str, dict]:
    payload = {'model': config.get('model'), 'voice': config.get('voice'),
               'input': record['text'], 'response_format': 'wav', 'stream': False}
    identity = {'provider': config.get('provider'), 'base_url': config.get('base_url'),
                'payload': payload, 'pipeline_version': VERSION}
    return digest(canonical(identity)), payload


def cached(record: dict, config: dict, directory: Path):
    key, payload = request_identity(record, config)
    path = directory / 'cache' / f"{record['id']}-{key[:20]}.wav"
    meta = path.with_suffix('.json')
    if path.exists() and meta.exists():
        saved = json.loads(meta.read_text())
        if saved.get('request_hash') == key and saved.get('sha256') == file_digest(path):
            audio_read(path)
            return saved, path, meta, key, payload
    return None, path, meta, key, payload


def estimate(records: list[dict], config: dict, directory: Path) -> dict:
    pending = [record for record in records if cached(record, config, directory)[0] is None]
    characters = sum(len(record['text']) for record in pending)
    price = config.get('price_per_1000_characters')
    words = sum(len(record['text'].split()) for record in records)
    return {'examples': len(records), 'cached': len(records) - len(pending), 'pending': len(pending),
            'pending_characters': characters, 'speech_minutes_estimate_150wpm': words / 150, 'corpus_sha256': digest(canonical(records)),
            'estimated_cost_first_attempt': characters / 1000 * price if price is not None else None,
            'estimated_cost_all_retries': characters / 1000 * price * (1 + config.get('retries', 2)) if price is not None else None,
            'currency': config.get('currency'), 'missing_configuration': missing_config(config),
            'estimate_basis': 'Unicode characters; verify billing/token/unit rules for configured provider; silence excluded'}


def _generate(records: list[dict], config: dict, directory: Path, *, dry_run: bool = False,
             transport=None) -> dict:
    report = estimate(records, config, directory)
    if dry_run:
        return report
    validate_config(config)
    if config['corpus_review_sha256'] != digest(canonical(records)):
        raise ValueError('corpus review SHA256 does not match selected corpus; review before generation')
    if len(records) > config['max_examples']:
        raise ValueError('corpus exceeds max_examples; no requests sent')
    ledger_path = directory / 'generation-ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'attempts': [], 'created': utc()}
    lock = threading.Lock()
    failures, successes = [], []

    def default_transport(payload: dict) -> bytes:
        headers = {'Content-Type': 'application/json'}
        if config.get('api_key_env'):
            headers['Authorization'] = 'Bearer ' + os.environ[config['api_key_env']]
        request = urllib.request.Request(config['base_url'], data=canonical(payload), headers=headers)
        with urllib.request.urlopen(request, timeout=config.get('timeout_seconds', 120)) as response:
            # A malformed response never becomes a cache hit; no secret headers are logged.
            return response.read(64 * 1024 * 1024)

    sender = transport or default_transport

    def one(record: dict) -> dict:
        saved, path, meta, key, payload = cached(record, config, directory)
        if saved:
            return saved
        for attempt in range(1 + config.get('retries', 2)):
            charge = len(record['text']) / 1000 * config['price_per_1000_characters']
            with lock:
                attempts = ledger['attempts']
                if (len(attempts) + 1 > config['max_requests'] or
                    sum(row['characters'] for row in attempts) + len(record['text']) > config['max_characters'] or
                    sum(row['reserved_cost'] for row in attempts) + charge > config['max_cost'] + 1e-12):
                    raise ValueError('explicit generation/cost ceiling reached')
                entry = {'id': record['id'], 'request_hash': key, 'time': utc(),
                         'characters': len(record['text']), 'reserved_cost': charge, 'status': 'reserved'}
                attempts.append(entry)
                json_write(ledger_path, ledger)  # Reserve before external call, including retries/crash.
            try:
                data = sender(payload)
                values, rate = audio_read(data)
                stats = audio_stats(values, rate)
                if stats['duration'] < .1 or stats['rms'] < 1e-5:
                    raise ValueError('empty or effectively silent generation')
                atomic(path, data)
                saved = {**record, **stats, 'audio_filepath': str(path), 'sha256': digest(data),
                         'request_hash': key, 'parameters': payload, 'provider': config['provider'],
                         'terms_url': config['terms_url'], 'terms_reviewed_on': config['terms_reviewed_on'],
                         'training_rights_confirmed': True, 'voice_rights_confirmed': True,
                         'generated_at': utc(), 'content_review': 'pending'}
                json_write(meta, saved)
                with lock:
                    entry['status'] = 'completed'
                    json_write(ledger_path, ledger)
                return saved
            except Exception as error:
                with lock:
                    entry['status'] = 'failed'
                    entry['error_type'] = type(error).__name__
                    json_write(ledger_path, ledger)
                if attempt == config.get('retries', 2):
                    raise
                time.sleep(min(8, 2 ** attempt))
        raise AssertionError('unreachable')

    # Cache checks, one request per example, bounded pool. Ledger caps include failed calls.
    with ThreadPoolExecutor(max_workers=config.get('concurrency', 1)) as pool:
        futures = {pool.submit(one, record): record for record in records}
        for future in as_completed(futures):
            record = futures[future]
            try:
                successes.append(future.result())
            except Exception as error:
                failures.append({'id': record['id'], 'error_type': type(error).__name__,
                                 'reason': 'ceiling' if 'ceiling' in str(error) else 'generation_failed'})
            json_write(directory / 'generation-failures.json', failures)
    order = {record['id']: i for i, record in enumerate(records)}
    successes.sort(key=lambda row: order[row['id']])
    atomic(directory / 'generation.jsonl', b''.join(canonical(row) + b'\n' for row in successes))
    return {**report, 'completed': len(successes), 'failed': len(failures), 'all_complete': not failures}


def generate(records: list[dict], config: dict, directory: Path, *, dry_run: bool = False,
             transport=None) -> dict:
    import fcntl
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / '.generation.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('another generation process owns this work directory') from error
        return _generate(records, config, directory, dry_run=dry_run, transport=transport)


def quiet_runs(values: np.ndarray, rate: int, *, threshold: float = .0006,
               minimum_seconds: float = .7) -> list[dict]:
    width = max(1, round(rate * .01))
    padded = np.pad(values, (0, (-len(values)) % width))
    rms = np.sqrt(np.mean(padded.reshape(-1, width) ** 2, axis=1))
    quiet = rms <= threshold
    edges = np.diff(np.r_[False, quiet, False].astype(np.int8))
    runs = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        a, b = start * width, min(len(values), end * width)
        if (b - a) / rate >= minimum_seconds:
            runs.append({'start': int(a), 'end': int(b), 'center': (a + b) / 2,
                         'seconds': (b - a) / rate})
    return runs


def envelope_signature(values: np.ndarray, size: int = 64) -> np.ndarray:
    if not len(values):
        return np.zeros(size)
    pieces = np.array_split(values, min(size, len(values)))
    energy = np.array([np.sqrt(np.mean(piece * piece)) for piece in pieces])
    energy = np.interp(np.linspace(0, 1, size), np.linspace(0, 1, len(energy)), energy)
    return energy / max(float(np.linalg.norm(energy)), 1e-12)


def active_bounds(values: np.ndarray, rate: int, threshold: float = .0006) -> tuple[int, int]:
    width = max(1, round(rate * .01))
    padded = np.pad(values, (0, (-len(values)) % width))
    rms = np.sqrt(np.mean(padded.reshape(-1, width) ** 2, axis=1))
    found = np.flatnonzero(rms > threshold)
    return (int(found[0] * width), min(len(values), int((found[-1] + 1) * width))) if len(found) else (0, 0)


def assemble(records: list[dict], config: dict, directory: Path) -> dict:
    rate = int(config.get('sample_rate', 24000))
    separator = round(float(config.get('separator_seconds', 2)) * rate)
    bookend = round(float(config.get('bookend_seconds', 2)) * rate)
    if separator < rate or bookend < rate:
        raise ValueError('separators/bookends must be at least one second')
    clips = []
    for record in records:
        saved, path, _, _, _ = cached(record, config, directory)
        if saved is None:
            raise ValueError('missing/invalid generation cache: ' + record['id'])
        with wave.open(str(path), 'rb') as audio:
            frames, source_rate = audio.getnframes(), audio.getframerate()
        target_length = math.ceil(frames * rate / source_rate)
        clips.append((record, path, saved, target_length))
    total = 2 * bookend + sum(length for _, _, _, length in clips) + max(0, len(clips) - 1) * separator
    if total * 2 + 44 >= 2 ** 32 - 1:
        raise ValueError('master exceeds RIFF WAV 4GiB; reduce sample rate or implement a single RF64 export, never split manual work')
    # Stream one WAV, no concatenated float copy of the entire 1-2h master in memory.
    master = directory / 'master_raw.wav'
    fd, temp = tempfile.mkstemp(prefix='.master-', dir=directory)
    os.fchmod(fd, 0o600)
    os.close(fd)
    entries, gaps, offset = [], [{'start': 0, 'end': bookend, 'kind': 'bookend'}], bookend
    try:
        with wave.open(temp, 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(rate)
            audio.writeframes(b'\0' * bookend * 2)
            for index, (record, path, saved, expected_length) in enumerate(clips):
                values, source_rate = audio_read(path)
                values = resample(values, source_rate, rate)
                if len(values) != expected_length:
                    raise ValueError('resampling length differs from master preflight')
                begin, end = active_bounds(values, rate)
                entry = {**record, 'order': index, 'sample_rate': rate,
                         'start_sample': offset, 'end_sample': offset + len(values),
                         'start_seconds': offset / rate, 'end_seconds': (offset + len(values)) / rate,
                         'duration': len(values) / rate, 'active_start_sample': offset + begin,
                         'active_end_sample': offset + end,
                         'source_audio_sha256': saved['sha256'],
                         'source_request_hash': saved['request_hash'],
                         'envelope_signature': envelope_signature(values[begin:end]).tolist(),
                         'content_review': 'pending'}
                entries.append(entry)
                audio.writeframes(np.rint(np.clip(values, -1, 32767 / 32768) * 32768).astype('<i2').tobytes())
                offset += len(values)
                if index + 1 < len(clips):
                    gaps.append({'start': offset, 'end': offset + separator, 'kind': 'separator'})
                    audio.writeframes(b'\0' * separator * 2)
                    offset += separator
            gaps.append({'start': offset, 'end': offset + bookend, 'kind': 'bookend'})
            audio.writeframes(b'\0' * bookend * 2)
        os.replace(temp, master)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    # Hash the master incrementally; avoid loading multi-hour data to hash.
    hashed = hashlib.sha256()
    with master.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hashed.update(block)
    manifest = {'schema_version': 1, 'pipeline_version': VERSION, 'created_at': utc(),
                'sample_rate': rate, 'channels': 1, 'sample_width': 2, 'samples': total,
                'duration': total / rate, 'master_file': str(master), 'master_sha256': hashed.hexdigest(),
                'speech_clip_seconds': sum(length for _, _, _, length in clips) / rate,
                'speech_active_span_seconds': sum(row['active_end_sample'] - row['active_start_sample'] for row in entries) / rate,
                'separator_seconds': separator / rate, 'bookend_seconds': bookend / rate,
                'generation_configuration_sha256': digest(canonical({k: v for k, v in config.items() if k != 'api_key_env'})),
                'clips': entries, 'gaps': gaps}
    json_write(directory / 'manifest.json', manifest)
    fields = ['id', 'group', 'split', 'text', 'order', 'sample_rate', 'start_sample', 'end_sample',
              'start_seconds', 'end_seconds', 'duration', 'source_audio_sha256']
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(entries)
    atomic(directory / 'manifest.csv', out.getvalue().encode())
    return {key: manifest[key] for key in ('master_file', 'master_sha256', 'duration', 'speech_clip_seconds', 'speech_active_span_seconds')}


def match_gaps(expected: list[dict], observed: list[dict], total_old: int, total_new: int,
               old_active: tuple[int, int], new_active: tuple[int, int]) -> dict[int, int]:
    """Monotonic sequence alignment, allowing extra internal pauses/missing boundaries.

    Position is normalized between first/last active audio to ignore global padding.
    Unmatched anchors quarantine neighbors. Never slide transcript indices forward.
    """
    n, m = len(expected), len(observed)
    costs = np.full((n + 1, m + 1), np.inf)
    moves = np.zeros((n + 1, m + 1), dtype=np.int8)
    costs[0, 0] = 0
    for i in range(1, n + 1):
        costs[i, 0], moves[i, 0] = i * 1.4, 1
    for j in range(1, m + 1):
        costs[0, j], moves[0, j] = j * 1.0, 2
    old_span, new_span = max(1, old_active[1] - old_active[0]), max(1, new_active[1] - new_active[0])
    for i in range(1, n + 1):
        a = expected[i - 1]
        apos = ((a['start'] + a['end']) / 2 - old_active[0]) / old_span
        for j in range(1, m + 1):
            b = observed[j - 1]
            bpos = (b['center'] - new_active[0]) / new_span
            position = abs(apos - bpos) * max(1, n - 1)
            # Bookends can grow with padding; they remain outside active speech.
            if a['kind'] == 'bookend':
                valid = (i == 1 and b['start'] == 0) or (i == n and b['end'] == total_new)
                match = 0 if valid else 10
            else:
                match = position * 3 + min(2, abs((a['end'] - a['start']) - (b['end'] - b['start'])) / max(1, a['end'] - a['start'])) * .15
            options = (costs[i - 1, j - 1] + match, costs[i - 1, j] + 1.4, costs[i, j - 1] + 1.0)
            move = int(np.argmin(options))
            costs[i, j], moves[i, j] = options[move], move
    mapping, i, j = {}, n, m
    while i or j:
        move = moves[i, j]
        if move == 0:
            mapping[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return mapping


def unique_identity_landmarks(clips: list[dict], observed: list[dict], values: np.ndarray, rate: int) -> dict[int, tuple[int, int]]:
    """Recover only uniquely identifiable, noncrossing intervals after a structural loss.

    Repeated durations/envelopes intentionally yield no identity landmarks. A long
    internal pause can be crossed; it is never automatically made a sample boundary.
    """
    intervals = []
    maximum = max((clip['active_end_sample'] - clip['active_start_sample'] for clip in clips), default=0) * 1.04 + round(.1 * rate)
    for a in range(len(observed) - 1):
        for b in range(a + 1, min(len(observed), a + 5)):
            first, last = observed[a]['end'], observed[b]['start']
            if last - first > maximum:
                break
            if last > first:
                intervals.append((a, b, last - first, envelope_signature(values[first:last])))
    candidates, owners = {}, {}
    for index, clip in enumerate(clips):
        duration = clip['active_end_sample'] - clip['active_start_sample']
        tolerance = max(round(.08 * rate), round(.03 * duration))
        signature = np.asarray(clip['envelope_signature'])
        matches = []
        for a, b, length, envelope in intervals:
            if abs(length - duration) <= tolerance and float(np.dot(signature, envelope)) >= .85:
                matches.append((a, b))
                owners.setdefault((a, b), set()).add(index)
        candidates[index] = matches
    unique = {index: matches[0] for index, matches in candidates.items()
              if len(matches) == 1 and len(owners[matches[0]]) == 1}
    ordered = sorted(unique)
    crossing = set()
    for left, right in zip(ordered, ordered[1:]):
        if unique[left][1] > unique[right][0]:
            crossing.update((left, right))
    return {index: bounds for index, bounds in unique.items() if index not in crossing}


def extract(manifest_path: Path, processed: Path, destination: Path,
            review_path: Path | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text())
    # Manifest is primary: its IDs/order never derive from detected silence count.
    ids = [row['id'] for row in manifest['clips']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate IDs in manifest')
    raw_path = Path(manifest['master_file'])
    if file_digest(raw_path) != manifest['master_sha256']:
        raise ValueError('raw master hash mismatch')
    source, source_rate = audio_read(raw_path)
    rate = int(manifest['sample_rate'])
    source = resample(source, source_rate, rate)
    source_count = len(source)
    if source_count != manifest['samples']:
        raise ValueError('raw master samples differ from manifest')
    old_active = active_bounds(source, rate)
    source_quiet = quiet_runs(source, rate)
    source_mapping = match_gaps(manifest['gaps'], source_quiet, source_count, source_count, old_active, old_active)
    del source  # Never keep both multi-hour float64 masters resident.
    values, input_rate = audio_read(processed)
    values = resample(values, input_rate, rate)
    new_active = active_bounds(values, rate)
    shortened_content = new_active[1] - new_active[0] < old_active[1] - old_active[0] - round(.15 * rate)
    review = json.loads(review_path.read_text()) if review_path else {}
    # Bind approval to the actual processed master, never recycle review on a changed export.
    processed_hash = file_digest(processed)
    if review and review.get('processed_master_sha256') != processed_hash:
        raise ValueError('review does not refer to this processed master SHA256')
    decisions = review.get('examples', {})
    observed = quiet_runs(values, rate)
    mapping = match_gaps(manifest['gaps'], observed, source_count, len(values), old_active, new_active)
    known_timeline = not shortened_content and abs(source_count - len(values)) <= round(.01 * rate)
    local_offsets = []
    for anchor, observed_index in mapping.items():
        source_index = source_mapping.get(anchor)
        if source_index is None:
            known_timeline = False
            continue
        original_gap, changed_gap = source_quiet[source_index], observed[observed_index]
        if max(abs(original_gap['start'] - changed_gap['start']), abs(original_gap['end'] - changed_gap['end'])) > round(.06 * rate):
            known_timeline = False
        if manifest['gaps'][anchor]['kind'] == 'separator':
            local_offsets.append(changed_gap['center'] - original_gap['center'])
    nonuniform_timeline = bool(local_offsets and max(local_offsets) - min(local_offsets) > round(.15 * rate))
    for gap in manifest['gaps']:
        center = (gap['start'] + gap['end']) // 2
        probe = values[max(0, center - round(.25 * rate)):center + round(.25 * rate)]
        if not len(probe) or np.sqrt(np.mean(probe * probe)) > .0006:
            known_timeline = False
    landmarks = {}
    if not known_timeline and (shortened_content or nonuniform_timeline):
        landmarks = unique_identity_landmarks(manifest['clips'], observed, values, rate)
        for index, (before, after) in landmarks.items():
            mapping[index], mapping[index + 1] = before, after
    rows, hashes, ratios = [], {}, []
    for index, clip in enumerate(manifest['clips']):
        reasons, segment = [], np.array([], dtype=np.float64)
        a, b = mapping.get(index), mapping.get(index + 1)
        start, end = 0, 0
        if not known_timeline and (a is None or b is None or b != a + 1):
            # Extra long INTERNAL silence can exist between the two separator anchors;
            # allow it only when the local duration/signature remains consistent.
            if a is None or b is None:
                reasons.append('missing_separator_anchor')
        if known_timeline or (a is not None and b is not None):
            if known_timeline:
                # Exact manifest offsets, after validating all separator interiors.
                start = max(0, clip['start_sample'] - round(.080 * rate))
                end = min(len(values), clip['end_sample'] + round(.120 * rate))
                first_known, last_known = active_bounds(values[start:end], rate)
                if last_known:
                    end = min(end, start + last_known + round(.120 * rate))
                    start = max(start, start + first_known - round(.080 * rate))
            else:
                start = max(0, observed[a]['end'] - round(.080 * rate))
                end = min(len(values), observed[b]['start'] + round(.120 * rate))
            if start >= end:
                reasons.append('missing_or_fused_segment')
            else:
                segment = values[start:end]
                first, last = active_bounds(segment, rate)
                active = max(0, last - first)
                original = max(1, clip['active_end_sample'] - clip['active_start_sample'])
                ratio = active / original
                ratios.append(ratio)
                if not .80 <= ratio <= 1.20:
                    reasons.append('duration_or_content_mismatch')
                signature = envelope_signature(segment[first:last])
                similarity = float(np.dot(signature, np.asarray(clip['envelope_signature'])))
                if similarity < .65:
                    reasons.append('content_envelope_mismatch')
                if not known_timeline and b != a + 1 and abs(ratio - 1) > .05:
                    reasons.append('ambiguous_internal_silence_or_missing_boundary')
                if not known_timeline and (shortened_content or nonuniform_timeline):
                    # A missing edge segment can make many equal-spaced layouts fit.
                    # Normalized positions cannot prove identity. Accept only a unique
                    # duration/envelope candidate; repeated layouts remain quarantined.
                    candidates = []
                    for other in manifest['clips']:
                        other_active = other['active_end_sample'] - other['active_start_sample']
                        tolerance = max(round(.08 * rate), round(.03 * other_active))
                        score = float(np.dot(signature, np.asarray(other['envelope_signature'])))
                        if abs(active - other_active) <= tolerance and score >= .85:
                            candidates.append(other['id'])
                    if candidates != [clip['id']]:
                        reasons.append('ambiguous_identity_after_missing_content')
        stats = audio_stats(segment, rate)
        if stats['duration'] < .15 or stats['rms'] < 1e-5:
            reasons.append('empty_or_silent')
        if stats['clipped_samples']:
            reasons.append('clipping')
        if stats['rms'] > .7:
            reasons.append('extreme_level')
        data = wav_bytes(segment, rate)
        hashed = digest(data)
        if hashed in hashes:
            reasons.append('duplicate_audio')
            rows[hashes[hashed]]['reasons'].append('duplicate_audio')
        else:
            hashes[hashed] = index
        decision = decisions.get(clip['id'], {})
        content_review = decision.get('content_review', 'pending')
        if content_review not in ('pending', 'approved', 'rejected'):
            raise ValueError('invalid content_review decision')
        if content_review == 'approved' and (not review.get('reviewer') or not review.get('reviewed_at') or not decision.get('evidence')):
            raise ValueError('approved content requires reviewer, reviewed_at and per-example evidence')
        if content_review == 'rejected':
            reasons.append('content_review_rejected')
        row = {key: clip[key] for key in ('id', 'group', 'split', 'text', 'order')}
        row['group_id'] = clip.get('group_id', clip['group'])
        row.update(stats, sha256=hashed, content_review=content_review, reasons=reasons,
                   processed_start_sample=start, processed_end_sample=end,
                   original_sample_rate=input_rate, source_master_sha256=manifest['master_sha256'],
                   processed_master_sha256=processed_hash, source_audio_sha256=clip['source_audio_sha256'],
                   pipeline_version=VERSION, provenance={'manifest_sha256': digest(manifest_path.read_bytes()),
                   'processed_source': str(processed), 'reviewer': review.get('reviewer'),
                   'reviewed_at': review.get('reviewed_at'), 'content_evidence': decision.get('evidence')})
        rows.append(row)
        # Temporarily keep data private. Final accepted/quarantine placement follows global checks.
        atomic(destination / 'staging' / (clip['id'] + '.wav'), data)
    # Coherent global time-stretch is prohibited; small local tails are not treated as stretch.
    valid_ratios = np.array([r for r in ratios if .8 <= r <= 1.2])
    if len(valid_ratios) >= 3 and abs(float(np.median(valid_ratios)) - 1) > .02 and float(np.std(valid_ratios)) < .015:
        for row in rows:
            row['reasons'].append('global_time_stretch_detected')
    for row in rows:
        row['reasons'] = sorted(set(row['reasons']))
        row['quality_status'] = 'quarantine' if row['reasons'] else 'accepted'
        row['alignment_status'] = 'quarantine' if row['reasons'] else 'aligned'
        folder = destination / ('quarantine' if row['reasons'] else 'clips')
        folder.mkdir(exist_ok=True, mode=0o700)
        final = folder / (row['id'] + '.wav')
        os.replace(destination / 'staging' / (row['id'] + '.wav'), final)
        old = destination / ('clips' if row['reasons'] else 'quarantine') / (row['id'] + '.wav')
        old.unlink(missing_ok=True)
        row['audio_filepath'] = str(final)
    atomic(destination / 'dataset.jsonl', b''.join(canonical(row) + b'\n' for row in rows))
    report = {'schema_version': 1, 'pipeline_version': VERSION, 'processed_master_sha256': processed_hash,
              'expected_examples': len(ids), 'output_examples': len(rows),
              'accepted': sum(row['quality_status'] == 'accepted' for row in rows),
              'quarantined': [row['id'] for row in rows if row['quality_status'] == 'quarantine'],
              'content_review_pending': [row['id'] for row in rows if row['content_review'] == 'pending'],
              'shortened_active_timeline': shortened_content,
              'nonuniform_separator_offsets': nonuniform_timeline,
              'unique_identity_landmarks': landmarks,
              'alignment_method': 'manifest_verified' if known_timeline else 'monotonic_separator_match',
              'matched_anchors': mapping, 'observed_quiet_runs': observed,
              'note': 'Envelope checks do not prove words/pronunciation. Content review or independent alignment is required before training.'}
    json_write(destination / 'validation-report.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('generate', 'assemble'):
        arg = sub.add_parser(command)
        arg.add_argument('--corpus', type=Path, default=ROOT / 'dataset/corpus.pt-BR.jsonl')
        arg.add_argument('--config', type=Path, required=True)
        arg.add_argument('--work-dir', type=Path, required=True)
        arg.add_argument('--stage', choices=('pilot', 'full'), default='pilot')
        if command == 'generate':
            arg.add_argument('--dry-run', action='store_true')
    arg = sub.add_parser('extract')
    arg.add_argument('--manifest', type=Path, required=True)
    arg.add_argument('--processed', type=Path, required=True)
    arg.add_argument('--output-dir', type=Path, required=True)
    arg.add_argument('--review', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        if args.command == 'extract':
            directory = private_directory(args.output_dir)
            result = extract(args.manifest, args.processed, directory, args.review)
        else:
            config = json.loads(args.config.read_text())
            records = selected(corpus_read(args.corpus), args.stage)
            directory = private_directory(args.work_dir)
            if args.command == 'generate':
                result = generate(records, config, directory, dry_run=args.dry_run)
            else:
                result = assemble(records, config, directory)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, wave.Error) as error:
        # HTTP exceptions can contain URLs; no credentials/text/payload included here.
        parser.exit(2, f'{type(error).__name__}: {error}\n')


if __name__ == '__main__':
    main()
