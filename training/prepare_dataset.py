#!/usr/bin/env python3
"""Import reviewed *processed* clips. Never calls a provider or downloads models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
import wave

import numpy as np
from scipy.signal import resample_poly

REPOSITORY = Path(__file__).resolve().parents[1]
RATE = 22050


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def private_destination(path: Path) -> Path:
    path = path.resolve()
    if path == REPOSITORY or REPOSITORY in path.parents:
        raise ValueError('private audio/manifests must be outside the repository')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def resample_pcm(values: np.ndarray, source_rate: int, target_rate: int = RATE) -> np.ndarray:
    from math import gcd
    if source_rate == target_rate:
        return np.asarray(values, dtype=np.float64).copy()
    divisor = gcd(source_rate, target_rate)
    return resample_poly(values, target_rate // divisor, source_rate // divisor)


def read_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), 'rb') as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError('reviewed input must be mono PCM16 WAV')
        rate = audio.getframerate()
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2').astype(np.float64) / 32768, rate


def write_pcm(path: Path, values: np.ndarray, rate: int) -> None:
    if not np.all(np.isfinite(values)) or np.max(np.abs(values), initial=0) >= 1:
        raise ValueError('non-finite/clipped audio; review instead of silently limiting')
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(np.round(values * 32768).astype('<i2').tobytes())
    path.chmod(0o600)


def select_records(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    selected, rejected, ids, groups = [], [], set(), {}
    for row in rows:
        identifier = row.get('id', '')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', identifier) or identifier in ids:
            raise ValueError('invalid or duplicate stable id')
        ids.add(identifier)
        split = row.get('split')
        if split not in {'train', 'validation', 'test'}:
            raise ValueError(f'{identifier}: invalid predetermined split')
        group = row.get('group_id', identifier)
        if group in groups and groups[group] != split:
            raise ValueError(f'{identifier}: near-duplicate group crosses splits')
        groups[group] = split
        if row.get('quality_status') != 'accepted' or row.get('alignment_status') != 'aligned' or row.get('content_review') != 'approved':
            rejected.append({'id': identifier, 'reason': 'quality/alignment/content review not approved'})
            continue
        if not row.get('source_master_sha256'):
            raise ValueError(f'{identifier}: processed-master provenance missing')
        text = unicodedata.normalize('NFC', row.get('spoken_text') or row.get('text', '')).strip()
        if not text or any(c in text for c in '\r\n\x00|'):
            raise ValueError(f'{identifier}: invalid reviewed transcript')
        # Do not guess how a provider read a date/number/abbreviation. A reviewer
        # can supply spoken_text; otherwise eSpeak receives the reviewed text.
        selected.append(dict(row, text=text))
    return selected, rejected


def prepare(manifest: Path, output: Path) -> dict:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected, rejected = select_records(rows)
    if not selected:
        raise ValueError('no reviewed processed clips are available')
    output = private_destination(output)
    audio_dir = output / 'audio'
    audio_dir.mkdir(mode=0o700, exist_ok=True)
    records = []
    for row in selected:
        source = Path(row['audio_filepath'])
        if not source.is_absolute():
            source = manifest.parent / source
        digest = sha256(source)
        if digest != row['sha256']:
            raise ValueError(f"{row['id']}: input hash mismatch")
        values, rate = read_pcm(source)
        duration = len(values) / rate
        if not .2 <= duration <= 30 or np.max(np.abs(values), initial=0) < 1e-4:
            raise ValueError(f"{row['id']}: empty or extreme clip requires review")
        if abs(float(row['duration']) - duration) > max(.002, 1 / rate):
            raise ValueError(f"{row['id']}: duration provenance mismatch")
        target = audio_dir / (row['id'] + '.wav')
        converted = resample_pcm(values, rate)
        write_pcm(target, converted, RATE)
        records.append(dict(row, audio_filepath=str(target), duration=len(converted) / RATE,
                            original_audio_sha256=digest, sha256=sha256(target), sample_rate=RATE))
    trainval = [row for row in records if row['split'] != 'test']
    if not any(r['split'] == 'train' for r in trainval) or not any(r['split'] == 'validation' for r in trainval):
        raise ValueError('both train and validation must have reviewed examples')
    with (output / 'metadata.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='|', lineterminator='\n')
        writer.writerows((Path(row['audio_filepath']).name, row['text']) for row in trainval)
    for name, content in [('dataset.jsonl', records), ('trainval.jsonl', trainval),
                          ('test.jsonl', [r for r in records if r['split'] == 'test'])]:
        (output / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in content))
        (output / name).chmod(0o600)
    summary = {'source_manifest_sha256': sha256(manifest), 'accepted': len(records),
               'excluded_pending_or_quarantined': rejected,
               'counts': {s: sum(row['split'] == s for row in records) for s in ('train', 'validation', 'test')},
               'sample_rate': RATE, 'normalization': 'no waveform loudness normalization; reviewed transcript NFC only'}
    (output / 'import-report.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    print(json.dumps(prepare(args.manifest, args.output), indent=2))


if __name__ == '__main__':
    main()
