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
import struct

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
    return resample_poly(values, target_rate // divisor, source_rate // divisor,
                         window=('kaiser', 10.0))


def read_pcm(path: Path) -> tuple[np.ndarray, int]:
    # wave.readframes alone silently returns short data for truncated RIFF files.
    # Check chunk boundaries before decoding either packed PCM16 or PCM24.
    size = path.stat().st_size
    with path.open('rb') as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b'RIFF' or header[8:] != b'WAVE':
            raise ValueError('expected RIFF PCM WAV')
        end = struct.unpack('<I', header[4:8])[0] + 8
        if end != size:
            raise ValueError('truncated WAV or inconsistent RIFF length')
        position, data_chunks, format_chunks = 12, [], 0
        while position < end:
            stream.seek(position)
            chunk = stream.read(8)
            if len(chunk) != 8:
                raise ValueError('truncated WAV chunk header')
            length = struct.unpack('<I', chunk[4:])[0]
            if position + 8 + length > end:
                raise ValueError('truncated WAV chunk payload')
            if chunk[:4] == b'data':
                data_chunks.append(length)
            elif chunk[:4] == b'fmt ':
                format_chunks += 1
                if length < 16:
                    raise ValueError('truncated WAV format chunk')
                encoding, channels, sample_rate, byte_rate, block_align, bits = struct.unpack(
                    '<HHIIHH', stream.read(16))
                if (bits not in (16, 24) or channels != 1 or
                        block_align != channels * (bits // 8) or
                        byte_rate != sample_rate * block_align):
                    raise ValueError('invalid mono PCM16/PCM24 WAV format metadata')
                if encoding not in (1, 0xfffe):
                    raise ValueError('WAV format must be integer PCM')
                if encoding == 0xfffe:
                    extension = stream.read(min(length - 16, 24))
                    pcm_guid = b'\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71'
                    if (len(extension) != 24 or struct.unpack('<H', extension[:2])[0] < 22 or
                            struct.unpack('<H', extension[2:4])[0] != bits or
                            extension[8:] != pcm_guid):
                        raise ValueError('unsupported extensible WAV: require full-width integer PCM')
            position += 8 + length + (length % 2)
        if len(data_chunks) != 1 or format_chunks != 1:
            raise ValueError('WAV must contain exactly one format and one audio data chunk')
    with wave.open(str(path), 'rb') as audio:
        width = audio.getsampwidth()
        if audio.getnchannels() != 1 or width not in (2, 3) or audio.getcomptype() != 'NONE':
            raise ValueError('reviewed input must be mono PCM16/PCM24 WAV')
        if data_chunks[0] % width:
            raise ValueError('truncated WAV sample frame')
        rate = audio.getframerate()
        if not 8000 <= rate <= 192000:
            raise ValueError('unsupported sample rate')
        frames = audio.getnframes()
        if not frames or frames / rate > 30:
            raise ValueError('empty or extreme WAV duration')
        raw = audio.readframes(frames)
        if len(raw) != frames * width:
            raise ValueError('truncated WAV samples')
        if width == 2:
            values = np.frombuffer(raw, dtype='<i2').astype(np.float64) / 32768
        else:
            packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            signed = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
            signed = (signed ^ 0x800000) - 0x800000
            values = signed.astype(np.float64) / 8388608
        return values, rate


def encode_pcm16(values: np.ndarray) -> bytes:
    if not np.all(np.isfinite(values)) or np.max(np.abs(values), initial=0) >= 1:
        raise ValueError('non-finite/clipped audio; review instead of silently limiting')
    quantized = np.round(values * 32768)
    # A float below +1 can still round to +32768. Casting that value directly
    # to int16 wraps to -32768 and creates an audible full-scale discontinuity.
    if np.min(quantized, initial=0) < -32768 or np.max(quantized, initial=0) > 32767:
        raise ValueError('clipped audio after PCM16 quantization; review instead of wrapping')
    return quantized.astype('<i2').tobytes()


def write_pcm(path: Path, values: np.ndarray, rate: int) -> None:
    encoded = encode_pcm16(values)
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(encoded)
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
        for field in ('source_master_sha256', 'processed_master_sha256'):
            if not isinstance(row.get(field), str) or not re.fullmatch(r'[0-9a-f]{64}', row[field]):
                raise ValueError(f'{identifier}: processed-master provenance missing: {field}')
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
    parser.add_argument('--input-schema', choices=['processed-single-master', 'aegis.audio_collection.v1'],
                        default='processed-single-master')
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--provenance', type=Path,
                        help='optional JSON mapping of real master/manifest paths and expected hashes')
    parser.add_argument('--split-plan', type=Path,
                        help='explicit SHA256-bound collection group plan; final test and original validation stay fixed')
    args = parser.parse_args()
    os.umask(0o077)
    if args.input_schema == 'aegis.audio_collection.v1':
        from .audio_collection import prepare_collection
        report = prepare_collection(args.manifest, args.output,
                                    args.dataset_root or args.manifest.parent, args.provenance, args.split_plan)
    else:
        if args.dataset_root or args.provenance or args.split_plan:
            parser.error('dataset-root/provenance/split-plan apply only to aegis.audio_collection.v1')
        report = prepare(args.manifest, args.output)
    print(json.dumps(report, indent=2))
    if report.get('status') == 'blocked':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
