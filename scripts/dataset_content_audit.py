#!/usr/bin/env python3
"""Offline content check using an EXISTING local faster-whisper model.

CPU is the default. CUDA requires explicit authorization and an exclusive GPU
window with >=9000MiB free. No download or transcript rewriting. ASR matches
alone never approve perceptual voice quality.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / 'gateway'))
from aegis_speech.normalization import SpeechTextNormalizer
from training.prepare_dataset import read_pcm, resample_pcm
try:
    from .dataset_pipeline import atomic, canonical, digest, file_digest, json_write, private_directory, utc
except ImportError:  # Direct script invocation.
    from dataset_pipeline import atomic, canonical, digest, file_digest, json_write, private_directory, utc


def tokens(text: str) -> list[str]:
    # Keep original text in every report. This only normalizes the comparison.
    if not text.strip():
        return []
    normalized = SpeechTextNormalizer().normalize(text).normalized_text.lower()
    normalized = ''.join(char for char in unicodedata.normalize('NFD', normalized) if unicodedata.category(char) != 'Mn')
    return re.findall(r'[a-z0-9]+', normalized)


def comparison(expected: str, recognized: str) -> dict:
    a, b = tokens(expected), tokens(recognized)
    dp = list(range(len(b) + 1))
    for i, token in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[j - 1] + 1, dp[j] + 1, dp[j - 1] + (token != other)))
        dp = current
    edits = dp[-1]
    return {'word_error_rate': edits / max(1, len(a)), 'word_edits': edits,
            'expected_tokens': a, 'recognized_tokens': b,
            'content_check': 'pass' if edits == 0 else 'review_required'}


def asr_waveform(path: Path) -> tuple[np.ndarray, int]:
    """Decode our validated PCM WAV directly; bypass PyAV file decoding.

    faster-whisper's ndarray contract is mono float32 at 16 kHz. Scale PCM
    correctly and resample with the same quality filter as dataset conversion;
    never normalize loudness, trim margins or rewrite the source file.
    """
    values, source_rate = read_pcm(path)
    waveform = resample_pcm(values, source_rate, 16000).astype(np.float32)
    if not np.all(np.isfinite(waveform)):
        raise ValueError('nonfinite ASR input waveform')
    return waveform, source_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path,
                        help='existing converted faster-whisper/CTranslate2 directory; no downloads')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--owner-authorization', help='explicit authorization reference required for CUDA')
    parser.add_argument('--exclusive-gpu-window', help='exclusive operational window reference required for CUDA')
    parser.add_argument('--resume', action='store_true', help='reuse only records bound to the same dataset/model/settings')
    args = parser.parse_args()
    if not args.model_dir.is_dir() or not (args.model_dir / 'model.bin').is_file():
        parser.error('existing local converted ASR model is required; this tool never downloads one')
    if not 1 <= args.cpu_threads <= 16:
        parser.error('cpu-threads must be between 1 and 16')
    if args.device == 'cuda':
        if not args.owner_authorization or not args.exclusive_gpu_window:
            parser.error('CUDA ASR requires --owner-authorization and --exclusive-gpu-window')
        from training.train_piper import gpu_available
        gpu_available(minimum_free_mib=9000)
    compute_type = 'float16' if args.device == 'cuda' else 'int8'
    directory = private_directory(args.output_dir)
    source = args.dataset.read_bytes()
    rows = [json.loads(line) for line in source.decode().splitlines() if line.strip()]
    identifiers = [row['id'] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError('duplicate dataset IDs')
    model_files = {str(path.relative_to(args.model_dir)): file_digest(path)
                   for path in args.model_dir.rglob('*') if path.is_file()}
    settings = {'dataset_sha256': digest(source), 'model_directory': str(args.model_dir.resolve()),
                'faster_whisper_version': version('faster-whisper'), 'model_files_sha256': model_files,
                'language': 'pt', 'device': args.device, 'compute_type': compute_type, 'beam_size': 5,
                'vad_filter': False, 'condition_on_previous_text': False, 'cpu_threads': args.cpu_threads,
                'expected_text_prompt': False, 'input_format': 'mono numpy.float32 PCM-scaled waveform',
                'input_sample_rate': 16000,
                'input_decoder': 'training.prepare_dataset.read_pcm; RIFF PCM16/PCM24; no PyAV file decoding',
                'input_resampling': 'scipy.signal.resample_poly Kaiser beta=10; no loudness normalization or trimming'}
    if args.device == 'cuda':
        settings.update(owner_authorization=args.owner_authorization, exclusive_gpu_window=args.exclusive_gpu_window,
                        gpu_minimum_free_mib=9000, cuda_capacity_check_passed=True)
    audit_path, provenance_path = directory / 'content-audit.jsonl', directory / 'content-audit-provenance.json'
    results = []
    if audit_path.exists() or provenance_path.exists():
        if not args.resume or not (audit_path.exists() and provenance_path.exists()):
            raise ValueError('audit output exists; use a fresh directory or --resume with a complete bound pair')
        previous = json.loads(provenance_path.read_text())
        if any(previous.get(key) != value for key, value in settings.items()):
            raise ValueError('ASR resume dataset/model/settings mismatch')
        results = [json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()]
    by_id = {row['id']: row for row in rows}
    seen = set()
    for result in results:
        row = by_id.get(result['id'])
        if row is None or result['id'] in seen or result.get('audio_sha256') != row['sha256'] or result.get('expected_text') != row['text']:
            raise ValueError('ASR resume record binding mismatch')
        if file_digest(Path(row['audio_filepath'])) != row['sha256']:
            raise ValueError('ASR resume audio changed: ' + row['id'])
        seen.add(row['id'])
    def save(complete=False):
        atomic(audit_path, b''.join(canonical(item) + b'\n' for item in results))
        audio_seconds = sum(item.get('duration_seconds', 0) for item in results)
        wall_seconds = sum(item.get('wall_seconds', 0) for item in results)
        total_audio = sum(row.get('duration', row.get('duration_seconds', 0)) for row in rows
                          if row.get('quality_status') == 'accepted')
        json_write(provenance_path, {**settings, 'created_at': utc(), 'complete': complete,
            'examples_checked': len(results), 'expected_examples': sum(row.get('quality_status') == 'accepted' for row in rows),
            'review_required': [item['id'] for item in results if item['content_check'] != 'pass'],
            'skipped_technical_ids': [row['id'] for row in rows if row.get('quality_status') != 'accepted'],
            'progress': {'audio_seconds_checked': audio_seconds, 'transcription_wall_seconds': wall_seconds,
                         'estimated_remaining_seconds': max(0., total_audio - audio_seconds) * wall_seconds / audio_seconds
                         if audio_seconds else None,
                         'note': 'ASR progress projection from completed examples; excludes initial model load and is not a training ETA'},
            'note': 'ASR is fallible. Review flagged items, numbers, first consonants and a stratified sample. Never replace original text silently.'})
    save()
    # CUDA is opt-in and capacity-gated; this script never stops production.
    from faster_whisper import WhisperModel
    model = WhisperModel(str(args.model_dir.resolve()), device=args.device, compute_type=compute_type,
                         cpu_threads=args.cpu_threads, num_workers=1, local_files_only=True)
    for row in rows:
        if row['quality_status'] != 'accepted' or row['id'] in seen:
            continue
        audio = Path(row['audio_filepath'])
        if file_digest(audio) != row['sha256']:
            raise ValueError('dataset clip SHA256 mismatch: ' + row['id'])
        started = time.perf_counter()
        waveform, source_rate = asr_waveform(audio)
        segments, info = model.transcribe(waveform, language='pt', beam_size=5,
                                       vad_filter=False, condition_on_previous_text=False)
        # No expected-text prompt: prompting with the transcript could conceal omissions.
        segments = list(segments)
        recognized = ' '.join(segment.text.strip() for segment in segments)
        results.append({'id': row['id'], 'expected_text': row['text'], 'recognized_text': recognized,
                        'audio_sha256': row['sha256'], **comparison(row['text'], recognized),
                        'segments': [{'start': segment.start, 'end': segment.end, 'text': segment.text,
                                      'avg_logprob': segment.avg_logprob, 'no_speech_prob': segment.no_speech_prob}
                                     for segment in segments],
                        'duration_seconds': info.duration, 'wall_seconds': time.perf_counter() - started,
                        'source_sample_rate': source_rate, 'asr_input_sample_rate': 16000,
                        'asr_input_samples': len(waveform), 'asr_input_dtype': str(waveform.dtype),
                        'asr_input_sha256': digest(waveform.tobytes()),
                        'content_review': 'pending'})
        if file_digest(audio) != row['sha256'] or args.dataset.read_bytes() != source:
            raise ValueError('ASR input changed during audit')
        save()
    save(complete=True)
    print(json.dumps({'checked': len(results), 'review_required': sum(item['content_check'] != 'pass' for item in results)}))


if __name__ == '__main__':
    main()
