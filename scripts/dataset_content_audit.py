#!/usr/bin/env python3
"""Offline optional content check. Uses an EXISTING local faster-whisper model.

No model download, GPU allocation or transcript rewriting. WER flags require
review; a successful ASR match alone does not approve perceptual voice quality.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gateway'))
from aegis_speech.normalization import SpeechTextNormalizer
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path,
                        help='existing converted faster-whisper/CTranslate2 directory; no downloads')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if not args.model_dir.is_dir() or not (args.model_dir / 'model.bin').is_file():
        parser.error('existing local converted ASR model is required; this tool never downloads one')
    directory = private_directory(args.output_dir)
    # Device fixed to CPU to avoid competing with the production GPU.
    from faster_whisper import WhisperModel
    model = WhisperModel(str(args.model_dir.resolve()), device='cpu', compute_type='int8',
                         cpu_threads=2, num_workers=1, local_files_only=True)
    source = args.dataset.read_bytes()
    rows = [json.loads(line) for line in source.decode().splitlines() if line.strip()]
    results = []
    for row in rows:
        if row['quality_status'] != 'accepted':
            continue
        audio = Path(row['audio_filepath'])
        if file_digest(audio) != row['sha256']:
            raise ValueError('dataset clip SHA256 mismatch: ' + row['id'])
        segments, _ = model.transcribe(str(audio), language='pt', beam_size=5,
                                       vad_filter=False, condition_on_previous_text=False)
        # No expected-text prompt: prompting with the transcript could conceal omissions.
        recognized = ' '.join(segment.text.strip() for segment in segments)
        results.append({'id': row['id'], 'expected_text': row['text'], 'recognized_text': recognized,
                        'audio_sha256': row['sha256'], **comparison(row['text'], recognized),
                        'content_review': 'pending'})
        atomic(directory / 'content-audit.jsonl', b''.join(canonical(item) + b'\n' for item in results))
    model_files = {str(path.relative_to(args.model_dir)): file_digest(path)
                   for path in args.model_dir.rglob('*') if path.is_file()}
    json_write(directory / 'content-audit-provenance.json', {
        'created_at': utc(), 'dataset_sha256': digest(source), 'model_directory': str(args.model_dir.resolve()),
        'faster_whisper_version': version('faster-whisper'), 'model_files_sha256': model_files,
        'language': 'pt', 'device': 'cpu', 'compute_type': 'int8', 'beam_size': 5,
        'examples_checked': len(results), 'review_required': [item['id'] for item in results if item['content_check'] != 'pass'],
        'note': 'ASR is fallible. Review flagged items, numbers, first consonants and a stratified sample. Never replace original text silently.'})
    print(json.dumps({'checked': len(results), 'review_required': sum(item['content_check'] != 'pass' for item in results)}))


if __name__ == '__main__':
    main()
