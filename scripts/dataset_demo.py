#!/usr/bin/env python3
"""Create a tiny, clearly labelled synthetic tone master. Never speech/training data."""
import argparse
from pathlib import Path

import numpy as np
from dataset_pipeline import (atomic, assemble, audio_stats, canonical, digest, extract,
                              json_write, private_directory, request_identity, wav_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = private_directory(args.output_dir)
    rate = 8000
    config = {'provider': 'synthetic-fixture-no-api', 'base_url': None, 'model': 'tones', 'voice': None,
              'sample_rate': rate, 'separator_seconds': 2, 'bookend_seconds': 2}
    rows = []
    for index in range(6):
        row = {'id': f'synthetic-tone-{index:02}', 'group': f'synthetic-{index}',
               'group_id': f'synthetic-{index}', 'text': 'SYNTHETIC TONE; NOT SPEECH OR TRAINING DATA',
               'split': 'test', 'stage': 'pilot'}
        t = np.arange(round((1 + index * .1) * rate)) / rate
        values = .15 * np.sin(np.linspace(0, np.pi, len(t))) ** 2 * np.sin(2 * np.pi * (180 + 43 * index) * t)
        if index == 2:
            middle = len(values) // 2
            values = np.r_[values[:middle], np.zeros(round(1.3 * rate)), values[middle:]]
        data = wav_bytes(values, rate)
        hashed, payload = request_identity(row, config)
        path = directory / 'cache' / f"{row['id']}-{hashed[:20]}.wav"
        atomic(path, data)
        json_write(path.with_suffix('.json'), {**row, **audio_stats(values, rate),
                   'sha256': digest(data), 'request_hash': hashed, 'parameters': payload,
                   'provider': 'synthetic-fixture-no-api', 'content_review': 'rejected'})
        rows.append(row)
    atomic(directory / 'SYNTHETIC_NOT_FOR_TRAINING.txt',
           b'Tiny synthetic tone fixture only. No person, provider call, speech, Melodyne recipe or training approval.\n')
    summary = assemble(rows, config, directory)
    # Identity export exercises extraction, without pretending to apply a DAW effect.
    atomic(directory / 'master_processed.wav', (directory / 'master_raw.wav').read_bytes())
    output = private_directory(directory / 'processed')
    report = extract(directory / 'manifest.json', directory / 'master_processed.wav', output)
    # Synthetic examples must never be marked eligible for training.
    dataset = output / 'dataset.jsonl'
    import json
    records = [json.loads(line) for line in dataset.read_text().splitlines()]
    for record in records:
        record['content_review'] = 'rejected'
        record['synthetic_fixture'] = True
    atomic(dataset, b''.join(canonical(row) + b'\n' for row in records))
    print(json.dumps({'synthetic_only': True, 'summary': summary, 'validation_accepted': report['accepted'],
                      'training_eligible': 0}, indent=2))


if __name__ == '__main__':
    main()
