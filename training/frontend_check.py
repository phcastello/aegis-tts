#!/usr/bin/env python3
"""Check the reviewed corpus against the exact PT-BR phoneme inventory (no model)."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def check_inventory(rows, config, phonemize):
    if config.get('espeak', {}).get('voice') != 'pt-br':
        raise ValueError('frontend must explicitly be pt-br')
    mapping = config['phoneme_id_map']
    counts, unknown = Counter(), {}
    for row in rows:
        symbols = [p for sentence in phonemize(row['text']) for p in sentence]
        counts.update(symbols)
        missing = sorted({p for p in symbols if p not in mapping})
        if missing:
            unknown[row['id']] = missing
    return {'inventory_size': len(mapping), 'observed_symbols': dict(sorted(counts.items())),
            'unknown_by_id': unknown, 'passed': not unknown,
            'pronunciation': 'symbol coverage only; owner review still required'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--voice-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from piper.phonemize_espeak import EspeakPhonemizer, ESPEAK_DATA_DIR
    frontend = EspeakPhonemizer(ESPEAK_DATA_DIR)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    config = json.loads(args.voice_config.read_text())
    report = check_inventory(rows, config, lambda text: frontend.phonemize('pt-br', text))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise SystemExit('unknown phonemes: review frontend; never drop them silently')


if __name__ == '__main__':
    main()
