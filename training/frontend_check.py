#!/usr/bin/env python3
"""Check the reviewed corpus against the exact PT-BR phoneme inventory (no model)."""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

from .prepare_dataset import private_destination, sha256


def check_inventory(rows, config, phonemize, encode_sentence=None):
    if config.get('espeak', {}).get('voice') != 'pt-br':
        raise ValueError('frontend must explicitly be pt-br')
    mapping = config['phoneme_id_map']
    counts, unknown, phoneme_ids = Counter(), {}, {}
    for row in rows:
        sentences = phonemize(row['text'])
        symbols = [p for sentence in sentences for p in sentence]
        counts.update(symbols)
        missing = sorted({p for p in symbols if p not in mapping})
        if missing:
            unknown[row['id']] = missing
        elif encode_sentence is not None:
            phoneme_ids[row['id']] = list(itertools.chain.from_iterable(
                encode_sentence(sentence, mapping) for sentence in sentences))
    report = {'inventory_size': len(mapping), 'observed_symbols': dict(sorted(counts.items())),
            'unknown_by_id': unknown, 'passed': not unknown,
            'pronunciation': 'symbol coverage only; owner review still required'}
    if encode_sentence is not None:
        report['phoneme_ids_by_id'] = phoneme_ids
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--voice-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from piper.phonemize_espeak import EspeakPhonemizer, ESPEAK_DATA_DIR
    from piper.phoneme_ids import phonemes_to_ids
    frontend = EspeakPhonemizer(ESPEAK_DATA_DIR)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    config = json.loads(args.voice_config.read_text())
    report = check_inventory(rows, config,
        lambda text: frontend.phonemize('pt-br', text, vowel_clusters=None),
        lambda sentence, mapping: phonemes_to_ids(sentence, id_map=mapping))
    report.update(manifest_sha256=sha256(args.manifest),
                  voice_config_sha256=sha256(args.voice_config),
                  voice_config_path=str(args.voice_config.resolve()))
    private_destination(args.output.parent)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    args.output.chmod(0o600)
    if not report['passed']:
        raise SystemExit('unknown phonemes: review frontend; never drop them silently')


if __name__ == '__main__':
    main()
