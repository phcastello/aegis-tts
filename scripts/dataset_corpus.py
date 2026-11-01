#!/usr/bin/env python3
"""Build the reviewable PT-BR corpus, with immutable IDs and grouped splits.

The full stage is a draft expansion, never an instruction to spend/generate audio.
All close variants stay in their original group's preassigned split.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Seven controlled discourse contexts. Full corpus requires text review before synthesis.
CONTEXTS = [
    ('base', '', ''),
    ('contexto', 'Para contextualizar a situação: ', ''),
    ('atencao', 'Observe este detalhe. ', ''),
    ('relato', 'O registro descreve o seguinte: ', ''),
    ('revisao', '', ' Vale revisar esse ponto antes de continuar.'),
    ('exemplo', 'Considere este exemplo: ', ''),
]


def build(source: Path) -> list[dict]:
    records, category, ordinal = [], None, 0
    for line in source.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('# '):
            category, ordinal = line[2:], 0
            continue
        ordinal += 1
        # ID tied to content, not line number; changing a text creates a new ID/group.
        group = 'ptbr-' + hashlib.sha256(line.encode()).hexdigest()[:12]
        bucket = int(hashlib.sha256(('aegis-split-v1:' + group).encode()).hexdigest()[:8], 16) % 100
        split = 'test' if bucket < 10 else 'validation' if bucket < 20 else 'train'
        for suffix, prefix, ending in CONTEXTS:
            records.append({'id': group + '-' + suffix, 'group': group, 'split': split,
                            'group_id': group, 'category': category, 'stage': 'pilot' if suffix == 'base' and (category == 'extensoes' or ordinal <= 10) else 'full',
                            'text': prefix + line + ending, 'text_review': 'draft'})
    return records


if __name__ == '__main__':
    records = build(ROOT / 'dataset/corpus_source.txt')
    target = ROOT / 'dataset/corpus.pt-BR.jsonl'
    target.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in records))
    for stage in ('pilot', 'full'):
        rows = [row for row in records if stage == 'full' or row['stage'] == 'pilot']
        words = sum(len(row['text'].split()) for row in rows)
        print(json.dumps({'stage': stage, 'examples': len(rows), 'words': words,
                          'speech_minutes_estimate_at_150wpm': round(words / 150, 1)}))
