#!/usr/bin/env python3
"""Read-only corpus audit; private receipts and a conservative source split.

Run with the existing F5 environment (numpy, soundfile, safetensors). No torch,
GPU, downloads, extraction, waveform conversion, or changes to source files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
import unicodedata


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def canonical(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold()))


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, mode=0o700, exist_ok=True)
    import numpy as np
    import soundfile as sf
    started = time.monotonic()
    rows = load_rows(args.source / 'dataset.jsonl')
    prepared = {r['id']: r for split in ('train', 'validation', 'test')
                for r in load_rows(args.prepared / f'{split}.jsonl')}
    groups = {key: defaultdict(set) for key in ('group', 'audio', 'text', 'source_master', 'source_utterance')}
    records, errors = [], []
    for index, row in enumerate(rows):
        identifier = row['id']
        source = args.source / row['audio_filepath']
        p = prepared[identifier]
        digest = sha256(source)
        checks = {'source_hash': digest == row['audio_sha256'],
                  'source_text_file': (args.source / 'texts' / f'{identifier}.txt').read_text(encoding='utf-8-sig').strip() == row['text'].strip(),
                  'prepared_source_hash': p['source_audio_sha256'] == digest,
                  'prepared_original_text': p['original_text'] == row['text'],
                  'prepared_hash': sha256(p['audio_filepath']) == p['sha256'],
                  'mel_hash': sha256(p['mel_filepath']) == p['mel_sha256']}
        info = sf.info(source)
        audio, rate = sf.read(source, dtype='float32', always_2d=True)
        mono = audio.mean(axis=1)
        frame = max(1, round(rate * .01))
        padded = np.pad(mono, (0, (-len(mono)) % frame))
        rms10 = np.sqrt(np.mean(padded.reshape(-1, frame).astype('float64') ** 2, axis=1))
        active = np.flatnonzero(rms10 > 10 ** (-50 / 20))
        clipped = int(np.count_nonzero(np.abs(audio) >= 1))
        near = int(np.count_nonzero(np.abs(audio) >= .999))
        checks.update(format=rate == 48000 and info.channels == 1 and info.subtype == 'PCM_24',
                      frames=info.frames == row['frames'], finite=bool(np.isfinite(audio).all()),
                      duration=abs(info.duration - row['duration']) < 1 / rate,
                      nonempty_activity=bool(active.size), unclipped=clipped == 0)
        if not all(checks.values()):
            errors.append({'id': identifier, 'failed': [k for k, v in checks.items() if not v]})
        src = row.get('source_clip', {})
        keys = dict(group=row['group_id'], audio=digest, text=canonical(row['text']),
                    source_master=src.get('source_master', 'unknown'), source_utterance=src.get('id', identifier))
        for kind, value in keys.items():
            groups[kind][value].add(row['split'])
        records.append(dict(id=identifier, historical_split=row['split'], group_id=row['group_id'],
                            source_master=keys['source_master'], source_utterance=keys['source_utterance'],
                            source_hash=digest, prepared_hash=p['sha256'], mel_hash=p['mel_sha256'],
                            seconds=info.duration, sample_rate=rate, subtype=info.subtype,
                            peak=float(np.max(np.abs(audio))), rms=float(np.sqrt(np.mean(audio.astype('float64') ** 2))),
                            clipped_samples=clipped, near_clip_samples=near,
                            silence_fraction_10ms=float(np.mean(rms10 <= 10 ** (-50 / 20))),
                            leading_silence_seconds=float(active[0] * .01) if active.size else info.duration,
                            trailing_silence_seconds=float(max(0, info.duration - (active[-1] + 1) * .01)) if active.size else info.duration,
                            checks=checks))
        if (index + 1) % 100 == 0:
            print(json.dumps({'audited': index + 1, 'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
    # Individual source IDs and semantic groups are the primary split units.
    # Masters are subsequent editing assemblies, not original continuous takes.
    # Preserve historical partitions for comparisons; export hashes, not a new
    # derivative audio tree. Transcripts remain private in these manifests.
    historical_dir = args.output / 'historical-preserved'
    historical_dir.mkdir(mode=0o700, exist_ok=True)
    for split in ('train', 'validation', 'test'):
        with (historical_dir / f'{split}.jsonl').open('w') as stream:
            for row in rows:
                if row['split'] == split:
                    stream.write(json.dumps({**row, 'audio_filepath': str(args.source / row['audio_filepath'])}, ensure_ascii=False) + '\n')
    wordsets = {split: set(word for row in rows if row['split'] == split for word in canonical(row['text']).split())
                for split in ('train', 'validation', 'test')}
    lexical = {}
    for split in ('validation', 'test'):
        unseen = wordsets[split] - wordsets['train']
        lexical[split] = dict(unique_words=len(wordsets[split]), words_absent_from_train=len(unseen),
                              utterances_with_unseen_words=sum(bool(set(canonical(r['text']).split()) & unseen)
                                                               for r in rows if r['split'] == split))
    interval_overlaps = []
    intervals = defaultdict(list)
    for row in rows:
        src = row['source_clip']
        intervals[src['source_master']].append((src['exported_start_frame_in_processed_master'],
                                               src['exported_end_frame_in_processed_master'], row['id'], row['split']))
    for master, clips in intervals.items():
        clips.sort()
        for previous, current in zip(clips, clips[1:]):
            if previous[1] > current[0]:
                interval_overlaps.append(dict(master=master, ids=[previous[2], current[2]],
                                              cross_split=previous[3] != current[3]))
    # An optional more conservative sensitivity plan blocks assembled masters.
    # Conservatively block the six assembled masters. Quarantine entire groups
    # touching a split boundary; semantic variants stay together without joining
    # every master into a single connected component.
    masters = sorted(groups['source_master'])
    if len(masters) != 6:
        raise ValueError('conservative plan expects the six documented masters')
    master_splits = {m: ('train' if i < 4 else 'validation' if i == 4 else 'test') for i, m in enumerate(masters)}
    proposed_groups = defaultdict(set)
    for row in records:
        proposed_groups[row['group_id']].add(master_splits[row['source_master']])
    quarantined = {g for g, splits in proposed_groups.items() if len(splits) > 1}
    plan = []
    for row in rows:
        split = 'quarantine' if row['group_id'] in quarantined else master_splits[row['source_clip']['source_master']]
        plan.append(dict(id=row['id'], split=split, group_id=row['group_id'], source_master=row['source_clip']['source_master'],
                         source_sha256=row['audio_sha256'], audio_filepath=str(args.source / row['audio_filepath']),
                         text=row['text'], duration=row['duration']))
    for split in ('train', 'validation', 'test', 'quarantine'):
        with (args.output / f'{split}.jsonl').open('w') as stream:
            for row in plan:
                if row['split'] == split:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    # Identity anchors come only from validation in this plan; not optimized or
    # chosen by model output. Fixed duration order, evenly spaced coverage.
    anchors_pool = sorted((r for r in plan if r['split'] == 'validation' and 4 <= r['duration'] <= 18), key=lambda r: (r['duration'], r['id']))
    anchors = [anchors_pool[i * (len(anchors_pool) - 1) // 2] for i in range(3)]
    write_json(args.output / 'master-blocked-identity-anchors.private.json', anchors)
    selected_pool = sorted((r for r in rows if r['split'] == 'validation' and 4 <= r['duration'] <= 18), key=lambda r: (r['duration'], r['id']))
    selected_anchors = [{**selected_pool[i * (len(selected_pool) - 1) // 2],
                         'audio_filepath': str(args.source / selected_pool[i * (len(selected_pool) - 1) // 2]['audio_filepath'])}
                        for i in range(3)]
    write_json(args.output / 'identity-anchors.private.json', selected_anchors)
    duplicate_counts = {kind: sum(len(splits) > 1 for splits in mapping.values()) for kind, mapping in groups.items()}
    summary = dict(schema='aegis.v2.data-audit.v1', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   source_manifest_sha256=sha256(args.source / 'dataset.jsonl'),
                   source_count=len(rows), errors=errors,
                   historical_split_counts=dict(Counter(r['split'] for r in rows)),
                   historical_split_seconds={s: sum(r['seconds'] for r in records if r['historical_split'] == s) for s in ('train','validation','test')},
                   formats=dict(Counter(f"{r['sample_rate']}Hz/{r['subtype']}" for r in records)),
                   duration_seconds=sum(r['seconds'] for r in records),
                   clipped_samples=sum(r['clipped_samples'] for r in records),
                   near_clip_samples=sum(r['near_clip_samples'] for r in records),
                   cross_split_key_counts=duplicate_counts,
                   original_source_intervals_overlaps=interval_overlaps,
                   lexical_holdout=lexical,
                   selected_partition_policy='Preserve original individual source utterance ID and semantic groups. The six masters are documented subsequent assemblies, not six original recording sessions.',
                   selected_identity_anchor_ids=[r['id'] for r in selected_anchors],
                   correspondence_scope='Exact manifest/TXT/prepared original_text and hashes. No independent forced alignment, full-corpus ASR or auditory claim.',
                   exact_duplicate_groups=sum(len(v) > 1 for v in [[r for r in records if r['source_hash'] == k] for k in groups['audio']]),
                   silence_policy='10ms RMS <= -50dBFS, diagnostic; no trimming/VAD/word alignment',
                   silence_fraction_duration_weighted=sum(r['silence_fraction_10ms'] * r['seconds'] for r in records) / sum(r['seconds'] for r in records),
                   leading_silence_quantiles_seconds={str(q): float(np.quantile([r['leading_silence_seconds'] for r in records], q)) for q in (.0,.5,.95,1.)},
                   trailing_silence_quantiles_seconds={str(q): float(np.quantile([r['trailing_silence_seconds'] for r in records], q)) for q in (.0,.5,.95,1.)},
                   peak_quantiles={str(q): float(np.quantile([r['peak'] for r in records], q)) for q in (.0,.5,.95,1.)},
                   proposed_plan=dict(policy='masters 1-4 train, 5 validation, 6 test; quarantine whole boundary-spanning groups',
                                      counts=dict(Counter(r['split'] for r in plan)),
                                      seconds={s: sum(r['duration'] for r in plan if r['split'] == s) for s in ('train','validation','test','quarantine')},
                                      master_assignments=master_splits, quarantine_groups=sorted(quarantined),
                                      anchor_ids=[r['id'] for r in anchors],
                                      caveat='Historical F5 already saw items now in proposed holdouts; this plan is for new candidates, not a claim of unseen F5 loss.'),
                   versions=dict(numpy=np.__version__, soundfile=sf.__version__), elapsed_seconds=time.monotonic() - started)
    if args.archive:
        summary['archive'] = dict(path=str(args.archive), bytes=args.archive.stat().st_size, sha256=sha256(args.archive))
    if args.checkpoint:
        from safetensors import safe_open
        tensors, finite = 0, True
        with safe_open(args.checkpoint, framework='numpy', device='cpu') as f:
            shapes = {}
            for key in f.keys():
                tensor = f.get_tensor(key)
                shapes[key] = list(tensor.shape)
                tensors += 1
                finite = finite and bool(np.isfinite(tensor).all())
        summary['checkpoint'] = dict(path=str(args.checkpoint), bytes=args.checkpoint.stat().st_size,
                                     sha256=sha256(args.checkpoint), tensors=tensors, all_finite=finite,
                                     tensor_shapes_sha256=hashlib.sha256(json.dumps(shapes, sort_keys=True).encode()).hexdigest(),
                                     scope='All safetensors tensors read on CPU, shapes and finiteness; strict model/inference loading is a separate runtime check.')
    write_json(args.output / 'records.private.json', records)
    write_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
