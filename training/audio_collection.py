"""Explicit adapter for the owner's Windows aegis.audio_collection.v1 JSONL.

This validates structure and audio; it never grants auditory or contractual
approval. The legacy processed-single-master importer keeps its own gates.
See docs/audio-collection-v1.md for accepted field aliases and provenance input.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re
import tempfile
import unicodedata
import wave

import numpy as np

from .prepare_dataset import (RATE, REPOSITORY, private_destination, read_pcm,
                              resample_pcm, sha256, write_pcm)

SCHEMA = 'aegis.audio_collection.v1'
SPLITS = ('train', 'validation', 'test')
# Every alias is explicit; contradictory aliases always fail.
ALIASES = {
    'audio_filepath': ('audio_filepath', 'audio_path', 'wav_path', 'path', 'audio.path'),
    'duration': ('duration', 'duration_seconds', 'duration_sec', 'audio.duration_seconds'),
    'sample_rate': ('sample_rate', 'sample_rate_hz', 'audio.sample_rate'),
    'channels': ('channels', 'num_channels', 'audio.channels'),
    'sha256': ('sha256', 'audio_sha256', 'wav_sha256', 'audio.sha256'),
    'text': ('text', 'transcript', 'transcription'),
    'category': ('category', 'text_category'),
}


def field(row: dict, name: str, *, required: bool = True):
    found = []
    for alias in ALIASES.get(name, (name,)):
        value = row
        for part in alias.split('.'):
            value = value.get(part) if isinstance(value, dict) else None
        if value is not None:
            found.append(value)
    if not found:
        if required:
            raise ValueError(f'missing required field: {name}; accepted aliases: {ALIASES.get(name, (name,))}')
        return None
    if any(value != found[0] for value in found):
        raise ValueError(f'contradictory aliases for {name}')
    return found[0]


def local_path(root: Path, value) -> Path:
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('invalid relative audio/artifact path')
    windows = PureWindowsPath(value)
    if windows.is_absolute() or windows.drive or value.startswith(('/', '\\')):
        raise ValueError('absolute paths are forbidden; paths must be relative to dataset root')
    path = (root / value.replace('\\', '/')).resolve()
    if root not in path.parents:
        raise ValueError('audio/artifact path escapes dataset root (including symlinks)')
    return path


def normalized_text(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold()))


def dump(path: Path, data, *, lines=False):
    content = (''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in data)
               if lines else json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    path.write_text(content, encoding='utf-8')
    path.chmod(0o600)


def provenance_report(root: Path, provenance: Path | None) -> dict:
    report = {'master_manifest_status': 'unverified', 'artifacts': [],
              'note': 'No missing master hashes are inferred from clip hashes.'}
    if provenance is None:
        return report
    spec = json.loads(provenance.read_text(encoding='utf-8-sig'))
    if not isinstance(spec, dict) or not isinstance(spec.get('artifacts'), list):
        raise ValueError('provenance must contain artifacts: [{role,path,sha256?}]')
    report['input_sha256'] = sha256(provenance)
    for item in spec['artifacts']:
        if item.get('role') not in {'manifest', 'source_master', 'processed_master'}:
            raise ValueError('provenance artifact role must be manifest/source_master/processed_master')
        path = local_path(root, item['path'])
        artifact = {'role': item['role'], 'path': item['path'], 'status': 'unverified'}
        if path.is_file():
            digest = sha256(path)
            expected = item.get('sha256')
            if expected is not None and (not re.fullmatch('[0-9a-f]{64}', str(expected)) or expected != digest):
                raise ValueError('master/manifest provenance SHA256 mismatch')
            artifact.update(sha256=digest, bytes=path.stat().st_size,
                            status='verified' if expected else 'computed-no-prior-hash')
        else:
            artifact['reason'] = 'artifact absent; no hash fabricated'
        report['artifacts'].append(artifact)
    roles = {a['role'] for a in report['artifacts'] if a['status'] != 'unverified'}
    if {'manifest', 'source_master', 'processed_master'} <= roles:
        report['master_manifest_status'] = 'files-hashed-linkage-requires-review'
    return report


def check_companion(supplied: list[dict], original: list[dict], root: Path, split: str) -> None:
    """Check either full records or the Windows path/text/duration export.

    Reduced manifests intentionally omit IDs/groups: the primary JSONL remains
    their only source. Resolve a unique canonical path back to that primary
    record, never guess an ID from a filename or infer a grouping.
    """
    by_id = {r['id']: r for r in original}
    by_path = defaultdict(list)
    for row in original:
        by_path[local_path(root, field(row, 'audio_filepath'))].append(row)
    seen = set()
    for row in supplied:
        if not isinstance(row, dict):
            raise ValueError('companion record must be an object')
        path_value = field(row, 'audio_filepath', required=False)
        if 'id' in row:
            primary = by_id.get(row['id'])
            if primary is None:
                raise ValueError('companion ID is absent from primary partition')
        else:
            if path_value is None:
                raise ValueError('companion needs an ID or relative audio path')
            matches = by_path[local_path(root, path_value)]
            if len(matches) != 1:
                raise ValueError('companion audio path does not identify one primary record')
            primary = matches[0]
            # This is the explicit reduced Windows export, not an untyped
            # arbitrary record: duration and text must still be present.
            duration = field(row, 'duration')
            if (isinstance(duration, bool) or not isinstance(duration, (float, int)) or
                    not math.isfinite(duration)):
                raise ValueError('companion duration must be a finite number')
        identifier = primary['id']
        if identifier in seen:
            raise ValueError('duplicate primary record in companion partition')
        seen.add(identifier)
        if row.get('split', split) != split or field(row, 'text') != field(primary, 'text'):
            raise ValueError('companion text/split differs from primary record')
        if path_value is not None and local_path(root, path_value) != local_path(root, field(primary, 'audio_filepath')):
            raise ValueError('companion audio path differs from primary record')
        for name in ('duration', 'sample_rate', 'channels', 'sha256', 'category', 'group_id'):
            value = field(row, name, required=False)
            if value is not None and value != field(primary, name, required=False):
                raise ValueError(f'companion {name} differs from primary record')
    if seen != set(by_id) or len(supplied) != len(original):
        raise ValueError('companion count/identity set differs from primary partition')


def assignment_sha256(assignments: dict[str, str]) -> str:
    encoded = json.dumps(assignments, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def validated_split_plan(path: Path, source_hash: str, rows: list[dict]) -> tuple[dict, dict]:
    """Read an explicit, hash-bound plan; never rebalance by inference."""
    plan = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(plan, dict) or plan.get('source_manifest_sha256') != source_hash:
        raise ValueError('split plan is not bound to the exact primary manifest SHA256')
    assignments = plan.get('assignments')
    if (not isinstance(assignments, dict) or
            any(not isinstance(k, str) or v not in SPLITS for k, v in assignments.items())):
        raise ValueError('split plan assignments must map IDs to valid partitions')
    digest = assignment_sha256(assignments)
    if plan.get('split_assignment_sha256') != digest:
        raise ValueError('split plan assignment SHA256 mismatch')
    identifiers = [row.get('id') for row in rows]
    if (any(not isinstance(identifier, str) for identifier in identifiers) or
            len(set(identifiers)) != len(identifiers) or set(identifiers) != set(assignments)):
        raise ValueError('split plan must cover every unique primary ID exactly once')
    original_groups, new_groups = defaultdict(set), defaultdict(set)
    source_assignments = {}
    for row in rows:
        group, old = row.get('group_id'), row.get('split')
        if not isinstance(group, str) or not group.strip() or old not in SPLITS:
            raise ValueError('split plan requires valid source groups and partitions')
        new = assignments[row['id']]
        # Keep the final test AND the original validation immutable. This
        # explicit plan only expands validation with entire training groups.
        if (old == 'test') != (new == 'test') or (old != 'train' and new != old):
            raise ValueError('split plan must preserve original test/validation assignments')
        original_groups[group].add(old)
        new_groups[group].add(new)
        source_assignments[row['id']] = old
    if any(len(values) != 1 for values in original_groups.values()):
        raise ValueError('primary group already crosses partitions; plan cannot hide leakage')
    if any(len(values) != 1 for values in new_groups.values()):
        raise ValueError('split plan breaks a primary group across partitions')
    summary = {'path': str(path.resolve()), 'sha256': sha256(path),
               'source_manifest_sha256': source_hash,
               'source_assignment_sha256': assignment_sha256(source_assignments),
               'split_assignment_sha256': digest,
               'rule': plan.get('rule', 'explicit complete assignments'),
               'seed': plan.get('seed'),
               'original_counts': dict(Counter(source_assignments.values())),
               'planned_counts': dict(Counter(assignments.values())),
               'moved_ids': [row['id'] for row in rows if assignments[row['id']] != row['split']],
               'moved_group_ids': sorted({row['group_id'] for row in rows if assignments[row['id']] != row['split']}),
               'final_test_unchanged': True, 'original_validation_unchanged': True}
    return assignments, summary


def prepare_collection(manifest: Path, output: Path, dataset_root: Path,
                       provenance: Path | None = None,
                       split_plan: Path | None = None) -> dict:
    """Build a new immutable technical snapshot, with auditory review pending.

    Returns a blocked report if records fail, while preserving every rejection.
    Output must not exist; reruns always use a new versioned destination.
    """
    manifest, root, output = manifest.resolve(), dataset_root.resolve(), output.resolve()
    if not manifest.is_file() or not root.is_dir():
        raise FileNotFoundError('dataset manifest/root not available; transfer the real dataset first')
    if output.exists():
        raise ValueError('immutable prepared destination already exists; choose a new version')
    if output == REPOSITORY or REPOSITORY in output.parents:
        raise ValueError('private audio/manifests must be outside the repository')
    if output == root or root in output.parents:
        raise ValueError('prepared snapshot must be outside the original dataset root')
    parent = private_destination(output.parent)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=parent))
    staging.chmod(0o700)
    (staging / 'audio').mkdir(mode=0o700)
    rejected, candidates, structural_errors = [], [], []
    reasons = defaultdict(list)
    raw_rows, received = [], 0
    source_hash = sha256(manifest)
    for line_number, line in enumerate(manifest.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        received += 1
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('record must be a JSON object')
            json.dumps(row, allow_nan=False)
            raw_rows.append((line_number, row))
        except (ValueError, TypeError) as exc:
            rejected.append({'line': line_number, 'id': None, 'reasons': [str(exc)]})
    # Plan validation happens before decoding/conversion and cannot turn a
    # malformed or leaking primary dataset into an apparently clean split.
    assignments, split_plan_report = {}, None
    if split_plan is not None:
        if rejected:
            raise ValueError('cannot apply split plan to malformed primary JSONL records')
        assignments, split_plan_report = validated_split_plan(split_plan, source_hash, [r for _, r in raw_rows])
    ids = Counter(row.get('id') for _, row in raw_rows if isinstance(row.get('id'), str))
    for line_number, row in raw_rows:
        identifier = row.get('id')
        try:
            schema = row.get('schema', row.get('schema_version'))
            if 'schema' in row and 'schema_version' in row and row['schema'] != row['schema_version']:
                raise ValueError('contradictory schema/schema_version aliases')
            if schema is not None and schema != SCHEMA:
                raise ValueError(f'wrong schema: expected {SCHEMA}')
            if not isinstance(identifier, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', identifier):
                raise ValueError('invalid stable id')
            if ids[identifier] != 1:
                raise ValueError('duplicate stable id; all occurrences rejected')
            if row.get('split') not in SPLITS:
                raise ValueError('invalid predetermined split')
            group = row.get('group_id')
            if not isinstance(group, str) or not group.strip():
                raise ValueError('missing/invalid group_id; never infer groups from individual IDs')
            text = field(row, 'text')
            if not isinstance(text, str) or not text.strip() or any(c in text for c in '\r\n\x00|'):
                raise ValueError('invalid transcription')
            text = unicodedata.normalize('NFC', text).strip()
            category = field(row, 'category')
            if not isinstance(category, str) or not category.strip():
                raise ValueError('missing/invalid textual category')
            expected_duration = field(row, 'duration')
            rate, channels = field(row, 'sample_rate'), field(row, 'channels')
            if isinstance(expected_duration, bool) or not isinstance(expected_duration, (int, float)) or not math.isfinite(expected_duration):
                raise ValueError('duration must be a finite number')
            if (isinstance(rate, bool) or not isinstance(rate, int) or
                    isinstance(channels, bool) or not isinstance(channels, int) or channels != 1):
                raise ValueError('sample rate must be integer and channels must be mono')
            source = local_path(root, field(row, 'audio_filepath'))
            if not source.is_file():
                raise ValueError('WAV file missing')
            digest = sha256(source)
            expected_hash = field(row, 'sha256', required=False)
            if expected_hash is not None and (not isinstance(expected_hash, str) or not re.fullmatch('[0-9a-f]{64}', expected_hash) or expected_hash != digest):
                raise ValueError('input SHA256 mismatch or malformed declared hash')
            values, actual_rate = read_pcm(source)
            if sha256(source) != digest:
                raise ValueError('source WAV changed during import; transfer must finish first')
            with wave.open(str(source), 'rb') as audio:
                width = audio.getsampwidth()
            duration = len(values) / actual_rate
            if actual_rate != rate or abs(expected_duration - duration) > max(.002, 1 / rate):
                raise ValueError('sample rate/duration metadata differs from actual WAV')
            if not .2 <= duration <= 30:
                raise ValueError('empty or extreme clip duration')
            peak = float(np.max(np.abs(values), initial=0))
            rms = float(np.sqrt(np.mean(values ** 2)))
            rails = int(np.count_nonzero((values <= -1) | (values >= 1 - 1 / 2**(width * 8 - 1))))
            if peak < 1e-4 or rms < 1e-6:
                raise ValueError('silent/empty audio')
            if rails:
                raise ValueError(f'clipping: {rails} samples at PCM rails')
            converted = resample_pcm(values, actual_rate)
            target = staging / 'audio' / (identifier + '.wav')
            write_pcm(target, converted, RATE)  # refuses overshoot and quantization wraparound
            output_values, output_rate = read_pcm(target)
            if output_rate != RATE or abs(len(output_values) / RATE - duration) > 1 / RATE:
                raise ValueError('conversion changed duration unexpectedly')
            candidate = dict(id=identifier, group_id=group,
                split=assignments.get(identifier, row['split']), source_split=row['split'], category=category,
                text=text, audio_filepath=str(output / 'audio' / target.name),
                sample_rate=RATE, channels=1, duration=len(converted) / RATE, sha256=sha256(target),
                original_audio_filepath=str(source), original_audio_sha256=digest,
                source_hash_status='verified-against-declared' if expected_hash else 'computed-no-prior-hash',
                source_sample_rate=actual_rate, source_sample_width=width,
                source_duration=duration, source_peak=peak, source_rms=rms,
                peak=float(np.max(np.abs(output_values))), rms=float(np.sqrt(np.mean(output_values**2))),
                conversion_duration_delta_seconds=len(output_values) / RATE - duration,
                conversion_rms_delta=float(np.sqrt(np.mean(output_values**2))) - rms,
                quality_status='accepted', alignment_status='unverified', content_review='pending',
                schema=SCHEMA, source_record=row, source_line=line_number,
                provenance={'source_manifest_sha256': source_hash,
                            'master_manifest_status': 'unverified'})
            candidates.append(candidate)
        except (ValueError, OSError, EOFError, wave.Error, TypeError, KeyError) as exc:
            rejected.append({'line': line_number, 'id': identifier, 'reasons': [str(exc)]})
    # Reject both sides of leakage, including physically identical audio under new IDs.
    for key, label in [('group_id', 'group'), ('original_audio_sha256', 'identical audio'),
                       ('sha256', 'identical converted waveform')]:
        buckets = defaultdict(list)
        for row in candidates:
            buckets[row[key]].append(row)
        for bucket in buckets.values():
            if len({r['split'] for r in bucket}) > 1:
                for row in bucket:
                    reasons[row['id']].append(label + ' crosses splits')
    texts = defaultdict(list)
    for row in candidates:
        texts[normalized_text(row['text'])].append(row)
    for bucket in texts.values():
        if len({r['split'] for r in bucket}) > 1:
            for row in bucket:
                reasons[row['id']].append('duplicate normalized transcription crosses splits')
    records = []
    for row in candidates:
        if reasons[row['id']]:
            rejected.append({'line': row['source_line'], 'id': row['id'], 'reasons': reasons[row['id']]})
            (staging / 'audio' / (row['id'] + '.wav')).unlink()
        else:
            records.append(row)
    near_pairs, near_count = [], 0
    normalized = {row['id']: normalized_text(row['text']) for row in records}
    for index, first in enumerate(records):
        a = normalized[first['id']]
        for second in records[index + 1:]:
            if first['split'] == second['split'] or first['group_id'] == second['group_id']:
                continue
            b = normalized[second['id']]
            if min(len(a), len(b)) < 24 or min(len(a), len(b)) / max(len(a), len(b)) < .9:
                continue
            matcher = SequenceMatcher(None, a, b, autojunk=False)
            # quick_ratio is an upper bound, so this avoids expensive exact
            # comparisons without missing any pair above the same threshold.
            if matcher.quick_ratio() < .94:
                continue
            similarity = matcher.ratio()
            if similarity >= .94:
                near_count += 1
                if len(near_pairs) < 500:
                    near_pairs.append({'ids': [first['id'], second['id']], 'similarity': similarity,
                                       'reason': 'possible variant across splits; resolve before training'})
    try:
        source_provenance = provenance_report(root, provenance)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        source_provenance = {'master_manifest_status': 'unverified', 'error': str(exc)}
        structural_errors.append(str(exc))
    for row in records:
        row['provenance']['artifacts'] = source_provenance.get('artifacts', [])
        row['provenance']['master_manifest_status'] = source_provenance['master_manifest_status']
    # Companion split JSONL files are a consistency check, never the source of splits.
    for split in SPLITS:
        companion = root / (split + '.jsonl')
        if companion.is_file():
            try:
                supplied = [json.loads(line) for line in companion.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
                original = [r for _, r in raw_rows if r.get('split') == split]
                check_companion(supplied, original, root, split)
            except (ValueError, KeyError, TypeError) as exc:
                structural_errors.append(f'{split}.jsonl conflicts with primary dataset.jsonl: {exc}')
    trainval = [r for r in records if r['split'] != 'test']
    if sha256(manifest) != source_hash:
        structural_errors.append('primary manifest changed during import')
    for name, subset in [('dataset', records), ('trainval', trainval)] + [(s, [r for r in records if r['split'] == s]) for s in SPLITS]:
        dump(staging / (name + '.jsonl'), subset, lines=True)
    with (staging / 'metadata.csv').open('w', encoding='utf-8', newline='') as stream:
        # Upstream Piper v1.8.0 reads csv.reader(delimiter='|'). Standard CSV
        # escaping preserves literal quotes, including at transcript edges.
        csv.writer(stream, delimiter='|', lineterminator='\n').writerows(
            (Path(r['audio_filepath']).name, r['text']) for r in trainval)
    (staging / 'metadata.csv').chmod(0o600)
    counts = {s: sum(r['split'] == s for r in records) for s in SPLITS}
    if not counts['train'] or not counts['validation'] or not counts['test']:
        structural_errors.append('all three partitions require at least one technically valid example')
    category_sets = {s: sorted({r['category'] for r in records if r['split'] == s}) for s in SPLITS}
    all_categories = sorted({r['category'] for r in records})
    summary = {
        'schema': SCHEMA, 'status': 'blocked' if rejected or structural_errors or near_count else 'prepared-pending-review',
        'source_manifest_sha256': source_hash, 'received': received,
        'accepted': len(records), 'counts': counts, 'duration_seconds': sum(r['duration'] for r in records),
        'duration_seconds_by_split': {s: sum(r['duration'] for r in records if r['split'] == s) for s in SPLITS},
        'groups_by_split': {s: len({r['group_id'] for r in records if r['split'] == s}) for s in SPLITS},
        'source_sample_formats': dict(Counter(f"PCM{r['source_sample_width'] * 8}" for r in records)),
        'original_duration_seconds': sum(r['source_duration'] for r in records),
        'sample_rate': RATE, 'channels': 1, 'sample_format': 'PCM16',
        'dataset_sha256': sha256(staging / 'dataset.jsonl'), 'trainval_sha256': sha256(staging / 'trainval.jsonl'),
        'metadata_sha256': sha256(staging / 'metadata.csv'),
        'split_manifest_sha256': {s: sha256(staging / (s + '.jsonl')) for s in SPLITS},
        'rejected': rejected, 'structural_errors': structural_errors,
        'near_variant_pair_count': near_count, 'near_variant_pairs_first_500': near_pairs,
        'categories': all_categories, 'categories_by_split': category_sets,
        'missing_categories_by_split': {s: sorted(set(all_categories) - set(category_sets[s])) for s in SPLITS},
        'split_decision': ('apply explicit hash-bound group plan; preserve original test and validation; expand validation by whole training groups'
                           if split_plan_report else 'preserve original group assignments; no redistribution without explicit reviewed proposal; final test frozen'),
        'split_plan': split_plan_report,
        'provenance': source_provenance, 'content_review': 'pending', 'training_authorized': False,
        'normalization': 'none; PCM scaling only, Kaiser polyphase resampling, no speed/effect changes',
        'review_required': ['ASR/content audit', 'stratified owner listening and all suspicious examples',
                            'rights confirmation', 'cross-split near-variant resolution', 'decoder reconstruction'],
    }
    dump(staging / 'import-report.json', summary)
    # Refuse overwrites; immutable snapshot includes hashes and can be reverified.
    for path in staging.rglob('*'):
        if path.is_file():
            path.chmod(0o400)
    (staging / 'audio').chmod(0o500)
    staging.chmod(0o500)
    staging.rename(output)
    return summary
