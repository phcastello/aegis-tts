"""Private acoustic/listening review for a prepared aegis.audio_collection.v1.

Plan is the default. No download, ASR, voice generation or training occurs here.
Signal outliers are suspicions, never proof of an audible defect or voice identity.
Original JSONL and WAVs remain unchanged; approvals live in a separate bound file.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from .prepare_dataset import REPOSITORY, read_pcm, sha256

SCHEMA = 'aegis.collection_review.v1'
COLLECTION = 'aegis.audio_collection.v1'


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate IDs in ' + path.name)
    return rows


def private_write(path: Path, value) -> None:
    path = path.resolve()
    if path == REPOSITORY or REPOSITORY in path.parents:
        raise ValueError('review artifacts must be outside the Git checkout')
    if path.exists():
        raise ValueError('refuse to overwrite an existing review artifact')
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Exclusive creation, including concurrent invocations.
    finally:
        os.unlink(temporary)


def acoustic_features(path: Path) -> tuple[dict, list[str]]:
    audio, rate = read_pcm(path)
    window = max(1, round(rate * .02))
    framed = audio[:len(audio) // window * window].reshape(-1, window)
    if not len(framed):
        raise ValueError('audio too short for acoustic review')
    envelope = np.sqrt(np.mean(framed ** 2, axis=1))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio), initial=0))
    threshold = max(.001, float(np.max(envelope)) * .04)
    active = np.flatnonzero(envelope > threshold)
    frame_seconds = window / rate
    leading = float(active[0] * frame_seconds) if len(active) else len(audio) / rate
    trailing = float((len(envelope) - active[-1] - 1) * frame_seconds) if len(active) else len(audio) / rate
    # Averaged magnitude spectra retain texture statistics without inventing a
    # perceptual score. They do not establish robotization or reverb quality.
    power = np.mean(np.abs(np.fft.rfft(framed * np.hanning(window), axis=1)) ** 2, axis=0)
    frequencies = np.fft.rfftfreq(window, 1 / rate)
    total = max(float(np.sum(power)), 1e-20)
    centroid = float(np.sum(power * frequencies) / total)
    high_band = float(np.sum(power[frequencies >= 4000]) / total)
    flatness = float(np.exp(np.mean(np.log(np.maximum(power, 1e-20)))) / max(np.mean(power), 1e-20))
    features = {'duration_seconds': len(audio) / rate, 'rms_dbfs': 20 * math.log10(max(rms, 1e-12)),
                'peak': peak, 'leading_quiet_seconds': leading, 'trailing_quiet_seconds': trailing,
                'spectral_centroid_hz': centroid, 'high_band_power_ratio': high_band,
                'spectral_flatness': flatness,
                'power_above_11025hz_ratio': float(np.sum(power[frequencies > 11025]) / total),
                'first_frame_rms': float(envelope[0]), 'last_frame_rms': float(envelope[-1]),
                'tail_rms_ratio': float(np.sqrt(np.mean(audio[-min(len(audio), round(.12 * rate)):] ** 2)) / max(rms, 1e-12))}
    reasons = []
    if peak >= 32767 / 32768:
        reasons.append('PCM rail reached; inspect clipping')
    if rms < 1e-6:
        reasons.append('silent or almost empty')
    if envelope[0] > max(.002, rms * .3):
        reasons.append('energy at first 20ms; inspect first consonant/preceding leakage')
    if envelope[-1] > max(.002, rms * .3):
        reasons.append('energy at last 20ms; inspect final consonant/reverb/following leakage')
    if leading > 1 or trailing > 1:
        reasons.append('quiet margin above one second; inspect duration and boundaries')
    return features, reasons


def load_prepared(prepared: Path) -> tuple[list[dict], dict, dict]:
    report_path = prepared / 'import-report.json'
    report = json.loads(report_path.read_text())
    if report.get('schema') != COLLECTION:
        raise ValueError('explicit collection review requires aegis.audio_collection.v1 import')
    rows = jsonl(prepared / 'dataset.jsonl')
    if not rows:
        raise ValueError('empty collection')
    trainval = jsonl(prepared / 'trainval.jsonl')
    if trainval != [row for row in rows if row['split'] != 'test']:
        raise ValueError('trainval is not the exact ordered train/validation projection')
    for name, key in [('dataset.jsonl', 'dataset_sha256'), ('trainval.jsonl', 'trainval_sha256'), ('metadata.csv', 'metadata_sha256')]:
        if sha256(prepared / name) != report.get(key):
            raise ValueError('prepared manifest hash mismatch: ' + name)
    groups = {}
    for row in rows:
        if row.get('schema') != COLLECTION or row.get('quality_status') != 'accepted' or row.get('synthetic_fixture'):
            raise ValueError('review cannot promote a quarantined/non-collection/synthetic record')
        if groups.setdefault(row['group_id'], row['split']) != row['split']:
            raise ValueError('group crosses partitions')
        if sha256(Path(row['audio_filepath'])) != row['sha256']:
            raise ValueError('prepared audio SHA256 mismatch: ' + row['id'])
        if row.get('content_review') == 'rejected':
            raise ValueError('explicitly rejected content must be removed through a new import')
    bindings = {'prepared_directory': str(prepared.resolve()),
                'dataset_sha256': sha256(prepared / 'dataset.jsonl'),
                'trainval_sha256': sha256(prepared / 'trainval.jsonl'),
                'metadata_sha256': sha256(prepared / 'metadata.csv'),
                'import_report_sha256': sha256(report_path),
                'source_manifest_sha256': report['source_manifest_sha256']}
    return rows, report, bindings


def audit_inputs(rows: list[dict], bindings: dict, audit: Path | None, provenance: Path | None):
    from scripts.dataset_content_audit import comparison
    if (audit is None) != (provenance is None):
        raise ValueError('content-audit and audit-provenance must be supplied together')
    if audit is None:
        return {}, dict(bindings), ['ASR audit missing']
    checks = jsonl(audit)
    by_id = {row['id']: row for row in rows}
    prov = json.loads(provenance.read_text())
    if prov.get('dataset_sha256') != bindings['dataset_sha256'] or prov.get('examples_checked') != len(checks):
        raise ValueError('ASR provenance dataset/count mismatch')
    if prov.get('device') not in ('cpu', 'cuda') or prov.get('language') != 'pt' or not prov.get('model_files_sha256'):
        raise ValueError('ASR provenance must identify a local Portuguese model and device by hashes')
    if prov['device'] == 'cuda' and (prov.get('compute_type') != 'float16' or
            not prov.get('owner_authorization') or not prov.get('exclusive_gpu_window') or
            prov.get('gpu_minimum_free_mib', 0) < 9000 or prov.get('cuda_capacity_check_passed') is not True):
        raise ValueError('CUDA ASR provenance requires authorization, exclusive GPU window and capacity guard')
    flags = []
    for check in checks:
        row = by_id.get(check['id'])
        if row is None or check.get('audio_sha256') != row['sha256'] or check.get('expected_text') != row['text']:
            raise ValueError('ASR text/audio binding mismatch')
        expected = comparison(row['text'], check.get('recognized_text', ''))
        if any(check.get(key) != expected[key] for key in ('word_error_rate', 'word_edits', 'content_check')):
            raise ValueError('ASR comparison inconsistent: ' + row['id'])
        if expected['content_check'] != 'pass':
            flags.append(row['id'])
    if set(flags) != set(prov.get('review_required', [])):
        raise ValueError('ASR flag inventory mismatch')
    missing = sorted(set(by_id) - {check['id'] for check in checks})
    blockers = ['ASR coverage incomplete: ' + ', '.join(missing)] if missing else []
    if prov.get('complete') is not True:
        blockers.append('ASR audit is incomplete')
    return {check['id']: check for check in checks}, {
        **bindings, 'content_audit': str(audit.resolve()), 'content_audit_sha256': sha256(audit),
        'audit_provenance': str(provenance.resolve()), 'audit_provenance_sha256': sha256(provenance)}, blockers


def create_plan(prepared: Path, audit: Path | None = None, provenance: Path | None = None) -> dict:
    rows, report, bindings = load_prepared(prepared)
    checks, bindings, blockers = audit_inputs(rows, bindings, audit, provenance)
    technical_blockers = []
    if report.get('rejected') or report.get('structural_errors'):
        technical_blockers.append('import has rejected examples or structural errors; create a resolved immutable snapshot')
    if report.get('status') == 'blocked' or report.get('rejected') or report.get('structural_errors') or report.get('near_variant_pair_count'):
        blockers.append('import blocked; resolve rejects/structural errors/cross-split variants and make a new immutable import')
    features, flags = {}, {}
    for row in rows:
        values, reasons = acoustic_features(Path(row['audio_filepath']))
        values['words_per_second'] = len(row['text'].split()) / values['duration_seconds']
        original = Path(row.get('original_audio_filepath', ''))
        if original.is_file():
            if sha256(original) != row.get('original_audio_sha256'):
                raise ValueError('original audio changed since import: ' + row['id'])
            source_values, _ = acoustic_features(original)
            values['source_power_above_target_nyquist_ratio'] = source_values['power_above_11025hz_ratio']
            values['source_rms_dbfs'] = source_values['rms_dbfs']
            values['conversion_rms_db_difference'] = values['rms_dbfs'] - source_values['rms_dbfs']
            if source_values['power_above_11025hz_ratio'] > .01:
                reasons.append('source has >1% power above target Nyquist; review robotic texture after conversion')
            if abs(values['conversion_rms_db_difference']) > .5:
                reasons.append('conversion RMS differs by >0.5dB; inspect spectrum and level')
        features[row['id']] = values
        check = checks.get(row['id'])
        if check and check['content_check'] != 'pass':
            reasons.append('ASR words differ from expected transcript')
        if check and any(segment.get('avg_logprob', 0) < -1 or segment.get('no_speech_prob', 0) > .6 for segment in check.get('segments', [])):
            reasons.append('ASR low-confidence segment')
        flags[row['id']] = reasons
    # Robust distribution outliers are useful for timbre, level, texture and
    # expressive exceptions. Keep all reasons and never relabel them as defects.
    keys = ('rms_dbfs', 'spectral_centroid_hz', 'high_band_power_ratio', 'spectral_flatness', 'tail_rms_ratio', 'words_per_second')
    if len(rows) >= 10:
        for key in keys:
            values = np.array([features[row['id']][key] for row in rows])
            center = float(np.median(values))
            scale = float(np.median(np.abs(values - center))) * 1.4826
            if scale < 1e-8:
                continue
            for row, value in zip(rows, values):
                if abs(value - center) > 6 * scale:
                    flags[row['id']].append('distribution outlier: ' + key)
    strata = {}
    sample_reasons = {}
    def sample(identifier, reason):
        sample_reasons.setdefault(identifier, []).append(reason)
    for row in rows:
        strata.setdefault((row['category'], row['split']), []).append(row['id'])
    for stratum, identifiers in sorted(strata.items()):
        chosen = min(identifiers, key=lambda identifier: hashlib.sha256((bindings['dataset_sha256'] + ':' + identifier).encode()).hexdigest())
        sample(chosen, 'category/split: ' + '/'.join(stratum))
    for split in ('train', 'validation', 'test'):
        partition = [row for row in rows if row['split'] == split]
        if not partition:
            continue
        for key in ('duration_seconds', 'rms_dbfs', 'spectral_centroid_hz', 'tail_rms_ratio'):
            for direction, chooser in (('minimum', min), ('maximum', max)):
                row = chooser(partition, key=lambda item: (features[item['id']][key], item['id']))
                sample(row['id'], split + ' ' + direction + ' ' + key)
        for label, pattern in [('numbers/times', r'\d'), ('acronyms/names', r'\b(?:[A-ZÁÉÍÓÚÇ]{2,}|[A-ZÁÉÍÓÚÇ][a-záéíóúç]+)\b'), ('questions', r'\?')]:
            matching = [row for row in partition if re.search(pattern, row['text'])]
            if matching:
                sample(min(matching, key=lambda row: row['id'])['id'], split + ' ' + label)
    suspicious = {identifier: reasons for identifier, reasons in flags.items() if reasons}
    queue_ids = set(suspicious) | set(sample_reasons)
    review_queue = [{'id': row['id'], 'category': row['category'], 'split': row['split'], 'text': row['text'],
                     'audio_filepath': row['audio_filepath'],
                     'original_audio_filepath': row.get('original_audio_filepath'),
                     'mandatory': row['id'] in suspicious,
                     'reasons': flags[row['id']] + sample_reasons.get(row['id'], [])}
                    for row in rows if row['id'] in queue_ids]
    template = {**bindings, 'schema': SCHEMA, 'reviewer': '', 'reviewed_at': '', 'approval_reference': '',
                'training_rights_confirmed': False, 'voice_rights_confirmed': False,
                'rights_evidence_reference': '', 'auditory_approval_scope': 'sample',
                'auditory_approved': False, 'batch_sample_approved': False,
                'perceptual_sample_approved': False, 'sampled_ids': sorted(sample_reasons),
                'suspicious_reviews': {}, 'reconstruction_effect_approved': False,
                'reconstruction_report': '', 'reconstruction_report_sha256': ''}
    return {'schema': SCHEMA, 'status': 'pending-owner-review', 'bindings': bindings,
            'count': len(rows), 'blockers': blockers, 'technical_blockers': technical_blockers,
            'suspicions': suspicious, 'mandatory_review_ids': sorted(suspicious),
            'suggested_sampled_ids': sorted(sample_reasons), 'sample_reasons': sample_reasons,
            'review_queue': review_queue,
            'strata': [{'category': category, 'split': split, 'examples': len(ids)} for (category, split), ids in sorted(strata.items())],
            'acoustic_features': features, 'owner_review_template': template,
            'checklist': ['transcript: omissions/additions, numbers/times/acronyms/names',
                          'first and last consonants; leakage between utterances',
                          'vocal identity, robotic texture and spectral character',
                          'reverb level/consistency, timbre differences, expressive exceptions',
                          'final test is for data quality only; never tune hyperparameters against it'],
            'note': 'Listen to the stratified sample and every suspicious clip. Metrics and ASR do not grant auditory approval. Originals and absent master hashes are never fabricated.'}


def _validate_owner(plan: dict, owner: dict, prepared: Path, phase: str) -> None:
    blockers = plan['technical_blockers'] if phase == 'benchmark' else plan['blockers']
    if blockers:
        raise ValueError('review blocked: ' + '; '.join(blockers))
    if any(owner.get(key) != value for key, value in plan['bindings'].items()) or owner.get('schema') != SCHEMA:
        raise ValueError('owner review bindings do not match current inputs')
    for key in ('reviewer', 'reviewed_at', 'approval_reference'):
        if not isinstance(owner.get(key), str) or not owner[key].strip():
            raise ValueError('explicit owner review field required: ' + key)
    reviewed_at = datetime.fromisoformat(owner['reviewed_at'].replace('Z', '+00:00'))
    if reviewed_at.tzinfo is None:
        raise ValueError('reviewed_at requires timezone')
    for key in ('training_rights_confirmed', 'voice_rights_confirmed'):
        if owner.get(key) is not True:
            raise ValueError('explicit owner confirmation required: ' + key)
    global_approval = owner.get('auditory_approval_scope') == 'collection'
    if global_approval:
        # A real owner may approve their complete collection without claiming
        # which individual clips they heard. Preserve that scope and statement;
        # never manufacture sampled IDs or retroactively approve new findings.
        if owner.get('auditory_approved') is not True:
            raise ValueError('explicit owner confirmation required: auditory_approved')
    else:
        for key in ('batch_sample_approved', 'perceptual_sample_approved'):
            if owner.get(key) is not True:
                raise ValueError('explicit owner confirmation required: ' + key)
    samples = owner.get('sampled_ids', [])
    if not isinstance(samples, list) or len(samples) != len(set(samples)):
        raise ValueError('sampled_ids must be a unique list when supplied')
    if not global_approval and not set(plan['suggested_sampled_ids']) <= set(samples):
        raise ValueError('owner must confirm every suggested listening sample')
    known = set(plan['acoustic_features'])
    if not set(samples) <= known:
        raise ValueError('owner sample includes unknown ID')
    if phase == 'benchmark':
        # Mechanical timing uses temporary updates on train examples only.
        # It grants no quality clearance, persists no trained weights and does
        # not require listening to a decoder reconstruction that does not exist.
        return
    decisions = owner.get('suspicious_reviews', {})
    if not isinstance(decisions, dict) or set(decisions) != set(plan['mandatory_review_ids']):
        raise ValueError('every suspicious example requires an explicit decision')
    for identifier, decision in decisions.items():
        if decision.get('decision') != 'approved' or not isinstance(decision.get('evidence'), str) or not decision['evidence'].strip():
            raise ValueError('unapproved suspicious example; reject through a new import: ' + identifier)
    if owner.get('reconstruction_effect_approved') is not True:
        raise ValueError('explicit owner confirmation required: reconstruction_effect_approved')
    path = Path(owner.get('reconstruction_report', ''))
    if not path.is_file() or sha256(path) != owner.get('reconstruction_report_sha256'):
        raise ValueError('reconstruction report file/hash required')
    reconstruction = json.loads(path.read_text())
    recipe = json.loads((REPOSITORY / 'training/recipe.json').read_text())
    if reconstruction.get('checkpoint_sha256') != recipe['checkpoint']['sha256']:
        raise ValueError('reconstruction checkpoint mismatch')
    rows = jsonl(prepared / 'trainval.jsonl')
    source_hashes = {row[key] for row in rows for key in ('sha256', 'original_audio_sha256') if key in row}
    if reconstruction.get('source_sha256') not in source_hashes:
        raise ValueError('reconstruction source must be a real train/validation clip')
    for filename, key in [('A-processed-target-22050.wav', 'a_sha256'), ('B-posterior-decoder-22050.wav', 'b_sha256')]:
        audio = path.parent / filename
        if not audio.is_file() or sha256(audio) != reconstruction.get(key):
            raise ValueError('reconstruction A/B audio file/hash mismatch')


def build_review(prepared: Path, owner: dict, phase: str = 'pilot') -> dict:
    if phase not in ('benchmark', 'pilot'):
        raise ValueError('review phase must be benchmark or pilot')
    audit = Path(owner['content_audit']) if owner.get('content_audit') else None
    provenance = Path(owner['audit_provenance']) if owner.get('audit_provenance') else None
    plan = create_plan(prepared, audit, provenance)
    _validate_owner(plan, owner, prepared, phase)
    return {'schema': SCHEMA, 'status': 'owner-approved-for-' + phase, 'phase': phase, 'owner': owner,
            'plan_sha256': hashlib.sha256(canonical(plan)).hexdigest(),
            'owner_sha256': hashlib.sha256(canonical(owner)).hexdigest(),
            'note': ('Bound approval for mechanical timing only; no quality approval or trained weights retained.'
                     if phase == 'benchmark' else 'Bound approval for this immutable collection only; no production promotion.')}


def validate_training_review(prepared: Path, review_path: Path, phase: str = 'pilot') -> dict:
    review = json.loads(review_path.read_text())
    if review.get('schema') != SCHEMA or review.get('status') != 'owner-approved-for-' + phase:
        raise ValueError('owner-approved collection review required')
    expected = build_review(prepared, review.get('owner', {}), phase=phase)
    if review != expected:
        raise ValueError('collection review was modified or its source inputs changed')
    return review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--content-audit', type=Path)
    parser.add_argument('--audit-provenance', type=Path)
    parser.add_argument('--owner-review', type=Path)
    parser.add_argument('--write-review', action='store_true')
    parser.add_argument('--phase', choices=('benchmark', 'pilot'), default='pilot',
                        help='benchmark requires technical integrity and general authorization; pilot also requires completed quality gates')
    parser.add_argument('--output', type=Path, help='private new file; plan is printed unless output is supplied')
    args = parser.parse_args()
    try:
        if args.write_review:
            if args.owner_review is None or args.output is None:
                raise ValueError('--write-review requires --owner-review and --output')
            result = build_review(args.prepared, json.loads(args.owner_review.read_text()), phase=args.phase)
        else:
            result = create_plan(args.prepared, args.content_audit, args.audit_provenance)
        if args.output:
            private_write(args.output, result)
            print(json.dumps({'status': result['status'], 'output': str(args.output.resolve()), 'sha256': sha256(args.output)}))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'{type(error).__name__}: {error}\n')


if __name__ == '__main__':
    main()
