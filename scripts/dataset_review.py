#!/usr/bin/env python3
"""Plan or build a batch content review after owner-approved listening sampling.

No ASR execution, API calls, voice generation, training or transcript correction.
Default is a read-only plan. Writing requires --write-review plus a bound owner
approval. ASR flags remain pending unless explicitly reviewed, without blocking
passing examples. Technical quarantine is never bypassed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from dataset_content_audit import comparison
from dataset_pipeline import ROOT, canonical, digest, file_digest, json_write, private_directory


def jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate IDs: ' + path.name)
    return rows


def sha256_value(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def load_inputs(dataset: Path, audit: Path, audit_provenance: Path, corpus: Path) -> dict:
    rows, checks, texts = jsonl(dataset), jsonl(audit), jsonl(corpus)
    if not rows:
        raise ValueError('empty dataset')
    text_by_id = {row['id']: row for row in texts}
    by_id = {row['id']: row for row in rows}
    checks_by_id = {row['id']: row for row in checks}
    provenance = json.loads(audit_provenance.read_text())
    dataset_hash = file_digest(dataset)
    if provenance.get('dataset_sha256') != dataset_hash:
        raise ValueError('audit provenance refers to another dataset SHA256')
    if provenance.get('examples_checked') != len(checks):
        raise ValueError('audit example count mismatch')
    for row in rows:
        original = text_by_id.get(row['id'])
        if original is None or any(row.get(key) != original.get(key) for key in ('text', 'split')):
            raise ValueError('corpus text/split mismatch: ' + row['id'])
        if row.get('group_id', row.get('group')) != original.get('group_id', original.get('group')):
            raise ValueError('corpus group mismatch: ' + row['id'])
        if not original.get('category'):
            raise ValueError('corpus category required: ' + row['id'])
        if file_digest(Path(row['audio_filepath'])) != row['sha256']:
            raise ValueError('audio SHA256 mismatch: ' + row['id'])
    flags = set()
    for check in checks:
        row = by_id.get(check['id'])
        if row is None or row['quality_status'] != 'accepted' or row['alignment_status'] != 'aligned':
            raise ValueError('ASR audit includes unknown or quarantined ID: ' + check['id'])
        if check.get('expected_text') != row['text'] or check.get('audio_sha256') != row['sha256']:
            raise ValueError('ASR text/audio binding mismatch: ' + row['id'])
        # Recompute the comparison; a hand-edited pass flag cannot hide omissions.
        result = comparison(row['text'], check.get('recognized_text', ''))
        if any(check.get(key) != result[key] for key in ('content_check', 'word_edits', 'word_error_rate')):
            raise ValueError('ASR comparison fields inconsistent: ' + row['id'])
        if result['content_check'] != 'pass':
            flags.add(row['id'])
    if set(provenance.get('review_required', [])) != flags:
        raise ValueError('audit provenance flag inventory mismatch')
    masters = {row['processed_master_sha256'] for row in rows}
    sources = {row['source_master_sha256'] for row in rows}
    manifests = {row.get('provenance', {}).get('manifest_sha256') for row in rows}
    if any(len(values) != 1 or not sha256_value(next(iter(values))) for values in (masters, sources, manifests)):
        raise ValueError('batch must refer to one bound source/processed master and manifest')
    processed_hash = next(iter(masters))
    # Verify the actual export, not only its string in the JSONL.
    master_paths = {row.get('provenance', {}).get('processed_source') for row in rows}
    if None in master_paths or any(file_digest(Path(path)) != processed_hash for path in master_paths):
        raise ValueError('processed master file SHA256 mismatch')
    bindings = {'dataset_sha256': dataset_hash, 'corpus_sha256': file_digest(corpus),
                'content_audit_sha256': file_digest(audit),
                'audit_provenance_sha256': file_digest(audit_provenance),
                'source_master_sha256': next(iter(sources)), 'processed_master_sha256': processed_hash,
                'manifest_sha256': next(iter(manifests))}
    return {'rows': rows, 'texts': text_by_id, 'checks': checks_by_id, 'flags': flags,
            'bindings': bindings}


def plan_review(inputs: dict, owner: dict | None = None) -> dict:
    owner = owner or {}
    rows, checks, flags = inputs['rows'], inputs['checks'], inputs['flags']
    by_id = {row['id']: row for row in rows}
    decisions = owner.get('flagged_reviews', {})
    if not isinstance(decisions, dict):
        raise ValueError('flagged_reviews must be an object')
    for identifier, decision in decisions.items():
        if identifier not in flags:
            raise ValueError('explicit flagged review must refer to an ASR flag: ' + identifier)
        if decision.get('decision') not in ('approved', 'rejected') or not decision.get('evidence'):
            raise ValueError('flagged review requires decision and evidence: ' + identifier)
    eligible = []
    for row in rows:
        identifier = row['id']
        if (row['quality_status'] != 'accepted' or row['alignment_status'] != 'aligned' or
            row.get('synthetic_fixture') or row.get('content_review') == 'rejected' or identifier not in checks):
            continue
        if identifier not in flags or decisions.get(identifier, {}).get('decision') == 'approved':
            eligible.append(identifier)
    strata = {}
    for identifier in eligible:
        row, text = by_id[identifier], inputs['texts'][identifier]
        strata.setdefault((text['category'], row['split']), []).append(identifier)
    samples = owner.get('sampled_ids', [])
    if not isinstance(samples, list) or len(samples) != len(set(samples)):
        raise ValueError('sampled_ids must be a unique list')
    if any(identifier not in by_id for identifier in samples):
        raise ValueError('sampled_ids includes unknown ID')
    coverage = {stratum for stratum, identifiers in strata.items() if set(identifiers) & set(samples)}
    missing = sorted(set(strata) - coverage)
    missing_bindings = [key for key, value in inputs['bindings'].items() if owner.get(key) != value]
    approval_fields = ('reviewer', 'reviewed_at', 'approval_reference')
    missing_approval = [key for key in approval_fields if not isinstance(owner.get(key), str) or not owner[key].strip()]
    if owner.get('batch_sample_approved') is not True:
        missing_approval.append('batch_sample_approved=true')
    if owner.get('perceptual_sample_approved') is not True:
        missing_approval.append('perceptual_sample_approved=true')
    if owner.get('reviewed_at'):
        try:
            if not isinstance(owner['reviewed_at'], str):
                raise ValueError('string required')
            parsed = datetime.fromisoformat(owner['reviewed_at'].replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                raise ValueError('timezone required')
        except (ValueError, TypeError):
            raise ValueError('reviewed_at must be ISO8601 with timezone')
    required_samples = [identifiers[0] for _, identifiers in sorted(strata.items())]
    return {'bindings': inputs['bindings'], 'eligible_for_batch': eligible,
            'pending_flags': sorted(flags - {identifier for identifier, value in decisions.items() if value['decision'] in ('approved', 'rejected')}),
            'technical_or_content_ineligible': [row['id'] for row in rows if row['id'] not in eligible],
            'required_sample_strata': [{'category': category, 'split': split} for category, split in sorted(strata)],
            'missing_sample_strata': [{'category': category, 'split': split} for category, split in missing],
            'suggested_sampled_ids': required_samples, 'missing_bindings': missing_bindings,
            'missing_owner_approval': missing_approval,
            'ready_to_write': not missing_bindings and not missing_approval and not missing,
            'owner_review_template': {**inputs['bindings'], 'reviewer': '', 'reviewed_at': '',
                                     'approval_reference': '', 'batch_sample_approved': False,
                                     'perceptual_sample_approved': False, 'sampled_ids': required_samples,
                                     'flagged_reviews': {}},
            'note': 'ASR pass alone never approves content/voice quality. Owner must hear and approve the stratified sample; unresolved flags remain pending.'}


def build_review(inputs: dict, owner: dict) -> tuple[dict, dict]:
    plan = plan_review(inputs, owner)
    if not plan['ready_to_write']:
        raise ValueError('batch review requires matching hashes, explicit owner approval and sample coverage')
    approved = set(plan['eligible_for_batch'])
    examples = {}
    for row in inputs['rows']:
        identifier = row['id']
        decision = owner.get('flagged_reviews', {}).get(identifier, {})
        status = 'approved' if identifier in approved else 'pending'
        if decision.get('decision') == 'rejected' or row.get('content_review') == 'rejected':
            status = 'rejected'
        if identifier in inputs['flags'] and identifier in approved:
            evidence = 'Explicit review of ASR flag: ' + decision['evidence']
        elif identifier in approved:
            evidence = 'Independent ASR comparison passed; owner approved stratified listening/perceptual sample.'
        else:
            evidence = 'Not approved: technical quarantine, missing audit or unresolved content review.'
        check = inputs['checks'].get(identifier)
        examples[identifier] = {'content_review': status, 'audio_sha256': row['sha256'],
                                'evidence': evidence + ' Approval reference: ' + owner['approval_reference'],
                                'audit_record_sha256': digest(canonical(check)) if check else None}
    review = {**inputs['bindings'], 'reviewer': owner['reviewer'], 'reviewed_at': owner['reviewed_at'],
              'approval_reference': owner['approval_reference'], 'sampled_ids': owner['sampled_ids'],
              'batch_sample_approved': True, 'perceptual_sample_approved': True,
              'examples': examples, 'owner_review_sha256': digest(canonical(owner))}
    return review, plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--content-audit', type=Path, required=True)
    parser.add_argument('--audit-provenance', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=ROOT / 'dataset/corpus.pt-BR.jsonl')
    parser.add_argument('--owner-review', type=Path)
    parser.add_argument('--write-review', action='store_true', help='explicit opt-in; default only prints plan')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        inputs = load_inputs(args.dataset, args.content_audit, args.audit_provenance, args.corpus)
        owner = json.loads(args.owner_review.read_text()) if args.owner_review else {}
        plan = plan_review(inputs, owner)
        if args.write_review:
            if args.owner_review is None or args.output is None:
                raise ValueError('--write-review requires --owner-review and --output')
            if args.output.resolve() in {p.resolve() for p in (args.dataset, args.content_audit, args.audit_provenance, args.corpus, args.owner_review)}:
                raise ValueError('output must not overwrite any review input')
            review, plan = build_review(inputs, owner)
            private_directory(args.output.parent)
            json_write(args.output, review)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'{type(error).__name__}: {error}\n')


if __name__ == '__main__':
    main()
