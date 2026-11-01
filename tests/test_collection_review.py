"""Synthetic PCM fixtures validate gates, never real auditory approvals."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.dataset_content_audit import comparison
from training.audio_collection import prepare_collection
from training.collection_review import (acoustic_features, build_review, create_plan,
                                        private_write, validate_training_review)
from training.prepare_dataset import sha256, write_pcm


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False))


def lines(path, values):
    path.write_text(''.join(json.dumps(value, ensure_ascii=False) + '\n' for value in values))


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    rows = []
    texts = ['Ative os motores da nave.', 'São 08:45 no relógio.', 'Você ouviu aquela música?']
    for index, split in enumerate(('train', 'validation', 'test')):
        path = source / (split + '.wav')
        signal = .15 * np.sin(np.arange(38400) * 2 * np.pi * (220 + index * 70) / 48000)
        values = np.r_[np.zeros(4800), signal, np.zeros(4800)]
        write_pcm(path, values, 48000)
        rows.append({'schema': 'aegis.audio_collection.v1', 'id': 'item-' + str(index),
                     'group_id': 'group-' + str(index), 'split': split, 'category': 'category-' + str(index),
                     'text': texts[index], 'audio_path': path.name, 'duration_seconds': 1,
                     'sample_rate': 48000, 'channels': 1, 'audio_sha256': sha256(path)})
    manifest = source / 'dataset.jsonl'
    lines(manifest, rows)
    prepared = tmp_path / 'prepared'
    prepare_collection(manifest, prepared, source)
    prepared_rows = [json.loads(line) for line in (prepared / 'dataset.jsonl').read_text().splitlines()]
    audit = tmp_path / 'audit.jsonl'
    checks = [{'id': row['id'], 'expected_text': row['text'], 'recognized_text': row['text'],
               'audio_sha256': row['sha256'], **comparison(row['text'], row['text'])} for row in prepared_rows]
    lines(audit, checks)
    provenance = tmp_path / 'audit-provenance.json'
    save(provenance, {'dataset_sha256': sha256(prepared / 'dataset.jsonl'), 'examples_checked': len(checks),
                      'review_required': [], 'device': 'cpu', 'language': 'pt',
                      'model_files_sha256': {'model.bin': '1' * 64}, 'complete': True})
    plan = create_plan(prepared, audit, provenance)
    owner = {**plan['owner_review_template'], 'reviewer': 'Synthetic fixture',
             'reviewed_at': '2026-11-01T00:00:00+00:00', 'approval_reference': 'test only',
             'rights_evidence_reference': 'test only', 'training_rights_confirmed': True,
             'voice_rights_confirmed': True, 'batch_sample_approved': True,
             'perceptual_sample_approved': True, 'reconstruction_effect_approved': True,
             'suspicious_reviews': {identifier: {'decision': 'approved', 'evidence': 'test only'}
                                    for identifier in plan['mandatory_review_ids']}}
    reconstruction = tmp_path / 'roundtrip'
    reconstruction.mkdir()
    a = reconstruction / 'A-processed-target-22050.wav'
    b = reconstruction / 'B-posterior-decoder-22050.wav'
    write_pcm(a, np.zeros(22050), 22050)
    write_pcm(b, np.zeros(22050), 22050)
    recipe = json.loads((Path(__file__).resolve().parents[1] / 'training/recipe.json').read_text())
    report = reconstruction / 'reconstruction.json'
    save(report, {'checkpoint_sha256': recipe['checkpoint']['sha256'],
                  'source_sha256': prepared_rows[0]['sha256'], 'a_sha256': sha256(a), 'b_sha256': sha256(b)})
    owner.update(reconstruction_report=str(report), reconstruction_report_sha256=sha256(report))
    return prepared, audit, provenance, plan, owner


def test_plan_never_grants_approval_and_preserves_unknown_master_provenance(bundle):
    prepared, _, _, plan, _ = bundle
    assert plan['status'] == 'pending-owner-review'
    assert not plan['owner_review_template']['perceptual_sample_approved']
    assert len(plan['strata']) == 3
    assert all('master_sha256' not in key for key in plan['bindings'])
    assert plan == create_plan(prepared, Path(plan['bindings']['content_audit']), Path(plan['bindings']['audit_provenance']))


def test_asr_missing_blocks_real_review_but_still_produces_acoustic_plan(bundle):
    prepared, _, _, _, owner = bundle
    plan = create_plan(prepared)
    assert plan['count'] == 3 and plan['acoustic_features']
    assert 'ASR audit missing' in plan['blockers']
    owner.pop('content_audit')
    owner.pop('audit_provenance')
    with pytest.raises(ValueError, match='ASR audit missing'):
        build_review(prepared, owner)


def test_bound_owner_review_can_be_validated_without_touching_dataset(bundle, tmp_path):
    prepared, _, _, _, owner = bundle
    before = sha256(prepared / 'dataset.jsonl')
    review = build_review(prepared, owner)
    output = tmp_path / 'approved.json'
    private_write(output, review)
    assert output.stat().st_mode & 0o777 == 0o600
    assert validate_training_review(prepared, output) == review
    assert sha256(prepared / 'dataset.jsonl') == before
    with pytest.raises(ValueError, match='overwrite'):
        private_write(output, review)


@pytest.mark.parametrize('key', ['training_rights_confirmed', 'voice_rights_confirmed', 'batch_sample_approved',
                                'perceptual_sample_approved', 'reconstruction_effect_approved'])
def test_owner_gates_are_explicit(bundle, key):
    prepared, _, _, _, owner = bundle
    owner[key] = False
    with pytest.raises(ValueError, match=key):
        build_review(prepared, owner)


def test_asr_tampering_or_missing_record_never_passes(bundle):
    prepared, audit, provenance, _, _ = bundle
    checks = [json.loads(line) for line in audit.read_text().splitlines()]
    checks[0]['recognized_text'] = 'Palavras incorretas e outras.'
    lines(audit, checks)
    with pytest.raises(ValueError, match='comparison inconsistent'):
        create_plan(prepared, audit, provenance)


def test_asr_flag_needs_individual_decision(bundle):
    prepared, audit, provenance, _, owner = bundle
    checks = [json.loads(line) for line in audit.read_text().splitlines()]
    checks[0]['recognized_text'] = 'Ative os motores.'
    checks[0].update(comparison(checks[0]['expected_text'], checks[0]['recognized_text']))
    lines(audit, checks)
    prov = json.loads(provenance.read_text())
    prov['review_required'] = ['item-0']
    save(provenance, prov)
    plan = create_plan(prepared, audit, provenance)
    owner.update(plan['bindings'])
    owner['suspicious_reviews'] = {}
    assert 'item-0' in plan['mandatory_review_ids']
    with pytest.raises(ValueError, match='every suspicious'):
        build_review(prepared, owner)
    owner['suspicious_reviews'] = {identifier: {'decision': 'approved', 'evidence': 'Synthetic fixture explicit check'}
                                    for identifier in plan['mandatory_review_ids']}
    assert build_review(prepared, owner)['status'] == 'owner-approved-for-pilot'


def test_listening_coverage_and_reconstruction_are_bound(bundle):
    prepared, _, _, _, owner = bundle
    owner['sampled_ids'] = []
    with pytest.raises(ValueError, match='every suggested'):
        build_review(prepared, owner)
    owner['sampled_ids'] = ['item-0', 'item-1', 'item-2']
    report = Path(owner['reconstruction_report'])
    value = json.loads(report.read_text())
    value['source_sha256'] = '0' * 64
    save(report, value)
    owner['reconstruction_report_sha256'] = sha256(report)
    with pytest.raises(ValueError, match='real train/validation clip'):
        build_review(prepared, owner)


def test_edge_energy_is_suspicion_without_claiming_perceptual_defect(tmp_path):
    path = tmp_path / 'tone.wav'
    write_pcm(path, .2 * np.sin(np.arange(22050) * .07), 22050)
    features, reasons = acoustic_features(path)
    assert features['spectral_centroid_hz'] > 0
    assert any('first 20ms' in reason for reason in reasons)
    assert any('last 20ms' in reason for reason in reasons)


def test_global_auditory_approval_does_not_invent_listened_ids(bundle):
    prepared, _, _, _, owner = bundle
    owner['auditory_approval_scope'] = 'collection'
    owner['auditory_approved'] = True
    owner['sampled_ids'] = []
    owner['batch_sample_approved'] = False
    owner['perceptual_sample_approved'] = False
    owner.pop('rights_evidence_reference')
    review = build_review(prepared, owner)
    assert review['owner']['sampled_ids'] == []
    assert review['owner']['batch_sample_approved'] is False


def test_mechanical_benchmark_can_precede_asr_and_reconstruction_without_promoting_pilot(bundle, tmp_path):
    prepared, _, _, _, owner = bundle
    for key in ('content_audit', 'content_audit_sha256', 'audit_provenance', 'audit_provenance_sha256'):
        owner.pop(key, None)
    owner.update(auditory_approval_scope='collection', auditory_approved=True, sampled_ids=[],
                 batch_sample_approved=False, perceptual_sample_approved=False,
                 reconstruction_effect_approved=False, reconstruction_report='', reconstruction_report_sha256='')
    review = build_review(prepared, owner, phase='benchmark')
    output = tmp_path / 'benchmark-approval.json'
    private_write(output, review)
    assert validate_training_review(prepared, output, phase='benchmark') == review
    assert not review['owner']['reconstruction_effect_approved']
    with pytest.raises(ValueError, match='owner-approved collection review required'):
        validate_training_review(prepared, output)


def test_global_approval_does_not_resolve_new_suspicious_findings(bundle):
    prepared, audit, provenance, _, owner = bundle
    checks = [json.loads(line) for line in audit.read_text().splitlines()]
    checks[0]['recognized_text'] = 'Ative os motores.'
    checks[0].update(comparison(checks[0]['expected_text'], checks[0]['recognized_text']))
    lines(audit, checks)
    prov = json.loads(provenance.read_text())
    prov['review_required'] = ['item-0']
    save(provenance, prov)
    plan = create_plan(prepared, audit, provenance)
    owner.update(plan['bindings'])
    owner.update(auditory_approval_scope='collection', auditory_approved=True, sampled_ids=[])
    # Real suspicious findings require per-example evidence even after a global
    # authorization. Use a boundary warning fixture to exercise that distinction.
    owner['suspicious_reviews'] = {}
    with pytest.raises(ValueError, match='every suspicious'):
        build_review(prepared, owner)


def test_review_accepts_authorized_cuda_asr_and_rejects_missing_window(bundle):
    prepared, audit, provenance, _, _ = bundle
    prov = json.loads(provenance.read_text())
    prov.update(device='cuda', compute_type='float16', owner_authorization='test-only',
                exclusive_gpu_window='test-only-window', gpu_minimum_free_mib=9000,
                cuda_capacity_check_passed=True)
    save(provenance, prov)
    assert not create_plan(prepared, audit, provenance)['blockers']
    prov.pop('exclusive_gpu_window')
    save(provenance, prov)
    with pytest.raises(ValueError, match='CUDA ASR provenance requires'):
        create_plan(prepared, audit, provenance)
