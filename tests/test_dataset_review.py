"""Batch review tests with synthetic metadata/bytes; no ASR/API/voice operations."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('dataset_review', SCRIPTS / 'dataset_review.py')
review_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_tool)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False))


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


@pytest.fixture
def bundle(tmp_path):
    master = tmp_path / 'master_processed.wav'
    master.write_bytes(b'synthetic test master bytes; no voice')
    processed_hash = review_tool.file_digest(master)
    corpus, dataset, audit = [], [], []
    for index, (category, split) in enumerate([('numbers', 'train'), ('neutral', 'validation'), ('numbers', 'test'), ('neutral', 'train'), ('numbers', 'train')]):
        identifier = f'id-{index}'
        text = f'O exemplo de teste tem {index} itens.'
        row = {'id': identifier, 'text': text, 'group': identifier, 'group_id': identifier,
               'category': category, 'split': split, 'stage': 'pilot'}
        corpus.append(row)
        clip = tmp_path / f'{identifier}.wav'
        clip.write_bytes(('synthetic clip ' + identifier).encode())
        saved = {**row, 'audio_filepath': str(clip), 'sha256': review_tool.file_digest(clip),
                 'quality_status': 'accepted' if index < 4 else 'quarantine',
                 'alignment_status': 'aligned' if index < 4 else 'quarantine', 'content_review': 'pending',
                 'source_master_sha256': 'a' * 64, 'processed_master_sha256': processed_hash,
                 'provenance': {'manifest_sha256': 'b' * 64, 'processed_source': str(master)}}
        dataset.append(saved)
        if index < 4:
            recognized = text if index != 3 else 'Outra frase com palavras ausentes.'
            audit.append({'id': identifier, 'expected_text': text, 'recognized_text': recognized,
                          'audio_sha256': saved['sha256'], **review_tool.comparison(text, recognized)})
    paths = {name: tmp_path / (name + ('.jsonl' if name in ('dataset', 'audit', 'corpus') else '.json'))
             for name in ('dataset', 'audit', 'corpus', 'provenance')}
    for name, rows in (('dataset', dataset), ('audit', audit), ('corpus', corpus)):
        write_jsonl(paths[name], rows)
    write_json(paths['provenance'], {'dataset_sha256': review_tool.file_digest(paths['dataset']),
                                    'examples_checked': len(audit), 'review_required': ['id-3']})
    inputs = review_tool.load_inputs(paths['dataset'], paths['audit'], paths['provenance'], paths['corpus'])
    plan = review_tool.plan_review(inputs)
    owner = {**plan['owner_review_template'], 'reviewer': 'Synthetic test reviewer',
             'reviewed_at': '2026-11-01T00:00:00+00:00', 'approval_reference': 'test-only/no-real-review',
             'batch_sample_approved': True, 'perceptual_sample_approved': True}
    return paths, inputs, owner


def test_default_cli_plans_without_writing_even_with_approved_owner(bundle, tmp_path):
    paths, inputs, owner = bundle
    owner_path, output = tmp_path / 'owner.json', tmp_path / 'review.json'
    write_json(owner_path, owner)
    result = subprocess.run([sys.executable, str(SCRIPTS / 'dataset_review.py'),
                             '--dataset', str(paths['dataset']), '--content-audit', str(paths['audit']),
                             '--audit-provenance', str(paths['provenance']), '--corpus', str(paths['corpus']),
                             '--owner-review', str(owner_path), '--output', str(output)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['ready_to_write']
    assert not output.exists()


def test_one_batch_approval_builds_all_good_entries_without_per_id_approvals(bundle):
    _, inputs, owner = bundle
    result, plan = review_tool.build_review(inputs, owner)
    assert all(result['examples'][f'id-{index}']['content_review'] == 'approved' for index in range(3))
    assert result['examples']['id-3']['content_review'] == 'pending'  # Unreviewed ASR flag.
    assert result['examples']['id-4']['content_review'] == 'pending'  # Technical quarantine.
    assert len(owner['sampled_ids']) == 3
    assert all(value['audio_sha256'] for value in result['examples'].values())
    assert result['owner_review_sha256']
    assert plan['pending_flags'] == ['id-3']


def test_asr_alone_cannot_approve_and_sample_must_cover_category_split(bundle):
    _, inputs, owner = bundle
    with pytest.raises(ValueError, match='explicit owner approval'):
        review_tool.build_review(inputs, {})
    owner['perceptual_sample_approved'] = False
    with pytest.raises(ValueError, match='explicit owner approval'):
        review_tool.build_review(inputs, owner)
    owner['perceptual_sample_approved'] = True
    owner['sampled_ids'] = ['id-0']
    plan = review_tool.plan_review(inputs, owner)
    assert len(plan['missing_sample_strata']) == 2
    with pytest.raises(ValueError, match='sample coverage'):
        review_tool.build_review(inputs, owner)


@pytest.mark.parametrize('field', ['dataset_sha256', 'corpus_sha256', 'content_audit_sha256',
                                  'audit_provenance_sha256', 'source_master_sha256',
                                  'processed_master_sha256', 'manifest_sha256'])
def test_all_batch_binding_hashes_are_required(bundle, field):
    _, inputs, owner = bundle
    owner[field] = '0' * 64
    with pytest.raises(ValueError, match='matching hashes'):
        review_tool.build_review(inputs, owner)


def test_flags_require_specific_review_without_blocking_other_good_examples(bundle):
    _, inputs, owner = bundle
    owner['flagged_reviews'] = {'id-3': {'decision': 'approved', 'evidence': 'Synthetic fixture: explicit content check recorded.'}}
    owner['sampled_ids'].append('id-3')
    result, plan = review_tool.build_review(inputs, owner)
    assert result['examples']['id-3']['content_review'] == 'approved'
    assert 'Explicit review of ASR flag' in result['examples']['id-3']['evidence']
    assert not plan['pending_flags']
    owner['flagged_reviews']['id-3']['decision'] = 'rejected'
    result, _ = review_tool.build_review(inputs, owner)
    assert result['examples']['id-3']['content_review'] == 'rejected'
    assert result['examples']['id-0']['content_review'] == 'approved'


def test_modified_audio_or_processed_export_refused(bundle):
    paths, inputs, _ = bundle
    audio = Path(inputs['rows'][0]['audio_filepath'])
    audio.write_bytes(b'changed clip')
    with pytest.raises(ValueError, match='audio SHA256 mismatch'):
        review_tool.load_inputs(paths['dataset'], paths['audit'], paths['provenance'], paths['corpus'])


def test_tampered_asr_pass_cannot_hide_flag(bundle):
    paths, _, _ = bundle
    records = review_tool.jsonl(paths['audit'])
    records[-1]['content_check'] = 'pass'
    write_jsonl(paths['audit'], records)
    with pytest.raises(ValueError, match='comparison fields inconsistent'):
        review_tool.load_inputs(paths['dataset'], paths['audit'], paths['provenance'], paths['corpus'])


def test_synthetic_or_rejected_examples_never_promoted(bundle):
    _, inputs, owner = bundle
    inputs['rows'][0]['synthetic_fixture'] = True
    inputs['rows'][1]['content_review'] = 'rejected'
    result, _ = review_tool.build_review(inputs, owner)
    assert result['examples']['id-0']['content_review'] == 'pending'
    assert result['examples']['id-1']['content_review'] == 'rejected'


def test_explicit_cli_write_creates_private_review(bundle, tmp_path):
    paths, _, owner = bundle
    owner_path, output = tmp_path / 'owner.json', tmp_path / 'review.json'
    write_json(owner_path, owner)
    subprocess.run([sys.executable, str(SCRIPTS / 'dataset_review.py'),
                    '--dataset', str(paths['dataset']), '--content-audit', str(paths['audit']),
                    '--audit-provenance', str(paths['provenance']), '--corpus', str(paths['corpus']),
                    '--owner-review', str(owner_path), '--write-review', '--output', str(output)],
                   capture_output=True, text=True, check=True)
    assert output.stat().st_mode & 0o777 == 0o600
    assert json.loads(output.read_text())['examples']['id-0']['content_review'] == 'approved'


@pytest.mark.parametrize('binding', ['manifest', 'audio'])
def test_extractor_enforces_optional_batch_review_bindings(tmp_path, binding):
    import numpy as np
    import dataset_pipeline as pipeline
    rate = 8000
    tone = .1 * np.sin(np.arange(rate) * 2 * np.pi * 200 / rate)
    values = np.r_[np.zeros(2 * rate), tone, np.zeros(2 * rate)]
    raw, processed, manifest_path = tmp_path / 'raw.wav', tmp_path / 'processed.wav', tmp_path / 'manifest.json'
    data = pipeline.wav_bytes(values, rate)
    raw.write_bytes(data)
    processed.write_bytes(data)
    manifest = {'master_file': str(raw), 'master_sha256': pipeline.digest(data), 'sample_rate': rate,
                'samples': len(values), 'gaps': [{'start': 0, 'end': 2 * rate, 'kind': 'bookend'},
                                               {'start': 3 * rate, 'end': 5 * rate, 'kind': 'bookend'}],
                'clips': [{'id': 'synthetic', 'group': 'g', 'split': 'test', 'text': 'Teste sintético.',
                           'order': 0, 'start_sample': 2 * rate, 'end_sample': 3 * rate,
                           'active_start_sample': 2 * rate, 'active_end_sample': 3 * rate,
                           'envelope_signature': pipeline.envelope_signature(tone).tolist(),
                           'source_audio_sha256': pipeline.digest(pipeline.wav_bytes(tone, rate))}]}
    write_json(manifest_path, manifest)
    approval = {'processed_master_sha256': pipeline.digest(data),
                'manifest_sha256': '0' * 64 if binding == 'manifest' else pipeline.file_digest(manifest_path),
                'reviewer': 'Synthetic reviewer', 'reviewed_at': '2026-11-01T00:00:00+00:00',
                'examples': {'synthetic': {'content_review': 'approved', 'evidence': 'Synthetic binding test only.',
                                           'audio_sha256': '0' * 64}}}
    review = tmp_path / 'review.json'
    write_json(review, approval)
    target = tmp_path / 'extracted'
    target.mkdir()
    with pytest.raises(ValueError, match='manifest SHA256' if binding == 'manifest' else 'audio SHA256 mismatch'):
        pipeline.extract(manifest_path, processed, target, review)
