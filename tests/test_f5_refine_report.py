"""Private listening delivery integrity and honest pending perceptual scoring."""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import soundfile as sf

from training import f5_refine_report as report


def write(root, name, value):
    path = root / name
    path.parent.mkdir(mode=0o700, exist_ok=True, parents=True)
    path.write_text(json.dumps(value))


def reseal(root):
    names = ('preflight.json', 'gpu-complete.json', 'results.json', 'asr.json',
        'qwen/results.json', 'operations/gpu-window.json', 'operations/restored.json', 'operations/baseline.json')
    write(root, 'technical-audit.json', dict(owner_perceptual_approval=False,
        seal=dict(selected_checkpoint_sha256='unchanged',all_checkpoint_hashes={'best.safetensors':'unchanged'},
            no_weight_updates=True,additional_training_started=False,production_restored=True,
            container_id='same-original',image_id='same-image'),
        input_receipt_sha256={n:report.digest(root/n) for n in names}))


def fixture(root):
    def audio(identifier, i, stage, case):
        path = root / 'audio' / (identifier + '.wav')
        path.parent.mkdir(parents=True, exist_ok=True)
        # One deliberately overrange native tests refusal without audio edits.
        sf.write(path, np.linspace(0, 1.2 if i == 1 else .2 + i/100, 256), 24000, subtype='FLOAT')
        return dict(id=identifier, stage=stage, case=case, label=identifier,
            path=str(path), sha256=report.digest(path), text='Texto exibido CPU.', tts_text='Texto exibido cê pê u.',
            total_seconds=.1, audio_seconds=256/24000, rtf=9.375, peak=1.2 if i == 1 else .2+i/100,
            nfe=16 if identifier.endswith('-16') else 8,
            first_pcm_ms=100, first_useful_pcm_available_ms=100,
            onset_audio_offset_ms=20, estimated_first_audible_playback_ms=120)
    rows, qwen = [], []
    i = 0
    for case in report.CASES:
        for prefix, suffix in [('qwen-', ''), ('original-', '-16'), ('original-', '-8'), ('selected-', '')]:
            identifier = prefix + case + suffix
            i += 1
            value = audio(identifier, i, 'controlled', case)
            (qwen if prefix == 'qwen-' else rows).append(value)
    rows += [audio('selected-ordinary-block0', 15, 'blocks', 'ordinary'),
             audio('selected-repeat', 16, 'stability', 'ordinary'),
             audio('long-budget90', 17, 'duration', 'long')]
    refs = []
    for i in range(4):
        value = audio('source' + str(i), 20+i, 'source', 'source')
        refs.append(dict(id='source' + str(i), audio_filepath=value['path'], sha256=value['sha256'],
            duration=value['audio_seconds'], group_id='group' + str(i), text='Referência correta.'))
    write(root, 'references.json', {'references': refs})
    write(root, 'results.json', rows)
    write(root, 'qwen/results.json', qwen)
    write(root, 'asr.json', [dict(id=r['id'],audio_sha256=r['sha256'],expected=r['tts_text'],
        hypothesis='Texto exibido CPU.', word_error_rate=.5, word_edits=2,
        reference_tokens=r['tts_text'].split()) for r in [*rows, *qwen]])
    write(root, 'gpu-complete.json', dict(all_gpu_experiments_complete=True,no_weight_updates=True,
        checkpoint_sha256_after='unchanged',selected_reference='source0',selected_nfe=8,
        additional_training_started=False,owner_approval=False))
    write(root, 'preflight.json', dict(checkpoint='/private/checkpoints/best.safetensors',
        checkpoint_sha256='unchanged',checkpoint_hashes={'best.safetensors':'unchanged'}))
    write(root, 'reference-comparison.json', {'selected':dict(reference_id='source0',nfe=8,owner_approval=False)})
    write(root, 'operations/gpu-window.json', dict(status='production_restored',
        stop_started_utc='2026-11-03T03:00:00Z',stopped_utc='2026-11-03T03:01:00Z',
        gpu_work_ended_utc='2026-11-03T03:04:00Z',restored_utc='2026-11-03T03:05:00Z'))
    write(root, 'operations/restored.json', dict(container_id='same-original',image_id='same-image',running=True,health='healthy'))
    write(root, 'operations/baseline.json', dict(container_id='same-original',image_id='same-image',running=True,health='healthy'))
    reseal(root)
    return rows, qwen


def test_package_preserves_native_overrange_and_exports_small_offline_cohort(tmp_path):
    root = tmp_path / 'private'
    rows, qwen = fixture(root)
    before = {r['path']:Path(r['path']).read_bytes() for r in [*rows,*qwen]}
    archive = tmp_path / 'listening.zip'
    result = report.build(root, archive)
    assert result['archive_crc_verified'] and result['archive_entry_hashes_verified']
    assert result['declined_pcm16'] == 1
    assert result['listening_rows'] == 13  # 12 main + one duration diagnostic.
    assert result['source_references'] == 4
    assert all(Path(path).read_bytes() == content for path,content in before.items())
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        assert all('selected-repeat' not in n and 'block0' not in n for n in names)
        assert all('operations' not in n for n in names)
        assert package.testzip() is None
        artifacts = json.loads(package.read('artifacts-sha256.json'))
        assert all(hashlib.sha256(package.read(n)).hexdigest() == sha for n,sha in artifacts.items())
        metrics = json.loads(package.read('metrics.json'))
        assert len([r for r in metrics if r['stage'] == 'controlled']) == 12
        assert metrics[0]['display_text'] != metrics[0]['tts_text']
        html = package.read('index.html').decode()
        assert 'Diagnóstico de duração' in html
        assert 'NFE 8 não é declarado superior' in html
        scorecard = list(csv.DictReader(io.StringIO(package.read('scorecard.csv').decode('utf-8-sig'))))
        assert all(all(r[field] == '' for field in report.PERCEPTUAL_FIELDS) for r in scorecard)
        assert all(r['status'] == 'pending_owner_listening' for r in scorecard)
        summary = json.loads(package.read('report.json'))
        assert summary['production_restoration_verified']
        assert not summary['nfe8_superiority_established']
        assert summary['downtime_seconds'] == 300
    second = report.build(root, archive)
    assert second['native_files'] == result['native_files']
    assert second['playback_copies'] == result['playback_copies']


def test_incomplete_controlled_cohort_or_missing_rollback_refuses_package(tmp_path):
    root = tmp_path / 'private'
    rows, _ = fixture(root)
    write(root, 'operations/gpu-window.json', dict(status='running'))
    with pytest.raises(ValueError, match='production restoration'):
        report.build(root, tmp_path / 'refused.zip')
    assert not (tmp_path / 'refused.zip').exists()
    write(root, 'operations/gpu-window.json', dict(status='production_restored',
        stop_started_utc='2026-11-03T03:00:00Z',restored_utc='2026-11-03T03:05:00Z'))
    write(root, 'results.json', rows[1:])
    reseal(root)
    with pytest.raises(ValueError, match='incomplete controlled cohort'):
        report.build(root, tmp_path / 'refused.zip')
    assert not (tmp_path / 'refused.zip').exists()


def test_source_tamper_refuses_without_reencoding(tmp_path):
    root = tmp_path / 'private'
    rows, _ = fixture(root)
    source = Path(rows[0]['path'])
    source.write_bytes(source.read_bytes() + b'altered')
    with pytest.raises(ValueError, match='source audio changed'):
        report.build(root, tmp_path / 'refused.zip')
    assert not (tmp_path / 'refused.zip').exists()


def test_qwen_actual_frontend_text_keeps_original_request_and_asr_receipts(tmp_path):
    root = tmp_path / 'private'
    _, qwen = fixture(root)
    qwen_results_before = (root / 'qwen/results.json').read_bytes()
    original_asr = dict(id=qwen[0]['id'], audio_sha256=qwen[0]['sha256'],
        expected=qwen[0]['tts_text'], hypothesis='Texto exibido CPU.',word_error_rate=.25,word_edits=1,
        reference_tokens=qwen[0]['tts_text'].split())
    originals = json.loads((root/'asr.json').read_text())
    write(root, 'asr.json', [r for r in originals if r['id'] != qwen[0]['id']] + [original_asr])
    before_asr = (root / 'asr.json').read_bytes()
    actual = 'Texto exibido cê pê ú, via frontend original.'
    write(root, 'qwen/frontend-audit.json', dict(qwen_results_sha256=report.digest(root / 'qwen/results.json'),
        records=[dict(id=qwen[0]['id'],audio_sha256=qwen[0]['sha256'],
            request_text=qwen[0]['tts_text'],tts_text=actual,wer_scope='raw-request ASR; not directly comparable')]))
    reseal(root)
    archive = tmp_path / 'audited.zip'
    report.build(root, archive)
    with zipfile.ZipFile(archive) as package:
        rows = json.loads(package.read('metrics.json'))
        row = next(r for r in rows if r['id'] == qwen[0]['id'])
        assert row['request_text'] == qwen[0]['tts_text']
        assert row['tts_text'] == actual
        assert row['asr_expected'] == original_asr['expected']
        assert row['word_error_rate'] == original_asr['word_error_rate']
        assert row['frontend_audited']
        assert 'qwen/frontend-audit.json' in package.namelist()
    assert (root / 'qwen/results.json').read_bytes() == qwen_results_before
    assert (root / 'asr.json').read_bytes() == before_asr


@pytest.mark.parametrize('problem', ['training_started', 'unhealthy_rollback', 'different_image',
    'changed_checkpoint_seal', 'unsealed_receipt', 'asr_wrong_audio', 'asr_wrong_expected'])
def test_changed_or_contradictory_evidence_refuses_delivery(tmp_path, problem):
    root = tmp_path / 'private'
    fixture(root)
    if problem == 'training_started':
        value = json.loads((root/'gpu-complete.json').read_text())
        value['additional_training_started'] = True
        write(root, 'gpu-complete.json', value)
    elif problem in ('unhealthy_rollback', 'different_image'):
        value = json.loads((root/'operations/restored.json').read_text())
        value['health' if problem == 'unhealthy_rollback' else 'image_id'] = 'different'
        write(root, 'operations/restored.json', value)
    elif problem == 'changed_checkpoint_seal':
        value = json.loads((root/'technical-audit.json').read_text())
        value['seal']['all_checkpoint_hashes']['best.safetensors'] = 'different'
        write(root, 'technical-audit.json', value)
    elif problem == 'unsealed_receipt':
        value = json.loads((root/'asr.json').read_text())
        value[0]['hypothesis'] = 'Changed after audit'
        write(root, 'asr.json', value)
    else:
        value = json.loads((root/'asr.json').read_text())
        value[0]['audio_sha256' if problem == 'asr_wrong_audio' else 'expected'] = 'different'
        write(root, 'asr.json', value)
        reseal(root)  # The provenance check still rejects a contradictory ASR.
    with pytest.raises(ValueError):
        report.build(root, tmp_path/'refused.zip')
    assert not (tmp_path/'refused.zip').exists()
