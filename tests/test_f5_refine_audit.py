"""Final evidence must reject stale rollback, changed weights and wrong ASR."""
import hashlib
import json

import pytest

from training.f5_refine_audit import asr_for, distribution, pcm_proxy, seal_inputs


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sealed(root):
    checkpoint = root / 'weights/best.safetensors'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'unchanged best weights')
    other = checkpoint.parent / 'pilot.safetensors'
    other.write_bytes(b'preserved pilot weights')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.parent.iterdir()}
    write(root, 'preflight.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=hashes[checkpoint.name], checkpoint_hashes=hashes))
    write(root, 'gpu-complete.json', dict(all_gpu_experiments_complete=True, no_weight_updates=True,
          additional_training_started=False, checkpoint_sha256_after=hashes[checkpoint.name]))
    write(root, 'operations/gpu-window.json', dict(status='production_restored'))
    write(root, 'operations/baseline.json', dict(container_id='same', image_id='same-image'))
    write(root, 'operations/restored.json', dict(container_id='same', image_id='same-image', running=True, health='healthy'))
    return checkpoint, other


def test_final_audit_requires_current_healthy_rollback(tmp_path):
    sealed(tmp_path)
    assert seal_inputs(tmp_path)['production_restored']
    write(tmp_path, 'operations/gpu-window.json', dict(status='active'))
    with pytest.raises(ValueError, match='rollback'):
        seal_inputs(tmp_path)
    write(tmp_path, 'operations/gpu-window.json', dict(status='production_restored'))
    write(tmp_path, 'operations/restored.json', dict(container_id='different', image_id='same-image', running=True, health='healthy'))
    with pytest.raises(ValueError, match='differs'):
        seal_inputs(tmp_path)


def test_audit_seals_all_checkpoints_including_unselected_and_new_files(tmp_path):
    checkpoint, pilot = sealed(tmp_path)
    pilot.write_bytes(b'changed pilot')
    with pytest.raises(ValueError, match='checkpoint set or hashes changed'):
        seal_inputs(tmp_path)
    pilot.write_bytes(b'preserved pilot weights')
    (checkpoint.parent / 'unexpected.pt').write_bytes(b'extra training state')
    with pytest.raises(ValueError, match='checkpoint set or hashes changed'):
        seal_inputs(tmp_path)


def test_asr_alias_requires_same_audio_hash_and_expected_text():
    row = dict(id='alias', sha256='same-audio', tts_text='A janela ficou aberta.')
    receipt = dict(id='primary', audio_sha256='same-audio', expected=row['tts_text'],
                   hypothesis=row['tts_text'], word_edits=0, reference_tokens=['a', 'janela', 'ficou', 'aberta'], word_error_rate=0)
    assert asr_for(row, [receipt])['word_error_rate'] == 0
    with pytest.raises(ValueError, match='missing'):
        asr_for({**row, 'tts_text': 'Different text'}, [receipt])
    with pytest.raises(ValueError, match='provenance mismatch'):
        asr_for(row, [{**receipt, 'id': 'alias', 'audio_sha256': 'wrong-audio'}])


def test_acoustic_distribution_does_not_invent_missing_pitch_estimates():
    absent = distribution([None, None])
    assert absent['count'] == 0 and absent['median'] is None
    actual = distribution([100, None, 120, 140])
    assert actual['count'] == 3
    assert actual['median'] == 120
    assert actual['population_variance'] == pytest.approx(800/3)


def test_standard_pcm_proxy_refuses_overflow_and_preserves_arrival_schedule():
    np = pytest.importorskip('numpy')
    values = np.full(2400, .1)
    values[:480] = 0
    original = values.copy()
    row = dict(total_seconds=2, arrivals=[dict(ms=100, bytes=4800)],
               first_useful_pcm_available_ms=100, onset_audio_offset_ms=20)
    measured = pcm_proxy(row, values, 24000)
    assert measured['metrics']['first_useful_pcm_available_ms'] == 100
    assert measured['metrics']['onset_audio_offset_ms'] == 20
    assert measured['standard_minus_original_receipt_ms']['onset_audio_offset_ms'] == 0
    assert np.array_equal(values, original)
    assert pcm_proxy(row, np.ones(2400), 24000)['available'] is False
