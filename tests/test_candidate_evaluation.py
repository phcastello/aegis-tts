"""Mechanical probes only: no export, real voice synthesis, network or GPU."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from training import evaluate_piper as evaluation
from training.piper_adapter import PiperUnitAdapter
from training.prepare_dataset import sha256


def save(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def sources(tmp_path):
    experiment = tmp_path / 'pilot'
    (experiment / 'checkpoints').mkdir(parents=True)
    checkpoint = experiment / 'checkpoints/epoch=001.ckpt'
    checkpoint.write_bytes(b'not-real-model; never loaded')
    index = experiment / 'checkpoint-index.json'
    save(index, {checkpoint.name: {'sha256': sha256(checkpoint), 'bytes': checkpoint.stat().st_size,
                                  'epoch': 1, 'global_step': 42}})
    config = experiment / 'AegisVoicev2.0.onnx.json'
    save(config, {'espeak': {'voice': 'pt-br'}, 'audio': {'sample_rate': 22050},
                  'num_symbols': 256, 'num_speakers': 1, 'phoneme_id_map': {'a': [1]}})
    prepared = tmp_path / 'prepared'
    (prepared / 'audio').mkdir(parents=True)
    rows = [dict(id=f'item{i}', split=split, text=f'Texto mecânico de referência {i}.',
                 audio_filepath=str(prepared / 'audio' / f'item{i}.wav'))
            for i, split in enumerate(('train', 'validation', 'test'))]
    for name, subset in [('dataset', rows), ('trainval', rows[:2])]:
        (prepared / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in subset))
    with (prepared / 'metadata.csv').open('w', newline='') as stream:
        csv.writer(stream, delimiter='|').writerows((f'item{i}.wav', rows[i]['text']) for i in range(2))
    save(prepared / 'import-report.json', {'dataset_sha256': sha256(prepared / 'dataset.jsonl'),
                                         'trainval_sha256': sha256(prepared / 'trainval.jsonl'),
                                         'status': 'prepared-pending-review'})
    save(experiment / 'run.json', {'benchmark': False, 'base_sha256': evaluation.BASE_SHA256,
                                   'prepared_sha256': sha256(prepared / 'trainval.jsonl')})
    return checkpoint, index, config, prepared


def test_plan_is_read_only_and_never_exports_or_loads_voice(sources, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evaluation.subprocess, 'run', lambda *a, **k: pytest.fail('export in plan mode'))
    arguments = ['evaluate_piper', '--checkpoint', str(sources[0]), '--checkpoint-index', str(sources[1]),
                 '--voice-config', str(sources[2]), '--prepared', str(sources[3]), '--output', str(tmp_path / 'new')]
    monkeypatch.setattr(evaluation.sys, 'argv', arguments)
    evaluation.main()
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'plan-only' and report['promotion'] is False
    assert len(report['probes']) == 6
    assert not (tmp_path / 'new').exists()


@pytest.mark.parametrize('change', ['checkpoint_bytes', 'index_hash', 'benchmark', 'outside'])
def test_invalid_checkpoint_or_nonpilot_is_rejected_before_export(sources, tmp_path, change):
    checkpoint, index, config, prepared = sources
    if change == 'checkpoint_bytes':
        checkpoint.write_bytes(checkpoint.read_bytes() + b'altered')
    elif change == 'index_hash':
        value = json.loads(index.read_text()); value[checkpoint.name]['sha256'] = '0' * 64; save(index, value)
    elif change == 'benchmark':
        value = json.loads((index.parent / 'run.json').read_text()); value['benchmark'] = True; save(index.parent / 'run.json', value)
    else:
        other = tmp_path / 'outside.ckpt'; other.write_bytes(checkpoint.read_bytes()); checkpoint = other
    with pytest.raises(ValueError):
        evaluation.evaluation_plan(checkpoint, index, config, prepared)


def test_export_uses_same_interpreter_cpu_and_verifies_source_again(sources, tmp_path, monkeypatch):
    bindings = evaluation.validate_sources(*sources)
    output = tmp_path / 'export'; output.mkdir()
    observed = {}
    def fake_run(command, **kwargs):
        observed.update(command=command, environment=kwargs['env'])
        Path(command[-1]).write_bytes(b'fake-export-for-mechanical-test')
    monkeypatch.setattr(evaluation.subprocess, 'run', fake_run)
    model, config, result = evaluation.export_candidate(bindings, output)
    assert observed['command'][:3] == [evaluation.sys.executable, '-m', 'piper.train.export_onnx']
    assert observed['environment']['CUDA_VISIBLE_DEVICES'] == ''
    assert observed['environment']['OMP_NUM_THREADS'] == '1'
    assert result['model_sha256'] == sha256(model)
    assert config.read_bytes() == sources[2].read_bytes()
    sources[0].write_bytes(b'changed after plan')
    monkeypatch.setattr(evaluation.subprocess, 'run', lambda *a, **k: pytest.fail('loaded changed checkpoint'))
    with pytest.raises(ValueError, match='source changed'):
        evaluation.export_candidate(bindings, output)


def test_final_test_cannot_be_a_reference_and_seen_text_cannot_be_a_probe(sources, monkeypatch):
    with pytest.raises(ValueError, match='final test stays untouched'):
        evaluation.evaluation_plan(*sources, reference_ids=['item2'])
    assert evaluation.evaluation_plan(*sources, reference_ids=['item1'])['reference_ids'] == ['item1']
    monkeypatch.setattr(evaluation, 'PROMPTS', [('seen', 'Texto mecânico de referência 0.')])
    with pytest.raises(ValueError, match='already occurs'):
        evaluation.evaluation_plan(*sources)


def test_config_cannot_change_phoneme_type_with_the_same_symbol_map(sources, tmp_path):
    altered = tmp_path / 'altered-config.json'
    parameters = json.loads(sources[2].read_text())
    parameters['phoneme_type'] = 'text'
    save(altered, parameters)
    with pytest.raises(ValueError, match='phoneme_type'):
        evaluation.validate_sources(sources[0], sources[1], altered, sources[3])


def test_activity_timing_separates_first_chunk_silence_and_late_audible_delivery():
    pcm = np.r_[np.zeros(1200), np.full(1200, 500)].astype('<i2').tobytes()
    metrics = evaluation.activity_metrics(pcm, [{'seconds': .1, 'bytes': 2400},
                                               {'seconds': .9, 'bytes': 2400}], 1.0)
    assert metrics['first_chunk_wall_seconds'] == .1
    assert metrics['leading_pcm_silence_seconds'] == .05
    assert metrics['first_activity_available_wall_seconds'] == .9
    assert metrics['estimated_first_audible_playback_wall_seconds'] == .9
    assert metrics['real_time_factor'] == 10
    with pytest.raises(ValueError, match='inventory'):
        evaluation.activity_metrics(pcm, [], 1.0)


class MechanicalVoice:
    def synthesize(self, text):
        yield SimpleNamespace(sample_channels=1, sample_rate=22050,
                              audio_float_array=.1 * np.sin(np.arange(22050) * .05))


def test_capture_keeps_public_pcm_and_cancellation_releases_adapter(tmp_path):
    adapter = PiperUnitAdapter(MechanicalVoice())
    pcm, metrics = evaluation.capture(adapter.stream_unit('Exemplo mecânico.'))
    target = tmp_path / 'pcm.wav'; evaluation.write_wav(target, pcm)
    with wave.open(str(target)) as stream:
        assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) == (1, 2, 24000, 24000)
    assert metrics['audio_seconds'] == 1 and metrics['pcm_rail_samples'] == 0
    cancellation = evaluation.cancellation_probe(adapter, 'Exemplo mecânico longo.')
    assert cancellation['first_chunk_bytes'] > 0 and cancellation['bytes_after_cancellation'] == 0
    assert adapter._lock.acquire(blocking=False)
    adapter._lock.release()
    with pytest.raises(ValueError, match='entirely silent'):
        evaluation.capture(iter([b'\x00\x00' * 2400]))


def test_production_requires_url_and_explicit_authorization_and_reassembles_samples(monkeypatch):
    monkeypatch.setattr(evaluation, 'urlopen', lambda *a, **k: pytest.fail('unauthorized production request'))
    with pytest.raises(ValueError, match='separate authorization'):
        list(evaluation.production_chunks('http://localhost/v1/aegis/speech', None, 'Teste.'))
    class Response:
        headers = {'X-Aegis-Sample-Rate': '24000', 'X-Aegis-Channels': '1', 'X-Aegis-Audio-Format': 'pcm_s16le'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read1(self, count): return next(self.fragments)
    response = Response(); response.fragments = iter([b'\x01', b'\x00\x02', b'\x00', b''])
    requests = []
    def receive(request, **kwargs):
        requests.append(json.loads(request.data))
        return response
    monkeypatch.setattr(evaluation, 'urlopen', receive)
    assert b''.join(evaluation.production_chunks('http://localhost/v1/aegis/speech', 'test authorization', 'Teste.')) == b'\x01\x00\x02\x00'
    assert requests[-1]['text'] == 'Teste.' and 'input' not in requests[-1]
    response.fragments = iter([b'\x01\x00', b''])
    list(evaluation.production_chunks('http://localhost/v1/audio/speech', 'test authorization', 'Teste.'))
    assert requests[-1]['input'] == 'Teste.' and 'text' not in requests[-1] and 'priority' not in requests[-1]
    response.fragments = iter([b'\x01', b''])
    with pytest.raises(ValueError, match='truncated PCM16'):
        list(evaluation.production_chunks('http://localhost/v1/aegis/speech', 'test authorization', 'Teste.'))


def test_execute_records_private_artifacts_without_promoting(sources, tmp_path, monkeypatch):
    plan = evaluation.evaluation_plan(*sources)
    def fake_export(bindings, output):
        model = output / 'candidate.onnx'; model.write_bytes(b'fake-model')
        config = output / 'candidate.onnx.json'; config.write_bytes(sources[2].read_bytes())
        return model, config, {'model_sha256': sha256(model), 'config_sha256': sha256(config)}
    monkeypatch.setattr(evaluation, 'export_candidate', fake_export)
    output = tmp_path / 'evaluation'
    result = evaluation.evaluate(plan, output, authorization='mechanical test only',
                                 check_cancellation=True,
                                 adapter_factory=lambda *a: PiperUnitAdapter(MechanicalVoice()))
    assert result['status'] == 'awaiting-private-perceptual-review'
    assert result['promoted'] is False and len(result['samples']) == 6
    assert all(row['status'] == 'rendered' and row['audio_seconds'] == 1 for row in result['samples'])
    assert (output / 'evaluation.json').stat().st_mode & 0o777 == 0o600
    assert output.stat().st_mode & 0o777 == 0o700
    generated = [json.loads(line) for line in (output / 'generated-evaluation.jsonl').read_text().splitlines()]
    assert len(generated) == 6
    assert all(row['split'] == 'evaluation-only' and row['content_review'] == 'pending' for row in generated)
    with pytest.raises(ValueError, match='new private directory'):
        evaluation.evaluate(plan, output, authorization='test only')
