"""Operational contracts runnable without importing the optional CUDA stack."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from training import train_piper as runner
from training.prepare_dataset import sha256, write_pcm


def test_real_dataset_batch_arithmetic_and_validation_not_dropped():
    counts = runner.batch_counts(1025, 55, 2, 4)
    assert counts == {'train_minibatches_per_epoch': 512,
                      'validation_minibatches_per_epoch': 28,
                      'updates_per_optimizer_per_epoch': 128,
                      'dropped_train_examples_per_epoch': 1}
    assert runner.project_eta(2., 35., counts)['10']['seconds'] == 10590.
    assert runner.project_eta(2., 35., counts)['30']['seconds'] == 31770.
    assert runner.batch_counts(1025, 55, 1, 8)['updates_per_optimizer_per_epoch'] == 129


def test_cuda_measurements_summary_is_mean_median_max_not_hour_count():
    assert runner.timing_summary([1., 2., 9.]) == {
        'count': 3, 'mean_seconds': 4., 'median_seconds': 2., 'max_seconds': 9., 'total_seconds': 12.}
    for invalid in ([], [float('nan')], [-1.]):
        with pytest.raises(ValueError):
            runner.timing_summary(invalid)


@pytest.mark.parametrize('observed,complete', [(40, False), (450, True)])
def test_partial_benchmark_epoch_never_claims_full_epoch_eta(tmp_path, monkeypatch, observed, complete):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None,
        max_memory_allocated=lambda: 1024, max_memory_reserved=lambda: 2048))
    callback = runner.telemetry_callback(fake_torch, object, tmp_path,
        {'train_minibatches_per_epoch': 450})
    callback.epoch_start = 100.
    callback.epoch_times = [.1] * observed
    callback.validation_seconds = 25.
    monkeypatch.setattr(runner.time, 'perf_counter', lambda: 200.)
    callback.on_train_epoch_end(SimpleNamespace(current_epoch=0), None)
    record = callback.epochs[0]
    assert record['complete_training_epoch'] is complete
    assert record['observed_train_minibatches'] == observed
    assert record['expected_train_minibatches'] == 450
    if complete:
        assert record['eta_after_observed_epoch']['10']['remaining_seconds'] == 900.
    else:
        assert 'eta_after_observed_epoch' not in record
        assert 'partial training epoch' in record['eta_note']


def test_accumulation_includes_final_partial_group_without_underweighting():
    assert [runner.accumulation_group_size(i, 10, 4) for i in range(10)] == [4]*8 + [2]*2
    with pytest.raises(ValueError):
        runner.accumulation_group_size(10, 10, 4)


def test_fixed_split_checks_duplicate_ids_and_group_leakage_before_model_load():
    train = {'id': 'one', 'split': 'train', 'group_id': 'same'}
    with pytest.raises(ValueError, match='group crosses'):
        runner.partition_indices([train, {'id': 'two', 'split': 'validation', 'group_id': 'same'}])
    with pytest.raises(ValueError, match='duplicate'):
        runner.partition_indices([train, dict(train)])


def prepared_fixture(tmp_path, monkeypatch):
    prepared = tmp_path/'prepared'
    (prepared/'audio').mkdir(parents=True)
    rows = []
    for index, split in enumerate(('train', 'validation')):
        audio = prepared/'audio'/f'{index}.wav'
        write_pcm(audio, np.full(22050, .1), 22050)
        rows.append({'id': str(index), 'split': split, 'text': f'Texto {index}.',
                     'audio_filepath': str(audio), 'sha256': sha256(audio),
                     'content_review': 'approved', 'quality_status': 'accepted', 'alignment_status': 'aligned'})
    (prepared/'trainval.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with (prepared/'metadata.csv').open('w') as stream:
        csv.writer(stream, delimiter='|', lineterminator='\n').writerows(
            [Path(row['audio_filepath']).name, row['text']] for row in rows)
    (prepared/'import-report.json').write_text('{}')
    checkpoint = tmp_path/'base.ckpt'
    checkpoint.write_bytes(b'unit fixture, never a real checkpoint')
    monkeypatch.setattr(runner, 'BASE_SHA256', sha256(checkpoint))
    config = tmp_path/'config.json'
    config.write_text('{}')
    frontend = tmp_path/'frontend.json'
    frontend.write_text(json.dumps({'passed': True,
        'manifest_sha256': sha256(prepared/'trainval.jsonl'),
        'voice_config_path': str(config), 'voice_config_sha256': sha256(config),
        'phoneme_ids_by_id': {row['id']: [1,0,14,0,2] for row in rows}}))
    return prepared, checkpoint, frontend, config


def test_training_input_binding_rejects_stale_frontend_and_audio(tmp_path, monkeypatch):
    prepared, checkpoint, frontend, config = prepared_fixture(tmp_path, monkeypatch)
    assert len(runner.validate_inputs(prepared, checkpoint, None, frontend)[0]) == 2
    config.write_text('{"modified":true}')
    with pytest.raises(ValueError, match='frontend'):
        runner.validate_inputs(prepared, checkpoint, None, frontend)
    config.write_text('{}')
    (prepared/'audio/0.wav').write_bytes(b'altered')
    with pytest.raises(ValueError, match='prepared audio changed'):
        runner.validate_inputs(prepared, checkpoint, None, frontend)


def test_training_collection_never_bypasses_owner_review(tmp_path, monkeypatch):
    prepared, checkpoint, frontend, _ = prepared_fixture(tmp_path, monkeypatch)
    (prepared/'import-report.json').write_text(json.dumps({'schema': 'aegis.audio_collection.v1'}))
    with pytest.raises(ValueError, match='explicit hash-bound owner review'):
        runner.validate_inputs(prepared, checkpoint, None, frontend)


def test_old_frontend_report_cannot_skip_exact_cached_id_verification(tmp_path, monkeypatch):
    prepared, checkpoint, frontend, _ = prepared_fixture(tmp_path, monkeypatch)
    report = json.loads(frontend.read_text())
    del report['phoneme_ids_by_id']
    frontend.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='complete phoneme IDs'):
        runner.validate_inputs(prepared, checkpoint, None, frontend)


def test_isolated_frontend_captures_ids_and_never_encodes_unknown_phonemes():
    from training.frontend_check import check_inventory
    config = {'espeak': {'voice': 'pt-br'}, 'phoneme_id_map': {'a': [14], 'ɲ': [82]}}
    report = check_inventory([{'id': 'safe', 'text': 'known'}, {'id': 'unsafe', 'text': 'unknown'}],
        config, lambda text: [['a', 'ɲ']] if text == 'known' else [['missing']],
        lambda sentence, mapping: [value for symbol in sentence for value in mapping[symbol]])
    assert report['phoneme_ids_by_id'] == {'safe': [14, 82]}
    assert report['unknown_by_id'] == {'unsafe': ['missing']}
    assert not report['passed']


def test_metadata_or_import_rejection_cannot_be_silently_skipped(tmp_path, monkeypatch):
    prepared, checkpoint, frontend, _ = prepared_fixture(tmp_path, monkeypatch)
    (prepared/'metadata.csv').write_text('0.wav|texto adulterado\n')
    with pytest.raises(ValueError, match='metadata order'):
        runner.validate_inputs(prepared, checkpoint, None, frontend)
    (prepared/'import-report.json').write_text(json.dumps({'rejected': [{'id':'0'}]}))
    with pytest.raises(ValueError, match='unresolved'):
        runner.validate_inputs(prepared, checkpoint, None, frontend)


@pytest.mark.parametrize('use_scaler', [False, True])
def test_actual_autograd_manual_accumulation_updates_both_optimizers(use_scaler):
    torch = pytest.importorskip('torch')

    class GrowingScaler:
        scale = 8.

        def get_scale(self):
            return self.scale

    class ScaledOptimizer:
        def __init__(self, optimizer, scaler):
            self.optimizer, self.scaler = optimizer, scaler

        def zero_grad(self):
            self.optimizer.zero_grad()

        def step(self):
            for group in self.optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None:
                        parameter.grad.div_(self.scaler.scale)
            self.optimizer.step()
            # Exercise Lightning's per-optimizer scale update, including the
            # boundary between the G and D steps in one accumulated group.
            self.scaler.scale *= 2

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model_g = torch.nn.Linear(1, 1, bias=False)
            self.model_d = torch.nn.Linear(1, 1, bias=False)
            self.model_mrd = None
            with torch.no_grad():
                self.model_g.weight.fill_(.5)
                self.model_d.weight.fill_(.4)
            self.trainer = SimpleNamespace(num_training_batches=3)
            self.opts = [torch.optim.SGD(self.model_g.parameters(), lr=.01),
                         torch.optim.SGD(self.model_d.parameters(), lr=.01)]
            self.scaler = GrowingScaler() if use_scaler else None
            self.trainer.precision_plugin = SimpleNamespace(scaler=self.scaler)
            if self.scaler:
                self.opts = [ScaledOptimizer(opt, self.scaler) for opt in self.opts]
            self.aegis_accumulation = 2

        def optimizers(self):
            return self.opts

        def manual_backward(self, loss, **kwargs):
            if self.scaler:
                loss = loss * self.scaler.get_scale()
            loss.backward(**kwargs)

        def log(self, *args, **kwargs):
            pass

        def log_dict(self, *args, **kwargs):
            pass

        def _compute_loss(self, batch):
            generated = self.model_g(batch.x)
            loss_g = ((self.model_d(generated)-1)**2).mean()
            loss_d = (self.model_d(generated.detach())**2).mean()
            return loss_g, loss_d, {}

    Model = runner.accumulated_model_class(Base, torch)
    model = Model()
    batch = SimpleNamespace(phoneme_ids=torch.zeros((1, 1), dtype=torch.long), x=torch.ones((1, 1)))
    before_g = model.model_g.weight.detach().clone()
    before_d = model.model_d.weight.detach().clone()
    model.training_step(batch, 0)
    assert torch.equal(before_g, model.model_g.weight)
    assert torch.equal(before_d, model.model_d.weight)
    model.training_step(batch, 1)
    # dG = 2*(.4*.5-1)*.4 = -.64; dD = 2*.4*.5*.5 = .2.
    assert model.model_g.weight.item() == pytest.approx(.5064)
    assert model.model_d.weight.item() == pytest.approx(.398)
    model.training_step(batch, 2)
    assert model.model_g.weight.item() > .5064
    assert model.model_d.weight.item() < .398
