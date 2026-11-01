#!/usr/bin/env python3
"""Plan by default; real training/benchmark require reviewed data and a GPU window."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from .prepare_dataset import private_destination, sha256

BASE_SHA256 = 'abd3e6c7237aca5b7cf9c3cfc50945bfef44a83a594e00186f584d9d11fd355a'
SEED = 40733133


def partition_indices(rows: list[dict]) -> dict[str, list[int]]:
    result = {'train': [], 'validation': []}
    groups, ids = {}, set()
    for index, row in enumerate(rows):
        if row['split'] not in result:
            raise ValueError('final test examples must never enter the trainer')
        if row['id'] in ids:
            raise ValueError('duplicate stable id in trainer input')
        ids.add(row['id'])
        group = row.get('group_id', row['id'])
        if group in groups and groups[group] != row['split']:
            raise ValueError('group crosses splits in trainer input')
        groups[group] = row['split']
        result[row['split']].append(index)
    if not all(result.values()):
        raise ValueError('empty train or validation partition')
    return result


def batch_counts(train_count: int, validation_count: int, batch_size: int, accumulation: int) -> dict:
    if min(train_count, validation_count, batch_size, accumulation) < 1:
        raise ValueError('counts, batch size and accumulation must be positive')
    batches = train_count // batch_size if train_count > batch_size else 1
    return {'train_minibatches_per_epoch': batches,
            'validation_minibatches_per_epoch': math.ceil(validation_count / batch_size),
            'updates_per_optimizer_per_epoch': math.ceil(batches / accumulation),
            'dropped_train_examples_per_epoch': train_count % batch_size if train_count > batch_size else 0}


def timing_summary(seconds: list[float]) -> dict:
    if not seconds or any(not math.isfinite(x) or x < 0 for x in seconds):
        raise ValueError('finite nonnegative timing measurements required')
    return {'count': len(seconds), 'mean_seconds': statistics.mean(seconds),
            'median_seconds': statistics.median(seconds), 'max_seconds': max(seconds),
            'total_seconds': sum(seconds)}


def project_eta(train_seconds: float, validation_seconds: float, counts: dict, epochs=(10, 30)) -> dict:
    epoch = train_seconds * counts['train_minibatches_per_epoch'] + validation_seconds
    return {str(n): {'epochs': n, 'seconds': epoch * n, 'hours': epoch * n / 3600}
            for n in epochs}


def accumulation_group_size(index: int, total: int, accumulation: int) -> int:
    """Correct denominator for the final partial accumulation group as well."""
    if not 0 <= index < total or accumulation < 1:
        raise ValueError('invalid accumulation group')
    return min(accumulation, total - (index // accumulation) * accumulation)


def gpu_available(minimum_free_mib: int = 9000) -> None:
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, check=True, text=True)
    values = [int(x.strip()) for x in result.stdout.splitlines() if x.strip()]
    if not values or min(values) < minimum_free_mib:
        raise RuntimeError('exclusive GPU capacity unavailable; never stop production automatically')


def validate_inputs(prepared: Path, checkpoint: Path, review: Path | None,
                    frontend_report: Path | None, phase='pilot') -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in (prepared / 'trainval.jsonl').read_text().splitlines() if line.strip()]
    partition_indices(rows)
    report = json.loads((prepared / 'import-report.json').read_text())
    if report.get('status') == 'blocked' or report.get('rejected') or report.get('structural_errors'):
        raise ValueError('import has unresolved integrity/split failures')
    if report.get('schema') == 'aegis.audio_collection.v1':
        if not review:
            raise ValueError('audio collection requires explicit hash-bound owner review and rights')
        from .collection_review import validate_training_review
        validate_training_review(prepared, review, phase=phase)
    elif any(row.get('content_review') != 'approved' or row.get('quality_status') != 'accepted'
             or row.get('alignment_status') != 'aligned' for row in rows):
        raise ValueError('processed content has not been approved')
    with (prepared / 'metadata.csv').open() as stream:
        csv_rows = list(csv.reader(stream, delimiter='|'))
    expected_rows = [[Path(row['audio_filepath']).name, row['text']] for row in rows]
    if csv_rows != expected_rows:
        raise ValueError('metadata order or transcript diverged from reviewed manifest')
    if sha256(checkpoint) != BASE_SHA256:
        raise ValueError('public base checkpoint SHA256 mismatch')
    for row in rows:
        audio = Path(row['audio_filepath']).resolve()
        if audio.parent != (prepared / 'audio').resolve() or sha256(audio) != row['sha256']:
            raise ValueError('prepared audio changed since review/import or escaped prepared/audio')
    if not frontend_report:
        raise ValueError('a hash-bound successful frontend report is required')
    frontend = json.loads(frontend_report.read_text())
    if (frontend.get('passed') is not True
            or frontend.get('manifest_sha256') != sha256(prepared / 'trainval.jsonl')
            or frontend.get('voice_config_sha256') != sha256(Path(frontend['voice_config_path']))):
        raise ValueError('frontend inventory report is stale or failed')
    expected_ids = frontend.get('phoneme_ids_by_id', {})
    if set(expected_ids) != {row['id'] for row in rows} or any(not ids for ids in expected_ids.values()):
        raise ValueError('frontend report needs complete phoneme IDs; rerun isolated frontend_check')
    return rows, frontend


def build_data_module(rows, prepared: Path, output: Path, batch_size=2, frontend=None,
                      benchmark_warmup: int | None = None):
    from piper.train.vits.dataset import VitsDataModule
    from torch.utils.data import Subset
    phonemes_path = None
    expected_config = None
    if frontend:
        expected_config = json.loads(Path(frontend['voice_config_path']).read_text())
        phonemes_path = output / 'base-phoneme-id-map.json'
        encoded_map = json.dumps(expected_config['phoneme_id_map'], ensure_ascii=False, sort_keys=True)
        if phonemes_path.exists() and phonemes_path.read_text() != encoded_map:
            raise ValueError('cached pretrained phoneme map changed; use a new experiment')
        phonemes_path.write_text(encoded_map)
        phonemes_path.chmod(0o600)

    class FixedSplitDataModule(VitsDataModule):
        def prepare_data(self):
            if getattr(self, '_aegis_prepared', False):
                return
            super().prepare_data()
            if frontend:
                expected = json.loads(Path(frontend['voice_config_path']).read_text())
                actual = json.loads(self.config_path.read_text())
                for field in ('phoneme_id_map', 'num_symbols', 'num_speakers'):
                    if actual.get(field) != expected.get(field):
                        raise ValueError(f'prepared Piper frontend differs from verified base: {field}')
                if (actual['audio']['sample_rate'] != expected['audio']['sample_rate']
                        or actual['espeak']['voice'] != expected['espeak']['voice']):
                    raise ValueError('Piper frontend sample rate or eSpeak voice changed')
            self._aegis_prepared = True

        def setup(self, stage):
            super().setup(stage)
            full = self.train_dataset.dataset
            if len(full) != len(rows):
                raise RuntimeError('Piper skipped a clip; do not shift split membership')
            # Upstream tests Path truthiness rather than exists(); check each
            # cached file and identity, so an absent/corrupted cache cannot hide.
            for row, utt in zip(rows, full.utts):
                for name in ('phoneme_ids_path', 'audio_norm_path', 'audio_spec_path'):
                    if not getattr(utt, name).is_file():
                        raise RuntimeError(f"missing cache for {row['id']}: {name}")
                if utt.text != row['text']:
                    raise RuntimeError('cached utterance text/order changed')
            if expected_config:
                import torch
                from .prepare_dataset import read_pcm
                for row, utt in zip(rows, full.utts):
                    current_audio, rate = read_pcm(Path(row['audio_filepath']))
                    cached_audio = torch.load(utt.audio_norm_path, map_location='cpu', weights_only=True)
                    if rate != 22050 or not torch.equal(cached_audio, torch.tensor(current_audio, dtype=torch.float32)):
                        raise ValueError('cached audio differs from prepared waveform; rebuild cache: ' + row['id'])
                    # Native eSpeak remains in the isolated frontend check.
                    # Reinitializing it in Trainer setup produced native
                    # corruption after CUDA setup on this host. Compare every
                    # cached ID with the immutable, complete frontend result.
                    expected_ids = frontend.get('phoneme_ids_by_id', {}).get(row['id'])
                    if not expected_ids:
                        raise ValueError('isolated frontend phoneme IDs missing for ' + row['id'])
                    actual_ids = torch.load(utt.phoneme_ids_path, map_location='cpu', weights_only=True).tolist()
                    if expected_ids != actual_ids:
                        raise ValueError('cached phoneme IDs differ from exact base frontend; rebuild cache: ' + row['id'])
            indices = partition_indices(rows)
            self.train_dataset = Subset(full, indices['train'])
            self.val_dataset = Subset(full, indices['validation'])
            self.test_dataset = Subset(full, indices['validation'][:5])

        def train_dataloader(self):
            if benchmark_warmup is None:
                return super().train_dataloader()
            # Include the longest padded batch after warm-up, then a stable
            # representative permutation. No validation/test enters training.
            import random
            count = len(self.train_dataset)
            ordered = list(range(count))
            random.Random(SEED).shuffle(ordered)
            train_rows = [row for row in rows if row['split'] == 'train']
            longest = sorted(range(count), key=lambda i: train_rows[i]['duration'], reverse=True)[:batch_size]
            ordered = [index for index in ordered if index not in longest]
            offset = benchmark_warmup * batch_size
            ordered[offset:offset] = longest
            return self._make_dataloader(Subset(self.train_dataset, ordered), shuffle=False,
                                         drop_last=count > batch_size)

    return FixedSplitDataModule(csv_path=prepared / 'metadata.csv', audio_dir=prepared / 'audio',
        cache_dir=output / 'cache', config_path=output / 'AegisVoicev2.0.onnx.json',
        voice_name='AegisVoicev2.0-PiperPTBR-pilot', espeak_voice='pt-br', sample_rate=22050,
        batch_size=batch_size, num_symbols=256, num_speakers=1, num_workers=1,
        trim_silence=False, validation_split=0, num_test_examples=0,
        phonemes_path=phonemes_path, vowel_clusters=None,
        filter_length=1024, hop_length=256, win_length=1024, segment_size=8192)


def accumulated_model_class(base, torch):
    class AccumulatedVits(base):
        """Keep upstream loss/model; implement accumulation for manual G/D optimization."""
        def training_step(self, batch, batch_idx):
            opt_g, opt_d = self.optimizers()
            accumulation = self.aegis_accumulation
            total = int(self.trainer.num_training_batches)
            divisor = accumulation_group_size(batch_idx, total, accumulation)
            if batch_idx % accumulation == 0:
                opt_g.zero_grad()
                opt_d.zero_grad()
            loss_g, loss_d, metrics = self._compute_loss(batch)
            if not torch.isfinite(loss_g).all() or not torch.isfinite(loss_d).all():
                raise FloatingPointError('nonfinite generator/discriminator loss')
            # Only G receives gradients from its loss. Otherwise the upstream
            # discriminator forward contaminates D's accumulated gradients.
            self.manual_backward(loss_g / divisor, inputs=list(self.model_g.parameters()))
            disc = list(self.model_d.parameters())
            if self.model_mrd is not None:
                disc += list(self.model_mrd.parameters())
            self.manual_backward(loss_d / divisor, inputs=disc)
            if (batch_idx + 1) % accumulation == 0 or batch_idx + 1 == total:
                for parameter in list(self.model_g.parameters()) + disc:
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError('nonfinite gradient; checkpoint remains last complete epoch')
                scaler = getattr(getattr(self.trainer, 'precision_plugin', None), 'scaler', None)
                scale_before = scaler.get_scale() if scaler is not None else None
                opt_g.step()
                # Lightning's manual AMP plugin updates its scaler after each
                # optimizer step. D gradients were accumulated with the old
                # scale; preserve their value if G's step changed that scale.
                if scaler is not None and scaler.get_scale() != scale_before:
                    ratio = scaler.get_scale() / scale_before
                    for parameter in disc:
                        if parameter.grad is not None:
                            parameter.grad.mul_(ratio)
                opt_d.step()
            size = int(batch.phoneme_ids.shape[0])
            self.log('loss_g', loss_g, batch_size=size, on_step=True, on_epoch=True)
            self.log('loss_d', loss_d, batch_size=size, on_step=True, on_epoch=True)
            self.log_dict({f'train_{k}': v for k, v in metrics.items()},
                          batch_size=size, on_step=True, on_epoch=True)

        def validation_step(self, batch, batch_idx):
            loss_g, loss_d, metrics = self._compute_loss(batch)
            if not torch.isfinite(loss_g).all() or not torch.isfinite(loss_d).all():
                raise FloatingPointError('nonfinite validation loss')
            size = int(batch.phoneme_ids.shape[0])
            self.log('val_loss', loss_g, batch_size=size, on_step=False, on_epoch=True)
            self.log_dict({f'val_{key}': value for key, value in metrics.items()},
                          batch_size=size, on_step=False, on_epoch=True, sync_dist=True)
            return loss_g

        def on_train_epoch_end(self):
            # Lightning does not advance schedulers in manual optimization.
            for scheduler in self.lr_schedulers():
                scheduler.step()

    return AccumulatedVits


def telemetry_callback(torch, callback_base, output, counts, warmup=0):
    class Telemetry(callback_base):
        def __init__(self):
            self.times = []
            self.epochs = []
            self.epoch_times = []
            self.recent_losses = {}

        def synchronize(self):
            torch.cuda.synchronize()

        def emit(self, value):
            with (output / 'telemetry.jsonl').open('a') as stream:
                stream.write(json.dumps(value, allow_nan=False) + '\n')

        def on_train_epoch_start(self, trainer, model):
            self.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.epoch_start = time.perf_counter()
            self.epoch_times = []

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            self.synchronize()
            self.batch_start = time.perf_counter()

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            self.synchronize()
            elapsed = time.perf_counter() - self.batch_start
            self.epoch_times.append(elapsed)
            if trainer.current_epoch > 0 or batch_idx >= warmup:
                self.times.append(elapsed)
            losses = {key: float(value.detach().cpu()) for key, value in trainer.callback_metrics.items()
                      if key.startswith(('loss_', 'train_', 'val_')) and hasattr(value, 'numel') and value.numel() == 1}
            if any(not math.isfinite(x) for x in losses.values()):
                raise FloatingPointError('nonfinite training telemetry')
            divergence_warnings = []
            for key in ('loss_g_step', 'loss_d_step'):
                if key not in losses:
                    continue
                recent = self.recent_losses.setdefault(key, [])
                if len(recent) >= 32 and losses[key] > 10 * max(statistics.median(recent), 1e-6):
                    divergence_warnings.append(key + ' exceeds 10x recent median; inspect GAN stability')
                recent.append(losses[key])
                del recent[:-64]
            self.emit({'event': 'minibatch', 'epoch': trainer.current_epoch, 'batch': batch_idx,
                       'seconds': elapsed, 'warmup': trainer.current_epoch == 0 and batch_idx < warmup,
                       'optimizer_global_step': trainer.global_step, 'metrics': losses,
                       'divergence_warnings': divergence_warnings})

        def on_validation_start(self, trainer, model):
            self.synchronize()
            self.validation_start = time.perf_counter()

        def on_validation_end(self, trainer, model):
            self.synchronize()
            self.validation_seconds = time.perf_counter() - self.validation_start
            if not trainer.sanity_checking:
                self.emit({'event': 'validation', 'epoch': trainer.current_epoch,
                           'seconds': self.validation_seconds,
                           'metrics': {key: float(value.detach().cpu()) for key, value in trainer.callback_metrics.items()
                                       if key.startswith('val_') and hasattr(value, 'numel') and value.numel() == 1}})

        def on_train_epoch_end(self, trainer, model):
            self.synchronize()
            record = {'event': 'epoch', 'epoch': trainer.current_epoch,
                      'wall_seconds_including_validation_io': time.perf_counter() - self.epoch_start,
                      'train_timing': timing_summary(self.epoch_times),
                      'validation_seconds': getattr(self, 'validation_seconds', 0),
                      'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                      'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                      'host_max_rss_mib': __import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss / 1024}
            record['eta_after_observed_epoch'] = {
                str(n): {'remaining_seconds': max(0, n - trainer.current_epoch - 1) * record['wall_seconds_including_validation_io']}
                for n in (10, 30)}
            self.epochs.append(record)
            self.emit(record)
            (output / 'epoch-summary.json').write_text(json.dumps(self.epochs, indent=2))

        def on_exception(self, trainer, model, exception):
            self.emit({'event': 'failure', 'type': type(exception).__name__, 'message': str(exception),
                       'epoch': trainer.current_epoch, 'optimizer_global_step': trainer.global_step})

    return Telemetry()


def run(prepared: Path, checkpoint: Path, output: Path, resume: Path | None, *, max_epochs=10,
        batch_size=2, accumulation=4, precision='32-true', review=None, frontend_report=None,
        benchmark=False, warmup=8, measured=32, authorization=None, gpu_window=None):
    rows, frontend = validate_inputs(prepared, checkpoint, review, frontend_report,
                                     phase='benchmark' if benchmark else 'pilot')
    indices = partition_indices(rows)
    counts = batch_counts(len(indices['train']), len(indices['validation']), batch_size, accumulation)
    if benchmark and (warmup % accumulation or measured % accumulation or measured < 2 * accumulation
                      or warmup < accumulation or warmup + measured > counts['train_minibatches_per_epoch']):
        raise ValueError('benchmark needs warmup >= one update, >= two measured updates, complete groups fitting one epoch')
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from piper.train.vits.lightning import VitsModel
    torch.set_num_threads(1)
    seed_everything(SEED, workers=True)
    original = torch.load(checkpoint, map_location='cpu', weights_only=False)
    hparams = dict(original['hyper_parameters'])
    hparams.update(batch_size=batch_size, learning_rate=5e-5, learning_rate_d=2.5e-5,
                   sample_rate=22050, filter_length=1024, hop_length=256, win_length=1024,
                   mel_channels=80, mel_fmin=0.0, mel_fmax=None, mos_metric=None,
                   warmstart_ckpt=None, vocoder_warmstart_ckpt=None)
    model = accumulated_model_class(VitsModel, torch)(**hparams)
    model.aegis_accumulation = accumulation
    state, expected = original['state_dict'], model.state_dict()
    essential = [key for key in expected if key.startswith('model_g.')]
    if any(key not in state or state[key].shape != expected[key].shape for key in essential):
        raise ValueError('incompatible generator/posterior/vocoder checkpoint; no partial random voice')
    compatible = {k: v for k, v in state.items() if k in expected and v.shape == expected[k].shape}
    model.load_state_dict(compatible, strict=False)
    del original, state, compatible
    if any(not parameter.requires_grad for parameter in model.model_g.parameters()):
        raise ValueError('frozen generator parameters would not learn the processed voice')
    data = build_data_module(rows, prepared, output, batch_size, frontend,
                             warmup if benchmark else None)
    report = {'base_sha256': BASE_SHA256, 'prepared_sha256': sha256(prepared / 'trainval.jsonl'),
              'review_sha256': sha256(review) if review else None,
              'frontend_report_sha256': sha256(frontend_report),
              'profile': 'AegisVoicev2.0-PiperPTBR-pilot', 'seed': SEED,
              'batch_size': batch_size, 'manual_accumulation': accumulation,
              'precision': precision, 'learning_rate': 5e-5, 'learning_rate_d': 2.5e-5,
              'resume': str(resume) if resume else None, 'max_epochs': max_epochs,
              'benchmark': benchmark, 'counts': counts,
              'owner_authorization': authorization, 'exclusive_gpu_window': gpu_window,
              'piper_version': __import__('importlib.metadata', fromlist=['version']).version('piper-tts'),
              'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(),
              'phase': 'pilot then owner review; no automatic promotion'}
    repository = Path(__file__).resolve().parents[1]
    report['source_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repository,
        capture_output=True, text=True, check=True).stdout.strip()
    report['source_sha256'] = {name: sha256(repository / name) for name in (
        'training/train_piper.py', 'training/recipe.json', 'training/prepare_dataset.py',
        'training/audio_collection.py', 'training/collection_review.py', 'training/frontend_check.py')}
    if report['piper_version'] != '1.8.0':
        raise ValueError('expected isolated pinned piper-tts 1.8.0 environment')
    report_path = output / 'run.json'
    if resume:
        previous = json.loads(report_path.read_text())
        for key in ('prepared_sha256', 'review_sha256', 'base_sha256', 'frontend_report_sha256',
                    'batch_size', 'manual_accumulation', 'precision', 'seed'):
            if previous[key] != report[key]:
                raise ValueError(f'run identity changed: resume forbidden ({key})')
        if resume.resolve().parent != (output / 'checkpoints').resolve() or not resume.is_file():
            raise ValueError('resume must be a checkpoint from this private experiment')
        checkpoint_index = json.loads((output / 'checkpoint-index.json').read_text())
        if checkpoint_index.get(resume.name, {}).get('sha256') != sha256(resume):
            raise ValueError('resume checkpoint does not match the experiment hash index')
        report['previous_run_sha256'] = sha256(report_path)
    elif report_path.exists():
        raise ValueError('experiment already exists; resume or choose another output')
    report_path.write_text(json.dumps(report, indent=2))
    telemetry = telemetry_callback(torch, Callback, output, counts, warmup if benchmark else 0)
    callbacks = [telemetry]
    if not benchmark:
        class HashedCheckpoint(ModelCheckpoint):
            def _save_checkpoint(self, trainer, filepath):
                super()._save_checkpoint(trainer, filepath)
                index_path = output / 'checkpoint-index.json'
                index = json.loads(index_path.read_text()) if index_path.exists() else {}
                path = Path(filepath)
                index[path.name] = {'sha256': sha256(path), 'bytes': path.stat().st_size,
                                    'epoch': trainer.current_epoch, 'global_step': trainer.global_step}
                index_path.write_text(json.dumps(index, indent=2))

            def _remove_checkpoint(self, trainer, filepath):
                super()._remove_checkpoint(trainer, filepath)
                index_path = output / 'checkpoint-index.json'
                if index_path.exists():
                    index = json.loads(index_path.read_text())
                    index.pop(Path(filepath).name, None)
                    index_path.write_text(json.dumps(index, indent=2))

        callbacks.append(HashedCheckpoint(dirpath=output / 'checkpoints', save_top_k=2, save_last=True,
                                         monitor='val_mel', mode='min', every_n_epochs=1,
                                         filename='epoch={epoch:03d}-val_mel={val_mel:.5f}',
                                         auto_insert_metric_name=False))
    trainer = Trainer(accelerator='gpu', devices=1, precision=precision,
        max_epochs=1 if benchmark else max_epochs, accumulate_grad_batches=1,
        callbacks=callbacks, logger=CSVLogger(str(output), name='loss'), default_root_dir=str(output),
        log_every_n_steps=1, check_val_every_n_epoch=1, enable_progress_bar=True,
        enable_checkpointing=not benchmark, num_sanity_val_steps=0,
        limit_train_batches=warmup + measured if benchmark else 1.0)
    started = time.perf_counter()
    preprocessing_started = time.perf_counter()
    data.prepare_data()
    data.setup('fit')
    # Exercise every cached train/validation sample on CPU before allocating
    # training tensors on the GPU. Loading the final test remains forbidden.
    from piper.train.vits.dataset import UtteranceCollate
    collate = UtteranceCollate(is_multispeaker=False, segment_size=8192)
    for split in (data.train_dataset, data.val_dataset):
        for index in range(len(split)):
            utterance = split[index]
            if (not torch.isfinite(utterance.audio_norm).all()
                    or not torch.isfinite(utterance.spectrogram).all()
                    or utterance.phoneme_ids.numel() < 1):
                raise ValueError('nonfinite/empty cache tensor')
    longest = sorted(indices['train'], key=lambda i: rows[i]['duration'], reverse=True)[:batch_size]
    longest_batch = collate([data.train_dataset.dataset[i] for i in longest])
    result_preparation = {'seconds': time.perf_counter() - preprocessing_started,
                          'loaded_train_validation_examples': len(rows),
                          'longest_train_ids': [rows[i]['id'] for i in longest],
                          'longest_padded_audio_shape': list(longest_batch.audios.shape),
                          'longest_padded_spectrum_shape': list(longest_batch.spectrograms.shape)}
    del longest_batch
    (output / 'data-preflight.json').write_text(json.dumps(result_preparation, indent=2))
    trainer.fit(model, datamodule=data, ckpt_path=str(resume) if resume else None)
    result = {'status': 'benchmark-complete' if benchmark else 'pilot-complete-awaiting-perceptual-evaluation',
              'wall_seconds_including_preprocessing': time.perf_counter() - started,
              'epochs': telemetry.epochs, 'counts': counts, 'data_preflight': result_preparation,
              'checkpoints': [{'name': path.name, 'sha256': sha256(path), 'bytes': path.stat().st_size}
                              for path in sorted((output / 'checkpoints').glob('*.ckpt'))],
              'checkpoint_retention': 'last complete epoch plus two best val_mel checkpoints'}
    if benchmark:
        summary = timing_summary(telemetry.times)
        validation = telemetry.validation_seconds
        result.update(warmup_minibatches=warmup, measured_minibatches=measured,
                      training_minibatch_timing=summary, validation_seconds=validation,
                      projections=project_eta(summary['mean_seconds'], validation, counts),
                      conservative_max_batch_projections=project_eta(summary['max_seconds'], validation, counts),
                      interpretation='CUDA synchronized; forward, backward and both optimizer updates included. '
                      'Projected epoch omits checkpoints and residual I/O; update after full epoch. '
                      'CSV logger does not synthesize validation audio; final test never loaded.')
    (output / ('benchmark.json' if benchmark else 'result.json')).write_text(json.dumps(result, indent=2))
    return result


# Preserve the library entry point, now requiring explicit bound verification.
def train(prepared, checkpoint, output, resume, max_epochs=10, **kwargs):
    return run(prepared, checkpoint, output, resume, max_epochs=max_epochs, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--frontend-report', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--precision', choices=['32-true', '16-mixed'], default='32-true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--warmup-minibatches', type=int, default=8)
    parser.add_argument('--measured-minibatches', type=int, default=32)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--owner-authorization', help='identifier of explicit owner authorization')
    parser.add_argument('--exclusive-gpu-window', help='approved operational window identifier')
    args = parser.parse_args()
    if min(args.max_epochs, args.batch_size, args.accumulation) < 1:
        parser.error('epochs, batch size and accumulation must be positive')
    if not args.execute:
        plan = {'status': 'plan-only', 'downloads': False, 'training': False,
                'mode': 'benchmark' if args.benchmark else 'pilot', 'batch_size': args.batch_size,
                'precision': args.precision, 'manual_accumulation': args.accumulation,
                'requires': ['hash-bound owner auditory/rights approval', 'bound successful frontend check',
                             'local verified checkpoint', 'explicit exclusive GPU window >=9000MiB free']}
        if not args.benchmark:
            plan['requires'].append('completed ASR/suspicion review and documented reconstruction/effect risk')
        manifest = args.prepared / 'trainval.jsonl'
        if manifest.is_file():
            rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
            split = partition_indices(rows)
            plan['counts'] = batch_counts(len(split['train']), len(split['validation']), args.batch_size, args.accumulation)
        print(json.dumps(plan, indent=2))
        return
    if not args.owner_authorization or not args.exclusive_gpu_window or not args.frontend_report:
        parser.error('execution requires explicit authorization, approved GPU window and frontend report')
    if args.benchmark and args.resume:
        parser.error('benchmark uses a disposable fresh model; do not resume an experiment')
    gpu_available()
    os.umask(0o077)
    output = private_destination(args.output)
    print(json.dumps(run(args.prepared, args.checkpoint, output, args.resume, max_epochs=args.max_epochs,
        batch_size=args.batch_size, accumulation=args.accumulation, precision=args.precision,
        review=args.review, frontend_report=args.frontend_report, benchmark=args.benchmark,
        warmup=args.warmup_minibatches, measured=args.measured_minibatches,
        authorization=args.owner_authorization, gpu_window=args.exclusive_gpu_window), indent=2))


if __name__ == '__main__':
    main()
