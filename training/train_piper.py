#!/usr/bin/env python3
"""Plan by default. Training needs future explicit owner authorization and window."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

from .prepare_dataset import private_destination, sha256

BASE_SHA256 = 'abd3e6c7237aca5b7cf9c3cfc50945bfef44a83a594e00186f584d9d11fd355a'


def partition_indices(rows: list[dict]) -> dict[str, list[int]]:
    result = {'train': [], 'validation': []}
    for index, row in enumerate(rows):
        if row['split'] not in result:
            raise ValueError('final test examples must never enter the trainer')
        result[row['split']].append(index)
    if not all(result.values()):
        raise ValueError('empty train or validation partition')
    return result


def gpu_available(minimum_free_mib: int = 9000) -> None:
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, check=True, text=True)
    values = [int(x.strip()) for x in result.stdout.splitlines() if x.strip()]
    if not values or min(values) < minimum_free_mib:
        raise RuntimeError('exclusive GPU capacity unavailable; never stop production automatically')


def build_data_module(rows, prepared: Path, output: Path):
    from piper.train.vits.dataset import VitsDataModule
    from torch.utils.data import Subset

    class FixedSplitDataModule(VitsDataModule):
        def setup(self, stage):
            super().setup(stage)
            full = self.train_dataset.dataset
            if len(full) != len(rows):
                raise RuntimeError('Piper skipped a clip; do not shift split membership')
            indices = partition_indices(rows)
            self.train_dataset = Subset(full, indices['train'])
            self.val_dataset = Subset(full, indices['validation'])
            # Upstream synthesizes test_dataset at each validation epoch.
            # Use validation examples there; final test stays entirely offline.
            self.test_dataset = Subset(full, indices['validation'][:5])

    return FixedSplitDataModule(csv_path=prepared / 'metadata.csv', audio_dir=prepared / 'audio',
        cache_dir=output / 'cache', config_path=output / 'AegisVoicev2.0.onnx.json',
        voice_name='AegisVoicev2.0-PiperPTBR-pilot', espeak_voice='pt-br', sample_rate=22050,
        batch_size=2, num_symbols=256, num_speakers=1, num_workers=1,
        trim_silence=False, validation_split=0, num_test_examples=0,
        filter_length=1024, hop_length=256, win_length=1024, segment_size=8192)


def train(prepared: Path, checkpoint: Path, output: Path, resume: Path | None, max_epochs: int = 10):
    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from piper.train.vits.lightning import VitsModel
    torch.set_num_threads(1)
    seed_everything(40733133, workers=True)
    rows = [json.loads(line) for line in (prepared / 'trainval.jsonl').read_text().splitlines()]
    partition_indices(rows)
    with (prepared / 'metadata.csv').open() as stream:
        csv_rows = list(csv.reader(stream, delimiter='|'))
    expected_rows = [[Path(row['audio_filepath']).name, row['text']] for row in rows]
    if csv_rows != expected_rows:
        raise ValueError('metadata order or transcript diverged from reviewed manifest')
    # Only the pinned public artifact is loaded as a base; pickle checkpoint
    # inputs must remain operator-owned and must never come from API uploads.
    if sha256(checkpoint) != BASE_SHA256:
        raise ValueError('public base checkpoint SHA256 mismatch')
    for row in rows:
        if sha256(Path(row['audio_filepath'])) != row['sha256']:
            raise ValueError('prepared audio changed since review/import')
    original = torch.load(checkpoint, map_location='cpu', weights_only=False)
    hparams = dict(original['hyper_parameters'])
    hparams.update(batch_size=2, learning_rate=5e-5, learning_rate_d=2.5e-5,
                   sample_rate=22050, filter_length=1024, hop_length=256, win_length=1024,
                   mel_channels=80, mel_fmin=0.0, mel_fmax=None, mos_metric=None,
                   warmstart_ckpt=None, vocoder_warmstart_ckpt=None)
    model = VitsModel(**hparams)
    state = original['state_dict']
    essential = [k for k in model.state_dict() if k.startswith('model_g.')]
    if any(k not in state or state[k].shape != model.state_dict()[k].shape for k in essential):
        raise ValueError('incompatible generator/posterior/vocoder checkpoint; no partial random voice')
    compatible = {k: v for k, v in state.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(compatible, strict=False)
    # Generator, spectral posterior, flow, text encoder and decoder all train.
    if any(not parameter.requires_grad for parameter in model.model_g.parameters()):
        raise ValueError('frozen generator parameters would not learn the processed voice')
    data = build_data_module(rows, prepared, output)
    report = {'base_sha256': BASE_SHA256, 'prepared_sha256': sha256(prepared / 'trainval.jsonl'),
              'profile': 'AegisVoicev2.0-PiperPTBR-pilot', 'resume': str(resume) if resume else None,
              'max_epochs': max_epochs, 'phase': 'pilot then review; no automatic promotion'}
    report_path = output / 'run.json'
    if resume:
        previous = json.loads(report_path.read_text())
        if previous['prepared_sha256'] != report['prepared_sha256']:
            raise ValueError('dataset/splits changed: resume forbidden')
    report_path.write_text(json.dumps(report, indent=2))
    callback = ModelCheckpoint(dirpath=output / 'checkpoints', save_top_k=3, save_last=True,
                               monitor='val_mel', mode='min')
    trainer = Trainer(accelerator='gpu', devices=1, precision='32-true', max_epochs=max_epochs,
        accumulate_grad_batches=4, callbacks=[callback], default_root_dir=str(output),
        log_every_n_steps=10, check_val_every_n_epoch=1, enable_progress_bar=True)
    trainer.fit(model, datamodule=data, ckpt_path=str(resume) if resume else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--owner-authorization', help='identifier of explicit future authorization')
    parser.add_argument('--exclusive-gpu-window', help='operational window identifier, never stop production here')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'plan-only', 'downloads': False, 'training': False,
                          'requires': ['processed reviewed dataset', 'explicit owner authorization',
                                       'local verified checkpoint', 'exclusive GPU window >=9000MiB free']}, indent=2))
        return
    if not args.owner_authorization or not args.exclusive_gpu_window:
        parser.error('execution requires explicit future owner authorization and exclusive window')
    if args.max_epochs < 1:
        parser.error('max-epochs must be positive')
    gpu_available()
    os.umask(0o077)
    output = private_destination(args.output)
    train(args.prepared, args.checkpoint, output, args.resume, args.max_epochs)


if __name__ == '__main__':
    main()
