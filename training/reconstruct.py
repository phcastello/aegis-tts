#!/usr/bin/env python3
"""Future CPU spectral-posterior/decoder A/B. No downloads and no training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .prepare_dataset import private_destination, read_pcm, resample_pcm, sha256, write_pcm
from .train_piper import BASE_SHA256


def reconstruct(source: Path, checkpoint: Path, output: Path) -> dict:
    import torch
    from piper.train.vits.lightning import VitsModel
    from piper.train.vits.mel_processing import spectrogram_torch
    torch.set_num_threads(1)
    torch.manual_seed(40733133)
    if sha256(checkpoint) != BASE_SHA256:
        raise ValueError('expected pinned trusted public base checkpoint')
    values, rate = read_pcm(source)
    if not .3 <= len(values) / rate <= 12:
        raise ValueError('use a short synthetic already-processed sample, .3-12 seconds')
    values = resample_pcm(values, rate, 22050)
    output = private_destination(output)
    # Loading a local pickle is restricted to the hash-verified public artifact.
    original = torch.load(checkpoint, map_location='cpu', weights_only=False)
    hparams = dict(original['hyper_parameters'])
    hparams.update(mos_metric=None, warmstart_ckpt=None, vocoder_warmstart_ckpt=None)
    model = VitsModel(**hparams)
    expected_state = model.state_dict()
    state = original['state_dict']
    essential = [k for k in expected_state if k.startswith('model_g.')]
    if any(k not in state or state[k].shape != expected_state[k].shape for k in essential):
        raise ValueError('checkpoint generator/posterior/decoder incompatible')
    model.load_state_dict({k: v for k, v in state.items()
                           if k in expected_state and v.shape == expected_state[k].shape}, strict=False)
    model.eval()
    expected = (22050, 1024, 256, 1024)
    effective = tuple(getattr(model.hparams, name) for name in ('sample_rate','filter_length','hop_length','win_length'))
    if effective != expected:
        raise ValueError('checkpoint feature mismatch')
    started = time.perf_counter()
    with torch.inference_mode():
        waveform = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        spectrum = spectrogram_torch(waveform, 1024, 22050, 256, 1024, center=False)
        lengths = torch.tensor([spectrum.shape[-1]], dtype=torch.long)
        latent, _, _, mask = model.model_g.enc_q(spectrum, lengths)
        rendered = model.model_g.dec(latent * mask).squeeze().numpy().astype(np.float64)
    elapsed = time.perf_counter() - started
    write_pcm(output / 'A-processed-target-22050.wav', values, 22050)
    # Do not silently limit a vocoder that introduces clipping.
    write_pcm(output / 'B-posterior-decoder-22050.wav', rendered, 22050)
    from scipy.signal import stft
    def magnitude(x):
        return np.abs(stft(x, fs=22050, nperseg=1024, noverlap=768, boundary=None)[2])
    a, b = magnitude(values), magnitude(rendered)
    frames = min(a.shape[1], b.shape[1])
    log_spectral_error = float(np.mean(np.abs(np.log(np.maximum(a[:,:frames], 1e-5)) - np.log(np.maximum(b[:,:frames], 1e-5)))))
    report = {'source_sha256': sha256(source), 'checkpoint_sha256': BASE_SHA256,
              'a_sha256': sha256(output / 'A-processed-target-22050.wav'),
              'b_sha256': sha256(output / 'B-posterior-decoder-22050.wav'),
              'seed': 40733133, 'threads': 1, 'device': 'cpu',
              'target_seconds': len(values) / 22050, 'output_seconds': len(rendered) / 22050,
              'wall_seconds_local_monotonic': elapsed, 'rtf': elapsed / (len(values) / 22050),
              'peak': float(np.max(np.abs(rendered))), 'log_spectral_mae': log_spectral_error,
              'effect_preserved': 'requires private owner A/B; spectral loss cannot establish identity',
              'path': 'linear magnitude spectrum -> VITS posterior -> latent -> HiFi-GAN decoder; not mel-only'}
    (output / 'reconstruction.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--owner-authorization')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'plan-only', 'checkpoint_download': False,
                          'requires': 'explicit future owner authorization and local pinned weights'}))
        return
    if not args.owner_authorization:
        parser.error('execution requires explicit future owner authorization')
    os.umask(0o077)
    print(json.dumps(reconstruct(args.source, args.checkpoint, args.output), indent=2))


if __name__ == '__main__':
    main()
