#!/usr/bin/env python3
"""CPU posterior/decoder A/B for real treated clips; no downloads or training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.signal import stft

from .prepare_dataset import private_destination, read_pcm, resample_pcm, sha256, write_pcm
from .train_piper import BASE_SHA256, SEED


def load_base(checkpoint: Path):
    """Load one trusted base once for a series of private CPU comparisons."""
    import torch
    from piper.train.vits.lightning import VitsModel
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    if sha256(checkpoint) != BASE_SHA256:
        raise ValueError('expected pinned trusted public base checkpoint')
    original = torch.load(checkpoint, map_location='cpu', weights_only=False)
    hparams = dict(original['hyper_parameters'])
    hparams.update(mos_metric=None, warmstart_ckpt=None, vocoder_warmstart_ckpt=None)
    model = VitsModel(**hparams)
    expected_state = model.state_dict()
    state = original['state_dict']
    essential = [key for key in expected_state if key.startswith('model_g.')]
    if any(key not in state or state[key].shape != expected_state[key].shape for key in essential):
        raise ValueError('checkpoint generator/posterior/decoder incompatible')
    compatible = {key: value for key, value in state.items()
                  if key in expected_state and value.shape == expected_state[key].shape}
    model.load_state_dict(compatible, strict=False)
    model.eval()
    expected = (22050, 1024, 256, 1024)
    effective = tuple(getattr(model.hparams, name)
                      for name in ('sample_rate', 'filter_length', 'hop_length', 'win_length'))
    if effective != expected:
        raise ValueError('checkpoint feature mismatch')
    model.aegis_checkpoint_sha256 = BASE_SHA256
    model.aegis_load_report = {
        'loaded_parameters': len(compatible), 'generator_parameters_verified': len(essential),
        'unloaded_model_keys': sorted(set(expected_state) - set(compatible)),
        'unused_checkpoint_keys': sorted(set(state) - set(compatible)),
        'features': dict(zip(('sample_rate', 'filter_length', 'hop_length', 'win_length'), effective)),
        'num_symbols': model.hparams.num_symbols, 'num_speakers': model.hparams.num_speakers}
    return model


def magnitude(values: np.ndarray, rate=22050):
    return np.abs(stft(values, fs=rate, nperseg=1024, noverlap=768,
                       boundary=None, padded=False)[2])


def texture_metrics(values: np.ndarray, rate: int) -> dict:
    spectrum = magnitude(values, rate)
    power = spectrum ** 2
    mean_power = np.mean(power, axis=1)
    frequencies = np.fft.rfftfreq(1024, 1 / rate)
    total = max(float(mean_power.sum()), 1e-20)
    frame_energy = power.sum(axis=0)
    active = frame_energy > max(float(frame_energy.max(initial=0)) * .001, 1e-20)
    active_power = power[:, active] if active.any() else power
    flatness = np.exp(np.mean(np.log(np.maximum(active_power, 1e-20)), axis=0)) / np.maximum(np.mean(active_power, axis=0), 1e-20)
    edge = max(1, round(rate * .1))
    rms = float(np.sqrt(np.mean(values ** 2)))
    return {'rms_dbfs': float(20 * np.log10(max(rms, 1e-12))),
            'peak': float(np.max(np.abs(values), initial=0)),
            'spectral_centroid_hz': float(np.sum(mean_power * frequencies) / total),
            'spectral_flatness_active_mean': float(np.mean(flatness)),
            'power_above_4000hz_ratio': float(mean_power[frequencies > 4000].sum() / total),
            'power_above_8000hz_ratio': float(mean_power[frequencies > 8000].sum() / total),
            'power_above_11025hz_ratio': float(mean_power[frequencies > 11025].sum() / total),
            'first_100ms_rms': float(np.sqrt(np.mean(values[:edge] ** 2))),
            'last_100ms_rms': float(np.sqrt(np.mean(values[-edge:] ** 2)))}


def compare_spectra(target: np.ndarray, rendered: np.ndarray) -> dict:
    a, b = magnitude(target), magnitude(rendered)
    frames = min(a.shape[1], b.shape[1])
    a, b = a[:, :frames], b[:, :frames]
    floor = 1e-5
    log_difference = np.log(np.maximum(a, floor)) - np.log(np.maximum(b, floor))
    return {'log_spectral_mae': float(np.mean(np.abs(log_difference))),
            'log_spectral_distance_db': float(np.sqrt(np.mean((20 / np.log(10) * log_difference) ** 2))),
            'spectral_convergence': float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-20)),
            'compared_spectral_frames': frames}


def reconstruct(source: Path, checkpoint: Path, output: Path, *, model=None) -> dict:
    import torch
    from piper.train.vits.mel_processing import spectrogram_torch
    if model is None:
        model = load_base(checkpoint)
    elif getattr(model, 'aegis_checkpoint_sha256', None) != BASE_SHA256:
        raise ValueError('shared reconstruction model is not the verified base')
    torch.manual_seed(SEED)
    original_values, rate = read_pcm(source)
    if not .3 <= len(original_values) / rate <= 30:
        raise ValueError('use an existing treated clip between .3 and 30 seconds')
    values = resample_pcm(original_values, rate, 22050)
    output = private_destination(output)
    if any(output.iterdir()):
        raise ValueError('reconstruction output must be new or empty; preserve previous A/B artifacts')
    started = time.perf_counter()
    with torch.inference_mode():
        waveform = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        spectrum = spectrogram_torch(waveform, 1024, 22050, 256, 1024, center=False)
        lengths = torch.tensor([spectrum.shape[-1]], dtype=torch.long)
        latent, _, _, mask = model.model_g.enc_q(spectrum, lengths)
        rendered = model.model_g.dec(latent * mask).squeeze().numpy().astype(np.float64)
    elapsed = time.perf_counter() - started
    if not np.isfinite(rendered).all() or not len(rendered):
        raise ValueError('decoder reconstruction is empty or nonfinite')
    write_pcm(output / 'A-processed-target-22050.wav', values, 22050)
    # No normalization, limiter, effects or phase alignment disguise base error.
    write_pcm(output / 'B-posterior-decoder-22050.wav', rendered, 22050)
    source_metrics = texture_metrics(original_values, rate)
    target_metrics, output_metrics = texture_metrics(values, 22050), texture_metrics(rendered, 22050)
    comparison = compare_spectra(values, rendered)
    risks = []
    if source_metrics['power_above_11025hz_ratio'] > .01:
        risks.append('source has >1% spectral power above 22050Hz target Nyquist; inspect lost high-frequency texture')
    if abs(target_metrics['rms_dbfs'] - output_metrics['rms_dbfs']) > 3:
        risks.append('decoder changes overall RMS by >3dB')
    if abs(target_metrics['spectral_centroid_hz'] - output_metrics['spectral_centroid_hz']) > 500:
        risks.append('decoder changes spectral centroid by >500Hz; inspect robotic texture/sibilants')
    if abs(len(rendered) - len(values)) > 1024:
        risks.append('decoder duration differs by more than one analysis window')
    report = {'source_sha256': sha256(source), 'source_path': str(source.resolve()),
              'checkpoint_sha256': BASE_SHA256,
              'a_sha256': sha256(output / 'A-processed-target-22050.wav'),
              'b_sha256': sha256(output / 'B-posterior-decoder-22050.wav'),
              'seed': SEED, 'threads': 1, 'device': 'cpu',
              'source_sample_rate': rate, 'target_seconds': len(values) / 22050,
              'output_seconds': len(rendered) / 22050, 'duration_difference_seconds': (len(rendered) - len(values)) / 22050,
              'wall_seconds_local_monotonic': elapsed, 'rtf': elapsed / (len(values) / 22050),
              'peak': output_metrics['peak'], **comparison,
              'source_metrics': source_metrics, 'target_metrics': target_metrics, 'output_metrics': output_metrics,
              'checkpoint_compatibility': model.aegis_load_report, 'risks': risks,
              'effect_preserved': 'unreviewed; requires private owner A/B. Spectral metrics cannot establish vocal identity, robotization or reverb fidelity.',
              'interpretation': 'Base posterior/decoder ceiling before fine-tuning. Decoder remains trainable; initial mismatch is recorded as risk, not proof fine-tuning cannot learn it.',
              'path': 'linear magnitude spectrum -> VITS posterior -> latent -> HiFi-GAN decoder; not mel-only'}
    (output / 'reconstruction.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def reconstruct_many(manifest: Path, identifiers: list[str], checkpoint: Path, output: Path) -> dict:
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows) or not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError('unique manifest IDs and requested IDs required')
    selected = [by_id[identifier] for identifier in identifiers]
    if any(row['split'] not in ('train', 'validation') for row in selected):
        raise ValueError('final test is not used for reconstruction selection')
    for row in selected:
        if sha256(Path(row['audio_filepath'])) != row['sha256']:
            raise ValueError('reconstruction source hash changed')
    model = load_base(checkpoint)
    output = private_destination(output)
    reports = [reconstruct(Path(row['audio_filepath']), checkpoint, output / row['id'], model=model)
               for row in selected]
    summary = {'manifest_sha256': sha256(manifest), 'checkpoint_sha256': BASE_SHA256,
               'ids': identifiers, 'device': 'cpu', 'examples': reports,
               'owner_ab_approval': 'pending; no perceptual approval inferred'}
    (output / 'reconstruction-summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source', type=Path)
    inputs.add_argument('--manifest', type=Path, help='prepared trainval.jsonl, never final test')
    parser.add_argument('--ids', nargs='+', help='existing train/validation IDs for multiple comparisons')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--owner-authorization')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'plan-only', 'checkpoint_download': False,
                          'requires': 'owner authorization and local pinned weights; real treated audio only'}))
        return
    if not args.owner_authorization:
        parser.error('execution requires explicit owner authorization')
    if (args.manifest is None) != (args.ids is None):
        parser.error('--manifest and --ids must be supplied together')
    os.umask(0o077)
    result = (reconstruct_many(args.manifest, args.ids, args.checkpoint, args.output)
              if args.manifest else reconstruct(args.source, args.checkpoint, args.output))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
