"""CPU-only, hash-bound evidence after the refinement GPU window is restored.

Descriptors and ASR are auxiliary measurements. This module has no model load,
service control, synthesis, waveform alteration or training entry point.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def distribution(values):
    """Between-utterance distribution, excluding absent pitch candidates."""
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return dict(count=0, median=None, mean=None, population_variance=None, minimum=None, maximum=None)
    return dict(count=len(numbers), median=statistics.median(numbers),
                mean=statistics.mean(numbers), population_variance=statistics.pvariance(numbers),
                minimum=min(numbers), maximum=max(numbers))


def asr_for(row, asr):
    # The runner deliberately caches identical audio/text by hash, so an alias
    # can legitimately lack its own ASR id. Never reuse an unrelated hypothesis.
    exact = next((a for a in asr if a['id'] == row['id']), None)
    if exact is None:
        exact = next((a for a in asr if a['audio_sha256'] == row['sha256']
                      and a['expected'] == row['tts_text']), None)
    if exact is None:
        raise ValueError('ASR evidence missing for ' + row['id'])
    if exact['audio_sha256'] != row['sha256'] or exact['expected'] != row['tts_text']:
        raise ValueError('ASR provenance mismatch for ' + row['id'])
    return {k: exact[k] for k in ('hypothesis', 'word_edits', 'reference_tokens', 'word_error_rate')}


def summarize_asr(rows):
    edits = sum(r['asr']['word_edits'] for r in rows)
    words = sum(len(r['asr']['reference_tokens']) for r in rows)
    return dict(word_edits=edits, expected_words=words, pooled_wer=edits / words if words else None)


def numeric_descriptors(descriptor):
    f0 = descriptor['yin_active_f0']
    texture = descriptor['common24k_texture']
    return dict(yin_energy_active_median_hz=f0['median_hz'],
                yin_energy_active_p90_minus_p10_hz=None if f0['p90_hz'] is None or f0['p10_hz'] is None else f0['p90_hz'] - f0['p10_hz'],
                spectral_flatness=texture['spectral_flatness_active_mean'],
                spectral_centroid_hz=texture['spectral_centroid_hz'],
                power_above_4000hz_ratio=texture['power_above_4000hz_ratio'],
                rms_dbfs=texture['rms_dbfs'])


def pooled(rows):
    keys = next((list(r['acoustic_values']) for r in rows if r.get('acoustic_values')), [])
    return dict(observations=len(rows), content=summarize_asr(rows),
                finite_outputs=sum(r['waveform']['finite'] for r in rows),
                outside_unit_samples=sum(r['waveform']['outside_unit_samples'] for r in rows),
                outside_unit_outputs=sum(r['waveform']['outside_unit_samples'] > 0 for r in rows),
                acoustic_between_utterance_distributions={k: distribution([r.get('acoustic_values', {}).get(k) for r in rows]) for k in keys})


def compare(left, right):
    """Preserve identifiers and direction; never classify acoustic desirability."""
    keys = left.get('acoustic_values', {}).keys() & right.get('acoustic_values', {}).keys()
    deltas = {k: right['acoustic_values'][k] - left['acoustic_values'][k]
              for k in sorted(keys) if right['acoustic_values'][k] is not None and left['acoustic_values'][k] is not None}
    return dict(left=left['id'], right=right['id'],
                right_minus_left_word_edits=right['asr']['word_edits'] - left['asr']['word_edits'],
                right_minus_left_wer=right['asr']['word_error_rate'] - left['asr']['word_error_rate'],
                right_minus_left_acoustic_values=deltas,
                left_waveform=left['waveform'], right_waveform=right['waveform'])


def pcm_proxy(row, values, rate):
    """Standard signed PCM16 scaling, with refusal rather than clipping.

    Recompute measurements only; native WAV and runner receipts stay untouched.
    Qwen FLOAT analysis values reconstruct the original signed PCM16 exactly.
    """
    import numpy as np
    rounded = np.rint(np.asarray(values).reshape(-1) * 32768)
    if not np.isfinite(rounded).all() or (rounded < -32768).any() or (rounded > 32767).any():
        return dict(available=False, reason='PCM16 would require clipping or gain change')
    path = Path(__file__).resolve().parents[1] / 'benchmarks/architecture_compare/benchmark.py'
    spec = importlib.util.spec_from_file_location('aegis_refinement_pcm_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pcm = rounded.astype('<i2').tobytes()
    arrivals = row.get('arrivals') or [dict(ms=row['total_seconds'] * 1000, bytes=len(pcm))]
    metrics = module.pcm_metrics(pcm, arrivals, rate)
    names = ('first_pcm_ms', 'first_useful_pcm_available_ms', 'onset_audio_offset_ms', 'estimated_first_audible_playback_ms')
    deltas = {k: metrics[k] - row[k] for k in names if metrics[k] is not None and row.get(k) is not None}
    return dict(available=True, quantization='nearest signed PCM16: round(float *32768); refuse overflow; no waveform written',
                metrics=metrics, standard_minus_original_receipt_ms=deltas)


def seal_inputs(root):
    preflight = read(root / 'preflight.json')
    complete = read(root / 'gpu-complete.json')
    if not complete.get('all_gpu_experiments_complete') or not complete.get('no_weight_updates') or complete.get('additional_training_started') is not False:
        raise ValueError('completed inference-only GPU receipt required')
    window = read(root / 'operations/gpu-window.json')
    restored = read(root / 'operations/restored.json')
    baseline = read(root / 'operations/baseline.json')
    if window.get('status') != 'production_restored' or not restored.get('running') or restored.get('health') != 'healthy':
        raise ValueError('verified healthy production rollback required before CPU audit')
    if any(restored.get(k) != baseline.get(k) for k in ('container_id', 'image_id')):
        raise ValueError('restored container/image differs from operational baseline')
    checkpoint = Path(preflight['checkpoint'])
    actual = {p.name: digest(p) for p in sorted(checkpoint.parent.iterdir()) if p.is_file()}
    if actual != preflight['checkpoint_hashes']:
        raise ValueError('checkpoint set or hashes changed since preflight')
    expected = preflight['checkpoint_sha256']
    if actual.get(checkpoint.name) != expected or complete.get('checkpoint_sha256_after') != expected:
        raise ValueError('selected checkpoint hash mismatch')
    return dict(all_checkpoint_hashes=actual, selected_checkpoint_sha256=expected,
                production_restored=True, container_id=restored['container_id'], image_id=restored['image_id'],
                rollback_utc=window.get('restored_utc'), no_weight_updates=True,
                additional_training_started=False)


def finalize(root):
    os.umask(0o077)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[key] = '2'
    root = Path(root).resolve(strict=True)
    if any((p / '.git').exists() for p in (root, *root.parents)):
        raise ValueError('private audit must remain outside Git')
    seal = seal_inputs(root)
    # Imports occur only after receipt/hash checks, in a CPU-restricted process.
    import numpy as np
    import soundfile as sf
    from training.f5_acoustics import describe, METHOD

    inputs = read(root / 'results.json') + read(root / 'qwen/results.json')
    asr = read(root / 'asr.json')
    chosen = [r for r in inputs if r['stage'] in ('references', 'controlled', 'stability', 'duration')]
    identifiers = [r['id'] for r in chosen]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('duplicate synthesis identifiers')
    cache = root / 'acoustic-cache'
    cache.mkdir(mode=0o700, exist_ok=True)
    output = []
    for row in chosen:
        path = Path(row['path']).resolve(strict=True)
        if digest(path) != row['sha256']:
            raise ValueError('WAV hash mismatch for ' + row['id'])
        values, rate = sf.read(path, dtype='float64', always_2d=True)
        if values.shape[1] != 1 or not values.size:
            raise ValueError('nonempty mono output required: ' + row['id'])
        finite = bool(np.isfinite(values).all())
        outside = int(np.sum(abs(values) > 1))
        descriptor = describe(path, cache, row['sha256']) if finite else None
        item = dict(id=row['id'], stage=row['stage'], case=row['case'], path=str(path),
                    sha256=row['sha256'], nfe=row.get('nfe'), reference_id=row.get('reference_id'), seed=row.get('seed'),
                    asr=asr_for(row, asr),
                    waveform=dict(finite=finite, samples=int(values.size), sample_rate=rate,
                                  outside_unit_samples=outside, outside_unit_ratio=outside / values.size,
                                  peak=float(abs(values).max()) if finite else None),
                    acoustic_values=numeric_descriptors(descriptor) if descriptor else {},
                    standard_pcm16_proxy=pcm_proxy(row, values, rate),
                    timings={k: row[k] for k in ('total_seconds', 'rtf', 'first_pcm_ms', 'first_useful_pcm_available_ms', 'onset_audio_offset_ms', 'estimated_first_audible_playback_ms') if k in row})
        if row.get('blocks'):
            item['continuity'] = {k: row[k] for k in ('blocks', 'block_ids', 'boundaries', 'arrivals', 'immediate_playback_underruns_over_10ms', 'immediate_playback_underrun_ms') if k in row}
        output.append(item)
    by_id = {r['id']: r for r in output}
    refs = [r for r in output if r['stage'] == 'references']
    pairs = []
    for r in refs:
        if r['nfe'] != 16:
            continue
        mate = next((a for a in refs if a['nfe'] == 8 and (a['reference_id'], a['case'], a['seed']) == (r['reference_id'], r['case'], r['seed'])), None)
        if mate is None:
            raise ValueError('NFE16 reference result lacks matched NFE8 pair')
        pairs.append(compare(r, mate))
    if len(pairs) * 2 != len(refs):
        raise ValueError('NFE comparison reference sets differ')
    repeat = by_id.get('selected-repeat')
    ordinary = by_id.get('selected-ordinary')
    stability = None
    if repeat and ordinary:
        if (repeat['reference_id'], repeat['nfe']) != (ordinary['reference_id'], ordinary['nfe']):
            raise ValueError('seed repeat configuration does not match selected ordinary')
        stability = dict(**compare(ordinary, repeat), left_seed=ordinary['seed'], right_seed=repeat['seed'],
                         note='one second seed on ordinary sentence; limited sensitivity check')
    duration = None
    if 'long-budget90' in by_id and 'original-long-16' in by_id:
        duration = dict(**compare(by_id['original-long-16'], by_id['long-budget90']),
                        note='100% legacy and90% corrected decoding differ at onset; complete text retained; not a crop or selected default')
    report = dict(schema='aegis.f5.refinement.technical.v1', utc=datetime.now(timezone.utc).isoformat(),
                  device='cpu', seal=seal, acoustic_method=METHOD, observations=output,
                  input_receipt_sha256={name: digest(root / name) for name in ('preflight.json', 'gpu-complete.json', 'results.json', 'asr.json', 'qwen/results.json', 'operations/gpu-window.json', 'operations/restored.json', 'operations/baseline.json')},
                  matched_reference_nfe16_to_nfe8=pairs,
                  reference_pooled_by_nfe={str(nfe): pooled([r for r in refs if r['nfe'] == nfe]) for nfe in (8, 16)},
                  seed_sensitivity=stability, duration_100_to_90=duration,
                  owner_perceptual_approval=False, subjective_scores=None,
                  interpretation=[
                      'ASR WER is auxiliary, with orthographic ambiguities for names, acronyms, dates and numbers.',
                      'Finite samples, overshoot and descriptor variance flag measurable changes; they do not establish acoustic desirability.',
                      'NFE8 is not declared superior for sounding less human. Robotic identity, unwanted roughness, consonant preservation and continuity require owner listening.',
                      'YIN uses energy-active frames, including unvoiced sounds; its estimates are not a voiced/unvoiced classifier.',
                      'Acoustic population variance is across utterances/references, with matched text and seed within each NFE pair; it is not a subjective identity score.',
                      'F5 timings exclude HTTP and model load. Selected block FIFO timings include instrumentation/file I/O. Qwen includes the existing HTTP/production frontend.',
                      'PCM useful-onset is a digital amplitude/window proxy, not a proof that the first audio is intelligible.',
                      'Standard PCM16 proxy uses round(float*32768); the original F5 runner uses32767. Both receipts remain explicit; native audio is unchanged.',
                      'No audio, model weights, service configuration or training state is modified by this audit.'
                  ])
    destination = root / 'technical-audit.json'
    temp = destination.with_suffix('.json.tmp')
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.chmod(0o600)
    temp.replace(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.root)
    print(json.dumps(dict(output=str(args.root / 'technical-audit.json'), observations=len(result['observations']),
                          matched_nfe_pairs=len(result['matched_reference_nfe16_to_nfe8']), owner_perceptual_approval=False)))


if __name__ == '__main__':
    main()
