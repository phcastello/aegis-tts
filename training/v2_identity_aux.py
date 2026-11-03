#!/usr/bin/env python3
"""Read-only, bounded CPU identity clues; never an identity or listening verdict.

Reuses the local Qwen conditioning encoder and existing acoustic descriptors.
Its embeddings favor its own baseline and are not an independent verifier of
robotic timbre. Private manifests contain hashes, not reference transcripts.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import time
from types import SimpleNamespace

ROOT = Path('/var/tmp/aegis-v2-work')
REPO = Path(__file__).resolve().parents[1]
MODEL = Path('/home/phc/aegis-tts/models/Qwen3-TTS-12Hz-0.6B-Base')
SOURCE = ROOT/'qwen-sft-research/qwen_tts/core/models'
SOURCE_SHA = '25c42656bcf810f06ef6bc1839bd7083f3c8cfedac3a147c4060b4262b1c96a0'
CONFIG_SOURCE_SHA = 'f52867f14fde06a416dd14864d503ce6d13d0a08d5f5da30191e1c80c13f5d18'
MODEL_SHA = '180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6'
CLASS_NAMES = ('Res2NetBlock', 'SqueezeExcitationBlock', 'AttentiveStatisticsPooling',
               'TimeDelayNetBlock', 'SqueezeExcitationRes2NetBlock', 'Qwen3TTSSpeakerEncoder')
FUNCTION_NAMES = ('dynamic_range_compression_torch', 'mel_spectrogram')
LIMITATIONS = [
    'Auxiliary observations only: no automated winner, identity threshold, listening or robotic-timbre verdict.',
    'The ECAPA-like encoder belongs to Qwen itself; comparison is biased toward the Qwen baseline.',
    'The three reserved real anchors have different words from the common generated validation sentences.',
    'Speaker cosine can react to text, prosody, DSP and recording conditions; three anchors do not calibrate a verifier.',
    'Active-RMS matching is an analysis gain, not perceptual loudness/LUFS matching; no original WAV is changed.',
    'YIN uses an energy mask, not a voiced/unvoiced classifier; unvoiced and octave errors remain possible.',
    'High-frequency band power is not a phoneme-aligned sibilance or metallic-character score.',
    'Same-requested-text DTW still has effective-text/frontend and pronunciation confounds.',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def effective_text_sha(value):
    # F5 receipts preserve one string per synthesis block. Hash joined text
    # for comparison; retain the original receipt hash for block boundaries.
    if isinstance(value, list) and all(isinstance(v, str) for v in value): value = ' '.join(value)
    if not isinstance(value, str): raise ValueError('Effective text must be a string or list of block strings')
    return text_sha(value)


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temporary.chmod(0o600); temporary.replace(path)


def private_dir(path):
    path = Path(path).resolve()
    if path == REPO or REPO in path.parents: raise ValueError('Private output must be outside Git')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def cpu_guard():
    if (ROOT/'pause-asr').exists(): raise RuntimeError('CPU benchmark pause-asr is active; retry after release')
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        raise RuntimeError('CUDA already initialized; this helper is CPU-only')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[key] = '2'


def build_manifest(output, benchmark_dirs, piper_dirs=()):
    """Select only reserved anchors and rep0 of the seven public validation texts."""
    corpus_path = REPO/'benchmarks/v2/validation.pt-BR.json'
    corpus = {r['id']: r for r in json.loads(corpus_path.read_text(encoding='utf-8'))}
    source = ROOT/'data-audit/identity-anchors.private.json'
    anchors = json.loads(source.read_text(encoding='utf-8'))
    items = []
    for r in anchors:
        if r['split'] != 'validation': raise ValueError('Reserved validation anchors required')
        items.append(dict(id=r['id'], role='reserved_real_anchor', candidate='aegis-real',
            path=r['audio_filepath'], sha256=r['audio_sha256'], requested_text_sha256=text_sha(r['text']),
            effective_text_sha256=None, split='validation', group_id=r['group_id']))
    receipts = []
    for directory in benchmark_dirs:
        directory = Path(directory).resolve()
        receipt = directory/'results.jsonl'
        rows = [json.loads(line) for line in receipt.read_text(encoding='utf-8').splitlines() if line.strip()]
        manifest = directory/'manifest.json'
        if sha(corpus_path) != json.loads(manifest.read_text(encoding='utf-8'))['corpus_sha256']:
            raise ValueError('Benchmark validation corpus differs from frozen validation')
        receipts.extend([dict(path=str(p), sha256=sha(p)) for p in (receipt, manifest)])
        selected = set()
        for row in rows:
            if row.get('repetition') != 0 or row.get('case') not in corpus: continue
            if row['case'] in selected: raise ValueError('Duplicate rep0 case')
            selected.add(row['case'])
            if row.get('outcome') != 'ok': raise ValueError('Only successful final-product WAVs allowed')
            requested = text_sha(corpus[row['case']]['text'])
            if row.get('text_sha256', requested) != requested: raise ValueError('Requested text hash differs')
            wave = Path(row['wav']); wave = wave if wave.is_absolute() else directory/wave
            items.append(dict(id=directory.name+':'+row['case'], role='generated_validation',
                candidate=directory.name, case=row['case'], path=str(wave), sha256=row['wav_sha256'],
                requested_text_sha256=requested,
                effective_text_sha256=effective_text_sha(row['effective_text']) if row.get('effective_text') else None,
                input_scope='final user-facing product PCM WAV; rep0 only', source_receipt=str(receipt)))
        if selected != set(corpus): raise ValueError('Benchmark must have all seven validation cases')
    for directory in piper_dirs:
        directory = Path(directory).resolve(); receipt = directory/'results.json'
        rows = json.loads(receipt.read_text(encoding='utf-8'))
        receipts.append(dict(path=str(receipt), sha256=sha(receipt)))
        for row in rows:
            if row.get('case') not in corpus or row.get('outcome') != 'ok': raise ValueError('Validation successes required')
            items.append(dict(id=directory.name+':'+row['id'], role='generated_language_baseline',
                candidate=directory.name+':'+row['frontend'], case=row['case'], path=row['wav'], sha256=row['sha256'],
                requested_text_sha256=text_sha(corpus[row['case']]['text']),
                effective_text_sha256=text_sha(row['text']) if row.get('text') else None,
                input_scope='CPU language baseline final product PCM; not a trained finalist', source_receipt=str(receipt)))
    if len({r['id'] for r in items}) != len(items): raise ValueError('Duplicate item IDs')
    if len(items) > 40: raise ValueError('Bounded pass supports at most40 clips')
    result = dict(schema='aegis.identity-aux.manifest.v1', items=items, limitations=LIMITATIONS,
        validation_corpus_sha256=sha(corpus_path), anchor_manifest_sha256=sha(source), receipts=receipts)
    save(output, result)
    return result


def encoder_inventory():
    """Read the safetensors header and only the76 speaker tensors (~18 MB)."""
    path = MODEL/'model.safetensors'
    with path.open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        if size > 2*1024*1024: raise ValueError('Unexpected safetensors header size')
        raw = f.read(size); header = json.loads(raw)
        selected = {k: v for k, v in header.items() if k.startswith('speaker_encoder.')}
        h = hashlib.sha256()
        for name, row in sorted(selected.items()):
            h.update(name.encode()); h.update(json.dumps(row, sort_keys=True).encode())
            start, stop = row['data_offsets']; f.seek(8+size+start)
            values = f.read(stop-start)
            if len(values) != stop-start: raise ValueError('Truncated speaker tensor')
            h.update(values)
    inventory = ROOT/'operations/qwen-model-inventory.json'
    old = next(r for r in json.loads(inventory.read_text(encoding='utf-8'))['files'] if r['path']=='model.safetensors')
    st = path.stat()
    if (st.st_size, st.st_ino, st.st_dev) != (old['bytes'], old['inode'], old['device']):
        raise ValueError('Local Qwen model no longer matches inventoried file identity')
    if old['sha256'] != MODEL_SHA or len(selected) != 76: raise ValueError('Unexpected inventoried Qwen encoder')
    return dict(path=str(path), model_sha256_previously_verified=MODEL_SHA,
        full_model_not_rehashed_here=True, inventory_sha256=sha(inventory),
        selected_tensors=len(selected), selected_elements=sum(math.prod(v['shape']) for v in selected.values()),
        selected_bytes=sum(v['data_offsets'][1]-v['data_offsets'][0] for v in selected.values()),
        selected_content_and_header_sha256=h.hexdigest(), safetensors_header_sha256=hashlib.sha256(raw).hexdigest(),
        source_sha256=sha(SOURCE/'modeling_qwen3_tts.py'), config_source_sha256=sha(SOURCE/'configuration_qwen3_tts.py'),
        model_revision='5d83992436eae1d760afd27aff78a71d676296fc',
        source_revision='022e286b98fbec7e1e916cb940cdf532cd9f488e', license='Apache-2.0')


def load_encoder():
    """Compile unchanged upstream classes/frontend; no transformers or TTS load."""
    import numpy as np
    import torch
    from librosa.filters import mel
    from safetensors import safe_open
    metadata = encoder_inventory()
    if metadata['source_sha256'] != SOURCE_SHA or metadata['config_source_sha256'] != CONFIG_SOURCE_SHA:
        raise ValueError('Pinned Qwen source changed')
    torch.set_num_threads(2)
    tree = ast.parse((SOURCE/'modeling_qwen3_tts.py').read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in CLASS_NAMES+FUNCTION_NAMES]
    if {n.name for n in nodes} != set(CLASS_NAMES+FUNCTION_NAMES): raise ValueError('Incomplete source isolation')
    namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, librosa_mel_fn=mel,
                     Qwen3TTSSpeakerEncoderConfig=SimpleNamespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE/'modeling_qwen3_tts.py'), 'exec'), namespace)
    config_tree = ast.parse((SOURCE/'configuration_qwen3_tts.py').read_text(encoding='utf-8'))
    cls = next(n for n in config_tree.body if isinstance(n, ast.ClassDef) and n.name=='Qwen3TTSSpeakerEncoderConfig')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='__init__')
    defaults = dict(zip([a.arg for a in init.args.args[1:]], [ast.literal_eval(n) for n in init.args.defaults]))
    explicit = json.loads((MODEL/'config.json').read_text(encoding='utf-8'))['speaker_encoder_config']
    if set(explicit)-set(defaults): raise ValueError('Unknown speaker configuration keys')
    config = SimpleNamespace(**(defaults | explicit))
    model = namespace['Qwen3TTSSpeakerEncoder'](config).float().eval()
    with safe_open(str(MODEL/'model.safetensors'), framework='pt', device='cpu') as f:
        state = {k.removeprefix('speaker_encoder.'): f.get_tensor(k).float() for k in f.keys() if k.startswith('speaker_encoder.')}
    if any(not torch.isfinite(t).all() for t in state.values()): raise ValueError('Nonfinite encoder weights')
    model.load_state_dict(state, strict=True)
    metadata.update(strict_load=True, runtime_dtype='float32', checkpoint_dtype='BF16', effective_config=vars(config),
                    torch_version=torch.__version__, numpy_version=np.__version__)
    return model, namespace['mel_spectrogram'], metadata


def active_rms(values, rate=24000):
    """RMS over nonoverlapping10ms blocks above the fixed/relative energy gate."""
    import numpy as np
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all(): raise ValueError('Finite nonempty mono required')
    block = max(1, round(rate*.01)); powers = []
    for start in range(0, len(y), block): powers.append(float(np.mean(y[start:start+block]**2)))
    powers = np.asarray(powers); gate = max(10**(-55/20), math.sqrt(float(powers.max()))*10**(-30/20))
    selected = powers > gate**2
    if not selected.any(): raise ValueError('No active audio above absolute/relative RMS gate')
    # Last block may be shorter: preserve sample weighting rather than average block lengths.
    lengths = np.minimum(block, len(y)-np.arange(len(powers))*block)
    rms = math.sqrt(float(np.sum(powers[selected]*lengths[selected])/lengths[selected].sum()))
    return rms, dict(block_seconds=.01, absolute_gate_dbfs=-55, relative_gate_db=-30,
                     active_blocks=int(selected.sum()), total_blocks=len(powers), threshold=gate)


def matched_gains(waves, target_dbfs=-23., peak_ceiling=.95):
    import numpy as np
    if not -60 <= target_dbfs <= -6 or not 0 < peak_ceiling < 1: raise ValueError('Invalid gain policy')
    measurements = [(active_rms(y)[0], float(np.max(np.abs(y)))) for y in waves]
    target = min(10**(target_dbfs/20), *(peak_ceiling*rms/peak for rms, peak in measurements))
    gains = [target/rms for rms, peak in measurements]
    return gains, dict(method='global linear gain matching active RMS; no limiting/clipping; originals unchanged',
        requested_target_dbfs=target_dbfs, effective_target_dbfs=20*math.log10(target), peak_ceiling=peak_ceiling,
        perceptual_loudness_lufs=False)


def embed(values, model, mel_function):
    import numpy as np
    import torch
    with torch.no_grad():
        mels = mel_function(torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0),
            n_fft=1024, num_mels=128, sampling_rate=24000, hop_size=256,
            win_size=1024, fmin=0, fmax=12000).transpose(1, 2)
        vector = model(mels)[0].cpu().numpy().astype(np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm <= 0: raise ValueError('Invalid speaker embedding')
    return vector/norm, norm


def dtw_same_requested_text(a, b, a_text, b_text, max_cells=500000):
    """Constrained time alignment of same-request log-mels; not pronunciation scoring."""
    import numpy as np
    if not a_text or a_text != b_text: raise ValueError('DTW requires exactly the same requested text hash')
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]: raise ValueError('Matching feature axes required')
    if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Nonfinite DTW features')
    n, m = len(a), len(b)
    if min(n,m) < 1 or max(n,m) > 2000: raise ValueError('DTW frame budget exceeded')
    width = max(2, math.ceil(.25*max(n,m)))
    cells = sum(max(0, min(m,round(i*m/n)+width)-max(1,round(i*m/n)-width)+1) for i in range(1,n+1))
    if cells > max_cells: raise ValueError('DTW cell budget exceeded')
    prev = np.full(m+1, np.inf); prev[0] = 0.; counts = np.zeros(m+1, dtype=np.int64)
    for i in range(1,n+1):
        current = np.full(m+1, np.inf); current_counts = np.zeros(m+1, dtype=np.int64)
        center = round(i*m/n)
        lo, hi = max(1,center-width), min(m,center+width)
        distances = np.mean(np.abs(b[lo-1:hi]-a[i-1]), axis=1)
        for j in range(lo, hi+1):
            # Diagonal wins ties; complete closed-begin/closed-end path required.
            neighbors = (prev[j-1], prev[j], current[j-1]); index = int(np.argmin(neighbors))
            current[j] = neighbors[index]+distances[j-lo]
            current_counts[j] = (counts[j-1], counts[j], current_counts[j-1])[index]+1
        prev, counts = current, current_counts
    if not np.isfinite(prev[m]) or not counts[m]: raise ValueError('No complete DTW path within band')
    return dict(mean_absolute_log_mel_per_path_step=float(prev[m]/counts[m]), path_steps=int(counts[m]),
                frames=[n,m], band_frames=width, evaluated_cells=cells, same_requested_text=True,
                interpretation='auxiliary aligned spectral difference; not intelligibility or identity')


def analyze(manifest_path, output, encoder=True, dtw_cases=()):
    cpu_guard(); os.nice(19)
    import numpy as np
    import soundfile as sf
    from scipy.signal import stft
    from training.prepare_dataset import resample_pcm
    from training.reconstruct import texture_metrics
    from training.f5_acoustics import active_f0, METHOD as F0_METHOD
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    items = manifest['items']
    if not 1 <= len(items) <= 40 or len({r['id'] for r in items}) != len(items): raise ValueError('Invalid clip budget/IDs')
    started = time.perf_counter(); waves = []; native = []
    for item in items:
        cpu_guard(); path = Path(item['path']).resolve(strict=True)
        if sha(path) != item['sha256']: raise ValueError('Audio hash mismatch: '+item['id'])
        info = sf.info(path)
        if info.channels != 1 or not .2 <= info.duration <= 40: raise ValueError('Require mono clips of0.2–40s')
        y, rate = sf.read(path, dtype='float64', always_2d=False)
        if len(y) != info.frames or not np.isfinite(y).all(): raise ValueError('Nonfinite/truncated WAV')
        native.append(dict(sample_rate=rate, samples=len(y), seconds=len(y)/rate, subtype=info.subtype,
            texture=texture_metrics(y, rate), samples_at_or_above_unit=int(np.sum(np.abs(y)>=1)),
            samples_outside_unit=int(np.sum(np.abs(y)>1))))
        waves.append(resample_pcm(y, rate, 24000))
    gains, gain_policy = matched_gains(waves)
    model = mel_function = encoder_metadata = None
    if encoder:
        cpu_guard(); model, mel_function, encoder_metadata = load_encoder()
    rows = []; vectors = {}; dtw_features = {}
    for item, y, gain, original in zip(items, waves, gains, native):
        cpu_guard(); normalized = y*gain
        pitch, _, _ = active_f0(normalized)
        row = dict(item, native=original, analysis_gain=gain, analysis_gain_db=20*math.log10(gain),
            common24k_matched_texture=texture_metrics(normalized, 24000), yin_energy_gated_f0=pitch,
            active_rms=active_rms(normalized)[0], active_gate=active_rms(normalized)[1])
        power = np.abs(stft(normalized, fs=24000, nperseg=1024, noverlap=768, boundary=None, padded=False)[2])**2
        mean_power = power.mean(axis=1); frequencies = np.fft.rfftfreq(1024, 1/24000)
        total = max(float(mean_power.sum()), 1e-20)
        row['broad_high_frequency_energy'] = {f'{lo}_{hi}_hz_ratio':float(mean_power[(frequencies>=lo)&(frequencies<hi)].sum()/total)
            for lo, hi in [(4000,8000),(8000,12001)]}
        if encoder:
            vectors[item['id']], row['embedding_original_l2_norm'] = embed(normalized, model, mel_function)
        if item.get('case') in dtw_cases:
            import librosa
            magnitude = np.abs(librosa.stft(normalized.astype(np.float32), n_fft=1024, hop_length=480, center=True))
            matrix = librosa.filters.mel(sr=24000, n_fft=1024, n_mels=32, fmin=0, fmax=12000)
            dtw_features[item['id']] = np.log(np.maximum(matrix@magnitude,1e-5)).T
        rows.append(row)
        save(output/'progress.json',dict(completed=len(rows),total=len(items),last_id=item['id']))
    anchors = [r for r in rows if r['role']=='reserved_real_anchor']
    cosine = []; anchor_pairs = []
    if encoder:
        if len(anchors) != 3: raise ValueError('Exactly three reserved anchors required for encoder comparison')
        centroid = np.mean([vectors[r['id']] for r in anchors],axis=0); centroid /= np.linalg.norm(centroid)
        for a, b in itertools.combinations(anchors,2):
            anchor_pairs.append(dict(a=a['id'],b=b['id'],cosine=float(vectors[a['id']]@vectors[b['id']]),same_text=False))
        for row in rows:
            if row in anchors: continue
            cosine.append(dict(id=row['id'],candidate=row['candidate'],case=row.get('case'),
                anchor_centroid_cosine=float(vectors[row['id']]@centroid),
                per_anchor=[dict(anchor=a['id'],cosine=float(vectors[row['id']]@vectors[a['id']]),
                    same_requested_text=row['requested_text_sha256']==a['requested_text_sha256']) for a in anchors],
                verdict=None, text_confound=True))
        np.savez_compressed(output/'embeddings.private.npz',**vectors)
        (output/'embeddings.private.npz').chmod(0o600)
    dtw = []
    for a, b in itertools.combinations(rows,2):
        if a['id'] not in dtw_features or b['id'] not in dtw_features or a.get('case') != b.get('case'): continue
        if a['requested_text_sha256'] != b['requested_text_sha256']: continue
        try: pair = dtw_same_requested_text(dtw_features[a['id']],dtw_features[b['id']],a['requested_text_sha256'],b['requested_text_sha256'])
        except ValueError as error: pair = dict(skipped=str(error))
        dtw.append(dict(a=a['id'],b=b['id'],case=a['case'],**pair,
            effective_text_equal=(a['effective_text_sha256']==b['effective_text_sha256']) if a['effective_text_sha256'] and b['effective_text_sha256'] else None))
    result = dict(schema='aegis.identity-aux.results.v1',utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), manifest_sha256=sha(manifest_path), source_receipts=manifest.get('receipts',[]),
        items=rows, normalization=gain_policy, f0_method=F0_METHOD, encoder=encoder_metadata,
        cosine_auxiliary=cosine, anchor_pair_auxiliary=anchor_pairs, same_requested_text_dtw=dtw,
        limitations=LIMITATIONS, automatic_identity_verdict=None, heard_audio=False,
        cpu_only=True,cuda_initialized=sys.modules['torch'].cuda.is_initialized() if 'torch' in sys.modules else False,
        wall_seconds=time.perf_counter()-started, max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    save(output/'results.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['manifest','inventory','analyze'])
    parser.add_argument('--output',type=Path,default=ROOT/'identity-aux')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--benchmark-dir',type=Path,action='append',default=[])
    parser.add_argument('--piper-dir',type=Path,action='append',default=[])
    parser.add_argument('--without-encoder',action='store_true')
    parser.add_argument('--dtw-cases',nargs='*',default=[])
    args=parser.parse_args(); os.umask(0o077); output=private_dir(args.output)
    if args.action=='manifest':
        value=build_manifest(output/'manifest.private.json', args.benchmark_dir,args.piper_dir)
        print(json.dumps(dict(items=len(value['items']),manifest=str(output/'manifest.private.json'))))
    elif args.action=='inventory':
        value=encoder_inventory(); save(output/'encoder-inventory.json',value)
        print(json.dumps({k:value[k] for k in ['selected_tensors','selected_elements','selected_bytes','license']}))
    else:
        if args.manifest is None: raise ValueError('--manifest is required')
        if (output/'results.json').exists(): raise ValueError('Use a fresh output directory to preserve receipts')
        value=analyze(args.manifest,output,not args.without_encoder,args.dtw_cases)
        print(json.dumps({k:value[k] for k in ['cpu_only','cuda_initialized','wall_seconds','max_rss_mib']}))


if __name__=='__main__': main()
