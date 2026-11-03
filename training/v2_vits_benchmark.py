"""VITS complete-file benchmark using the common final-PCM onset metrics."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import wave

from training import v2_vits as vits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--root', type=Path, default=vits.ROOT)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--gpu-authorized', action='store_true')
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--candidate', default='vits-cv-pt-base')
    p.add_argument('--case', action='append')
    p.add_argument('--noise-scale', type=float, default=None,
                   help='Override native inference_noise_scale; omitted preserves model config')
    p.add_argument('--duration-noise-scale', type=float, default=None,
                   help='Override native inference_noise_scale_dp; omitted preserves model config')
    a = p.parse_args(); os.umask(0o077)
    for value in (a.noise_scale,a.duration_noise_scale):
        if value is not None and (not math.isfinite(value) or value < 0):
            p.error('noise scales must be finite and nonnegative')
    a.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    spec = importlib.util.spec_from_file_location('v2bench', vits.REPO/'benchmarks/v2/benchmark.py')
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    args = SimpleNamespace(root=a.root, device=a.device, gpu_authorized=a.gpu_authorized,
                           threads=a.threads, checkpoint=a.checkpoint)
    started = time.perf_counter()
    torch, np, model, config, normalizer = vits.bootstrap(args)
    step = vits.load_trained(args, model, torch)
    if a.noise_scale is not None:model.inference_noise_scale=a.noise_scale
    if a.duration_noise_scale is not None:model.inference_noise_scale_dp=a.duration_noise_scale
    sampling=dict(inference_noise_scale=float(model.inference_noise_scale),
                  inference_noise_scale_dp=float(model.inference_noise_scale_dp),
                  length_scale=float(model.length_scale))
    model.to(a.device).eval()
    if a.device == 'cuda': torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    from scipy.signal import resample_poly
    corpus = vits.REPO/'benchmarks/v2/validation.pt-BR.json'
    cases = json.loads(corpus.read_text())
    if a.case: cases = [case for case in cases if case['id'] in a.case]
    if not cases: raise ValueError('no benchmark cases')
    b.atomic_json(a.output/'manifest.json', dict(candidate=a.candidate, checkpoint=str(a.checkpoint),
                  checkpoint_sha256=vits.digest(a.checkpoint) if a.checkpoint else vits.BASE_SHA,
                  step=step, corpus_sha256=vits.digest(corpus), code_sha256=vits.digest(__file__),
                  engine_code_sha256=vits.digest(vits.__file__), load_seconds=load_seconds,
                  sampling=sampling,requested_noise_overrides=dict(noise_scale=a.noise_scale,duration_noise_scale=a.duration_noise_scale),
                  format='PCM16LE mono 24000Hz', native_rate=22050, concurrency=1,
                  warmup='One non-corpus synthesis recorded separately before warm matrix; no slow call discarded',
                  normalization='f5-refinement-1 experimental PT-BR normalization, A-é-jis owner target; checkpoint graphemes',
                  normalizer_code_sha256=vits.digest(vits.REPO/'training/f5_refine_text.py'),
                  resample='scipy resample_poly 160/147 Kaiser10',
                  scope='complete-file engine: frontend, generator and integrated vocoder, final resample, PCM conversion, WAV write and close; no HTTP',
                  dsp='none; dataset already processed robotic. Base model not target identity.',
                  onset_rule=b.RULE, runtime='coqui-tts0.28.0', torch=torch.__version__))
    rows = []
    order = [(-1, dict(id='non-corpus-warmup',size='short',text='Esta é uma verificação de aquecimento.'))]
    order += [(rep, case) for rep in range(a.repetitions) for case in cases]
    for repetition, case in order:
            seed = 40733133 + max(0,repetition)
            torch.manual_seed(seed)
            if a.device == 'cuda':
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            start = time.perf_counter()
            ids = vits.tokens(model, normalizer, case['text'])
            token = torch.tensor(ids, device=a.device)[None]
            output = model.inference(token, aux_input=dict(speaker_ids=torch.zeros(1,dtype=torch.long,device=a.device),
                                                           language_ids=torch.zeros(1,dtype=torch.long,device=a.device)))
            raw = output['model_outputs'].squeeze().float().cpu().numpy()
            final = resample_poly(raw, 160, 147, window=('kaiser',10)).astype(np.float32)
            row = dict(candidate=a.candidate,case=case['id'],size=case['size'],repetition=repetition,seed=seed,
                       condition='first-model-synthesis-warmup' if repetition == -1 else 'warm',
                       source_text=case['text'],effective_text=normalizer.normalize(case['text']).normalized_text.lower(),
                       scope='complete-file local engine including all conversion and WAV close; no HTTP',
                       streaming='one complete file; no neural streaming',step=step,device=a.device,sampling=sampling)
            if not final.size or not np.isfinite(final).all() or np.abs(final).max() > 1:
                import soundfile as sf
                sf.write(a.output/f"{case['id']}-r{repetition:02d}-invalid.wav",final,24000,subtype='FLOAT')
                row.update(outcome='failed',error='nonfinite, empty or overshoot; raw FLOAT preserved, no concealed clipping',
                           peak=float(np.abs(final).max()) if final.size and np.isfinite(final).all() else None)
            else:
                pcm = np.rint(final*32767).astype('<i2').tobytes()
                path = a.output/f"{case['id']}-r{repetition:02d}.wav"
                with wave.open(str(path),'wb') as wav:
                    wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(pcm)
                elapsed = (time.perf_counter()-start)*1000
                chunks = [dict(ms=elapsed,bytes=len(pcm))]
                row.update(eof_ms=elapsed,rtf=elapsed/1000/(len(pcm)/48000),chunks=chunks,outcome='ok',
                           wav=str(path),pcm_sha256=hashlib.sha256(pcm).hexdigest(),wav_sha256=vits.digest(path),
                           **b.pcm_metrics(pcm,chunks))
                if repetition > 0:
                    path.unlink(); row.pop('wav');row.pop('wav_sha256')
            if a.device == 'cuda':
                row.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                           peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
            if repetition == -1:
                b.atomic_json(a.output/'warmup.json',row)
                print(json.dumps(dict(warmup=True,outcome=row['outcome'],seconds=row.get('eof_ms',0)/1000)),flush=True)
                continue
            rows.append(row)
            with (a.output/'results.jsonl').open('a') as stream:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            b.atomic_json(a.output/'summary.json',b.summarize(rows))
            print(json.dumps({k:row.get(k) for k in ('case','repetition','outcome','ttfa_estimated_playback_ms','rtf','error')}),flush=True)


if __name__ == '__main__':
    main()
