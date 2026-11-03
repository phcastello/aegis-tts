"""Common validation for specialized F5. All output is local; no HTTP claim.

Preserve raw FLOAT evidence, reject rather than clip out-of-range conversion,
record warm-up separately, and finish the cohort even if one PCM output fails.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import wave
import numpy as np
import soundfile as sf
from training import f5_evaluate as ev
from training.f5_refine_segments import decode_generated, linguistic_chunks
from training.f5_refine_text import normalize_text


def write_pcm(path, pcm):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(pcm)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--nfe', type=int, default=8)
    p.add_argument('--delivery', choices=['sentence-files', 'whole-file'], default='sentence-files')
    a=p.parse_args()
    if a.repetitions<1 or a.nfe<1:p.error('positive repetitions and nfe required')
    os.umask(0o077);a.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    source=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('aegis_v2_benchmark',source/'benchmarks/v2/benchmark.py')
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    checkpoint=ev.ROOT/'checkpoints/best.safetensors'
    expected_sha='b3faf799babfcdf33768beb8dcc26f7f94c3c297b3b3a89549ab32cb9380fb44'
    if ev.digest(checkpoint)!=expected_sha:raise ValueError('specialized checkpoint integrity failure')
    corpus=Path(b.__file__).parent/'validation.pt-BR.json'
    cases=json.loads(corpus.read_text())
    manifest=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),status='loading',
        command=sys.argv,checkpoint=str(checkpoint),sha256=expected_sha,nfe=a.nfe,repetitions=a.repetitions,
        corpus=str(corpus),corpus_sha256=ev.digest(corpus),code_sha256=ev.digest(__file__),
        metric_code_sha256=ev.digest(b.__file__),normalizer_code_sha256=ev.digest(source/'training/f5_refine_text.py'),
        decoder_code_sha256=ev.digest(source/'training/f5_refine_segments.py'),
        versions={k:importlib.metadata.version(k) for k in ('torch','numpy','soundfile','vocos','safetensors')},
        normalization='f5-refinement-1; lower-case after normalization',
        target_output='mono 24kHz PCM16LE; FLOAT evidence retained unchanged',
        dsp='none; target dataset already robotic',precision='bfloat16 acoustic autocast; float32 vocoder decode',
        seed='40733133 + repetition',delivery=a.delivery,
        scope='local engine/file timing includes normalization, acoustic model, vocoder, conversion and required WAV close; no HTTP or device delay',
        warmup_policy='one non-corpus synthesis recorded separately before warm matrix; not discarded or mislabeled cold',
        onset_rule=b.RULE,
        raw_evidence_policy='FLOAT WAV for every repetition-zero case and every failed PCM conversion; no clipping, gain, trim or normalization',
        performance_evidence_policy='do not compare if concurrent CPU/GPU load differs; monitor externally',
        source_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=source,capture_output=True,text=True,check=True).stdout.strip())
    b.atomic_json(a.output/'manifest.json',manifest)
    torch=ev.bootstrap();started=time.perf_counter()
    model=ev.load_model(checkpoint);v=ev.vocos('cuda')
    r=next(row for row in (json.loads(x) for x in (ev.ROOT/'prepared/train.jsonl').read_text().splitlines()) if row['id']=='aegis_ptbr_0586')
    melpath=ev.ROOT/'prepared/mels'/f"{r['id']}.npy"
    m=np.load(melpath);cond=torch.from_numpy(np.ascontiguousarray(m.T if m.shape[0]==100 else m))[None].cuda()
    torch.cuda.synchronize();load_s=time.perf_counter()-started
    from f5_tts.model.utils import convert_char_to_pinyin
    from f5_probe import duration_frames
    ref_text=r['text'].strip()+' ';ref_frames=int(cond.shape[1])
    manifest.update(status='warming',load_s=load_s,reference_id=r['id'],reference_mel=str(melpath),
        reference_mel_sha256=ev.digest(melpath),reference_frames=ref_frames,
        reference_audio_sha256=ev.digest(r['audio_filepath']),reference_text_sha256=hashlib.sha256(r['text'].encode()).hexdigest(),
        vocoder_sha256=ev.digest(ev.OLD/'f5/vocos/pytorch_model.bin'),cuda_runtime=torch.version.cuda,
        device=torch.cuda.get_device_name(),gpu_resident_allocated_mib=torch.cuda.memory_allocated()/2**20)
    b.atomic_json(a.output/'manifest.json',manifest)

    def synthesize(case, rep, warmup=False):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();chunks=[];pcms=[];values_all=[];blocks=[];effective=[];invalid=False
        spans=linguistic_chunks(case['text'])
        for j,span in enumerate(spans):
            block_start=time.perf_counter();text=normalize_text(span)['tts_text'].lower();effective.append(text)
            tokens=convert_char_to_pinyin([ref_text+text]);frames=duration_frames(ref_frames,ref_text,text,90)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                generated,trajectory=model.sample(cond=cond,text=tokens,duration=frames,steps=a.nfe,
                    cfg_strength=2.,sway_sampling_coef=-1.,seed=40733133+rep,use_epss=False)
            del trajectory
            with torch.inference_mode():
                z,plan=decode_generated(v,generated,ref_frames);values=z[0].float().cpu().numpy()
            values_all.append(values)
            finite=bool(np.isfinite(values).all());peak=float(abs(values).max()) if finite and values.size else None
            outside=int((abs(values)>1).sum()) if finite else None
            block=dict(index=j,samples=len(values),finite=finite,peak_float=peak,outside_unit_samples=outside,
                engine_ready_ms=(time.perf_counter()-start)*1000,engine_block_ms=(time.perf_counter()-block_start)*1000)
            if not finite or not values.size or outside:
                invalid=True;block['outcome']='failed_PCMconversion'
                # Preserve raw evidence; never clip/scale or invent PCM arrival.
            else:
                pcm=np.rint(values*32767).astype('<i2').tobytes();pcms.append(pcm)
                if a.delivery=='sentence-files':
                    path=a.output/f"{case['id']}-r{rep:02d}-b{j:02d}.wav"
                    write_pcm(path,pcm)
                    chunks.append(dict(ms=(time.perf_counter()-start)*1000,bytes=len(pcm)))
                    block.update(path=str(path),available_ms=chunks[-1]['ms'])
                block['outcome']='ok'
            blocks.append(block);del generated,z
        engine_complete_ms=(time.perf_counter()-start)*1000
        values=np.concatenate(values_all);finite=bool(np.isfinite(values).all())
        peak=float(abs(values).max()) if finite and values.size else None
        outside=int((abs(values)>1).sum()) if finite else None
        row=dict(candidate='f5-specialized-nfe'+str(a.nfe),case=case['id'],size=case.get('size','warmup'),
            repetition=rep,condition='warmup' if warmup else 'warm',seed=40733133+rep,
            scope=manifest['scope'],delivery=a.delivery,
            streaming='sequential complete sentence files; full ODE per block; no HTTP and no within-block neural streaming' if a.delivery=='sentence-files' else 'whole file only; no streaming',
            effective_text=effective,source_text=case['text'],engine_complete_ms=engine_complete_ms,
            engine_rtf=engine_complete_ms/1000/(len(values)/24000) if len(values) else None,
            audio_seconds=len(values)/24000,blocks=blocks,
            outcome='failed_PCMconversion' if invalid else 'ok',peak_float=peak,outside_unit_samples=outside,
            nonfinite_samples=int((~np.isfinite(values)).sum()),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        if not invalid:
            pcm=b''.join(pcms)
            if a.delivery=='whole-file':
                path=a.output/f"{case['id']}-r{rep:02d}.wav";write_pcm(path,pcm)
                chunks=[dict(ms=(time.perf_counter()-start)*1000,bytes=len(pcm))]
                row.update(wav=str(path),wav_sha256=ev.digest(path))
            elif rep==0 or warmup:
                # Combined comparison WAV is evidence after delivery, excluded from block timing.
                path=a.output/f"{case['id']}.wav";write_pcm(path,pcm)
                row.update(wav=str(path),wav_sha256=ev.digest(path))
            eof_ms=chunks[-1]['ms']
            row.update(eof_ms=eof_ms,rtf=eof_ms/1000/(len(pcm)/48000),chunks=chunks,
                pcm_sha256=hashlib.sha256(pcm).hexdigest(),**b.pcm_metrics(pcm,chunks))
            if row['first_speech_pcm_available_ms'] is None:row['outcome']='no_speech'
        else:
            row.update(error='FLOAT exceeds PCM16 range or is nonfinite/empty; no clipping and no final-output TTFA/RTF reported',
                partial_valid_blocks=sum(x['outcome']=='ok' for x in blocks))
        if rep==0 or invalid or warmup:
            raw=a.output/f"{case['id']}-r{rep:02d}-raw-float.wav"
            sf.write(raw,values,24000,subtype='FLOAT')
            row.update(raw_float_wav=str(raw),raw_float_sha256=ev.digest(raw),raw_float_archive_scope='debug evidence saved after measured delivery; unchanged samples')
        return row

    try:
        warm=synthesize(dict(id='warmup_noncorpus',text='Esta é uma verificação de aquecimento.'),rep=0,warmup=True)
        b.atomic_json(a.output/'warmup.json',warm)
        manifest.update(status='benchmarking',warmup_outcome=warm['outcome'],warmup_record='warmup.json')
        b.atomic_json(a.output/'manifest.json',manifest);rows=[]
        with (a.output/'results.jsonl').open('x') as f:
            for rep in range(a.repetitions):
                for case in cases:
                    row=synthesize(case,rep);rows.append(row)
                    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                    b.atomic_json(a.output/'summary.json',b.summarize(rows))
                    print(json.dumps({k:row.get(k) for k in ('case','repetition','outcome','peak_float','outside_unit_samples','ttfa_estimated_playback_ms','rtf','engine_rtf','peak_allocated_mib')},ensure_ascii=False),flush=True)
        manifest.update(status='completed',completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),observations=len(rows),
            failures=sum(x['outcome']!='ok' for x in rows),raw_float_cases=sum('raw_float_wav' in x for x in rows))
        b.atomic_json(a.output/'manifest.json',manifest)
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc),failed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        b.atomic_json(a.output/'manifest.json',manifest)
        raise

if __name__=='__main__':main()
