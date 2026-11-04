"""Specialized F5 common delivery executor; default is a stdlib-only validation plan.

Sentence delivery means complete local WAV files, each after a complete ODE.
It does not imply HTTP delivery or neural PCM streaming.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from training import v2_delivery_protocol as protocol

CHECKPOINT=Path('/var/tmp/aegisvoice-v2-training-20261102/checkpoints/best.safetensors')
CHECKPOINT_SHA='b3faf799babfcdf33768beb8dcc26f7f94c3c297b3b3a89549ab32cb9380fb44'
OLD=Path('/var/tmp/aegis-v2-evaluation-20261102')
TARGET=Path('/var/tmp/aegisvoice-v2-training-20261102')

class OOVTokensError(ValueError):
    def __init__(self,tokens,chunk_index):
        self.oov_tokens=sorted(set(tokens));self.failed_chunk_index=chunk_index
        super().__init__('OOV tokens before ODE: '+repr(self.oov_tokens))

def failure_row(args,case,rep,error,warmup=False):
    return dict(candidate=args.candidate,case=case['id'],size=case.get('size','warmup'),repetition=rep,
                condition='first-external-after-load' if warmup and rep==0 else 'common-external-warmup' if warmup else 'warm',
                final_test=args.final_test,seed=protocol.SEED,outcome='failed',error=repr(error),
                oov_tokens=getattr(error,'oov_tokens',[]),failed_chunk_index=getattr(error,'failed_chunk_index',None))

def settings(args):
    return dict(checkpoint=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,nfe=16,cfg_strength=2.,sway_sampling_coef=-1.,
                delivery=args.delivery,reference_id='aegis_ptbr_0586',reference_partition='training',
                precision='BF16 acoustic autocast, FP32 model/vocoder parameters',dsp='none',
                frontend='f5-refinement-1 then lowercase per linguistic chunk',
                scope='Local engine/files; includes frontend, ODE, Vocos, PCM conversion and WAV close; excludes HTTP and hardware playback',
                internal_warmups=0,raw_policy='Unchanged FLOAT for repetition0 and failed PCM conversion; no clipping/gain/trim')

class Backend:
    def __init__(self,args):
        # All neural imports occur after owner/GPU-empty/budget checks.
        import numpy as np
        import soundfile as sf
        from training import f5_evaluate as ev
        from training.f5_refine_segments import decode_generated,linguistic_chunks
        from training.f5_refine_text import normalize_text
        if protocol.sha(CHECKPOINT)!=CHECKPOINT_SHA:raise ValueError('Specialized F5 checkpoint integrity failure')
        self.np=np;self.sf=sf;self.ev=ev;self.decode=decode_generated;self.chunks=linguistic_chunks;self.normalize=normalize_text;self.args=args
        self.torch=ev.bootstrap();started=time.perf_counter();self.model=ev.load_model(CHECKPOINT);self.vocoder=ev.vocos('cuda')
        rows=[json.loads(x) for x in (TARGET/'prepared/train.jsonl').read_text().splitlines()]
        selected=[r for r in rows if r['id']=='aegis_ptbr_0586']
        if len(selected)!=1:raise ValueError('Expected exact specialized F5 training reference0586')
        row=selected[0];melpath=TARGET/'prepared/mels'/f"{row['id']}.npy";mel=np.load(melpath)
        self.cond=self.torch.from_numpy(np.ascontiguousarray(mel.T if mel.shape[0]==100 else mel))[None].cuda()
        self.ref_text=row['text'].strip()+' ';self.ref_frames=int(self.cond.shape[1])
        self.torch.cuda.synchronize()
        if any(p.device.type!='cuda' for model in (self.model,self.vocoder) for p in model.parameters()):raise ValueError('Complete acoustic model and vocoder must remain on GPU')
        if any(p.dtype!=self.torch.float32 for model in (self.model,self.vocoder) for p in model.parameters()):raise ValueError('Expected actual FP32 model/vocoder parameters')
        if any(p.device.type!='cuda' for model in (self.model,self.vocoder) for p in model.buffers()):raise ValueError('Model/vocoder buffers must remain on GPU')
        from f5_tts.model.utils import convert_char_to_pinyin
        from f5_probe import duration_frames
        self.tokens=convert_char_to_pinyin;self.duration=duration_frames
        self.bindings={str(CHECKPOINT):CHECKPOINT_SHA,str(melpath):protocol.sha(melpath),str(Path(row['audio_filepath'])):protocol.sha(row['audio_filepath']),
                       str(OLD/'f5/vocos/pytorch_model.bin'):protocol.sha(OLD/'f5/vocos/pytorch_model.bin'),
                       str(OLD/'f5/vocos/config.yaml'):protocol.sha(OLD/'f5/vocos/config.yaml'),str(OLD/'f5/vocab.hf.txt'):protocol.sha(OLD/'f5/vocab.hf.txt'),
                       str(TARGET/'prepared/train.jsonl'):protocol.sha(TARGET/'prepared/train.jsonl')}
        self.metadata=dict(load_seconds=time.perf_counter()-started,reference_id=row['id'],reference_frames=self.ref_frames,
                           reference_text_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),artifact_sha256=self.bindings,
                           versions={k:importlib.metadata.version(k) for k in ('torch','numpy','soundfile','vocos','safetensors')},
                           device=self.torch.cuda.get_device_name(),cuda_runtime=self.torch.version.cuda,
                           actual_backend_flags=dict(matmul_allow_tf32=self.torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=self.torch.backends.cudnn.allow_tf32,
                                                     cudnn_benchmark=self.torch.backends.cudnn.benchmark,cudnn_deterministic=self.torch.backends.cudnn.deterministic,
                                                     deterministic_algorithms=self.torch.are_deterministic_algorithms_enabled()),
                           gpu_resident_allocated_mib=self.torch.cuda.memory_allocated()/2**20)

    def generate(self,case,rep,warmup=False):
        torch=self.torch;np=self.np;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();arrivals=[];pcms=[];all_values=[];blocks=[];effective=[];invalid=False;reserved_samples=0
        for index,span in enumerate(self.chunks(case['text'])):
            tick=time.perf_counter();text=self.normalize(span)['tts_text'].lower();effective.append(text)
            tokens=self.tokens([self.ref_text+text])
            oov=sorted({token for token in tokens[0] if token not in self.model.vocab_char_map})
            if oov:raise OOVTokensError(oov,index)
            frames=self.duration(self.ref_frames,self.ref_text,text,90)
            reserved_samples+=max(0,frames-self.ref_frames)*256
            if reserved_samples>24000*180:raise ValueError('Declared180s audio bound exceeded; no truncation')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                generated,trajectory=self.model.sample(cond=self.cond,text=tokens,duration=frames,steps=16,cfg_strength=2.,sway_sampling_coef=-1.,seed=protocol.SEED,use_epss=False)
            del trajectory
            with torch.inference_mode():z,decode_plan=self.decode(self.vocoder,generated,self.ref_frames);values=z[0].float().cpu().numpy()
            all_values.append(values);finite=bool(np.isfinite(values).all());outside=int((abs(values)>1).sum()) if finite else None
            block=dict(index=index,samples=len(values),finite=finite,peak_float=float(abs(values).max()) if finite and values.size else None,
                       outside_unit_samples=outside,engine_ready_ms=(time.perf_counter()-start)*1000,
                       engine_block_ms=(time.perf_counter()-tick)*1000)
            if not finite or not values.size or outside:invalid=True;block['outcome']='failed_PCMconversion'
            else:
                pcm=np.rint(values*32767).astype('<i2').tobytes();pcms.append(pcm);block['outcome']='ok'
                if self.args.delivery=='sentence-files':
                    protocol.output_guard(self.args.run,len(pcm)+4096)
                    path=self.args.run/f"{case['id']}-r{rep:02d}-b{index:02d}.wav";protocol.write_pcm(path,pcm)
                    arrivals.append(dict(ms=(time.perf_counter()-start)*1000,bytes=len(pcm)));block.update(path=str(path),available_ms=arrivals[-1]['ms'])
            blocks.append(block);del generated,z
        if not all_values:raise ValueError('No generated chunks')
        values=np.concatenate(all_values);engine_ms=(time.perf_counter()-start)*1000;finite=bool(np.isfinite(values).all())
        row=dict(candidate=self.args.candidate,case=case['id'],size=case.get('size','warmup'),repetition=rep,seed=protocol.SEED,
                 condition='first-external-after-load' if warmup and rep==0 else 'common-external-warmup' if warmup else 'warm',
                 final_test=self.args.final_test,source_text=case['text'],effective_text=effective,delivery=self.args.delivery,
                 streaming='Sequential complete sentence WAVs; full ODE per file; no HTTP or within-file streaming' if self.args.delivery=='sentence-files' else 'Whole file only; no streaming',
                 scope=settings(self.args)['scope'],engine_complete_ms=engine_ms,engine_rtf=engine_ms/1000/(len(values)/24000) if values.size else None,
                 audio_seconds=len(values)/24000,blocks=blocks,outcome='failed_PCMconversion' if invalid else 'ok',
                 peak_float=float(abs(values).max()) if finite and values.size else None,outside_unit_samples=int((abs(values)>1).sum()) if finite else None,
                 nonfinite_samples=int((~np.isfinite(values)).sum()),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                 peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,oov_tokens=[])
        if not invalid:
            pcm=b''.join(pcms)
            if self.args.delivery=='whole-file':
                protocol.output_guard(self.args.run,len(pcm)+4096)
                path=self.args.run/f"{case['id']}-r{rep:02d}.wav";protocol.write_pcm(path,pcm)
                arrivals=[dict(ms=(time.perf_counter()-start)*1000,bytes=len(pcm))];row.update(wav=str(path),wav_sha256=protocol.sha(path))
            elif rep==0 and not warmup:
                protocol.output_guard(self.args.run,len(pcm)+4096)
                path=self.args.run/f"{case['id']}-r00.wav";protocol.write_pcm(path,pcm);row.update(wav=str(path),wav_sha256=protocol.sha(path))
            eof=arrivals[-1]['ms'];row.update(eof_ms=eof,rtf=eof/1000/(len(pcm)/48000),chunks=arrivals,pcm_sha256=hashlib.sha256(pcm).hexdigest(),**protocol.metrics().pcm_metrics(pcm,arrivals))
            if row['first_speech_pcm_available_ms'] is None:row['outcome']='no_speech'
        else:row.update(error='Raw FLOAT out of range/nonfinite/empty; no PCM output TTFA/RTF; no clipping or gain',partial_valid_blocks=sum(b['outcome']=='ok' for b in blocks))
        if (rep==0 and not warmup) or invalid:
            protocol.output_guard(self.args.run,values.nbytes+4096)
            raw=self.args.run/f"{case['id']}-r{rep:02d}-raw-float.wav";self.sf.write(raw,values,24000,subtype='FLOAT')
            row.update(raw_float_wav=str(raw),raw_float_sha256=protocol.sha(raw))
        return row

def execute(args,plan):
    owner=protocol.owner_guard();before=protocol.budget(args.run,protocol.OUTPUT_CAP);gpu_before=protocol.gpu_set([])
    started=time.perf_counter();backend=None;rows=[];jobs=[];warmups=[];error=None;sampler=None
    protocol.save(args.run/'manifest.json',dict(plan=plan,plan_sha256=protocol.sha(args.run/'plan.json'),status='loading',owner=owner,preflight=before,gpu_before=gpu_before))
    try:
        with protocol.metrics().GpuSampler(args.run/'gpu.jsonl',True) as sampler:
            backend=Backend(args);protocol.gpu_set([os.getpid()])
            protocol.save(args.run/'startup.json',dict(process=protocol.identity(os.getpid()),load=backend.metadata,
                          load_complete_elapsed_seconds=time.perf_counter()-started,
                          scope='New worker loads saved specialized weights. Model loading is separate from first synthesis/TTFA; no gateway readiness exists.',internal_warmups=0))
            for index in range(3):
                case=dict(id='common_warmup',text=protocol.WARMUP)
                try:row=backend.generate(case,index,True)
                except Exception as exc:row=failure_row(args,case,index,exc,True)
                warmups.append(row);protocol.save(args.run/'warmups.json',warmups)
                if warmups[-1]['outcome']!='ok':raise RuntimeError('Common external warmup failed')
                protocol.output_guard(args.run)
            with (args.run/'results.jsonl').open('x') as stream:
                for rep in range(args.repetitions):
                    for case in plan['cohort']['cases']:
                        protocol.gpu_set([os.getpid()])
                        try:row=backend.generate(case,rep)
                        except Exception as exc:row=failure_row(args,case,rep,exc)
                        rows.append(row);protocol.append_row(stream,row);job=protocol.asr_job(row,case,plan['cohort']['spoken'][case['id']],args.run)
                        if job:jobs.append(job)
                        protocol.save(args.run/'summary.json',protocol.metrics().summarize(rows));protocol.output_guard(args.run)
            protocol.assert_matrix(rows,plan['cohort'],args.repetitions)
        if sampler.error:raise RuntimeError('GPU sampler failed: '+sampler.error)
        protocol.save(args.run/'gpu-overlap-audit.json',protocol.audit_gpu_samples(args.run/'gpu.jsonl',[os.getpid()]))
    except BaseException as exc:error=repr(exc);raise
    finally:
        sources_unchanged=protocol.source_bindings(plan['source_hashes'])==plan['source_hashes']
        artifacts_unchanged=backend is not None and all(protocol.sha(path)==digest for path,digest in backend.bindings.items())
        final_budget_error=None
        try:final_budget=protocol.budget(args.run,0)
        except Exception as exc:final_budget_error=repr(exc);final_budget=dict(error=final_budget_error)
        protocol.save(args.run/'asr-jobs.json',dict(jobs=jobs,expected_conditions=plan['cohort']['case_denominator'],filtered=False))
        protocol.save(args.run/'execution.json',dict(completed_utc=protocol.utc(),expected_observations=plan['cohort']['case_denominator']*args.repetitions,
                      observations=len(rows),successes=sum(r['outcome']=='ok' for r in rows),error=error or final_budget_error,final_budget_error=final_budget_error,sources_unchanged=sources_unchanged,
                      artifacts_unchanged=artifacts_unchanged,sampled_vram_peak_used_mib=sampler.peak if sampler else None,
                      final_working=final_budget,scope='Process exit/GPU-empty must be verified by owner worker; no persistent service'))
        if not sources_unchanged or (backend is not None and not artifacts_unchanged):raise RuntimeError('Source/artifact changed; evidence invalid')
        if final_budget_error and error is None:raise RuntimeError('Final budget guard failed; execution invalid, receipt preserved')

def main():
    parser=argparse.ArgumentParser(description=__doc__);protocol.common_arguments(parser)
    parser.add_argument('--delivery',choices=('sentence-files','whole-file'),default='sentence-files')
    args=parser.parse_args();os.umask(0o077)
    sources=[Path(__file__),protocol.REPO/'training/f5_evaluate.py',protocol.REPO/'training/f5_refine_segments.py',protocol.REPO/'training/f5_refine_text.py',
             protocol.REPO/'gateway/aegis_speech/normalization.py',protocol.REPO/'benchmarks/architecture_compare/f5_probe.py']
    sources.extend(OLD/'f5/source/src/f5_tts/model'/name for name in ('cfm.py','utils.py','modules.py','backbones/dit.py'))
    plan=protocol.make_plan(args,'f5-specialized-nfe16',settings(args),sources)
    if protocol.plan_or_verify(args,plan):execute(args,plan)
    print(json.dumps(dict(status='executed' if args.action=='execute' else 'prepared_only',run=str(args.run),gpu_executed=args.action=='execute')))

if __name__=='__main__':main()
