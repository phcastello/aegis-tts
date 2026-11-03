"""Small fixed TRAIN0586 reference control; default is a CPU-only plan.

Execution belongs exclusively to the owner GPU worker. No training, service,
continuation, cache merging, model download or quality promotion occurs here.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import wave

from training import v2_voxcpm_cache_probe as capacity
from training import v2_voxcpm_optimized_probe as optimized
from training import v2_voxcpm_probe as pilot
from training import v2_voxcpm_streaming_probe as native

REPO = Path(__file__).resolve().parents[1]
ROOT = optimized.ROOT
CASES = ('v_command', 'v_pressure', 'v_long')
REFERENCE = Path('/var/tmp/aegisvoice-v2-training-20261102/prepared/audio/aegis_ptbr_0586.wav')
REFERENCE_SHA = '197b86b91143efa8ca83b07af7eb0ea26b16b9253dcf0c8a17fb0531d94c5397'
CHECKPOINT = ROOT/'identity-aux/snapshots/voxcpm2-step800-78680488bcf4/checkpoint.pt'
CHECKPOINT_SHA = 'b9160041e0773799ab20e2066fb526f7151425bb8055491add375b1007501f33'
OUTPUT_CAP = 128*1024**2
RESERVE = 1024**3  #512MiB cache +128MiB output +128MiB checkpoint overlap +256MiB margin.


def generation(text):
    return dict(target_text=text, min_len=2, max_len=400, inference_timesteps=10,
                cfg_value=2., retry_badcase=False, streaming_prefix_len=4, seed=40733133)


def guard(text_tokens, prompt_cache, maximum=400, slots=1024):
    """Exact prefix length, full requested generation and one spare slot."""
    if any(type(v) is not int or v<0 for v in (text_tokens,maximum,slots)) or not maximum or not slots:
        raise ValueError('Nonnegative integer counts and positive maximum/capacity required')
    patches=0
    if prompt_cache is not None:
        if not isinstance(prompt_cache,dict) or set(prompt_cache)!={'mode','ref_audio_feat'} or prompt_cache['mode']!='reference':
            raise ValueError('Only None or fixed reference-only cache is allowed; no continuation/merge')
        feat=prompt_cache['ref_audio_feat']
        if tuple(feat.shape)!=(64,4,64) or str(feat.dtype)!='torch.float32' or feat.device.type!='cpu':
            raise ValueError('Expected immutable CPU FP32 reference features [64,4,64]')
        patches=int(feat.shape[0])
    prefix=text_tokens+1+(patches+2 if patches else 0)
    needed=prefix+maximum+1
    if needed>slots:raise ValueError(f'KV needs {needed} slots, capacity={slots}; refuse truncation')
    return dict(text_tokens=text_tokens,reference_patches=patches,prefix_slots=prefix,
                requested_max_len=maximum,reserved_slots=needed,capacity=slots,spare_slots=slots-needed)


@contextlib.contextmanager
def guarded_cache_entry(model,state):
    original=model._generate_with_prompt_cache
    present='_generate_with_prompt_cache' in vars(model)
    previous=vars(model).get('_generate_with_prompt_cache')
    def dispatch(*args,**kwargs):
        if args:raise ValueError('Use named cached-generation arguments')
        text=kwargs.get('target_text')
        if not isinstance(text,str) or not text.strip():raise ValueError('Nonempty target text required')
        state['guard']=guard(len(model.text_tokenizer(text)),kwargs.get('prompt_cache'),kwargs.get('max_len',2000))
        return original(**kwargs)
    model._generate_with_prompt_cache=dispatch
    try:yield
    finally:
        if present:model._generate_with_prompt_cache=previous
        else:delattr(model,'_generate_with_prompt_cache')


def cached_audio(model,kwargs,prompt_cache):
    """Close the actual upstream tuple iterator on EOF, error or cancellation."""
    stream=None
    try:
        stream=iter(model.generate_with_prompt_cache_streaming(**kwargs,prompt_cache=prompt_cache))
        for item in stream:
            if not isinstance(item,tuple) or len(item)!=3:raise ValueError('Expected native (wav,tokens,features) tuple')
            yield item[0]
    finally:
        if stream is not None and hasattr(stream,'close'):stream.close()


def trials():
    #Three excluded calls per mode/case; subsequent n3 have counterbalanced A/B order.
    warm=[('warmup',case,mode,rep) for case in CASES for rep in range(3) for mode in ('A','B')]
    measured=[('warm',case,mode,rep) for rep in range(3) for case in CASES
              for mode in (('A','B') if rep%2==0 else ('B','A'))]
    return warm+measured


def private(path):
    path=path.resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Artifacts/cache must stay inside the accounted private working root')
    if any((p/'.git').exists() for p in (path,*path.parents)):raise ValueError('Private artifacts must stay outside Git')
    path.mkdir(parents=True,exist_ok=True,mode=0o700)
    return path


def bindings(args):
    data=REFERENCE.parents[1]
    rows=[json.loads(x) for x in (data/'train.jsonl').read_text().splitlines() if x.strip()]
    found=[r for r in rows if r['id']=='aegis_ptbr_0586']
    if len(found)!=1:raise ValueError('Require unique TRAIN0586 binding')
    row=found[0]
    val=[json.loads(x) for x in (data/'validation.jsonl').read_text().splitlines() if x.strip()]
    if row['split']!='train' or any(r['id']==row['id'] or r['group_id']==row['group_id'] for r in val):
        raise ValueError('Reference must be TRAIN with no validation source-group overlap')
    if Path(row['audio_filepath'])!=REFERENCE or row['sha256']!=REFERENCE_SHA or pilot.sha(REFERENCE)!=REFERENCE_SHA:
        raise ValueError('TRAIN0586 WAV binding differs')
    sidecar=args.checkpoint.with_suffix('.sha256.json')
    cp=json.loads(sidecar.read_text())
    if cp['step'] not in (800,1400) or cp['sha256']!=args.checkpoint_sha256 or pilot.sha(args.checkpoint)!=args.checkpoint_sha256:
        raise ValueError('Frozen saved step800/1400 checkpoint binding differs')
    code=[Path(__file__),REPO/'training/v2_voxcpm_cache_probe.py',Path(native.__file__),
          Path(optimized.__file__),Path(pilot.__file__),REPO/'benchmarks/v2/benchmark.py',
          REPO/'benchmarks/v2/validation.pt-BR.json',REPO/'gateway/aegis_speech/experimental_text.py',
          REPO/'gateway/aegis_speech/experimental_refinement.py',
          args.source/'src/voxcpm/model/voxcpm2.py',args.source/'src/voxcpm/model/utils.py']
    revision=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    if revision!=pilot.SOURCE_REV or subprocess.check_output(['git','-C',str(args.source),'status','--porcelain'],text=True).strip():
        raise ValueError('Require clean pinned Vox source')
    return dict(source_revision=revision,reference_sha256=REFERENCE_SHA,reference_text_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
        reference_group=row['group_id'],reference_split='train',checkpoint_step=cp['step'],
        checkpoint_sha256=args.checkpoint_sha256,checkpoint_sidecar_sha256=pilot.sha(sidecar),
        manifests={n:pilot.sha(data/n) for n in ('train.jsonl','validation.jsonl')},
        model_metadata={n:pilot.sha(args.model/n) for n in pilot.METADATA_BLOBS},
        code={str(p):pilot.sha(p) for p in code})


def load_with_capacity(args,torch,cls,lora_cls,state):
    #Reuse the existing loader hook only; restore all shared process hooks before timing.
    with capacity.capacity_hooks(1024,2,state):
        model,loaded=native.load_model(args,torch,cls,lora_cls)
        if '_generate' not in vars(model):raise RuntimeError('Expected loader zero-shot guard hook')
        delattr(model,'_generate')  #Restore pinned class method; legacy equivalence is explicitly guarded below.
    return model,loaded


@contextlib.contextmanager
def signals_for_cleanup():
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGHUP)}
    def stop(sig,_frame):raise SystemExit(128+sig)
    try:
        for sig in previous:signal.signal(sig,stop)
        yield
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def output_check(args,additional=0):
    from training.v2_language_monitor import physical_usage
    used=physical_usage([args.run])['allocated_bytes']
    if used+additional>OUTPUT_CAP:raise RuntimeError('Private output128MiB cap would be exceeded')
    if shutil.disk_usage(args.run).free<5*1024**3+additional:raise RuntimeError('Physical reserve below5GiB plus pending output')
    return used


def collect(model,kwargs,cache,*,legacy=False,cancel_after=0):
    import numpy as np
    captured=[]
    def convert(chunk):
        value=native.native_numpy(chunk);captured.append(value.copy());return value
    factory=(lambda:model.generate_streaming(**kwargs)) if legacy else (lambda:cached_audio(model,kwargs,cache))
    pcm,row=native.collect_stream(factory,convert=convert,cancel_after_native_chunks=cancel_after)
    raw=np.concatenate(captured) if captured else np.empty(0,dtype=np.float32)
    row['pcm_sha256']=hashlib.sha256(pcm).hexdigest()
    row['native_nonfinite_samples']=int((~np.isfinite(raw)).sum())
    row['native_overshoot_samples']=int((np.abs(raw)>1).sum())
    row['float_capture']='Copy of each native CPU chunk inside timer, equally for A/B; no model-state copy'
    return pcm,row,raw


def save_audio(args,label,pcm,raw,row):
    import soundfile as sf
    output_check(args,len(pcm)+raw.nbytes+16384)
    if raw.size:
        path=args.run/(label+'-native48000-float.wav');sf.write(path,raw,48000,subtype='FLOAT')
        row.update(raw_wav=str(path),raw_wav_sha256=pilot.sha(path))
    if pcm:
        path=args.run/(label+'-24000.wav')
        with wave.open(str(path),'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000);f.writeframes(pcm)
        row.update(wav=str(path),sha256=pilot.sha(path))


def execute(args):
    parent=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().decode(errors='replace')
    if 'v2_gpu_worker' not in parent:raise RuntimeError('Exclusive owner GPU-worker execution required')
    if not (ROOT/'pause-asr').exists():raise RuntimeError('Require owner pause-asr before timing')
    if (args.run/'recipe.json').exists():raise ValueError('Fresh execution directory required')
    optimized.require_idle_cpu();before=bindings(args)
    weights=pilot.verify(args);pilot.guard_gpu(args);optimized.configure_cache(args)
    inventory=optimized.working_usage()
    if inventory.get('permission_errors') or inventory['allocated_bytes']+RESERVE>40_000_000_000:
        raise RuntimeError('Working40GB budget cannot cover declared cache/output/checkpoint reserve')
    if shutil.disk_usage(args.run).free<5*1024**3+RESERVE:raise RuntimeError('Need5GiB physical reserve plus1GiB peak')
    output_check(args)
    sys.path.insert(0,str(REPO/'gateway'))
    from aegis_speech.experimental_text import normalizer_for
    corpus={r['id']:r for r in json.loads((REPO/'benchmarks/v2/validation.pt-BR.json').read_text())}
    effective={case:normalizer_for('ptbr-numbers-v1').normalize(corpus[case]['text']).normalized_text for case in CASES}
    torch,cls,lora_cls,*_=pilot.imports(args)
    torch.set_float32_matmul_precision('highest')
    stop=threading.Event();watch=threading.Thread(target=optimized.budget_watch,args=(args,stop,inventory),daemon=True)
    original_replay=torch.cuda.CUDAGraph.replay;replays=[0];state={};records=[];status='failed';error=None;weights_after=None
    def replay(graph,*a,**kw):replays[0]+=1;return original_replay(graph,*a,**kw)
    with signals_for_cleanup():
        watch.start()
        try:
            model,loaded=load_with_capacity(args,torch,cls,lora_cls,state)
            pilot.atomic_json(args.run/'recipe.json',dict(utc=optimized.utc(),bindings=before,weights=weights,
                loader=loaded,kv=capacity.cache_record(model,1024),cache_limit_mib=512,output_cap_bytes=OUTPUT_CAP,
                working=inventory,reserved_extra_bytes=RESERVE,checkpoint_overlap_reserve_bytes=128*1024**2,
                generation=generation('<fixed validation text>'),frontend='ptbr-numbers-v1',topology='lora-r8-BF16-unmerged',
                matmul_precision=torch.get_float32_matmul_precision(),resampling=native.HalfRateFIR().recipe(),
                scope='Native local client only; immediate playback, no gateway/startup320 or auditory approval',
                warmup='Three excluded calls per case/mode; n3 A/B counterbalanced, same seed',trials=trials()))
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
            ref=model.build_prompt_cache(reference_wav_path=str(REFERENCE),trim_silence_vad=False)
            torch.cuda.synchronize();build_ms=(time.perf_counter()-begin)*1000
            feat=ref['ref_audio_feat'];guard(0,ref)
            if not torch.isfinite(feat).all():raise ValueError('Reference cache contains nonfinite features')
            feature_sha=hashlib.sha256(feat.contiguous().numpy().tobytes()).hexdigest()
            pilot.atomic_json(args.run/'reference-cache.json',dict(cold_build_ms=build_ms,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                shape=list(feat.shape),dtype=str(feat.dtype),device=str(feat.device),bytes=feat.numel()*feat.element_size(),
                feature_sha256=feature_sha,reference_sha256=REFERENCE_SHA,residency=optimized.residency(model,torch),
                staging='Official API returns CPU latent inputs, packs CPU and transfers prefix per request; model/VAE/KV stay CUDA',
                reuse='Fixed cache once; prefill/KV are recomputed each request; no merging'))
            with guarded_cache_entry(model,state):
                for case in CASES:
                    kw=generation(effective[case]);guard(len(model.text_tokenizer(kw['target_text'])),None)
                    checks=[];pairs=[]
                    for legacy in (True,False):
                        optimized.require_idle_cpu();torch.cuda.synchronize();torch.compiler.cudagraph_mark_step_begin()
                        pcm,row,raw=collect(model,kw,None,legacy=legacy)
                        row['guard']=guard(len(model.text_tokenizer(kw['target_text'])),None)
                        checks.append(row);pairs.append((legacy,pcm,row,raw))
                    equal=all(checks[0][k]==checks[1][k] for k in ('outcome','pcm_sha256','native_float32_sha256','native_samples','audio_seconds'))
                    for legacy,pcm,row,raw in pairs:
                        label='equivalence-'+case+('-legacy' if legacy else '-cached')
                        if not equal or row['outcome']!='ok':save_audio(args,label,pcm,raw,row)
                        pilot.atomic_json(args.run/(label+'.json'),row)
                    pilot.atomic_json(args.run/('equivalence-'+case+'.json'),dict(case=case,exact_equal=equal,rows=checks))
                    if not equal or checks[0]['outcome']!='ok':raise RuntimeError('Zero-cache API equivalence failed: '+case)
                for condition,case,mode,rep in trials():
                    if not (ROOT/'pause-asr').exists():raise RuntimeError('Owner timing pause was removed')
                    optimized.require_idle_cpu();output_check(args);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    torch.cuda.CUDAGraph.replay=replay if condition=='warmup' else original_replay
                    torch.compiler.cudagraph_mark_step_begin();kw=generation(effective[case])
                    pcm,row,raw=collect(model,kw,ref if mode=='B' else None)
                    torch.cuda.synchronize();label=f'{condition}-{case}-{mode}-r{rep:02d}'
                    tokens=len(model.text_tokenizer(kw['target_text']));cap=min(tokens*6+10,400)
                    row.update(condition=condition,case=case,mode=mode,repetition=rep,
                        requested_text=corpus[case]['text'],text=kw['target_text'],generation=kw,guard=state['guard'],
                        text_token_count=tokens,effective_ar_length_cap=cap,potential_length_cap_hit=len(row['native_chunks'])>=cap,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
                    if (condition=='warm' and rep==0) or row['outcome']!='ok' or row['potential_length_cap_hit']:
                        save_audio(args,label,pcm,raw,row)
                    output_check(args,len(json.dumps(row).encode())*3+16384)
                    pilot.atomic_json(args.run/(label+'.json'),row);records.append(row)
                    with (args.run/'results.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                    pilot.atomic_json(args.run/'progress.json',dict(utc=optimized.utc(),condition=condition,case=case,mode=mode,repetition=rep,outcome=row['outcome']))
                    optimized.residency(model,torch);capacity.cache_record(model,1024)
                    if row['outcome']!='ok' or row['potential_length_cap_hit']:raise RuntimeError('Invalid/capped native response: '+label)
            if feature_sha!=hashlib.sha256(feat.contiguous().numpy().tobytes()).hexdigest():raise RuntimeError('Fixed reference features changed')
            if not replays[0]:raise RuntimeError('No actual CUDAgraph replay proved during warmup')
            weights_after=pilot.verify(args)
            if bindings(args)!=before or weights_after!=weights:raise RuntimeError('Input/source/checkpoint/model hashes changed')
            summary={m:native.summarize([r for r in records if r['mode']==m],corpus) for m in ('A','B')}
            pilot.atomic_json(args.run/'summary.json',dict(modes=summary,cudagraph_replays_in_warmup=replays[0],
                actual_cuda_graph_replay=True,diagnostics=optimized.diagnostics(torch),
                quality='No ASR/listening/identity approval; future pinned ASR and paired analysis required',
                next_product_gate='If useful, all7 and actual gateway320ms with zero gaps and material TTFA gain'))
            output_check(args);status='completed_measurement'
        except BaseException as exc:
            error=repr(exc);raise
        finally:
            torch.cuda.CUDAGraph.replay=original_replay;stop.set();watch.join(2)
            try:after=bindings(args);binding_error=None
            except Exception as exc:after=None;binding_error=repr(exc)
            pilot.atomic_json(args.run/'execution.json',dict(status=status,error=error,utc=optimized.utc(),
                records=len(records),bindings_after=after,binding_audit_error=binding_error,
                source_and_input_bindings_unchanged=after==before,weights_after=weights_after,
                saved_weight_files_verified_unchanged=weights_after==weights))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['plan','execute'],default='plan')
    p.add_argument('--source',type=Path,default=pilot.ROOT/'source');p.add_argument('--model',type=Path,default=pilot.ROOT/'model')
    p.add_argument('--run',type=Path,default=pilot.ROOT/'reference0586-step800')
    p.add_argument('--cache',type=Path,default=pilot.ROOT/'compile-cache')
    p.add_argument('--checkpoint',type=Path,default=CHECKPOINT);p.add_argument('--checkpoint-sha256',default=CHECKPOINT_SHA)
    p.add_argument('--gpu-authorized',action='store_true')
    a=p.parse_args();os.umask(0o077);a.run=private(a.run);a.cache=private(a.cache)
    a.topology='lora';a.rank=8;a.optimize=True;a.matmul_precision='highest';a.cache_limit_mib=512
    if a.action=='plan':
        plan=dict(status='prepared_only',bindings=bindings(a),source_rev=pilot.SOURCE_REV,model_rev=pilot.MODEL_REV,
            generation=generation('<numbers frontend, frozen validation only>'),cases=CASES,trials=trials(),
            reference=str(REFERENCE),checkpoint=str(a.checkpoint),frontend='ptbr-numbers-v1',kv_capacity=1024,
            cache_cap_mib=512,output_cap_mib=128,reserved_extra_bytes=RESERVE,
            gpu_execution_performed=False,quality='Unmeasured; no automatic quality decision',
            proposed_command=[sys.executable,'-m','training.v2_voxcpm_reference_probe','--action','execute',
                '--run',str(a.run),'--cache',str(a.cache),'--checkpoint',str(a.checkpoint),
                '--checkpoint-sha256',a.checkpoint_sha256,'--gpu-authorized'])
        pilot.atomic_json(a.run/'plan.json',plan);print(json.dumps(plan,indent=2));return
    execute(a)


if __name__=='__main__':main()
