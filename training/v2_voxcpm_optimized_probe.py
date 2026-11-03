"""Pinned VoxCPM2 optimization probe; default is CPU-only metadata planning.

GPU execution is allowed only as a private GPU-worker job after other backends
exit. Uses official optimize(); local LoRA folding is a separately tested shim.
No quality/speed decision is made by this engineering probe.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

from training import v2_voxcpm_probe as pilot

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/var/tmp/aegis-v2-work')


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()


def build_trials(cases,repetitions,optimize):
    if not cases or len(cases)!=len(set(cases)):raise ValueError('Unique, nonempty validation cases required')
    if repetitions<3:raise ValueError('At least3 warmed repetitions required')
    warmup=[]
    for index,case in enumerate(cases):
        first=('compile-first' if optimize else 'cold-first') if index==0 else 'case-first'
        warmup.extend([(first,case,-1),('warmup1',case,-1),('warmup2',case,-1)])
    return warmup+[('warm',case,rep) for rep in range(repetitions) for case in cases]


def fold_lora(model,torch):
    """Remove all low-rank branches; preserve checkpoint/base files unchanged."""
    changed=[]
    for name,module in list(model.named_modules()):
        if module.__class__.__name__!='LoRALinear':continue
        if module.training:raise ValueError('LoRA fold requires eval mode')
        parent=model
        parts=name.split('.')
        for part in parts[:-1]:parent=getattr(parent,part)
        with torch.no_grad():
            merged=module.weight.detach().float()
            if module.r>0 and module.lora_A is not None:
                merged=merged+(module.lora_B.detach().float()@module.lora_A.detach().float())*module.scaling.detach().float()
            layer=torch.nn.Linear(module.in_features,module.out_features,bias=module.bias is not None,
                                  device=module.weight.device,dtype=module.weight.dtype)
            layer.weight.copy_(merged.to(layer.weight.dtype))
            if module.bias is not None:layer.bias.copy_(module.bias)
            layer.requires_grad_(False);layer.eval()
        setattr(parent,parts[-1],layer)
        changed.append(dict(name=name,shape=list(layer.weight.shape),dtype=str(layer.weight.dtype),device=str(layer.weight.device)))
    return changed


def compiled_targets(model):
    values={'base_lm.forward_step':model.base_lm.forward_step,
            'residual_lm.forward_step':model.residual_lm.forward_step,
            'feat_encoder':model.feat_encoder,'feat_decoder.estimator':model.feat_decoder.estimator}
    return {name:bool(hasattr(value,'_torchdynamo_orig_callable') or hasattr(value,'_orig_mod')) for name,value in values.items()}


def diagnostics(torch):
    result={}
    try:
        from torch._dynamo.utils import counters
        result['dynamo_counters']={str(k):{str(a):int(b) if isinstance(b,(int,float)) else str(b) for a,b in v.items()} for k,v in counters.items()}
        from torch._inductor import metrics
        result['generated_kernel_count']=int(metrics.generated_kernel_count)
        from torch._inductor.cudagraph_trees import get_manager
        manager=get_manager(0,create_if_none_exists=False)
        nodes=[];seen=set()
        if manager:
            pending=[n for v in manager.roots.values() for n in v]
            while pending:
                node=pending.pop()
                if id(node) in seen:continue
                seen.add(id(node));nodes.append(node)
                pending.extend(n for values in getattr(node,'children',{}).values() for n in values)
        result['recorded_cudagraph_nodes']=sum(getattr(n,'graph',None) is not None for n in nodes)
        result['replay_scope']='Actual warmup replays are counted by the separate temporary replay hook; nodes alone do not prove replay'
    except Exception as e:result['diagnostics_error']=repr(e)
    return result


def residency(model,torch):
    cpu=[];parameter_bytes=0;buffer_bytes=0
    for name,p in model.named_parameters():
        parameter_bytes+=p.numel()*p.element_size()
        if p.device.type!='cuda':cpu.append('parameter:'+name)
    for name,value in model.named_buffers():
        buffer_bytes+=value.numel()*value.element_size()
        if value.device.type!='cuda':cpu.append('buffer:'+name)
    for name,module in model.named_modules():
        for attr,value in vars(module).items():
            if torch.is_tensor(value) and value.device.type!='cuda':cpu.append('ordinary:'+name+'.'+attr)
    for name in ('base_lm','residual_lm'):
        if getattr(model,name).kv_cache.kv_cache.device.type!='cuda':cpu.append('cache:'+name)
    if cpu:raise ValueError('CPU-resident model tensor/cache: '+repr(cpu[:30]))
    return dict(all_model_parameters_buffers_and_kv_on_cuda=True,parameter_bytes=parameter_bytes,buffer_bytes=buffer_bytes)


def cpu_check(args):
    from types import SimpleNamespace
    from unittest.mock import patch
    os.environ['CUDA_VISIBLE_DEVICES']=''
    torch,cls,*_=pilot.imports(args)
    from voxcpm.modules.layers.lora import LoRALinear
    torch.manual_seed(40733133)
    proofs=[]
    for zero in (True,False):
        base=torch.nn.Linear(7,5)
        layer=LoRALinear(base,r=3,alpha=3,dropout=0).eval()
        with torch.no_grad():
            if not zero:layer.lora_B.normal_(0,.01)
        model=torch.nn.Sequential(layer).eval();x=torch.randn(2,4,7)
        before=model(x).detach();folded=fold_lora(model,torch);after=model(x).detach()
        error=float((before-after).abs().max())
        if error>2e-6:raise AssertionError('LoRA folding numeric mismatch')
        if any(m.__class__.__name__=='LoRALinear' for m in model.modules()):raise AssertionError('LoRA branch remains after fold')
        if any(p.device.type!='cpu' for p in model.parameters()):raise AssertionError('CPU proof touched GPU parameters')
        proofs.append(dict(zero_lora=zero,max_abs_error=error,folded_layers=len(folded),all_parameters_cpu=True))
    bf16_proofs=[]
    for strength in (.01,.00001):
        base=torch.nn.Linear(7,5).to(torch.bfloat16)
        layer=LoRALinear(base,r=3,alpha=3,dropout=0.).eval()
        with torch.no_grad():layer.lora_B.normal_(0,strength)
        if layer.lora_A.dtype!=torch.float32 or layer.lora_B.dtype!=torch.float32:raise AssertionError('Expected FP32 adapter')
        original=layer.weight.detach().clone();delta=layer.lora_B.detach().float()@layer.lora_A.detach().float()
        model=torch.nn.Sequential(layer).eval();x=torch.randn(2,4,7).to(torch.bfloat16)
        with torch.autocast('cpu',dtype=torch.bfloat16):before=model(x).detach()
        fold_lora(model,torch);after=model(x).detach()
        difference=(before.float()-after.float());merged_delta=model[0].weight.detach().float()-original.float()
        bf16_proofs.append(dict(lora_B_std=strength,base_dtype=str(original.dtype),adapter_dtype='torch.float32',
           baseline='nonzero LoRA under BF16 autocast',merged='delta computed FP32 then merged weight rounded BF16',
           max_abs_error=float(difference.abs().max()),relative_l2_error=float(difference.norm()/before.float().norm().clamp_min(1e-20)),
           collapsed_update_elements=int(((delta!=0)&(merged_delta==0)).sum()),nonzero_update_elements=int((delta!=0).sum()),
           retained_delta_norm_ratio=float(merged_delta.norm()/delta.norm().clamp_min(1e-20)),
           all_parameters_cpu=all(p.device.type=='cpu' for p in model.parameters())))
    requests=[]
    def spy(value,**kw):
        requests.append(dict(mode=kw.get('mode'),fullgraph=kw.get('fullgraph')))
        if isinstance(value,torch.nn.Module):
            wrapper=torch.nn.Sequential(value);wrapper._orig_mod=value;return wrapper
        def wrapper(*a,**k):return value(*a,**k)
        wrapper._torchdynamo_orig_callable=value;return wrapper
    fake=SimpleNamespace(device='cuda',base_lm=SimpleNamespace(forward_step=lambda x:x),
                         residual_lm=SimpleNamespace(forward_step=lambda x:x),feat_encoder=torch.nn.Identity(),
                         feat_decoder=SimpleNamespace(estimator=torch.nn.Identity()))
    with patch.object(torch,'compile',side_effect=spy):cls.optimize(fake)
    if len(requests)!=4 or not all(compiled_targets(fake).values()):raise AssertionError('official optimize target coverage mismatch')
    if torch.cuda.is_initialized():raise AssertionError('CPU preflight initialized CUDA')
    record=dict(status='pass',utc=utc(),scope='tiny real CPU LoRA folding plus mocked official optimize dispatch; no full-model load or CUDA compile',
                torch=torch.__version__,cuda_build=torch.version.cuda,cuda_initialized=False,
                tiny_lora_proofs=proofs,bf16_nonzero_lora_proofs=bf16_proofs,official_optimize_requests=requests,
                limitation='GPU compilation, CUDAGraph recording, warmed speed and audio are not measured in this action',
                code_sha256=pilot.sha(__file__))
    pilot.atomic_json(args.run/'cpu-check.json',record);print(json.dumps(record),flush=True)


def working_usage():
    from training.v2_language_monitor import physical_usage
    paths=sorted(Path('/var/tmp').glob('aegis*'))+[Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip'),REPO,Path('/home/phc/.cache')]
    return physical_usage(paths)


def configure_cache(args):
    cache=args.cache.resolve();cache.mkdir(parents=True,exist_ok=True,mode=0o700)
    for name in ('tmp','extensions','cuda'):(cache/name).mkdir(exist_ok=True,mode=0o700)
    os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),
                      TORCHINDUCTOR_COMPILE_THREADS='1',MAX_JOBS='1',TMPDIR=str(cache/'tmp'),
                      TORCH_EXTENSIONS_DIR=str(cache/'extensions'),CUDA_CACHE_PATH=str(cache/'cuda'))
    import tempfile
    tempfile.tempdir=str(cache/'tmp')
    receipt=ROOT/'voxcpm2/compile-prereqs.json'
    if receipt.exists():
        info=json.loads(receipt.read_text(encoding='utf-8'))
        if info.get('headers',{}).get('manifest_sha256')!=pilot.sha(Path(info['headers']['manifest'])):
            raise ValueError('Compiler-header manifest changed')
        os.environ.update(info.get('env_for_child_only',{}))
    return cache


def budget_watch(args,stop,initial):
    from training.v2_language_monitor import physical_usage
    initial_cache=physical_usage([args.cache])['allocated_bytes'];last_full=time.monotonic()
    while not stop.wait(5):
        cache=physical_usage([args.cache])['allocated_bytes'];free=shutil.disk_usage(args.run).free
        working=initial['allocated_bytes']+max(0,cache-initial_cache)
        if time.monotonic()-last_full>60:
            working=working_usage()['allocated_bytes'];last_full=time.monotonic()
        reason=None
        if cache>args.cache_limit_mib*1024**2:reason='private compile cache cap exceeded'
        elif free<5*1024**3:reason='filesystem reserve below5GiB'
        elif working>40_000_000_000:reason='working budget exceeds40GB'
        if reason:
            pilot.atomic_json(args.run/'safety-stop.json',dict(reason=reason,cache_bytes=cache,free_bytes=free,working_bytes=working,utc=utc()))
            # Stop this probe PID only; never its process group/GPU queue/trainer.
            os.kill(os.getpid(),signal.SIGTERM);return


def require_idle_cpu():
    active=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            tokens=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if not any(x.endswith('asr_score.py') for x in tokens):continue
            state=(proc/'stat').read_text(encoding='utf-8').split(') ',1)[1].split()[0]
            if state not in ('T','Z'):active.append(int(proc.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if active:raise RuntimeError('Pause/wait for CPU ASR before speed probe; active PIDs='+repr(active))


def execute(args):
    parent=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().decode(errors='replace')
    if 'v2_gpu_worker' not in parent:raise RuntimeError('GPU execution must be a private GPU-worker job')
    if (args.run/'recipe.json').exists() or (args.run/'results.jsonl').exists():
        raise ValueError('Use a fresh run directory; existing inference evidence must not be overwritten or mixed acrossNFE/precision')
    if not (ROOT/'pause-asr').exists():raise RuntimeError('Create pause-asr and confirm CPU ASR suspended before this benchmark')
    require_idle_cpu();pilot.verify(args);pilot.guard_gpu(args)
    cache=configure_cache(args);inventory=working_usage()
    from training.v2_language_monitor import physical_usage
    initial_cache=physical_usage([cache])
    estimated_extra=max(1024**3,(args.cache_limit_mib+256)*1024**2)
    if inventory['allocated_bytes']+estimated_extra>40_000_000_000:raise RuntimeError('Insufficient working budget for declared compile/output peak')
    if shutil.disk_usage(args.run).free<5*1024**3+estimated_extra:raise RuntimeError('Need5GiB reserve plus declared compile margin')
    pilot.atomic_json(args.run/'preflight.json',dict(utc=utc(),working=inventory,estimated_extra_bytes=estimated_extra,
                        cache_limit_mib=args.cache_limit_mib,headers_receipt=str(ROOT/'voxcpm2/compile-prereqs.json'),
                        compile_cache_before=initial_cache,
                        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),asr_idle=True))
    stop=threading.Event();watch=threading.Thread(target=budget_watch,args=(args,stop,inventory),daemon=True);watch.start()
    torch,cls,lora_cls,*_=pilot.imports(args)
    import numpy as np
    torch.set_float32_matmul_precision(args.matmul_precision)
    if args.checkpoint and args.topology=='base':raise ValueError('Base topology must not silently drop an adapter checkpoint')
    begin=time.perf_counter();previous=torch.get_default_dtype()
    lora=None if args.topology=='base' else lora_cls(enable_lm=True,enable_dit=True,enable_proj=False,r=args.rank,alpha=args.rank,dropout=0.)
    try:
        torch.set_default_dtype(torch.bfloat16)
        model=cls.from_local(str(args.model),optimize=False,training=True,device='cpu',lora_config=lora)
    finally:torch.set_default_dtype(previous)
    if lora is not None:
        for name,p in model.named_parameters():
            if p.requires_grad:p.data=p.data.float()
    adapter_sha=None
    if args.checkpoint:
        expected=json.loads(args.checkpoint.with_suffix('.sha256.json').read_text(encoding='utf-8'))['sha256']
        adapter_sha=pilot.sha(args.checkpoint)
        if expected!=adapter_sha:raise ValueError('Adapter checkpoint hash mismatch')
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if state['config']['rank']!=args.rank:raise ValueError('Adapter rank mismatch')
        pilot.copy_lora_strict(model,state['lora'],torch)
    model.eval();folded=fold_lora(model,torch) if args.topology=='merged' else []
    for p in model.parameters():p.requires_grad_(False)
    # Native generation has no outer autocast: nonmerged branches need BF16.
    if args.topology=='lora':
        for name,p in model.named_parameters():
            if 'lora_' in name:p.data=p.data.to(torch.bfloat16)
    model=pilot.model_to_gpu(model,torch).eval();torch.cuda.synchronize()
    loaded_ms=(time.perf_counter()-begin)*1000
    resident=residency(model,torch)
    optimize_warnings=''
    if args.optimize:
        warning=io.StringIO()
        with contextlib.redirect_stderr(warning):model.optimize()
        optimize_warnings=warning.getvalue()
        targets=compiled_targets(model)
        if not all(targets.values()):raise RuntimeError('Official optimize silently/partially disabled: '+repr(targets)+' '+optimize_warnings)
    else:targets=compiled_targets(model)
    manifest=dict(utc=utc(),source_rev=pilot.SOURCE_REV,model_rev=pilot.MODEL_REV,code_sha256=pilot.sha(__file__),
                  synthesis_helper_sha256=pilot.sha(pilot.__file__),
                  validation_corpus_sha256=pilot.sha(REPO/'benchmarks/v2/validation.pt-BR.json'),
                  topology=args.topology,optimize_requested=args.optimize,compiled_wrappers=targets,optimize_warnings=optimize_warnings,
                  adapter_sha256=adapter_sha,lora_layer_count=sum(m.__class__.__name__=='LoRALinear' for m in model.modules()),
                  folded_layers=folded,model_load_ms=loaded_ms,residency=resident,
                  inference=dict(nfe=args.nfe,inference_timesteps=args.nfe,
                                 nfe_label='upstream timestep argument; default CFGZeroStar skips first estimator call',
                                 source_default_use_cfg_zero_star=True,
                                 expected_dit_evaluations_per_patch=args.nfe-max(1,int((args.nfe+1)*.04)),
                                 cfg=2.,seed=40733133,max_len=400,retry_badcase=False,retry_badcase_ratio_threshold=6.,normalization=False,denoising=False),
                  arithmetic=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                                  requested_float32_matmul_precision=args.matmul_precision,
                                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                                  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')),
                  delivery='whole file; final24k PCM after resample_poly Kaiser10; no HTTP or hypothetical streaming TTFA')
    pilot.atomic_json(args.run/'recipe.json',manifest)
    corpus={r['id']:r for r in json.loads((REPO/'benchmarks/v2/validation.pt-BR.json').read_text(encoding='utf-8'))}
    if set(args.cases)-set(corpus):raise ValueError('Only fixed validation cases allowed')
    records=[]
    ar_counts={'base_lm':0,'residual_lm':0}
    for name in ar_counts:
        module=getattr(model,name);original=module.forward_step
        def counted_step(*a,_name=name,_original=original,**kw):
            ar_counts[_name]+=1;return _original(*a,**kw)
        module.forward_step=counted_step
    estimator_calls=[0];original_estimator=model.feat_decoder.estimator.forward
    def counted_estimator(*a,**kw):
        estimator_calls[0]+=1;return original_estimator(*a,**kw)
    model.feat_decoder.estimator.forward=counted_estimator
    replay_count=[0];original_replay=torch.cuda.CUDAGraph.replay
    def counted_replay(graph,*a,**kw):
        replay_count[0]+=1;return original_replay(graph,*a,**kw)
    if args.optimize:torch.cuda.CUDAGraph.replay=counted_replay
    try:
        # First call includes lazy compile. Next2 calls allow graph warmup and
        # recording; none are relabeled as warmed benchmark observations.
        trials=build_trials(args.cases,args.repetitions,args.optimize)
        for condition,case,rep in trials:
            # The replay hook proves actual execution during cold/warmup only;
            # restore the class before all timed warm repetitions.
            if condition=='warm':torch.cuda.CUDAGraph.replay=original_replay
            require_idle_cpu();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            torch.compiler.cudagraph_mark_step_begin()
            args.text=corpus[case]['text'];label=f'{condition}-{case}-r{rep:02d}'
            before_counts=ar_counts.copy()
            before_estimator=estimator_calls[0]
            row=pilot.synthesize(args,model,torch,np,label)
            row.update(condition=condition,case=case,repetition=rep,topology=args.topology,optimize_requested=args.optimize,
                       matmul_precision=args.matmul_precision,
                       ar_forward_steps={k:ar_counts[k]-before_counts[k] for k in ar_counts},
                       dit_estimator_calls=estimator_calls[0]-before_estimator,
                       cudagraph_replays_observed_warmup=replay_count[0],
                       diagnostics=diagnostics(torch) if args.optimize else {})
            steps=row['ar_forward_steps']['base_lm']
            row['text_token_count']=len(model.text_tokenizer(args.text))
            row['effective_ar_length_cap']=min(row['text_token_count']*6+10,400)
            row['potential_length_cap_hit']=steps>=row['effective_ar_length_cap']
            row['eos_semantics']='Length-cap guard only; normal stop prediction does not prove complete pronunciation/content'
            row['amortized_native_ms_per_base_lm_step']=row['native_ms']/steps if steps else None
            row['step_cost_scope']='whole native call divided by counted AR forward_step calls; includes prefill/DiT/VAE, not isolated kernel timing'
            residency(model,torch);records.append(row)
            with (args.run/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
            pilot.atomic_json(args.run/'progress.json',dict(utc=utc(),condition=condition,case=case,repetition=rep,eof_ms=row.get('eof_ms'),rtf=row.get('rtf')))
        if args.optimize:
            diag=diagnostics(torch)
            if not diag.get('generated_kernel_count',0):raise RuntimeError('No generated kernels observed; do not label compiled execution verified')
        else:diag={}
        import statistics
        summary={}
        for case in args.cases:
            observed=[r for r in records if r['condition']=='warm' and r['case']==case]
            selected=[r for r in observed if r.get('outcome')=='ok' and r.get('first_speech_pcm_available_ms') is not None]
            cap_hits=sum(r['potential_length_cap_hit'] for r in selected)
            rtf=statistics.median(r['rtf'] for r in selected) if selected else None
            summary[case]=dict(n=len(selected),observations=len(observed),failed_or_no_speech=len(observed)-len(selected),
                              eof_p50_ms=statistics.median(r['eof_ms'] for r in selected) if selected else None,
                              rtf_p50=rtf,potential_length_cap_hits=cap_hits,
                              speed_gate_passed=bool(len(selected)>=3 and len(selected)==len(observed) and rtf<1 and not cap_hits),
                              speed_gate_scope='Performance triage only; omitting/repeating/shortened speech still requires common ASR/audio checks',
                              max_vram_allocated_mib=max((r['peak_allocated_mib'] for r in selected),default=None),p95=None)
        pilot.atomic_json(args.run/'summary.json',dict(cases=summary,diagnostics=diag,nfe=args.nfe,matmul_precision=args.matmul_precision,
                         compile_and_warmup_separate=True,warmup_policy='3 excluded calls for EVERY case before the warm matrix',
                         cudagraph_replays_observed_warmup=replay_count[0],cudagraph_execution_verified=replay_count[0]>0,
                         replay_proof='process-local CUDAGraph.replay wrapper during per-case warmups; restored before all warm benchmark observations',
                         quality='NFE/precision ablations are new inference configurations; no quality equivalence, hearing or identity claims; common audio/ASR required'))
    finally:
        torch.cuda.CUDAGraph.replay=original_replay
        stop.set();watch.join(2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['plan','cpu-check','execute'],default='plan')
    p.add_argument('--source',type=Path,default=pilot.ROOT/'source');p.add_argument('--model',type=Path,default=pilot.ROOT/'model')
    p.add_argument('--run',type=Path,default=pilot.ROOT/'optimized-preparation');p.add_argument('--cache',type=Path,default=pilot.ROOT/'compile-cache')
    p.add_argument('--topology',choices=['base','lora','merged'],default='base');p.add_argument('--optimize',action='store_true')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--rank',type=int,default=8);p.add_argument('--gpu-authorized',action='store_true')
    p.add_argument('--cache-limit-mib',type=int,default=512);p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--nfe',type=int,choices=[10,6,4],default=10)
    p.add_argument('--matmul-precision',choices=['highest','high'],default='highest',help='Separate TF32 ablation; highest is measured control')
    p.add_argument('--cases',nargs='+',default=['v_reply']);a=p.parse_args();os.umask(0o077)
    if a.repetitions<3:p.error('At least3 warm repetitions per case are required')
    if a.rank<1 or a.cache_limit_mib<1:p.error('Positive rank and compile-cache cap required')
    build_trials(a.cases,a.repetitions,a.optimize)
    a.run.mkdir(parents=True,exist_ok=True,mode=0o700)
    if a.action=='plan':
        value=dict(source_rev=pilot.SOURCE_REV,code_sha256=pilot.sha(__file__),actions=['CPU imports/tiny merge proof only now','Later exclusive fresh-process base-eager','base official optimize','adapter eager vs merged eager/compiled'],
                   pending_gpu=True,official_compile=dict(targets=4,mode='reduce-overhead',fullgraph=True,device_must_equal='cuda'),
                   topology=a.topology,optimize=a.optimize,checkpoint=str(a.checkpoint) if a.checkpoint else None,
                   compiler_cache_cap_mib=a.cache_limit_mib,estimated_peak_extra_bytes=max(1024**3,(a.cache_limit_mib+256)*1024**2),
                   inference=dict(nfe=a.nfe,cfg=2.,seed=40733133,matmul_precision=a.matmul_precision),
                   validation_cases=a.cases,warmup='Each case gets3 excluded calls before any warm observations',
                   quality='NFE6/4 and TF32 high are new configurations; speed is not quality evidence',
                   next_gate='v_reply n>=3 and RTFp50<1 without length-cap hit; only then seven-case matrix + native streaming + ASR/content/identity checks',
                   cpu_model_residency='Full CPU load proof is existing cpu-check/cpu-check.json; this action does not load weights',
                   execution_guard='private GPU-worker parent; empty GPU; all CPU ASR suspended; pause-asr present; budget<40GB incl cache')
        pilot.atomic_json(a.run/'plan.json',value);print(json.dumps(value,indent=2));return
    if a.action=='cpu-check':cpu_check(a);return
    try:execute(a)
    except BaseException as e:
        pilot.atomic_json(a.run/'failure.json',dict(utc=utc(),type=type(e).__name__,error=repr(e),traceback=traceback.format_exc(),code_sha256=pilot.sha(__file__)))
        raise


if __name__=='__main__':main()
