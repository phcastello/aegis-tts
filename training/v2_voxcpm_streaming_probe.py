"""Native pinned VoxCPM2 streaming probe; default action only prints a plan.

GPU execution is exclusively an owner GPU-worker job. CPU tests exercise the
stateful 48->24k FIR/client adapter without importing torch or model weights.
Shared optimized/file probes and common measurement helpers are read-only.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import wave

from training import v2_voxcpm_optimized_probe as optimized
from training import v2_voxcpm_probe as pilot

REPO=Path(__file__).resolve().parents[1]
RATE=24000
SEED=40733133


class HalfRateFIR:
    """Bounded-history causal FIR, aligned to scipy resample_poly Kaiser10.

    A 41-tap filter delays20 native samples (10 outputs). Hold these outputs
    until future input is known; flush20 zeros at a normal EOF and trim exactly
    to ceil(input_samples/2). This compensates delay without introducing audio
    padding. Native chunk timestamps include the lookahead wait. upfirdn on
    phase-aligned history uses the same summation/order as the batch reference.
    """
    native_rate=48000
    final_rate=24000
    lookahead_native_samples=20

    def __init__(self):
        import numpy as np
        from scipy.signal import firwin
        self.h=firwin(41,.5,window=('kaiser',10.)).astype(np.float32)
        self.history=np.empty(0,dtype=np.float32);self.input_samples=0
        self.causal_outputs=0;self.output_samples=0;self.closed=False

    def _feed(self,values):
        import numpy as np
        from scipy.signal import upfirdn
        values=np.asarray(values,dtype=np.float32)
        if values.ndim!=1 or not np.isfinite(values).all():raise ValueError('Finite mono float samples required')
        if not values.size:return np.empty(0,dtype=np.float32)
        old=self.input_samples;start=old-len(self.history)
        if start%2:raise AssertionError('FIR history lost decimation phase')
        joined=np.concatenate((self.history,values))
        end=old+len(values);causal_count=(end+1)//2
        causal=upfirdn(self.h,joined,up=1,down=2)
        first=max(self.causal_outputs,10)
        result=causal[first-start//2:causal_count-start//2].copy()
        self.input_samples=end;self.causal_outputs=causal_count
        # Preserve all40 FIR predecessors plus one sample when phase requires.
        next_start=max(0,end-40);next_start-=next_start%2
        self.history=joined[next_start-start:].copy()
        self.output_samples+=len(result)
        return result

    def push(self,values):
        if self.closed:raise RuntimeError('Resampler already closed')
        return self._feed(values)

    def flush(self):
        import numpy as np
        if self.closed:raise RuntimeError('Resampler already closed')
        native=self.input_samples;expected=(native+1)//2
        result=self._feed(np.zeros(20,dtype=np.float32)) if native else np.empty(0,dtype=np.float32)
        self.closed=True;self.history=np.empty(0,dtype=np.float32)
        if self.output_samples!=expected:raise AssertionError('FIR final length differs from batch reference')
        return result

    def cancel(self):
        import numpy as np
        self.closed=True;self.history=np.empty(0,dtype=np.float32)

    def recipe(self):
        return dict(native_rate=48000,final_rate=24000,taps=41,window=['kaiser',10.],
                    filter_sha256=hashlib.sha256(self.h.tobytes()).hexdigest(),
                    phase='global input sample0',lookahead_native_samples=20,
                    lookahead_ms=20/48,normal_eof='zero extension20samples; compensate10output delay; ceil(N/2)',
                    cancel='discard pending FIR state; no synthetic tail or normal flush',
                    reference="scipy.signal.resample_poly(float32,1,2,window=('kaiser',10.))")


def metrics_module():
    spec=importlib.util.spec_from_file_location('aegis_stream_metrics',REPO/'benchmarks/v2/benchmark.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def native_numpy(chunk):
    """Native model yields [1,T] CPU tensors; mocks can yield mono NumPy."""
    import numpy as np
    if hasattr(chunk,'detach'):chunk=chunk.detach().float().cpu().numpy()
    values=np.asarray(chunk,dtype=np.float32)
    if values.ndim==2 and values.shape[0]==1:values=values[0]
    if values.ndim!=1:raise ValueError('Native stream must be mono, batch1')
    return values


def collect_stream(factory,*,clock=time.perf_counter,sink=None,cancelled=None,
                   cancel_after_native_chunks=0,convert=native_numpy):
    """Local PCM client timing, including lazy generation/FIR/quantization/flush.

    The sink receives headerless PCM; its completion is the local availability
    timestamp. No HTTP/device delay is implied. Generator.close() runs on EOF,
    cancel and errors so native VAE monkeypatches cannot leak across requests.
    """
    import numpy as np
    begin=clock();resampler=HalfRateFIR();stream=None;chunks=[];native=[];payload=[]
    native_hash=hashlib.sha256();native_count=0;native_samples=0;native_peak=0.
    eof_native_ms=None;outcome='ok';error=None;closed=False;flushed=False
    def stop_requested():return bool(cancelled and cancelled())
    def emit(values,origin):
        if not values.size:return
        if not np.isfinite(values).all() or float(np.abs(values).max())>1.:
            raise ValueError('PCM out of range/nonfinite; no clipping or gain is permitted')
        data=np.rint(values*32767).astype('<i2').tobytes()
        if sink:sink(data)
        payload.append(data)
        chunks.append(dict(ms=(clock()-begin)*1000,bytes=len(data),source=origin,
                           output_samples=len(data)//2))
    try:
        stream=iter(factory())
        while True:
            if stop_requested():outcome='cancelled';break
            try:chunk=next(stream)
            except StopIteration:eof_native_ms=(clock()-begin)*1000;break
            ready=(clock()-begin)*1000;values=convert(chunk)
            if not np.isfinite(values).all():raise ValueError('Native chunk is nonfinite')
            native_count+=1;native_samples+=len(values);native_hash.update(values.tobytes())
            if values.size:native_peak=max(native_peak,float(np.abs(values).max()))
            native.append(dict(index=native_count-1,ready_ms=ready,samples=len(values),
                               cumulative_samples=native_samples,empty=not bool(values.size)))
            if stop_requested():outcome='cancelled';break
            emit(resampler.push(values),'native:'+str(native_count-1))
            if cancel_after_native_chunks and native_count>=cancel_after_native_chunks:
                outcome='cancelled';break
        if outcome=='ok':emit(resampler.flush(),'normal_eof_flush');flushed=True
        else:resampler.cancel()
    except Exception as exc:
        outcome='failed';error=repr(exc);resampler.cancel()
    finally:
        if stream is not None:
            try:
                close=getattr(stream,'close',None)
                if close:close();closed=True
            except Exception as exc:outcome='failed';error='generator close: '+repr(exc)
    pcm=b''.join(payload);eof_ms=(clock()-begin)*1000
    if outcome=='ok' and not pcm:outcome='failed';error='Native EOF without playable PCM'
    metrics=metrics_module().pcm_metrics(pcm,chunks,rate=RATE)
    row=dict(outcome=outcome,error=error,native_chunks=native,pcm_chunks=chunks,
             native_samples=native_samples,native_seconds=native_samples/48000,
             native_float32_sha256=native_hash.hexdigest(),native_peak=native_peak,
             native_eof_ms=eof_native_ms,last_pcm_available_ms=chunks[-1]['ms'] if chunks else None,
             eof_ms=eof_ms,rtf=eof_ms/1000/metrics['audio_seconds'] if metrics['audio_seconds'] else None,
             normal_flush_completed=flushed,generator_close_called=closed,
             local_client='direct native iterator -> CPU FIR -> PCM bytes sink; no HTTP/device timing',
             partial_pcm_delivered=bool(pcm) and outcome!='ok',resampling=resampler.recipe(),**metrics)
    return pcm,row


def trials(cases,repetitions,optimize):
    """Three excluded calls for every case, then fixed warm repetitions."""
    return optimized.build_trials(cases,repetitions,optimize)


def generation_kwargs(args,text):
    return dict(target_text=text,max_len=400,inference_timesteps=args.nfe,cfg_value=2.,
                retry_badcase=False,retry_badcase_ratio_threshold=6.,
                trim_silence_vad=False,streaming_prefix_len=4,seed=SEED)


def load_model(args,torch,cls,lora_cls):
    if args.checkpoint and args.topology=='base':raise ValueError('Base topology cannot discard an adapter')
    begin=time.perf_counter();previous=torch.get_default_dtype()
    lora=None if args.topology=='base' else lora_cls(enable_lm=True,enable_dit=True,enable_proj=False,r=args.rank,alpha=args.rank,dropout=0.)
    try:
        torch.set_default_dtype(torch.bfloat16)
        model=cls.from_local(str(args.model),optimize=False,training=True,device='cpu',lora_config=lora)
    finally:torch.set_default_dtype(previous)
    if lora is not None:
        for p in model.parameters():
            if p.requires_grad:p.data=p.data.float()
    adapter_sha=None
    if args.checkpoint:
        expected=json.loads(args.checkpoint.with_suffix('.sha256.json').read_text())['sha256']
        adapter_sha=pilot.sha(args.checkpoint)
        if adapter_sha!=expected:raise ValueError('Adapter checkpoint hash mismatch')
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if state['config']['rank']!=args.rank:raise ValueError('Adapter rank mismatch')
        pilot.copy_lora_strict(model,state['lora'],torch)
    model.eval();folded=optimized.fold_lora(model,torch) if args.topology=='merged' else []
    for p in model.parameters():p.requires_grad_(False)
    if args.topology=='lora':
        for name,p in model.named_parameters():
            if 'lora_' in name:p.data=p.data.to(torch.bfloat16)
    model=pilot.model_to_gpu(model,torch).eval();torch.cuda.synchronize()
    resident=optimized.residency(model,torch);warnings=''
    if args.optimize:
        capture=io.StringIO()
        with contextlib.redirect_stderr(capture):model.optimize()
        warnings=capture.getvalue()
    targets=optimized.compiled_targets(model)
    if args.optimize and not all(targets.values()):raise RuntimeError('Official optimization disabled/partial: '+repr(targets)+warnings)
    return model,dict(model_load_ms=(time.perf_counter()-begin)*1000,adapter_sha256=adapter_sha,
                      folded_layers=folded,residency=resident,compiled_wrappers=targets,optimize_warnings=warnings)


def summarize(records,corpus):
    import statistics
    summary={}
    for case in dict.fromkeys(row['case'] for row in records):
        all_rows=[r for r in records if r['case']==case and r['condition']=='warm']
        good=[r for r in all_rows if r['outcome']=='ok' and r.get('first_speech_pcm_available_ms') is not None]
        value=dict(n=len(good),observations=len(all_rows),size=corpus[case]['size'],p95=None,
                   failed_or_no_speech=len(all_rows)-len(good),
                   potential_length_cap_hits=sum(r['potential_length_cap_hit'] for r in good))
        for key in ['first_pcm_ms','first_speech_pcm_available_ms','ttfa_estimated_playback_ms',
                    'eof_ms','rtf','audio_seconds','underrun_total_ms']:
            value[key+'_p50']=statistics.median(r[key] for r in good) if good else None
        value['gap_events']=sum(r['underruns_over_10ms'] for r in good)
        throughput=bool(len(good)>=3 and len(good)==len(all_rows)
             and not value['potential_length_cap_hits'] and value['rtf_p50']<1)
        value['rtf_below_one_observed']=throughput
        value['mandatory_long_rtf_gate_passed']=throughput if value['size']=='long' else None
        value['rejected_by_short_rtf_alone']=False
        value['short_startup_caveat']='Short RTF>1 alone does not reject native streaming; inspect TTFA/gaps and mandatory long RTF<1.'
        summary[case]=value
    return summary


def execute(args):
    """Prepared future runner; not called by CPU tests or default plan."""
    parent=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().decode(errors='replace')
    if 'v2_gpu_worker' not in parent:raise RuntimeError('Run exclusively through the private GPU worker')
    if (args.run/'recipe.json').exists() or (args.run/'results.jsonl').exists():raise ValueError('Fresh run required for each NFE/topology/precision')
    root=pilot.ROOT.parent
    if not (root/'pause-asr').exists():raise RuntimeError('Pause CPU ASR explicitly before a performance measurement')
    optimized.require_idle_cpu()
    corpus={r['id']:r for r in json.loads((REPO/'benchmarks/v2/validation.pt-BR.json').read_text())}
    if set(args.cases)-set(corpus):raise ValueError('Only frozen validation cases allowed')
    pilot.verify(args);pilot.guard_gpu(args);optimized.configure_cache(args)
    inventory=optimized.working_usage();extra=max(1024**3,(args.cache_limit_mib+256)*1024**2)
    if inventory['allocated_bytes']+extra>40_000_000_000:raise RuntimeError('Working budget cannot cover declared compile/output margin')
    if shutil.disk_usage(args.run).free<5*1024**3+extra:raise RuntimeError('Need5GiB filesystem reserve plus declared peak')
    pilot.atomic_json(args.run/'preflight.json',dict(utc=optimized.utc(),working=inventory,
                      estimated_extra_bytes=extra,cache_limit_mib=args.cache_limit_mib,asr_idle=True))
    torch,cls,lora_cls,*_=pilot.imports(args)
    torch.set_float32_matmul_precision(args.matmul_precision)
    stop=threading.Event();watch=threading.Thread(target=optimized.budget_watch,args=(args,stop,inventory),daemon=True);watch.start()
    original_replay=torch.cuda.CUDAGraph.replay;replays=[0];records=[]
    try:
        model,loaded=load_model(args,torch,cls,lora_cls)
        pilot.atomic_json(args.run/'recipe.json',dict(utc=optimized.utc(),code_sha256=pilot.sha(__file__),
             shared_pilot_sha256=pilot.sha(pilot.__file__),shared_optimized_helpers_sha256=pilot.sha(optimized.__file__),
             common_metrics_sha256=pilot.sha(REPO/'benchmarks/v2/benchmark.py'),
             validation_corpus_sha256=pilot.sha(REPO/'benchmarks/v2/validation.pt-BR.json'),
             source_rev=pilot.SOURCE_REV,model_rev=pilot.MODEL_REV,topology=args.topology,nfe=args.nfe,
             generation=generation_kwargs(args,'<fixed validation text>'),optimize_requested=args.optimize,
             nfe_label='upstream timestep argument; CFGZeroStar skips initial estimator call',
             matmul_precision=torch.get_float32_matmul_precision(),
             matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
             TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'),
             warmup='Three excluded calls for EVERY case before the warm matrix; compile-first separate',
             resampling=HalfRateFIR().recipe(),delivery='Native streaming to local PCM client; not HTTP/device timing',
             artifact_io='WAV/JSON trace written after timed client EOF; no artifact IO included in stream delivery',**loaded))
        def counted_replay(graph,*a,**kw):replays[0]+=1;return original_replay(graph,*a,**kw)
        if args.optimize:torch.cuda.CUDAGraph.replay=counted_replay
        for condition,case,rep in trials(args.cases,args.repetitions,args.optimize):
            if condition=='warm':torch.cuda.CUDAGraph.replay=original_replay
            optimized.require_idle_cpu();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            torch.compiler.cudagraph_mark_step_begin()
            kwargs=generation_kwargs(args,corpus[case]['text']);label=f'{condition}-{case}-r{rep:02d}'
            pcm,row=collect_stream(lambda:model.generate_streaming(**kwargs),cancel_after_native_chunks=args.cancel_after_native_chunks)
            torch.cuda.synchronize()
            row.update(condition=condition,case=case,repetition=rep,text=corpus[case]['text'],nfe=args.nfe,
                       topology=args.topology,optimize_requested=args.optimize,matmul_precision=args.matmul_precision,
                       peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                       cudagraph_replays_observed_warmup=replays[0],generation=kwargs,
                       text_token_count=len(model.text_tokenizer(kwargs['target_text'])))
            row['effective_ar_length_cap']=min(int(row['text_token_count']*6+10),400)
            row['potential_length_cap_hit']=len(row['native_chunks'])>=row['effective_ar_length_cap']
            row['cap_note']='Guard only; EOS does not prove correct/complete Portuguese'
            if pcm:
                path=args.run/(label+'-24000.wav')
                with wave.open(str(path),'wb') as wav:
                    wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(pcm)
                row.update(wav=str(path),sha256=pilot.sha(path))
            optimized.residency(model,torch)
            pilot.atomic_json(args.run/(label+'.json'),row);records.append(row)
            with (args.run/'results.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n');stream.flush()
            pilot.atomic_json(args.run/'progress.json',dict(utc=optimized.utc(),condition=condition,case=case,repetition=rep,
                              outcome=row['outcome'],eof_ms=row['eof_ms'],rtf=row['rtf']))
        diag=optimized.diagnostics(torch) if args.optimize else {}
        replay_proven=bool(replays[0]>0)
        pilot.atomic_json(args.run/'summary.json',dict(cases=summarize(records,corpus),nfe=args.nfe,
                   matmul_precision=args.matmul_precision,diagnostics=diag,
                   cudagraph_replays_observed_warmup=replays[0],cudagraph_execution_verified=replay_proven,
                   quality='No hearing/identity/ASR claims; each NFE is a distinct configuration needing common audio checks',
                   required_next_gate='Mandatory long RTF<1 plus TTFA/gaps/content/identity; short startup RTF alone cannot reject streaming'))
        if args.optimize and not replay_proven:raise RuntimeError('Official wrappers present but actual CUDAgraph replay is unproven')
    finally:
        torch.cuda.CUDAGraph.replay=original_replay;stop.set();watch.join(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['plan','execute'],default='plan')
    parser.add_argument('--source',type=Path,default=pilot.ROOT/'source');parser.add_argument('--model',type=Path,default=pilot.ROOT/'model')
    parser.add_argument('--run',type=Path,default=pilot.ROOT/'streaming-preparation');parser.add_argument('--cache',type=Path,default=pilot.ROOT/'compile-cache')
    parser.add_argument('--nfe',type=int,choices=[4,6,10],default=10);parser.add_argument('--matmul-precision',choices=['highest','high'],default='highest')
    parser.add_argument('--optimize',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--topology',choices=['base','merged','lora'],default='base');parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--rank',type=int,default=8);parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--cache-limit-mib',type=int,default=512);parser.add_argument('--gpu-authorized',action='store_true')
    parser.add_argument('--cases',nargs='+',default=['v_reply']);parser.add_argument('--cancel-after-native-chunks',type=int,default=0)
    args=parser.parse_args();os.umask(0o077);trials(args.cases,args.repetitions,args.optimize)
    if args.cancel_after_native_chunks<0 or args.rank<1 or args.cache_limit_mib<1:
        parser.error('Positive rank/cache cap and nonnegative cancel chunk count required')
    if args.action=='plan':
        print(json.dumps(dict(status='prepared_only',source_rev=pilot.SOURCE_REV,model_rev=pilot.MODEL_REV,
              native_api='VoxCPM2Model.generate_streaming',native_patch_samples=7680,native_patch_ms=160,
              overlap='none: upstream StreamingVAEDecoder carries causal conv state and yields newest patch only',
              generation=generation_kwargs(args,'<fixed validation text>'),nfe=args.nfe,optimize=args.optimize,
              compiler_cache_cap_mib=args.cache_limit_mib,
              resampler=HalfRateFIR().recipe(),trials=trials(args.cases,args.repetitions,args.optimize),
              measurement_scope='native local PCM client only; no HTTP/device claim',gpu_execution_performed=False,
              quality='unmeasured; ASR/common audio needed for each NFE/topology; long RTF<1 required',
              pending='exclusive queued GPU validation, cold/compile and per-case warmup separate'),indent=2));return
    execute(args)


if __name__=='__main__':main()
