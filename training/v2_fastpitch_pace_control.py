"""CPU-only FastPitch pace1.0/.8 control. Plan is default; never launches ASR."""
from __future__ import annotations
import argparse,ast,datetime,hashlib,json,math,os,random,resource,shutil,signal,sys,threading,time,unicodedata,wave
from pathlib import Path

from training import v2_fastpitch_benchmark as common

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/var/tmp/aegis-v2-work')
PACES=(1.,.8)
SEED=40733133
DISK_CAP=50_000_000
RSS_CAP=5*1024**3


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def usage(path):return sum(p.stat().st_blocks*512 for p in path.rglob('*') if p.is_file())


def expected_frontend(cases):
    # Read the exact pure normalization function without importing Torch.
    from training.f5_refine_text import normalize_text
    tree=ast.parse((REPO/'training/fastpitch_experiment.py').read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='normalize')
    scope={'unicodedata':unicodedata};exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned FastPitch normalize>','exec'),scope)
    return {r['id']:scope['normalize'](normalize_text(r['text'])['tts_text']) for r in cases}


def bindings(args):
    paths=[Path(__file__),REPO/'training/v2_fastpitch_benchmark.py',REPO/'training/fastpitch_experiment.py',
           REPO/'training/f5_refine_text.py',REPO/'gateway/aegis_speech/normalization.py',REPO/'training/v2_language_monitor.py',
           REPO/'benchmarks/v2/validation.pt-BR.json',REPO/'benchmarks/v2/spoken-validation.pt-BR.json',
           args.source/'UPSTREAM.json',args.source/'fastpitch/model.py',args.bigvgan/'config.json',args.bigvgan/'revision.json']
    return {str(p):common.sha(p) for p in paths}


def planned(args):
    cases=read(REPO/'benchmarks/v2/validation.pt-BR.json');assert len(cases)==7
    tree=ast.parse((args.source/'fastpitch/model.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FastPitch')
    infer=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='infer')
    assert 'pace' in [a.arg for a in infer.args.args]
    effective=expected_frontend(cases)
    previous=ROOT/'fastpitch/language/common-probes-current7000-refinement'
    old=read(previous/'manifest.json');assert old['frontend_mode']=='refinement'
    assert old['code_sha256']==common.sha(common.__file__)
    assert old['loader_code_sha256']==common.sha(REPO/'training/fastpitch_experiment.py')
    prior=[json.loads(s) for s in (previous/'results.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    for case in cases:
        row=next(r for r in prior if r['case']==case['id'] and r['repetition']==0)
        assert row['effective_text']==effective[case['id']]
    pair_order=[dict(case=c['id'],pace=p,seed=SEED) for i,c in enumerate(cases) for p in (PACES if i%2==0 else PACES[::-1])]
    checkpoint=None
    if args.checkpoint is not None or args.checkpoint_sha256 is not None:
        if args.checkpoint is None or args.checkpoint_sha256 is None:raise ValueError('Require both frozen checkpoint and SHA')
        if not args.checkpoint.is_file() or args.checkpoint.is_symlink() or common.sha(args.checkpoint)!=args.checkpoint_sha256:raise ValueError('Exact frozen checkpoint mismatch')
        checkpoint=dict(path=str(args.checkpoint),sha256=args.checkpoint_sha256,bytes=args.checkpoint.stat().st_size,required_step=8000)
    return dict(status='prepared_only',hypothesis='At the same exact step8000, pace0.8 expands predicted grapheme/token durations ~1.25x; a0.7704 validation duration ratio would approach0.963. It may reduce omissions in rendered speech; only paired audio/ASR can establish the effect.',source_hashes=bindings(args),checkpoint=checkpoint,checkpoint_required_step=8000,checkpoint_substitution_allowed=False,
        pace_values=list(PACES),pair_order=pair_order,cases=cases,effective_texts=effective,observations=14,per_case_per_pace=1,warmups=0,
        precision='CPU FP32 acoustic/vocoder; compare pace within CPU8000, not against GPU/BF16 latency',seed=SEED,frontend='same refinement as7000, followed by pinned FastPitch NFC/lowercase',
        rss_cap_bytes=RSS_CAP,rss_watch_seconds=.25,minimum_available_ram_bytes=8*1024**3,ram_estimate='~2GiB model/load states plus up to~3GiB activations/library headroom;5GiB RSS is a ceiling hypothesis, not a measured guarantee. Abort on resource cap.',
        disk_cap_bytes=DISK_CAP,worst_pcm14x60sec_bytes=14*(int(60*24000)*2+44),retained_audio='PCM only, mono24k16bit; raw finite/peak/overshoot/bound counts and FLOAT SHA recorded, no expanded FLOAT WAVs',
        maximum_audio_seconds=60.,wall_budget_seconds=1800,models_resident_at_once=2,backend_instances_at_once=1,asr='Jobs only, immutable scorer4ee9 after separate root technical approval; no ASR execution in this helper',
        decisions='Retain every condition/case failure in n14; never compare selective successful subsets. No product pace change, A2000 timing, identity approval or final data.',torch_imported='torch' in sys.modules,gpu_used=False,tts_executed=False)


class PaceBackend(common.Backend):
    def generate_paced(self,text,pace):
        t=self.torch;seq=self.fp.tokens(text)[None].to('cpu')
        if seq.shape[1]>self.max_tokens:raise ValueError('effective text exceeds token bound')
        with t.inference_mode(),common.bounded_regulator(self.model_module,t,self.max_frames):
            mel,lens,durations,*_=self.acoustic.infer(seq,pace=pace)
            frames=int(lens[0])
            if not 0<frames<=self.max_frames or not bool(t.isfinite(mel).all()):raise ValueError('invalid acoustic mel/length')
            expected=(durations.float()/pace+.5).long().sum(dim=1)
            if int(expected[0])!=frames:raise ValueError('pace duration/regulated frame disagreement')
            audio=self.vocoder(mel[:,:,:frames].float()).squeeze().float().cpu().numpy()
        if audio.size!=frames*256:raise ValueError('vocoder output differs from acoustic frames/hop')
        return audio,dict(mel_frames=frames,predicted_duration_sum=float(durations.sum()),rounded_duration_frames=int(expected[0]),text_tokens=seq.shape[1])


def execute(args,plan):
    if not args.cpu_authorized:raise ValueError('Root technical review required before --cpu-authorized')
    if (ROOT/'pause-asr').exists():raise RuntimeError('CPU timing pause active')
    if plan['checkpoint'] is None:raise ValueError('A frozen exact step8000 checkpoint is required')
    if (args.output/'results.jsonl').exists():raise ValueError('Fresh execution output required')
    from training.v2_language_monitor import physical_usage
    roots=sorted(Path('/var/tmp').glob('aegis*'))+[Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip'),REPO,Path('/home/phc/.cache')]
    allocated=physical_usage(roots);free=shutil.disk_usage(args.output).free
    available=int(next(l for l in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines() if l.startswith('MemAvailable:')).split()[1])*1024
    if allocated['permission_errors'] or allocated['allocated_bytes']+DISK_CAP>40_000_000_000 or free<5*1024**3+DISK_CAP or available<8*1024**3:raise RuntimeError('CPU control space/RAM gate; preserve checkpoint and defer, not a model-feasibility result')
    common.json_atomic(args.output/'preflight.json',dict(working=allocated,output_peak_bytes=DISK_CAP,physical_free_bytes=free,available_ram_bytes=available))
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='2'
    import numpy as np
    import torch
    assert not torch.cuda.is_initialized()
    stop=threading.Event();aborted=[];started=time.monotonic()
    def interrupt(_sig,_frame):raise RuntimeError('CPU control resource/deadline stop; training untouched')
    previous=signal.signal(signal.SIGTERM,interrupt)
    def watch():
        while not stop.wait(.25):
            fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in Path('/proc/self/status').read_text(encoding='utf-8').splitlines() if ':' in line}
            rss=int(fields['VmRSS'].split()[0])*1024
            if rss>RSS_CAP or time.monotonic()-started>1800:
                aborted.append(dict(rss_bytes=rss,seconds=time.monotonic()-started));common.json_atomic(args.output/'resource-stop.json',aborted[-1]);os.kill(os.getpid(),signal.SIGTERM);return
    monitor=threading.Thread(target=watch,daemon=True);monitor.start();rows=[];jobs=[];backend=None
    try:
        backend=PaceBackend(args)
        if backend.metadata['step']!=8000:raise ValueError('Only exact step8000; no substitute checkpoint')
        common.json_atomic(args.output/'manifest.json',dict(plan=plan,backend=backend.metadata,scope='CPU control only; no A2000 latency measurement'))
        metrics=common.metrics_module();spoken=read(REPO/'benchmarks/v2/spoken-validation.pt-BR.json')['cases'];case_map={x['id']:x for x in plan['cases']}
        for item in plan['pair_order']:
            case=case_map[item['case']];pace=item['pace'];label=f'fastpitch-step8000-cpu-fp32-pace{pace:g}-refinement'
            row=dict(candidate=label,case=case['id'],pace=pace,repetition=0,seed=SEED,source_text=case['text'],effective_text=plan['effective_texts'][case['id']],device='cpu',precision='FP32',checkpoint_step=8000,scope='CPU analysis control; timing is not A2000 product latency')
            tick=time.perf_counter()
            try:
                if aborted:raise RuntimeError('Control aborted by resource bound')
                if (ROOT/'pause-asr').exists():raise RuntimeError('CPU pause resumed; do not execute remaining inference')
                random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
                assert backend.fp.normalize(common.frontend('refinement')(case['text'])['tts_text'])==row['effective_text']
                raw,details=backend.generate_paced(common.frontend('refinement')(case['text'])['tts_text'],pace)
                raw=np.asarray(raw,dtype=np.float32).reshape(-1);finite=bool(np.isfinite(raw).all())
                row.update(details,finite=finite,samples=len(raw),native_float_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),peak=float(abs(raw).max()) if raw.size and finite else None,nonfinite_samples=int((~np.isfinite(raw)).sum()),outside_unit=int((abs(raw)>1).sum()),samples_at_generator_bound=int((abs(raw)>=1).sum()))
                if not raw.size or not finite or row['outside_unit']:raise ValueError('empty/nonfinite/overshoot; no gain/clipping permitted')
                pcm=np.rint(raw*32767).astype('<i2').tobytes();path=args.output/f'{case["id"]}-pace{pace:g}.wav'
                if usage(args.output)+len(pcm)+16384>DISK_CAP:raise RuntimeError('50MB output cap; no selective WAV dropping')
                common.write_pcm(path,pcm);measure=metrics.pcm_metrics(pcm,[dict(ms=0.,bytes=len(pcm))])
                row.update(wav=str(path),wav_sha256=common.sha(path),pcm_sha256=hashlib.sha256(pcm).hexdigest(),audio_seconds=len(raw)/24000,onset_audio_offset_ms=measure['onset_audio_offset_ms'],clipped_samples=measure['clipped_samples'],outcome='ok' if measure['first_speech_pcm_available_ms'] is not None else 'no_speech')
                jobs.append(dict(candidate=label,case=case['id'],wav=str(path),expected_verbatim=case['text'],expected_spoken=spoken[case['id']],expected_effective=row['effective_text'],generation_outcome=row['outcome'],cohort='validation-step8000-cpu-fp32-fixedpace-pair',input_audio_sha256=row['wav_sha256']))
            except Exception as error:row.update(outcome='failed',error=(type(error).__name__+': '+str(error))[:512])
            row['cpu_elapsed_ms']=(time.perf_counter()-tick)*1000;rows.append(row)
            with (args.output/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            common.json_atomic(args.output/'asr-jobs.json',dict(jobs=jobs,expected_conditions=14,missing_or_invalid=[dict(case=x['case'],pace=x['pace'],outcome=x['outcome']) for x in rows if x['outcome']!='ok'],asr_executed=False))
        assert len(rows)==14
    finally:
        stop.set();monitor.join(2);signal.signal(signal.SIGTERM,previous)
        unchanged=bindings(args)==plan['source_hashes'] and common.sha(args.checkpoint)==args.checkpoint_sha256
        common.json_atomic(args.output/'execution.json',dict(observations=len(rows),expected=14,successes=sum(x['outcome']=='ok' for x in rows),failures=sum(x['outcome']!='ok' for x in rows),source_checkpoint_unchanged=unchanged,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,allocated_output_bytes=usage(args.output),cpu_only=True,cuda_initialized=torch.cuda.is_initialized(),aborted=aborted,asr_executed=False,final_test_opened=False))
        if not unchanged:raise RuntimeError('Source/checkpoint changed; control invalid')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=['plan','execute'],default='plan')
    p.add_argument('--output',type=Path,default=ROOT/'fastpitch/pace-control-step8000-preparation')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--checkpoint-sha256');p.add_argument('--cpu-authorized',action='store_true')
    p.add_argument('--source',type=Path,default=ROOT/'fastpitch/source');p.add_argument('--bigvgan',type=Path,default=ROOT/'bigvgan')
    a=p.parse_args();os.umask(0o077);os.nice(19);a.output=a.output.resolve()
    if any((x/'.git').exists() for x in (a.output,*a.output.parents)):raise ValueError('Private control output must stay outside Git')
    a.output.mkdir(parents=True,exist_ok=True,mode=0o700)
    if a.checkpoint:a.checkpoint=a.checkpoint.resolve()
    a.device='cpu';a.precision='fp32';a.threads=2;a.max_tokens=2048;a.max_audio_seconds=60.
    plan=planned(a)
    if a.action=='plan':
        path=a.output/('plan.json' if plan['checkpoint'] else 'plan.pending.json')
        if path.exists():raise ValueError('Preserve previous plan; choose a new output')
        common.json_atomic(path,plan);print(json.dumps(dict(status='prepared_only',plan=str(path),checkpoint_bound=plan['checkpoint'] is not None,torch_imported='torch' in sys.modules)));return
    if (a.output/'plan.json').exists() and read(a.output/'plan.json')!=plan:raise ValueError('Frozen plan differs')
    common.json_atomic(a.output/'plan.json',plan);execute(a,plan)


if __name__=='__main__':main()
