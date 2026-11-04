"""Seven fixed validation texts, CPU FP32, pace1 only. Default is a Torch-free plan."""
from __future__ import annotations
import argparse,fcntl,gc,importlib.metadata,json,math,os,random,resource,shutil,signal,stat,subprocess,sys,threading,time,wave
from pathlib import Path
from training import v2_fastpitch_benchmark as common
from training import v2_fastpitch_pace_control as pace

REPO=Path(__file__).resolve().parents[1];ROOT=Path('/var/tmp/aegis-v2-work')
SEED=40733133;DISK_CAP=50_000_000;RSS_CAP=5*1024**3;WALL=1800
PACE_SHA='67bee5515049dae75d61898783efd86090e2bb26555fec917f007299de8cd13e'
SCORER_SHA='4ee9f7da2cc8c733badb4d6d7c248a96ce0204a394fc26dc7652801a1410f241'
BASELINE=ROOT/'fastpitch/pace-control-step8000-eval-v2'
ASR_ROOT=Path('/var/tmp/aegis-v2-evaluation-20261102/asr')

class ControlAbort(BaseException):
    """Escape per-case Exception handlers while retaining the outer receipt/finally."""

def interrupt(*_):raise ControlAbort('Owned CPU control resource/time bound; training untouched')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):common.json_atomic(path,value)
def bindings(a):
    paths=[Path(__file__),Path(pace.__file__),Path(common.__file__),REPO/'training/fastpitch_experiment.py',REPO/'training/f5_refine_text.py',
           REPO/'gateway/aegis_speech/normalization.py',REPO/'training/v2_language_monitor.py',REPO/'benchmarks/v2/benchmark.py',
           REPO/'benchmarks/v2/asr_score.py',REPO/'training/f5_asr_metrics.py',REPO/'benchmarks/v2/validation.pt-BR.json',
           REPO/'benchmarks/v2/spoken-validation.pt-BR.json',a.source/'UPSTREAM.json',a.source/'fastpitch/model.py',
           a.bigvgan/'config.json',a.bigvgan/'revision.json',a.asr_root/'download-receipt.json']
    return {str(p):common.sha(p) for p in paths}
def git_common():
    path=Path(subprocess.check_output(['git','-C',str(REPO),'rev-parse','--git-common-dir'],text=True).strip())
    return path.resolve() if path.is_absolute() else (REPO/path).resolve()
def output_usage(output):
    frozen=output/'checkpoint/checkpoint.pt'
    return sum(p.lstat().st_blocks*512 for p in output.rglob('*') if p.is_file() and p!=frozen)
def finite_tree(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,list):return all(finite_tree(v) for v in value)
    return True

def freeze(a):
    source=a.checkpoint;receipt=read(source.with_suffix('.sha256.json'));st=source.lstat()
    if source.is_symlink() or not stat.S_ISREG(st.st_mode) or st.st_uid!=os.getuid() or receipt.get('step')!=a.expected_step or receipt.get('sha256')!=a.checkpoint_sha256 or receipt.get('bytes')!=st.st_size:
        raise ValueError('Source checkpoint owner/step/SHA/bytes differ')
    folder=a.output/'checkpoint';folder.mkdir(mode=0o700,exist_ok=True);target=folder/'checkpoint.pt'
    if not target.exists():os.link(source,target)
    linked=target.lstat()
    if target.is_symlink() or (linked.st_dev,linked.st_ino)!=(st.st_dev,st.st_ino) or common.sha(target)!=a.checkpoint_sha256:
        raise ValueError('Frozen checkpoint differs; preserve for review')
    value=dict(source=str(source),path=str(target),step=a.expected_step,sha256=a.checkpoint_sha256,bytes=st.st_size,device=st.st_dev,inode=st.st_ino,links=linked.st_nlink)
    save(folder/'checkpoint.sha256.json',dict(step=a.expected_step,sha256=a.checkpoint_sha256,bytes=st.st_size));save(folder/'freeze.json',value)
    return value

def planned(a):
    if common.sha(pace.__file__)!=PACE_SHA or common.sha(REPO/'benchmarks/v2/asr_score.py')!=SCORER_SHA:raise ValueError('Immutable pace/scorer source changed')
    cases=read(REPO/'benchmarks/v2/validation.pt-BR.json');old=read(BASELINE/'plan.json');rows=[json.loads(x) for x in (BASELINE/'results.jsonl').read_text().splitlines() if x]
    if len(cases)!=7 or len({c['id'] for c in cases})!=7 or cases!=old['cases']:raise ValueError('Fixed validation-seven roster changed')
    effective=pace.expected_frontend(cases)
    prior=[r for r in rows if r['pace']==1.]
    if len(prior)!=7 or any(r['outcome']!='ok' or r['seed']!=SEED or r['effective_text']!=effective[r['case']] for r in prior):raise ValueError('CPU8000 pace1 baseline is not comparable')
    if old['source_hashes'][str(REPO/'training/fastpitch_experiment.py')]!=common.sha(REPO/'training/fastpitch_experiment.py') or old['source_hashes'][str(Path(common.__file__))]!=common.sha(common.__file__):raise ValueError('Acoustic/backend source differs from CPU8000')
    baseline_asr=read(BASELINE/'asr-scores.json');baseline_rows=[r for r in baseline_asr['attempts'] if r['candidate']=='fastpitch-step8000-cpu-fp32-pace1-refinement']
    if baseline_asr['script_sha256']!=SCORER_SHA or len(baseline_rows)!=7 or not all(r['success'] for r in baseline_rows):raise ValueError('CPU8000 ASR baseline incomplete or different')
    baseline_edits=sum(r['comparisons']['verbatim']['word_edits'] for r in baseline_rows);baseline_words=sum(len(r['comparisons']['verbatim']['reference_tokens']) for r in baseline_rows)
    if (baseline_edits,baseline_words)!=(42,135):raise ValueError('Unexpected CPU8000 pace1 baseline score')
    return dict(status='prepared_only',expected_step=a.expected_step,checkpoint=freeze(a),source_hashes=bindings(a),cases=cases,effective_texts=effective,seed=SEED,pace=1.,observations=7,
        precision='CPU FP32 acoustic and pinned BigVGAN',threads=2,niceness=19,models_resident_at_once=2,backend_instances=1,warmups=0,per_case=1,
        frontend='same refinement then FastPitch NFC/lowercase as CPU8000',baseline=dict(step=8000,cohort='CPU FP32 pace1 seven fixed validation cases',plan_sha256=common.sha(BASELINE/'plan.json'),results_sha256=common.sha(BASELINE/'results.jsonl'),raw_verbatim_word_edits=baseline_edits,reference_words=baseline_words),
        disk_cap_bytes=DISK_CAP,temporary_checkpoint='Hardlink only; existing inode excluded from derived-output50MB cap, included/deduplicated in global40GB with Git common; remove after sealing if source retained',
        rss_cap_bytes=RSS_CAP,minimum_available_ram_bytes=8*1024**3,wall_seconds=WALL,asr=dict(scorer_sha256=SCORER_SHA,receipt_sha256=common.sha(a.asr_root/'download-receipt.json'),python=str(a.asr_python),settings=baseline_asr['settings'],versions=baseline_asr['versions']),
        policy='Validation7 only; no final-data flag. All7 outcomes remain; ASR only after7 valid WAVs and no public ASR active. Fresh output after partial; no A2000/product latency or identity approval.',torch_imported='torch' in sys.modules,gpu_used=False,tts_executed=False)

def preflight(a):
    from training.v2_language_monitor import physical_usage
    paths=sorted(Path('/var/tmp').glob('aegis*'))+[Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip'),REPO,Path('/home/phc/.cache'),git_common()]
    used=physical_usage(paths);free=shutil.disk_usage(a.output).free
    available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
    result=dict(working=used,git_common=str(paths[-1]),extra_output_bytes=DISK_CAP,free_bytes=free,available_ram_bytes=available)
    if used['permission_errors'] or used['allocated_bytes']+DISK_CAP>40_000_000_000 or free<5*1024**3+DISK_CAP or available<8*1024**3:raise RuntimeError('Operational CPU control disk/RAM gate; not model infeasibility')
    save(a.output/'preflight.json',result);return result

def active_asr(exclude=()):
    found=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) in exclude:continue
        try:
            cmd=(path/'cmdline').read_bytes().decode(errors='replace').split('\0');state=(path/'stat').read_text().rsplit(')',1)[1].split()[0]
            if state!='Z' and any(c.endswith('asr_score.py') for c in cmd):found.append(int(path.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return found
def child_identity(pid):
    p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,start_ticks=fields[19],uid=p.stat().st_uid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def stop_owned(child,proof,output):
    if child is None or child.poll() is not None:return
    command=Path(f'/proc/{child.pid}/cmdline').read_bytes().decode().split('\0')
    if child_identity(child.pid)!=proof or proof['uid']!=os.getuid() or str(output/'asr-jobs.json') not in command or str(REPO/'benchmarks/v2/asr_score.py') not in command:raise RuntimeError('Refuse signal to unverified ASR child')
    child.terminate();child.wait(timeout=10)
def validate_rows(rows,cases):
    if len(rows)!=7 or {r['case'] for r in rows}!={c['id'] for c in cases} or any(r.get('outcome')!='ok' for r in rows):raise ValueError('All7 valid conditions required; never score a filtered subset')
    for row in rows:
        with wave.open(row['wav'],'rb') as wav:
            if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())!=(24000,1,2) or wav.getnframes()<=0:raise ValueError('Invalid complete PCM WAV')
            if len(wav.readframes(wav.getnframes()))!=wav.getnframes()*2:raise ValueError('Truncated PCM WAV payload')
        if common.sha(row['wav'])!=row['wav_sha256']:raise ValueError('WAV changed before ASR')
def audit_asr(receipt,jobs,plan):
    attempts=receipt.get('attempts',[])
    expected={(j['case'],j['input_audio_sha256']) for j in jobs};observed={(r.get('case'),r.get('audio_sha256')) for r in attempts}
    if receipt.get('script_sha256')!=SCORER_SHA or receipt.get('model_receipt_sha256')!=plan['asr']['receipt_sha256'] or not receipt.get('completed_utc') or receipt.get('deadline_reached') or len(attempts)!=7 or observed!=expected or not all(r.get('success') for r in attempts):raise ValueError('Incomplete/different ASR: preserve denominator7, no filtered score')
    if receipt['settings']!=plan['asr']['settings'] or receipt['versions']!=plan['asr']['versions']:raise ValueError('ASR settings/versions differ from CPU8000')
    by_case={j['case']:j for j in jobs}
    for row in attempts:
        job=by_case[row['case']]
        if row['candidate']!=job['candidate'] or any(row.get(k)!=v for k,v in job.items() if k.startswith('expected_')):raise ValueError('ASR candidate/expected-text policy differs')
    return dict(word_edits=sum(r['comparisons']['verbatim']['word_edits'] for r in attempts),reference_words=sum(len(r['comparisons']['verbatim']['reference_tokens']) for r in attempts),character_edits=sum(r['comparisons']['verbatim']['character_edits'] for r in attempts),reference_characters=sum(r['comparisons']['verbatim']['reference_characters'] for r in attempts))
def cleanup(a,plan):
    cp=Path(plan['checkpoint']['path']);source=Path(plan['checkpoint']['source'])
    if not cp.exists():return dict(status='already_absent')
    st=cp.lstat()
    if cp.is_symlink() or st.st_uid!=os.getuid() or cp.resolve().parent!=(a.output/'checkpoint').resolve():raise ValueError('Unsafe temporary checkpoint cleanup')
    try:original=source.lstat()
    except FileNotFoundError:return dict(status='preserved_for_review',reason='Original retained inode no longer available')
    if source.is_symlink() or not stat.S_ISREG(original.st_mode) or original.st_uid!=os.getuid() or (original.st_dev,original.st_ino)!=(st.st_dev,st.st_ino) or st.st_nlink<2:return dict(status='preserved_for_review',reason='Original retained inode no longer safely available')
    users=subprocess.run(['fuser',str(cp)],capture_output=True,timeout=10)
    if users.returncode!=1 or users.stdout or users.stderr:return dict(status='preserved_for_review',reason='Temporary link in use or fuser inconclusive')
    cp.unlink();return dict(status='removed_only_owned_temporary_hardlink',inode=st.st_ino,links_before=st.st_nlink,links_after=source.stat().st_nlink,weight_data_blocks_freed=0)

def execute(a,plan):
    started=time.monotonic();plan_sha=common.sha(a.output/'plan.json')
    if not a.cpu_authorized:raise ValueError('Root review required before neural execution')
    if (a.output/'results.jsonl').exists() or (a.output/'execution.json').exists():raise ValueError('Fresh execution output required after any partial run')
    if (ROOT/'pause-asr').exists() or active_asr():raise RuntimeError('CPU pause/public ASR active; defer without scoring')
    preflight(a);os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='2'
    for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[key]='1'
    import numpy as np
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only control requires uninitialized CUDA')
    stop=threading.Event();aborted=[];rows=[];jobs=[];backend=None;child=None;child_proof=None;summary=None;failure=None;rss_peaks=[0]
    previous=signal.signal(signal.SIGTERM,interrupt)
    def watch():
        while not stop.wait(.25):
            rss=int(next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024
            if child is not None and child.poll() is None:
                try:rss+=int(next(x for x in Path(f'/proc/{child.pid}/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])*1024
                except (FileNotFoundError,ProcessLookupError):pass
            rss_peaks[0]=max(rss_peaks[0],rss)
            if rss>RSS_CAP or time.monotonic()-started>WALL:
                aborted.append(dict(rss_bytes=rss,seconds=time.monotonic()-started));save(a.output/'resource-stop.json',aborted[-1]);os.kill(os.getpid(),signal.SIGTERM);return
    monitor=threading.Thread(target=watch,daemon=True);monitor.start()
    try:
        class_args=argparse.Namespace(**vars(a));class_args.checkpoint=Path(plan['checkpoint']['path']);class_args.device='cpu';class_args.precision='fp32';class_args.threads=2;class_args.max_tokens=2048;class_args.max_audio_seconds=60.
        backend=pace.PaceBackend(class_args)
        if backend.metadata['step']!=a.expected_step:raise ValueError('Loaded model is not the required exact step')
        versions={k:importlib.metadata.version(k) for k in ('torch','numpy','librosa','soundfile','scipy')}
        save(a.output/'manifest.json',dict(plan=plan,backend=backend.metadata,versions=versions,command=sys.argv,models=2,backends=1,scope='CPU language control only; no product timing'))
        metrics=common.metrics_module();spoken=read(REPO/'benchmarks/v2/spoken-validation.pt-BR.json')['cases']
        for case in plan['cases']:
            row=dict(candidate=f'fastpitch-step{a.expected_step}-cpu-fp32-pace1-refinement',case=case['id'],pace=1.,seed=SEED,source_text=case['text'],effective_text=plan['effective_texts'][case['id']],checkpoint_step=a.expected_step,precision='FP32',device='cpu');tick=time.perf_counter()
            try:
                if aborted or (ROOT/'pause-asr').exists():raise ControlAbort('CPU control aborted/paused')
                random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
                effective=common.frontend('refinement')(case['text'])['tts_text']
                if backend.fp.normalize(effective)!=row['effective_text']:raise ValueError('Frontend changed')
                raw,details=backend.generate_paced(effective,1.);raw=np.asarray(raw,dtype=np.float32).reshape(-1)
                valid=bool(np.isfinite(raw).all());row.update(details,finite=valid,samples=len(raw),native_float_sha256=__import__('hashlib').sha256(raw.tobytes()).hexdigest(),peak=float(abs(raw).max()) if raw.size and valid else None,outside_unit=int((abs(raw)>1).sum()),samples_at_generator_bound=int((abs(raw)>=1).sum()))
                if not raw.size or not valid or row['outside_unit']:raise ValueError('Empty/nonfinite/overshoot; no gain/clipping')
                pcm=np.rint(raw*32767).astype('<i2').tobytes()
                if output_usage(a.output)+len(pcm)+16384>DISK_CAP:raise RuntimeError('Derived output50MB cap')
                path=a.output/(case['id']+'.wav');common.write_pcm(path,pcm);measure=metrics.pcm_metrics(pcm,[dict(ms=0.,bytes=len(pcm))])
                row.update(wav=str(path),wav_sha256=common.sha(path),audio_seconds=len(raw)/24000,outcome='ok' if measure['first_speech_pcm_available_ms'] is not None else 'no_speech',clipped_samples=measure['clipped_samples'],onset_audio_offset_ms=measure['onset_audio_offset_ms'])
                jobs.append(dict(candidate=row['candidate'],case=case['id'],wav=str(path),expected_verbatim=case['text'],expected_spoken=spoken[case['id']],expected_effective=row['effective_text'],generation_outcome=row['outcome'],cohort='validation-common7-cpu-fp32-pace1',input_audio_sha256=row['wav_sha256']))
            except Exception as error:row.update(outcome='failed',error=(type(error).__name__+': '+str(error))[:512])
            row['cpu_elapsed_ms']=(time.perf_counter()-tick)*1000;rows.append(row)
            with (a.output/'results.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        del backend;backend=None;gc.collect()
        save(a.output/'asr-jobs.json',dict(jobs=jobs,expected_conditions=7,missing_or_invalid=[r['case'] for r in rows if r['outcome']!='ok']))
        validate_rows(rows,plan['cases'])
        while (ROOT/'pause-asr').exists() or active_asr():
            if time.monotonic()-started>WALL:raise RuntimeError('ASR defer bound; no filtered result')
            time.sleep(2)
        scorer=REPO/'benchmarks/v2/asr_score.py'
        if common.sha(scorer)!=SCORER_SHA:raise ValueError('Pinned ASR source changed')
        command=[str(a.asr_python),str(scorer),'--jobs',str(a.output/'asr-jobs.json'),'--output',str(a.output/'asr-scores.json'),'--asr-root',str(a.asr_root),'--deadline-seconds',str(max(1,WALL-(time.monotonic()-started)))]
        with (a.output/'asr.log').open('ab') as log:child=subprocess.Popen(command,cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        child_proof=child_identity(child.pid);save(a.output/'asr-process.json',dict(**child_proof,command=command))
        while child.poll() is None:
            if active_asr((child.pid,)) or (ROOT/'pause-asr').exists():stop_owned(child,child_proof,a.output);raise RuntimeError('Another ASR/pause appeared; owned ASR stopped, result incomplete')
            if (a.output/'asr.log').stat().st_size>1024**2:
                with (a.output/'asr.log').open('r+b') as log:log.seek(-512*1024,2);tail=log.read();log.seek(0);log.write(tail);log.truncate()
            time.sleep(.5)
        if child.returncode:raise RuntimeError('Owned ASR failed')
        summary=audit_asr(read(a.output/'asr-scores.json'),jobs,plan);save(a.output/'comparison.json',dict(step=a.expected_step,after=summary,baseline=plan['baseline'],scope='Same seven source/effective texts, seed, FP32 CPU pace1 only; no GPU latency or identity verdict'))
    except BaseException as error:failure=(type(error).__name__+': '+str(error))[:1024];raise
    finally:
        stop_owned(child,child_proof,a.output)
        stop.set();monitor.join(2);signal.signal(signal.SIGTERM,previous)
        unchanged=bindings(a)==plan['source_hashes'] and common.sha(plan['checkpoint']['path'])==a.checkpoint_sha256 and common.sha(a.output/'plan.json')==plan_sha
        artifacts={str(p):common.sha(p) for p in a.output.glob('*') if p.is_file() and p.name not in ('control.lock','execution.json')}
        save(a.output/'execution.json',dict(expected=7,observations=len(rows),successes=sum(r['outcome']=='ok' for r in rows),failures=7-sum(r['outcome']=='ok' for r in rows),source_checkpoint_unchanged=unchanged,plan_sha256=plan_sha,artifact_sha256=artifacts,error=failure,asr_complete=summary is not None,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,sampled_process_tree_rss_peak_bytes=rss_peaks[0],cuda_initialized=torch.cuda.is_initialized(),cpu_only=True,final_test_opened=False,target50_opened=False,aborted=aborted,allocated_derived_output_bytes=output_usage(a.output)))
        if unchanged:save(a.output/'temporary-checkpoint-cleanup.json',cleanup(a,plan))
        if not unchanged:raise RuntimeError('Source/checkpoint changed; control invalid')

def main():
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group();group.add_argument('--plan',dest='action',action='store_const',const='plan');group.add_argument('--execute',dest='action',action='store_const',const='execute');p.set_defaults(action='plan')
    p.add_argument('--expected-step',type=int,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-authorized',action='store_true')
    p.add_argument('--source',type=Path,default=ROOT/'fastpitch/source');p.add_argument('--bigvgan',type=Path,default=ROOT/'bigvgan');p.add_argument('--asr-root',type=Path,default=ASR_ROOT);p.add_argument('--asr-python',type=Path,default=Path('/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python'));a=p.parse_args()
    if a.expected_step<=0 or len(a.checkpoint_sha256)!=64 or any(c not in '0123456789abcdef' for c in a.checkpoint_sha256):p.error('Positive exact step and lowercase SHA256 required')
    os.umask(0o077);os.nice(19);a.output=a.output.resolve()
    if any((q/'.git').exists() for q in (a.output,*a.output.parents)):p.error('Private output must stay outside Git')
    a.output.mkdir(parents=True,exist_ok=True,mode=0o700);lock=(a.output/'control.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.action=='plan':
        if (a.output/'plan.json').exists():raise ValueError('Preserve previous plan; use a new output')
        plan=planned(a);save(a.output/'plan.json',plan);print(json.dumps(dict(status='prepared_only',plan=str(a.output/'plan.json'),torch_imported='torch' in sys.modules)));return
    plan=read(a.output/'plan.json')
    if plan['expected_step']!=a.expected_step or plan['checkpoint']['sha256']!=a.checkpoint_sha256 or bindings(a)!=plan['source_hashes']:raise ValueError('Frozen plan differs; no neural execution')
    execute(a,plan)

if __name__=='__main__':main()
