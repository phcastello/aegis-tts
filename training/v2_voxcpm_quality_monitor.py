"""Local CPU ASR of frozen VoxCPM validation audio; never controls GPU/training."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from training.v2_language_monitor import physical_usage, ticks, utc
from training.v2_fastpitch_quality_monitor import read, sha, write

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/var/tmp/aegis-v2-work')
PYTHON='/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python'
CP_LIMIT=70*1024**2
RESERVE=384*1024**2
KEEP_CHECKPOINTS=3

def sync_dir(path):
    fd=os.open(path,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def process_identity(pid):
    """Read-only /proc proof; exec changes argv while preserving PID/start."""
    try:
        root=Path(f'/proc/{int(pid)}');before=(root/'stat').read_text().rsplit(') ',1)[1].split()
        command=(root/'cmdline').read_bytes().decode(errors='replace').split('\0')
        after=(root/'stat').read_text().rsplit(') ',1)[1].split()
        if before[19]!=after[19] or after[0]=='Z':return None
        return dict(pid=int(pid),ppid=int(after[1]),start_ticks=after[19],uid=root.stat().st_uid,command=[v for v in command if v])
    except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError,IndexError):return None

def flag(command,name):
    try:return command[command.index(name)+1]
    except (ValueError,IndexError):return next((v.split('=',1)[1] for v in command if v.startswith(name+'=')),None)

def trainer_binding(root,run,worker,previous=None):
    """Health evidence only: identify the worker's actual child after a gate exec."""
    if worker.get('status')!='running':return None
    parent=process_identity(worker.get('worker_pid',-1));child=process_identity(worker.get('pid',-1))
    if not parent or not child or parent['uid']!=os.getuid() or child['uid']!=os.getuid() or child['ppid']!=parent['pid']:return None
    if 'training.v2_gpu_worker' not in parent['command'] or flag(parent['command'],'--root')!=str(root):return None
    anchor=read(root/'operations/gpu-window.json',{})
    if anchor and (anchor.get('child_pid')!=parent['pid'] or str(anchor.get('child_start'))!=parent['start_ticks']):return None
    command=child['command']
    module=flag(command,'-m')=='training.v2_voxcpm_train'
    script=str(REPO/'training/v2_voxcpm_train.py') in command
    if not (module or script) or flag(command,'--run')!=str(run) or flag(command,'--action')!='train':return None
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    identity=dict(**child,worker_pid=parent['pid'],worker_start_ticks=parent['start_ticks'],boot_id=boot,
                  job_started_utc=worker.get('started_utc'),job_command=worker.get('job',{}).get('command',[]))
    if previous and previous.get('boot_id')==boot and previous.get('job_started_utc')==identity['job_started_utc']:
        if any(previous.get(k)!=identity[k] for k in ('pid','start_ticks','worker_pid','worker_start_ticks')):return None
    return identity

def owned(active):
    """Only signal the boot/UID/start/session-bound scorer with our exact jobs."""
    pid=active['pid']
    try:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        return (Path('/proc/sys/kernel/random/boot_id').read_text().strip()==active['boot_id']
            and ticks(pid)==str(active['start_ticks']) and Path(f'/proc/{pid}').stat().st_uid==os.getuid()
            and os.getpgid(pid)==pid and os.getsid(pid)==pid and active['jobs'] in command
            and str(REPO/'benchmarks/v2/asr_score.py') in command)
    except (FileNotFoundError,ProcessLookupError,PermissionError):return False

def pause_child(active,pause,force=False):
    if not force and bool(active.get('paused'))==pause:return
    if not owned(active):raise RuntimeError('Refuse signal to an unverified ASR process')
    os.killpg(active['pid'],signal.SIGSTOP if pause else signal.SIGCONT);active['paused']=pause

def checkpoint(run,step,target):
    """Sidecars precede/follow atomic publication; defer until step+SHA agree."""
    if target.exists():
        saved=read(target.with_suffix('.sha256.json'),{})
        if target.is_symlink() or target.stat().st_uid!=os.getuid() or target.stat().st_size>CP_LIMIT:raise ValueError('Unsafe existing checkpoint link')
        if saved.get('step')==step and sha(target)==saved.get('sha256'):return saved
        target.unlink()  # Uncommitted monitor-only link after a crash; never a trainer path.
    for label in ('last','best'):
        source=run/(label+'.pt');receipt=run/(label+'.sha256.json');before=read(receipt,{})
        if before.get('step')!=step or not source.exists():continue
        stat=source.lstat()
        if source.is_symlink() or stat.st_uid!=os.getuid() or not 0<stat.st_size<=CP_LIMIT:raise ValueError('Unsafe/oversize checkpoint')
        os.link(source,target)
        if ((target.stat().st_dev,target.stat().st_ino)==(stat.st_dev,stat.st_ino)
                and sha(target)==before.get('sha256') and read(receipt)==before):
            return dict(**before,source=str(source),binding='Trainer-validated sidecar step and exact frozen-file SHA; no model load')
        target.unlink()
    return None

def retention(jobs,best,training_best,active,keep=6):
    selected={j['key'] for j in sorted(jobs,key=lambda j:(j['step'],j['key']))[-keep:]}
    selected.update(j['key'] for j in jobs if j['step'] in (0,training_best))
    return selected|{k for k in (best,active) if k}

def checkpoint_retention(jobs,best,best_checkpoint=None):
    bound=[j for j in jobs if j.get('checkpoint') and not j.get('removed_utc')]
    selected={j['key'] for j in sorted(bound,key=lambda j:(j['step'],j['key']))[-KEEP_CHECKPOINTS:]}
    # Keep the recoverable best until a newly selected best obtains its exact link.
    return selected|({best,best_checkpoint}&{j['key'] for j in bound})

def freeze(source,target,corpus,spoken,policy):
    raw=source.read_bytes();data=json.loads(raw);rows=data['results'];source_jobs=(source.parent/'asr-jobs.json').read_bytes()
    declared=json.loads(source_jobs)['jobs'];ok={r['case']:r for r in rows if r.get('outcome')=='ok'}
    if len(rows)!=7 or {r['case'] for r in rows}!=set(corpus) or len(declared)!=len(ok) or {r['case'] for r in declared}!=set(ok):raise ValueError('Expected seven unique validation outcomes and their exact successful jobs')
    if 'expected_nfe' in policy and any(r.get('nfe')!=policy['expected_nfe'] for r in rows):raise ValueError('Native probe NFE differs from the declared comparison policy')
    key=hashlib.sha256(raw+source_jobs+json.dumps(policy,sort_keys=True).encode()).hexdigest();directory=target/f'step-{data["step"]:08d}-{key[:12]}'
    if directory.exists():return read(directory/'source.json')
    staging=target/('.snapshot-'+key[:12]);shutil.rmtree(staging,ignore_errors=True);staging.mkdir(mode=0o700)
    try:
        jobs=[];hashes={}
        for item in declared:
            case=item['case'];row=ok[case];wave=Path(row['wav']);dest=staging/(case+'.wav')
            if wave.is_symlink() or wave.resolve().parent!=source.parent.resolve() or wave.stat().st_uid!=os.getuid() or item['wav']!=str(wave) or item['expected_verbatim']!=corpus[case]['text']:raise ValueError('Unexpected validation audio/text binding')
            os.link(wave,dest);hashes[case]=sha(dest)
            if hashes[case]!=row['sha256']:raise ValueError('WAV changed while snapshotting')
            jobs.append(dict(candidate='voxcpm2-aegis',case=case,cohort='validation-seven-native-eager',wav=str(directory/(case+'.wav')),
                expected_verbatim=corpus[case]['text'],expected_spoken=spoken[case],expected_effective=corpus[case]['text']))
        (staging/'source-results.json').write_bytes(raw);(staging/'source-jobs.json').write_bytes(source_jobs);write(staging/'jobs.json',dict(jobs=jobs))
        job=dict(key=key,step=data['step'],directory=str(directory),status='pending',created_utc=utc(),policy=policy,
            source_results_sha256=hashlib.sha256(raw).hexdigest(),source_jobs_sha256=hashlib.sha256(source_jobs).hexdigest(),jobs_sha256=sha(staging/'jobs.json'),
            audio_hashes=hashes,outcomes={r['case']:r.get('outcome') for r in rows},nfe={r['case']:r.get('nfe') for r in rows},product=False,case_denominator=7)
        write(staging/'source.json',job)
        for path in staging.iterdir():
            with path.open('rb') as stream:os.fsync(stream.fileno())
        sync_dir(staging);staging.rename(directory);sync_dir(target);return job
    finally:
        if staging.exists():shutil.rmtree(staging)

class Monitor:
    def __init__(self,args):
        self.a=args;self.run=args.run.resolve();self.root=args.root.resolve();self.out=self.run/'quality-monitor';self.probes=self.out/'probes';self.probes.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.state=read(self.out/'state.json',dict(jobs={},best_asr=None));self.child=read(self.out/'active.json');self.proc=None;self.running=True;self.last_poll=self.last_budget=self.last_heartbeat=0.
        self.scorer=REPO/'benchmarks/v2/asr_score.py';self.corpus=REPO/'benchmarks/v2/validation.pt-BR.json';self.spoken=REPO/'benchmarks/v2/spoken-validation.pt-BR.json'
        self.policy={name:sha(path) for name,path in dict(scorer=self.scorer,corpus=self.corpus,spoken=self.spoken,asr=Path(args.asr_root)/'download-receipt.json').items()}
        self.policy['expected_nfe']=args.expected_nfe
        self.log=logging.getLogger('voxcpm-quality');self.log.setLevel(logging.INFO);self.log.addHandler(RotatingFileHandler(self.out/'monitor.log',maxBytes=1024**2,backupCount=1))
        write(self.out/'recipe.json',dict(utc=utc(),pid=os.getpid(),start_ticks=ticks(os.getpid()),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),command=sys.argv,code_sha256=sha(__file__),policy=self.policy,
            run=str(self.run),poll_seconds=args.poll_seconds,child_poll_seconds=1,heartbeat_seconds=1800,timeout_active_seconds=args.timeout,keep_latest=6,keep_latest_exact_checkpoints=KEEP_CHECKPOINTS,checkpoint_limit_bytes=CP_LIMIT,estimated_extra_bytes=RESERVE,
            scope='CPU2 nice19 pinned/offline ASR only; seven-case denominator; eager NFE recorded per case; no product latency, listening, model approval or GPU control',pause_file=str(self.root/'pause-asr')))
        for path in self.probes.glob('step-*/source.json'):
            job=read(path)
            if Path(job['directory']).resolve()==path.parent.resolve():self.state['jobs'].setdefault(job['key'],job)
        if self.child and not owned(self.child):self.complete(self.child,None);self.child=None;(self.out/'active.json').unlink(missing_ok=True)

    def save(self):write(self.out/'state.json',self.state)
    def review(self,kind,**values):
        path=self.out/'review-needed.json';data=read(path,dict(notices=[]));data['notices']=[n for n in data['notices'] if n['kind']!=kind]
        data['notices'].append(dict(kind=kind,utc=utc(),**values));data['notices']=data['notices'][-100:];data['interpretation']='ASR/health review only; no automatic training action or voice verdict';write(path,data);self.log.info(json.dumps(data['notices'][-1]))
    def budget_ok(self):
        usage=self.state.get('budget',{});return bool(usage) and not usage.get('permission_errors') and usage['allocated_bytes']+RESERVE<=40_000_000_000 and shutil.disk_usage(self.root).free>=5*1024**3+RESERVE
    def observe(self):
        now=time.monotonic();pause=(self.root/'pause-asr').exists()
        if not pause and now-self.last_budget>=300:
            roots=sorted(Path('/var/tmp').glob('aegis*'))+[Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip'),REPO,Path('/home/phc/.cache')]
            self.state['budget']=physical_usage(roots);self.last_budget=now
            if not self.budget_ok():self.review('budget_blocks_new_ASR',usage=self.state['budget'],free_bytes=shutil.disk_usage(self.root).free)
        progress=read(self.run/'progress.json',{});history=[]
        worker=read(self.root/'worker-current.json',{});previous=self.state.get('trainer_identity')
        identity=trainer_binding(self.root,self.run,worker,previous)
        if identity:self.state['trainer_identity']=identity
        binding=identity or previous;command=worker.get('job',{}).get('command',[])
        for path in list((self.root/'queue').glob('*.json'))+list((self.root/'done').glob('*.json')):
            item=read(path,{});job_command=item.get('job',item).get('command',[])
            direct=str(self.run) in job_command and 'training.v2_voxcpm_train' in job_command
            gated=bool(binding and job_command==binding['job_command'] and (not item.get('started_utc') or item.get('started_utc')==binding['job_started_utc']))
            if direct or gated:history.append((item.get('started_utc',''),item,path))
        matching=bool(identity or (binding and binding['job_command']==command and binding['job_started_utc']==worker.get('started_utc')))
        if matching:
            for _,item,path in list(history):
                if path.parent.name=='queue' and item.get('command')==command:history.append((worker.get('started_utc',''),worker,path))
        running=identity is not None
        age=time.time()-(self.run/'progress.json').stat().st_mtime if progress else (time.time()-dt.datetime.fromisoformat(worker['started_utc']).timestamp() if running and worker.get('started_utc') else None)
        if progress and not running:self.review('trainer_not_running',worker=worker if matching else None,last_progress=progress)
        if running and age is not None and age>600:self.review('progress_stale',age_seconds=age,step=progress.get('step'))
        if any(isinstance(v,(int,float)) and not math.isfinite(v) for v in progress.values()):self.review('nonfinite_progress',progress=progress)
        for _,item,path in sorted(history,key=lambda r:(r[0],r[2].name))[-1:]:
            log=self.root/'logs'/(path.stem+'.log')
            if log.exists():
                with log.open('rb') as stream:stream.seek(max(0,log.stat().st_size-32768));tail=stream.read().decode(errors='replace')
                if re.search(r'\b(nan|nonfinite|out of memory|traceback)\b',tail,re.I):self.review('trainer_log_error',log=str(log),status=item.get('status'))
            if item.get('status') in ('failed','succeeded'):self.review('trainer_exit',result=item)
        status=dict(utc=utc(),progress=progress,progress_file=str(self.run/'progress.json'),trainer_running=running,
            trainer_phase='updates_or_validation' if running and progress else 'initialization_or_initial_validation' if running else 'not_verified_running',
            trainer_identity=identity,progress_age_seconds=age,active_asr=self.child,paused=pause,working_bytes=self.state.get('budget',{}).get('allocated_bytes'),free_bytes=shutil.disk_usage(self.root).free)
        write(self.out/'status.json',status)
        if now-self.last_heartbeat>=1800:write(self.out/'heartbeat.json',status);self.log.info(json.dumps(dict(event='heartbeat',**status)));self.last_heartbeat=now
    def discover(self):
        if (self.root/'pause-asr').exists() or not self.budget_ok():return
        if any(sha(path)!=self.policy[name] for name,path in dict(scorer=self.scorer,corpus=self.corpus,spoken=self.spoken,asr=Path(self.a.asr_root)/'download-receipt.json').items()):raise ValueError('Scoring policy changed')
        corpus={r['id']:r for r in read(self.corpus)};spoken=read(self.spoken)['cases'];history=self.run/'validation.jsonl'
        for line in history.read_text().splitlines() if history.exists() else []:
            try:
                record=json.loads(line);source=Path(record['probes']['directory'])/'results.json'
                if source.parent.parent.resolve()!=(self.run/'probes').resolve():raise ValueError('Unexpected trainer probe directory')
                if not source.exists():continue
                job=freeze(source,self.probes,corpus,spoken,self.policy)
                if job['step']!=record['step']:raise ValueError('Validation step mismatch')
                self.state['jobs'].setdefault(job['key'],job)
            except (FileNotFoundError,json.JSONDecodeError):continue
        self.capture_checkpoints();self.save()
    def capture_checkpoints(self):
        jobs=[j for j in self.state['jobs'].values() if not j.get('removed_utc')]
        latest={j['key'] for j in sorted(jobs,key=lambda j:(j['step'],j['key']))[-KEEP_CHECKPOINTS:]}
        for job in sorted(jobs,key=lambda j:(j['key']!=self.state.get('best_asr'),-j['step'],j['key'])):
            selected=job['key']==self.state.get('best_asr')
            if job.get('checkpoint') or (job['key'] not in latest and not selected):continue
            target=Path(job['directory'])/'checkpoint.pt';binding=checkpoint(self.run,job['step'],target)
            if binding:
                job['checkpoint']=binding;write(target.with_suffix('.sha256.json'),binding);sync_dir(target.parent)
                if selected:self.state['best_checkpoint']=job['key']
            elif selected and read(self.run/'last.sha256.json',{}).get('step',-1)>job['step'] and read(self.run/'best.sha256.json',{}).get('step')!=job['step']:
                self.review('best_ASR_checkpoint_unavailable',step=job['step'],reason='Trainer receipts already advanced; no mismatched weights retained')
    def start(self):
        if not self.running or self.child or (self.root/'pause-asr').exists() or not self.budget_ok():return
        pending=sorted((j for j in self.state['jobs'].values() if j['status']=='pending'),key=lambda j:j['step'])
        if not pending:return
        job=pending[0];directory=Path(job['directory'])
        if any(sha(path)!=self.policy[name] for name,path in dict(scorer=self.scorer,corpus=self.corpus,spoken=self.spoken,asr=Path(self.a.asr_root)/'download-receipt.json').items()):raise ValueError('Scoring policy changed before launch')
        if job['policy']!=self.policy or sha(directory/'jobs.json')!=job['jobs_sha256'] or sha(directory/'source-results.json')!=job['source_results_sha256'] or sha(directory/'source-jobs.json')!=job['source_jobs_sha256'] or any(sha(directory/(case+'.wav'))!=value for case,value in job['audio_hashes'].items()):raise ValueError('Frozen ASR inputs changed')
        if not job['audio_hashes']:self.complete(dict(key=job['key']),0);return
        command=['nice','-n','19',self.a.python,str(self.scorer),'--jobs',str(directory/'jobs.json'),'--output',str(directory/'scores.json'),'--asr-root',str(self.a.asr_root),'--deadline-seconds','31536000']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        if not self.running or (self.root/'pause-asr').exists():return
        with (directory/'asr.log').open('ab') as stream:self.proc=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        self.child=dict(key=job['key'],pid=self.proc.pid,start_ticks=ticks(self.proc.pid),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),jobs=str(directory/'jobs.json'),command=command,paused=False,active_seconds=0.,last_monotonic=time.monotonic(),started_utc=utc())
        job['status']='running';write(self.out/'active.json',self.child);self.save()
    def complete(self,active,code):
        job=self.state['jobs'][active['key']]
        if job['status'] in ('complete','failed'):return
        directory=Path(job['directory']);receipt=read(directory/'scores.json',{});attempts=receipt.get('attempts',[])
        success=[r for r in attempts if r.get('success')];observed={(r.get('case'),r.get('audio_sha256')) for r in success};expected=set(job['audio_hashes'].items())
        comparable=(code in (0,None) and receipt.get('completed_utc') and not receipt.get('deadline_reached') and receipt.get('script_sha256')==job['policy']['scorer'] and receipt.get('model_receipt_sha256')==job['policy']['asr'] and len(attempts)==len(success)==len(expected)==7 and expected==observed and all(sha(directory/(case+'.wav'))==value for case,value in expected))
        metrics=dict(step=job['step'],key=job['key'],scores_sha256=sha(directory/'scores.json') if (directory/'scores.json').exists() else None,product=False,case_denominator=7,successful_cases=len(success),synthesis_failures=7-len(expected),asr_failures=len(attempts)-len(success),missing_cases=max(0,7-len(attempts)),comparable=bool(comparable),comparisons={})
        if comparable:
            for kind in ('verbatim','spoken','effective'):
                edits=sum(r['comparisons'][kind]['word_edits'] for r in success);words=sum(len(r['comparisons'][kind]['reference_tokens']) for r in success)
                metrics['comparisons'][kind]=dict(word_edits=edits,reference_words=words,wer=edits/max(1,words))
        job.update(status='complete' if comparable else 'failed',metrics=metrics,finished_utc=utc(),exit_code=code);old=self.state['jobs'].get(self.state.get('best_asr'),{})
        if comparable and (old.get('policy')!=job['policy'] or metrics['comparisons']['verbatim']['wer']<old.get('metrics',{}).get('comparisons',{}).get('verbatim',{}).get('wer',float('inf'))):
            self.state['best_asr']=job['key']
            if job.get('checkpoint'):self.state['best_checkpoint']=job['key']
            else:self.review('best_ASR_checkpoint_deferred',step=job['step'],reason='Validation precedes checkpoint publication; retry exact step/SHA without holding up ASR')
        write(directory/'metrics.json',metrics)
        baseline=next((j for j in self.state['jobs'].values() if j['step']==0 and j.get('policy')==job['policy'] and j.get('metrics',{}).get('comparable')),None)
        write(directory/'comparison-to-initial.json',dict(product=False,case_denominator=7,filtered=False,comparable=bool(comparable and baseline),
            before=baseline.get('metrics') if baseline else None,after=metrics,
            interpretation='Same seven validation cases and fixed policy only; failures never disappear from the denominator; ASR cannot approve pronunciation/identity'))
        history=self.out/'metrics-history.jsonl'
        if not history.exists() or not any((r.get('key'),r.get('scores_sha256'))==(job['key'],metrics['scores_sha256']) for r in map(json.loads,history.read_text().splitlines())):
            with history.open('a') as stream:stream.write(json.dumps(metrics)+'\n');stream.flush();os.fsync(stream.fileno())
        self.review('ASR_ready_review_only' if comparable else 'ASR_incomplete',step=job['step'],metrics=metrics,scores=str(directory/'scores.json'));self.save()
    def poll_child(self):
        if not self.child:return
        active=self.child;code=self.proc.poll() if self.proc and self.proc.pid==active['pid'] else None
        if code is not None or ticks(active['pid'])!=str(active['start_ticks']):self.complete(active,code);self.child=self.proc=None;(self.out/'active.json').unlink(missing_ok=True);return
        now=time.monotonic()
        if not active['paused']:active['active_seconds']+=max(0.,now-active['last_monotonic'])
        active['last_monotonic']=now;pause_child(active,(self.root/'pause-asr').exists())
        if active['active_seconds']>self.a.timeout:
            if owned(active):os.killpg(active['pid'],signal.SIGKILL)
            self.review('ASR_timeout',active=active)
        write(self.out/'active.json',active);log=Path(self.state['jobs'][active['key']]['directory'])/'asr.log'
        if log.stat().st_size>1024**2:
            with log.open('r+b') as stream:stream.seek(-512*1024,2);tail=stream.read();stream.seek(0);stream.write(tail);stream.truncate()
    def cleanup(self):
        if (self.root/'pause-asr').exists():return
        jobs=list(self.state['jobs'].values());keep=retention(jobs,self.state.get('best_asr'),read(self.run/'best.sha256.json',{}).get('step'),self.child['key'] if self.child else None)
        checkpoints=checkpoint_retention(jobs,self.state.get('best_asr'),self.state.get('best_checkpoint'));keep.update(checkpoints)
        for job in jobs:
            path=Path(job['directory'])
            if not path.exists():continue
            if path.is_symlink() or path.resolve().parent!=self.probes.resolve() or path.stat().st_uid!=os.getuid():raise ValueError('Unsafe cleanup path')
            files=[str(path)]+[str(p) for p in path.iterdir() if p.is_file()]
            if subprocess.run(['fuser',*files],capture_output=True,timeout=10).returncode!=1:continue
            # Keep latest exact validation weights independently of ASR rank/status.
            if job.get('checkpoint') and job['key'] not in checkpoints:
                (path/'checkpoint.pt').unlink(missing_ok=True);(path/'checkpoint.sha256.json').unlink(missing_ok=True);job.pop('checkpoint',None)
            if job['key'] in keep or job['status']=='running':continue
            allocated=physical_usage([path])['allocated_bytes'];shutil.rmtree(path);job.update(removed_utc=utc(),status='expired_unscored' if job['status']=='pending' else job['status'])
            with (self.out/'cleanup.jsonl').open('a') as stream:stream.write(json.dumps(dict(utc=utc(),directory=str(path),allocated_bytes=allocated,reason='Outside latest6 audio/latest3 exact checkpoints/initial/bestASR/training-best/active'))+'\n')
    def shutdown(self):
        if self.child and owned(self.child):pause_child(self.child,True,force=True);write(self.out/'active.json',self.child)
        self.save()
    def loop(self):
        while self.running:
            try:
                self.poll_child();now=time.monotonic()
                if now-self.last_poll>=self.a.poll_seconds:self.observe();self.discover();self.cleanup();self.save();self.last_poll=now
                if not self.a.once:self.start()
            except Exception as exc:self.log.exception('monitor_iteration_failed');self.review('monitor_error',error=repr(exc))
            if self.a.once:return
            time.sleep(1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'voxcpm2/aegis-lora');p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--python',default=PYTHON);p.add_argument('--asr-root',type=Path,default=Path('/var/tmp/aegis-v2-evaluation-20261102/asr'));p.add_argument('--expected-nfe',type=int,choices=(4,6,10),default=10);p.add_argument('--poll-seconds',type=float,default=30.);p.add_argument('--timeout',type=float,default=1800.);p.add_argument('--once',action='store_true');args=p.parse_args()
    if min(args.poll_seconds,args.timeout)<=0:p.error('Positive cadence/deadline required')
    os.umask(0o077);os.environ['CUDA_VISIBLE_DEVICES']='';out=args.run/'quality-monitor';out.mkdir(parents=True,exist_ok=True,mode=0o700)
    lock=(out/'monitor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);monitor=Monitor(args)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:setattr(monitor,'running',False))
    try:monitor.loop()
    finally:monitor.shutdown()

if __name__=='__main__':main()
