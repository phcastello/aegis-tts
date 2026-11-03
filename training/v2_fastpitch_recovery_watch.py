"""CPU-only FastPitch OOM recovery controller; GPU training only via owner queue.

Retries require BOTH failed done receipt and final CUDA OOM log, a verified
finite checkpoint, idle/exclusive GPU, and unchanged scientific recipe. Normal
SIGINT/exit0, NaN, unrelated errors and stale progress never trigger a retry.
"""
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
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/var/tmp/aegis-v2-work')
RUN=ROOT/'fastpitch/language'
PYTHON='/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python'


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:return default
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temporary=path.with_name(path.name+f'.new-{os.getpid()}')
    with temporary.open('w',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.chmod(0o600);os.replace(temporary,path)
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)
def ticks(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except (FileNotFoundError,ProcessLookupError):return None
def trainer_command(job):return job.get('recovery',{}).get('trainer_command',job.get('command',[]))


def flags(command):
    if 'train' not in command:raise ValueError('not a training command')
    result={};items=command[command.index('train')+1:];index=0
    while index<len(items):
        key=items[index]
        if not key.startswith('--'):raise ValueError('unexpected positional training token')
        if '=' in key:key,value=key.split('=',1)
        elif index+1<len(items) and not items[index+1].startswith('--'):index+=1;value=items[index]
        else:value=True
        if key in result:raise ValueError('duplicate training flag '+key)
        result[key]=value;index+=1
    return result
def invariant(job):
    values=flags(trainer_command(job))
    for key in ('--batch-size','--accumulation','--resume','--resume-allow-resource-change','--max-steps'):values.pop(key,None)
    return values
def same_output(job,run):
    try:return flags(trainer_command(job)).get('--output')==str(run)
    except (ValueError,KeyError):return False
def next_resources(batch,accum):
    if batch*accum!=16:raise ValueError('effective batch must remain16')
    options={8:(4,4),4:(2,8),2:(1,16)}
    return options.get(batch)
def classify(done,tail):
    code=done.get('returncode')
    if code in (0,130,143,-2,-15) or done.get('status')=='succeeded':return 'normal_stop'
    if done.get('emergency'):return 'other_error'
    exceptions=re.findall(r'^([\w.]+(?:Error|Exception)):\s*(.*)$',tail,re.M)
    if not exceptions:return 'other_error'
    name,message=exceptions[-1]
    if 'CUDA out of memory' in message and name in ('torch.OutOfMemoryError','torch.cuda.OutOfMemoryError','RuntimeError'):
        # Direct Python unhandled exception exits1. A killed process may retain
        # an OOM from an older appended log, so signals/abort codes never retry.
        return 'cuda_oom' if done.get('status')=='failed' and code==1 else 'other_error'
    return 'other_error'
def gpu_processes():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
    return [int(s.strip()) for s in p.stdout.splitlines() if s.strip()]
def worker_ready(root):
    if (root/'finish-gpu').exists() or (root/'disable-fp-auto-recovery').exists():return False
    window=read(root/'gpu-window-authorized.json',{})
    if not (window.get('status')=='active' and window.get('production_stopped') and window.get('owner_authorized')):return False
    current=read(root/'worker-current.json',{})
    pid=current.get('worker_pid')
    if not pid or ticks(pid) is None:return False
    command=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
    if 'v2_gpu_worker' not in command:return False
    with (root/'gpu-worker.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return current
        else:fcntl.flock(lock,fcntl.LOCK_UN);return False
def publish(path,job):
    """Atomic exclusive publication; a crash cannot overwrite another job."""
    temporary=path.with_name(path.name+f'.pending-{os.getpid()}')
    write(temporary,job)
    try:os.link(temporary,path)
    except FileExistsError:
        if read(path)!=job:raise ValueError('queue filename collision')
    finally:temporary.unlink(missing_ok=True)
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


def retry_command(job,batch,accum):
    command=trainer_command(job).copy()
    for flag,value in (('--batch-size',str(batch)),('--accumulation',str(accum))):
        i=command.index(flag);command[i+1]=value
    for flag in ('--resume','--resume-allow-resource-change'):
        if flag not in command:command.append(flag)
    command=[x for x in command if not x.startswith('PYTORCH_CUDA_ALLOC_CONF=')]
    if command[0]=='/usr/bin/env':command.insert(1,'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True')
    else:command=['/usr/bin/env','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',*command]
    return command


def guarded_launch(ticket_path):
    """Called ONLY by GPU-worker; exec keeps a single trainer PID, no sibling."""
    ticket=read(ticket_path);root=Path(ticket['root']);run=Path(ticket['run'])
    output=ticket_path.with_suffix('.launch.json')
    try:
        parent=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().decode(errors='replace')
        if 'v2_gpu_worker' not in parent or not worker_ready(root):raise RuntimeError('exclusive GPU-worker/owner window unavailable')
        if gpu_processes():raise RuntimeError('GPU compute process already active')
        if any(path.name!=ticket['queue_name'] for path in (root/'queue').glob('*.json')):
            raise RuntimeError('new owner task queued; suppress racing recovery')
        source_done=root/'done'/ticket['source_job']
        source_log=root/'logs'/(Path(ticket['source_job']).stem+'.log')
        if sha(source_done)!=ticket['fault_done_sha256'] or sha(source_log)!=ticket['fault_log_sha256']:
            raise RuntimeError('source fault receipt/log changed; request review')
        checkpoint=read(run/'checkpoint.json',{})
        audit=read(ticket['audit_path'],{})
        if (checkpoint.get('sha256')!=ticket['checkpoint_sha256'] or audit.get('status')!='pass'
                or audit.get('sha256')!=ticket['checkpoint_sha256'] or audit.get('cuda_initialized')
                or not audit.get('checks') or not all(audit['checks'].values())
                or sha(ticket['audit_path'])!=ticket['audit_sha256']
                or sha(REPO/'training/fastpitch_experiment.py')!=ticket['trainer_code_sha256']
                or sha(run/'last.pt')!=ticket['checkpoint_sha256']):
            raise RuntimeError('audited checkpoint changed; preserve state and request review')
        # Another owner-scheduled recovery after the fault supersedes this intent.
        for path in (root/'done').glob('*.json'):
            done=read(path,{})
            same=same_output(done.get('job',{}),run)
            if same and done.get('started_utc','')>ticket['fault_finished_utc']:
                raise RuntimeError('a newer training job already ran after this fault; suppress stale retry')
        write(output,dict(status='exec_training',utc=utc(),pid=os.getpid(),checkpoint_step=audit['step'],checkpoint_sha256=audit['sha256'],
                          trainer_command=ticket['trainer_command'],exclusive_gpu_processes=[],effective_batch=16))
        # Hashing/scanning can take seconds; give a newly queued owner task
        # priority again immediately before the serial worker execs training.
        if (not worker_ready(root) or gpu_processes()
                or any(path.name!=ticket['queue_name'] for path in (root/'queue').glob('*.json'))):
            raise RuntimeError('owner task/window changed during launch audit; defer to owner')
        command=ticket['trainer_command'];os.execvpe(command[0],command,os.environ)
    except Exception as e:
        write(output,dict(status='refused',utc=utc(),error=repr(e)));raise SystemExit(75)


class Watcher:
    def __init__(self,args):
        self.a=args;self.root=args.root.resolve();self.run=args.run.resolve();self.out=self.run/'recovery-watch';self.out.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.state=read(self.out/'state.json',dict(jobs={},handled={},intents={},conditions=[]))
        initial=read(self.root/'queue'/args.initial_job) or read(self.root/'done'/args.initial_job,{}).get('job')
        if not initial:raise ValueError('initial queue/done job is required')
        self.baseline=invariant(initial)
        if self.baseline.get('--output')!=str(self.run):raise ValueError('initial output differs from watched run')
        if flags(trainer_command(initial)).get('--precision')!='bf16':raise ValueError('expected BF16 scientific recipe')
        self.state['jobs'].setdefault(args.initial_job,initial)
        initial_done=read(self.root/'done'/args.initial_job,{})
        current=read(self.root/'worker-current.json',{})
        floor=initial_done.get('started_utc') or (current.get('started_utc') if current.get('job')==initial else None)
        if floor is None:
            floor=dt.datetime.fromtimestamp((self.root/'queue'/args.initial_job).stat().st_mtime,dt.timezone.utc).isoformat()
        self.floor=self.state.setdefault('start_floor_utc',floor)
        self.baseline_step=int(read(self.run/'checkpoint.json',{}).get('step',0));self.last_heartbeat=0.;self.running=True
        self.log=logging.getLogger('fp-recovery');self.log.setLevel(logging.INFO)
        handler=RotatingFileHandler(self.out/'watcher.log',maxBytes=1024*1024,backupCount=1)
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'));self.log.addHandler(handler)
        write(self.out/'recipe.json',dict(utc=utc(),pid=os.getpid(),pgid=os.getpgrp(),pid_start_ticks=ticks(os.getpid()),code_sha256=sha(__file__),
              initial_job=args.initial_job,initial_command=trainer_command(initial),invariant=self.baseline,poll_seconds=15,heartbeat_seconds=1800,
              recovery_ladder=['8x2 allocator expandable','4x4','2x8','1x16'],effective_batch=16,
              rule='Failed done receipt + final CUDA OOM; finite hashed full checkpoint only; no restart for manualSIGINT/exit0/NaN/nonOOM/stale',
              scope='CPU monitor/auditor and atomic owner-queue publication only; GPU training runs only in GPU-worker after launch guard'))

    def event(self,name,**kw):self.log.info(json.dumps(dict(event=name,**kw),ensure_ascii=False))
    def save(self):write(self.out/'state.json',self.state)
    def review(self,kind,**kw):
        path=self.run/'review-needed.json';data=read(path,dict(created_utc=utc(),notices=[]))
        data['notices']=[x for x in data.get('notices',[]) if x.get('kind')!=kind]
        data['notices'].append(dict(kind=kind,utc=utc(),**kw));data['updated_utc']=utc();write(path,data)
        self.event('review_needed',kind=kind)
    def matching(self,job):
        try:return invariant(job)==self.baseline
        except (ValueError,KeyError):return False

    def discover(self):
        for path in (self.root/'queue').glob('*.json'):
            job=read(path,{})
            if same_output(job,self.run):self.state['jobs'].setdefault(path.name,job)
        # Adopt short owner jobs that started/finished between monitor polls.
        for path in (self.root/'done').glob('*.json'):
            done=read(path,{})
            if done.get('started_utc','')>=self.floor and same_output(done.get('job',{}),self.run):
                self.state['jobs'].setdefault(path.name,done['job'])

    def reconcile_intents(self):
        """Adopt a published retry after crash between queue link and state save."""
        for key,intent in self.state['intents'].items():
            name=intent['job_name'];queued=read(self.root/'queue'/name)
            done=read(self.root/'done'/name)
            if queued is None and done is None:continue
            actual=queued if queued is not None else done.get('job')
            if actual!=intent['job']:raise ValueError('recovery transaction content changed: '+name)
            self.state['jobs'][name]=intent['job']
            intent['status']='finished' if done is not None else 'queued'
            source=intent['job']['recovery']['source_job']
            self.state['handled'].setdefault(source,dict(action='queued_retry',queue_name=name,
                         reason='Reconciled atomic queue publication after controller restart',utc=utc()))

    def health(self):
        progress=read(self.run/'progress.json',{});checkpoint=read(self.run/'checkpoint.json',{})
        current=read(self.root/'worker-current.json',{})
        active=self.matching(current.get('job',{})) and current.get('status')=='running' and ticks(current.get('pid',-1)) is not None
        age=time.time()-(self.run/'progress.json').stat().st_mtime if (self.run/'progress.json').exists() else None
        conditions=[]
        if active and age is not None and age>600:conditions.append('stale_progress')
        if any(isinstance(v,(int,float)) and not math.isfinite(v) for v in progress.values()):conditions.append('nonfinite_progress')
        if not worker_ready(self.root):conditions.append('worker_or_owner_window_unavailable')
        for condition in conditions:
            if condition not in self.state['conditions']:self.review(condition,progress=progress,checkpoint=checkpoint)
        self.state['conditions']=conditions
        status=dict(utc=utc(),active_training=active,worker_job=current.get('job'),pid=current.get('pid'),progress=progress,
                    saved_checkpoint=checkpoint,progress_is_not_a_checkpoint=True,last_progress_age_seconds=age,
                    free_bytes=os.statvfs(self.root).f_bavail*os.statvfs(self.root).f_frsize,
                    conditions=conditions,pending_recoveries={k:v['status'] for k,v in self.state['intents'].items()})
        write(self.out/'status.json',status)
        if time.monotonic()-self.last_heartbeat>=1800:
            write(self.out/'heartbeat.json',status);self.event('heartbeat',step=progress.get('step'),checkpoint_step=checkpoint.get('step'),active_training=active)
            self.last_heartbeat=time.monotonic()

    def newer_owner_training(self,source_name,done):
        for name,job in self.state['jobs'].items():
            if name==source_name or job.get('recovery'):continue
            result=read(self.root/'done'/name)
            queued=self.root/'queue'/name
            if result and result.get('started_utc','')>done['finished_utc']:return name
            if queued.exists() and queued.stat().st_mtime>dt.datetime.fromisoformat(done['finished_utc']).timestamp():return name
        return None

    def handle_done(self,name,job):
        path=self.root/'done'/name;done=read(path)
        if not done or name in self.state['handled']:return
        if done.get('job')!=job:
            self.review('job_filename_reused_or_receipt_changed',job=name)
            self.state['handled'][name]=dict(action='review_only_job_identity_mismatch',utc=utc());self.save();return
        if not self.matching(job):
            self.state['handled'][name]=dict(action='owner_changed_recipe_no_auto_retry',utc=utc());self.save();return
        log=self.root/'logs'/(Path(name).stem+'.log')
        if not log.exists():self.review('missing_failure_log',job=name);return
        with log.open('rb') as f:f.seek(max(0,log.stat().st_size-262144));tail=f.read().decode(errors='replace')
        outcome=classify(done,tail)
        if outcome=='normal_stop':
            self.state['handled'][name]=dict(action='no_restart',reason='normal/signal exit; owner diagnostic interruption respected',utc=utc());self.save();return
        if outcome!='cuda_oom':
            self.review('non_oom_failure',job=name,done=done,log=str(log));self.state['handled'][name]=dict(action='review_only',utc=utc());self.save();return
        external=self.newer_owner_training(name,done)
        if external:
            self.state['handled'][name]=dict(action='superseded_by_owner_job',new_job=external,utc=utc());self.save();return
        values=flags(trainer_command(job));resource=next_resources(int(values['--batch-size']),int(values['--accumulation']))
        if resource is None:
            self.review('oom_at_minimum_microbatch',job=name,checkpoint=read(self.run/'checkpoint.json'));self.state['handled'][name]=dict(action='review_only',utc=utc());self.save();return
        # Never create another GPU-active process or race a queued owner task.
        if list((self.root/'queue').glob('*.json')) or not worker_ready(self.root) or gpu_processes():return
        if os.statvfs(self.root).f_bavail*os.statvfs(self.root).f_frsize<5*1024**3:
            self.review('disk_reserve_blocks_recovery',job=name);return
        fault_hash=sha(path);key=fault_hash[:16]
        audit=self.out/('checkpoint-audit-'+key+'.json')
        command=['nice','-n','19',PYTHON,'-m','training.v2_fastpitch_checkpoint_audit','--run',str(self.run),'--output',str(audit)]
        audit_log=self.out/'checkpoint-audit.log'
        if audit_log.exists() and audit_log.stat().st_size>1024*1024:
            with audit_log.open('r+b') as f:f.seek(-262144,2);recent=f.read();f.seek(0);f.write(recent);f.truncate()
        with audit_log.open('ab') as output:
            result=subprocess.run(command,cwd=REPO,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                                  stdout=output,stderr=subprocess.STDOUT,timeout=180)
        if not self.running:return  # SIGINT during audit must veto publication.
        verified=read(audit,{})
        if result.returncode==2:return
        if result.returncode or verified.get('status')!='pass' or verified.get('cuda_initialized'):
            self.review('checkpoint_invalid_no_retry',job=name,audit=str(audit));self.state['handled'][name]=dict(action='review_only',utc=utc());self.save();return
        # Reconcile again after CPU audit in case the owner scheduled a recovery.
        self.discover()
        if self.newer_owner_training(name,done) or list((self.root/'queue').glob('*.json')) or not worker_ready(self.root) or gpu_processes():return
        batch,accum=resource;training=retry_command(job,batch,accum)
        if invariant(dict(command=training))!=self.baseline:raise ValueError('scientific recipe changed in retry')
        ticket_path=self.out/('retry-'+key+'.json');queue_name=f'zz-fp-oom-{key}-b{batch}a{accum}.json'
        ticket=dict(root=str(self.root),run=str(self.run),source_job=name,queue_name=queue_name,fault_finished_utc=done['finished_utc'],
                    fault_done_sha256=fault_hash,fault_log_sha256=sha(log),cause='final CUDA OOM confirmed after failed queue job',
                    checkpoint_sha256=verified['sha256'],checkpoint_step=verified['step'],checkpoint_epoch=verified['epoch'],checkpoint_cursor=verified['cursor'],
                    audit_path=str(audit),audit_sha256=sha(audit),invariant=self.baseline,trainer_command=training,
                    trainer_code_sha256=sha(REPO/'training/fastpitch_experiment.py'),
                    last_unsaved_progress_diagnostic=read(self.run/'progress.json',{}),
                    old_resources=[int(values['--batch-size']),int(values['--accumulation'])],new_resources=[batch,accum],effective_batch=16,
                    state_policy='Restore model/Adam/scheduler/scaler/RNG/data cursor from validated saved checkpoint; never progress.json',utc=utc())
        write(ticket_path,ticket)
        queued=dict(command=[PYTHON,'-m','training.v2_fastpitch_recovery_watch','--guarded-ticket',str(ticket_path)],cwd=str(REPO),
                    recovery=dict(transaction=key,source_job=name,ticket=str(ticket_path),trainer_command=training))
        self.state['intents'][key]=dict(status='intent',job_name=queue_name,job=queued,ticket=str(ticket_path));self.save()
        if not self.running or not worker_ready(self.root):return
        publish(self.root/'queue'/queue_name,queued)
        self.state['intents'][key]['status']='queued';self.state['jobs'][queue_name]=queued
        self.state['handled'][name]=dict(action='queued_retry',queue_name=queue_name,checkpoint_step=verified['step'],resources=[batch,accum],utc=utc());self.save()
        self.event('retry_queued',source=name,job=queue_name,checkpoint_step=verified['step'],microbatch=batch,accumulation=accum)
        self.review('oom_recovered_to_queue',job=queue_name,checkpoint_step=verified['step'],ticket=str(ticket_path),meaning='Queued, not yet proof of successful restart')

    def loop(self):
        while self.running:
            try:
                self.discover();self.reconcile_intents();self.health()
                for name,job in list(self.state['jobs'].items()):self.handle_done(name,job)
                self.save()
            except Exception as error:
                self.log.exception('watch_iteration_failed');self.review('recovery_watcher_error',error=repr(error))
            time.sleep(15)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--initial-job',default='073-fastpitch-allocator-recovery.json');parser.add_argument('--guarded-ticket',type=Path)
    args=parser.parse_args();os.umask(0o077)
    if args.guarded_ticket:return guarded_launch(args.guarded_ticket)
    output=args.run/'recovery-watch';output.mkdir(parents=True,exist_ok=True,mode=0o700)
    lock=(output/'watcher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    watcher=Watcher(args)
    def stop(*_):watcher.running=False;watcher.save()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    watcher.loop()


if __name__=='__main__':main()
