"""Local CPU supervision and immutable per-epoch ASR for a VITS language run.

Never touches the GPU, training controls, model weights, data or final test.
The owner reviews review-needed.json; ASR never automatically rejects a model.
"""
from __future__ import annotations
import argparse
import collections
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

REPO=Path(__file__).resolve().parents[1]


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default
def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.chmod(0o600);temporary.replace(path)
def ticks(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except (FileNotFoundError,ProcessLookupError):return None
def alive(pid,start_ticks):return bool(start_ticks and ticks(pid)==str(start_ticks))


def physical_usage(paths):
    """Allocated blocks, global hardlink dedup, no symlink traversal."""
    seen=set();total=0;missing=[];errors=[];per_root={}
    for root in paths:
        subtotal=0;stack=[Path(root)]
        while stack:
            path=stack.pop()
            try:
                stat=path.lstat()
                if path.is_symlink():continue
                key=(stat.st_dev,stat.st_ino)
                if key in seen:continue
                seen.add(key);subtotal+=stat.st_blocks*512
                if path.is_dir():
                    with os.scandir(path) as entries:stack.extend(Path(e.path) for e in entries)
            except FileNotFoundError:missing.append(str(path))
            except PermissionError:errors.append(str(path))
        per_root[str(root)]=subtotal;total+=subtotal
    return dict(allocated_bytes=total,roots=per_root,missing_count=len(missing),permission_errors=errors[:20],
                policy='allocated st_blocks; hardlinks deduplicated globally; symlinks not followed')


def job_key(manifest_sha,audio_hashes,corpus_sha,spoken_sha,scorer_sha):
    value=dict(manifest=manifest_sha,audio=sorted(audio_hashes.items()),corpus=corpus_sha,spoken=spoken_sha,scorer=scorer_sha)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def retained_jobs(jobs,critical,active,keep=6):
    epochs=sorted([j for j in jobs if j['label'].startswith('epoch-')],key=lambda j:(j['epoch'],j['step'],j['key']))
    retain={j['key'] for j in epochs[-keep:]}
    retain.update(j['key'] for j in jobs if j['label']=='initial')
    retain.update(critical)
    if active:retain.add(active)
    return retain


class Monitor:
    def __init__(self,args):
        self.a=args;self.run=args.run.resolve();self.root=args.root.resolve();self.out=self.run/'asr'
        self.out.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.state=read(self.out/'state.json',dict(jobs={},conditions=[],best_asr=None))
        self.child=None;self.paused=False;self.running=True;self.last_heartbeat=0.;self.last_budget=0.
        self.corpus_path=REPO/'benchmarks/v2/validation.pt-BR.json'
        self.spoken_path=REPO/'benchmarks/v2/spoken-validation.pt-BR.json'
        self.scorer=REPO/'benchmarks/v2/asr_score.py'
        self.corpus={c['id']:c for c in read(self.corpus_path)}
        self.spoken=read(self.spoken_path)['cases']
        self.corpus_sha=sha(self.corpus_path);self.spoken_sha=sha(self.spoken_path)
        self.log=logging.getLogger('aegis-local-monitor');self.log.setLevel(logging.INFO)
        handler=RotatingFileHandler(self.out/'monitor.log',maxBytes=1024*1024,backupCount=1)
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'));self.log.addHandler(handler)
        recipe=dict(start_utc=utc(),pid=os.getpid(),pid_start_ticks=ticks(os.getpid()),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    command=sys.argv,script_sha256=sha(__file__),scorer_sha256=sha(self.scorer),
                    run=str(self.run),poll_seconds=args.poll_seconds,heartbeat_seconds=1800,budget_poll_seconds=300,
                    cpu_asr=dict(python=args.python,device='cpu',threads=2,nice=19,cuda_visible_devices='',deadline_seconds=1800),
                    corpus_sha256=self.corpus_sha,spoken_corpus_sha256=self.spoken_sha,
                    pause_file=str(self.root/'pause-asr'),retention='6 latest epoch directories plus initial, best ASR reference and current training best epoch; compact metrics history retained',
                    budget_scope='All /var/tmp/aegis* + source ZIP + experimental worktree + conservative whole ~/.cache; production models/cache/protected and immutable Docker rollback image excluded; no new image built',
                    changes_forbidden='No training stop, weight/data edits, GPU use, final-test access or auditory claims')
        write(self.out/'monitor-recipe.json',recipe)
        self.recover_active()

    def event(self,event_name,**values):
        self.log.info(json.dumps(dict(event=event_name,**values),ensure_ascii=False))

    def save_state(self):write(self.out/'state.json',self.state)

    def review(self,kind,key,**values):
        path=self.run/'review-needed.json';record=read(path,dict(created_utc=utc(),notices=[]))
        record['notices']=[r for r in record['notices'] if r['key']!=key]
        record['notices'].append(dict(kind=kind,key=key,utc=utc(),**values));record['notices']=record['notices'][-100:]
        record['updated_utc']=utc();record['interpretation']='Review required; no automatic model rejection or auditory assessment'
        write(path,record);self.event('review_needed',kind=kind,key=key)

    def recover_active(self):
        item=read(self.out/'active.json')
        if item and alive(item['pid'],item['start_ticks']):
            self.child=item;self.paused=bool(item.get('paused'));self.event('reattached_asr',pid=item['pid'],key=item['key'])
        elif item:
            # A process that ended while the monitor was down is validated using
            # its persisted receipt; an incomplete receipt is safely retried.
            self.complete(item,None);(self.out/'active.json').unlink(missing_ok=True)

    def budget_roots(self):
        return sorted(Path('/var/tmp').glob('aegis*'))+[Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip'),REPO,Path('/home/phc/.cache')]

    def observe_health(self):
        progress=read(self.run/'progress.json',{})
        worker=read(self.root/'worker-current.json',{})
        command=worker.get('job',{}).get('command',[])
        matching=str(self.run) in command and 'train' in command and 'training.v2_vits' in command
        running=matching and worker.get('status')=='running' and ticks(worker.get('pid',-1)) is not None
        completed=matching and worker.get('status') in ('failed','succeeded')
        done=None
        for path in sorted((self.root/'done').glob('*.json'),reverse=True):
            item=read(path,{})
            if str(self.run) in item.get('job',{}).get('command',[]) and 'train' in item['job']['command']:
                done=item;break
        if done and not running:completed=True
        now=time.time();age=now-(self.run/'progress.json').stat().st_mtime if (self.run/'progress.json').exists() else float('inf')
        free=shutil.disk_usage(self.run).free
        pause=(self.root/'pause-asr').exists()
        if not pause and time.monotonic()-self.last_budget>=300:
            self.state['budget']=dict(utc=utc(),**physical_usage(self.budget_roots()));self.last_budget=time.monotonic()
            write(self.out/'budget-latest.json',self.state['budget'])
        conditions={}
        if completed:conditions['trainer_finished']=dict(worker_result=done or worker)
        elif not running:conditions['trainer_process_missing']=dict(worker_pid=worker.get('pid'),worker_status=worker.get('status'))
        elif age>600:conditions['training_stale']=dict(last_progress_age_seconds=age,step=progress.get('step'))
        if any(isinstance(v,(float,int)) and not math.isfinite(v) for v in progress.values()):conditions['nonfinite_progress']=dict(progress=progress)
        log=self.run/'training.log'
        if log.exists():
            with log.open('rb') as f:f.seek(max(0,log.stat().st_size-65536));tail=f.read().decode(errors='replace')
            if re.search(r'\b(?:nan|nonfinite|out of memory|traceback)\b',tail,re.I):conditions['training_log_error']=dict(log=str(log))
        if free<5*1024**3:conditions['disk_reserve']=dict(free_bytes=free,threshold_bytes=5*1024**3)
        budget=self.state.get('budget',{})
        if budget.get('allocated_bytes',0)>40_000_000_000:conditions['working_budget']=dict(**budget,threshold_bytes=40_000_000_000,action='root review; conservative cache attribution; no automatic deletion/stop')
        for key,value in conditions.items():
            if key not in self.state.get('conditions',[]):self.review(key,key,**value)
        self.state['conditions']=sorted(conditions)
        status=dict(utc=utc(),trainer_running=running,completed=completed,progress=progress,last_progress_age_seconds=age if math.isfinite(age) else None,
                    free_bytes=free,working_bytes=budget.get('allocated_bytes'),conditions=sorted(conditions),asr_paused=pause,active_asr=self.child)
        write(self.out/'status.json',status)
        if time.monotonic()-self.last_heartbeat>=1800:
            write(self.out/'heartbeat.json',status);self.event('heartbeat',step=progress.get('step'),epoch=progress.get('epoch'),trainer_running=running,
                free_bytes=free,working_bytes=budget.get('allocated_bytes'),conditions=sorted(conditions));self.last_heartbeat=time.monotonic()

    def discover(self):
        if sha(self.corpus_path)!=self.corpus_sha or sha(self.spoken_path)!=self.spoken_sha:
            self.review('validation_policy_changed','validation_policy_changed');return
        sources=[]
        initial=self.run/'probes/initial/results.json'
        if initial.exists():sources.append(initial)
        sources.extend(sorted((self.run/'probes').glob('epoch-*/results.json')))
        for path in sources:
            try:
                raw=path.read_bytes();data=json.loads(raw);items=data['results'];manifest_sha=hashlib.sha256(raw).hexdigest()
                if len(items)!=7 or {r['case'] for r in items}!=set(self.corpus):raise ValueError('expected exactly seven validation cases')
                audio_hashes={r['case']:r['audio']['sha256'] for r in items}
                key=job_key(manifest_sha,audio_hashes,self.corpus_sha,self.spoken_sha,sha(self.scorer))
                if key in self.state['jobs']:continue
                label=path.parent.name;directory=self.out/(label+'-'+key[:16]);temp=self.out/(directory.name+'.snapshot')
                if temp.exists():shutil.rmtree(temp)
                temp.mkdir(mode=0o700);audio_dir=temp/'audio';audio_dir.mkdir(mode=0o700)
                jobs=[]
                for row in items:
                    case=row['case'];source=Path(row['audio']['path'])
                    if source.is_symlink() or source.resolve()!=(path.parent/(case+'.wav')).resolve():raise ValueError('unexpected source audio path')
                    target=audio_dir/(case+'.wav')
                    try:os.link(source,target)
                    except OSError:shutil.copyfile(source,target)
                    if sha(target)!=row['audio']['sha256']:raise ValueError('probe changed while snapshotting; retry next poll')
                    jobs.append(dict(candidate='vits-language',cohort='validation-seven',case=case,
                                     wav=str(directory/'audio'/(case+'.wav')),expected_verbatim=self.corpus[case]['text'],
                                     expected_spoken=self.spoken[case],expected_effective=row['effective_text']))
                # Stable input receipt and immutable inode snapshots, even if
                # trainer later atomically replaces/removes its own paths.
                (temp/'source-results.json').write_bytes(raw)
                write(temp/'jobs.json',dict(jobs=jobs))
                manifest=dict(key=key,label=label,epoch=data['epoch'],step=data['step'],status='pending',directory=str(directory),
                              source_results_path=str(path),source_results_sha256=manifest_sha,audio_hashes=audio_hashes,
                              jobs_sha256=sha(temp/'jobs.json'),scorer_sha256=sha(self.scorer),created_utc=utc())
                write(temp/'snapshot.json',manifest)
                if directory.exists():shutil.rmtree(temp)
                else:temp.replace(directory)
                self.state['jobs'][key]=manifest;self.save_state()
                self.review('new_epoch' if label!='initial' else 'initial_probe',key,epoch=data['epoch'],step=data['step'],snapshot=str(directory),asr='pending')
            except (ValueError,FileNotFoundError,KeyError,json.JSONDecodeError) as e:
                self.event('snapshot_retry',source=str(path),error=str(e))

    def start_asr(self):
        if self.child or (self.root/'pause-asr').exists():return
        pending=sorted([j for j in self.state['jobs'].values() if j['status'] in ('pending','retry')],key=lambda j:(j['epoch'],j['step']))
        if not pending:return
        job=pending[0];directory=Path(job['directory'])
        if sha(directory/'jobs.json')!=job['jobs_sha256'] or sha(directory/'source-results.json')!=job['source_results_sha256']:
            raise ValueError('ASR job/source manifest integrity failure')
        if sha(self.scorer)!=job['scorer_sha256']:
            job['status']='expired_policy';self.save_state();return
        for case,expected in job['audio_hashes'].items():
            if sha(directory/'audio'/(case+'.wav'))!=expected:raise ValueError('ASR snapshot integrity failure')
        command=['nice','-n','19',self.a.python,str(self.scorer),'--jobs',str(directory/'jobs.json'),'--output',str(directory/'scores.json')]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
             'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'}
        with (directory/'asr.log').open('ab') as output:
            proc=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        self.proc=proc
        self.child=dict(key=job['key'],pid=proc.pid,start_ticks=ticks(proc.pid),command=command,paused=False,started_utc=utc())
        write(self.out/'active.json',self.child);job['status']='running';self.save_state();self.event('asr_started',pid=proc.pid,key=job['key'])

    def complete(self,active,exit_code):
        job=self.state['jobs'].get(active['key'])
        if not job:return
        directory=Path(job['directory']);receipt=read(directory/'scores.json',{})
        attempts=receipt.get('attempts',[])
        expected={(case,value) for case,value in job['audio_hashes'].items()}
        observed={(r.get('case'),r.get('audio_sha256')) for r in attempts if r.get('success')}
        valid=expected==observed and len(attempts)==7 and receipt.get('script_sha256')==job['scorer_sha256']
        if valid:
            metrics=read(directory/'scores-summary.json')
            if not metrics or sum(g['successes'] for g in metrics['groups'].values())!=7:raise ValueError('Incomplete ASR summary')
            job['status']='complete';job['summary']=metrics
            write(directory/'summary.json',metrics)
            score=sum(r['comparisons']['spoken']['word_edits'] for r in attempts)/max(1,sum(len(r['comparisons']['spoken']['reference_tokens']) for r in attempts))
            job['spoken_wer']=score;previous=self.state.get('best_asr')
            if previous is None or score<self.state['jobs'][previous].get('spoken_wer',float('inf')):self.state['best_asr']=job['key']
            with (self.out/'metrics-history.jsonl').open('a') as f:
                f.write(json.dumps(dict(utc=utc(),key=job['key'],epoch=job['epoch'],step=job['step'],spoken_wer=score,
                                        receipt_sha256=sha(directory/'scores.json'),summary=metrics),ensure_ascii=False)+'\n')
            self.review('asr_ready',job['key']+'-asr',epoch=job['epoch'],step=job['step'],spoken_wer=score,
                        scores=str(directory/'scores.json'),summary=str(directory/'summary.json'),interpretation='ASR evidence only; no listening, model rejection or identity conclusion')
        else:
            job['status']='retry' if not receipt.get('completed_utc') and job.get('retries',0)<2 else 'failed';job['retries']=job.get('retries',0)+1
            self.review('asr_failed',job['key']+'-failed',exit_code=exit_code,successful_cases=len(observed),directory=str(directory))
        job['last_exit_code']=exit_code;job['last_finished_utc']=utc();self.save_state()

    def poll_asr(self):
        if not self.child:return
        pid=self.child['pid']
        code=self.proc.poll() if getattr(self,'proc',None) and self.proc.pid==pid else None
        if code is not None or not alive(pid,self.child['start_ticks']):
            self.complete(self.child,code);self.child=None;self.paused=False;(self.out/'active.json').unlink(missing_ok=True);return
        pause=(self.root/'pause-asr').exists()
        if pause!=self.paused:
            os.killpg(pid,signal.SIGSTOP if pause else signal.SIGCONT);self.paused=pause;self.child['paused']=pause
            write(self.out/'active.json',self.child);self.event('asr_paused' if pause else 'asr_resumed',pid=pid)

    def cleanup(self):
        jobs=list(self.state['jobs'].values());critical={self.state.get('best_asr')}
        best=read(self.run/'probes/best/results.json',{})
        for job in jobs:
            if best and job['epoch']==best.get('epoch') and job['step']==best.get('step'):critical.add(job['key'])
        active=self.child['key'] if self.child else None
        keep=retained_jobs(jobs,critical,active)
        for job in jobs:
            directory=Path(job['directory'])
            if job['key'] in keep or not directory.exists():continue
            if job['status'] in ('pending','retry'):
                job['status']='expired_unscored'
                self.review('asr_retention_expired',job['key']+'-expired',epoch=job['epoch'],step=job['step'],reason='Outside latest6 while ASR pending; training continues')
            if job['status'] not in ('complete','failed','expired_unscored'):continue
            if directory.resolve().parent!=self.out or directory.is_symlink():raise ValueError('unsafe monitor cleanup path')
            shutil.rmtree(directory);job['snapshot_removed_utc']=utc();self.event('snapshot_cleaned',key=job['key'])

    def run_loop(self):
        while self.running:
            try:
                self.observe_health();self.discover();self.poll_asr();self.start_asr();self.cleanup();self.save_state()
            except Exception as e:
                self.log.exception('monitor_iteration_failed')
                try:self.review('monitor_error',type(e).__name__,error=repr(e))
                except Exception:self.log.exception('review_notice_write_failed')
            if self.a.once:return
            time.sleep(self.a.poll_seconds)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=Path('/var/tmp/aegis-v2-work/vits/language'))
    p.add_argument('--root',type=Path,default=Path('/var/tmp/aegis-v2-work'));p.add_argument('--python',default='/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python')
    p.add_argument('--poll-seconds',type=float,default=30);p.add_argument('--once',action='store_true');args=p.parse_args()
    os.umask(0o077);directory=args.run/'asr';directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    lock=(directory/'monitor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    monitor=Monitor(args)
    def stop(*_):
        # ASR is a detached CPU process; persist its identity for reattachment.
        # Never send a signal to the trainer.
        monitor.running=False;monitor.save_state()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    monitor.run_loop()


if __name__=='__main__':main()
