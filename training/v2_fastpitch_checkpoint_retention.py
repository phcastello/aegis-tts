"""Retain exact FastPitch native-validation states without loading tensors.

Separate from the live trainer/ASR monitor. Only this supervisor's hardlinks
are created/removed. One pending state plus the best recoverable ASR state;
neither ASR nor retention approves pronunciation or voice identity.
"""
import argparse
import fcntl
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from training.v2_language_monitor import physical_usage, ticks, utc
from training.v2_fastpitch_quality_monitor import read, sha, write, freeze_snapshot

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/var/tmp/aegis-v2-work')
CP_LIMIT=600_000_000
OVERLAP=550_000_000

def sync_dir(path):
    fd=os.open(path,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def budget(root,checkpoint_bytes):
    roots=sorted(Path('/var/tmp').glob('aegis*'))+[Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip'),REPO,Path('/home/phc/.cache')]
    usage=physical_usage(roots);extra=checkpoint_bytes+max(OVERLAP,checkpoint_bytes)+64*1024**2
    free=shutil.disk_usage(root).free
    return dict(**usage,free_bytes=free,estimated_extra_bytes=extra,
                allowed=not usage['permission_errors'] and usage['allocated_bytes']+extra<=40_000_000_000 and free>=5*1024**3+extra)


def safe_file(path,limit):
    stat=path.lstat()
    if path.is_symlink() or stat.st_uid!=os.getuid() or not 0<stat.st_size<=limit:
        raise ValueError('Unexpected owner/type/size: '+str(path))
    return stat


def in_use(directory):
    paths=[str(directory)]+[str(p) for p in directory.iterdir() if p.is_file()]
    return subprocess.run(['fuser',*paths],capture_output=True,timeout=10).returncode!=1


def remove_stage(stage,snapshots):
    if stage.is_symlink() or stage.resolve().parent!=snapshots.resolve() or not stage.name.startswith('.stage-') or stage.stat().st_uid!=os.getuid():
        raise ValueError('Unsafe retention staging cleanup path')
    if in_use(stage):return False
    shutil.rmtree(stage)
    return True


def bind_checkpoint(run,stage,packet):
    """Link then verify the entire inode against the owner's exact-step receipt.

    A later trainer publication does not invalidate an already frozen inode
    that matches the earlier full SHA. A new inode linked against an old
    receipt is rejected. No model format or finite-tensor claim is invented.
    """
    receipt=read(run/'checkpoint.json',{})
    if receipt.get('step')!=packet['step']:return None
    source=run/'last.pt';stat=safe_file(source,CP_LIMIT)
    if Path(receipt.get('checkpoint','')).resolve()!=source.resolve():raise ValueError('Unexpected trainer checkpoint path')
    record=dict(step=packet['step'],packet=packet,checkpoint_sha256=receipt['sha256'],checkpoint_bytes=stat.st_size,
                trainer_receipt=receipt,created_utc=utc(),status='pending_asr',product=False)
    write(stage/'intent.json',record);sync_dir(stage)
    target=stage/'checkpoint.pt';os.link(source,target)
    linked=target.stat()
    if (linked.st_dev,linked.st_ino)!=(stat.st_dev,stat.st_ino) or sha(target)!=receipt['sha256']:
        target.unlink();return None
    write(stage/'checkpoint.sha256.json',dict(step=record['step'],sha256=record['checkpoint_sha256'],bytes=record['checkpoint_bytes']))
    return record


def commit_stage(stage,snapshots,record):
    key=hashlib.sha256(f"{record['step']}:{record['packet']['sha256']}:{record['checkpoint_sha256']}".encode()).hexdigest()
    directory=snapshots/f'step-{record["step"]:08d}-{key[:12]}'
    record.update(key=key,directory=str(directory));write(stage/'entry.json',record);sync_dir(stage)
    if directory.exists():
        if not remove_stage(stage,snapshots):raise RuntimeError('Duplicate staging is in use; preserved for review: '+str(stage))
    else:stage.rename(directory);sync_dir(snapshots)
    return record


def recover_stage(stage,snapshots):
    intent=read(stage/'intent.json')
    if not intent:return None
    cp=stage/'checkpoint.pt';packet=stage/'input-mels.pt'
    if not cp.exists() or not packet.exists():return None
    safe_file(cp,CP_LIMIT);safe_file(packet,64*1024**2)
    if sha(cp)!=intent['checkpoint_sha256'] or sha(packet)!=intent['packet']['sha256']:raise ValueError('Uncommitted retention payload failed SHA; preserve for review')
    write(stage/'checkpoint.sha256.json',dict(step=intent['step'],sha256=intent['checkpoint_sha256'],bytes=cp.stat().st_size))
    return commit_stage(stage,snapshots,intent)


class Keeper:
    def __init__(self,args):
        self.a=args;self.run=args.run.resolve();self.root=args.root.resolve();self.out=self.run/'asr-checkpoints';self.snapshots=self.out/'snapshots'
        self.snapshots.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.state=read(self.out/'state.json',dict(snapshots={},best=None));self.running=True;self.last_heartbeat=0.
        self.log=logging.getLogger('fp-checkpoint-retention');self.log.setLevel(logging.INFO)
        self.log.addHandler(RotatingFileHandler(self.out/'supervisor.log',maxBytes=1024**2,backupCount=1))
        if not (self.root/'pause-asr').exists():self.reconcile_stages()
        write(self.out/'recipe.json',dict(utc=utc(),pid=os.getpid(),start_ticks=ticks(os.getpid()),code_sha256=sha(__file__),
            run=str(self.run),poll_seconds=args.poll_seconds,heartbeat_seconds=1800,checkpoint_limit_bytes=CP_LIMIT,
            overlap_reserve_bytes=OVERLAP,scope='CPU filesystem/SHA only; no trainer/monitor mutation, tensors, ASR, GPU or automatic voice approval'))

    def reconcile_stages(self):
        for path in self.snapshots.glob('step-*/entry.json'):
            record=read(path)
            if Path(record['directory']).resolve()==path.parent.resolve():self.state['snapshots'].setdefault(record['key'],record)
        for stage in self.snapshots.glob('.stage-*'):
            if stage.is_symlink() or stage.stat().st_uid!=os.getuid():raise ValueError('Unsafe retention staging directory')
            record=recover_stage(stage,self.snapshots)
            if record:self.state['snapshots'][record['key']]=record
            elif not (stage/'checkpoint.pt').exists() and not remove_stage(stage,self.snapshots):self.review('staging_in_use',directory=str(stage))

    def review(self,kind,**values):
        data=read(self.out/'review-needed.json',dict(notices=[]))
        data['notices']=[n for n in data['notices'] if n['kind']!=kind]
        data['notices'].append(dict(kind=kind,utc=utc(),**values));write(self.out/'review-needed.json',data)
        self.log.info(json.dumps(data['notices'][-1]))

    def scores(self,quality):
        observed=quality.get('best_asr');jobs=quality.get('jobs',{})
        policy=jobs.get(observed,{}).get('policy_sha256');scored=[]
        for record in self.state['snapshots'].values():
            matches=[j for j in jobs.values() if j['step']==record['step'] and j['snapshot']['sha256']==record['packet']['sha256']]
            if not matches and record['status']=='pending_asr' and any(j['step']>record['step'] for j in jobs.values()):
                record['status']='discardable';record['discard_reason']='ASR monitor moved to another native packet without scoring this captured step'
            for job in matches:
                record['quality_job']=job['key'];metrics=job.get('metrics',{})
                if job['status']=='complete' and metrics.get('source_sha256')==record['packet']['sha256'] and metrics.get('policy_sha256')==job['policy_sha256'] and math.isfinite(metrics.get('wer',float('nan'))):
                    record.update(status='scored',metrics=metrics,policy_sha256=job['policy_sha256'])
                elif job['status'] in ('failed','expired_unscored'):record['status']='discardable'
            if record.get('status')=='scored' and record.get('policy_sha256')==policy and not record.get('removed_utc'):scored.append(record)
        if scored:self.state['best']=min(scored,key=lambda r:(r['metrics']['wer'],r['step']))['key']
        self.state['observed_best_asr']=observed
        if observed and not any(r.get('quality_job')==observed and not r.get('removed_utc') for r in self.state['snapshots'].values()):
            self.review('observed_best_checkpoint_unavailable',step=jobs[observed]['step'],reason='No exact retained checkpoint; never substitute later or mel-best weights')

    def capture(self,quality):
        receipt=read(self.run/'checkpoint.json',{});step=receipt.get('step');validation=read(self.run/'validation-latest.json',{})
        recipe=read(self.run/'quality-monitor/recipe.json',{});jobs=quality.get('jobs',{})
        candidates=[j for j in jobs.values() if j['step']==step and j['status'] in ('pending','running','complete')]
        scheduled=(type(step) is int and step>=recipe.get('first_step',4000) and (step-recipe.get('first_step',4000))%recipe.get('every_steps',2000)==0 and validation.get('step')==step)
        if not candidates and not scheduled:return
        if any(r['step']==step for r in self.state['snapshots'].values()):return
        outstanding=[r for r in self.state['snapshots'].values() if r['key']!=self.state.get('best') and not r.get('removed_utc')]
        if outstanding:return  # One pending slot; also wait for obsolete links still in use.
        source=self.run/'last.pt';stat=safe_file(source,CP_LIMIT);usage=budget(self.root,stat.st_size)
        write(self.out/'budget-latest.json',dict(utc=utc(),**usage))
        if not usage['allowed']:self.review('budget_blocks_capture',budget=usage);return
        stage=self.snapshots/f'.stage-{os.getpid()}';stage.mkdir(mode=0o700)
        try:
            if candidates:
                job=max(candidates,key=lambda j:j['step']);source_packet=Path(job['directory'])/'input-mels.pt'
                if source_packet.resolve().parent.parent!=(self.run/'quality-monitor/probes').resolve():raise ValueError('Unexpected native snapshot directory')
                safe_file(source_packet,64*1024**2);os.link(source_packet,stage/'input-mels.pt')
                packet=job['snapshot'].copy()
                if packet['step']!=step or sha(stage/'input-mels.pt')!=packet['sha256']:raise ValueError('Frozen native packet changed')
            else:packet=freeze_snapshot(self.run,stage/'input-mels.pt')
            if packet['step']!=step:return
            record=bind_checkpoint(self.run,stage,packet)
            if record:
                record=commit_stage(stage,self.snapshots,record);self.state['snapshots'][record['key']]=record
                self.log.info(json.dumps(dict(event='exact_state_retained',step=step,checkpoint_sha256=record['checkpoint_sha256'],packet_sha256=packet['sha256'])))
        finally:
            if stage.exists() and not (stage/'checkpoint.pt').exists() and not remove_stage(stage,self.snapshots):self.review('staging_in_use',directory=str(stage))

    def cleanup(self):
        pending=sorted((r for r in self.state['snapshots'].values() if r['status']=='pending_asr' and not r.get('removed_utc')),key=lambda r:r['step'])
        keep={self.state.get('best')}|{r['key'] for r in pending[:1]}
        for record in self.state['snapshots'].values():
            path=Path(record['directory'])
            if record['key'] in keep or not path.exists():continue
            if path.is_symlink() or path.resolve().parent!=self.snapshots.resolve() or path.stat().st_uid!=os.getuid():raise ValueError('Unsafe retention cleanup path')
            if in_use(path):self.review('old_retention_in_use',directory=str(path));continue
            allocated=physical_usage([path])['allocated_bytes']
            exclusive=sum(p.stat().st_blocks*512 for p in path.iterdir() if p.is_file() and p.stat().st_nlink==1)
            shutil.rmtree(path);record['removed_utc']=utc()
            with (self.out/'cleanup.jsonl').open('a') as stream:stream.write(json.dumps(dict(utc=utc(),directory=str(path),unlinked_allocated_bytes=allocated,exclusive_allocated_bytes_reclaimed=exclusive,checkpoint_sha256=record['checkpoint_sha256'],reason='Own obsolete link; no fuser use; original last/best untouched'))+'\n')

    def iteration(self):
        if (self.root/'pause-asr').exists():
            self.publish_status(paused=True)
            return
        self.reconcile_stages()
        quality=read(self.run/'quality-monitor/state.json',{})
        self.scores(quality);self.cleanup();self.capture(quality);self.scores(quality);self.cleanup()
        write(self.out/'state.json',self.state)
        self.publish_status(paused=False)

    def publish_status(self,paused):
        status=dict(utc=utc(),paused=paused,best_recoverable=self.state.get('best'),observed_best_asr=self.state.get('observed_best_asr'),
            retained=[{k:r.get(k) for k in ('key','step','status','directory','checkpoint_sha256')} for r in self.state['snapshots'].values() if not r.get('removed_utc')],product=False)
        write(self.out/'status.json',status)
        if time.monotonic()-self.last_heartbeat>=1800:write(self.out/'heartbeat.json',status);self.last_heartbeat=time.monotonic()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'fastpitch/language');p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--poll-seconds',type=float,default=2.);p.add_argument('--once',action='store_true');a=p.parse_args()
    if not math.isfinite(a.poll_seconds) or a.poll_seconds<=0:p.error('Positive finite poll required')
    os.umask(0o077);a.run=a.run.resolve()
    if any((p/'.git').exists() for p in (a.run,*a.run.parents)):p.error('Private retention artifacts must stay outside Git')
    out=a.run/'asr-checkpoints';out.mkdir(parents=True,exist_ok=True,mode=0o700)
    lock=(out/'supervisor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);keeper=Keeper(a)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:setattr(keeper,'running',False))
    while keeper.running:
        try:keeper.iteration()
        except Exception as error:keeper.log.exception('retention_iteration_failed');keeper.review('retention_error',error=repr(error))
        if a.once:return
        time.sleep(a.poll_seconds)


if __name__=='__main__':main()
