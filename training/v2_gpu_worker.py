"""Private sequential GPU task queue, run INSIDE av2_ops rollback guard.

A completed/failed task never hides its exit code. No training deadline. The
agent supplies the next concrete command via atomic JSON in queue/. Long jobs
write their own step/checkpoint evidence. Polling and progress logging are local.
"""
import argparse,datetime,fcntl,json,os,signal,subprocess,time
from pathlib import Path

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,value):
 tmp=path.with_suffix('.new');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();os.umask(0o077);root=a.root
 lock=(root/'gpu-worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for d in ('queue','done','logs'): (root/d).mkdir(exist_ok=True)
 started=time.monotonic();idle=started;last_heartbeat=0;child=None
 def terminate(sig,frame):
  if child is not None and child.poll() is None:child.send_signal(signal.SIGINT);child.wait(timeout=120)
  raise SystemExit(128+sig)
 for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,terminate)
 while not (root/'finish-gpu').exists():
  tasks=sorted((root/'queue').glob('*.json'))
  if not tasks:
   if time.monotonic()-idle>3600:write(root/'worker-exit.json',dict(utc=utc(),reason='idle1h',no_training_time_limit=True));return
   time.sleep(2);continue
  task=tasks[0];job=json.loads(task.read_text());assert isinstance(job['command'],list)
  if os.statvfs(root).f_bavail*os.statvfs(root).f_frsize<5*1024**3:raise RuntimeError('reserve below5GiB before start')
  result=dict(job=job,started_utc=utc(),worker_pid=os.getpid(),status='running')
  log=root/'logs'/(task.stem+'.log')
  with log.open('ab') as output:
   child=subprocess.Popen(job['command'],cwd=job.get('cwd'),stdout=output,stderr=subprocess.STDOUT)
   result['pid']=child.pid;write(root/'worker-current.json',result)
   while child.poll() is None:
    if time.monotonic()-last_heartbeat>1800:
     write(root/'heartbeat.json',dict(utc=utc(),job=task.stem,pid=child.pid,free_bytes=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize,log_bytes=log.stat().st_size,next_check_seconds=1800));last_heartbeat=time.monotonic()
    if os.statvfs(root).f_bavail*os.statvfs(root).f_frsize<5*1024**3:child.send_signal(signal.SIGINT);result['emergency']='disk reserve below5GiB'
    if log.stat().st_size>32*1024**2:
     # Retain recent tail only while child append fd stays on same inode.
     with log.open('r+b') as f:f.seek(-4*1024**2,2);tail=f.read();f.seek(0);f.write(tail);f.truncate()
    time.sleep(2)
  result.update(status='succeeded' if child.returncode==0 else 'failed',returncode=child.returncode,finished_utc=utc());write(root/'done'/task.name,result);task.unlink();idle=time.monotonic();write(root/'worker-current.json',result)
 write(root/'worker-exit.json',dict(utc=utc(),reason='agent_finished_gpu_tasks'))
if __name__=='__main__':main()
