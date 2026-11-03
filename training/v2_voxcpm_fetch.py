"""Download only two immutable VoxCPM2 weights, with project/disk peak gates."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
from training.v2_voxcpm_probe import WEIGHTS,MODEL_REV,ROOT,atomic_json,sha

def allocated(roots):
 seen=set();total=0;per={}
 for root in roots:
  count=0
  if not root.exists():continue
  paths=[root] if root.is_file() else root.rglob('*')
  for path in paths:
   try:st=path.lstat()
   except FileNotFoundError:continue
   if path.is_symlink() or not path.is_file():continue
   key=(st.st_dev,st.st_ino)
   if key in seen:continue
   seen.add(key);count+=st.st_blocks*512
  total+=count;per[str(root)]=count
 return total,per

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--model',type=Path,default=ROOT/'model');p.add_argument('--ceiling',type=int,default=40_000_000_000);p.add_argument('--reserve-overlap',type=int,default=2_500_000_000);a=p.parse_args();os.umask(0o077)
 resources=Path('/var/tmp/aegis-v2-work/resources-current.json');known=json.loads(resources.read_text());roots=[Path(x) for x in known['roots']]+[Path('/home/phc/aegis-v2-lab')]
 if ROOT.parent not in roots:raise ValueError('working roots omit Vox directory')
 with urllib.request.urlopen('https://huggingface.co/api/models/openbmb/VoxCPM2/revision/'+MODEL_REV+'?blobs=true',timeout=30) as f:info=json.load(f)
 if info['sha']!=MODEL_REV or info.get('cardData',{}).get('license')!='apache-2.0':raise ValueError('remote revision/license metadata changed')
 siblings={r['rfilename']:r for r in info['siblings']};remaining=0
 for name,(expected,size) in WEIGHTS.items():
  item=siblings[name]
  if item['lfs']['sha256']!=expected or item['size']!=size:raise ValueError('remote LFS integrity metadata mismatch '+name)
  dest=a.model/name
  if dest.exists():
   if dest.stat().st_size!=size or sha(dest)!=expected:raise ValueError('existing local weights mismatch '+name)
  else:remaining+=size
 current,per=allocated(roots);free=shutil.disk_usage(a.model).free
 plan=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),model_revision=MODEL_REV,remote_license=info['cardData']['license'],
  current_allocated_unique_bytes=current,remaining_weight_bytes=remaining,checkpoint_overlap_reserve_bytes=a.reserve_overlap,
  projected_project_peak_bytes=current+remaining+a.reserve_overlap,project_ceiling_bytes=a.ceiling,
  available_volume_bytes=free,minimum_volume_reserve_bytes=5_000_000_000,roots=per,
  peak_policy='direct .part->final on same volume, no duplicate caches; include simultaneous FP/VITS checkpoints in overlap reserve',files={k:dict(sha256=v[0],bytes=v[1]) for k,v in WEIGHTS.items()})
 if plan['projected_project_peak_bytes']>a.ceiling:raise ValueError('project peak would exceed 40GB: '+json.dumps(plan))
 if free-remaining-a.reserve_overlap<5_000_000_000:raise ValueError('volume minimum reserve would be violated')
 atomic_json(ROOT/'download-plan.json',plan);print(json.dumps({k:v for k,v in plan.items() if k!='roots'}),flush=True)
 if not a.execute:return
 results=[]
 for name,(expected,size) in WEIGHTS.items():
  dest=a.model/name
  if dest.exists():results.append(dict(name=name,sha256=expected,bytes=size,status='existing verified'));continue
  part=dest.with_name(dest.name+'.part')
  if part.exists():raise ValueError('inspect existing partial file before restarting: '+str(part))
  h=hashlib.sha256();count=0;last_log=0;start=time.monotonic()
  request=urllib.request.Request('https://huggingface.co/openbmb/VoxCPM2/resolve/'+MODEL_REV+'/'+name,headers={'Accept-Encoding':'identity'})
  with urllib.request.urlopen(request,timeout=90) as response,part.open('xb') as f:
   while True:
    block=response.read(8*1024*1024)
    if not block:break
    count+=len(block)
    if count>size:raise ValueError('download exceeds expected length')
    f.write(block);h.update(block)
    if count-last_log>=128*1024*1024:
     if shutil.disk_usage(a.model).free<5_000_000_000+a.reserve_overlap:raise ValueError('disk reserve at risk during download; partial preserved')
     progress=dict(name=name,bytes=count,total_bytes=size,seconds=time.monotonic()-start,utc=dt.datetime.now(dt.timezone.utc).isoformat())
     atomic_json(ROOT/'download-progress.json',progress);print(json.dumps(progress),flush=True);last_log=count
   f.flush();os.fsync(f.fileno())
  if count!=size or h.hexdigest()!=expected:raise ValueError('download integrity failure, partial retained: '+name)
  # Re-read the completed local file before it becomes the loader's final input.
  if sha(part)!=expected:raise ValueError('local re-read SHA mismatch '+name)
  part.replace(dest);directory=os.open(a.model,os.O_RDONLY)
  try:os.fsync(directory)
  finally:os.close(directory)
  results.append(dict(name=name,bytes=count,sha256=expected,seconds=time.monotonic()-start,source='official pinned HF revision; remote LFS SHA checked',status='verified local final'))
  atomic_json(ROOT/'download-receipt.json',dict(revision=MODEL_REV,files=results,completed=False))
 current,per=allocated(roots);free=shutil.disk_usage(a.model).free
 if current+a.reserve_overlap>a.ceiling or free<5_000_000_000:raise ValueError('post-download budget check requires intervention; weights retained')
 atomic_json(ROOT/'download-receipt.json',dict(revision=MODEL_REV,files=results,completed=True,final_project_allocated_unique_bytes=current,free_volume_bytes=free,roots=per))
 print(json.dumps(dict(completed=True,project_bytes=current,free_volume_bytes=free)),flush=True)
if __name__=='__main__':main()
