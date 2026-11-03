"""CPU-only target feature audit/preparation under an owned pauseable process group."""
import argparse
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import time
from training.v2_language_monitor import alive,physical_usage,read,sha,ticks,utc,write
from training.v2_fastpitch_target_plan import RECIPE_SHA

ROOT=Path('/var/tmp/aegis-v2-work');REPO=Path(__file__).resolve().parents[1]
PEAK_EXTRA=530_000_000+32*1024**2+55_000_000


def budget(root):
    roots=sorted(Path('/var/tmp').glob('aegis*'))+[Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip'),REPO,Path('/home/phc/.cache')]
    result=physical_usage(roots);free=__import__('shutil').disk_usage(root).free
    return dict(utc=utc(),usage=result,free_bytes=free,estimated_peak_extra_bytes=PEAK_EXTRA,
        allowed=not result['permission_errors'] and result['allocated_bytes']+PEAK_EXTRA<=40_000_000_000 and free>=5*1024**3+PEAK_EXTRA)


def owned_signal(pid,start,sig):
    if not alive(pid,start):return False
    if Path(f'/proc/{pid}').stat().st_uid!=os.getuid() or os.getpgid(pid)!=pid or os.getsid(pid)!=pid:
        raise RuntimeError('Refuse signal: CPU child ownership/group identity differs')
    os.killpg(pid,sig)
    return True


def wait_pause(root):
    while (root/'pause-asr').exists():time.sleep(.25)


def worker(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden')
    wait_pause(a.root)
    recipe=REPO/'training/fastpitch_experiment.py'
    if sha(recipe)!=RECIPE_SHA:raise RuntimeError('Reviewed recipe changed')
    usage=budget(a.root);write(a.preparation/'pre-extraction-budget.json',usage)
    if not usage['allowed']:raise RuntimeError('Global40GB/free5GiB peak budget blocks target feature stage')
    wait_pause(a.root)
    import torch
    from training import fastpitch_experiment as f
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    prep=a.preparation;output=a.features;plan=read(prep/'plan.json');reuse=read(prep/'reuse-plan.json')['reusable']
    selected={};source_manifests={}
    for split in ('train','validation'):
        path=prep/(split+'.jsonl');expected=plan['manifests'][split]['sha256']
        if sha(path)!=expected:raise RuntimeError('Frozen target manifest changed')
        source_manifests[str(path)]=expected
        for row in f.rows(path):
            audio=Path(row['audio_filepath']);audio_stat=audio.lstat()
            if audio.is_symlink() or not stat.S_ISREG(audio_stat.st_mode) or audio_stat.st_uid!=os.getuid():raise ValueError('Original audio owner/type differs')
            selected[row['id']]=row
    if len(selected)!=1080:raise RuntimeError('Unexpected selected target count')
    output.mkdir(parents=True,exist_ok=True);(output/'features').mkdir(exist_ok=True)
    valid=[];failed=[];started=time.monotonic()
    write(prep/'feature-stage-progress.json',dict(utc=utc(),phase='audit_legacy',checked=0,total=len(reuse)))
    for index,row in enumerate(reuse,1):
        wait_pause(a.root);source=Path(row['source']);expected=selected[row['id']]
        try:
            st=source.lstat()
            if source.is_symlink() or not stat.S_ISREG(st.st_mode) or st.st_uid!=os.getuid() or sha(source)!=row['feature_sha256']:
                raise ValueError('Legacy feature owner/type/SHA mismatch')
            item=torch.load(source,map_location='cpu',weights_only=True);audio=Path(expected['audio_filepath'])
            if (item['id']!=row['id'] or item['normalized_text']!=f.normalize(expected['text'])
                or not torch.equal(item['text'],f.tokens(expected['text'])) or len(item['text'])!=len(f.normalize(expected['text']))+2
                or Path(item['audio_filepath']).resolve()!=audio.resolve() or item['audio_sha256']!=expected['source_audio_sha256']
                or sha(audio)!=expected['source_audio_sha256'] or item['mel_config']!=f.MEL_CONFIG
                or item.get('pitch_method')!='world' or item['mel'].shape!=(100,row['frames']) or item['pitch'].shape!=(1,row['frames'])
                or not torch.isfinite(item['mel']).all() or not torch.isfinite(item['pitch']).all() or (item['pitch']<0).any()):
                raise ValueError('Legacy feature/audio/text/token/acoustic identity mismatch')
            valid.append(row)
        except Exception as error:failed.append(dict(id=row['id'],error=repr(error)))
        if index%25==0 or index==len(reuse):
            progress=dict(utc=utc(),phase='audit_legacy',checked=index,total=len(reuse),passed=len(valid),failed=len(failed))
            write(prep/'feature-stage-progress.json',progress);print(json.dumps(progress),flush=True)
    write(prep/'legacy-selected-full-audit.json',dict(utc=utc(),passed=len(valid),failed=failed,seconds=time.monotonic()-started,
        checks='Entire feature/audio SHA; provenance path/id/text/tokens; real owner/type; mel/pitch config/shapes/finite; selected train/validation only'))
    for row in valid:
        wait_pause(a.root);target=output/'features'/(row['id']+'.pt')
        if target.exists():
            if target.is_symlink() or sha(target)!=row['feature_sha256']:raise ValueError('Existing target cache differs; preserve for review')
        else:os.link(row['source'],target)
    wait_pause(a.root)
    write(prep/'feature-stage-progress.json',dict(utc=utc(),phase='prepare',reused=len(valid),missing=1080-len(valid)))
    f.prepare(argparse.Namespace(output=str(output),manifests=list(source_manifests),exclude_id=[],limit=0,workers=2,pitch_method='world'))
    wait_pause(a.root);features=read(output/'features.json');manifests={s:f.rows(output/(s+'.jsonl')) for s in ('train','validation')}
    if {s:len(v) for s,v in manifests.items()}!={'train':1025,'validation':55}:raise ValueError('Wrong prepared split sizes')
    write(prep/'feature-stage-progress.json',dict(utc=utc(),phase='audit_complete_view',total=1080))
    f.audit_features(features,manifests['train']+manifests['validation'],output)
    sums=[0.,0.,0];minimum=float('inf');maximum=0.;frames=0
    for row in manifests['train']:
        wait_pause(a.root);pitch=f.load_feature(row)['pitch'];voiced=pitch[pitch>0].double();frames+=pitch.numel()
        sums=[sums[0]+float(voiced.sum()),sums[1]+float(voiced.square().sum()),sums[2]+voiced.numel()]
        if voiced.numel():minimum=min(minimum,float(voiced.min()));maximum=max(maximum,float(voiced.max()))
    mean=sums[0]/sums[2];std=math.sqrt(max(1.,sums[1]/sums[2]-mean*mean))
    if not math.isclose(features['pitch_mean'],mean,rel_tol=1e-12) or not math.isclose(features['pitch_std'],std,rel_tol=1e-12) or features['pitch_voiced_frames']!=sums[2]:
        raise ValueError('Pitch statistics are not exactly new train-only WORLD Hz')
    wait_pause(a.root);usage=physical_usage([output]);final_budget=budget(a.root)
    result=dict(utc=utc(),status='qualified_cache_only_not_model',recipe_sha256=RECIPE_SHA,worker_source_sha256=sha(__file__),
        counts={s:len(v) for s,v in manifests.items()},source_manifests=source_manifests,
        feature_manifests_sha256={s:sha(output/(s+'.jsonl')) for s in manifests},feature_config_sha256=sha(output/'features.json'),
        reused=len(valid),newly_prepared=1080-len(valid),legacy_failures=failed,allocated_bytes=usage['allocated_bytes'],
        pitch=dict(mean_hz=mean,std_hz=std,positive_frames=sums[2],total_train_frames=frames,min_positive_hz=minimum,max_hz=maximum,statistics_scope='1025train only; WORLD raw Hz labels; unvoiced0'),
        final_budget=final_budget,seconds=time.monotonic()-started,final_test='Never opened; no tensors/audio outside frozen train/validation selected',
        training_gate='No target training authorized by this cache; public language must qualify first')
    write(output/'qualification.json',result);write(prep/'feature-stage-progress.json',dict(utc=utc(),phase='complete',result=str(output/'qualification.json')))
    print(json.dumps(result),flush=True)


def supervise(a):
    a.preparation.mkdir(parents=True,exist_ok=True);lock=(a.preparation/'feature-stage.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=read(a.preparation/'feature-stage-status.json',{})
    if previous.get('child_pid') and alive(previous['child_pid'],previous.get('child_start_ticks')):raise RuntimeError('Previous CPU child still exists; preserve and review, do not duplicate')
    log=logging.getLogger('target-features');log.setLevel(logging.INFO)
    log.addHandler(RotatingFileHandler(a.preparation/'feature-stage.log',maxBytes=1024**2,backupCount=1))
    status=dict(utc=utc(),status='waiting_if_paused',supervisor_pid=os.getpid(),supervisor_start_ticks=ticks(os.getpid()),
        source_sha256=sha(__file__),recipe_sha256=RECIPE_SHA,pause_file=str(a.root/'pause-asr'),cuda_visible_devices='',threads=2)
    write(a.preparation/'feature-stage-launch.json',status);write(a.preparation/'feature-stage-status.json',status);wait_pause(a.root)
    receipt=a.preparation/'pre-extraction-budget.json'
    if not read(receipt,{}).get('allowed') or time.time()-receipt.stat().st_mtime>600:raise RuntimeError('Fresh allowed pre-extraction budget receipt required')
    wait_pause(a.root)
    command=[sys.executable,'-m','training.v2_fastpitch_target_features','--action','worker','--root',str(a.root),'--preparation',str(a.preparation),'--features',str(a.features)]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMBA_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
    proc=subprocess.Popen(command,cwd=REPO,env=env,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    start=ticks(proc.pid);status.update(child_pid=proc.pid,child_start_ticks=start,child_pgid=os.getpgid(proc.pid),command=command,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    os.set_blocking(proc.stdout.fileno(),False);selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
    paused=False;last=0.;stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    while proc.poll() is None:
        desired=(a.root/'pause-asr').exists() or stopping
        if desired!=paused:
            if owned_signal(proc.pid,start,signal.SIGSTOP if desired else signal.SIGCONT):paused=desired
            last=0.
        for key,_ in selector.select(.25):
            block=os.read(key.fd,65536)
            if block:log.info(block.decode(errors='replace').rstrip())
        if time.monotonic()-last>=2:
            status.update(utc=utc(),status='paused' if paused else 'running',paused=paused,progress=read(a.preparation/'feature-stage-progress.json',{}))
            write(a.preparation/'feature-stage-status.json',status);last=time.monotonic()
        if stopping:
            status.update(utc=utc(),status='supervisor_stopped_child_preserved_paused',paused=paused)
            write(a.preparation/'feature-stage-status.json',status);return
    for key,_ in selector.select(0):
        block=os.read(key.fd,65536)
        if block:log.info(block.decode(errors='replace').rstrip())
    status.update(utc=utc(),status='complete' if proc.returncode==0 else 'failed',exit_code=proc.returncode,paused=False,progress=read(a.preparation/'feature-stage-progress.json',{}))
    write(a.preparation/'feature-stage-status.json',status)
    if proc.returncode:raise SystemExit(proc.returncode)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=('supervisor','worker'),default='supervisor')
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--preparation',type=Path,default=ROOT/'fastpitch/aegis-preparation')
    p.add_argument('--features',type=Path,default=ROOT/'fastpitch/aegis-features-v2');a=p.parse_args();os.umask(0o077)
    if any((parent/'.git').exists() for parent in (a.features,*a.features.parents)):p.error('Keep private feature data outside Git')
    if a.action=='worker':worker(a)
    else:supervise(a)


if __name__=='__main__':main()
