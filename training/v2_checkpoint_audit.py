"""Read-only CPU audit of an atomically published private VITS checkpoint."""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,time
from pathlib import Path


def write(path,value):
    temporary=path.with_suffix('.new');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-step',type=int,default=1);p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=30)
    a=p.parse_args();os.umask(0o077);os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1'
    a.output.mkdir(mode=0o700,parents=True,exist_ok=True)
    path=a.run/'last.pt';receipt=a.run/'last.pt.json';initial=path.stat()
    observations=[];last_seen=None
    while True:
        stat=path.stat();metadata=receipt.read_bytes();r=json.loads(metadata)
        key=(stat.st_ino,stat.st_size,r['step'])
        if key!=last_seen:
            observations.append(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inode=stat.st_ino,bytes=stat.st_size,
                                     metadata_step=r['step'],temporary_visible=(a.run/'last.pt.tmp').exists()))
            write(a.output/'observed-transitions.json',observations);last_seen=key
        if r['step'] < a.min_step:
            if not a.watch:raise ValueError('checkpoint is before requested minimum step')
            time.sleep(a.interval);continue
        # Open once: atomic renames cannot change this file descriptor's object.
        with path.open('rb') as stream:
            opened=os.fstat(stream.fileno());h=hashlib.sha256()
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            if receipt.read_bytes()!=metadata or path.stat().st_ino!=opened.st_ino:
                time.sleep(a.interval);continue
            import torch
            torch.set_num_threads(1);stream.seek(0)
            checkpoint=torch.load(stream,map_location='cpu',weights_only=False)
        checks=dict(hash_matches=h.hexdigest()==r['sha256'],size_matches=opened.st_size==r['bytes'],
                    step_matches=checkpoint['step']==r['step'],epoch_matches=checkpoint['epoch']==r['epoch'],
                    cursor_matches=checkpoint['cursor']==r['cursor'],precision_fp32=checkpoint['precision']=='fp32',
                    stage_language=checkpoint.get('stage')=='language',optimizer_count=len(checkpoint['optimizers'])==2,
                    scheduler_count=len(checkpoint['schedulers'])==2,
                    rng_keys={'python','numpy','torch','cuda'}<=checkpoint['rng'].keys(),
                    permutation_complete=sorted(checkpoint['permutation'])==list(range(len(checkpoint['permutation']))),
                    valid_cursor=0<=checkpoint['cursor']<=len(checkpoint['permutation']),
                    cuda_hidden=os.environ['CUDA_VISIBLE_DEVICES']=='',
                    published_new_inode=opened.st_ino!=initial.st_ino)
        finite_count=0
        def finite(value):
            nonlocal finite_count
            if torch.is_tensor(value):
                finite_count+=1
                return not value.is_floating_point() or bool(torch.isfinite(value).all())
            if isinstance(value,dict):return all(finite(x) for x in value.values())
            if isinstance(value,(list,tuple)):return all(finite(x) for x in value)
            return True
        checks['model_finite']=finite(checkpoint['model']);checks['optimizer_states_finite']=finite(checkpoint['optimizers'])
        checks['checkpoint_and_receipt_still_same_object']=receipt.read_bytes()==metadata and path.stat().st_ino==opened.st_ino
        result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint=str(path),sha256=h.hexdigest(),
                    inode=opened.st_ino,initial_inode=initial.st_ino,bytes=opened.st_size,step=checkpoint['step'],epoch=checkpoint['epoch'],
                    cursor=checkpoint['cursor'],dataset_fingerprint=checkpoint['dataset_fingerprint'],checks=checks,tensors_checked=finite_count,
                    optimizer_steps=[sorted({int(state['step'].item()) for state in optimizer['state'].values() if 'step' in state})
                                     for optimizer in checkpoint['optimizers']],
                    scheduler_epochs=[s['last_epoch'] for s in checkpoint['schedulers']],observations=observations,
                    scope='CPU read of one open descriptor; full SHA, model/optimizer finiteness and resumed-state metadata; no GPU/model execution')
        write(a.output/'result.json',result);print(json.dumps(result),flush=True)
        if not all(checks.values()):raise SystemExit(1)
        return


if __name__=='__main__':main()
