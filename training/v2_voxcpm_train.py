"""Whole-utterance Aegis LoRA training, with complete bounded recovery state.

Default is a metadata-only plan. Real training requires the exclusive GPU
worker and already-local pinned weights. No final-test utterance is loaded.
"""
from __future__ import annotations
import argparse
import collections
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import unicodedata
from types import SimpleNamespace

from training import v2_voxcpm_probe as pilot
from training import v2_voxcpm_cache_probe as capacity
from training.v2_language_monitor import physical_usage

ROOT=pilot.ROOT.parent
REPO=Path(__file__).resolve().parents[1]
PREPARED=Path('/var/tmp/aegisvoice-v2-training-20261102/prepared')
RESERVE=512*1024**2


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+f'.tmp-{os.getpid()}')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def read(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:return default


def budget():
    roots=sorted(Path('/var/tmp').glob('aegis*'))+[Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip'),REPO,Path('/home/phc/.cache')]
    usage=physical_usage(roots);free=shutil.disk_usage(ROOT).free
    if usage['permission_errors'] or usage['allocated_bytes']+RESERVE>40_000_000_000 or free<5*1024**3+RESERVE:
        raise RuntimeError('Working budget/resumable-checkpoint reserve unavailable: '+json.dumps(dict(usage=usage,free=free)))
    return dict(**usage,free_bytes=free,estimated_next_peak_extra_bytes=RESERVE)


def load_data(args):
    rows={};manifests={};seen={kind:{} for kind in ('id','group','text','audio')}
    for split in ('train','validation'):
        path=args.prepared/(split+'.jsonl');manifests[split]=pilot.sha(path)
        rows[split]=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        for row in rows[split]:
            if 'original_text' not in row:raise ValueError('Original transcript missing: '+row['id'])
            row['_text']=unicodedata.normalize('NFC',row['original_text'])
            if not row['_text'].strip() or not row.get('source_audio_sha256') or not row.get('source_audio_filepath'):
                raise ValueError('Original transcript/source audio binding missing: '+row['id'])
            keys=dict(id=row['id'],group=row.get('group_id',row['id']),
                      text=' '.join(row['_text'].casefold().split()),audio=row['source_audio_sha256'])
            for kind,key in keys.items():
                previous=seen[kind].setdefault(key,split)
                if previous!=split or (kind=='id' and sum(r['id']==row['id'] for r in rows[split])!=1):
                    raise ValueError('Duplicate/cross-partition '+kind+': '+row['id'])
        if not rows[split]:raise ValueError('Empty '+split)
    original_counts={s:len(v) for s,v in rows.items()}
    if args.pilot_longest:rows['train']=[max(rows['train'],key=lambda row:row['duration'])]
    elif args.train_limit:rows['train']=rows['train'][:args.train_limit]
    if args.validation_limit:rows['validation']=rows['validation'][:args.validation_limit]
    fingerprint=dict(manifests=manifests,selection={s:[r['id'] for r in v] for s,v in rows.items()},
                     text_policy='NFC original_text, original spelling/numerals; no extra frontend/DSP',
                     audio_policy='Original bound mono WAV ->16k float32 resample_poly Kaiser10 at read; whole utterance, no crop/trim/gain',
                     original_counts=original_counts,test='50 held-out target rows and six final sentences are not read')
    data=dict(rows=rows,fingerprint=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest(),manifest=fingerprint)
    return data


class AudioCache:
    def __init__(self,mib,np):self.limit=mib*1024**2;self.np=np;self.bytes=0;self.items=collections.OrderedDict()
    def get(self,row):
        path=Path(row['source_audio_filepath']);s=path.stat();identity=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
        key=(str(path),row['source_audio_sha256']);cached=self.items.pop(key,None)
        if cached:
            self.bytes-=cached[1].nbytes
            if cached[0]==identity:self.items[key]=cached;self.bytes+=cached[1].nbytes;return cached[1]
        values,_,_=pilot.audio(row,self.np)
        if values.nbytes<=self.limit:
            while self.items and self.bytes+values.nbytes>self.limit:
                _,(_,old)=self.items.popitem(last=False);self.bytes-=old.nbytes
            self.items[key]=(identity,values);self.bytes+=values.nbytes
        return values


def make_batch(row,b,cache):
    t=b.t;values=cache.get(row)
    ids=t.tensor(b.model.text_tokenizer(row['_text']),device=b.device,dtype=t.int32)
    if not ids.numel():raise ValueError('Empty native tokenization: '+row['id'])
    packed=b.packer.process_tts_data(t.from_numpy(values.copy()).to(b.device),ids)
    tokens,features,text_mask,audio_mask,loss_mask,labels=packed[:6]
    if len(tokens)>b.max_sequence:raise ValueError('Whole utterance exceeds sequence limit; refuse crop: '+row['id'])
    if int(loss_mask.sum())<=0 or int(labels.sum())!=1 or int(labels[-2])!=1:
        raise ValueError('Native audio mask/stop-label contract failed')
    return [v[None] for v in (tokens,text_mask,features,audio_mask,loss_mask)]+[t.arange(len(tokens),device=b.device)[None],labels[None]],len(tokens)


def rng(t,np,device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=t.get_rng_state(),
                cuda=t.cuda.get_rng_state_all() if device=='cuda' else [])


def restore_rng(value,t,np,device):
    random.setstate(value['python']);np.random.set_state(value['numpy']);t.set_rng_state(value['torch'])
    if device=='cuda':t.cuda.set_rng_state_all(value['cuda'])


def repair_inference_caches(model,t):
    """Replace persistent inference tensors outside inference_mode before autograd."""
    repaired=[]
    with t.inference_mode(False):
        for name,module in model.named_modules():
            for key,value in list(module._buffers.items()):
                if t.is_tensor(value) and t.is_inference(value):
                    module._buffers[key]=value.clone();repaired.append(name+'.buffer:'+key)
            for key,value in list(vars(module).items()):
                if t.is_tensor(value) and not isinstance(value,t.nn.Parameter) and t.is_inference(value):
                    setattr(module,key,value.clone());repaired.append(name+'.attr:'+key)
        for name in ('base_lm','residual_lm'):
            lm=getattr(model,name,None);kv=getattr(lm,'kv_cache',None)
            if kv is not None and t.is_inference(kv.kv_cache):
                kv.kv_cache=kv.kv_cache.clone();repaired.append(name+'.kv_cache')
    return repaired


@contextlib.contextmanager
def validation_context(b,seed):
    state=rng(b.t,b.np,b.device);training=b.model.training
    try:
        random.seed(seed);b.np.random.seed(seed);b.t.manual_seed(seed)
        if b.device=='cuda':b.t.cuda.manual_seed_all(seed)
        b.model.eval()
        with b.t.no_grad():yield
    finally:
        b.last_repaired=repair_inference_caches(b.model,b.t)
        b.model.train(training)
        if hasattr(b.model,'audio_vae'):b.model.audio_vae.eval()
        restore_rng(state,b.t,b.np,b.device)


def loss(batch,b):
    with b.t.autocast(b.device,dtype=b.t.bfloat16,enabled=b.device=='cuda'):
        out=b.model(*batch)
        total=out['loss/diff']+out['loss/stop']
    if not b.t.isfinite(total):raise FloatingPointError('Nonfinite diff+stop loss')
    return total


def validation(args,b,data,cache,step):
    values=[]
    with validation_context(b,args.seed+100000):
        for index,row in enumerate(data['rows']['validation']):
            b.t.manual_seed(args.seed+100000+index)
            batch,_=make_batch(row,b,cache);values.append(float(loss(batch,b)));del batch
    return sum(values)/len(values)


def native_probes(args,b,step):
    cases=read(REPO/'benchmarks/v2/validation.pt-BR.json')
    if args.probe_limit:cases=cases[:args.probe_limit]
    directory=args.run/'probes'/f'step-{step:08d}-{time.time_ns()}'
    directory.mkdir(parents=True,exist_ok=False,mode=0o700)
    rows=[];jobs=[]
    with validation_context(b,args.seed+200000):
        for case in cases:
            command=SimpleNamespace(run=directory,text=case['text'],nfe=args.nfe)
            row=pilot.synthesize(command,b.model,b.t,b.np,case['id'])
            rows.append(dict(case=case['id'],**row))
            if row['outcome']=='ok':
                jobs.append(dict(candidate='voxcpm2-aegis-step'+str(step),case=case['id'],wav=row['wav'],
                                 expected_verbatim=case['text'],cohort='validation-seven-native-eager'))
    atomic_json(directory/'results.json',dict(step=step,results=rows,scope='Eager full-GPU validation audio only; no compiled latency or auditory claim'))
    atomic_json(directory/'asr-jobs.json',dict(jobs=jobs))
    return dict(directory=str(directory),cases=len(rows),valid_pcm=len(jobs),inference_caches_repaired=b.last_repaired)


def finite(value,t):
    if t.is_tensor(value):return not value.is_floating_point() or bool(t.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v,t) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite(v,t) for v in value)
    return not isinstance(value,float) or math.isfinite(value)


def audit_state(s,t):
    required={'lora','optimizer','scheduler','scaler','rng','step','epoch','cursor','permutation','best','bad','lr_state','config','provenance','parameter_names'}
    if required-set(s) or not finite(s,t):raise ValueError('Missing/nonfinite saved state')
    n=s['config']['training_count'];order=s['permutation']
    if sorted(order)!=list(range(n)) or not 0<=s['cursor']<=n:raise ValueError('Invalid saved permutation/cursor')
    if any(type(s[k]) is not int or s[k]<0 for k in ('step','epoch','bad')):raise ValueError('Invalid saved counters')
    names=s['parameter_names'];opt=s['optimizer'];ids=[i for group in opt['param_groups'] for i in group['params']]
    if set(names)!=set(s['lora']) or len(names)!=len(set(names)) or len(names)!=len(ids) or len(ids)!=len(set(ids)):raise ValueError('LoRA/Adam parameter mapping mismatch')
    if any('lora_' not in name or s['lora'][name].dtype!=t.float32 for name in names):raise ValueError('Saved trainable tensors must be FP32 LoRA only')
    if s['step'] and set(opt['state'])!=set(ids):raise ValueError('Incomplete Adam coverage')
    for name,ident in zip(names,ids):
        if not s['step']:continue
        state=opt['state'][ident]
        if not {'step','exp_avg','exp_avg_sq'}<=set(state):raise ValueError('Missing Adam moments')
        if float(state['step'])!=s['step']:raise ValueError('Adam step differs from completed update')
        if any(state[key].shape!=s['lora'][name].shape or state[key].dtype!=t.float32 for key in ('exp_avg','exp_avg_sq')):raise ValueError('Adam moment shape/dtype mismatch')
    if s['scheduler']['last_epoch']!=s['step']:raise ValueError('Scheduler step mismatch')
    if set(s['rng'])!={'python','numpy','torch','cuda'}:raise ValueError('Incomplete saved RNG')
    random.Random().setstate(s['rng']['python'])
    import numpy as np
    np.random.RandomState().set_state(s['rng']['numpy']);t.Generator(device='cpu').set_state(s['rng']['torch'])
    if s['provenance']['device']=='cuda' and (not s['rng']['cuda'] or any(v.dtype!=t.uint8 or not v.numel() for v in s['rng']['cuda'])):raise ValueError('Missing CUDA RNG')
    reduction=s['lr_state']['reductions'];multiplier=s['lr_state']['multiplier']
    if reduction not in (0,1) or not math.isclose(multiplier,.3**reduction):raise ValueError('Invalid LR reduction state')
    expected=s['config']['lr']*min(1.,(s['step']+1)/s['config']['warmup'])*multiplier
    if len(s['scheduler']['_last_lr'])!=len(opt['param_groups']) or any(not math.isclose(g['lr'],expected,rel_tol=1e-10) or not math.isclose(g['lr'],lr,rel_tol=1e-10) for g,lr in zip(opt['param_groups'],s['scheduler']['_last_lr'])):raise ValueError('Inconsistent saved learning rates')
    return s


def publish(path,temporary,receipt):
    pending=path.with_suffix('.pending.json');sidecar=path.with_suffix('.sha256.json')
    atomic_json(pending,receipt);os.replace(temporary,path);atomic_json(sidecar,receipt)
    pending.unlink()


def save_checkpoint(args,b,s,is_best=False):
    budget();path=args.run/'last.pt';temporary=path.with_name(f'.last-{os.getpid()}.tmp')
    b.t.save(s,temporary)
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    checked=audit_state(b.t.load(temporary,map_location='cpu',weights_only=False),b.t)
    if checked['step']!=s['step'] or pilot.checksum_tensors(checked['lora'].items(),b.t)!=pilot.checksum_tensors(s['lora'].items(),b.t):
        raise ValueError('Temporary checkpoint failed exact LoRA reload')
    receipt=dict(sha256=pilot.sha(temporary),step=s['step'],bytes=temporary.stat().st_size)
    publish(path,temporary,receipt)
    if is_best:
        temporary=args.run/'.best.tmp';temporary.unlink(missing_ok=True);os.link(path,temporary)
        publish(args.run/'best.pt',temporary,receipt)


def load_checkpoint(path,t):
    path=Path(path);actual=pilot.sha(path);receipt=read(path.with_suffix('.sha256.json'),{})
    pending=read(path.with_suffix('.pending.json'),{})
    if actual!=receipt.get('sha256') and actual!=pending.get('sha256'):raise ValueError('Checkpoint SHA mismatch')
    state=audit_state(t.load(path,map_location='cpu',weights_only=False),t)
    if actual!=receipt.get('sha256'):atomic_json(path.with_suffix('.sha256.json'),pending)
    path.with_suffix('.pending.json').unlink(missing_ok=True)
    return state


def emit(args,event):
    record=dict(utc=pilot.dt.datetime.now(pilot.dt.timezone.utc).isoformat(),**event)
    path=args.run/'steps.jsonl'
    if path.exists() and path.stat().st_size>5*1024**2:
        tail=path.read_bytes()[-2*1024**2:].split(b'\n',1)[-1];temp=path.with_suffix('.tmp');temp.write_bytes(tail);temp.replace(path)
    with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
    atomic_json(args.run/'progress.json',record);print(json.dumps(record),flush=True)


def cleanup_probes(args,best_step):
    directories=sorted((args.run/'probes').glob('step-*'),key=lambda p:p.stat().st_mtime_ns)
    complete=[p for p in directories if (p/'results.json').exists()]
    keep=set(complete[-3:])
    for step in (0,best_step):
        matches=[p for p in complete if read(p/'results.json')['step']==step]
        if matches:keep.add(matches[-1])
    for path in directories:
        if path in keep:continue
        if path.is_symlink() or path.resolve().parent!=(args.run/'probes').resolve() or path.stat().st_uid!=os.getuid():raise ValueError('Unsafe probe cleanup path')
        import subprocess
        files=[str(p) for p in path.rglob('*') if p.is_file()]
        if files and subprocess.run(['fuser',*files],capture_output=True,timeout=10).returncode!=1:continue
        allocated=physical_usage([path])['allocated_bytes'];shutil.rmtree(path)
        with (args.run/'cleanup.jsonl').open('a') as stream:stream.write(json.dumps(dict(path=str(path),allocated_bytes=allocated,reason='Closed native validation probes outside latest3, initial and current best; no fuser use'))+'\n')


def training_config(args,b,data):
    return dict(rank=args.rank,lr=args.lr,warmup=args.warmup,seed=args.seed,accumulation=args.accumulation,
        microbatch=1,training_count=len(data['rows']['train']),data_fingerprint=data['fingerprint'],
        source_rev=pilot.SOURCE_REV,model_rev=pilot.MODEL_REV,base_weights=pilot.WEIGHTS,
        precision='BF16 backbone/autocast; FP32 AudioVAE/LoRA/Adam',objective='native loss/diff + loss/stop',
        saved_tensors_cpu=args.saved_tensors_cpu,max_sequence=args.max_sequence,validation_every=args.validate_every,
        validation_count=len(data['rows']['validation']),validation_noise='seed+100000+row_index; all training RNG restored',
        nfe=args.nfe,kv_cache_capacity=1024,patience=args.patience,min_delta=args.min_delta,
        plateau_lr_factor=.3,max_lr_reductions=1,scope='engineering pilot' if args.pilot_longest or args.train_limit or args.validation_limit or args.probe_limit else 'full frozen target specialization')


def _train_locked(args,b,data):
    t,np=b.t,b.np;args.run.mkdir(parents=True,exist_ok=True,mode=0o700)
    config=training_config(args,b,data);named=[(n,p) for n,p in b.model.named_parameters() if p.requires_grad]
    if not named or any('lora_' not in n or p.dtype!=t.float32 for n,p in named):raise ValueError('Unexpected trainable set')
    params=[p for _,p in named];optimizer=t.optim.AdamW(params,lr=args.lr,weight_decay=.01)
    lr_state=dict(multiplier=1.,reductions=0)
    scheduler=t.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/args.warmup)*lr_state['multiplier'])
    scaler=t.amp.GradScaler('cuda',enabled=False)
    provenance=dict(code_sha256=pilot.sha(__file__),pilot_sha256=pilot.sha(pilot.__file__),
                    cache_helper_sha256=pilot.sha(capacity.__file__),torch=t.__version__,
                    cuda_runtime=t.version.cuda,device=b.device)
    if hasattr(b,'cache_state'):provenance['cache_at_load']=b.cache_state['actual']
    step=epoch=cursor=bad=0;best=None;order=np.random.default_rng(args.seed).permutation(len(data['rows']['train'])).tolist()
    if args.resume:
        path=args.run/'last.pt' if str(args.resume)=='last' else Path(args.resume)
        saved=load_checkpoint(path,t);old=dict(saved['config']);new=dict(config)
        old_resource=old.pop('saved_tensors_cpu');new_resource=new.pop('saved_tensors_cpu')
        if old!=new or (old_resource!=new_resource and not args.resume_allow_resource_change):raise ValueError('Resume scientific/data/resource configuration mismatch')
        if saved['provenance']!=provenance and not args.resume_allow_code_change:raise ValueError('Resume code/runtime change requires explicit recorded override')
        if old_resource!=new_resource or saved['provenance']!=provenance:
            atomic_json(args.run/f'resume-change-{saved["step"]}.json',dict(before=saved['config'],after=config,before_provenance=saved['provenance'],after_provenance=provenance,checkpoint_sha256=pilot.sha(path)))
        pilot.copy_lora_strict(b.model,saved['lora'],t);optimizer.load_state_dict(saved['optimizer'])
        lr_state.update(saved['lr_state']);scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        step,epoch,cursor,best,bad=[saved[k] for k in ('step','epoch','cursor','best','bad')];order=saved['permutation']
        restore_rng(saved['rng'],t,np,b.device)
    else:
        random.seed(args.seed);np.random.seed(args.seed);t.manual_seed(args.seed)
    atomic_json(args.run/'config.json',dict(config=config,provenance=provenance,data=data['manifest']))
    cache=AudioCache(args.audio_cache_mib,np);stop=[False];previous_handlers={}
    for sig in (signal.SIGINT,signal.SIGTERM):
        previous_handlers[sig]=signal.getsignal(sig);signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def snapshot():
        return dict(lora={n:p.detach().cpu().clone() for n,p in named},optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng(t,np,b.device),
            step=step,epoch=epoch,cursor=cursor,permutation=order.copy(),best=best,bad=bad,
            lr_state=lr_state.copy(),config=config,provenance=provenance,parameter_names=[n for n,_ in named])
    def evaluate():
        started=time.monotonic();score=validation(args,b,data,cache,step);loss_seconds=time.monotonic()-started
        probes=native_probes(args,b,step)
        with (args.run/'validation.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step,loss=score,loss_seconds=loss_seconds,total_seconds=time.monotonic()-started,
                         utc=pilot.dt.datetime.now(pilot.dt.timezone.utc).isoformat(),probes=probes))+'\n');stream.flush()
        return score
    last_save=time.monotonic();initial_hash=pilot.checksum_tensors(named,t);initial_step=step
    try:
        b.model.train()
        if hasattr(b.model,'audio_vae'):b.model.audio_vae.eval()
        if best is None:
            best=evaluate();save_checkpoint(args,b,snapshot(),True);last_save=time.monotonic()
        while step<args.max_steps and not stop[0]:
            started=time.monotonic();optimizer.zero_grad(set_to_none=True);total=0.;lengths=[]
            if b.device=='cuda':t.cuda.reset_peak_memory_stats()
            before=params[1 if len(params)>1 else 0].detach().flatten()[:128].clone()
            for _ in range(args.accumulation):
                if cursor==len(order):
                    epoch+=1;cursor=0;order=np.random.default_rng(args.seed+epoch).permutation(len(order)).tolist()
                row=data['rows']['train'][order[cursor]];batch,length=make_batch(row,b,cache);cursor+=1
                saving=t.autograd.graph.save_on_cpu(pin_memory=True) if args.saved_tensors_cpu and b.device=='cuda' else contextlib.nullcontext()
                with saving:
                    value=loss(batch,b);(value/args.accumulation).backward()
                total+=float(value.detach())/args.accumulation;lengths.append(length);del value,batch
            norm=t.nn.utils.clip_grad_norm_(params,1.)
            if not t.isfinite(norm):raise FloatingPointError('Nonfinite gradients; previous checkpoint preserved')
            optimizer.step();scheduler.step();step+=1
            if b.device=='cuda':t.cuda.synchronize()
            change=float((params[1 if len(params)>1 else 0].detach().flatten()[:128]-before).abs().max())
            emit(args,dict(event='step',step=step,epoch=epoch,cursor=cursor,loss=total,seconds=time.monotonic()-started,
                tokens=lengths,weight_delta=change,gradient_norm=float(norm),lr=optimizer.param_groups[0]['lr'],
                vram_peak_bytes=t.cuda.max_memory_allocated() if b.device=='cuda' else 0,
                vram_reserved_peak_bytes=t.cuda.max_memory_reserved() if b.device=='cuda' else 0,
                cpu_peak_rss_bytes=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss*1024))
            if step%args.validate_every==0:
                score=evaluate();improved=score<best-args.min_delta
                if improved:best=score;bad=0
                else:bad+=1
                save_checkpoint(args,b,snapshot(),improved);last_save=time.monotonic()
                cleanup_probes(args,read(args.run/'best.sha256.json',{}).get('step'))
                if bad>=args.patience:
                    if not lr_state['reductions']:
                        lr_state.update(multiplier=.3,reductions=1);bad=0
                        for group,base in zip(optimizer.param_groups,scheduler.base_lrs):group['lr']=base*min(1.,(scheduler.last_epoch+1)/args.warmup)*.3
                        scheduler._last_lr=[g['lr'] for g in optimizer.param_groups]
                        save_checkpoint(args,b,snapshot());last_save=time.monotonic()
                    else:emit(args,dict(event='plateau',step=step,bad=bad));break
            elif time.monotonic()-last_save>=args.checkpoint_seconds:
                save_checkpoint(args,b,snapshot());last_save=time.monotonic()
        save_checkpoint(args,b,snapshot())
        final_hash=pilot.checksum_tensors(named,t)
        atomic_json(args.run/'update-proof.json',dict(initial_step=initial_step,final_step=step,initial_sha256=initial_hash,final_sha256=final_hash,weights_changed=initial_hash!=final_hash))
        if step>initial_step and initial_hash==final_hash:raise RuntimeError('No actual LoRA update')
        emit(args,dict(event='stopped',step=step,signal=stop[0],checkpoint=str(args.run/'last.pt')))
    finally:
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)


def train(args,b,data):
    args.run.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (args.run/'training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _train_locked(args,b,data)


def bootstrap(args):
    parent=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().decode(errors='replace')
    if 'v2_gpu_worker' not in parent:raise RuntimeError('Training requires the exclusive GPU worker')
    budget();pilot.verify(args);pilot.guard_gpu(args)
    available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if available<13_000_000_000:raise RuntimeError('Need measured13GB available RAM for strict CPU staging of existing base/VAE')
    t,cls,lora_cls,_,_,packer_cls=pilot.imports(args)
    import numpy as np
    random.seed(args.seed);np.random.seed(args.seed);t.manual_seed(args.seed)
    state={}
    with capacity.capacity_hooks(1024,2,state):model=pilot.load_model(args,t,cls,lora_cls)
    count=pilot.enable_checkpointing(model,t)
    if not count:raise RuntimeError('No LM-layer activation checkpointing was enabled')
    packer=packer_cls(1,args.max_sequence,model.config.patch_size,model.config.feat_dim,model.audio_vae)
    return SimpleNamespace(t=t,np=np,model=model,packer=packer,device='cuda',max_sequence=args.max_sequence,
                           checkpointed_layers=count,cache_state=state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['plan','train'],default='plan');p.add_argument('--prepared',type=Path,default=PREPARED)
    p.add_argument('--source',type=Path,default=pilot.ROOT/'source');p.add_argument('--model',type=Path,default=pilot.ROOT/'model')
    p.add_argument('--run',type=Path,default=pilot.ROOT/'aegis-lora');p.add_argument('--gpu-authorized',action='store_true')
    p.add_argument('--rank',type=int,default=8);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--warmup',type=int,default=100)
    p.add_argument('--accumulation',type=int,default=4);p.add_argument('--seed',type=int,default=40733133)
    p.add_argument('--max-steps',type=int,default=20000);p.add_argument('--validate-every',type=int,default=200)
    p.add_argument('--checkpoint-seconds',type=float,default=1800);p.add_argument('--patience',type=int,default=8);p.add_argument('--min-delta',type=float,default=.001)
    p.add_argument('--max-sequence',type=int,default=1024);p.add_argument('--audio-cache-mib',type=int,default=256)
    p.add_argument('--saved-tensors-cpu',action='store_true');p.add_argument('--resume',nargs='?',const='last',type=str)
    p.add_argument('--resume-allow-resource-change',action='store_true');p.add_argument('--resume-allow-code-change',action='store_true')
    p.add_argument('--nfe',type=int,choices=[4,6,10],default=4);p.add_argument('--train-limit',type=int,default=0)
    p.add_argument('--validation-limit',type=int,default=0);p.add_argument('--probe-limit',type=int,default=0);p.add_argument('--pilot-longest',action='store_true')
    args=p.parse_args();os.umask(0o077)
    args.run=args.run.resolve()
    if any((directory/'.git').exists() for directory in (args.run,*args.run.parents)):
        p.error('Private training plans/checkpoints/audio must stay outside every Git checkout')
    if not all(math.isfinite(v) for v in (args.lr,args.min_delta,args.checkpoint_seconds)) or min(args.rank,args.lr,args.warmup,args.accumulation,args.max_steps,args.validate_every,args.checkpoint_seconds,args.patience,args.max_sequence)<=0 or min(args.audio_cache_mib,args.train_limit,args.validation_limit,args.probe_limit,args.min_delta)<0:
        p.error('Positive training cadence/resources and nonnegative limits required')
    data=load_data(args)
    if args.action=='plan':
        plan=dict(status='prepared_only',counts={s:len(v) for s,v in data['rows'].items()},seconds={s:sum(r['duration'] for r in v) for s,v in data['rows'].items()},
            longest={s:dict(id=max(v,key=lambda r:r['duration'])['id'],seconds=max(r['duration'] for r in v)) for s,v in data['rows'].items()},
            data=data['manifest'],data_fingerprint=data['fingerprint'],code_sha256=pilot.sha(__file__),source_rev=pilot.SOURCE_REV,model_rev=pilot.MODEL_REV,
            hypothesis='Aegis LoRA learns robotic identity while preserving qualified PT-BR; validation audio/ASR and identity decide advancement',
            future_gates=['Longest whole-utterance GPU pilot','Three real updates/checkpoint/fresh resume4 + native audio','Only then full1025/55 specialization'],
            checkpoint='FP32 LoRA + complete Adam/scheduler/scaler/RNG/permutation/cursor; atomic validated last/best, SHA sidecars',
            input_policy=data['manifest']['audio_policy'],cache_mib=args.audio_cache_mib,working_peak_extra_bytes=RESERVE,heldout_test='Not loaded or synthesized')
        args.run.mkdir(parents=True,exist_ok=True,mode=0o700);atomic_json(args.run/'plan.json',plan)
        print(json.dumps({k:v for k,v in plan.items() if k!='data'},indent=2));return
    train(args,bootstrap(args),data)


if __name__=='__main__':main()
