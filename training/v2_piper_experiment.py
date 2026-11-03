"""Private Piper Cadu full G+D specialization; never imported by production.

Explicit local research only. New stage loads both pretrained networks and
starts fresh Adam/schedulers; own resume restores the complete paired state.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace,ModuleType
import urllib.request

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/var/tmp/aegis-v2-work/piper')
DATA=Path('/var/tmp/aegisvoice-v2-training-20261102/prepared')
BASE_SHA='abd3e6c7237aca5b7cf9c3cfc50945bfef44a83a594e00186f584d9d11fd355a'
BASE_BYTES=845888150
BASE_REV='f0a4c43d31b4e533feac35ae3ac25a56eebab8e8'
BASE_PATH='pt/pt_BR/cadu/medium/epoch=5195-step=109116.ckpt'
CONFIG_SHA='9433ddafa54e7d5c973d6160a29c046e39d65a2dc8892305591c7b93656c2253'
MAS_SHA='471d99b9c4de3da39d1b002c1422bae7a8991063983f827131e089dc142550b9'
TAG='639388b6317fc4731e91d53da42aea68fd4166ff'

def utc():return dt.datetime.now(dt.timezone.utc).isoformat()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def sync_directory(path):
    descriptor=os.open(path,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)

def atomic_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path);sync_directory(path.parent)

def private(path):
    path=Path(path).resolve()
    if any((p/'.git').exists() for p in (path,*path.parents)):raise ValueError('private output must be outside Git')
    path.mkdir(parents=True,exist_ok=True,mode=0o700);return path

def usage():
    from training.v2_language_monitor import physical_usage
    roots=sorted(Path('/var/tmp').glob('aegis*'))+[REPO,Path('/home/phc/.cache'),Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip')]
    return physical_usage(roots)

def budget(path,extra=0,reserve=550_000_000):
    current=usage();free=shutil.disk_usage(path).free
    if current['permission_errors']:raise RuntimeError('cannot account for all project roots')
    if current['allocated_bytes']+extra+reserve>40_000_000_000:raise RuntimeError('project40GB peak exceeded')
    if free-extra-reserve<5*1024**3:raise RuntimeError('volume reserve below5GiB')
    return dict(project=current['allocated_bytes'],extra=extra,reserve=reserve,volume_free=free)

def fetch(a):
    target=ROOT/'downloads/pt_BR-cadu-medium.ckpt';private(target.parent)
    with (target.parent/'cadu-fetch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if target.exists():
            if target.stat().st_size!=BASE_BYTES or sha(target)!=BASE_SHA:raise ValueError('existing Cadu checkpoint mismatch')
            return dict(path=str(target),verified=True,existing=True)
        before=budget(target.parent,BASE_BYTES)
        temp=target.with_name(target.name+'.part')
        if temp.exists():raise ValueError('inspect existing partial download before retry')
        url='https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/'+BASE_REV+'/'+BASE_PATH
        h=hashlib.sha256();count=0;reported=0;started=time.monotonic()
        request=urllib.request.Request(url,headers={'Accept-Encoding':'identity'})
        with urllib.request.urlopen(request,timeout=60) as response,temp.open('xb') as f:
            while block:=response.read(8*1024**2):
                count+=len(block)
                if count>BASE_BYTES:raise ValueError('download exceeded pinned size')
                f.write(block);h.update(block)
                if count-reported>=128*1024**2:
                    if shutil.disk_usage(target.parent).free<5*1024**3+550_000_000:raise RuntimeError('volume reserve at risk')
                    atomic_json(ROOT/'full-download-progress.json',dict(bytes=count,total=BASE_BYTES,utc=utc()))
                    print(json.dumps(dict(download_bytes=count,total=BASE_BYTES)),flush=True);reported=count
            f.flush();os.fsync(f.fileno())
        if count!=BASE_BYTES or h.hexdigest()!=BASE_SHA or sha(temp)!=BASE_SHA:raise ValueError('full download SHA/length verification failed')
        os.replace(temp,target);sync_directory(target.parent)
        receipt=dict(path=str(target),bytes=count,sha256=BASE_SHA,revision=BASE_REV,url=url,verified=True,before=before,
                     seconds=time.monotonic()-started,utc=utc(),license_scope='private local research; no upload; Lessac lineage unresolved')
        atomic_json(ROOT/'full-download-receipt.json',receipt);return receipt

def config(a):
    if sha(a.voice_config)!=CONFIG_SHA:raise ValueError('Cadu configuration SHA mismatch')
    value=json.loads(a.voice_config.read_text(encoding='utf-8'));m=value['phoneme_id_map']
    if len(m)!=161 or m.get('\u0303')!=[141] or value['num_symbols']!=256 or value['num_speakers']!=1 or value['audio']['sample_rate']!=22050 or value['espeak']['voice']!='pt-br':
        raise ValueError('Cadu original161phonemes/nasal/architecture required')
    return value

def read_rows(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def prepare(a):
    """Separate process: native eSpeak never initializes inside Torch trainer."""
    from training.frontend_check import check_inventory
    sys.path.insert(0,str(a.overlay))
    from piper.phonemize_espeak import EspeakPhonemizer,ESPEAK_DATA_DIR
    from piper.phoneme_ids import phonemes_to_ids
    cfg=config(a);frontend=EspeakPhonemizer(ESPEAK_DATA_DIR);result={};bindings={};ids=set();seen={k:{} for k in ('group','audio','text')}
    for split in ('train','validation'):
        path=a.prepared/(split+'.jsonl');bindings[split]=sha(path);rows=read_rows(path)
        if not rows:raise ValueError('empty partition')
        for row in rows:
            ident=row['id'];text=row['text']
            if ident in ids:raise ValueError('duplicate stable ID')
            ids.add(ident)
            for key,value in [('group',row.get('group_id',ident)),('audio',row['sha256']),('text',' '.join(text.casefold().split()))]:
                previous=seen[key].setdefault(value,split)
                if previous!=split:raise ValueError('train/validation leakage: '+key)
            if sha(row['audio_filepath'])!=row['sha256']:raise ValueError('prepared WAV SHA mismatch: '+ident)
        phones={r['text']:frontend.phonemize('pt-br',r['text'],vowel_clusters=None) for r in rows}
        report=check_inventory(rows,cfg,lambda t:phones[t],lambda sentence,m:phonemes_to_ids(sentence,id_map=m))
        if not report['passed']:raise ValueError('unknown training phonemes: '+repr(report['unknown_by_id']))
        for row in rows:row['_ids']=report['phoneme_ids_by_id'][row['id']]
        result[split]=rows
    corpus=REPO/'benchmarks/v2/validation.pt-BR.json';cases=json.loads(corpus.read_text(encoding='utf-8'))
    for case in cases:
        sentences=frontend.phonemize('pt-br',case['text'],vowel_clusters=None);case['_sentences']=[]
        for sentence in sentences:
            if not sentence or set(sentence)-set(cfg['phoneme_id_map']):raise ValueError('probe OOV/empty sentence')
            case['_sentences'].append(phonemes_to_ids(sentence,id_map=cfg['phoneme_id_map']))
    bindings.update(corpus=sha(corpus),voice_config=CONFIG_SHA,frontend_code=sha(REPO/'training/frontend_check.py'),
                    phonemizer=sha(a.overlay/'piper/phonemize_espeak.py'),phoneme_ids=sha(a.overlay/'piper/phoneme_ids.py'))
    value=dict(rows=result,probes=cases,bindings=bindings,policy='native PTBR map161; vowel_clusters=None; strict OOV; preparedWAV→22050RAM; finaltest never read',
               fingerprint=hashlib.sha256(json.dumps(bindings,sort_keys=True).encode()).hexdigest(),created_utc=utc())
    atomic_json(a.data_manifest,value);return dict(path=str(a.data_manifest),fingerprint=value['fingerprint'],counts={s:len(r) for s,r in result.items()})

def ensure_data(a):
    if not a.data_manifest.exists():
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        subprocess.run([sys.executable,'-m','training.v2_piper_experiment','prepare','--prepared',str(a.prepared),'--overlay',str(a.overlay),
                        '--voice-config',str(a.voice_config),'--data-manifest',str(a.data_manifest),'--run',str(a.run)],cwd=REPO,env=env,check=True)
    value=json.loads(a.data_manifest.read_text(encoding='utf-8'))
    expected=dict(train=sha(a.prepared/'train.jsonl'),validation=sha(a.prepared/'validation.jsonl'),
                  corpus=sha(REPO/'benchmarks/v2/validation.pt-BR.json'),voice_config=CONFIG_SHA,
                  frontend_code=sha(REPO/'training/frontend_check.py'),phonemizer=sha(a.overlay/'piper/phonemize_espeak.py'),phoneme_ids=sha(a.overlay/'piper/phoneme_ids.py'))
    if value['bindings']!=expected:raise ValueError('stale precomputed frontend; use new data-manifest')
    fingerprint=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
    if value['fingerprint']!=fingerprint:raise ValueError('data fingerprint mismatch')
    return value

def install_mas(torch,np):
    path=Path('/var/tmp/aegis-v2-work/vits/deps/monotonic_alignment_search/core.cpython-312-x86_64-linux-gnu.so')
    if sha(path)!=MAS_SHA:raise ValueError('MAS binary pin mismatch')
    spec=importlib.util.spec_from_file_location('_aegis_piper_mas.core',path);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    module=ModuleType('piper.train.vits.monotonic_align')
    def maximum_path(scores,mask):
        frames=mask.sum(1)[:,0].long();tokens=mask.sum(2)[:,0].long()
        if bool((tokens<1).any()) or bool((frames<tokens).any()) or not bool(torch.isfinite(scores).all()):raise ValueError('invalid MAS dimensions/scores')
        values=np.ascontiguousarray(scores.detach().float().transpose(1,2).cpu().numpy())
        paths=np.zeros(values.shape,np.int32)
        core.maximum_path_c(paths,values,tokens.cpu().numpy().astype(np.int32),frames.cpu().numpy().astype(np.int32))
        return torch.from_numpy(paths).transpose(1,2).to(device=scores.device,dtype=scores.dtype)
    module.maximum_path=maximum_path;sys.modules[module.__name__]=module

def strict_copy(model,state,torch):
    expected=model.state_dict()
    if set(state)!=set(expected):raise ValueError('strict network key set mismatch')
    for key,value in state.items():
        if value.shape!=expected[key].shape or value.dtype!=expected[key].dtype or not bool(torch.isfinite(value).all()):raise ValueError('network shape/dtype/nonfinite mismatch: '+key)
    model.load_state_dict(state,strict=True)

def bootstrap(a):
    if a.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    elif not a.gpu_authorized:raise ValueError('CUDA requires parent exclusiveworker authorization')
    else:
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if occupied:raise RuntimeError('GPU already occupied; no concurrent backend allowed')
    sys.path.insert(0,str(a.overlay));import numpy as np;import torch
    torch.set_num_threads(a.threads);config(a)
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    if available<3*1024**3:raise RuntimeError('CPU strict loader requires3GiB availableRAM margin')
    source=a.overlay/'piper/train/vits'
    for name,expected in {'models.py':'a27efdfeb4ccccd2c68037741b15ea4bb56b61615d33c27561b962ae6e4bff03','lightning.py':'360da7954710c6a8e7593399995f9e29d42cb166b7b10ea2f0a2f45b30937f7c'}.items():
        if sha(source/name)!=expected:raise ValueError('Piper source tag mismatch')
    source_hashes={str(p.relative_to(a.overlay)):sha(p) for p in sorted(source.rglob('*.py')) if '__pycache__' not in p.parts}
    install_mas(torch,np)
    from piper.train.vits.models import SynthesizerTrn,MultiPeriodDiscriminator
    from piper.train.vits import commons,losses,mel_processing
    from training.v2_fastpitch_benchmark import load_bound
    saved=None;own=a.resume or a.checkpoint
    if own:
        saved=load_own(own,SimpleNamespace(torch=torch));hp=saved['model_hparams'];gs,ds=saved['g'],saved['d'];provenance=saved['config']['base']
        if provenance['source_hashes']!=source_hashes or provenance['torch']!=torch.__version__ or provenance['base_sha256']!=BASE_SHA:raise ValueError('own checkpoint source/runtime/base mismatch')
    else:
        if a.base.stat().st_size!=BASE_BYTES:raise ValueError('base checkpoint length mismatch')
        original=load_bound(a.base,BASE_SHA,torch);hp=original['hyper_parameters'];state=original['state_dict']
        gs={k.removeprefix('model_g.'):v for k,v in state.items() if k.startswith('model_g.')};ds={k.removeprefix('model_d.'):v for k,v in state.items() if k.startswith('model_d.')}
        if len(gs)+len(ds)!=len(state):raise ValueError('unknown full checkpoint network prefix')
        provenance=dict(base_sha256=BASE_SHA,base_epoch=original['epoch'],base_global_step=original['global_step'],original_lightning=original['pytorch-lightning_version'],
                        source_tag=TAG,source_hashes=source_hashes,g_tensors=len(gs),d_tensors=len(ds),config_sha256=CONFIG_SHA,mas_sha256=MAS_SHA,torch=torch.__version__)
    if len(gs)+len(ds)!=784 or hp['num_symbols']!=256 or hp['num_speakers']!=1 or hp['sample_rate']!=22050:raise ValueError('Cadu full checkpoint architecture/keycount mismatch')
    keys=('inter_channels','hidden_channels','filter_channels','n_heads','n_layers','kernel_size','p_dropout','resblock',
          'resblock_kernel_sizes','resblock_dilation_sizes','upsample_rates','upsample_initial_channel','upsample_kernel_sizes','gin_channels','use_sdp')
    g=SynthesizerTrn(n_vocab=256,spec_channels=513,segment_size=32,n_speakers=1,**{k:hp[k] for k in keys})
    d=MultiPeriodDiscriminator(use_spectral_norm=hp['use_spectral_norm'])
    strict_copy(g,gs,torch);strict_copy(d,ds,torch)
    provenance.update(g_parameters=sum(p.numel() for p in g.parameters()),d_parameters=sum(p.numel() for p in d.parameters()))
    if not own:del original,state
    del gs,ds
    g.to(a.device);d.to(a.device)
    for network in (g,d):
        for module in network.modules():
            for hook in module._forward_pre_hooks.values():
                if hook.__class__.__name__=='WeightNorm':
                    with torch.no_grad():hook(module,None)
        if any(p.device.type!=a.device for p in network.parameters()) or any(b.device.type!=a.device for b in network.buffers()):raise RuntimeError('network not resident on selecteddevice')
    return SimpleNamespace(torch=torch,np=np,g=g,d=d,hp=hp,commons=commons,losses=losses,mel=mel_processing,provenance=provenance,device=a.device,saved_state=saved)

def rng(torch,np,device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if device=='cuda' else [])

def restore_rng(value,torch,np,device):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
    if device=='cuda':torch.cuda.set_rng_state_all(value['cuda'])

@contextmanager
def preserve_rng(b,seed=None):
    state=rng(b.torch,b.np,b.device)
    try:
        if seed is not None:b.torch.manual_seed(seed);b.np.random.seed(seed);random.seed(seed)
        yield
    finally:restore_rng(state,b.torch,b.np,b.device)

class AudioCache:
    def __init__(self,b,limit_mib):self.b=b;self.limit=limit_mib*1024**2;self.items=OrderedDict();self.bytes=0
    def get(self,row):
        key=(row['id'],row['sha256'])
        if key in self.items:self.items.move_to_end(key);return self.items[key]
        import soundfile as sf
        from scipy.signal import resample_poly
        t,np=self.b.torch,self.b.np
        if sha(row['audio_filepath'])!=row['sha256']:raise ValueError('prepared input WAV changed: '+row['id'])
        audio,rate=sf.read(row['audio_filepath'],dtype='float32')
        if audio.ndim!=1 or not audio.size or not np.isfinite(audio).all():raise ValueError('invalid inputaudio')
        if rate!=22050:
            divisor=math.gcd(rate,22050);audio=resample_poly(audio,22050//divisor,rate//divisor,window=('kaiser',10)).astype(np.float32)
        if np.abs(audio).max()>1:raise ValueError('input resample overshoot; no clipping')
        wave=t.from_numpy(audio.copy())
        spec=self.b.mel.spectrogram_torch(wave[None],1024,22050,256,1024,center=False)[0]
        if spec.shape[1]<len(row['_ids']) or spec.shape[1]<32:raise ValueError('audio too short for MAS/32frame segment')
        value=(wave,spec);size=sum(x.numel()*x.element_size() for x in value)
        while self.items and self.bytes+size>self.limit:
            _,removed=self.items.popitem(last=False);self.bytes-=sum(x.numel()*x.element_size() for x in removed)
        if size<=self.limit:self.items[key]=value;self.bytes+=size
        return value

def make_batch(rows,b,cache):
    t=b.torch;values=[(row,*cache.get(row)) for row in rows];values.sort(key=lambda v:v[2].shape[1],reverse=True)
    n=len(values);ids=t.zeros(n,max(len(r['_ids']) for r,_,_ in values),dtype=t.long)
    wave=t.zeros(n,1,max(8192,max(len(w) for _,w,_ in values)));spec=t.zeros(n,513,max(s.shape[1] for _,_,s in values))
    for i,(row,w,s) in enumerate(values):ids[i,:len(row['_ids'])]=t.tensor(row['_ids']);wave[i,0,:len(w)]=w;spec[i,:,:s.shape[1]]=s
    return SimpleNamespace(x=ids.to(b.device),xl=t.tensor([len(r['_ids']) for r,_,_ in values],device=b.device),wave=wave.to(b.device),
                           spec=spec.to(b.device),sl=t.tensor([s.shape[1] for _,_,s in values],device=b.device))

def acoustic_forward(b,batch):
    yhat,duration,_,offsets,_,mask,(_,zp,mp,lp,_,lq)=b.g(batch.x,batch.xl,batch.spec,batch.sl)
    targetmel=b.mel.spec_to_mel_torch(batch.spec.float(),1024,80,22050,0.,None)
    targetmel=b.commons.slice_segments(targetmel,offsets,32)
    predictedmel=b.mel.mel_spectrogram_torch(yhat.squeeze(1).float(),1024,80,22050,256,1024,0.,None)
    y=b.commons.slice_segments(batch.wave,offsets*256,8192);yhat=yhat[...,:y.shape[-1]]
    mel=b.torch.nn.functional.l1_loss(targetmel,predictedmel)
    kl=b.losses.kl_loss(zp,lq,mp,lp,mask)
    return y,yhat,mel,duration.float().sum(),kl

def compute_loss(b,batch):
    y,yhat,mel,duration,kl=acoustic_forward(b,batch)
    _,fake,realmap,fakemap=b.d(y,yhat)
    generator=b.losses.generator_loss(fake)[0]+b.losses.feature_loss(realmap,fakemap)+45*mel+duration+kl
    real,fake,_,_=b.d(y,yhat.detach());critic=b.losses.discriminator_loss(real,fake)[0]
    return generator,critic,dict(mel=float(mel.detach()),duration=float(duration.detach()),kl=float(kl.detach()))

def pair_update(b,optimizers,batches,loss_function=None):
    loss_function=loss_function or compute_loss
    for optimizer in optimizers:optimizer.zero_grad(set_to_none=True)
    metrics=[]
    for batch in batches:
        generator,critic,values=loss_function(b,batch)
        if not bool(b.torch.isfinite(generator)) or not bool(b.torch.isfinite(critic)):raise FloatingPointError('nonfinite loss')
        b.torch.autograd.backward(generator/len(batches),inputs=list(b.g.parameters()))
        b.torch.autograd.backward(critic/len(batches),inputs=list(b.d.parameters()));metrics.append(values)
    for network in (b.g,b.d):
        if any(p.grad is not None and not bool(b.torch.isfinite(p.grad).all()) for p in network.parameters()):raise FloatingPointError('nonfinite accumulated gradients')
    for optimizer in optimizers:optimizer.step()
    for network in (b.g,b.d):
        if any(not bool(b.torch.isfinite(p).all()) for p in network.parameters()):raise FloatingPointError('nonfinite updated weights; discard unsaved pair')
    return {k:sum(v[k] for v in metrics)/len(metrics) for k in metrics[0]}

def synthesize(b,sentences,seed=40733133,max_seconds=60.):
    t=b.torch;original=b.commons.sequence_mask;maximum=int(max_seconds*22050/256);parts=[];training=b.g.training
    def bounded(lengths,max_length=None):
        if bool((lengths<1).any()) or bool((lengths>maximum).any()):raise ValueError('predicted duration exceeds audio bounds; no truncation')
        return original(lengths,max_length)
    with preserve_rng(b,seed),t.inference_mode():
        b.g.eval();b.commons.sequence_mask=bounded
        try:
            for ids in sentences:
                token=t.tensor(ids,dtype=t.long,device=b.device)[None]
                audio=b.g.infer(token,t.tensor([token.shape[1]],device=b.device),noise_scale=0.,noise_scale_w=0.,length_scale=1.)[0]
                parts.append(audio.squeeze().float().cpu().numpy())
            result=b.np.concatenate(parts)
            if not result.size or not b.np.isfinite(result).all() or len(result)>max_seconds*22050:raise ValueError('invalid/bounded native inference')
            return result
        finally:b.commons.sequence_mask=original;b.g.train(training)

def audio_artifact(path,values,b):
    import soundfile as sf
    from training.prepare_dataset import encode_pcm16,resample_pcm
    from training.v2_fastpitch_benchmark import write_pcm,metrics_module
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700);raw=path.with_name(path.stem+'-raw22050.wav')
    sf.write(raw,values,22050,subtype='FLOAT')
    result=dict(raw=str(raw),raw_sha256=sha(raw),native_samples=len(values),native_peak=float(b.np.abs(values).max()),native_outside=int((b.np.abs(values)>1).sum()))
    try:
        final=resample_pcm(values,22050,24000);pcm=encode_pcm16(final);write_pcm(path,pcm)
        qc=metrics_module().pcm_metrics(pcm,[dict(ms=0.,bytes=len(pcm))])
        speech=qc['first_speech_pcm_available_ms'] is not None
        for key in ('first_pcm_ms','first_speech_pcm_available_ms','ttfa_estimated_playback_ms','ttfa_margin_playback_ms','underruns_over_10ms','underrun_total_ms','underrun_max_ms','underrun_events'):qc.pop(key,None)
        result.update(outcome='ok' if speech else 'no_speech',wav=str(path),wav_sha256=sha(path),pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                      timing='onset audio offsets/QC only; no synthesis latency or streaming measurement',**qc)
    except ValueError as error:result.update(outcome='failed',error=str(error))
    return result

def validation(b,rows,cache,limit):
    values=[];training=b.g.training
    with preserve_rng(b,40733133),b.torch.inference_mode():
        b.g.eval()
        try:
            for row in rows[:limit]:
                value=float(acoustic_forward(b,make_batch([row],b,cache))[2])
                if not math.isfinite(value):raise FloatingPointError('nonfinite validationmel; preserve previous checkpoint')
                values.append(value)
        finally:b.g.train(training)
    return sum(values)/len(values)

def probes(a,b,data,label,step):
    directory=a.run/'probes'/label;private(directory)
    spoken=json.loads((REPO/'benchmarks/v2/spoken-validation.pt-BR.json').read_text(encoding='utf-8'))['cases'];rows=[];jobs=[]
    for case in data['probes']:
        try:row=audio_artifact(directory/(case['id']+'.wav'),synthesize(b,case['_sentences']),b)
        except Exception as error:row=dict(outcome='failed',error=repr(error))
        row.update(case=case['id'],source_text=case['text'],effective_text=case['text'],step=step,frontend='native',sampling=dict(noise_scale=0.,noise_w=0.,length_scale=1.),timing='training probe only; not productionbenchmark')
        rows.append(row)
        if row.get('wav'):jobs.append(dict(candidate='piper-cadu-target',case=case['id'],cohort='validation-rep0',wav=row['wav'],expected_verbatim=case['text'],expected_spoken=spoken[case['id']],expected_effective=case['text'],generation_outcome=row['outcome']))
    atomic_json(directory/'results.json',rows);atomic_json(directory/'asr-jobs.json',dict(jobs=jobs));return rows

def snapshot(a,b,data,step,best=False):
    label=f'step-{step:08d}';directory=a.run/'probes'/label
    if not (directory/'results.json').exists():probes(a,b,data,label,step)
    pointer=dict(step=step,directory=str(directory),checkpoint_sha256=sha(a.run/'last.pt'))
    atomic_json(a.run/'last-probes.json',pointer)
    if best:atomic_json(a.run/'best-probes.json',pointer)
    keep={label}
    if (a.run/'best-probes.json').exists():keep.add(Path(json.loads((a.run/'best-probes.json').read_text())['directory']).name)
    directories=sorted((a.run/'probes').glob('step-*'))
    keep.update(p.name for p in directories[-6:])
    for path in directories:
        if path.name not in keep:
            if path.parent!=(a.run/'probes') or path.is_symlink():raise RuntimeError('unexpected probe cleanup path')
            shutil.rmtree(path)

def audit_state(state,torch):
    required={'g','d','model_hparams','optimizers','optimizer_names','schedulers','scaler','rng','step','epoch','cursor','permutation','config','best','stale','reduced'}
    if not required<=set(state) or len(state['optimizers'])!=2 or len(state['schedulers'])!=2:raise ValueError('incomplete paired checkpoint')
    def finite(value):
        if torch.is_tensor(value):
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):raise ValueError('nonfinite checkpoint tensor')
        elif isinstance(value,dict):
            for x in value.values():finite(x)
        elif isinstance(value,(list,tuple)):
            for x in value:finite(x)
        elif isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite checkpoint scalar')
    finite(state)
    if state['step']<0 or state['epoch']<0 or not 0<=state['cursor']<=len(state['permutation']):raise ValueError('invalid paired step/epoch/cursor')
    if state['permutation'] and sorted(state['permutation'])!=list(range(state['config']['train_count'])):raise ValueError('invalid sampler permutation')
    if not {'python','numpy','torch','cuda'}<=set(state['rng']):raise ValueError('incomplete RNG state')
    for network,optimizer,names in zip((state['g'],state['d']),state['optimizers'],state['optimizer_names']):
        ids=[i for group in optimizer['param_groups'] for i in group['params']]
        if len(ids)!=len(names) or set(names)!=set(network):raise ValueError('Adam parameter name binding mismatch')
        if state['step'] and set(ids)!=set(optimizer['state']):raise ValueError('incomplete Adam parameter coverage')
        for identifier,name in zip(ids,names):
            if identifier not in optimizer['state']:continue
            value=optimizer['state'][identifier]
            if not {'step','exp_avg','exp_avg_sq'}<=set(value):raise ValueError('missing Adam moments')
            for key in ('exp_avg','exp_avg_sq'):
                if value[key].shape!=network[name].shape or value[key].dtype!=network[name].dtype:raise ValueError('Adam moment shape/dtype mismatch')
            if float(value['step'])!=state['step']:raise ValueError('Adam step differs from completed G+D pair counter')
    return state

def save_checkpoint(a,b,state,best=False):
    target=a.run/'last.pt';temp=a.run/'last.pt.tmp';budget(a.run,max(BASE_BYTES,target.stat().st_size if target.exists() else 0)+16*1024**2)
    with temp.open('wb') as f:b.torch.save(state,f);f.flush();os.fsync(f.fileno())
    checked=b.torch.load(temp,map_location='cpu',weights_only=False);audit_state(checked,b.torch);del checked
    digest=sha(temp)
    receipt=dict(sha256=digest,bytes=temp.stat().st_size,step=state['step'],epoch=state['epoch'],cursor=state['cursor'],utc=utc())
    atomic_json(a.run/'last.pt.pending.json',receipt)
    os.replace(temp,target);sync_directory(a.run)
    atomic_json(target.with_suffix('.pt.json'),receipt)
    (a.run/'last.pt.pending.json').unlink();sync_directory(a.run)
    if best:
        link=a.run/'best.pt.tmp';link.unlink(missing_ok=True);os.link(target,link)
        atomic_json(a.run/'best.pt.pending.json',receipt);os.replace(link,a.run/'best.pt');sync_directory(a.run)
        atomic_json(a.run/'best.pt.json',receipt);(a.run/'best.pt.pending.json').unlink();sync_directory(a.run)
    return receipt

def load_own(path,b):
    from training.v2_fastpitch_benchmark import load_bound
    receipt_path=path.with_suffix(path.suffix+'.json');pending=path.with_suffix(path.suffix+'.pending.json')
    receipt=json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else None
    actual=sha(path)
    if receipt is None or actual!=receipt['sha256']:
        candidate=json.loads(pending.read_text(encoding='utf-8')) if pending.exists() else None
        if not candidate or actual!=candidate['sha256']:raise ValueError('checkpoint differs from receipt/pending publication')
        value=audit_state(load_bound(path,candidate['sha256'],b.torch),b.torch)
        atomic_json(receipt_path,candidate);pending.unlink();sync_directory(path.parent);return value
    return audit_state(load_bound(path,receipt['sha256'],b.torch),b.torch)

def emit(a,kind,**values):
    from training.v2_vits import event
    event(a.run,kind,**values)

def train(a,b,data):
    existing={}
    for path in (a.run/'last.pt',a.run/'best.pt'):
        if path.exists():existing[(path.stat().st_dev,path.stat().st_ino)]=path.stat().st_size
    bare=sum(v.numel()*v.element_size() for network in (b.g,b.d) for v in network.state_dict().values())+2*1024**2
    predicted=2*BASE_BYTES+bare if a.action=='pilot' else 3*BASE_BYTES
    budget(a.run,max(0,predicted-sum(existing.values()))+16*1024**2)
    t,np=b.torch,b.np;random.seed(a.seed);np.random.seed(a.seed);t.manual_seed(a.seed)
    optimizers=[t.optim.AdamW(b.g.parameters(),lr=a.lr_gen,betas=(.8,.99),eps=1e-9),t.optim.AdamW(b.d.parameters(),lr=a.lr_disc,betas=(.5,.9),eps=1e-9)]
    schedulers=[t.optim.lr_scheduler.ExponentialLR(optimizers[0],gamma=.999875),t.optim.lr_scheduler.ExponentialLR(optimizers[1],gamma=.9999)]
    scaler=t.amp.GradScaler('cuda',enabled=False)
    cfg=dict(schema='aegis.piper.paired.v1',base=b.provenance,data_fingerprint=data['fingerprint'],data_manifest_sha256=sha(a.data_manifest),code_sha256=sha(__file__),
             batch_size=a.batch_size,accumulation=a.accumulation,precision='FP32',device=b.device,seed=a.seed,lr_gen=a.lr_gen,lr_disc=a.lr_disc,
             val_limit=a.val_limit,patience=a.patience,min_delta=a.min_delta,frontend='native161',initialization='strict pretrainedG+D, fresh Adam/schedulers/RNG/counters',license_scope='private research only; no redistribution',
             train_count=len(data['rows']['train']),validation_count=len(data['rows']['validation']),
             scheduler='once/completedepoch; Ggamma.999875/Dgamma.9999; plateauonefactor.3')
    step=epoch=cursor=stale=0;permutation=[];best=math.inf;reduced=False
    if a.resume:
        state=b.saved_state if b.saved_state is not None else load_own(a.resume,b)
        if state['config']!=cfg:raise ValueError('resume config/data/source mismatch')
        strict_copy(b.g,state['g'],t);strict_copy(b.d,state['d'],t)
        for optimizer,saved in zip(optimizers,state['optimizers']):optimizer.load_state_dict(saved)
        for scheduler,saved in zip(schedulers,state['schedulers']):scheduler.load_state_dict(saved)
        scaler.load_state_dict(state['scaler']);step,epoch,cursor=state['step'],state['epoch'],state['cursor'];permutation=state['permutation'];best,stale,reduced=state['best'],state['stale'],state['reduced']
        restore_rng(state['rng'],t,np,b.device);b.saved_state=None;del state;emit(a,'resumed',step=step,epoch=epoch,cursor=cursor)
    elif (a.run/'recipe.json').exists():raise ValueError('existing run requires explicit resume')
    atomic_json(a.run/'recipe.json',cfg);cache=AudioCache(b,a.cache_mib);b.g.train();b.d.train();stop=False;lastsave=time.monotonic()
    def signal_stop(sig,frame):
        nonlocal stop
        stop=True
    old={sig:signal.signal(sig,signal_stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    def state():return dict(g=b.g.state_dict(),d=b.d.state_dict(),model_hparams=b.hp,optimizers=[o.state_dict() for o in optimizers],optimizer_names=[[n for n,_ in network.named_parameters()] for network in (b.g,b.d)],schedulers=[s.state_dict() for s in schedulers],scaler=scaler.state_dict(),
                           rng=rng(t,np,b.device),step=step,epoch=epoch,cursor=cursor,permutation=permutation,config=cfg,best=best,stale=stale,reduced=reduced)
    if not a.resume:
        initial=validation(b,data['rows']['validation'],cache,a.val_limit);best=initial
        receipt=save_checkpoint(a,b,state(),True);emit(a,'initial-validation',step=0,mel=initial,checkpoint=receipt)
        probes(a,b,data,'initial',0)
        pointer=dict(step=0,directory=str(a.run/'probes/initial'),checkpoint_sha256=receipt['sha256'])
        atomic_json(a.run/'best-probes.json',pointer);atomic_json(a.run/'last-probes.json',pointer)
    try:
        while step<a.max_updates and not stop:
            if cursor>=len(permutation):permutation=list(range(len(data['rows']['train'])));random.shuffle(permutation);cursor=0
            end=min(cursor+a.batch_size*a.accumulation,len(permutation));selected=permutation[cursor:end]
            started=time.monotonic()
            batches=[make_batch([data['rows']['train'][i] for i in selected[j:j+a.batch_size]],b,cache) for j in range(0,len(selected),a.batch_size)]
            g0=next(b.g.parameters()).detach().clone();d0=next(b.d.parameters()).detach().clone()
            metrics=pair_update(b,optimizers,batches);step+=1;cursor=end
            if b.device=='cuda':t.cuda.synchronize()
            metrics.update(delta_g=float((next(b.g.parameters())-g0).abs().max()),delta_d=float((next(b.d.parameters())-d0).abs().max()))
            emit(a,'update',step=step,epoch=epoch,cursor=cursor,seconds=time.monotonic()-started,**metrics)
            complete=cursor==len(permutation)
            if complete:
                epoch+=1
                for scheduler in schedulers:scheduler.step()
            if complete or step==a.max_updates:
                value=validation(b,data['rows']['validation'],cache,a.val_limit);improved=value<best*(1-a.min_delta)
                if improved:best=value;stale=0
                else:stale+=1
                if stale>=a.patience:
                    if reduced:stop=True
                    else:
                        for optimizer in optimizers:
                            for group in optimizer.param_groups:group['lr']*=.3
                        reduced=True;stale=0;emit(a,'lr-reduced',step=step,factor=.3)
                receipt=save_checkpoint(a,b,state(),improved);lastsave=time.monotonic();emit(a,'validation',step=step,epoch=epoch,mel=value,improved=improved,checkpoint=receipt)
                snapshot(a,b,data,step,improved)
            elif time.monotonic()-lastsave>=a.checkpoint_seconds:save_checkpoint(a,b,state());lastsave=time.monotonic()
        if stop:save_checkpoint(a,b,state());snapshot(a,b,data,step)
        emit(a,'completed',step=step,epoch=epoch,cursor=cursor,stopped=stop,peak_allocated_mib=t.cuda.max_memory_allocated()/2**20 if b.device=='cuda' else None)
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['fetch','prepare','check','reconstruct','pilot','train','infer'])
    p.add_argument('--base',type=Path,default=ROOT/'downloads/pt_BR-cadu-medium.ckpt');p.add_argument('--overlay',type=Path,default=ROOT/'deps')
    p.add_argument('--voice-config',type=Path,default=ROOT.parent/'piper-research/cadu-config.json');p.add_argument('--prepared',type=Path,default=DATA)
    p.add_argument('--data-manifest',type=Path,default=ROOT/'target-native161.json');p.add_argument('--run',type=Path,default=ROOT/'pilot')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--gpu-authorized',action='store_true');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-updates',type=int,default=3);p.add_argument('--resume',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--accumulation',type=int,default=1);p.add_argument('--cache-mib',type=int,default=128)
    p.add_argument('--lr-gen',type=float,default=5e-5);p.add_argument('--lr-disc',type=float,default=2.5e-5);p.add_argument('--seed',type=int,default=40733133)
    p.add_argument('--val-limit',type=int,default=55);p.add_argument('--patience',type=int,default=12);p.add_argument('--min-delta',type=float,default=.005)
    p.add_argument('--checkpoint-seconds',type=int,default=1800);p.add_argument('--case',default='v_command');p.add_argument('--reconstruct-id')
    a=p.parse_args();os.umask(0o077)
    if a.checkpoint and a.action in ('pilot','train'):raise ValueError('--checkpoint is diagnostic inference; use --resume for own fullstate continuation or pinned --base for freshstage')
    if a.resume and a.action not in ('pilot','train'):raise ValueError('--resume is for training; use --checkpoint for diagnostic loading')
    if min(a.max_updates,a.batch_size,a.accumulation,a.threads,a.val_limit,a.patience,a.checkpoint_seconds)<1 or a.cache_mib<0:raise ValueError('positive counts/cadences required')
    if any(not math.isfinite(x) or x<=0 for x in (a.lr_gen,a.lr_disc)) or not 0<=a.min_delta<1:raise ValueError('invalid LR/min_delta')
    if a.action=='fetch':print(json.dumps(fetch(a),indent=2));return
    private(a.run);private(a.data_manifest.parent)
    if a.action=='prepare':print(json.dumps(prepare(a),indent=2));return
    data=ensure_data(a)
    with (a.run/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);b=bootstrap(a)
        if a.action=='check':
            atomic_json(a.run/'cpu-check.json',dict(**b.provenance,code_sha256=sha(__file__),data_fingerprint=data['fingerprint'],counts={s:len(r) for s,r in data['rows'].items()},cuda_initialized=b.torch.cuda.is_initialized(),device=a.device,all_finite=True,strict_G_D=True,utc=utc()))
        elif a.action in ('pilot','train'):train(a,b,data)
        else:
            if a.checkpoint:
                state=b.saved_state if b.saved_state is not None else load_own(a.checkpoint,b);strict_copy(b.g,state['g'],b.torch);b.saved_state=None;del state
            if a.action=='reconstruct':
                row=next((r for r in data['rows']['validation'] if a.reconstruct_id is None or r['id']==a.reconstruct_id),None)
                if row is None:raise ValueError('reserved validation ID absent')
                wave,spec=AudioCache(b,a.cache_mib).get(row)
                with preserve_rng(b,a.seed),b.torch.inference_mode():
                    b.g.eval();z,_,_,mask=b.g.enc_q(spec[None].to(b.device),b.torch.tensor([spec.shape[1]],device=b.device));values=b.g.dec(z*mask).squeeze().float().cpu().numpy()
                target=audio_artifact(a.run/'target.wav',wave.numpy(),b)
                result=audio_artifact(a.run/'reconstruction.wav',values,b);result.update(target=target,source_id=row['id'],source_sha256=row['sha256'],target_samples=len(wave),path='linear513→posterior→decoder; no identity/listening claim')
            else:
                case=next((r for r in data['probes'] if r['id']==a.case),None)
                if case is None:raise ValueError('unknown validation case')
                result=audio_artifact(a.run/'inference.wav',synthesize(b,case['_sentences'],a.seed),b);result.update(case=case['id'],text=case['text'],sampling='zero_noise0/0 seedfixed',device=a.device)
            result.update(code_sha256=sha(__file__),data_fingerprint=data['fingerprint'],checkpoint_sha256=sha(a.checkpoint) if a.checkpoint else BASE_SHA,
                          cuda_initialized=b.torch.cuda.is_initialized(),device=a.device)
            atomic_json(a.run/(a.action+'.json'),result)

if __name__=='__main__':main()
