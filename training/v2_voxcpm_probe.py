"""Pinned VoxCPM2 pilot: plan/imports, VAE roundtrip, LoRA update/resume, GPU synthesis.

Default action is a metadata-only plan. GPU actions require already-local hashes,
explicit CLI authorization and an empty GPU. This module never downloads weights.
The pilot is not the final data/training recipe; all source clips stay intact.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path('/var/tmp/aegis-v2-work/voxcpm2')
SOURCE_REV='cce58a5b59303c9bd63d12f23afe6a49a4a80c59'
MODEL_REV='32279effe8c19989596f05d353d1447f51d9e915'
WEIGHTS={'model.safetensors':('f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d',4580080592),
         'audiovae.pth':('94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1',376951122)}
METADATA_BLOBS={'config.json':'792f1c223ed607f9e508c0a0deb15dd9532483be',
 'special_tokens_map.json':'8619dda6f3eb6d60d0a1bb274820054e46f41699',
 'tokenization_voxcpm2.py':'e7d768677298d058fa6ef8b160e3ca4430997fad',
 'tokenizer.json':'41a5c2a8dba4058dd1ad73fb898abf5e4f64f0f9',
 'tokenizer_config.json':'fecf4cdae73b57053cac2ad34c67febbe4e4f08b'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def atomic_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.chmod(0o600);tmp.replace(path)

def verify(args,vae_only=False):
 commit=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
 dirty=subprocess.check_output(['git','-C',str(args.source),'status','--porcelain'],text=True).strip()
 if commit!=SOURCE_REV or dirty:raise ValueError('source must be clean at the immutable pin')
 wanted=['audiovae.pth'] if vae_only else list(WEIGHTS)
 for name in wanted:
  expected,size=WEIGHTS[name];p=args.model/name
  if not p.is_file() or p.stat().st_size!=size or sha(p)!=expected:raise ValueError('weight integrity failure: '+name)
 if vae_only:
  names=['config.json']
 else:names=METADATA_BLOBS
 for name in names:
  data=(args.model/name).read_bytes();blob=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
  if blob!=METADATA_BLOBS[name]:raise ValueError('metadata integrity failure: '+name)
 for name in ('audiovae.safetensors','pytorch_model.bin'):
  if (args.model/name).exists():raise ValueError('unexpected alternative weights: '+name)
 return {name:sha(args.model/name) for name in wanted}

def imports(args):
 for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[k]='1'
 for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
 sys.path.insert(0,str(args.source/'src'))
 import torch
 torch.set_num_threads(2)
 from voxcpm.model.voxcpm2 import VoxCPM2Model,LoRAConfig
 from voxcpm.modules.audiovae import AudioVAEV2,AudioVAEConfigV2
 # This immutable helper has no relative imports. Direct loading avoids the
 # training package's unrelated eager datasets/argbind/UI dependency closure.
 spec=importlib.util.spec_from_file_location('aegis_vox_packer',args.source/'src/voxcpm/training/packers.py')
 pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)
 return torch,VoxCPM2Model,LoRAConfig,AudioVAEV2,AudioVAEConfigV2,pack.AudioFeatureProcessingPacker

def guard_gpu(args):
 if not args.gpu_authorized:raise ValueError('GPU action needs --gpu-authorized, and must run through the exclusive project queue')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 if apps:raise RuntimeError('another compute process occupies GPU: '+apps)
 args.run.mkdir(parents=True,exist_ok=True,mode=0o700)

def model_to_gpu(model,torch):
 model.to('cuda');model.device='cuda';model.config.device='cuda'
 # These caches are ordinary Python objects, not registered buffers; .to()
 # does not move them. Recreate both explicitly and verify complete residency.
 for lm in (model.base_lm,model.residual_lm):lm.setup_cache(1,model.config.max_length,'cuda',torch.bfloat16)
 model.audio_vae.float()
 # Legacy weight_norm stores a derived .weight Tensor outside parameters and
 # buffers. Recompute it from the CUDA weight_g/v before checking residency.
 for module in model.modules():
  for hook in module._forward_pre_hooks.values():
   if hook.__class__.__name__=='WeightNorm':hook(module,None)
 for name,p in model.named_parameters():
  if p.device.type!='cuda':raise RuntimeError('CPU-resident model parameter: '+name)
 for name,value in model.named_buffers():
  if value.device.type!='cuda':raise RuntimeError('CPU-resident model buffer: '+name)
 for name,module in model.named_modules():
  for attr,value in vars(module).items():
   if torch.is_tensor(value) and value.device.type!='cuda':
    raise RuntimeError('CPU-resident ordinary Tensor/cache: '+name+'.'+attr)
 for lm in (model.base_lm,model.residual_lm):
  if lm.kv_cache.kv_cache.device.type!='cuda':raise RuntimeError('CPU KV cache')
 return model

def load_model(args,torch,cls,lora_cls,to_cuda=True):
 lora=lora_cls(enable_lm=True,enable_dit=True,enable_proj=False,r=args.rank,alpha=args.rank,dropout=0.)
 previous=torch.get_default_dtype()
 try:
  # Upstream training=True otherwise allocates the frozen 2B backbone FP32.
  # Build/load on CPU in BF16, keeping frozen AudioVAE explicitly FP32.
  torch.set_default_dtype(torch.bfloat16)
  model=cls.from_local(str(args.model),optimize=False,training=True,device='cpu',lora_config=lora)
 finally:torch.set_default_dtype(previous)
 for name,p in model.named_parameters():
  if p.requires_grad:p.data=p.data.float()
 return model_to_gpu(model,torch) if to_cuda else model

def copy_lora_strict(model,state,torch):
 expected={name:p for name,p in model.named_parameters() if p.requires_grad}
 if set(state)!=set(expected):
  raise ValueError('LoRA key mismatch: missing='+str(sorted(set(expected)-set(state)))+' extra='+str(sorted(set(state)-set(expected))))
 for name,p in expected.items():
  value=state[name]
  if not torch.is_tensor(value) or tuple(value.shape)!=tuple(p.shape) or value.dtype!=p.dtype:
   raise ValueError('LoRA shape/dtype mismatch: '+name)
  if not torch.isfinite(value).all():raise ValueError('nonfinite LoRA state: '+name)
 with torch.no_grad():
  for name,p in expected.items():p.copy_(state[name].to(p.device))

def cpu_check(args,torch,cls,lora_cls):
 import resource
 from collections import Counter
 from safetensors import safe_open
 memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
 predicted_peak=11_000_000_000;reserve=2_000_000_000
 if memory['MemAvailable']<predicted_peak+reserve:
  raise RuntimeError('insufficient measured RAM for CPU load: '+json.dumps(memory))
 args.run.mkdir(parents=True,exist_ok=True,mode=0o700)
 atomic_json(args.run/'cpu-check-preflight.json',dict(**memory,predicted_peak_bytes=predicted_peak,reserve_bytes=reserve,CUDA_VISIBLE_DEVICES=''))
 start=time.perf_counter();model=load_model(args,torch,cls,lora_cls,to_cuda=False)
 state=model.state_dict();lora={n:p for n,p in model.named_parameters() if p.requires_grad}
 if not lora or any('lora_' not in n for n in lora):raise RuntimeError('unexpected trainable set')
 dtype_counts=Counter()
 for name,p in model.named_parameters():
  expected=torch.float32 if name.startswith('audio_vae.') or name in lora else torch.bfloat16
  if p.device.type!='cpu' or p.dtype!=expected:raise ValueError('CPU load dtype/residency mismatch: '+name)
  dtype_counts[str(p.dtype)]+=p.numel()
 base_keys=set()
 with safe_open(str(args.model/'model.safetensors'),framework='pt',device='cpu') as f:
  for name in f.keys():
   base_keys.add(name)
   if name not in state or tuple(f.get_slice(name).get_shape())!=tuple(state[name].shape):raise ValueError('base checkpoint shape/key mismatch: '+name)
  probe='base_lm.layers.0.self_attn.q_proj.weight'
  if not torch.equal(f.get_slice(probe)[:1,:64],state[probe][:1,:64]):raise ValueError('base tensor values were not loaded')
 vae_state=torch.load(args.model/'audiovae.pth',map_location='cpu',weights_only=True,mmap=True)
 vae_state=vae_state.get('state_dict',vae_state)
 for name,value in vae_state.items():
  name='audio_vae.'+name;base_keys.add(name)
  if name not in state or tuple(value.shape)!=tuple(state[name].shape):raise ValueError('VAE checkpoint shape/key mismatch: '+name)
 if base_keys!=set(state)-set(lora):raise ValueError('base+VAE key set is not exact')
 copy_lora_strict(model,{n:p.detach().clone() for n,p in lora.items()},torch)
 layer=next(m for m in model.modules() if m.__class__.__name__=='LoRALinear')
 x=torch.randn(1,2,layer.in_features,dtype=torch.bfloat16,requires_grad=True)
 with torch.autocast('cpu',dtype=torch.bfloat16):output=layer(x)
 if output.dtype!=torch.bfloat16:raise ValueError('FP32 LoRA promoted BF16 activation')
 output.float().square().mean().backward()
 if layer.lora_B.grad is None or layer.lora_B.grad.dtype!=torch.float32 or not torch.isfinite(layer.lora_B.grad).all():raise ValueError('LoRA gradient contract failed')
 for p in lora.values():p.grad=None
 if torch.cuda.is_initialized():raise RuntimeError('CPU check initialized CUDA unexpectedly')
 record=dict(status='pass',scope='strict full CPU load + one real LoRALinear autocast/backward; no full-model forward/optimizer/GPU',
  seconds=time.perf_counter()-start,base_keys=len(base_keys),lora_tensors=len(lora),lora_parameters=sum(p.numel() for p in lora.values()),
  parameter_elements_by_dtype=dict(dtype_counts),lora_activation_dtype=str(output.dtype),cuda_initialized=False,
  peak_cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,code_sha256=sha(__file__))
 atomic_json(args.run/'cpu-check.json',record);print(json.dumps(record),flush=True)

def enable_checkpointing(model,torch):
 from torch.utils.checkpoint import checkpoint
 count=0
 # Non-reentrant supports frozen inputs + trainable LoRA parameters and the
 # nested (hidden, KV) return. Inference forward_step remains untouched.
 for module in model.modules():
  if module.__class__.__name__!='MiniCPMDecoderLayer':continue
  original=module.forward
  def wrapped(*args,_original=original,_module=module,**kwargs):
   if _module.training and torch.is_grad_enabled():
    return checkpoint(_original,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
   return _original(*args,**kwargs)
  module.forward=wrapped;count+=1
 return count

def audio(row,np):
 import soundfile as sf
 from scipy.signal import resample_poly
 path=Path(row.get('source_audio_filepath',row.get('audio_filepath',row.get('audio'))))
 expected=row.get('source_audio_sha256') if row.get('source_audio_filepath') else row.get('sha256')
 if expected and sha(path)!=expected:raise ValueError('source WAV SHA mismatch')
 data,sr=sf.read(path,dtype='float32',always_2d=True)
 if data.shape[1]!=1 or not data.size or not np.isfinite(data).all():raise ValueError('finite nonempty mono source required')
 d=math.gcd(sr,16000);values=resample_poly(data[:,0],16000//d,sr//d,window=('kaiser',10)).astype(np.float32)
 return values,path,sr

def make_batch(row,model,packer,torch,np):
 values,path,sr=audio(row,np)
 text=row.get('original_text',row.get('text'))
 ids=torch.tensor(model.text_tokenizer(text),device='cuda',dtype=torch.int32)
 # Exact upstream packing, complete source utterance; no transcript/audio crop.
 packed=packer.process_tts_data(torch.from_numpy(values).to('cuda'),ids)
 tokens,features,text_mask,audio_mask,loss_mask,labels=packed[:6]
 if len(tokens)>512:raise ValueError('pilot sequence exceeds 512-token cap; select another whole clip')
 return [x[None] for x in (tokens,text_mask,features,audio_mask,loss_mask)]+[torch.arange(len(tokens),device='cuda')[None],labels[None]],len(tokens)

def cpu_optimizer(params,torch,lr):
 masters=[torch.nn.Parameter(p.detach().float().cpu().clone()) for p in params]
 return torch.optim.AdamW(masters,lr=lr,weight_decay=.01),masters

def checksum_tensors(named,torch):
 h=hashlib.sha256()
 for name,p in named:
  h.update(name.encode());h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def save_state(path,model,optimizer,scheduler,step,cursor,torch,np,config):
 state=dict(lora={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
  optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),amp_scaler=None,
  precision='BF16 backbone/VAE FP32/LoRA FP32/autocast BF16',step=step,cursor=cursor,config=config,
  rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
 tmp=path.with_suffix('.pt.tmp');torch.save(state,tmp)
 with tmp.open('rb') as f:os.fsync(f.fileno())
 check=torch.load(tmp,map_location='cpu',weights_only=False)
 if check['step']!=step or checksum_tensors(check['lora'].items(),torch)!=checksum_tensors(state['lora'].items(),torch):raise ValueError('new checkpoint failed reload verification')
 tmp.replace(path);atomic_json(path.with_suffix('.sha256.json'),dict(sha256=sha(path),step=step,bytes=path.stat().st_size))

def restore(path,model,optimizer,scheduler,masters,torch,np,config):
 receipt=json.loads(path.with_suffix('.sha256.json').read_text(encoding='utf-8'))
 if sha(path)!=receipt['sha256']:raise ValueError('resume integrity failure')
 state=torch.load(path,map_location='cpu',weights_only=False)
 if state['config']!=config:raise ValueError('resume configuration/data partition mismatch')
 copy_lora_strict(model,state['lora'],torch)
 params=[p for p in model.parameters() if p.requires_grad]
 if masters is not None:
  for master,p in zip(masters,params):master.data.copy_(p.detach().cpu())
 optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
 r=state['rng'];random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
 return state['step'],state['cursor']

def synthesize(args,model,torch,np,label):
 import soundfile as sf
 from scipy.signal import resample_poly
 text=args.text
 nfe=int(getattr(args,'nfe',10))
 if nfe<1:raise ValueError('inference timesteps must be positive')
 model.eval();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 # Model's native generator, no normalizer/denoiser/retry or CPU weight offload.
 original_data=[(p,p.data) for p in model.parameters() if p.requires_grad]
 try:
  # Native generation has no outer autocast. Retain the FP32 master tensors
  # unchanged while materializing BF16 LoRA inference weights, then restore.
  for p,_ in original_data:p.data=p.data.to(torch.bfloat16)
  values=model.generate(target_text=text,max_len=400,inference_timesteps=nfe,cfg_value=2.,retry_badcase=False,
                        retry_badcase_ratio_threshold=6.,seed=40733133)
 finally:
  for p,data in original_data:p.data=data
 values=np.asarray(values,dtype=np.float32).reshape(-1)
 torch.cuda.synchronize();native_ms=(time.perf_counter()-start)*1000
 final=resample_poly(values,1,2,window=('kaiser',10)).astype(np.float32)
 finite=bool(np.isfinite(final).all());peak=float(abs(final).max()) if finite and final.size else None
 row=dict(text=text,native_ms=native_ms,nfe=nfe,inference_timesteps=nfe,
  nfe_label='upstream inference_timesteps argument, not a measured estimator-evaluation count',
  cfg_value=2.,seed=40733133,max_len=400,retry_badcase=False,retry_badcase_ratio_threshold=6.,
  final_sample_rate=24000,native_sample_rate=48000,peak=peak,
  outside_unit=int((abs(final)>1).sum()) if finite else None,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,scope='whole-file native engine incl resample/PCM/WAV close; no HTTP',
  streaming='native generator supports streaming but this pilot measures whole file only')
 if finite and final.size and peak<=1:
  pcm=np.rint(final*32767).astype('<i2').tobytes();path=args.run/(label+'-24000.wav')
  import wave
  with wave.open(str(path),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(pcm)
  elapsed=(time.perf_counter()-start)*1000
  spec=importlib.util.spec_from_file_location('aegis_v2_metrics',Path(__file__).resolve().parents[1]/'benchmarks/v2/benchmark.py')
  b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
  row.update(outcome='ok',wav=str(path),sha256=sha(path),eof_ms=elapsed,rtf=elapsed/1000/(len(final)/24000),**b.pcm_metrics(pcm,[dict(ms=elapsed,bytes=len(pcm))]))
 else:row.update(outcome='failed_PCMconversion',error='no clipping or gain applied; no product TTFA')
 raw=args.run/(label+'-raw48000.wav');sf.write(raw,values,48000,subtype='FLOAT');row['raw_float']=str(raw)
 atomic_json(args.run/(label+'.json'),row);return row

def roundtrip(args,torch,vae_cls,vae_config):
 import numpy as np
 import soundfile as sf
 config=json.loads((args.model/'config.json').read_text(encoding='utf-8'))['audio_vae_config'];vae=vae_cls(config=vae_config(**config))
 state=torch.load(args.model/'audiovae.pth',map_location='cpu',weights_only=True);vae.load_state_dict(state.get('state_dict',state),strict=True);vae.to('cuda').eval()
 rows=[json.loads(x) for x in args.manifest.read_text(encoding='utf-8').splitlines()][:args.samples];results=[]
 for row in rows:
  if row.get('split') not in ('validation','val'):raise ValueError('VAE decision needs reserved validation anchors, not training/final test')
  values,path,sr=audio(row,np);tensor=torch.from_numpy(values)[None,None].cuda();torch.cuda.synchronize();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
  with torch.inference_mode():out=vae.decode(vae.encode(tensor,16000)).squeeze().float().cpu().numpy()
  elapsed=time.perf_counter()-start
  expected_padded_samples=math.ceil(len(values)/int(vae.hop_length))*int(vae.decode_chunk_size)
  if not out.size or not np.isfinite(out).all() or len(out)!=expected_padded_samples:
   failed=args.run/(row['id']+'-invalid-untrimmed-vae48k.wav')
   sf.write(failed,out,48000,subtype='FLOAT')
   atomic_json(args.run/'roundtrip-failed.json',dict(id=row['id'],outcome='failed_roundtrip_contract',
    actual_samples=int(out.size),expected_padded_samples=expected_padded_samples,
    nonfinite_samples=int((~np.isfinite(out)).sum()),raw_float=str(failed)))
   raise ValueError('VAE output nonfinite/empty or wrong padded length; untrimmed FLOAT preserved')
  out=out[:len(values)*3]
  dest=args.run/(row['id']+'-vae48k.wav');sf.write(dest,out,48000,subtype='FLOAT')
  control=args.run/(row['id']+'-encoder16k.wav');sf.write(control,values,16000,subtype='FLOAT')
  results.append(dict(id=row['id'],text=row.get('original_text',row['text']),source=str(path),source_rate=sr,source_sha256=sha(path),
   encoder_input=str(control),encoder_input_sha256=sha(control),reconstructed=str(dest),reconstructed_sha256=sha(dest),
   seconds=elapsed,peak=float(abs(out).max()),outside_unit=int((abs(out)>1).sum()),source_seconds=len(values)/16000,
   peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,padding_policy='remove only tail beyond resampled input length*3',
   interpretation='16kHz conditioning/48kHz decoder super-resolution, not proof of robotic identity'))
  atomic_json(args.run/'roundtrip.json',results)


def pilot(args,torch,cls,lora_cls,packer_cls):
 import numpy as np
 random.seed(40733133);np.random.seed(40733133);torch.manual_seed(40733133)
 model=load_model(args,torch,cls,lora_cls);layers=enable_checkpointing(model,torch) if args.checkpoint_layers else 0
 params=[p for p in model.parameters() if p.requires_grad]
 if not params:raise RuntimeError('no trainable LoRA parameters')
 if any(p.dtype!=torch.float32 for p in params):raise RuntimeError('LoRA precision must be FP32')
 if args.optimizer_device=='cpu':optimizer,masters=cpu_optimizer(params,torch,args.lr)
 else:optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=.01);masters=None
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
 all_rows=[json.loads(x) for x in args.manifest.read_text(encoding='utf-8').splitlines()]
 if any(r.get('split','train')!='train' for r in all_rows):raise ValueError('pilot requires train split only')
 rows=[r for r in all_rows if 3<=r['duration']<=args.max_seconds]
 if len(rows)<2:raise ValueError('need >=2 complete short training clips; do not crop audio/text')
 rows=sorted(rows,key=lambda r:r['id'])
 config=dict(source_rev=SOURCE_REV,model_rev=MODEL_REV,manifest_sha256=sha(args.manifest),rank=args.rank,
  lr=args.lr,max_seconds=args.max_seconds,optimizer_device=args.optimizer_device,checkpoint_layers=args.checkpoint_layers,
  code_sha256=sha(__file__),sample_ids=[r['id'] for r in rows],grad_accum=args.accum,
  precision='BF16 frozen base, FP32 VAE, FP32 LoRA',pilot_only=True)
 step=cursor=0
 if args.resume:step,cursor=restore(args.resume,model,optimizer,scheduler,masters,torch,np,config)
 atomic_json(args.run/'config.json',dict(**config,checkpoint_layers_wrapped=layers,trainable_parameters=sum(p.numel() for p in params),
  initial_step=step,initial_cursor=cursor,total_model_parameters=sum(p.numel() for p in model.parameters()),infer_full_gpu_verified=True))
 packer=packer_cls(1,512,model.config.patch_size,model.config.feat_dim,model.audio_vae)
 before=checksum_tensors([(n,p) for n,p in model.named_parameters() if p.requires_grad],torch)
 model.train();model.audio_vae.eval()
 for target in range(step+1,args.steps+1):
  start=time.perf_counter();torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
  for p in params:p.grad=None
  losses=[];tokens=[]
  for micro in range(args.accum):
   row=rows[cursor%len(rows)];batch,length=make_batch(row,model,packer,torch,np);cursor+=1;tokens.append(length)
   with torch.autocast('cuda',dtype=torch.bfloat16):out=model(*batch);loss=out['loss/diff']+out['loss/stop']
   if not torch.isfinite(loss):raise ValueError('nonfinite loss')
   (loss/args.accum).backward();losses.append(float(loss.detach()));del out,loss,batch
  norm=torch.nn.utils.clip_grad_norm_(params,1.)
  if not torch.isfinite(norm):raise ValueError('nonfinite gradients; preserve the previous valid checkpoint')
  if masters is not None:
   for master,p in zip(masters,params):master.grad=p.grad.detach().float().cpu() if p.grad is not None else None
   optimizer.step()
   with torch.no_grad():
    for master,p in zip(masters,params):p.copy_(master.to(p.device))
  else:optimizer.step()
  scheduler.step();torch.cuda.synchronize()
  record=dict(step=target,cursor=cursor,loss=sum(losses)/len(losses),tokens=tokens,grad_norm=float(norm),seconds=time.perf_counter()-start,
   peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
  with (args.run/'steps.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
  save_state(args.run/f'step-{target:04d}.pt',model,optimizer,scheduler,target,cursor,torch,np,config)
  print(json.dumps(record),flush=True)
 after=checksum_tensors([(n,p) for n,p in model.named_parameters() if p.requires_grad],torch)
 if args.steps>step and before==after:raise RuntimeError('optimizer did not change actual LoRA weights')
 atomic_json(args.run/'update-proof.json',dict(initial_step=step,final_step=args.steps,initial_sha256=before,final_sha256=after,weights_changed=before!=after))
 synthesize(args,model,torch,np,'after-step-'+str(args.steps))


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=['plan','imports','cpu-check','roundtrip','pilot','infer'],default='plan')
 p.add_argument('--source',type=Path,default=ROOT/'source');p.add_argument('--model',type=Path,default=ROOT/'model');p.add_argument('--run',type=Path,default=ROOT/'pilot')
 p.add_argument('--manifest',type=Path,default=Path('/var/tmp/aegisvoice-v2-training-20261102/prepared/train.jsonl'))
 p.add_argument('--samples',type=int,default=3);p.add_argument('--gpu-authorized',action='store_true');p.add_argument('--steps',type=int,default=3);p.add_argument('--resume',type=Path)
 p.add_argument('--rank',type=int,default=8);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--max-seconds',type=float,default=6.)
 p.add_argument('--accum',type=int,default=1);p.add_argument('--optimizer-device',choices=['cuda','cpu'],default='cuda');p.add_argument('--checkpoint-layers',action=argparse.BooleanOptionalAction,default=True)
 p.add_argument('--text',default='A válvula azul permanece fechada durante a revisão.');a=p.parse_args();os.umask(0o077)
 if a.rank<1 or a.accum<1 or a.samples<1 or not 1<=a.steps<=10:p.error('positive rank/accum/samples and pilot1..10steps required; long training needs bounded retention and validation recipe')
 if a.action=='plan':
  print(json.dumps(dict(source_rev=SOURCE_REV,model_rev=MODEL_REV,weights=WEIGHTS,required_metadata=METADATA_BLOBS,
   dependencies='reuse F5 environment; no optional UI/denoise/data-framework packages required for this native pilot',
   sequence=['VAE roundtrip','full-GPU base inference and PT-BR validation','pilot3 + fresh-process resume4 + synthesis','only then consider long specialization'],
   vram='BF16 frozen backbone, FP32 AudioVAE/LoRA, microbatch1, per-layer nonreentrant checkpointing; unmeasured on A2000',
   storage='weights4.957GB + source0.011GB + metadata0.004GB + two resumed LoRA checkpoints~0.2GB; no duplicate full-model saves',
   local_weights={name:dict(exists=(a.model/name).is_file(),expected_bytes=expected[1]) for name,expected in WEIGHTS.items()},
   download_receipt=str(ROOT/'download-receipt.json') if (ROOT/'download-receipt.json').exists() else None,
   status='GPU/forward/update unmeasured; inspect local receipts for actual weight acquisition state'),indent=2));return
 if a.action=='imports':
  os.environ['CUDA_VISIBLE_DEVICES']='';t=time.perf_counter();torch,*_=imports(a)
  print(json.dumps(dict(imports='pass',seconds=time.perf_counter()-t,torch=torch.__version__,cuda_build=torch.version.cuda,cuda_execution=False)));return
 if a.action=='cpu-check':
  os.environ['CUDA_VISIBLE_DEVICES']='';verify(a);torch,cls,lora_cls,*_=imports(a)
  cpu_check(a,torch,cls,lora_cls);return
 verify(a,vae_only=a.action=='roundtrip');guard_gpu(a);torch,cls,lora_cls,vae_cls,vae_config,packer_cls=imports(a)
 if a.action=='roundtrip':roundtrip(a,torch,vae_cls,vae_config)
 elif a.action=='pilot':pilot(a,torch,cls,lora_cls,packer_cls)
 elif a.action=='infer':
  import numpy as np
  model=load_model(a,torch,cls,lora_cls)
  if a.resume:
   if sha(a.resume)!=json.loads(a.resume.with_suffix('.sha256.json').read_text(encoding='utf-8'))['sha256']:raise ValueError('LoRA integrity failure')
   state=torch.load(a.resume,map_location='cpu',weights_only=False)
   copy_lora_strict(model,state['lora'],torch)
  synthesize(a,model,torch,np,'inference')
if __name__=='__main__':main()
