"""Optional experimental VoxCPM2 runtime. Importing this module never imports Torch.

All neural operations, warm-up, native iterator steps, close and cache cleanup
belong to one persistent inference thread. No training module is imported.
Paths/configuration are operator owned; public requests cannot select weights.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

from voxcpm_audio import HalfRateFIR, native_numpy, pcm16_bytes

SOURCE_REV = 'cce58a5b59303c9bd63d12f23afe6a49a4a80c59'
MODEL_REV = '32279effe8c19989596f05d353d1447f51d9e915'
WEIGHTS = {
    'model.safetensors': ('f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d',4580080592),
    'audiovae.pth': ('94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1',376951122),
}
METADATA = {
    'config.json':'792f1c223ed607f9e508c0a0deb15dd9532483be',
    'special_tokens_map.json':'8619dda6f3eb6d60d0a1bb274820054e46f41699',
    'tokenization_voxcpm2.py':'e7d768677298d058fa6ef8b160e3ca4430997fad',
    'tokenizer.json':'41a5c2a8dba4058dd1ad73fb898abf5e4f64f0f9',
    'tokenizer_config.json':'fecf4cdae73b57053cac2ad34c67febbe4e4f08b',
}
NATIVE_PROFILE = 'AegisVoicev2.0-VoxCPM2-experimental'


class BackendTextTooLong(ValueError): pass
class GenerationLimitReached(RuntimeError): pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class VoxConfig:
    source: Path
    model: Path
    adapter: Path
    adapter_sha256: str
    cache: Path
    gpu_lock: Path
    compiler_prerequisites: Path | None = None
    dsp: str = 'passthrough'
    max_len: int = 400
    capacity: int = 1024
    nfe: int = 10
    rank: int = 8
    seed: int = 40733133
    max_buffer_chunks: int = 2
    client_stall_seconds: float = 10.
    cache_limit_mib: int = 512
    instrument_replays: bool = False
    frontend: str = 'original-nfc'

    def __post_init__(self):
        if (self.max_len,self.capacity,self.nfe,self.rank) != (400,1024,10,8):
            raise ValueError('Experimental validated recipe requires max_len400/KV1024/NFE10/rank8')
        if self.dsp not in {'passthrough','world-w04-p07-m02'}:
            raise ValueError('Explicit passthrough or existing WORLD recipe required')
        if not re.fullmatch('[0-9a-f]{64}',self.adapter_sha256): raise ValueError('Pinned adapter SHA256 required')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32: raise ValueError('Invalid fixed seed')
        if not 1 <= self.max_buffer_chunks <= 4 or not .1 <= self.client_stall_seconds <= 10:
            raise ValueError('Bounded bridge buffer/stall required')
        if self.cache_limit_mib != 512: raise ValueError('Validated compile cache budget is512MiB')
        if type(self.instrument_replays) is not bool: raise ValueError('instrument_replays must be boolean')
        if self.frontend not in {'original-nfc','f5-refinement-1'}:raise ValueError('Explicit original/refinement frontend required')
        for name in ('source','model','adapter','cache','gpu_lock'):
            if not isinstance(getattr(self,name),Path) or not getattr(self,name).is_absolute():
                raise ValueError('Operator paths must be absolute: '+name)

    @classmethod
    def read(cls,path):
        value=json.loads(Path(path).read_text(encoding='utf-8'))
        allowed=set(cls.__dataclass_fields__)
        if not isinstance(value,dict) or set(value)-allowed: raise ValueError('Unknown experimental backend configuration')
        for name in ('source','model','adapter','cache','gpu_lock','compiler_prerequisites'):
            if value.get(name) is not None: value[name]=Path(value[name]).resolve()
        return cls(**value)

    @property
    def profile(self):
        suffix='Passthrough' if self.dsp=='passthrough' else 'W04-P07-M02'
        frontend='OriginalNFC' if self.frontend=='original-nfc' else 'F5Refinement1'
        return f'AegisVoicev2.0-VoxCPM2-LoRA-r8-N10-{suffix}-{frontend}-Experimental'


def capacity_guard(tokens, config):
    if type(tokens) is not int or tokens < 1: raise ValueError('Nonempty exact tokenizer output required')
    needed=tokens+config.max_len+2
    if needed>config.capacity:
        raise BackendTextTooLong('Segment exceeds the1024-slot text+audio reservation; no truncation')
    return dict(text_tokens=tokens,reserved_slots=needed,capacity=config.capacity,
                audio_patch_limit=min(tokens*6+10,config.max_len))


def verify_local(config):
    commit=subprocess.check_output(['git','-C',str(config.source),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(config.source),'status','--porcelain'],text=True).strip()
    if commit!=SOURCE_REV or dirty: raise ValueError('Vox source must be clean at the pinned revision')
    if any((config.model/n).exists() for n in ('audiovae.safetensors','pytorch_model.bin')):
        raise ValueError('Unpinned alternate model/AudioVAE loader precedence')
    for name,(expected,size) in WEIGHTS.items():
        path=config.model/name
        if not path.is_file() or path.stat().st_size!=size or digest(path)!=expected:
            raise ValueError('Pinned weight integrity failure: '+name)
    for name,expected in METADATA.items():
        value=(config.model/name).read_bytes()
        if hashlib.sha1(f'blob {len(value)}\0'.encode()+value).hexdigest()!=expected:
            raise ValueError('Pinned metadata integrity failure: '+name)
    receipt=json.loads(config.adapter.with_suffix('.sha256.json').read_text(encoding='utf-8'))
    if receipt['sha256']!=config.adapter_sha256 or receipt.get('bytes')!=config.adapter.stat().st_size:
        raise ValueError('Pinned adapter sidecar differs')
    return receipt


def prepare_cache(config):
    config.cache.mkdir(parents=True,exist_ok=True,mode=0o700)
    for name in ('tmp','extensions','cuda'): (config.cache/name).mkdir(exist_ok=True,mode=0o700)
    os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(config.cache/'inductor'),TRITON_CACHE_DIR=str(config.cache/'triton'),
        TORCHINDUCTOR_COMPILE_THREADS='1',MAX_JOBS='1',TMPDIR=str(config.cache/'tmp'),
        TORCH_EXTENSIONS_DIR=str(config.cache/'extensions'),CUDA_CACHE_PATH=str(config.cache/'cuda'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import tempfile
    tempfile.tempdir=str(config.cache/'tmp')
    if config.compiler_prerequisites:
        receipt=json.loads(config.compiler_prerequisites.read_text(encoding='utf-8'))
        headers=receipt['headers']
        if digest(headers['manifest'])!=headers['manifest_sha256']: raise ValueError('Compiler header manifest differs')
        values=receipt.get('env_for_child_only',{})
        if set(values)-{'CPATH','CC','CXX'}: raise ValueError('Unexpected compiler environment keys')
        os.environ.update(values)


def cache_budget(config):
    # Count allocated bytes, including the compilation temp directory. Never
    # remove a file that a compiler or active runtime could still be using.
    seen=set();size=0
    for parent,_,files in os.walk(config.cache):
        for name in files:
            try: st=(Path(parent)/name).stat()
            except FileNotFoundError: continue
            if (st.st_dev,st.st_ino) not in seen: size+=st.st_blocks*512;seen.add((st.st_dev,st.st_ino))
    if size>config.cache_limit_mib*1024**2 or shutil.disk_usage(config.cache).free<5*1024**3:
        raise RuntimeError('Compile cache budget or filesystem reserve exceeded')
    return size


@contextmanager
def strict_base_load(cls):
    """Upstream permits missing LoRA; reject every other missing/extra key."""
    own='load_state_dict' in cls.__dict__;original=cls.load_state_dict
    def checked(model,state,*args,**kwargs):
        result=original(model,state,strict=False)
        lora={n for n,p in model.named_parameters() if 'lora_' in n}
        if set(result.missing_keys)!=lora or result.unexpected_keys or set(state)!=set(model.state_dict())-lora:
            raise ValueError('Base+AudioVAE key set must be exact, excluding only new LoRA branches')
        return result
    cls.load_state_dict=checked
    try: yield
    finally:
        if own: cls.load_state_dict=original
        else: delattr(cls,'load_state_dict')


def copy_adapter(model,state,torch):
    expected={n:p for n,p in model.named_parameters() if p.requires_grad}
    if not expected or set(state)!=set(expected) or any('lora_' not in n for n in expected):
        raise ValueError('Exact trainable LoRA key set required')
    for name,p in expected.items():
        value=state[name]
        if not torch.is_tensor(value) or value.dtype!=torch.float32 or value.shape!=p.shape or not torch.isfinite(value).all():
            raise ValueError('FP32 finite LoRA shape/dtype mismatch: '+name)
    with torch.no_grad():
        for name,p in expected.items(): p.copy_(state[name])


class NativeSession:
    """Model lifetime belongs entirely to the backend's persistent worker."""
    def __init__(self,config):
        self.config=config;self.model=None;self.torch=None;self.lease=None
        self.last_observation={};self.before_generation={}
        self.replays=0;self.replay_original=None;self.replay_hook=None
        self.load_observation={}

    def load(self):
        config=self.config
        verify_local(config);prepare_cache(config);cache_budget(config)
        config.gpu_lock.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.lease=config.gpu_lock.open('a+b')
        fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if apps: raise RuntimeError('Experimental backend requires an otherwise idle GPU')
        memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
        if memory['MemAvailable']<13_000_000_000: raise RuntimeError('Need13GB available RAM for the measured CPU-first loading route')
        sys.path.insert(0,str(config.source/'src'))
        import torch
        from voxcpm.model.voxcpm2 import VoxCPM2Model,LoRAConfig
        import voxcpm.model.voxcpm2 as module
        if not Path(module.__file__).resolve().is_relative_to(config.source.resolve()): raise ValueError('Vox import escaped pinned source')
        self.torch=torch;torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
        old=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with strict_base_load(VoxCPM2Model):
                model=VoxCPM2Model.from_local(str(config.model),optimize=False,training=True,device='cpu',
                    lora_config=LoRAConfig(enable_lm=True,enable_dit=True,enable_proj=False,r=8,alpha=8,dropout=0.))
        finally: torch.set_default_dtype(old)
        self.model=model
        for p in model.parameters():
            if p.requires_grad: p.data=p.data.float()
        # Hash and deserialize the same open inode. A concurrent atomic save
        # cannot substitute another checkpoint between checking and loading.
        with config.adapter.open('rb') as stream:
            h=hashlib.sha256()
            for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
            if h.hexdigest()!=config.adapter_sha256: raise ValueError('Adapter checkpoint changed')
            stream.seek(0);state=torch.load(stream,map_location='cpu',weights_only=False)
        saved=state['config']
        if (saved.get('rank'),saved.get('source_rev'),saved.get('model_rev'))!=(8,SOURCE_REV,MODEL_REV):
            raise ValueError('Adapter topology/source/model provenance differs')
        copy_adapter(model,state['lora'],torch);del state
        for name,p in model.named_parameters():
            p.requires_grad_(False)
            if 'lora_' in name: p.data=p.data.to(torch.bfloat16)
        model.config.max_length=1024
        model.to('cuda');model.device='cuda';model.config.device='cuda';model.audio_vae.float()
        for lm in (model.base_lm,model.residual_lm): lm.setup_cache(1,1024,'cuda',torch.bfloat16)
        for layer in model.modules():
            for hook in layer._forward_pre_hooks.values():
                if hook.__class__.__name__=='WeightNorm': hook(layer,None)
        self.check_residency();model.eval().optimize()
        compiled=(model.base_lm.forward_step,model.residual_lm.forward_step,model.feat_encoder,model.feat_decoder.estimator)
        if not all(hasattr(f,'_torchdynamo_orig_callable') or hasattr(f,'_orig_mod') for f in compiled):
            raise RuntimeError('Official reduce-overhead optimization was disabled or partial')
        self.instrument_replays()
        torch.cuda.synchronize()
        self.load_observation=self.observe()
        return self

    def instrument_replays(self):
        if not self.config.instrument_replays:return
        self.replay_original=self.torch.cuda.CUDAGraph.replay
        def counted(graph,*args,**kwargs):
            self.replays+=1
            return self.replay_original(graph,*args,**kwargs)
        self.replay_hook=counted;self.torch.cuda.CUDAGraph.replay=counted

    def token_count(self,text): return len(self.model.text_tokenizer(text))

    def check_residency(self):
        torch=self.torch;model=self.model
        for name,p in model.named_parameters():
            expected=torch.float32 if name.startswith('audio_vae.') else torch.bfloat16
            if p.device.type!='cuda' or p.dtype!=expected or p.requires_grad:
                raise RuntimeError('Neural parameter residency/dtype mismatch: '+name)
        for name,tensor in model.named_buffers():
            if tensor.device.type!='cuda': raise RuntimeError('CPU model buffer: '+name)
        for name,module in model.named_modules():
            for attr,tensor in vars(module).items():
                if torch.is_tensor(tensor) and tensor.device.type!='cuda': raise RuntimeError('CPU ordinary tensor: '+name+'.'+attr)
        for lm in (model.base_lm,model.residual_lm):
            if lm.kv_cache.max_length!=1024 or lm.kv_cache.kv_cache.device.type!='cuda': raise RuntimeError('KV cache residency/capacity differs')

    def generate(self,text):
        cache_budget(self.config)
        self.before_generation=self.observe()
        self.torch.cuda.reset_peak_memory_stats()
        self.torch.compiler.cudagraph_mark_step_begin()
        return self.model.generate_streaming(target_text=text,max_len=400,inference_timesteps=10,cfg_value=2.,
            retry_badcase=False,retry_badcase_ratio_threshold=6.,trim_silence_vad=False,streaming_prefix_len=4,seed=self.config.seed)

    def reset(self):
        if self.model is None:return
        # Preserve addresses captured by CUDA graphs. Closing the native
        # generator first releases StreamingVAEDecoder's context/monkeypatches.
        with self.torch.no_grad():
            for lm in (self.model.base_lm,self.model.residual_lm):
                lm.kv_cache.kv_cache.zero_();lm.kv_cache.current_length=0
        self.torch.cuda.synchronize()
        cache_budget(self.config)
        self.last_observation=self.observe()
        self.last_observation['generation_new_unique_graphs']=max(0,
            self.last_observation.get('compile_unique_graphs',0)-self.before_generation.get('compile_unique_graphs',0))
        if self.config.instrument_replays:
            self.last_observation['generation_cudagraph_replays']=self.replays-self.before_generation.get('cudagraph_replays_total',0)

    def observe(self):
        """Numeric diagnostics only; counters help identify novel-text recompiles."""
        values=dict(cuda_allocated_bytes=self.torch.cuda.memory_allocated(),
            cuda_reserved_bytes=self.torch.cuda.memory_reserved(),
            cuda_peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=self.torch.cuda.max_memory_reserved(),compile_cache_bytes=cache_budget(self.config))
        values['cudagraph_replays_instrumented']=int(self.config.instrument_replays)
        if self.config.instrument_replays:values['cudagraph_replays_total']=self.replays
        for line in Path('/proc/self/status').read_text().splitlines():
            key=line.split(':')[0]
            if key in {'VmRSS','VmHWM'}:values['process_'+key.lower()+'_bytes']=int(line.split()[1])*1024
            elif key=='Threads':values['process_threads']=int(line.split()[1])
        try:
            from torch._dynamo.utils import counters
            from torch._inductor import metrics
            values.update(compile_unique_graphs=int(counters['stats']['unique_graphs']),
                compile_generated_kernels=int(metrics.generated_kernel_count))
        except (ImportError,AttributeError,KeyError):values['compile_counters_unavailable']=1
        return values

    def close(self):
        try:self.reset()
        finally:
            self.model=None
            try:
                if self.torch is not None:
                    import gc
                    gc.collect();self.torch.cuda.empty_cache()
            finally:
                try:
                    if self.replay_original is not None:
                        if self.torch.cuda.CUDAGraph.replay is not self.replay_hook:
                            raise RuntimeError('Owned CUDA graph instrumentation changed')
                        self.torch.cuda.CUDAGraph.replay=self.replay_original
                        self.replay_original=None;self.replay_hook=None
                finally:
                    if self.lease is not None:self.lease.close();self.lease=None


class VoxBackend:
    """Bounded async bridge; the executor has one thread for its full lifetime."""
    def __init__(self,config,session_factory=NativeSession):
        self.config=config;self.factory=session_factory;self.session=None
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='aegis-voxcpm-owner')
        self.state='not_started';self.initializing=None;self.ready_event=asyncio.Event()
        self.warmup=[];self.failure_code=None;self._stops=set();self._closed=False
        self.load_seconds=0.;self.ready_seconds=0.

    @property
    def is_ready(self):return self.state=='ready' and not self._closed

    async def start(self):
        if self.initializing is None:
            self.state='loading';self.initializing=asyncio.create_task(self._initialize(),name='aegis-voxcpm-warmup')

    async def _initialize(self):
        initialized=time.perf_counter()
        def work():
            self.session=self.factory(self.config)
            try:
                before=time.perf_counter();self.session.load();self.load_seconds=time.perf_counter()-before
                warmup_text='Preparando a voz experimental da Aegis.'
                if self.config.frontend=='f5-refinement-1':
                    from aegis_speech.experimental_text import RefinedTextNormalizer
                    warmup_text=RefinedTextNormalizer().normalize(warmup_text).normalized_text
                for index in range(3):
                    start=time.perf_counter();count=0
                    for block in self._pcm(warmup_text,threading.Event()): count+=len(block)
                    if not count:raise RuntimeError('Warm-up returned no PCM')
                    self.warmup.append(dict(call=index,condition='cold/compile-first' if index==0 else 'warmup',
                        thread_id=threading.get_ident(),pcm_bytes=count,seconds=time.perf_counter()-start,
                        frontend=self.config.frontend,text_sha256=hashlib.sha256(warmup_text.encode()).hexdigest(),
                        diagnostics=dict(getattr(self.session,'last_observation',{}))))
            except BaseException:
                self.session.close();self.session=None;raise
        try:
            await asyncio.get_running_loop().run_in_executor(self.executor,work)
            self.ready_seconds=time.perf_counter()-initialized
            if not self._closed:self.state='ready';self.ready_event.set()
        except Exception as error:
            self.state='failed';self.failure_code='VOX_INITIALIZATION_FAILED'
            import traceback
            # Startup has no user input. Keep the actionable trace in the
            # operator's bounded stderr log, never in an HTTP response.
            traceback.print_exception(error,file=sys.stderr)

    def validate(self,texts):
        if not self.is_ready:raise RuntimeError('Vox backend is not ready')
        # The tokenizer is immutable CPU-only. Avoid queuing preflight behind
        # an active GPU stream, which would defeat the gateway request queue.
        return [capacity_guard(self.session.token_count(text),self.config) for text in texts]

    def numeric_metrics(self):
        values=dict(ready=int(self.is_ready),warmup_completed=len(self.warmup),
            load_seconds=self.load_seconds,ready_seconds=self.ready_seconds,initialization_failed=int(self.state=='failed'))
        if self.session is not None:values.update(getattr(self.session,'last_observation',{}))
        startup=[getattr(self.session,'load_observation',{})] if self.session is not None else []
        startup.extend(row.get('diagnostics',{}) for row in self.warmup)
        for diagnostics in startup:
            for key,value in diagnostics.items():
                if key.startswith(('cuda_peak_','process_vmhwm','compile_cache_')):
                    values['startup_max_'+key]=max(values.get('startup_max_'+key,0),value)
        return {'aegis_voxcpm_'+name:float(value) for name,value in values.items()}

    def _pcm(self,text,stop):
        iterator=None;resampler=HalfRateFIR();completed=False;chunks=0;emitted=0
        guard=capacity_guard(self.session.token_count(text),self.config)
        try:
            iterator=iter(self.session.generate(text))
            while not stop.is_set():
                try:chunk=next(iterator)
                except StopIteration:completed=True;break
                chunks+=1
                if stop.is_set():break
                block=pcm16_bytes(resampler.push(native_numpy(chunk)))
                if block:emitted+=len(block);yield block
            if completed and not stop.is_set():
                if chunks>=guard['audio_patch_limit']:
                    raise GenerationLimitReached('Native audio patch limit reached; completion is uncertain')
                block=pcm16_bytes(resampler.flush())
                if block:emitted+=len(block);yield block
                if not emitted:raise RuntimeError('Native EOF without playable PCM')
        finally:
            if not resampler.closed:resampler.cancel()
            try:
                if iterator is not None and hasattr(iterator,'close'):iterator.close()
            finally:
                try:self.session.reset()
                except BaseException:
                    self.state='failed';self.failure_code='VOX_RESET_FAILED';raise

    async def stream(self,text,cancelled):
        if not self.is_ready:raise RuntimeError('Vox backend is not ready')
        if cancelled.is_set():return
        self.validate([text]);loop=asyncio.get_running_loop();queue=asyncio.Queue(self.config.max_buffer_chunks)
        stop=threading.Event();self._stops.add(stop)
        def produce():
            iterator=self._pcm(text,stop)
            try:
                for block in iterator:
                    if stop.is_set():break
                    pending=asyncio.run_coroutine_threadsafe(queue.put(block),loop);start=time.monotonic()
                    while True:
                        try:pending.result(timeout=.05);break
                        except TimeoutError:
                            if stop.is_set() or time.monotonic()-start>=self.config.client_stall_seconds:
                                pending.cancel()
                                if not stop.is_set():raise TimeoutError('Vox client stalled')
                                return
            finally:iterator.close()
        future=asyncio.wrap_future(self.executor.submit(produce));cancel=asyncio.create_task(cancelled.wait());get=None
        try:
            while not cancelled.is_set():
                if future.done() and queue.empty():await future;break
                get=asyncio.create_task(queue.get())
                finished,_=await asyncio.wait((get,future,cancel),return_when=asyncio.FIRST_COMPLETED)
                if cancel in finished:break
                if get in finished:
                    block=get.result();get=None
                    if not cancelled.is_set():yield block
                elif future in finished:
                    get.cancel();await asyncio.gather(get,return_exceptions=True);get=None
                    await future
        finally:
            stop.set();cancel.cancel()
            if get is not None:get.cancel()
            await asyncio.gather(cancel,*([get] if get is not None else []),return_exceptions=True)
            try:
                try:await asyncio.shield(future)
                except asyncio.CancelledError:
                    await future;raise
            finally:self._stops.discard(stop)

    async def close(self):
        if self.state=='closed':return
        self._closed=True;self.state='closing'
        for stop in list(self._stops):stop.set()
        def work():
            try:
                if self.session:self.session.close()
            finally:self.session=None
        try:
            if self.initializing:await asyncio.shield(self.initializing)
        finally:
            try:
                cleanup=asyncio.get_running_loop().run_in_executor(self.executor,work)
                try:await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup;raise
            finally:self.executor.shutdown(wait=True);self.state='closed'
