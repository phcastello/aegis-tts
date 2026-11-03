"""Prepare or probe the experimental gateway; default is a GPU-free plan.

Execute is exclusively an owner GPU-worker job. It owns and stops only its
uvicorn child, leaves production/Qwen alone and never opens the final corpus.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import http.client
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import wave
import zipfile

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/var/tmp/aegis-v2-work')
PYTHON=Path('/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python')
WHEELS={
 'fastapi':dict(version='0.115.12',bytes=95164,sha256='e94613d6c05e27be7ffebdd6ea5f388112e5e430c8f7d6494a9d1d88d43e814d',
  url='https://files.pythonhosted.org/packages/50/b3/b51f09c2ba432a576fe63758bddc81f78f0c6309d9e5c10d194313bf021e/fastapi-0.115.12-py3-none-any.whl'),
 'starlette':dict(version='0.46.2',bytes=72037,sha256='595633ce89f8ffa71a015caed34a5b2dc1c0cdb3f0f1fbd1e69339cf2abeec35',
  url='https://files.pythonhosted.org/packages/8b/0c/9d30a4ebeb6db2b25a841afbb80f6ef9a854fc3b41be131d249a977b4959/starlette-0.46.2-py3-none-any.whl'),
 'uvicorn':dict(version='0.34.0',bytes=62315,sha256='023dc038422502fa28a09c7a30bf2b6991512da7dcdb8fd35fe57cfc154126f4',
  url='https://files.pythonhosted.org/packages/61/14/33a3a1352cfa71812a3a21e8c9bfb83f60b0011f5e36f2b1399d51928209/uvicorn-0.34.0-py3-none-any.whl'),
}
RUNTIME_FILES=['gateway/default_reference_gateway.py','gateway/voxcpm_runtime.py','gateway/voxcpm_audio.py',
 'gateway/voxcpm_adapter.py','gateway/aegis_speech/experimental_text.py','gateway/aegis_speech/experimental_refinement.py','benchmarks/v2/benchmark.py']


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def save(path,value):
 path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
 with temporary.open('w',encoding='utf-8') as f:
  json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 temporary.chmod(0o600);temporary.replace(path)


def private(path):
 path=Path(path).resolve()
 if any((p/'.git').exists() for p in (path,*path.parents)):raise ValueError('Artifacts must stay outside Git')
 path.mkdir(parents=True,exist_ok=True,mode=0o700);return path


def working_usage():
 from training.v2_language_monitor import physical_usage
 roots=sorted(Path('/var/tmp').glob('aegis*'))+[Path('/home/phc/aegis-tts/dataset_final_124_7ppm.zip'),REPO,Path('/home/phc/.cache')]
 return physical_usage(roots)


def prepare_overlay(args):
 overlay=private(args.overlay);downloads=private(overlay.parent/'wheels')
 if (overlay/'manifest.json').exists():return verify_overlay(overlay)
 if list(overlay.iterdir()):raise ValueError('Overlay must be empty or have its verified manifest')
 if shutil.disk_usage(overlay).free<5*1024**3+5*1024**2:raise RuntimeError('Insufficient overlay reserve')
 receipts=[]
 for name,row in WHEELS.items():
  path=downloads/(name+'-'+row['version']+'-py3-none-any.whl')
  if not path.exists():
   temporary=path.with_suffix('.part')
   with urllib.request.urlopen(row['url'],timeout=30) as source,temporary.open('wb') as output:
    shutil.copyfileobj(source,output,length=65536)
   if temporary.stat().st_size!=row['bytes'] or sha(temporary)!=row['sha256']:raise ValueError('Wheel checksum failure')
   temporary.replace(path)
  if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Cached wheel differs')
  with zipfile.ZipFile(path) as wheel:
   for entry in wheel.infolist():
    target=(overlay/entry.filename).resolve()
    if not target.is_relative_to(overlay) or entry.filename.endswith('.pth'):raise ValueError('Unsafe wheel path')
    top=Path(entry.filename).parts[0]
    if top!=name and not top.startswith(name+'-'):raise ValueError('Unexpected wheel top-level module')
   wheel.extractall(overlay)
  receipts.append(dict(name=name,**row,path=str(path)))
 files=[dict(path=str(p.relative_to(overlay)),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(overlay.rglob('*')) if p.is_file()]
 save(overlay/'manifest.json',dict(wheels=receipts,files=files,scope='Only FastAPI/Starlette/Uvicorn; no Torch/Pydantic/Transformers overlay'))
 return verify_overlay(overlay)


def verify_overlay(overlay):
 value=json.loads((overlay/'manifest.json').read_text(encoding='utf-8'))
 expected={r['path'] for r in value['files']}
 # Ignore only generated Python bytecode; no extra .py, .pth or native module
 # may shadow the existing F5 environment's neural packages.
 actual={str(p.relative_to(overlay)) for p in overlay.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=overlay/'manifest.json'}
 if actual!=expected:raise ValueError('Unexpected/absent overlay files')
 for row in value['files']:
  if sha(overlay/row['path'])!=row['sha256']:raise ValueError('Overlay file changed')
 if {r['name']:r['sha256'] for r in value['wheels']}!={n:r['sha256'] for n,r in WHEELS.items()}:
  raise ValueError('Wheel pins changed')
 return dict(overlay=str(overlay),manifest_sha256=sha(overlay/'manifest.json'),files=len(expected),
             retained_bytes=sum(p.stat().st_size for p in overlay.parent.rglob('*') if p.is_file()))


def child_environment(args):
 env=os.environ.copy();env.update(PYTHONPATH=os.pathsep.join((str(args.overlay),str(REPO/'gateway'))),
  PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
  HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
  AEGIS_TTS_STREAM_STARTUP_MS=f'{getattr(args,"startup_ms",0.):g}')
 return env


def startup_ms(value):
 try:amount=float(value)
 except (ValueError,TypeError) as error:raise argparse.ArgumentTypeError('startup-ms must be finite and between0 and500') from error
 if not math.isfinite(amount) or not 0<=amount<=500:
  raise argparse.ArgumentTypeError('startup-ms must be finite and between0 and500')
 return amount


def candidate_name(args):
 return f'voxcpm2-lora-gateway-{args.frontend}-startup{args.startup_ms:g}ms'


def cpu_check(args):
 verify_overlay(args.overlay)
 env=child_environment(args);env['CUDA_VISIBLE_DEVICES']='';env.pop('AEGIS_TTS_BACKEND',None);env.pop('AEGIS_VOXCPM_CONFIG',None)
 code="""import sys,json,resource,importlib,importlib.metadata,importlib.util
from pathlib import Path
import fastapi,starlette,uvicorn,pydantic,httpx,torch
import default_reference_gateway,voxcpm_runtime
assert default_reference_gateway.BACKEND_KIND=='qwen'
assert str(Path(torch.__file__).resolve()).startswith(str(Path(sys.prefix).resolve())+'/')
assert str(Path(pydantic.__file__).resolve()).startswith(str(Path(sys.prefix).resolve())+'/')
sys.path.insert(0,sys.argv[1]+'/src')
from voxcpm.model.voxcpm2 import VoxCPM2Model,LoRAConfig
assert not torch.cuda.is_initialized()
names=['fastapi','starlette','uvicorn','pydantic','torch','transformers','httpx','anyio','h11','click','PyYAML']
print(json.dumps(dict(cpu_only=True,cuda_initialized=False,model_constructed=False,python=sys.executable,
 versions={n:importlib.metadata.version(n) for n in names},torch_path=torch.__file__,pydantic_path=pydantic.__file__,
 fastapi_path=fastapi.__file__,uvicorn_path=uvicorn.__file__,source=sys.argv[1],
 simplejson_available=importlib.util.find_spec('simplejson') is not None,
 sortedcontainers_available=importlib.util.find_spec('sortedcontainers') is not None,
 peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)))
"""
 result=subprocess.run([str(args.python),'-c',code,str(args.source)],env=env,cwd=REPO,
  capture_output=True,text=True,encoding='utf-8',timeout=120)
 if result.returncode:raise RuntimeError('Actual target-Python CPU imports failed: '+result.stderr[-6000:])
 value=json.loads(result.stdout.splitlines()[-1]);save(args.run/'cpu-import-check.json',value);return value


class LimitedLog:
 def __init__(self,path,limit=32*1024**2,retain=4*1024**2):
  self.path=Path(path);self.limit=limit;self.retain=retain;self.truncations=0;self.total_bytes=0
 def pump(self,stream):
  with self.path.open('wb') as output:
   while True:
    block=stream.read1(4096) if hasattr(stream,'read1') else stream.read(4096)
    if not block:break
    output.write(block);output.flush();self.total_bytes+=len(block)
    if output.tell()>self.limit:
     output.seek(-self.retain,os.SEEK_END);tail=output.read() if output.readable() else None
     # A single writer owns this file; reopen read-only before truncating.
     if tail is None:
      with self.path.open('rb') as source:source.seek(-self.retain,os.SEEK_END);tail=source.read()
     output.seek(0);output.truncate();output.write(tail);output.flush();self.truncations+=1


def process_identity(pid):
 raw=Path(f'/proc/{pid}/stat').read_text();tail=raw[raw.rfind(')')+2:].split()
 return dict(pid=pid,start_ticks=int(tail[19]),uid=Path(f'/proc/{pid}').stat().st_uid)


@contextmanager
def cleanup_signals():
 previous={signum:signal.getsignal(signum) for signum in (signal.SIGTERM,signal.SIGINT)}
 def interrupted(signum,frame):
  # Do not let a second cooperative signal interrupt ownership cleanup.
  for value in previous:signal.signal(value,signal.SIG_IGN)
  raise KeyboardInterrupt('Gateway probe interrupted by signal'+str(signum))
 for signum in previous:signal.signal(signum,interrupted)
 try:yield
 finally:
  for signum,handler in previous.items():signal.signal(signum,handler)


@contextmanager
def defer_spawn_signals():
 """Finish binding Popen/identity before delivering a cooperative stop.

 This changes only the caller's Python handlers, never the signal mask.
 Caught handlers revert to defaults on exec, so uvicorn can receive TERM.
 """
 previous={signum:signal.getsignal(signum) for signum in (signal.SIGTERM,signal.SIGINT)};pending=[]
 def remember(signum,frame):
  if not pending:pending.append(signum)
 for signum in previous:signal.signal(signum,remember)
 try:yield
 finally:
  for signum,handler in previous.items():signal.signal(signum,handler)
  if pending:
   signum=pending[0];handler=previous[signum]
   if callable(handler):handler(signum,None)
   elif handler!=signal.SIG_IGN:raise KeyboardInterrupt('Deferred spawn signal'+str(signum))


def stop_owned(process,identity):
 if process.poll() is not None:return
 try:current=process_identity(process.pid)
 except FileNotFoundError:
  if process.poll() is not None:return
  raise
 if current!=identity or identity['uid']!=os.getuid():raise RuntimeError('Owned child identity changed; refuse signal')
 process.send_signal(signal.SIGTERM)
 try:process.wait(timeout=30)
 except subprocess.TimeoutExpired:
  if process_identity(process.pid)!=identity:raise RuntimeError('Child identity changed before SIGKILL')
  process.kill();process.wait(timeout=10)


def request(port,path,method='GET',payload=None,timeout=30):
 connection=http.client.HTTPConnection('127.0.0.1',port,timeout=timeout)
 body=json.dumps(payload,ensure_ascii=False).encode() if payload is not None else None
 connection.request(method,path,body,{'Content-Type':'application/json','Accept-Encoding':'identity'})
 response=connection.getresponse()
 return connection,response


def complete_request(port,path,method='GET',payload=None,timeout=30):
 connection,response=request(port,path,method,payload,timeout)
 try:return response.status,dict((k.lower(),v) for k,v in response.getheaders()),response.read(16*1024**2)
 finally:connection.close()


def wav_contract(data):
 if len(data)<=44:raise ValueError('Header-only/empty WAV')
 with wave.open(io.BytesIO(data)) as wav:
  if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())!=(24000,1,2):raise ValueError('WAV format differs')
  frames=wav.getnframes();pcm=wav.readframes(frames)
  if frames<1 or len(pcm)!=frames*2:raise ValueError('Incomplete WAV')
  if len(data)!=44+len(pcm):raise ValueError('Unexpected/truncated gateway PCM WAV container')
 return dict(frames=frames,pcm_bytes=len(pcm),sample_rate=24000,channels=1,pcm_sha256=hashlib.sha256(pcm).hexdigest())


def status_receipt(row,status,body):
 value=json.loads(body) if status==200 else dict(status='unavailable',http_status=status)
 identifier=row['headers']['x-aegis-request-id']
 if value.get('request_id',identifier)!=identifier:raise ValueError('Status request ID differs')
 return dict(value,case=row['case'],repetition=row['repetition'],request_id=identifier)


def contracts(args,profile):
 results={}
 status,headers,body=complete_request(args.port,'/v1/audio/speech','POST',
  dict(input='Esta fala inédita verifica o arquivo completo e as pausas.',stream=False,response_format='wav'),120)
 if status!=200 or headers.get('x-aegis-voice-profile')!=profile or headers.get('x-aegis-tts-frontend')!=args.frontend:
  raise ValueError('WAV/profile/frontend contract failed')
 results['wav']=wav_contract(body)
 path=args.run/'contract-complete.wav';path.write_bytes(body);results['wav']['sha256']=sha(path)
 connection=response=None
 try:
  connection,response=request(args.port,'/v1/aegis/speech','POST',
   dict(text='Esta solicitação verifica o cancelamento em uma frase longa. '*12,stream=True,response_format='pcm'),120)
  if response.status!=200:raise ValueError('Cancellation request failed')
  if response.getheader('X-Aegis-Voice-Profile')!=profile or response.getheader('X-Aegis-TTS-Frontend')!=args.frontend:
   raise ValueError('Streaming profile/frontend differs')
  identifier=response.getheader('X-Aegis-Request-Id');first=response.read1(4096)
  if len(first)%2:first+=response.read(1)
  if len(first)<2 or len(first)%2 or first[:4]==b'RIFF':raise ValueError('Cancel must follow a valid PCM block')
  status,_,body=complete_request(args.port,'/v1/aegis/speech/'+identifier,'DELETE')
  if status!=200:raise ValueError('DELETE contract failed')
  results['cancel']=dict(request_id=identifier,first_pcm_bytes=len(first),delete=json.loads(body))
 finally:
  if connection is not None:connection.close()
 deadline=time.monotonic()+30
 while time.monotonic()<deadline:
  status,_,body=complete_request(args.port,'/v1/aegis/speech/'+identifier)
  if status!=200:raise ValueError('Cancellation status unavailable')
  state=json.loads(body)['status']
  if state=='cancelled':break
  time.sleep(.2)
 if state!='cancelled':raise ValueError('Cancellation failed to finish: '+state)
 results['cancel']['final_status']=state
 status,headers,body=complete_request(args.port,'/v1/audio/speech','POST',
  dict(input='Outra frase inédita confirma a retomada depois do cancelamento.',stream=False,response_format='wav'),120)
 if status!=200 or headers.get('x-aegis-voice-profile')!=profile:raise ValueError('Next generation after cancel failed')
 results['next_after_cancel']=wav_contract(body)
 for route in ('/health','/ready','/v1/aegis/queue'):
  status,_,body=complete_request(args.port,route)
  if status!=200:raise ValueError('Post-contract health/readiness failed')
  results[route]=status
 return results


def execute(args):
 parent=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().decode(errors='replace')
 if 'v2_gpu_worker' not in parent:raise RuntimeError('Execute only as an exclusive owner GPU-worker job')
 if not (ROOT/'pause-asr').exists():raise RuntimeError('Pause ASR/CPU competitors for performance measurements')
 if args.checkpoint is None or args.checkpoint_sha256 is None:raise ValueError('Stable checkpoint and pinned SHA required')
 if (args.run/'recipe.json').exists():raise ValueError('Fresh output required')
 from training.v2_voxcpm_optimized_probe import require_idle_cpu
 require_idle_cpu()
 verify_overlay(args.overlay)
 cpu_check(args)
 if sha(args.checkpoint)!=args.checkpoint_sha256:raise ValueError('Selected checkpoint differs')
 receipt=json.loads(args.checkpoint.with_suffix('.sha256.json').read_text(encoding='utf-8'))
 if receipt['sha256']!=args.checkpoint_sha256:raise ValueError('Checkpoint sidecar differs')
 usage=working_usage();extra=1024**3
 if usage['permission_errors'] or usage['allocated_bytes']+extra>40_000_000_000 or shutil.disk_usage(args.run).free<5*1024**3+extra:
  raise RuntimeError('Need40GB working budget,1GiB experiment margin and5GiB free reserve')
 with socket.socket() as check:check.bind(('127.0.0.1',args.port))
 if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
  raise RuntimeError('GPU is not idle')
 code_hashes={p:sha(REPO/p) for p in RUNTIME_FILES}
 config=dict(source=str(args.source),model=str(args.model),adapter=str(args.checkpoint),adapter_sha256=args.checkpoint_sha256,
  cache=str(args.cache),gpu_lock=str(args.run.parent/'gateway-gpu.lock'),
  compiler_prerequisites=str(args.compiler_prerequisites) if args.compiler_prerequisites else None,dsp='passthrough',instrument_replays=True,
  frontend=args.frontend)
 config_path=args.run/'runtime.private.json';save(config_path,config)
 env=child_environment(args);env.update(AEGIS_TTS_BACKEND='voxcpm2',AEGIS_VOXCPM_CONFIG=str(config_path),
  AEGIS_TTS_ONSET_TRIM='off',AEGIS_TTS_PROFILE_LATENCY='1',
  AEGIS_TTS_SOURCE_COMMIT=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
 front={'original-nfc':'OriginalNFC','f5-refinement-1':'F5Refinement1','ptbr-numbers-v1':'PtbrNumbers1'}[args.frontend]
 profile=f'AegisVoicev2.0-VoxCPM2-LoRA-r8-N10-Passthrough-{front}-Experimental'
 command=[str(args.python),'-m','uvicorn','default_reference_gateway:app','--host','127.0.0.1','--port',str(args.port),
  '--workers','1','--loop','asyncio','--http','h11','--no-access-log','--log-level','warning']
 recipe=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),code_hashes=code_hashes,runner_sha256=sha(__file__),
  checkpoint=dict(path=str(args.checkpoint),sha256=args.checkpoint_sha256,step=receipt.get('step')),working=usage,
  planned_extra_bytes=extra,config_sha256=sha(config_path),command=command,
  frontend=args.frontend,startup_ms=args.startup_ms,candidate=candidate_name(args),
  scope='Experimental HTTP gateway; service warmup3 noncorpus calls only; no per-case warmup; final corpus unopened')
 save(args.run/'recipe.json',recipe)
 process=None;identity=None;logger=LimitedLog(args.run/'gateway.log');log_thread=None;stop=threading.Event();watch=None;startup_sampler=None
 try:
  spec=importlib.util.spec_from_file_location('aegis_gateway_probe_metrics',REPO/'benchmarks/v2/benchmark.py')
  common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
  startup_sampler=common.GpuSampler(args.run/'startup-gpu.jsonl',True);startup_sampler.__enter__()
  begun=time.perf_counter()
  with defer_spawn_signals():
   process=subprocess.Popen(command,cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
   identity=process_identity(process.pid)
  save(args.run/'process.json',dict(**identity,command=command))
  log_thread=threading.Thread(target=logger.pump,args=(process.stdout,),daemon=True);log_thread.start()
  def budget_watch():
   while not stop.wait(5):
    from training.v2_language_monitor import physical_usage
    # The runtime's CacheGuard owns cache traversal. Read its numeric snapshot;
    # do not add a second cache crawl to the client benchmark process.
    cache=None
    try:
     status,_,body=complete_request(args.port,'/metrics',timeout=2)
     if status==200:
      metrics={line.split()[0]:float(line.split()[1]) for line in body.decode().splitlines() if line and not line.startswith('#')}
      cache=int(metrics.get('aegis_voxcpm_compile_cache_bytes',initial_cache))
    except (OSError,http.client.HTTPException,ValueError):pass
    free=shutil.disk_usage(args.run).free
    output=physical_usage([args.run])['allocated_bytes']
    # No full project crawl during latency observations. Exclusive GPU/paused
    # ASR plus bounded cache/output use the preflight's1GiB experiment margin.
    estimated=usage['allocated_bytes']+max(0,(cache if cache is not None else initial_cache)-initial_cache)+output
    if (cache is not None and cache>512*1024**2) or free<5*1024**3 or output>512*1024**2 or estimated>40_000_000_000:
     save(args.run/'safety-stop.json',dict(cache_bytes=cache,free_bytes=free,estimated_working_bytes=estimated))
     stop_owned(process,identity);return
  initial_cache=0
  watch=threading.Thread(target=budget_watch,daemon=True);watch.start()
  startup=[];deadline=time.monotonic()+args.ready_timeout
  while time.monotonic()<deadline:
   if process.poll() is not None:raise RuntimeError('Gateway exited during startup')
   try:
    status,headers,_=complete_request(args.port,'/ready',timeout=2)
    startup.append(dict(elapsed_seconds=time.perf_counter()-begun,status=status))
    if headers.get('x-aegis-tts-backend-state')=='failed':raise RuntimeError('Gateway initialization failed; inspect bounded gateway.log')
    if status==200:
     if headers.get('x-aegis-tts-dsp-variant')!='passthrough':raise ValueError('Gateway DSP mislabeled')
     break
   except (OSError,http.client.HTTPException):pass
   save(args.run/'progress.json',dict(phase='startup',elapsed_seconds=time.perf_counter()-begun,pid=process.pid))
   time.sleep(1)
  else:raise TimeoutError('Gateway readiness timeout')
  ready_observed_seconds=time.perf_counter()-begun
  startup_sampler.__exit__(None,None,None)
  if startup_sampler.error:raise RuntimeError('Startup GPU sampling failed: '+startup_sampler.error)
  for line in (args.run/'startup-gpu.jsonl').read_text(encoding='utf-8').splitlines():
   if {int(p[0]) for p in json.loads(line)['processes']}-{process.pid}:raise RuntimeError('Another GPU process overlapped startup')
  save(args.run/'startup.json',dict(startup_seconds=ready_observed_seconds,polls=startup,poll_interval_seconds=1,
       sampled_gpu_peak_mib=startup_sampler.peak,
       warmup='Backend three calls of one noncorpus phrase, same persistent inference thread; no case prewarm'))
  status,_,body=complete_request(args.port,'/metrics');(args.run/'metrics-before.txt').write_bytes(body)
  benchmark=args.run/'benchmark'
  command=[str(args.python),str(REPO/'benchmarks/v2/benchmark.py'),'run','--endpoint',f'http://127.0.0.1:{args.port}/v1/audio/speech',
   '--candidate',candidate_name(args),'--output',str(benchmark),'--repetitions','3','--condition','warm',
   '--sample-gpu','--expected-pids',str(process.pid),'--cold-evidence',str(args.run/'startup.json')]
  with (args.run/'benchmark.log').open('wb') as output:
   completed=subprocess.run(command,cwd=REPO,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=2400)
  if completed.returncode:raise RuntimeError('Common HTTP benchmark failed')
  rows=[json.loads(l) for l in (benchmark/'results.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
  if len(rows)!=21:raise ValueError('Common seven-case n3 matrix is incomplete')
  if any(r.get('outcome')!='ok' for r in rows):raise ValueError('Common HTTP matrix contains failures/no speech')
  statuses=[]
  for row in rows:
   if row['headers'].get('x-aegis-voice-profile')!=profile or row['headers'].get('x-aegis-tts-frontend')!=args.frontend:
    raise ValueError('Matrix contains a different voice profile/frontend')
   identifier=row['headers'].get('x-aegis-request-id')
   if not identifier:raise ValueError('Missing gateway request ID')
   status,_,body=complete_request(args.port,'/v1/aegis/speech/'+identifier)
   statuses.append(status_receipt(row,status,body))
  save(args.run/'request-statuses.json',statuses)
  if any(row['status']!='completed' for row in statuses):
   raise ValueError('HTTP PCM may be partial: at least one gateway request did not complete')
  save(args.run/'service-warmup-cohort.json',dict(
   matrix='Service warm only. Each case rep0 is novel after readiness, reps1/2 repeated; no per-case warmup',
   rows=[dict(case=r['case'],repetition=r['repetition'],request_id=r['headers'].get('x-aegis-request-id'),
     model_text_condition='novel_after_service_ready' if r['repetition']==0 else 'repeated_after_service_ready') for r in rows]))
  corpus={r['id']:r for r in json.loads((REPO/'benchmarks/v2/validation.pt-BR.json').read_text(encoding='utf-8'))}
  spoken=json.loads((REPO/'benchmarks/v2/spoken-validation.pt-BR.json').read_text(encoding='utf-8'))['cases']
  jobs=[];events=[]
  for row in rows:
   identifier=row['headers'].get('x-aegis-request-id')
   if identifier:
    status,_,body=complete_request(args.port,'/v1/aegis/speech/'+identifier+'/events')
    if status==200:events.append(dict(case=row['case'],repetition=row['repetition'],request_id=identifier,sse=body.decode('utf-8')))
   if row['repetition']==0:
    from gateway.aegis_speech.experimental_text import normalizer_for
    normalizer=normalizer_for(args.frontend)
    jobs.append(dict(candidate=candidate_name(args),case=row['case'],wav=str(benchmark/row['wav']),
     expected_verbatim=corpus[row['case']]['text'],expected_spoken=spoken[row['case']],
     expected_effective=normalizer.normalize(corpus[row['case']]['text']).normalized_text,frontend=args.frontend,
     cohort='validation-gateway-service-only-warmup-rep0',product_audio_valid=True))
  save(args.run/'request-events.json',events);save(args.run/'asr-jobs.json',dict(jobs=jobs))
  status,_,body=complete_request(args.port,'/metrics');(args.run/'metrics-after.txt').write_bytes(body)
  save(args.run/'contracts.json',contracts(args,profile))
  recipe['status']='pass';recipe['benchmark_observations']=len(rows)
 except BaseException as error:
  recipe.update(status='failed',error=repr(error));raise
 finally:
  stop.set()
  if watch:watch.join(5)
  if startup_sampler:startup_sampler.__exit__(None,None,None)
  cleanup_error=None
  try:
   if process is not None and identity is not None:stop_owned(process,identity)
  except Exception as error:cleanup_error=repr(error)
  finally:
   if log_thread:log_thread.join(5)
  recipe.update(service_left_running=bool(process is not None and process.poll() is None),logs_total_bytes=logger.total_bytes,log_truncations=logger.truncations,
   owned_process=identity,
   checkpoint_unchanged=sha(args.checkpoint)==args.checkpoint_sha256,
   runtime_sources_unchanged={p:sha(REPO/p)==h for p,h in code_hashes.items()},final_working=working_usage())
  if cleanup_error or not recipe['checkpoint_unchanged'] or not all(recipe['runtime_sources_unchanged'].values()):
   recipe.update(status='failed',final_guard_error=cleanup_error or 'Checkpoint/runtime changed during probe')
  save(args.run/'result.json',recipe)
  if cleanup_error:raise RuntimeError('Owned gateway cleanup failed: '+cleanup_error)
  if not recipe['checkpoint_unchanged'] or not all(recipe['runtime_sources_unchanged'].values()):
   raise RuntimeError('Checkpoint/runtime changed during probe; evidence invalid')
 return recipe


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--action',choices=['plan','prepare-overlay','cpu-check','execute'],default='plan')
 p.add_argument('--run',type=Path,default=ROOT/'voxcpm-gateway/preparation')
 p.add_argument('--overlay',type=Path,default=ROOT/'voxcpm-gateway/deps')
 p.add_argument('--python',type=Path,default=PYTHON)
 p.add_argument('--source',type=Path,default=ROOT/'voxcpm2/source');p.add_argument('--model',type=Path,default=ROOT/'voxcpm2/model')
 p.add_argument('--cache',type=Path,default=ROOT/'voxcpm2/compile-cache')
 p.add_argument('--compiler-prerequisites',type=Path,default=ROOT/'voxcpm2/compile-prereqs.json')
 p.add_argument('--checkpoint',type=Path);p.add_argument('--checkpoint-sha256')
 p.add_argument('--frontend',choices=['original-nfc','f5-refinement-1','ptbr-numbers-v1'],default='original-nfc')
 p.add_argument('--startup-ms',type=startup_ms,default=0.)
 p.add_argument('--port',type=int,default=8002);p.add_argument('--ready-timeout',type=float,default=600)
 a=p.parse_args();os.umask(0o077);a.run=private(a.run)
 # A venv interpreter is normally a symlink to /usr/bin/python3. Resolving
 # that symlink changes sys.prefix and drops the entire selected environment.
 a.python=a.python.absolute()
 for name in ('overlay','source','model','cache','compiler_prerequisites','checkpoint'):
  if getattr(a,name) is not None:setattr(a,name,getattr(a,name).resolve())
 if a.port!=8002:raise ValueError('Prepared probe is isolated on port8002')
 if a.action=='plan':
  value=dict(action='plan',gpu_executed=False,service_launched=False,python=str(a.python),overlay=str(a.overlay),
   pending_checkpoint=str(a.checkpoint) if a.checkpoint else None,pending_sha256=a.checkpoint_sha256,frontend=a.frontend,
   startup_ms=a.startup_ms,candidate=candidate_name(a),
   matrix='seven validation phrases n3, service-only warmup; final six unopened',
   order=['overlay CPU import check','exclusive GPU worker','start one uvicorn8002','readiness and cold startup',
    'common HTTP benchmark without per-case warmup','WAV/PCM/firstPCM DELETE/nextcall/readiness','stop only owned uvicorn'],
   cpu_commands=['--action prepare-overlay','--action cpu-check'],
   execute_template='--action execute --run NEW_PRIVATE_RUN --checkpoint STABLE_SELECTED_CP --checkpoint-sha256 SHA256')
  save(a.run/'plan.json',value)
 elif a.action=='prepare-overlay':value=prepare_overlay(a)
 elif a.action=='cpu-check':value=cpu_check(a)
 else:
  with cleanup_signals():value=execute(a)
 print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
