#!/usr/bin/env python3
"""Reproducible, GUI-free Graillon 3 LV2/JACK laboratory.

The module is intentionally self contained: normal TTS dependencies never load
JACK.  All subprocesses use argv lists and a private JACK server environment.
"""
from __future__ import annotations
import argparse, csv, fcntl, hashlib, html, json, math, os, platform, re, shutil, signal, subprocess, sys, tempfile, time, wave
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LAB=ROOT/'experiments/glados-filter-lab'
URI='https://www.auburnsounds.com/products/Graillon.html40733133#mono'
NOTES=('c','c_sharp','d','d_sharp','e','f','f_sharp','g','g_sharp','a','a_sharp','b')
DEFAULTS={
 'AEGIS_GRAILLON_PLUGIN_URI':URI, 'AEGIS_GRAILLON_BUNDLE':str(LAB/'plugins/lv2/Auburn Sounds Graillon 3.lv2'),
 'AEGIS_GRAILLON_INFO_FILE':str(LAB/'graillon-lv2-info.txt'), 'AEGIS_GRAILLON_LV2_PATH':f'{LAB}/plugins/lv2:/usr/lib/lv2:/usr/local/lib/lv2:{Path.home()}/.lv2',
 'AEGIS_GRAILLON_JACK_SERVER':'aegis-glados-lab','AEGIS_GRAILLON_SAMPLE_RATE':'24000','AEGIS_GRAILLON_BLOCK_SIZE':'256','AEGIS_GRAILLON_PERIODS':'3','AEGIS_GRAILLON_REALTIME':'0',
 'AEGIS_GRAILLON_STARTUP_TIMEOUT_SECONDS':'15','AEGIS_GRAILLON_SHUTDOWN_TIMEOUT_SECONDS':'10','AEGIS_GRAILLON_RENDER_PADDING_SECONDS':'2','AEGIS_GRAILLON_PREROLL_MS':'500','AEGIS_GRAILLON_POSTROLL_MS':'500',
 'AEGIS_GRAILLON_JACKD_BIN':'jackd','AEGIS_GRAILLON_JALV_BIN':'jalv','AEGIS_GRAILLON_JACK_LSP_BIN':'jack_lsp','AEGIS_GRAILLON_JACK_CONNECT_BIN':'jack_connect','AEGIS_GRAILLON_FFMPEG_BIN':'ffmpeg','AEGIS_GRAILLON_FFPROBE_BIN':'ffprobe','AEGIS_GRAILLON_LV2LS_BIN':'lv2ls','AEGIS_GRAILLON_LV2INFO_BIN':'lv2info',
 'AEGIS_GRAILLON_PRESETS_CONFIG':str(ROOT/'config/graillon-presets.yaml'),'AEGIS_GRAILLON_INPUT':str(ROOT/'samples/glados-lab/source.wav'),'AEGIS_GRAILLON_OUTPUT_DIR':str(ROOT/'samples/glados-lab/graillon-results'),'AEGIS_GRAILLON_REPORT_DIR':str(ROOT/'reports/glados-lab/graillon'),
 'AEGIS_GRAILLON_KEEP_JACK_RUNNING':'0','AEGIS_GRAILLON_KEEP_RAW_OUTPUTS':'1','AEGIS_GRAILLON_LOG_LEVEL':'INFO','AEGIS_GRAILLON_INTEGRATION_TESTS':'0','AEGIS_GRAILLON_ASSUME_YES':'0'}
BOOLS={'AEGIS_GRAILLON_REALTIME','AEGIS_GRAILLON_KEEP_JACK_RUNNING','AEGIS_GRAILLON_KEEP_RAW_OUTPUTS','AEGIS_GRAILLON_INTEGRATION_TESTS','AEGIS_GRAILLON_ASSUME_YES'}
INTS={'AEGIS_GRAILLON_SAMPLE_RATE','AEGIS_GRAILLON_BLOCK_SIZE','AEGIS_GRAILLON_PERIODS','AEGIS_GRAILLON_STARTUP_TIMEOUT_SECONDS','AEGIS_GRAILLON_SHUTDOWN_TIMEOUT_SECONDS','AEGIS_GRAILLON_RENDER_PADDING_SECONDS','AEGIS_GRAILLON_PREROLL_MS','AEGIS_GRAILLON_POSTROLL_MS'}
PATHS={k for k in DEFAULTS if k.endswith(('_BUNDLE','_FILE','_CONFIG','_INPUT','_DIR'))}

class LabError(RuntimeError): pass
def run(argv,env=None,**kw): return subprocess.run(argv,env=env,text=True,capture_output=True,check=False,**kw)
def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''): h.update(x)
 return h.hexdigest()
def bool_value(value):
 if str(value).lower() in ('1','true','yes','on'): return True
 if str(value).lower() in ('0','false','no','off'): return False
 raise LabError(f'booleano inválido: {value!r}')
def load_dotenv(path):
 """Strict dotenv reader; it deliberately never evaluates shell syntax."""
 got={}
 if not path or not Path(path).exists(): return got
 for n,line in enumerate(Path(path).read_text(encoding='utf8').splitlines(),1):
  line=line.strip()
  if not line or line.startswith('#'): continue
  m=re.fullmatch(r'([A-Z][A-Z0-9_]*)=(.*)',line)
  if not m: raise LabError(f'{path}:{n}: linha dotenv inválida')
  key,val=m.groups()
  if key not in DEFAULTS: raise LabError(f'{path}:{n}: chave desconhecida {key}')
  if val.startswith(('"',"'")): raise LabError(f'{path}:{n}: aspas não são aceitas no dotenv')
  got[key]=val
 return got
def validate_config(d):
 for k in d:
  if k not in DEFAULTS: raise LabError(f'chave desconhecida: {k}')
 for k in BOOLS: bool_value(d[k])
 for k in INTS:
  try: n=int(d[k])
  except ValueError: raise LabError(f'{k} precisa ser inteiro')
  if n<=0: raise LabError(f'{k} precisa ser positivo')
 if int(d['AEGIS_GRAILLON_SAMPLE_RATE']) not in (8000,12000,16000,22050,24000,44100,48000,96000): raise LabError('sample rate não suportado pelo laboratório')
 if int(d['AEGIS_GRAILLON_BLOCK_SIZE']) & (int(d['AEGIS_GRAILLON_BLOCK_SIZE'])-1): raise LabError('block size precisa ser potência de dois')
 if not re.fullmatch(r'https?://[^\s]+#mono',d['AEGIS_GRAILLON_PLUGIN_URI']): raise LabError('URI LV2 mono inválida')
 if not re.fullmatch(r'[A-Za-z0-9_.-]+',d['AEGIS_GRAILLON_JACK_SERVER']): raise LabError('nome do servidor JACK inválido')
 for k in PATHS:
  if '\x00' in d[k]: raise LabError(f'caminho inválido em {k}')
 for k in DEFAULTS:
  if k.endswith('_BIN') and (not d[k] or '/' in d[k] and not Path(d[k]).is_file()): raise LabError(f'executável inválido em {k}')
 return d
def config(args=None):
 envfile=getattr(args,'env_file',None) if args else None
 if envfile is None:
  candidate=LAB/'.env.graillon'; envfile=str(candidate) if candidate.exists() else None
 d=dict(DEFAULTS); d.update(load_dotenv(envfile)); d.update({k:os.environ[k] for k in DEFAULTS if k in os.environ})
 # CLI intentionally only overrides paths supplied by the operator.
 for attribute,key in [('input','AEGIS_GRAILLON_INPUT'),('output_dir','AEGIS_GRAILLON_OUTPUT_DIR'),('preset_config','AEGIS_GRAILLON_PRESETS_CONFIG'),('config','AEGIS_GRAILLON_PRESETS_CONFIG')]:
  value=getattr(args,attribute,None) if args else None
  if value: d[key]=value
 return validate_config(d)
def child_env(c,runtime=None):
 e=os.environ.copy(); e.update({'LV2_PATH':c['AEGIS_GRAILLON_LV2_PATH'],'JACK_DEFAULT_SERVER':c['AEGIS_GRAILLON_JACK_SERVER'],'JACK_NO_AUDIO_RESERVATION':'1'})
 if runtime: e['XDG_RUNTIME_DIR']=str(runtime)
 return e
def ports_from_lv2info(text):
 ports=[]; blocks=re.split(r'\n\s*Port (\d+):',text)
 for i in range(1,len(blocks),2):
  index,body=int(blocks[i]),blocks[i+1]
  symbol=re.search(r'^\s*Symbol:\s+(\S+)',body,re.M); name=re.search(r'^\s*Name:\s+(.+)',body,re.M)
  if not symbol: continue
  direction='input' if 'InputPort' in body else 'output' if 'OutputPort' in body else None
  typ='control' if 'ControlPort' in body else 'audio' if 'AudioPort' in body else 'event' if 'AtomPort' in body else 'unknown'
  item={'index':index,'symbol':symbol.group(1),'name':name.group(1).strip() if name else '', 'direction':direction,'type':typ}
  for label,key in [('Minimum','minimum'),('Maximum','maximum'),('Default','default')]:
   v=re.search(r'^\s*'+label+r':\s+([-+.0-9eE]+)',body,re.M)
   if v: item[key]=float(v.group(1))
  ports.append(item)
 return ports
def discover(c):
 info=Path(c['AEGIS_GRAILLON_INFO_FILE'])
 p=run([c['AEGIS_GRAILLON_LV2INFO_BIN'],c['AEGIS_GRAILLON_PLUGIN_URI']],env=child_env(c))
 text=p.stdout if p.returncode==0 else (info.read_text(encoding='utf8') if info.exists() else '')
 if not text: raise LabError('lv2info falhou e não há inventário local: '+p.stderr.strip())
 ports=ports_from_lv2info(text)
 return {'plugin_uri':c['AEGIS_GRAILLON_PLUGIN_URI'],'bundle':c['AEGIS_GRAILLON_BUNDLE'],'ports':ports,'lv2info_source':'live' if p.returncode==0 else 'file','lv2info_error':p.stderr.strip() if p.returncode else None}
def read_yaml(path):
 try:
  import yaml
 except ImportError as exc: raise LabError('PyYAML é necessário') from exc
 return yaml.safe_load(Path(path).read_text(encoding='utf8'))
def semantic_map(): return read_yaml(LAB/'graillon-semantic-map.yaml')
def validate_inventory(inv):
 sm=semantic_map(); by={p['symbol']:p for p in inv['ports']}
 missing=[s for s in sm.values() if s not in by]
 if missing: raise LabError('símbolos obrigatórios ausentes: '+', '.join(missing))
 for name,s in sm.items():
  p=by[s]
  if name in ('audio_input','audio_output','midi_input','latency_output'): continue
  if p['type']!='control' or p['direction']!='input' or p.get('minimum')!=0.0 or p.get('maximum')!=1.0: raise LabError(f'porta incompatível para {name}: {p}')
 if by[sm['audio_input']]['type']!='audio' or by[sm['audio_input']]['direction']!='input': raise LabError('entrada de áudio incompatível')
 if by[sm['audio_output']]['type']!='audio' or by[sm['audio_output']]['direction']!='output': raise LabError('saída de áudio incompatível')
 return by
def parse_ttl_presets(bundle):
 text=(Path(bundle)/'manifest.ttl').read_text(encoding='utf8')
 result={}
 for m in re.finditer(r'(vendor:preset\d+)\s+a pset:Preset\s*;(.+?)(?=\n\s*vendor:preset\d+\s|\Z)',text,re.S):
  uri,label=m.group(1),re.search(r'rdfs:label\s+"([^"]+)"',m.group(2))
  if not label: continue
  controls={s:float(v) for s,v in re.findall(r'lv2:symbol\s+"([^"]+)";\s*pset:value\s+([-+.0-9eE]+)',m.group(2))}
  result[label.group(1)]={'uri':URI.rsplit('#',1)[0]+'#'+uri.split(':',1)[1],'controls':controls,'source':'manifest.ttl'}
 return result
def base_controls(c,which):
 ps=parse_ttl_presets(c['AEGIS_GRAILLON_BUNDLE'])
 name={'default':'Default','official-hard-tune':'Hard Tune','round2-hard-tune-base':'Hard Tune'}.get(which,which)
 if name not in ps: raise LabError(f'estado base ausente: {which}')
 return ps[name]
def resolve_preset(pre,inv,c):
 by=validate_inventory(inv); sm=semantic_map(); result=dict(base_controls(c,pre.get('base_state','default'))['controls'])
 controls=pre.get('controls',{}) or {}; raw=pre.get('raw_controls',{}) or {}
 if not isinstance(controls,dict) or not isinstance(raw,dict): raise LabError(f"{pre.get('id')}: controls devem ser mapas")
 for k,v in controls.items():
  if k not in sm: raise LabError(f"{pre.get('id')}: controle semântico desconhecido {k}")
  raw[sm[k]]=v
 for s,v in raw.items():
  if s not in by: raise LabError(f"{pre.get('id')}: símbolo inexistente {s}")
  p=by[s]
  if p['type']!='control' or p['direction']!='input': raise LabError(f"{pre.get('id')}: {s} não é controle de entrada")
  try: v=float(v)
  except (TypeError,ValueError): raise LabError(f"{pre.get('id')}: valor inválido para {s}")
  if not p['minimum']<=v<=p['maximum']: raise LabError(f"{pre.get('id')}: {s} fora da faixa")
  result[s]=v
 # The original G-series lab inferred a few values from its preset ID.  That
 # compatibility path must never run for the controlled round-2 schema: its
 # YAML is deliberately explicit and is the source of truth for every
 # comparison dimension.
 if not pre.get('round2') and pre['id'] not in ('G00','G01','G02'):
  allowed={'G07':{'g','a'},'G08':{'g','a','d'},'G09':{'c','g','a'},'G10':{'c','d_sharp','f','g','g_sharp','a_sharp'},'G11':{'c','d','e','f','g','a','b'}}.get(pre['id'],set(NOTES))
  for n in NOTES: result[sm['allow_'+n]]=1.0 if n in allowed else 0.0
  for k,v in {'correction_enabled':1,'correction_amount':1,'wet_mix':1,'dry_mix':0,'midi_enabled':0,'pitch_shift':.5,'formant_shift':.5,'formant_enabled':0,'fx_enabled':0,'ptm_enabled':0,'output_gain':.5}.items(): result[sm[k]]=v
 return result
def jalv_args(c,controls,name='graillon'):
 # -p writes control-output changes (including the LV2 latency port) to the
 # per-run Jalv log; it does not request a UI.
 args=[c['AEGIS_GRAILLON_JALV_BIN'],'-i','-p','-x','-n',name]
 for symbol,value in sorted(controls.items()): args.extend(['-c',f'{symbol}={float(value):.9g}'])
 return args+[c['AEGIS_GRAILLON_PLUGIN_URI']]
class Runtime:
 def __init__(self,c,run_dir): self.c,self.run_dir,self.lock,self.runtime,self.procs=c,run_dir,None,None,[]
 def __enter__(self):
  lock=LAB/'.graillon.lock'; lock.parent.mkdir(parents=True,exist_ok=True); self.lock=open(lock,'w')
  try: fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError: raise LabError('outro laboratório Graillon já possui o lock')
  existing=run([self.c['AEGIS_GRAILLON_JACK_LSP_BIN']],env=child_env(self.c))
  if existing.returncode==0:
   raise LabError('já existe um servidor JACK com este nome; ele não pertence a esta execução')
  xd=os.environ.get('XDG_RUNTIME_DIR'); valid=xd and Path(xd).is_dir() and os.access(xd,os.W_OK)
  if valid: self.runtime=None
  else:
   self.runtime=Path(tempfile.mkdtemp(prefix='aegis-graillon-',dir='/tmp')); self.runtime.chmod(0o700)
  return self
 @property
 def env(self): return child_env(self.c,self.runtime)
 def start(self,argv,log):
  f=open(log,'w',encoding='utf8'); p=subprocess.Popen(argv,env=self.env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); self.procs.append((p,f,argv)); return p
 def __exit__(self,*exc):
  for p,f,_ in reversed(self.procs):
   if p.poll() is None:
    p.terminate()
    try:p.wait(timeout=int(self.c['AEGIS_GRAILLON_SHUTDOWN_TIMEOUT_SECONDS']))
    except subprocess.TimeoutExpired: p.kill(); p.wait()
   f.close()
  if self.runtime: shutil.rmtree(self.runtime,ignore_errors=True)
  if self.lock: self.lock.close()
def wait_ports(c,want,timeout):
 end=time.monotonic()+timeout
 while time.monotonic()<end:
  p=run([c['AEGIS_GRAILLON_JACK_LSP_BIN']],env=child_env(c))
  if p.returncode==0 and all(x in p.stdout for x in want): return p.stdout
  time.sleep(.1)
 raise LabError('portas JACK não apareceram: '+', '.join(want))
def start_stack(c,runtime,controls,run_dir):
 # -d dummy avoids physical audio; -n names the isolated server.
 # JACK2's dummy backend has no periods option.  It is deliberately launched
 # non-realtime and with zero physical capture/playback ports.
 jack=runtime.start([c['AEGIS_GRAILLON_JACKD_BIN'],'--no-realtime','-n',c['AEGIS_GRAILLON_JACK_SERVER'],'-d','dummy','-C','0','-P','0','-r',c['AEGIS_GRAILLON_SAMPLE_RATE'],'-p',c['AEGIS_GRAILLON_BLOCK_SIZE']],run_dir/'jackd.log')
 time.sleep(.25)
 if jack.poll() is not None: raise LabError('jackd dummy terminou; veja jackd.log')
 name='aegis-graillon-plugin'; jalv=runtime.start(jalv_args(c,controls,name),run_dir/'jalv.log')
 wait_ports(c,[name+':input_65',name+':output_0'],int(c['AEGIS_GRAILLON_STARTUP_TIMEOUT_SECONDS']))
 return name,jack,jalv
def wav_info(path,c):
 p=run([c['AEGIS_GRAILLON_FFPROBE_BIN'],'-v','error','-show_entries','format=duration:stream=sample_rate,channels','-of','json',str(path)])
 if p.returncode: raise LabError('WAV inválido: '+p.stderr.strip())
 d=json.loads(p.stdout); s=d.get('streams',[{}])[0]
 if not s.get('sample_rate'): raise LabError('entrada não possui stream de áudio')
 return {'duration_seconds':float(d['format'].get('duration',0)),'sample_rate':int(s['sample_rate']),'channels':int(s['channels'])}
def pcm_from_wav(src,c):
 p=subprocess.run([c['AEGIS_GRAILLON_FFMPEG_BIN'],'-v','error','-i',str(src),'-f','s16le','-ar',c['AEGIS_GRAILLON_SAMPLE_RATE'],'-ac','1','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
 if p.returncode: raise LabError(p.stderr.decode('utf8','replace').strip())
 return p.stdout
def write_wav(path,pcm,rate):
 path.parent.mkdir(parents=True,exist_ok=True)
 with wave.open(str(path),'wb') as w: w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);w.writeframes(pcm)
def audio_metrics(path,c):
 p=run([c['AEGIS_GRAILLON_FFMPEG_BIN'],'-hide_banner','-i',str(path),'-af','ebur128=peak=true,astats=metadata=1:reset=0','-f','null','-'])
 def last(rx):
  m=re.findall(rx,p.stderr); return float(m[-1]) if m else None
 return {'integrated_lufs':last(r'\bI:\s*(-?[\d.]+) LUFS'),'true_peak_db':last(r'Peak level dB:\s*(-?[\d.]+)'),'rms_db':last(r'RMS level dB:\s*(-?[\d.]+)')}
def rms_difference(a,b):
 n=min(len(a),len(b))//2
 if not n:return None
 import array
 aa=array.array('h',a[:n*2]);bb=array.array('h',b[:n*2]); return math.sqrt(sum((x-y)**2 for x,y in zip(aa,bb))/n)/32768
def listening_html(records,blind=False):
 order=[x for x in records if x.get('success')]; import random; random.Random('graillon').shuffle(order)
 cards=[]
 for i,p in enumerate(order,1):
  title=f'Sample {i:02d}' if blind else f"{p['id']} — {p['name']}"
  cards.append(f"<article><h2>{html.escape(title)}</h2><audio controls src=\"normalized/{html.escape(p['normalized_file'])}\"></audio><p>{'' if blind else html.escape(p['description'])}</p><p>GLaDOS / força hard-tune / inteligibilidade / identidade / artefatos / fadiga: <input aria-label=notes></p></article>")
 return '<!doctype html><meta charset=utf-8><title>Graillon listening lab</title><style>body{font:15px system-ui;margin:2rem;max-width:900px}article{border-top:1px solid #ccc;padding:1rem}</style><h1>Laboratório Graillon'+(' cego' if blind else '')+'</h1><p>Original <audio controls src="original.wav"></audio></p>'+''.join(cards)
def write_manifest(c,out,records,meta):
 out.mkdir(parents=True,exist_ok=True); shutil.copy2(meta['source'],out/'original.wav')
 for directory in ('raw','normalized','diagnostics'): (out/directory).mkdir(exist_ok=True)
 manifest={**meta,'plugin_uri':c['AEGIS_GRAILLON_PLUGIN_URI'],'bundle':c['AEGIS_GRAILLON_BUNDLE'],'lv2_path':c['AEGIS_GRAILLON_LV2_PATH'],'jack_server':c['AEGIS_GRAILLON_JACK_SERVER'],'sample_rate':int(c['AEGIS_GRAILLON_SAMPLE_RATE']),'block_size':int(c['AEGIS_GRAILLON_BLOCK_SIZE']),'optional_feature_status':{'ptm':'unknown','tune_keyboard':'unknown','doubler':'unknown'},'presets':records}
 (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 fields=['id','name','success','classification','output_sha256','rms_difference_from_control','integrated_lufs','true_peak_db','plugin_latency_frames','xrun_count','error']
 with (out/'manifest.csv').open('w',newline='',encoding='utf8') as f: w=csv.DictWriter(f,fields);w.writeheader();w.writerows([{k:x.get(k) for k in fields} for x in records])
 (out/'listening.html').write_text(listening_html(records),encoding='utf8');(out/'listening-blind.html').write_text(listening_html(records,True),encoding='utf8')
def preflight(c):
 checks={}; checks['not_root']=os.geteuid()!=0; checks['x86_64']=platform.machine()=='x86_64'; bundle=Path(c['AEGIS_GRAILLON_BUNDLE']); checks['bundle']=bundle.is_dir(); checks['binary']=any(bundle.glob('*.so'))
 e=child_env(c)
 for key,var,args in [('lv2ls','AEGIS_GRAILLON_LV2LS_BIN',[c['AEGIS_GRAILLON_LV2LS_BIN']]),('lv2info','AEGIS_GRAILLON_LV2INFO_BIN',[c['AEGIS_GRAILLON_LV2INFO_BIN'],c['AEGIS_GRAILLON_PLUGIN_URI']]),('jalv','AEGIS_GRAILLON_JALV_BIN',[c['AEGIS_GRAILLON_JALV_BIN'],'-V']),('jack','AEGIS_GRAILLON_JACKD_BIN',[c['AEGIS_GRAILLON_JACKD_BIN'],'--version']),('ffmpeg','AEGIS_GRAILLON_FFMPEG_BIN',[c['AEGIS_GRAILLON_FFMPEG_BIN'],'-version']),('ffprobe','AEGIS_GRAILLON_FFPROBE_BIN',[c['AEGIS_GRAILLON_FFPROBE_BIN'],'-version'])]:
  p=run(args,env=e)
  # Jalv 1.6 prints valid help/version output but exits 1 (a known CLI
  # convention); executing and producing its banner is sufficient here.
  checks[key]=(p.returncode==0 or (key=='jalv' and 'jalv ' in (p.stdout+p.stderr).lower()))
 p=run([c['AEGIS_GRAILLON_LV2LS_BIN']],env=e); checks['uri_found']=p.returncode==0 and c['AEGIS_GRAILLON_PLUGIN_URI'] in p.stdout
 checks['isolated_server_available']=run([c['AEGIS_GRAILLON_JACK_LSP_BIN']],env=e).returncode!=0
 checks['input_wav']=Path(c['AEGIS_GRAILLON_INPUT']).is_file()
 def writable_target(value):
  p=Path(value)
  while not p.exists(): p=p.parent
  return os.access(p,os.W_OK)
 checks['writable_dirs']=all(writable_target(c[k]) for k in ('AEGIS_GRAILLON_OUTPUT_DIR','AEGIS_GRAILLON_REPORT_DIR'))
 try: import jack; checks['python_jack_client']=True
 except ImportError: checks['python_jack_client']=False
 return checks
def version(c,bin,args):
 p=run([bin]+args,env=child_env(c)); return (p.stdout or p.stderr).splitlines()[0] if (p.stdout or p.stderr) else None
def inspect(c):
 inv=discover(c); validate_inventory(inv); (LAB/'graillon-ports.json').write_text(json.dumps({'plugin_uri':inv['plugin_uri'],'bundle':inv['bundle'],'ports':inv['ports']},indent=2),encoding='utf8')
 hard=base_controls(c,'official-hard-tune'); payload={'preset_name':'Hard Tune','preset_uri':hard['uri'],'source':hard['source'],'controls':hard['controls']}; state=LAB/'states';state.mkdir(exist_ok=True);(state/'hard-tune-controls.json').write_text(json.dumps(payload,indent=2),encoding='utf8'); return inv,payload
def command_preflight(args):
 c=config(args); checks=preflight(c); payload={'checks':checks,'effective_environment':{k:c[k] for k in sorted(c)},'plugin_uri':c['AEGIS_GRAILLON_PLUGIN_URI'],'bundle':c['AEGIS_GRAILLON_BUNDLE'],'versions':{'jalv':version(c,c['AEGIS_GRAILLON_JALV_BIN'],['-V']),'jack':version(c,c['AEGIS_GRAILLON_JACKD_BIN'],['--version']),'ffmpeg':version(c,c['AEGIS_GRAILLON_FFMPEG_BIN'],['-version'])}}
 print(json.dumps(payload,ensure_ascii=False,indent=2)); return 0 if all(checks.values()) else 2
def command_inspect(args):
 c=config(args);inv,hard=inspect(c);print(json.dumps({'inventory':str(LAB/'graillon-ports.json'),'hard_tune':hard},ensure_ascii=False,indent=2));return 0
def render_one(c,pre,inv,out,source,run_dir):
 # Import at the last responsible moment: inspecting never needs libjack.
 try: from graillon_pipe import process_pcm
 except ImportError as exc: raise LabError('python-jack-client ausente; instale -r experiments/glados-filter-lab/requirements-graillon.txt') from exc
 try: import jack  # noqa: F401
 except ImportError as exc: raise LabError('python-jack-client ausente; instale a dependência isolada requirements-graillon.txt') from exc
 controls=resolve_preset(pre,inv,c); pcm=pcm_from_wav(source,c); rate=int(c['AEGIS_GRAILLON_SAMPLE_RATE']); start=time.monotonic()
 with Runtime(c,run_dir) as rt:
  plugin,_,_=start_stack(c,rt,controls,run_dir)
  processed,diag=process_pcm(pcm,c,rt,plugin,run_dir)
 record={'id':pre['id'],'name':pre['name'],'description':pre['description'],'base_state':pre.get('base_state'),'semantic_controls':pre.get('controls',{}),'lv2_controls':controls,'jalv_command':jalv_args(c,controls,plugin),'connections':diag['connections'],'xrun_count':diag['xruns'],'plugin_latency_frames':None,'plugin_latency_ms':None,'render_seconds':round(time.monotonic()-start,4),'audio_duration_seconds':len(processed)/2/rate,'raw_file':f'preset-{pre["id"]}-raw.wav','compensated_file':f'preset-{pre["id"]}-compensated.wav','normalized_file':f'preset-{pre["id"]}-normalized.wav','success':True}
 raw=out/'raw'/record['raw_file'];write_wav(raw,processed,rate); write_wav(out/'normalized'/record['normalized_file'],processed,rate); shutil.copy2(raw,out/'raw'/record['compensated_file']);record.update(audio_metrics(raw,c));record['output_sha256']=sha_file(raw)
 return record,processed
def command_render(args,all_presets=False):
 c=config(args); data=read_yaml(c['AEGIS_GRAILLON_PRESETS_CONFIG'])
 if data.get('round')=='round2':
  from graillon_round2 import run as run_round2
  return run_round2(args)
 if data.get('round')=='round3':
  from graillon_round3 import run as run_round3
  return run_round3(args)
 if data.get('round')=='round4':
  from graillon_round3 import run as run_round4
  return run_round4(args)
 inv,_=inspect(c);
 if data.get('engine')!='graillon_lv2': raise LabError('configuração não é graillon_lv2')
 source=Path(c['AEGIS_GRAILLON_INPUT']); wav_info(source,c); presets=[p for p in data['presets'] if p.get('enabled',True) and (all_presets or p['id']==args.preset)]
 if not presets: raise LabError('preset não encontrado')
 out=Path(c['AEGIS_GRAILLON_OUTPUT_DIR']); timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); run_dir=Path(c['AEGIS_GRAILLON_REPORT_DIR'])/'runs'/timestamp;run_dir.mkdir(parents=True)
 (run_dir/'preflight.json').write_text(json.dumps(preflight(c),indent=2),encoding='utf8')
 (run_dir/'environment.json').write_text(json.dumps({k:c[k] for k in sorted(c)},indent=2),encoding='utf8')
 records=[];control_pcm=None
 for pre in presets:
  try:
   preset_run_dir=run_dir/pre['id'];preset_run_dir.mkdir(parents=True,exist_ok=True)
   record,pcm=render_one(c,pre,inv,out,source,preset_run_dir)
   if pre['id']=='G00':control_pcm=pcm
   if control_pcm is not None:
    diff=rms_difference(pcm,control_pcm);record['rms_difference_from_control']=diff;record['classification']='identical_to_control' if record['output_sha256']==sha_bytes(control_pcm) else 'near_identical_to_control' if diff is not None and diff<.001 else 'materially_different'
   records.append(record)
  except Exception as exc: records.append({'id':pre['id'],'name':pre['name'],'description':pre['description'],'success':False,'error':str(exc)})
 write_manifest(c,out,records,{'source':str(source),'executed_at':datetime.now(timezone.utc).isoformat(),'jalv_version':version(c,c['AEGIS_GRAILLON_JALV_BIN'],['-V']),'jack_version':version(c,c['AEGIS_GRAILLON_JACKD_BIN'],['--version'])})
 (run_dir/'controls-effective.json').write_text(json.dumps({x['id']:x.get('lv2_controls',{}) for x in records},indent=2),encoding='utf8')
 (run_dir/'metrics.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf8')
 (run_dir/'connections.txt').write_text('\n'.join(' -> '.join(p) for x in records for p in x.get('connections',[])),encoding='utf8')
 # One render process is intentionally created per preset; retain its logs in
 # child directories and provide stable top-level entry points for automation.
 for log in ('jackd.log','jalv.log'):
  pieces=[]
  for child in sorted(run_dir.glob('*/'+log)):
   pieces.append('## '+str(child.relative_to(run_dir))+'\n'+child.read_text(encoding='utf8',errors='replace'))
  (run_dir/log).write_text('\n'.join(pieces) if pieces else 'No plugin process started.\n',encoding='utf8')
 (run_dir/'summary.md').write_text('\n'.join(f"- {x['id']}: {'ok' if x.get('success') else x.get('error')}" for x in records),encoding='utf8');print(json.dumps({'rendered':sum(x.get('success',False) for x in records),'failed':sum(not x.get('success',False) for x in records),'output':str(out),'report':str(run_dir)},ensure_ascii=False));return 0 if any(x.get('success') for x in records) else 2
def main():
 # Allow global overrides before *or after* the command, matching the examples
 # in README and avoiding a surprising argparse ordering constraint.
 raw=sys.argv[1:]; global_tokens=[]; rest=[]; i=0
 while i<len(raw):
  if raw[i] in ('--env-file','--input','--output-dir','--preset-config','--config','--base-engine'):
   if i+1>=len(raw): print('graillon_lab: valor ausente para '+raw[i],file=sys.stderr);return 2
   global_tokens.extend(raw[i:i+2]);i+=2
  else: rest.append(raw[i]);i+=1
 ap=argparse.ArgumentParser();ap.add_argument('--env-file');ap.add_argument('--input');ap.add_argument('--output-dir');ap.add_argument('--preset-config');ap.add_argument('--config');ap.add_argument('--base-engine'); sub=ap.add_subparsers(dest='cmd',required=True);sub.add_parser('preflight');sub.add_parser('inspect');r=sub.add_parser('render');r.add_argument('--preset',required=True);sub.add_parser('render-all');s=sub.add_parser('stream-test');s.add_argument('--preset',default='G03'); args=ap.parse_args(global_tokens+rest)
 def interrupted(signum, frame): raise KeyboardInterrupt
 previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
 try:
  if args.cmd=='preflight': return command_preflight(args)
  if args.cmd=='inspect': return command_inspect(args)
  if args.cmd=='render': return command_render(args)
  if args.cmd=='render-all': return command_render(args,True)
  if args.cmd=='stream-test': return command_render(args)
 except KeyboardInterrupt:
  print('graillon_lab: cancelado; cleanup dos processos próprios concluído',file=sys.stderr);return 130
 except (LabError,OSError,json.JSONDecodeError) as exc: print('graillon_lab: '+str(exc),file=sys.stderr);return 2
 finally:
  for sig,handler in previous.items(): signal.signal(sig,handler)
if __name__=='__main__': sys.exit(main())
