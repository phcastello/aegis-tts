#!/usr/bin/env python3
"""PCM stdin/stdout adapter for one private headless Graillon JACK session."""
from __future__ import annotations
import argparse, os, signal, sys, threading, time
from collections import deque
from pathlib import Path

# graillon_lab imports this file as a module; importing it itself does not start
# JACK and therefore remains safe in unit tests and production environments.
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))

class PipeError(RuntimeError): pass
def log(message): print('graillon_pipe: '+message,file=sys.stderr,flush=True)
def aligned_chunks(reader,chunk_bytes=8192):
 """Yield s16le-aligned fragments, retaining an odd final byte as an error."""
 carry=b''
 while True:
  data=reader.read(chunk_bytes)
  if not data: break
  data=carry+data; take=len(data)&~1
  if take: yield data[:take]
  carry=data[take:]
 if carry: raise PipeError('stdin terminou com amostra s16le incompleta')

class JackBridge:
 """Bounded Python JACK client.  JACK callbacks never perform I/O or logging."""
 def __init__(self,c,client_name):
  try:
   import jack, numpy as np
  except ImportError as exc: raise PipeError('python-jack-client não instalado') from exc
  self.jack,self.np=jack,np; self.rate=int(c['AEGIS_GRAILLON_SAMPLE_RATE']);self.block=int(c['AEGIS_GRAILLON_BLOCK_SIZE']);self.limit=self.block*64
  self.in_q=deque();self.out_q=deque();self.in_count=self.out_count=0;self.lock=threading.Condition();self.xruns=0
  self.client=jack.Client(client_name,no_start_server=True,servername=c['AEGIS_GRAILLON_JACK_SERVER']);self.send=self.client.outports.register('to_graillon');self.receive=self.client.inports.register('from_graillon')
  @self.client.set_process_callback
  def process(frames):
   with self.lock:
    need=frames; pieces=[]
    while need and self.in_q:
     a=self.in_q[0]; n=min(need,len(a));pieces.append(a[:n]);need-=n
     if n==len(a):self.in_q.popleft()
     else:self.in_q[0]=a[n:]
     self.in_count-=n
    out=self.send.get_array();out.fill(0)
    if pieces: out[:frames-need]=self.np.concatenate(pieces)
    incoming=self.receive.get_array().copy()
    if self.out_count+len(incoming)<=self.limit:
     self.out_q.append(incoming);self.out_count+=len(incoming);self.lock.notify_all()
    else: self.xruns+=1
  @self.client.set_xrun_callback
  def xrun(delay): self.xruns+=1
 def activate(self): self.client.activate()
 def close(self):
  try:self.client.deactivate()
  finally:self.client.close()
 def feed(self,pcm):
  a=self.np.frombuffer(pcm,dtype='<i2').astype(self.np.float32)/32768.0; pos=0
  with self.lock:
   while pos<len(a):
    while self.in_count>=self.limit: self.lock.wait(.05)
    n=min(self.limit-self.in_count,len(a)-pos);self.in_q.append(a[pos:pos+n].copy());self.in_count+=n;pos+=n;self.lock.notify_all()
 def pop(self):
  with self.lock:
   if not self.out_q:return b''
   a=self.out_q.popleft();self.out_count-=len(a);self.lock.notify_all()
  return self.np.clip(a*32768,-32768,32767).astype('<i2').tobytes()
 def idle(self):
  with self.lock:return self.in_count==0

def process_pcm(pcm,c,runtime,plugin_name,run_dir):
 """Process finite PCM and return captured s16le.  Queues are bounded."""
 from graillon_lab import wait_ports
 bridge=JackBridge(c,'aegis-graillon-bridge'); bridge.activate()
 try:
  wait_ports(c,['aegis-graillon-bridge:to_graillon','aegis-graillon-bridge:from_graillon'],int(c['AEGIS_GRAILLON_STARTUP_TIMEOUT_SECONDS']))
  con=[('aegis-graillon-bridge:to_graillon',plugin_name+':input_65'),(plugin_name+':output_0','aegis-graillon-bridge:from_graillon')]
  for src,dst in con:
   import subprocess
   p=subprocess.run([c['AEGIS_GRAILLON_JACK_CONNECT_BIN'],src,dst],env=runtime.env,text=True,capture_output=True)
   if p.returncode: raise PipeError('jack_connect falhou: '+p.stderr.strip())
  padding=b'\0'*(int(c['AEGIS_GRAILLON_PREROLL_MS'])*bridge.rate//1000*2)
  payload=padding+pcm+b'\0'*(int(c['AEGIS_GRAILLON_POSTROLL_MS'])*bridge.rate//1000*2)
  # Feeding can block on the bounded input queue.  It must not prevent this
  # thread from draining the bounded capture queue, otherwise a long file
  # looks like an xrun and loses most of its tail.
  done=threading.Event(); feeder=threading.Thread(target=lambda:(bridge.feed(payload),done.set()),daemon=True); feeder.start()
  out=bytearray(); deadline=time.monotonic()+max(5,len(payload)/2/bridge.rate+int(c['AEGIS_GRAILLON_RENDER_PADDING_SECONDS']))
  drained_at=None
  while time.monotonic()<deadline:
   chunk=bridge.pop()
   if chunk: out.extend(chunk);drained_at=None
   if done.is_set() and bridge.idle():
    if drained_at is None: drained_at=time.monotonic()
    elif time.monotonic()-drained_at>=.25: break
   time.sleep(.002)
  feeder.join(timeout=1)
  while (chunk:=bridge.pop()): out.extend(chunk)
  return bytes(out),{'connections':con,'xruns':bridge.xruns}
 finally: bridge.close()

def pipe_main():
 from graillon_lab import LabError, Runtime, config, discover, read_yaml, resolve_preset, start_stack
 ap=argparse.ArgumentParser();ap.add_argument('--preset',required=True);ap.add_argument('--env-file'); ns=ap.parse_args()
 stopped=False
 def stop(*_):
  nonlocal stopped;stopped=True
 for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,stop)
 try:
  c=config(ns); inv=discover(c); pre=next(p for p in read_yaml(c['AEGIS_GRAILLON_PRESETS_CONFIG'])['presets'] if p['id']==ns.preset); controls=resolve_preset(pre,inv,c)
  try: import jack  # noqa: F401
  except ImportError as exc: raise PipeError('python-jack-client ausente; instale requirements-graillon.txt') from exc
  run_dir=Path(c['AEGIS_GRAILLON_REPORT_DIR'])/'runs'/time.strftime('%Y%m%dT%H%M%SZ')/'pipe';run_dir.mkdir(parents=True,exist_ok=True)
  with Runtime(c,run_dir) as rt:
   plugin,_,_=start_stack(c,rt,controls,run_dir)
   # Stream fragments remain aligned; each independent callback session uses a
   # single plugin instance and no bytes are ever logged to stderr.
   whole=b''.join(aligned_chunks(sys.stdin.buffer))
   if not stopped:
    out,_=process_pcm(whole,c,rt,plugin,run_dir)
    try: sys.stdout.buffer.write(out);sys.stdout.buffer.flush()
    except BrokenPipeError: return 141
  return 0
 except (LabError,PipeError,OSError) as exc: log(str(exc));return 2
if __name__=='__main__': sys.exit(pipe_main())
