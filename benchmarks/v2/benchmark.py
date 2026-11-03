#!/usr/bin/env python3
"""Measure final PCM at the client; never starts/stops a model or service."""
from __future__ import annotations
import argparse
import array
import datetime as dt
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import wave

HERE = Path(__file__).resolve().parent
RATE = 24000
RULE = {"frame_ms":20,"rms_dbfs":-45.0,"consecutive_frames":2,"margin_ms":20,
        "gap_threshold_ms":10,"playback":"immediate FIFO, no device delay"}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.chmod(tmp,0o600);tmp.replace(path)

def pcm_metrics(pcm,chunks,rate=RATE):
    """Chunks are client arrival ms/bytes. Headerless PCM16LE, even after splits."""
    if len(pcm)%2:raise ValueError('truncated PCM16 sample')
    samples=array.array('h');samples.frombytes(pcm)
    if sys.byteorder!='little':samples.byteswap()
    frame=round(rate*RULE['frame_ms']/1000);threshold=32768*10**(RULE['rms_dbfs']/20)
    active=0;onset=None;confirmed=None
    for start in range(0,len(samples)-frame+1,frame):
        values=samples[start:start+frame]
        rms=math.sqrt(sum(v*v for v in values)/frame)
        active=active+1 if rms>=threshold else 0
        if active>=RULE['consecutive_frames']:
            onset=start-(RULE['consecutive_frames']-1)*frame;confirmed=start+frame;break
    margin=max(0,onset-round(rate*RULE['margin_ms']/1000)) if onset is not None else None
    received=0;first_pcm=None;available=None;playback_end=None;onset_playback=None;margin_playback=None
    gaps=[];previous=-1
    for item in chunks:
        arrival=float(item['ms']);size=int(item['bytes'])
        if arrival<previous or size<0:raise ValueError('invalid arrival schedule')
        previous=arrival;before=received//2;received+=size;after=received//2
        if before==after:continue
        if first_pcm is None:first_pcm=arrival
        if playback_end is not None and arrival-playback_end>RULE['gap_threshold_ms']:
            gaps.append({'at_ms':arrival,'duration_ms':arrival-playback_end})
        play_start=max(arrival,playback_end) if playback_end is not None else arrival
        if onset is not None and before<=onset<after:
            onset_playback=play_start+(onset-before)*1000/rate
        if margin is not None and before<=margin<after:
            margin_playback=play_start+(margin-before)*1000/rate
        if confirmed is not None and available is None and after>=confirmed:available=arrival
        playback_end=play_start+(after-before)*1000/rate
    if received!=len(pcm):raise ValueError('arrival schedule does not match audio')
    return dict(first_pcm_ms=first_pcm,first_speech_pcm_available_ms=available,
        ttfa_estimated_playback_ms=onset_playback,ttfa_margin_playback_ms=margin_playback,
        onset_audio_offset_ms=onset*1000/rate if onset is not None else None,
        onset_margin_audio_offset_ms=margin*1000/rate if margin is not None else None,
        audio_seconds=len(samples)/rate,peak_pcm16=max((abs(x) for x in samples),default=0),
        clipped_samples=sum(x in (-32768,32767) for x in samples),underruns_over_10ms=len(gaps),
        underrun_total_ms=sum(x['duration_ms'] for x in gaps),underrun_max_ms=max((x['duration_ms'] for x in gaps),default=0),
        underrun_events=gaps,onset_rule=RULE)

def gpu_snapshot():
    def run(query):
        p=subprocess.run(['nvidia-smi',query,'--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=5)
        return [line.strip() for line in p.stdout.splitlines() if line.strip()]
    rows=run('--query-gpu=uuid,name,memory.total,memory.used')
    processes=run('--query-compute-apps=pid,used_gpu_memory')
    return dict(monotonic=time.monotonic(),devices=[r.split(', ') for r in rows],processes=[r.split(', ') for r in processes])

class GpuSampler:
    def __init__(self,path,enabled):
        self.path=path;self.enabled=enabled;self.stop=threading.Event();self.thread=None;self.peak=None;self.error=None
    def __enter__(self):
        if self.enabled:
            self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
        return self
    def loop(self):
        try:
            with self.path.open('x') as f:
                os.chmod(self.path,0o600)
                while not self.stop.is_set():
                    row=gpu_snapshot()
                    self.peak=max(self.peak or 0,max(float(r[3]) for r in row['devices']))
                    f.write(json.dumps(row)+'\n');f.flush();self.stop.wait(.5)
        except Exception as e:self.error=repr(e)
    def __exit__(self,*args):
        self.stop.set()
        if self.thread:self.thread.join(7)

def endpoint(value):
    p=urlsplit(value)
    if p.scheme!='http' or p.hostname not in ('127.0.0.1','::1') or not p.port or p.query or p.fragment or p.username:
        raise ValueError('endpoint must be literal HTTP loopback with port and path')
    return p

def receive(url,payload,timeout):
    p=endpoint(url);connection=http.client.HTTPConnection(p.hostname,p.port,timeout=timeout)
    body=json.dumps(payload,ensure_ascii=False).encode();raw=bytearray();chunks=[];row={}
    started=time.perf_counter()
    try:
        connection.request('POST',p.path or '/',body,{'Content-Type':'application/json','Accept-Encoding':'identity'})
        response=connection.getresponse();row['headers_ms']=(time.perf_counter()-started)*1000
        row['http_status']=response.status
        row['headers']={k.lower():v for k,v in response.getheaders() if k.lower() in ('content-type','content-encoding','transfer-encoding','content-length') or k.lower().startswith('x-aegis-')}
        if response.status!=200:raise ValueError('HTTP '+str(response.status))
        ctype=response.getheader('content-type','').split(';')[0].strip().lower()
        if ctype not in ('audio/pcm','audio/raw','application/octet-stream'):raise ValueError('not raw PCM16LE: '+ctype)
        if response.getheader('content-encoding','identity')!='identity':raise ValueError('compressed transport')
        rate=response.getheader('x-aegis-sample-rate')
        if rate and int(rate)!=RATE:raise ValueError('final rate must be 24000')
        channels=response.getheader('x-aegis-channels')
        if channels and int(channels)!=1:raise ValueError('final channels must be mono')
        while True:
            part=response.read1(65536)
            now=(time.perf_counter()-started)*1000
            if not part:break
            chunks.append({'ms':now,'bytes':len(part)});raw.extend(part)
            if now>timeout*1000:raise TimeoutError('overall deadline')
            if len(raw)>RATE*2*180:raise ValueError('audio exceeds 180s')
        row['eof_ms']=(time.perf_counter()-started)*1000
        if raw[:4] in (b'RIFF',b'OggS',b'fLaC'):raise ValueError('container header in raw PCM')
        if not raw:raise ValueError('empty PCM')
        row.update(pcm_metrics(raw,chunks));row['rtf']=row['eof_ms']/1000/row['audio_seconds']
        row['outcome']='ok' if row['first_speech_pcm_available_ms'] is not None else 'no_speech'
    except Exception as e:
        row.update(outcome='failed',error=repr(e),failed_after_ms=(time.perf_counter()-started)*1000)
    finally:connection.close()
    row['chunks']=chunks;row['pcm_sha256']=hashlib.sha256(raw).hexdigest();row['pcm_bytes']=len(raw)
    return bytes(raw),row

def percentile(values,p):
    return sorted(values)[math.ceil(len(values)*p)-1] if values else None

def summarize(rows):
    result={}
    for group in sorted({(r.get('candidate'),r.get('condition'),r.get('case')) for r in rows}):
        selected=[r for r in rows if (r.get('candidate'),r.get('condition'),r.get('case'))==group]
        good=[r for r in selected if r.get('outcome')=='ok'];metrics={}
        for name in ('first_pcm_ms','first_speech_pcm_available_ms','ttfa_estimated_playback_ms','eof_ms','rtf','audio_seconds','underrun_total_ms'):
            values=[r[name] for r in good if r.get(name) is not None]
            metrics[name]=dict(n=len(values),p50=statistics.median(values) if values else None,
                p95=percentile(values,.95) if len(values)>=20 else None,
                min=min(values) if values else None,max=max(values) if values else None,
                stdev=statistics.stdev(values) if len(values)>1 else None)
        result['/'.join(str(x) for x in group)]=dict(observations=len(selected),successes=len(good),failures=len(selected)-len(good),metrics=metrics)
    return dict(groups=result,p95_policy='nearest-rank only with >=20 observations per case/condition; otherwise observed max',
        warning='digital onset/FIFO proxy excludes audio device delay; listening/ASR and identity assessed separately')

def private_output(value):
    path=Path(value).expanduser().resolve()
    if any((p/'.git').exists() for p in (path,*path.parents)):raise ValueError('output must be outside Git checkouts')
    path.mkdir(parents=True,exist_ok=False,mode=0o700)
    return path

def run(args):
    os.umask(0o077);endpoint(args.endpoint)
    corpus=HERE/('test.pt-BR.json' if args.final_test else 'validation.pt-BR.json')
    cases=json.loads(corpus.read_text())
    if args.final_test:
        if not args.decision:raise ValueError('final test requires pre-recorded --decision file')
        decision=json.loads(Path(args.decision).read_text())
        required={'material_ttfa_gain_ms','max_wer_regression_pp','identity_rejection_rule','selected_candidate','validation_evidence'}
        if not required<=decision.keys():raise ValueError('decision file lacks: '+str(sorted(required-decision.keys())))
        if decision['selected_candidate']!=args.candidate:raise ValueError('decision candidate mismatch')
    if args.case:cases=[c for c in cases if c['id'] in args.case]
    if not cases:raise ValueError('no selected cases')
    out=private_output(args.output);snapshot=gpu_snapshot() if args.sample_gpu or args.expected_pids else None
    if args.expected_pids:
        actual={int(x[0]) for x in snapshot['processes']}
        if actual!=set(args.expected_pids):raise ValueError('GPU PID set mismatch: '+str(actual))
    if args.condition=='cold-process' and not args.cold_evidence:raise ValueError('cold-process requires --cold-evidence startup record')
    command=subprocess.run(['git','rev-parse','HEAD'],cwd=HERE,capture_output=True,text=True)
    manifest=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),command=sys.argv,source_commit=command.stdout.strip(),
        harness_sha256=digest(__file__),corpus=str(corpus),corpus_sha256=digest(corpus),candidate=args.candidate,endpoint=args.endpoint,
        sample_rate=RATE,format='s16le mono',concurrency=1,repetitions=args.repetitions,condition=args.condition,
        python=sys.version,platform=platform.platform(),onset_rule=RULE,expected_pids=args.expected_pids,gpu_before=snapshot,
        payload=json.loads(args.payload),text_field=args.text_field,seed_field=args.seed_field,seed=args.seed,
        audio_retention='first repetition per case plus every failed PCM response',cold_evidence=args.cold_evidence,
        cold_evidence_sha256=digest(args.cold_evidence) if args.cold_evidence else None,
        decision_sha256=digest(args.decision) if args.decision else None,scope='client HTTP final output; connection establishment included; no warm-up sent/discarded')
    atomic_json(out/'manifest.json',manifest);rows=[]
    order=[(rep,case) for rep in range(args.repetitions) for case in cases]
    if args.shuffle:random.Random(args.seed).shuffle(order)
    with GpuSampler(out/'gpu.jsonl',args.sample_gpu) as sampler, (out/'results.jsonl').open('x') as f:
        for rep,case in order:
            payload=json.loads(args.payload);payload[args.text_field]=case['text']
            if args.seed_field:payload[args.seed_field]=args.seed+rep
            pcm,row=receive(args.endpoint,payload,args.timeout)
            row.update(candidate=args.candidate,condition=args.condition,case=case['id'],size=case['size'],repetition=rep,
                requested_seed=args.seed+rep if args.seed_field else None,text_sha256=hashlib.sha256(case['text'].encode()).hexdigest(),
                sampled_vram_peak_used_mib=sampler.peak)
            if pcm and not len(pcm)%2 and (rep==0 or row['outcome']!='ok'):
                path=out/(case['id']+f'-r{rep:02d}.wav')
                with wave.open(str(path),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(RATE);w.writeframes(pcm)
                row.update(wav=path.name,wav_sha256=digest(path))
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno());rows.append(row)
            atomic_json(out/'summary.json',summarize(rows))
            print(json.dumps({k:row.get(k) for k in ('case','repetition','outcome','first_pcm_ms','first_speech_pcm_available_ms','ttfa_estimated_playback_ms','eof_ms','rtf','error')}),flush=True)
            if args.stop_on_error and row['outcome']!='ok':break
    atomic_json(out/'completed.json',dict(completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),observations=len(rows),
        sampled_vram_peak_used_mib=sampler.peak,sampler_error=sampler.error,
        limitation='nvidia-smi 500ms samples may miss short peaks; no allocator peak or pre-run loading peak'))

def record_file(args):
    """For file-only engines: elapsed must include generation, DSP, write and close."""
    with wave.open(args.wav,'rb') as w:
        if (w.getframerate(),w.getnchannels(),w.getsampwidth())!=(RATE,1,2):raise ValueError('need final mono PCM16 WAV 24000Hz')
        pcm=w.readframes(w.getnframes())
    row=pcm_metrics(pcm,[dict(ms=args.elapsed*1000,bytes=len(pcm))])
    row.update(candidate=args.candidate,condition=args.condition,case=args.case[0],repetition=args.repetition,
        eof_ms=args.elapsed*1000,rtf=args.elapsed/row['audio_seconds'],outcome='ok' if row['first_speech_pcm_available_ms'] is not None else 'no_speech',
        wav=args.wav,wav_sha256=digest(args.wav),streaming='whole file available only after synthesis/DSP/write/close',
        scope='engine/file; no HTTP; elapsed supplied by instrumented runner',onset_rule=RULE)
    print(json.dumps(row,ensure_ascii=False,allow_nan=False))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    r=sub.add_parser('run');r.add_argument('--endpoint',default='http://127.0.0.1:8001/v1/audio/speech');r.add_argument('--candidate',required=True)
    r.add_argument('--output',required=True);r.add_argument('--repetitions',type=int,default=3);r.add_argument('--case',action='append')
    r.add_argument('--payload',default='{"response_format":"pcm","stream":true}');r.add_argument('--text-field',default='input')
    r.add_argument('--condition',choices=['warm','first-request','cold-process'],default='warm');r.add_argument('--cold-evidence')
    r.add_argument('--sample-gpu',action='store_true');r.add_argument('--expected-pids',type=int,nargs='+');r.add_argument('--timeout',type=float,default=180)
    r.add_argument('--seed',type=int,default=40733133);r.add_argument('--seed-field');r.add_argument('--shuffle',action='store_true')
    r.add_argument('--final-test',action='store_true');r.add_argument('--decision');r.add_argument('--stop-on-error',action='store_true');r.set_defaults(func=run)
    s=sub.add_parser('summarize');s.add_argument('results');s.set_defaults(func=lambda a:print(json.dumps(summarize([json.loads(x) for x in Path(a.results).read_text().splitlines() if x]),indent=2)))
    f=sub.add_parser('file');f.add_argument('--wav',required=True);f.add_argument('--elapsed',required=True,type=float);f.add_argument('--candidate',required=True)
    f.add_argument('--case',required=True,action='append');f.add_argument('--repetition',type=int,default=0);f.add_argument('--condition',default='warm');f.set_defaults(func=record_file)
    a=p.parse_args()
    if hasattr(a,'repetitions') and a.repetitions<1:p.error('repetitions must be positive')
    a.func(a)
if __name__=='__main__':main()
