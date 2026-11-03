#!/usr/bin/env python3
"""Local CPU-only fixed-ASR evidence. No network, TTS, training or auditory claims."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from training.f5_asr_metrics import plain_word_comparison,plain_word_tokens,METRIC_POLICY
HERE=Path(__file__).resolve().parent
ASR=Path('/var/tmp/aegis-v2-evaluation-20261102/asr')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.chmod(0o600);tmp.replace(path)

def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        now=[i]
        for j,y in enumerate(b,1):now.append(min(now[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=now
    return prev[-1]

def score(expected,hypothesis):
    row=plain_word_comparison(expected,hypothesis)
    ref=' '.join(plain_word_tokens(expected));hyp=' '.join(plain_word_tokens(hypothesis));edits=distance(ref,hyp)
    row.update(character_edits=edits,reference_characters=len(ref),character_error_rate=edits/max(1,len(ref)),
        cer_policy='Same historical token normalization; CER over tokens joined with single spaces, including those spaces')
    return row

def jobs_for_qwen(directory):
    spoken=json.loads((HERE/'spoken-validation.pt-BR.json').read_text())['cases']
    corpus=json.loads((HERE/'validation.pt-BR.json').read_text())
    jobs=[]
    for case in corpus:
        wav=Path(directory)/(case['id']+'-r00.wav')
        if not wav.exists():continue
        jobs.append(dict(candidate='qwen-production',case=case['id'],wav=str(wav),expected_verbatim=case['text'],
            expected_spoken=spoken[case['id']],cohort='validation-rep0'))
    return jobs

def summary(receipt):
    grouped={}
    for row in receipt['attempts']:
        key=row['candidate']+'/'+row.get('cohort','unspecified')
        dest=grouped.setdefault(key,dict(observations=0,successes=0,failures=0,comparisons={}))
        dest['observations']+=1
        if not row.get('success'):dest['failures']+=1;continue
        dest['successes']+=1
        for kind,sc in row['comparisons'].items():
            s=dest['comparisons'].setdefault(kind,dict(word_edits=0,reference_words=0,character_edits=0,reference_characters=0,cases=[]))
            s['word_edits']+=sc['word_edits'];s['reference_words']+=len(sc['reference_tokens']);s['character_edits']+=sc['character_edits'];s['reference_characters']+=sc['reference_characters'];s['cases'].append(row['case'])
    for g in grouped.values():
        for s in g['comparisons'].values():s['wer']=s['word_edits']/max(1,s['reference_words']);s['cer']=s['character_edits']/max(1,s['reference_characters'])
    return dict(groups=grouped,note='Compare only identical case/cohort sets and expected-text policy. Spelling/number/acronym/name differences are auxiliary ASR mismatches, not auditory pronunciation or identity judgments.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--qwen-dir');p.add_argument('--jobs');p.add_argument('--output',required=True)
    p.add_argument('--asr-root',default=str(ASR));p.add_argument('--deadline-seconds',type=float,default=1800);args=p.parse_args()
    os.umask(0o077);asr=Path(args.asr_root);receipt=json.loads((asr/'download-receipt.json').read_text())
    if receipt['repository']!='Systran/faster-whisper-small' or receipt['revision']!='536b0662742c02347bc0e980a01041f333bce120':raise ValueError('ASR revision mismatch')
    for item in receipt['files']:
        if sha(asr/'model'/item['file'])!=item['sha256']:raise ValueError('ASR model integrity failure: '+item['file'])
    jobs=jobs_for_qwen(args.qwen_dir) if args.qwen_dir else []
    if args.jobs:
        data=json.loads(Path(args.jobs).read_text());jobs+=data['jobs'] if isinstance(data,dict) else data
    if not jobs:raise ValueError('no audio jobs')
    out=Path(args.output).resolve();out.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    if any((x/'.git').exists() for x in (out.parent,*out.parents)):raise ValueError('private output must be outside Git')
    result=json.loads(out.read_text()) if out.exists() else dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),attempts=[],
        settings=dict(repository=receipt['repository'],revision=receipt['revision'],device='cpu',compute_type='int8',cpu_threads=2,num_workers=1,
            language='pt',beam_size=5,temperature=0,vad_filter=False,condition_on_previous_text=False,initial_prompt=None,
            resampling='scipy resample_poly Kaiser beta10 to16k float32',trim=False,gain=False,niceness=os.nice(0)),
        versions={k:importlib.metadata.version(k) for k in ('faster-whisper','ctranslate2','numpy','scipy','soundfile','av')},
        script_sha256=sha(__file__),model_receipt_sha256=sha(asr/'download-receipt.json'),model_files=receipt['files'],metric_policy=METRIC_POLICY,
        interpretation='Local ASR evidence only. No listening was performed. Robotic timbre can cause recognizer errors; review divergent cases against audio before rejection.')
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from faster_whisper import WhisperModel
    start=time.perf_counter();model=WhisperModel(str(asr/'model'),device='cpu',compute_type='int8',cpu_threads=2,num_workers=1,local_files_only=True)
    result['load_seconds']=time.perf_counter()-start;save(out,result)
    for job in jobs:
        path=Path(job['wav']);audio_sha=sha(path)
        expected={k.removeprefix('expected_'):v for k,v in job.items() if k.startswith('expected_')}
        key=dict(candidate=job['candidate'],case=job['case'],audio_sha256=audio_sha,expected=expected)
        if any(r.get('key')==key for r in result['attempts']):continue
        if time.perf_counter()-start>args.deadline_seconds:result['deadline_reached']=True;save(out,result);break
        row=dict(**job,key=key,audio_sha256=audio_sha);tick=time.perf_counter()
        try:
            data,rate=sf.read(path,dtype='float32',always_2d=True)
            if data.shape[1]!=1 or not data.size or not np.isfinite(data).all():raise ValueError('finite nonempty mono audio required')
            divisor=math.gcd(rate,16000);waveform=resample_poly(data[:,0],16000//divisor,rate//divisor,window=('kaiser',10.)).astype(np.float32)
            row.update(source_rate=rate,source_seconds=len(data)/rate,source_peak=float(abs(data).max()),asr_input_sha256=hashlib.sha256(waveform.tobytes()).hexdigest())
            segments,info=model.transcribe(waveform,language='pt',beam_size=5,temperature=0,vad_filter=False,condition_on_previous_text=False,initial_prompt=None)
            segments=list(segments);hypothesis=' '.join(s.text.strip() for s in segments)
            row.update(success=True,hypothesis=hypothesis,segments=[dict(start=s.start,end=s.end,text=s.text,avg_logprob=s.avg_logprob,no_speech_prob=s.no_speech_prob) for s in segments],
                comparisons={kind:score(text,hypothesis) for kind,text in expected.items()})
        except Exception as e:row.update(success=False,error=repr(e))
        row['asr_wall_seconds']=time.perf_counter()-tick;result['attempts'].append(row);result['last_update_utc']=dt.datetime.now(dt.timezone.utc).isoformat();save(out,result)
        save(out.with_name(out.stem+'-summary.json'),summary(result))
        print(json.dumps({k:row.get(k) for k in ('candidate','case','success','hypothesis','asr_wall_seconds','error')},ensure_ascii=False),flush=True)
    result['last_run_seconds']=time.perf_counter()-start;result['completed_utc']=dt.datetime.now(dt.timezone.utc).isoformat();save(out,result)
if __name__=='__main__':main()
