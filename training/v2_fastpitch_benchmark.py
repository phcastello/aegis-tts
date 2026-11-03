"""Common-validation FastPitch + pinned BigVGAN complete-file benchmark.

No service control or training. Explicit checkpoint hash, exact model/vocoder
representations, three warmups per case, no final-test access, no HTTP claims.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import wave

REPO=Path(__file__).resolve().parents[1]
BIG_SOURCE_REV='7d2b454564a6c7d014227f635b7423881f14bdac'
BIG_MODEL_REV='c329ede9e9bbc100ddf5c91e2330a61921262370'
BIG_WEIGHT_SHA='6f9c5715550c9d0f11159ceb8935638da5aeb19e27d1e63677632df095e376f5'
BIG_CONFIG_SHA='d77e2c96583ca2296ac112a56ec7cc6bd5da4bf7681ceff18448bedc4fcf6512'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def json_atomic(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def metrics_module():
    spec=importlib.util.spec_from_file_location('aegis_fp_common_metrics',REPO/'benchmarks/v2/benchmark.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def load_bound(path,expected,torch,weights_only=False):
    """Hash and load one open object; reject a concurrent publication."""
    path=Path(path)
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno());h=hashlib.sha256()
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=expected:raise ValueError('artifact SHA256 mismatch: '+str(path))
        stream.seek(0);state=torch.load(stream,map_location='cpu',weights_only=weights_only)
        after=path.stat()
        if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
            raise RuntimeError('artifact replaced while loading; select a frozen checkpoint')
    return state


def validate_pair(config,h,vocab,mel_config):
    features=config['features']
    if features['vocab']!=vocab or features['mel']!=mel_config:raise ValueError('checkpoint frontend/mel differs from the pinned FastPitch recipe')
    if (config['model']['n_symbols'],config['model']['n_mel_channels'])!=(len(vocab),100):raise ValueError('checkpoint architecture/frontend mismatch')
    if config.get('alignment_precision')!='FP32 cumulative duration boundaries and pitch sums; acoustic outputs retain selected dtype':
        raise ValueError('checkpoint predates qualified FP32 cumulative alignment policy')
    expected=dict(sampling_rate=24000,n_fft=1024,hop_size=256,win_size=1024,num_mels=100,fmin=0,fmax=None)
    if any(h.get(k)!=v for k,v in expected.items()):raise ValueError('BigVGAN acoustic representation is incompatible')
    if math.prod(h['upsample_rates'])!=256:raise ValueError('BigVGAN upsample ratio differs from acoustic hop')
    if not math.isfinite(features['pitch_mean']) or not math.isfinite(features['pitch_std']) or features['pitch_std']<=0:
        raise ValueError('invalid training pitch normalization')


@contextmanager
def bounded_regulator(module,torch,max_frames):
    original=module.regulate_len
    def bounded(durations,encoded,pace=1.,mel_max_len=None):
        if not math.isfinite(pace) or pace<=0 or not bool(torch.isfinite(durations).all()) or bool((durations<0).any()):raise ValueError('nonfinite/negative durations or invalid pace')
        # Match the pinned non-JIT regulator expression exactly, including .5
        # ties. imports() promotes its cumulative boundaries to FP32 only.
        reps=(durations.float()/pace+0.5).long()
        if bool((reps<0).any()) or bool((reps>max_frames).any()):raise ValueError('predicted duration exceeds declared audio bound; no truncation')
        frames=reps.sum(dim=1)
        if bool((frames<=0).any()):raise ValueError('zero predicted duration; acoustic model is not ready')
        if bool((frames>max_frames).any()):raise ValueError('predicted duration exceeds declared audio bound; no truncation')
        return original(durations,encoded,pace,mel_max_len)
    module.regulate_len=bounded
    try:yield
    finally:module.regulate_len=original


class Backend:
    def __init__(self,args):
        if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
        import numpy as np
        import torch
        from training import fastpitch_experiment as fp
        self.torch,self.np,self.device=torch,np,args.device
        self.max_frames=int(args.max_audio_seconds*24000/256)
        self.max_tokens=args.max_tokens;self.precision=args.precision;torch.set_num_threads(args.threads)
        checkpoint=load_bound(args.checkpoint,args.checkpoint_sha256,torch)
        config=checkpoint['config'];h=json.loads((args.bigvgan/'config.json').read_text(encoding='utf-8'))
        validate_pair(config,h,fp.VOCAB,fp.MEL_CONFIG)
        if sha(args.bigvgan/'config.json')!=BIG_CONFIG_SHA:raise ValueError('BigVGAN config pin mismatch')
        revision=json.loads((args.bigvgan/'revision.json').read_text(encoding='utf-8'))
        if revision['revision']!=BIG_MODEL_REV:raise ValueError('BigVGAN model revision mismatch')
        source=args.bigvgan/'source'
        if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=BIG_SOURCE_REV:
            raise ValueError('BigVGAN source revision mismatch')
        if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():raise ValueError('BigVGAN source has modifications')
        cls,_,_,_=fp.imports(args.source)
        import fastpitch.model as model_module
        self.model_module=model_module;self.fp=fp
        self.acoustic=cls(**config['model']);self.acoustic.load_state_dict(checkpoint['model'],strict=True)
        if not all(bool(torch.isfinite(p).all()) for p in self.acoustic.parameters()):raise ValueError('nonfinite acoustic weights')
        for name,value in [('pitch_mean',config['features']['pitch_mean']),('pitch_std',config['features']['pitch_std'])]:
            if not math.isclose(float(getattr(self.acoustic,name)[0]),value,rel_tol=1e-6,abs_tol=1e-6):raise ValueError('checkpoint pitch buffer differs from feature normalization')
        self.metadata=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=args.checkpoint_sha256,step=checkpoint['step'],epoch=checkpoint['epoch'],
                           checkpoint_config=config,checkpoint_provenance=checkpoint.get('provenance'),
                           fastpitch_source_pin=fp.UPSTREAM,fastpitch_source_manifest_sha256=sha(args.source/'UPSTREAM.json'),
                           loader_code_sha256=sha(fp.__file__),bigvgan_source_revision=BIG_SOURCE_REV,bigvgan_model_revision=BIG_MODEL_REV,
                           bigvgan_weight_sha256=BIG_WEIGHT_SHA,bigvgan_config_sha256=BIG_CONFIG_SHA,
                           vocoder_training_normalize_volume=h.get('normalize_volume'),runtime_waveform_normalization='none')
        del checkpoint
        sys.path.insert(0,str(source))
        from bigvgan import BigVGAN
        from env import AttrDict
        self.vocoder=BigVGAN(AttrDict(h),use_cuda_kernel=False)
        state=load_bound(args.bigvgan/'bigvgan_generator.pt',BIG_WEIGHT_SHA,torch,weights_only=True)
        self.vocoder.load_state_dict(state['generator'],strict=True);del state
        self.vocoder.remove_weight_norm()
        self.acoustic.to(args.device).eval();self.vocoder.to(args.device).eval()
        if not all(bool(torch.isfinite(p).all()) for p in self.vocoder.parameters()):raise ValueError('nonfinite vocoder weights')
        for model in (self.acoustic,self.vocoder):
            if any(p.device.type!=args.device for p in model.parameters()) or any(b.device.type!=args.device for b in model.buffers()):raise RuntimeError('backend has nonresident parameters/buffers')
            if any((b.is_floating_point() or b.is_complex()) and not bool(torch.isfinite(b).all()) for b in model.buffers()):raise ValueError('nonfinite model buffer')
        self.synchronize()
        self.metadata.update(load_vram=self.peaks(),resident_allocated_mib=torch.cuda.memory_allocated()/2**20 if args.device=='cuda' else None)

    def synchronize(self):
        if self.device=='cuda':self.torch.cuda.synchronize()

    def reset_peak(self):
        if self.device=='cuda':self.torch.cuda.reset_peak_memory_stats()

    def peaks(self):
        if self.device!='cuda':return dict(peak_allocated_mib=None,peak_reserved_mib=None)
        return dict(peak_allocated_mib=self.torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=self.torch.cuda.max_memory_reserved()/2**20)

    def generate(self,text):
        t=self.torch;seq=self.fp.tokens(text)[None].to(self.device)
        if seq.shape[1]>self.max_tokens:raise ValueError('effective text exceeds token bound')
        with t.inference_mode(),bounded_regulator(self.model_module,t,self.max_frames):
            with t.autocast(self.device,dtype=t.bfloat16,enabled=self.precision=='bf16'):
                mel,lens,*_=self.acoustic.infer(seq)
            frames=int(lens[0])
            if not 0<frames<=self.max_frames or not bool(t.isfinite(mel).all()):raise ValueError('invalid acoustic mel/length')
            audio=self.vocoder(mel[:,:,:frames].float()).squeeze().float().cpu().numpy()
        if audio.size!=frames*256:raise ValueError('vocoder output length differs from acoustic frames/hop')
        return audio


def write_pcm(path,pcm):
    with wave.open(str(path),'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(pcm)


def frontend(mode):
    if mode=='native':
        return lambda text:dict(tts_text=text,applied_rules=[],warnings=[])
    if mode=='refinement':
        from training.f5_refine_text import normalize_text
        return normalize_text
    raise ValueError('unknown frontend mode')


def measure_case(backend,case,normalizer,metrics,output,candidate,repetition,warmup=False,clock=time.perf_counter,frontend_mode='refinement'):
    import numpy as np
    backend.synchronize();backend.reset_peak();started=clock();raw=None;pcm=None
    row=dict(candidate=candidate,case=case['id'],size=case['size'],repetition=repetition,
             condition='warmup-case' if warmup else 'warm',source_text=case['text'],frontend_mode=frontend_mode,
             scope='local complete file: frontend+FastPitch+BigVGAN+PCM conversion+WAV write/close; no HTTP',streaming='one whole file, no streaming')
    path=output/(case['id']+('-warm' if warmup else '-r')+f'{repetition:02d}.wav')
    try:
        normalized=normalizer(case['text']);effective=normalized['tts_text'];row['effective_text']=backend.fp.normalize(effective)
        row.update(refinement_rules=normalized.get('applied_rules',[]),frontend_warnings=normalized.get('warnings',[]))
        raw=np.asarray(backend.generate(effective),dtype=np.float32).reshape(-1)
        backend.synchronize()
        finite=bool(np.isfinite(raw).all());row.update(finite=finite,samples=len(raw),peak=float(abs(raw).max()) if finite and raw.size else None,
                                                     outside_unit=int((abs(raw)>1).sum()) if finite else None,
                                                     samples_at_generator_bound=int((abs(raw)>=1).sum()) if finite else None,
                                                     fraction_at_generator_bound=float((abs(raw)>=1).mean()) if finite and raw.size else None)
        if not raw.size or not finite:raise ValueError('empty or nonfinite output')
        if len(raw)>backend.max_frames*256:raise ValueError('output exceeds declared audio bound')
        if row['outside_unit']:raise ValueError('overshoot; PCM conversion refused without gain/clipping')
        pcm=np.rint(raw*32767).astype('<i2').tobytes();write_pcm(path,pcm)
        elapsed=(clock()-started)*1000;chunks=[dict(ms=elapsed,bytes=len(pcm))]
        measured=metrics.pcm_metrics(pcm,chunks)
        row.update(measured,eof_ms=elapsed,rtf=elapsed/1000/(len(raw)/24000),chunks=chunks,
                   outcome='ok' if measured['first_speech_pcm_available_ms'] is not None else 'no_speech',
                   pcm_sha256=hashlib.sha256(pcm).hexdigest(),wav_sha256=sha(path))
        if not warmup and repetition==0:row['wav']=str(path)
        else:path.unlink()
    except Exception as error:
        row.update(outcome='failed',error=repr(error),failed_after_ms=(clock()-started)*1000)
    row.update(backend.peaks())
    # Diagnostic FLOAT is not required for serving the final PCM, so its I/O
    # follows the end-to-end timer. Retain first repetition only, bounded.
    if raw is not None and raw.size and repetition==0:
        import soundfile as sf
        raw_path=output/(case['id']+('-warmup' if warmup else '')+'-raw.wav')
        sf.write(raw_path,raw,24000,subtype='FLOAT');row.update(raw_float=str(raw_path),raw_float_sha256=sha(raw_path))
    return row


def asr_job(case,row,spoken):
    if not row.get('wav'):return None
    return dict(candidate=row['candidate'],case=case['id'],wav=row['wav'],expected_verbatim=case['text'],
                expected_spoken=spoken[case['id']],expected_effective=row['effective_text'],expected_frontend=row['effective_text'],
                cohort='validation-rep0',artifact='final-PCM16',generation_outcome=row['outcome'],
                frontend_mode=row['frontend_mode'],
                product_audio_valid=row['outcome']=='ok',effective_alias_policy='expected_frontend aliases expected_effective for existing common ASR tooling')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,default=Path('/var/tmp/aegis-v2-work/fastpitch/source'))
    p.add_argument('--bigvgan',type=Path,default=Path('/var/tmp/aegis-v2-work/bigvgan'));p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--gpu-authorized',action='store_true');p.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    p.add_argument('--frontend',choices=['native','refinement'],default='refinement',
                   help='native uses training NFC/lowercase/punctuation only; refinement adds explicit experimental PT-BR numbers/lexicon')
    p.add_argument('--candidate',default='fastpitch-bigvgan');p.add_argument('--repetitions',type=int,default=3);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-audio-seconds',type=float,default=60.);p.add_argument('--max-tokens',type=int,default=2048)
    a=p.parse_args();os.umask(0o077)
    if a.repetitions<1 or not math.isfinite(a.max_audio_seconds) or a.max_audio_seconds<=0 or a.max_tokens<1:p.error('positive counts/bounds required')
    if len(a.checkpoint_sha256)!=64 or any(c not in '0123456789abcdef' for c in a.checkpoint_sha256):p.error('exact lowercase checkpoint SHA256 required')
    if any((path/'.git').exists() for path in (a.output.resolve(),*a.output.resolve().parents)):p.error('private output must be outside Git')
    if a.device=='cuda':
        if not a.gpu_authorized:p.error('CUDA benchmark requires --gpu-authorized and exclusive worker queue')
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if active:raise RuntimeError('another GPU compute process is active; keep backends isolated')
    a.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[key]='1'
    metrics=metrics_module();load_start=time.perf_counter();backend=Backend(a);load_seconds=time.perf_counter()-load_start
    normalizer=frontend(a.frontend)
    corpus=REPO/'benchmarks/v2/validation.pt-BR.json';spoken_path=REPO/'benchmarks/v2/spoken-validation.pt-BR.json'
    cases=json.loads(corpus.read_text(encoding='utf-8'));spoken=json.loads(spoken_path.read_text(encoding='utf-8'))['cases']
    manifest=dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),command=sys.argv,**backend.metadata,
                  corpus_sha256=sha(corpus),spoken_corpus_sha256=sha(spoken_path),code_sha256=sha(__file__),metric_code_sha256=sha(REPO/'benchmarks/v2/benchmark.py'),
                  frontend_mode=a.frontend,
                  frontend=('f5-refinement-1 experimental numbers/lexicon then ' if a.frontend=='refinement' else '')+'pinned FastPitch training NFC/lowercase/punctuation mapping and exact vocabulary IDs',
                  frontend_training_match=a.frontend=='native',
                  normalizer_sha256=sha(REPO/'training/f5_refine_text.py') if a.frontend=='refinement' else None,
                  number_words_sha256=sha(REPO/'gateway/aegis_speech/normalization.py') if a.frontend=='refinement' else None,
                  precision=dict(acoustic_autocast=a.precision,parameters='FP32',vocoder='FP32'),
                  generator_bound='Pinned BigVGAN intrinsically clamps output [-1,1]; count abs(raw)>=1 before PCM conversion, no subsequent clipping/gain',
                  native_sample_rate=24000,final_format='PCM16LE mono24000Hz',dsp='none, no gain/trim/limiter',load_seconds=load_seconds,
                  max_audio_seconds=a.max_audio_seconds,max_tokens=a.max_tokens,warmup='3 retained observations per validation case before all measured repetitions',
                  repetitions=a.repetitions,concurrency=1,onset_rule=metrics.RULE,scope='complete-file local backend, all required processing and WAV close included; no HTTP or streaming claim')
    json_atomic(a.output/'manifest.json',manifest);rows=[];jobs=[]
    order=[(True,warm,case) for case in cases for warm in range(3)]
    order += [(False,rep,case) for rep in range(a.repetitions) for case in cases]
    for warmup,rep,case in order:
        backend.torch.manual_seed(40733133+rep)
        row=measure_case(backend,case,normalizer,metrics,a.output,a.candidate,rep,warmup,frontend_mode=a.frontend)
        row.update(requested_seed=40733133+rep,precision=a.precision,checkpoint_step=backend.metadata['step'],device=a.device,
                   first_model_synthesis=warmup and rep==0 and case['id']==cases[0]['id'])
        log=a.output/('warmup.jsonl' if warmup else 'results.jsonl')
        with log.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        if not warmup:
            rows.append(row);json_atomic(a.output/'summary.json',metrics.summarize(rows))
            if rep==0:
                job=asr_job(case,row,spoken)
                if job:jobs.append(job)
            json_atomic(a.output/'asr-jobs.json',dict(jobs=jobs,missing_rep0_cases=[c['id'] for c in cases if c['id'] not in {j['case'] for j in jobs}],
                                                   generation_manifest=str(a.output/'manifest.json'),policy='verbatim/spoken/effective remain separate; final test not opened'))
        print(json.dumps({k:row.get(k) for k in ('case','repetition','condition','outcome','ttfa_estimated_playback_ms','rtf','error')},ensure_ascii=False),flush=True)
    json_atomic(a.output/'completed.json',dict(observations=len(rows),warmups=3*len(cases),successes=sum(r['outcome']=='ok' for r in rows),
                                              failures=sum(r['outcome']!='ok' for r in rows),final_test_opened=False))


if __name__=='__main__':main()
