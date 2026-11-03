"""Private Cadu ONNX PT-BR language gate, CPU only; no production integration.

Reuses the reviewed phoneme inventory, experimental text normalizer and strict
PCM conversion. Uses raw phoneme_ids_to_audio because runtime synthesize clips
even when normalization is disabled. It never downloads or trains anything.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/var/tmp/aegis-v2-work/piper')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.chmod(0o600);temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--overlay',type=Path,default=ROOT/'deps')
    p.add_argument('--run',type=Path,default=ROOT/'base-cadu-cpu')
    p.add_argument('--frontends',nargs='+',choices=['raw','refinement'],default=['raw','refinement'])
    p.add_argument('--zero-noise',action='store_true',help='Separate deterministic hypothesis: set both ONNX noise inputs0')
    args=p.parse_args();os.umask(0o077);os.environ['CUDA_VISIBLE_DEVICES']=''
    if len(args.frontends)!=len(set(args.frontends)):raise ValueError('Unique frontend variants required')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
    os.nice(19);sys.path.insert(0,str(args.overlay))
    if (args.run/'recipe.json').exists():raise ValueError('Fresh private output required')
    args.run.mkdir(parents=True,exist_ok=True,mode=0o700)
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf
    from piper import PiperVoice
    from piper.config import PiperConfig,SynthesisConfig
    from piper.phonemize_espeak import EspeakPhonemizer,ESPEAK_DATA_DIR
    from piper.phoneme_ids import phonemes_to_ids
    from training.frontend_check import check_inventory
    from training.f5_refine_text import normalize_text
    from training.prepare_dataset import encode_pcm16,resample_pcm
    from training.evaluate_piper import write_wav
    planned=json.loads((ROOT.parent/'NEXT-piper-base.json').read_text())
    candidate=planned['candidates'][0];model=ROOT/'downloads/pt_BR-cadu-medium.onnx'
    config_path=Path(candidate['config']['local']);config=json.loads(config_path.read_text())
    if model.stat().st_size!=candidate['onnx']['bytes'] or sha(model)!=candidate['onnx']['sha256_expected_lfs']:
        raise ValueError('Cadu ONNX size/SHA mismatch')
    if sha(config_path)!=candidate['config']['sha256']:raise ValueError('Config SHA mismatch')
    mapping=config['phoneme_id_map']
    if len(mapping)!=161 or mapping.get('\u0303')!=[141] or config['espeak']['voice']!='pt-br':
        raise ValueError('Exact PT-BR161symbol map with preserved combining tilde required')
    effective_config=json.loads(json.dumps(config))
    if args.zero_noise:effective_config['inference'].update(noise_scale=0.,noise_w=0.)
    save(args.run/'voice-config-effective.json',effective_config)
    cfg=PiperConfig.from_dict(effective_config);cfg.vowel_clusters=None
    frontend=EspeakPhonemizer(ESPEAK_DATA_DIR)
    corpus_path=REPO/'benchmarks/v2/validation.pt-BR.json'
    corpus=json.loads(corpus_path.read_text());spoken_path=REPO/'benchmarks/v2/spoken-validation.pt-BR.json'
    spoken=json.loads(spoken_path.read_text())['cases'];items=[];phones={}
    for mode in args.frontends:
        for case in corpus:
            normalized=normalize_text(case['text']) if mode=='refinement' else dict(tts_text=case['text'],applied_rules=[],warnings=[])
            ident=mode+'-'+case['id'];text=normalized['tts_text']
            phones[ident]=frontend.phonemize('pt-br',text,vowel_clusters=None)
            items.append(dict(id=ident,case=case['id'],frontend=mode,text=text,
                 expected_verbatim=case['text'],expected_spoken=spoken[case['id']],expected_effective=text,
                 refinement_rules=normalized.get('applied_rules',[]),frontend_warnings=normalized.get('warnings',[])))
    phone_by_text={row['text']:phones[row['id']] for row in items}
    report=check_inventory(items,config,lambda text:phone_by_text[text],
                           lambda sentence,m:phonemes_to_ids(sentence,id_map=m))
    report.update(vowel_clusters=None,combining_tilde_id=mapping['\u0303'],config_sha256=sha(config_path),
                  corpus_sha256=sha(corpus_path),expected_cases=7*len(args.frontends),actual_cases=len(items))
    save(args.run/'frontend.json',report)
    if not report['passed']:raise ValueError('Unknown phonemes; never silently drop them: '+repr(report['unknown_by_id']))
    overlay_manifest=[dict(path=str(path.relative_to(args.overlay)),bytes=path.stat().st_size,sha256=sha(path))
                      for path in sorted(args.overlay.rglob('*')) if path.is_file() and '__pycache__' not in path.parts]
    save(args.run/'overlay-manifest.json',overlay_manifest)
    ort.set_seed(40733133);np.random.seed(40733133)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    if session.get_providers()!=['CPUExecutionProvider']:raise ValueError('CPUExecutionProvider only required')
    voice=PiperVoice(session=session,config=cfg);syn=SynthesisConfig(normalize_audio=False,volume=1.)
    save(args.run/'recipe.json',dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),pid=os.getpid(),
         command=sys.argv,niceness=os.nice(0),code_sha256=sha(__file__),model=str(model),model_sha256=sha(model),
         model_revision=candidate['onnx']['revision'],config_sha256=sha(config_path),map_symbols=161,
         effective_config_sha256=sha(args.run/'voice-config-effective.json'),effective_inference=effective_config['inference'],
         variant='zero-noise' if args.zero_noise else 'default-noise',
         runtime_version=importlib.metadata.version('piper-tts'),onnxruntime=ort.__version__,numpy=np.__version__,
         providers=session.get_providers(),intra_op_threads=2,inter_op_threads=1,seed=40733133,
         overlay_manifest_sha256=sha(args.run/'overlay-manifest.json'),
         phonemizer='bundled eSpeak pt-br; Unicode NFD codepoints preserved; vowel_clusters=None; strict OOV failure',
         synthesis='PiperVoice.phoneme_ids_to_audio raw session.run; same per-sentence frontend boundaries as synthesize',
         gain=1.,normalize_audio=False,limiter=False,clipping=False,raw_rate=22050,final_rate=24000,
         resampling='concatenate full sentence outputs, then existing resample_pcm Kaiser10',
         corpus_sha256=sha(corpus_path),spoken_reference_sha256=sha(spoken_path),
         normalizer_sha256=sha(REPO/'training/f5_refine_text.py'),final_test_opened=False,
         timing='CPU language qualification concurrent with GPU FastPitch077; not production/A2000 latency benchmark',
         license_scope='Private local research only; repositoryMIT/runtimeGPL/Lessac lineage unresolved; no redistribution/use conclusion'))
    records=[];jobs=[]
    for item in items:
        started=time.perf_counter();row=dict(item);raw=[];sentence_receipts=[]
        try:
            ort.set_seed(40733133)
            for index,phonemes in enumerate(phones[item['id']]):
                if not phonemes:raise ValueError('Empty phoneme sentence')
                missing=set(phonemes)-set(mapping)
                if missing:raise ValueError('Missing phonemes '+repr(missing))
                ids=phonemes_to_ids(phonemes,id_map=mapping)
                if ids[0:2]!=[1,0] or ids[-1]!=2:raise ValueError('BOS/pad/EOS recipe differs')
                values=np.asarray(voice.phoneme_ids_to_audio(ids,syn),dtype=np.float32)
                if values.ndim!=1 or not values.size:raise ValueError('Empty/nonmono ONNX output')
                raw.append(values)
                sentence_receipts.append(dict(index=index,phonemes=phonemes,phoneme_ids=ids,samples=len(values),
                         finite=bool(np.isfinite(values).all()),nonfinite=int((~np.isfinite(values)).sum()),
                         peak=float(np.abs(values).max()) if np.isfinite(values).all() else None,
                         samples_mod_hop256=len(values)%256))
            values=np.concatenate(raw);path=args.run/(item['id']+'-raw22050.wav')
            sf.write(path,values,22050,subtype='FLOAT');row.update(raw_float=str(path),raw_sha256=sha(path),
                native_samples=len(values),native_seconds=len(values)/22050,empty=False,
                nonfinite_samples=int((~np.isfinite(values)).sum()),native_overshoot_samples=int((np.abs(values)>1).sum()),
                native_peak=float(np.abs(values).max()) if np.isfinite(values).all() else None,
                sentence_receipts=sentence_receipts,sentence_outputs_complete=len(raw)==len(phones[item['id']]))
            if not np.isfinite(values).all():raise ValueError('Native output nonfinite')
            final=resample_pcm(values.astype(np.float64),22050,24000)
            if len(final)!=math.ceil(len(values)*160/147):raise ValueError('Resampling length mismatch')
            final_float=args.run/(item['id']+'-float24000.wav');sf.write(final_float,final,24000,subtype='FLOAT')
            row.update(final_samples=len(final),audio_seconds=len(final)/24000,final_peak=float(np.abs(final).max()),
                       final_overshoot_samples=int((np.abs(final)>1).sum()),final_float=str(final_float),final_float_sha256=sha(final_float))
            pcm=encode_pcm16(final);wav=args.run/(item['id']+'-pcm24000.wav');write_wav(wav,pcm)
            restored,rate=sf.read(wav,dtype='int16',always_2d=True)
            if rate!=24000 or restored.shape!=(len(final),1):raise ValueError('PCM WAV incomplete/truncated')
            row.update(outcome='ok',wav=str(wav),sha256=sha(wav),pcm_wav_complete=True,
                       semantic_truncation='Not established by WAV integrity; compare ASR complete text/tail, no hearing claim')
        except Exception as error:
            row.update(outcome='failed',error=repr(error),empty=not bool(raw),sentence_receipts=sentence_receipts)
        row.update(wall_seconds=time.perf_counter()-started,max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        records.append(row);save(args.run/(item['id']+'.json'),row)
        if row.get('raw_float') and not row.get('nonfinite_samples'):
            source=row.get('wav',row['raw_float']);diagnostic=row['outcome']!='ok'
            jobs.append(dict(candidate='piper-cadu-base-'+('zero-noise-' if args.zero_noise else '')+item['frontend'],case=item['case'],wav=source,
                         expected_verbatim=item['expected_verbatim'],expected_spoken=item['expected_spoken'],expected_effective=item['expected_effective'],
                         cohort='validation-CPU-FP077-concurrent'+('-zero-noise' if args.zero_noise else '')+('-raw-FLOAT-diagnostic' if diagnostic else ''),
                         product_pcm_valid=not diagnostic))
        save(args.run/'results.json',records);save(args.run/'asr-jobs.json',dict(jobs=jobs))
        save(args.run/'progress.json',dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),completed=len(records),total=len(items),case=item['id'],outcome=row['outcome']))
        print(json.dumps({k:row.get(k) for k in ['id','outcome','audio_seconds','native_peak','final_peak','wall_seconds','error']},ensure_ascii=False),flush=True)
    save(args.run/'done.json',dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),cases=len(records),
         ok=sum(r['outcome']=='ok' for r in records),failed=sum(r['outcome']!='ok' for r in records),asr_jobs=len(jobs),
         cpu_only=True,torch_imported='torch' in sys.modules,max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))


if __name__=='__main__':main()
