"""Private AegisVoice v2 acoustic/inference evidence; no production mutation."""
from __future__ import annotations
import argparse
import gc
import hashlib
import html
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path('/var/tmp/aegisvoice-v2-training-20261102')
OLD = Path('/var/tmp/aegis-v2-evaluation-20261102')
SOURCE = OLD / 'f5/source'
BASE = OLD / 'f5/model_last.safetensors'
RATE, HOP = 24000, 256
# Use the repository's exact standalone benchmark modules. The F5 environment
# has an unrelated regular package named 'benchmarks' that shadows namespaces.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks/architecture_compare'))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def write_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False));tmp.replace(path)

def bootstrap(cpu=False):
    if cpu: os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):os.environ[key]='1'
    sys.path.insert(0,str(SOURCE/'src'))
    import torch
    torch.set_num_threads(2)
    return torch

def vocos(device):
    from vocos import Vocos
    import torch
    v=Vocos.from_hparams(str(OLD/'f5/vocos/config.yaml'))
    v.load_state_dict(torch.load(OLD/'f5/vocos/pytorch_model.bin',map_location='cpu',weights_only=True), strict=True)
    return v.eval().to(device)

def refs(root, target_seconds=6.0):
    rows=[json.loads(x) for x in (root/'prepared/train.jsonl').read_text().splitlines()]
    # Choose complete original clips, never crop to manufacture a short transcript.
    return sorted(rows,key=lambda x:abs(x['duration']-target_seconds))[:3]

def reference(root, device):
    import soundfile as sf
    import torch
    from f5_tts.model.modules import MelSpec
    r=refs(root, target_seconds=10.0)[0]
    a,sr=sf.read(r['audio_filepath'],dtype='float32')
    assert sr==RATE and a.ndim==1
    mel=MelSpec(mel_spec_type='vocos')(torch.from_numpy(a)[None]).transpose(1,2).contiguous().to(device)
    return r,mel,len(a)//HOP

def corpora():
    p=Path(__file__).resolve().parents[1]/'benchmarks/architecture_compare'
    return json.loads((p/'corpus.pt-BR.json').read_text())+json.loads((p/'corpus.supplement.pt-BR.json').read_text())

def save_audio(path, values):
    import soundfile as sf
    import numpy as np
    values=np.asarray(values,dtype=np.float32).reshape(-1)
    if not values.size or not np.isfinite(values).all():raise ValueError('nonfinite/empty output')
    path.parent.mkdir(parents=True,exist_ok=True)
    sf.write(path,values,RATE,subtype='FLOAT')
    return dict(path=str(path),sha256=digest(path),audio_seconds=len(values)/RATE,peak=float(abs(values).max()),outside_unit=int((abs(values)>1).sum()))

def synthesize(model,v,root,label,cases,nfes,repeats=1):
    import torch
    import numpy as np
    from f5_tts.model.utils import convert_char_to_pinyin
    from f5_probe import duration_frames
    from benchmark import pcm_metrics
    dev=next(model.parameters()).device
    r,cond,frames=reference(root,dev)
    ref_text=r['text'].strip()+' '
    records=[];out=root/'evaluations'/label;out.mkdir(parents=True,exist_ok=True)
    write_json(out/'reference.json',r)
    for nfe in nfes:
        for case in cases:
            for rep in range(repeats):
                identifier=f"{case['id']}-nfe{nfe}-r{rep:02d}"
                tokens=convert_char_to_pinyin([ref_text+case['text'].lower()])
                oov=sorted({c for c in tokens[0] if c not in model.vocab_char_map})
                duration=duration_frames(frames,ref_text,case['text'].lower(),90)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    generated,trajectory=model.sample(cond=cond,text=tokens,duration=duration,steps=nfe,cfg_strength=2.0,sway_sampling_coef=-1.0,seed=40733133+rep,use_epss=False)
                torch.cuda.synchronize();diffusion=time.perf_counter()-start
                with torch.inference_mode():
                    wave=v.decode(generated[:,frames:,:].transpose(1,2).float())
                torch.cuda.synchronize()
                values=wave[0].float().cpu().numpy()
                elapsed=time.perf_counter()-start
                row=dict(label=label,case=case['id'],text=case['text'],effective_text=case['text'].lower(),nfe=nfe,repetition=rep,seed=40733133+rep,oov_tokens=oov,cold_generation=not records,reference_cache_hit=True,
                    diffusion_seconds=diffusion,total_seconds=elapsed,ttfa_ms=elapsed*1000,rtf=elapsed/(len(values)/RATE),
                    streaming='whole utterance; no PCM before complete ODE+Vocos',resident_allocated_mib=torch.cuda.memory_allocated()/2**20,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                    **save_audio(out/(identifier+'.wav'),values))
                if abs(values).max()<=1:
                    row.update(pcm_metrics(np.rint(values*32767).astype('<i2').tobytes(),[{'ms':elapsed*1000,'bytes':len(values)*2}],RATE))
                write_json(out/(identifier+'.json'),row);records.append(row)
                print(json.dumps({k:row[k] for k in ('label','case','nfe','total_seconds','rtf','peak_allocated_mib')}),flush=True)
                del generated,trajectory,wave
    write_json(out/'results.json',records)
    return records

def quick_synthesis(model,run_root,label):
    """Engine hook: preserve training RNG/mode; allocate only small frozen Vocos."""
    import torch
    root=Path(run_root);states=(random.getstate(),torch.get_rng_state(),torch.cuda.get_rng_state_all());was_training=model.training
    try:
        v=vocos(next(model.parameters()).device)
        model.eval()
        return synthesize(model,v,root,label,[corpora()[0],corpora()[-1]],[16])
    finally:
        model.train(was_training);random.setstate(states[0]);torch.set_rng_state(states[1]);torch.cuda.set_rng_state_all(states[2])
        if 'v' in locals():del v
        gc.collect();torch.cuda.empty_cache()

def roundtrip(root):
    torch=bootstrap(cpu=True)
    import numpy as np
    import soundfile as sf
    from f5_tts.model.modules import MelSpec
    from training.reconstruct import texture_metrics,compare_spectra
    v=vocos('cpu');mel=MelSpec(mel_spec_type='vocos');records=[]
    for r in refs(root):
        original=Path(r.get('source_audio_filepath',r.get('source_path',r['audio_filepath'])))
        # Locate preserved original by ID if manifest uses another provenance field.
        candidates=list((root/'extracted').glob('**/'+r['id']+'.wav'))
        if candidates:original=candidates[0]
        out=root/'roundtrip'/r['id'];out.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,out/'A-original-treated.wav')
        a,asr=sf.read(original,dtype='float32');b,sr=sf.read(r['audio_filepath'],dtype='float32')
        assert sr==RATE
        with torch.inference_mode():c=v.decode(mel(torch.from_numpy(b)[None]))[0].numpy()
        row=dict(id=r['id'],text=r['text'],original_path=str(original),original_sha256=digest(original),original_rate=asr,
            converted=save_audio(out/'B-input-24000.wav',b),vocos=save_audio(out/'C-vocos-24000.wav',c),
            source_metrics=texture_metrics(a,asr),converted_metrics=texture_metrics(b,sr),vocos_metrics=texture_metrics(c,sr),
            comparison=compare_spectra(b,c),perceptual_approval='pending owner listening; objective descriptors cannot prove robotic identity')
        write_json(out/'results.json',row);records.append(row)
    write_json(root/'roundtrip/results.json',records)
    print(json.dumps({'roundtrips':len(records)}))

def load_model(path):
    from f5_tts.model import CFM,DiT
    from f5_tts.model.utils import get_tokenizer
    from safetensors.torch import load_file
    from f5_probe import ARCH
    mapping,size=get_tokenizer(str(OLD/'f5/vocab.hf.txt'),'custom')
    model=CFM(transformer=DiT(**{**ARCH,'checkpoint_activations':False},text_num_embeds=size,mel_dim=100),mel_spec_kwargs={'mel_spec_type':'vocos'},odeint_kwargs={'method':'euler'},vocab_char_map=mapping)
    model.load_state_dict(load_file(str(path)),strict=True)
    return model.eval().cuda()

def final(root):
    torch=bootstrap()
    authorization=root/'gpu-window-authorized.json'
    if not authorization.is_file():raise ValueError('GPU authorization missing')
    auth=json.loads(authorization.read_text())
    if not auth.get('owner_authorized') or auth.get('status') != 'active' or not auth.get('production_stopped'):
        raise ValueError('GPU authorization must be active with production_stopped=true')
    paths=[('base',BASE)]
    for label in ('pilot','intermediate','best','last'):
        p=root/'checkpoints'/(label+'.safetensors')
        if p.exists():paths.append((label,p))
    # Engine may expose final raw weights through its own selection manifest.
    selection=root/'checkpoints/selection.json'
    if selection.exists():
        for label,p in json.loads(selection.read_text()).items():
            if isinstance(p,str) and Path(p).suffix=='.safetensors' and Path(p).exists() and (label,Path(p)) not in paths:paths.append((label,Path(p)))
    summaries=[]
    for label,p in paths:
        torch.cuda.reset_peak_memory_stats();load_started=time.perf_counter()
        model=load_model(p);v=vocos('cuda');torch.cuda.synchronize()
        from benchmark import gpu_snapshot
        load_evidence=dict(driver_snapshot=gpu_snapshot(),seconds=time.perf_counter()-load_started,resident_allocated_mib=torch.cuda.memory_allocated()/2**20,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,strict_load=True,dtype='FP32 parameters; BF16 autocast')
        write_json(root/'evaluations'/label/'load.json',load_evidence)
        cases=corpora() if label in ('base','best') else [corpora()[0],corpora()[6],corpora()[-1]]
        rows=synthesize(model,v,root,label,cases,[8,16,32],repeats=2 if label=='best' else 1)
        summaries.append(dict(label=label,checkpoint=str(p),sha256=digest(p),rows=len(rows),load=load_evidence))
        if label in ('base','best'):heldout(model,root,label)
        if label=='best':block_streaming(model,v,root)
        del model,v;gc.collect();torch.cuda.empty_cache()
    write_json(root/'evaluations/checkpoint-manifest.json',summaries)

def heldout(model,root,label):
    """Evaluate final untouched split only after model selection; never tunes it."""
    import torch
    import numpy as np
    import statistics
    from f5_tts.model.utils import convert_char_to_pinyin
    rows=[json.loads(line) for line in (root/'prepared/test.jsonl').read_text().splitlines()]
    losses=[];python_rng=random.getstate()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for i,row in enumerate(rows):
                random.seed(3000017+i);torch.manual_seed(3000017+i);torch.cuda.manual_seed_all(3000017+i)
                m=np.load(row['mel_filepath'],allow_pickle=False)
                if m.shape[0]==100:m=m.T
                inp=torch.from_numpy(np.ascontiguousarray(m))[None].cuda()
                loss,_,_=model(inp,text=convert_char_to_pinyin([row['text']]),lens=torch.tensor([inp.shape[1]],device='cuda'))
                if not torch.isfinite(loss):raise ValueError('nonfinite final test loss')
                losses.append(float(loss))
    finally:random.setstate(python_rng)
    write_json(root/'evaluations'/label/'heldout-test.json',dict(label=label,count=len(rows),loss=statistics.mean(losses),per_item_losses=losses,policy='final test only after training selection completed; no hyperparameter decisions'))

def block_streaming(model,v,root):
    """Sentence-block delivery measured with actual CUDA generation; no HTTP claim."""
    import torch
    import soundfile as sf
    import numpy as np
    full=next(c for c in corpora() if c['id']=='long_stability')
    pieces=[x.strip()+'.' for x in full['text'].split('.') if x.strip()]
    rows=[];chunks=[];start=time.perf_counter()
    for i,text in enumerate(pieces):
        batch=synthesize(model,v,root,'best-blocks',[{'id':f'block-{i:02d}','text':text}],[16])[0]
        chunks.append(sf.read(batch['path'],dtype='float32')[0]);rows.append(dict(block=i,available_ms=(time.perf_counter()-start)*1000,**batch))
    boundaries=[]
    for a,b in zip(chunks,chunks[1:]):
        boundaries.append(dict(sample_jump=float(abs(b[0]-a[-1])),last100ms_rms=float(np.sqrt(np.mean(a[-2400:]**2))),first100ms_rms=float(np.sqrt(np.mean(b[:2400]**2)))))
    meta=save_audio(root/'evaluations/best-blocks/concatenated.wav',np.concatenate(chunks))
    write_json(root/'evaluations/block-streaming.json',dict(blocks=rows,boundaries=boundaries,concatenated=meta,first_block_ms=rows[0]['available_ms'],total_ms=(time.perf_counter()-start)*1000,
        interpretation='Delivery by complete sentence blocks; full ODE solve per block. No true within-block neural streaming. Playback continuity depends on block generation vs duration; listen to unmodified joins.'))

def asr(root):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import math
    import soundfile as sf
    import numpy as np
    from scipy.signal import resample_poly
    from faster_whisper import WhisperModel
    from training.f5_asr_metrics import plain_word_comparison as comparison
    settings=dict(repository='Systran/faster-whisper-small',revision='536b0662742c02347bc0e980a01041f333bce120',device='cpu',compute_type='int8',cpu_threads=4,num_workers=1,language='pt',beam_size=5,temperature=0,vad_filter=False,condition_on_previous_text=False,resampling='scipy resample_poly Kaiser10 to16k; no gain or trimming',scoring='NFD/accents/case/punctuation only; no gateway normalization; numeric formatting can inflate WER')
    settings_hash=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    write_json(root/'asr/settings.json',dict(settings=settings,sha256=settings_hash,model_receipt_sha256=digest(OLD/'asr/download-receipt.json')))
    model=WhisperModel(str(OLD/'asr/model'),device='cpu',compute_type='int8',cpu_threads=4,num_workers=1,local_files_only=True)
    def transcribe(path):
        audio,rate=sf.read(path,dtype='float32')
        d=math.gcd(rate,16000)
        audio=resample_poly(audio,16000//d,rate//d,window=('kaiser',10)).astype(np.float32)
        seg,_=model.transcribe(audio,language='pt',beam_size=5,temperature=0,vad_filter=False,condition_on_previous_text=False)
        return ' '.join(s.text.strip() for s in seg)
    result_path=root/'asr/results.json'
    old_records=json.loads(result_path.read_text()) if result_path.exists() else []
    cache={r['cache_key']:r for r in old_records if r.get('settings_sha256')==settings_hash and r.get('cache_key')}
    cache_dir=root/'asr/cache';cache_dir.mkdir(parents=True,exist_ok=True)
    records=[]
    def check(source,metadata):
        sha=digest(source)
        key=hashlib.sha256(json.dumps([metadata,sha,settings_hash],sort_keys=True).encode()).hexdigest()
        cached_path=cache_dir/(key+'.json')
        if cached_path.exists():record=json.loads(cached_path.read_text())
        elif key in cache:record=cache[key]
        else:
            started=time.perf_counter();text=transcribe(source)
            record=dict(**metadata,hypothesis=text,audio_sha256=sha,settings_sha256=settings_hash,cache_key=key,seconds=time.perf_counter()-started,**comparison(metadata['reference'],text))
        write_json(cached_path,record)
        records.append(record);write_json(result_path,records)
        print(json.dumps({k:record[k] for k in ('label','case','word_error_rate')},ensure_ascii=False),flush=True)
    for p in sorted((root/'evaluations').glob('*/*.json')):
        if p.name in ('results.json','reference.json'):continue
        row=json.loads(p.read_text())
        if not isinstance(row,dict) or not row.get('path') or not row.get('text'):continue
        check(row['path'],dict(label=row['label'],case=row['case'],nfe=row['nfe'],repetition=row['repetition'],reference=row['text']))
    for p in sorted((root/'roundtrip').glob('*/results.json')):
        row=json.loads(p.read_text())
        for name,source in [('original',p.parent/'A-original-treated.wav'),('converted',p.parent/'B-input-24000.wav'),('vocos',p.parent/'C-vocos-24000.wav')]:
            check(source,dict(label='roundtrip-'+name,case=row['id'],reference=row['text']))
    write_json(result_path,records)

def package(root):
    dest=root/'listening';dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for sub in ('roundtrip','evaluations'):
        for p in sorted((root/sub).rglob('*.wav')):
            rel=p.relative_to(root);q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True)
            if not q.exists():os.link(p,q)
            rows.append((str(rel),digest(p)))
    qwen=OLD/'qwen-fixed-cohort'
    for p in sorted(qwen.glob('*.wav')):
        q=dest/'qwen'/p.name;q.parent.mkdir(exist_ok=True)
        if not q.exists():os.link(p,q)
        rows.append((str(q.relative_to(dest)),digest(p)))
    conditioning=None
    if (root/'prepared/train.jsonl').exists():
        conditioning=refs(root,target_seconds=10.0)[0]
        for name,key in [('A-conditioning-original.wav','source_audio_filepath'),('B-conditioning-24000.wav','audio_filepath')]:
            source=Path(conditioning[key]);q=dest/'reference'/name;q.parent.mkdir(exist_ok=True)
            if not q.exists():os.link(source,q)
            rows.append((str(q.relative_to(dest)),digest(source)))
    metadata={'dataset_zip_sha256':'5bbd8288d9ab795a83ffd1599b0c5a32c00b401cb903b368a183d7ee7580c8df','source_commit':'45868abaabd74aabf03cd673d6dd00cc9d9c01ae','base_revision':'0af67bc061699cab46218c150948910fde209614','base_sha256':'c25e2678bdd4304071e3d0e39a283a7d23ffbc23678080c5b0b741dda2e07f0f','vocab_sha256':'2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c','vocos_sha256':'97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a','owner_perceptual_approval':False}
    if conditioning:metadata['conditioning_reference']=conditioning
    for name,source in [('checkpoints',root/'evaluations/checkpoint-manifest.json'),('training',root/'training-result.json'),('asr',root/'asr/results.json')]:
        if source.exists():metadata[name]=json.loads(source.read_text())
    write_json(dest/'experiment.json',metadata)
    texts={c['id']:c['text'] for c in corpora()}
    if conditioning:
        texts.update({'A-conditioning-original':conditioning['original_text'],'B-conditioning-24000':conditioning['original_text']})
    body=['<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>AegisVoice v2 — avaliação privada</title><style>body{font:16px system-ui;max-width:1000px;margin:2em auto}article{padding:1em;border-bottom:1px solid #bbb}audio{width:100%}code{font-size:11px}</style><h1>AegisVoice v2 — comparação privada</h1><p>Candidatas experimentais. A voz definitiva depende da escuta do proprietário. Sem efeitos reaplicados. F5 usa frases completas; blocos não demonstram streaming neural.</p>']
    for path,sha in rows:
        case=Path(path).stem.split('-nfe')[0]
        phrase=texts.get(case,'')
        body.append(f'<article><p>{html.escape(path)}</p><p>{html.escape(phrase)}</p><audio controls preload="none" src="{html.escape(path,quote=True)}"></audio><code>SHA-256 {sha}</code></article>')
    body.append('</html>');(dest/'index.html').write_text('\n'.join(body));write_json(dest/'manifest-sha256.json',dict(rows))
    print(json.dumps({'listening':str(dest),'audio_files':len(rows)}))

def main():
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,default=ROOT);p.add_argument('--action',choices=['roundtrip','final','checkpoint','asr','package'],required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--label',default='pilot-recovered')
    args=p.parse_args()
    if args.action=='checkpoint':
        bootstrap();auth=json.loads((args.run_root/'gpu-window-authorized.json').read_text())
        if not (auth.get('status')=='active' and auth.get('production_stopped')):raise ValueError('active exclusive GPU window required')
        model=load_model(args.checkpoint);quick_synthesis(model,args.run_root,args.label)
    else:{'roundtrip':roundtrip,'final':final,'asr':asr,'package':package}[args.action](args.run_root)
if __name__=='__main__':main()
