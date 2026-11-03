"""Bounded inference-only experiments for the already trained AegisVoice F5.

Private results, immutable weights, no production configuration writes. The
existing av2_ops guard owns the one authorized GPU window and final rollback.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import wave

from training import f5_evaluate as ev
from training.f5_asr_metrics import plain_word_comparison
from training.f5_refine_segments import decode_generated, linguistic_chunks

ROOT = Path('/var/tmp/aegisvoice-v2-refinement-20261103')
TRAIN = ev.ROOT
CHECKPOINT = TRAIN / 'checkpoints/best.safetensors'
BEST_SHA = 'b3faf799babfcdf33768beb8dcc26f7f94c3c297b3b3a89549ab32cb9380fb44'
SEED = 40733133
SHORT = 'A janela ficou aberta, mas a chuva já passou.'
LEXICAL = 'Aegis, confira a CPU e a API. O SUS confirmou o CPF, e a URL usa HTTPS.'


def controlled_cases():
    long = next(c['text'] for c in ev.corpora() if c['id'] == 'long_stability')
    return [dict(id='ordinary', text=SHORT), dict(id='lexical', text=LEXICAL),
            dict(id='long', text=long)]


def write(root, name, value):
    ev.write_json(root / name, value)


def references(root):
    value = json.loads((root / 'references.json').read_text())
    rows = value['references'] if isinstance(value, dict) else value
    if len(rows) != 4 or len({r['group_id'] for r in rows}) != 4:
        raise ValueError('four references from distinct train groups required')
    control = next(r for r in rows if r['id'] == 'aegis_ptbr_0586')
    return [control] + [r for r in rows if r is not control]


def reference_cases(root):
    value = json.loads((root / 'references.json').read_text())
    return value['cases']


def preflight(root):
    from safetensors import safe_open
    rows = references(root)
    assert ev.digest(CHECKPOINT) == BEST_SHA
    train_ids = {json.loads(s)['id'] for s in (TRAIN / 'prepared/train.jsonl').read_text().splitlines()}
    for r in rows:
        assert r['id'] in train_ids and ev.digest(r['audio_filepath']) == r['sha256']
        assert ev.digest(r['source_audio_filepath']) == r['source_audio_sha256']
        assert ev.digest(r['mel_filepath']) == r['mel_sha256']
    with safe_open(str(CHECKPOINT), framework='pt', device='cpu') as trained, safe_open(str(ev.BASE), framework='pt', device='cpu') as base:
        assert set(trained.keys()) == set(base.keys())
        shapes = {k: trained.get_slice(k).get_shape() for k in trained.keys()}
        assert all(v == base.get_slice(k).get_shape() for k, v in shapes.items())
    from training.f5_refine_text import normalize_text, pronunciation_cases
    for c in pronunciation_cases():
        assert c['display_text']
    example = normalize_text(LEXICAL)
    assert example['display_text'] == LEXICAL and 'Égide' not in example['tts_text']
    free = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
    assert free > 2 * 1024**3
    write(root, 'preflight.json', dict(checkpoint=str(CHECKPOINT), checkpoint_sha256=BEST_SHA,
        strict_key_shape_check=len(shapes), train_only_references=[r['id'] for r in rows],
        disk_free_bytes=free, extra_results_budget_bytes=512*1024**2,
        no_weight_updates=True, desired_name_pronunciation='A-é-jis',
        desired_name_provenance='owner explicitly confirmed A-é-jis on 2026-11-03',
        checkpoint_hashes={p.name:ev.digest(p) for p in sorted((TRAIN/'checkpoints').iterdir()) if p.is_file()},
        previous_checkpoints='read-only; optimizer and all previous candidates preserved'))


def baseline(root):
    import numpy as np
    from benchmark import pcm_metrics
    out = root / 'qwen'; out.mkdir(exist_ok=True)
    results = []
    for case in controlled_cases():
        payload = json.dumps(dict(text=case['text'], stream=True, response_format='pcm', interrupt_policy='reject_if_busy')).encode()
        request = urllib.request.Request('http://127.0.0.1:8001/v1/aegis/speech', data=payload, headers={'Content-Type':'application/json'})
        body = bytearray(); arrivals = []; start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=180) as response:
            assert response.headers.get_content_type() == 'application/octet-stream'
            assert int(response.headers.get('X-Aegis-Sample-Rate','24000')) == ev.RATE
            while block := response.read1(4096):
                arrivals.append(dict(ms=(time.perf_counter()-start)*1000, bytes=len(block)))
                body.extend(block)
        total = time.perf_counter()-start
        assert body and len(body)%2 == 0
        path = out / (case['id']+'.wav')
        with wave.open(str(path),'wb') as audio:
            audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(ev.RATE);audio.writeframes(body)
        row = dict(id='qwen-'+case['id'], stage='controlled', label='Qwen atual', case=case['id'],
            text=case['text'], tts_text=case['text'], path=str(path), sha256=ev.digest(path),
            total_seconds=total, rtf=total/(len(body)/2/ev.RATE), arrivals=arrivals,
            note='current production frontend and protected lexicon preserved',
            **pcm_metrics(bytes(body),arrivals,ev.RATE))
        results.append(row)
        write(root,'qwen/results.json',results)
        print(json.dumps(dict(qwen=case['id'],total_seconds=total)),flush=True)


class Experiment:
    def __init__(self, root):
        self.root=root; self.rows=[];self.asr_rows=[]
        self.torch=ev.bootstrap();self.refs=references(root);self.control=self.refs[0]
        self.model=ev.load_model(CHECKPOINT);self.v=ev.vocos('cuda');self.cond={}
        self.torch.cuda.synchronize()
        write(root,'load.json',dict(checkpoint_sha256=BEST_SHA,strict=True,
            resident_mib=self.torch.cuda.memory_allocated()/2**20,
            source_revision='45868abaabd74aabf03cd673d6dd00cc9d9c01ae',
            dtype='FP32 weights; BF16 autocast; Euler; CFG2; sway-1', seed=SEED))

    def conditioning(self, ref):
        import numpy as np
        if ref['id'] not in self.cond:
            m=np.load(ref['mel_filepath'],allow_pickle=False)
            if m.shape[0] == 100:m=m.T
            self.cond[ref['id']]=self.torch.from_numpy(np.ascontiguousarray(m))[None].cuda()
        return self.cond[ref['id']]

    def duration(self, ref, text):
        from f5_probe import duration_frames
        return duration_frames(int(ref['duration']*ev.RATE)//ev.HOP, ref['text'].strip()+' ',text.lower(),90)

    def sample(self, ref, text, nfe, generated_frames=None, seed=SEED):
        from f5_tts.model.utils import convert_char_to_pinyin
        torch=self.torch;cond=self.conditioning(ref)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        tokens=convert_char_to_pinyin([ref['text'].strip()+' '+text.lower()])
        oov=sorted({c for c in tokens[0] if c not in self.model.vocab_char_map})
        if oov:raise ValueError(f'OOV tokens: {oov}')
        duration=self.duration(ref,text) if generated_frames is None else cond.shape[1]+generated_frames
        # DiT's precomputed text positions are stricter than CFM's generic cap.
        maximum=self.model.transformer.text_embed.precompute_max_pos
        if duration > maximum or len(tokens[0]) >= duration:
            raise ValueError('context budget exceeded; split complete text instead of truncating it')
        frontend=time.perf_counter()-started
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            generated,trajectory=self.model.sample(cond=cond,text=tokens,duration=duration,
                steps=nfe,cfg_strength=2.0,sway_sampling_coef=-1.0,seed=seed,use_epss=False)
        torch.cuda.synchronize()
        del trajectory
        return generated,dict(started=started,frontend_seconds=frontend,
            diffusion_seconds=time.perf_counter()-started-frontend,total_frames=int(duration),
            conditioning_frames=int(cond.shape[1]),reference_id=ref['id'],nfe=nfe,seed=seed,
            duration_policy='upstream UTF8 byte-ratio' if generated_frames is None else 'explicit generated mel-frame budget',
            cold_generation=not self.rows)

    def decode(self, generated, timing, *, legacy=False, context=32):
        torch=self.torch
        with torch.inference_mode():
            if legacy:
                # Exact previous experiment: one copied conditioning frame remains.
                wave=self.v.decode(generated[:,timing['conditioning_frames']-1:,:].transpose(1,2).float())
                plan=dict(policy='legacy floor reference length; cold Vocos edge')
            else:
                wave,plan=decode_generated(self.v,generated,timing['conditioning_frames'],lookback_frames=context)
        torch.cuda.synchronize()
        values=wave[0].float().cpu().numpy()
        elapsed=time.perf_counter()-timing['started']
        return values,dict(**{k:v for k,v in timing.items() if k!='started'},total_seconds=elapsed,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,decode_plan=plan)

    def record(self, identifier, stage, label, case, text, tts_text, values, timing, **extra):
        import numpy as np
        from benchmark import pcm_metrics
        row=dict(id=identifier,stage=stage,label=label,case=case,text=text,tts_text=tts_text,
            **timing,**ev.save_audio(self.root/'audio'/(identifier+'.wav'),values),**extra)
        row['rtf']=row['total_seconds']/row['audio_seconds']
        if row['peak'] <= 1:
            pcm=np.rint(values*32767).astype('<i2').tobytes()
            row.update(pcm_metrics(pcm,[dict(ms=row['total_seconds']*1000,bytes=len(pcm))],ev.RATE))
        else:row['pcm16_declined']='native FLOAT exceeds unit range; no clipping/normalization'
        self.rows.append(row);write(self.root,'results.json',self.rows)
        print(json.dumps(dict(id=identifier,seconds=row['total_seconds'],rtf=row['rtf']),ensure_ascii=False),flush=True)
        return row

    def synth(self, identifier, stage, label, case, text, ref, nfe, *, tts_text=None, generated_frames=None, legacy=False, seed=SEED):
        tts_text=tts_text or text
        generated,timing=self.sample(ref,tts_text,nfe,generated_frames,seed)
        values,receipt=self.decode(generated,timing,legacy=legacy)
        row=self.record(identifier,stage,label,case,text,tts_text,values,receipt)
        del generated
        return row

    def transcribe(self, row):
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        if not hasattr(self,'asr'):
            from faster_whisper import WhisperModel
            self.asr=WhisperModel(str(ev.OLD/'asr/model'),device='cpu',compute_type='int8',cpu_threads=2,num_workers=1,local_files_only=True)
            write(self.root,'asr-settings.json',dict(repository='Systran/faster-whisper-small',revision='536b0662742c02347bc0e980a01041f333bce120',device='cpu',compute_type='int8',cpu_threads=2,
                language='pt',beam_size=5,temperature=0,vad_filter=False,condition_on_previous_text=False,
                interpretation='auxiliary only; name/acronym/number spelling ambiguity; never owner approval'))
        cached=next((r for r in self.asr_rows if r['audio_sha256']==row['sha256'] and r['expected']==row['tts_text']),None)
        if cached:return cached
        a,rate=sf.read(row['path'],dtype='float32');d=math.gcd(rate,16000)
        a=resample_poly(a,16000//d,rate//d,window=('kaiser',10)).astype(np.float32)
        start=time.perf_counter()
        segments,_=self.asr.transcribe(a,language='pt',beam_size=5,temperature=0,vad_filter=False,condition_on_previous_text=False)
        segments=list(segments);hypothesis=' '.join(s.text.strip() for s in segments)
        receipt=dict(id=row['id'],audio_sha256=row['sha256'],expected=row['tts_text'],hypothesis=hypothesis,
            segments=[dict(start=s.start,end=s.end,text=s.text) for s in segments],
            seconds=time.perf_counter()-start,**plain_word_comparison(row['tts_text'],hypothesis))
        self.asr_rows.append(receipt);write(self.root,'asr.json',self.asr_rows)
        print(json.dumps(dict(asr=row['id'],wer=receipt['word_error_rate'],hypothesis=hypothesis),ensure_ascii=False),flush=True)
        return receipt

    def descriptor(self,row):
        from training.f5_acoustics import describe
        cache=self.root/'acoustic-cache';cache.mkdir(exist_ok=True)
        return describe(row['path'],cache,row['sha256'])

    def grid(self):
        import numpy as np
        cases=reference_cases(self.root);scores=[]
        control_frames=int(self.conditioning(self.control).shape[1])
        durations={c['id']:self.duration(self.control,c['text'])-control_frames for c in cases}
        # Equal generated frame budgets, same text/seed/weights across references.
        for ref in self.refs:
            for nfe in (8,16):
                checks=[]
                for c in cases[:2]:
                    row=self.synth(f"ref-{ref['id']}-{c['id']}-{nfe}",'references',ref['id'],c['id'],c['text'],ref,nfe,generated_frames=durations[c['id']])
                    checks.append(self.transcribe(row))
                scores.append(dict(reference_id=ref['id'],nfe=nfe,word_edits=sum(c['word_edits'] for c in checks),reference_words=sum(len(c['reference_tokens']) for c in checks)))
        # Select two references by aggregate intelligibility; no single outlier.
        ref_order=sorted(self.refs,key=lambda r:(sum(s['word_edits'] for s in scores if s['reference_id']==r['id']),self.refs.index(r)))
        for ref in ref_order[:2]:
            for nfe in (8,16):
                c=cases[2];row=self.synth(f"confirm-{ref['id']}-{nfe}",'references',ref['id'],c['id'],c['text'],ref,nfe,generated_frames=durations[c['id']])
                check=self.transcribe(row)
                score=next(s for s in scores if s['reference_id']==ref['id'] and s['nfe']==nfe)
                score['word_edits']+=check['word_edits'];score['reference_words']+=len(check['reference_tokens'])
                score['confirmation_included']=True
        # Acoustic distances are tie-breakers only, not an identity classifier.
        refdesc={r['id']:self.descriptor(dict(path=r['audio_filepath'],sha256=r['sha256'])) for r in self.refs}
        def vector(d):
            f=d['yin_active_f0'];t=d['common24k_texture']
            return np.array([f['median_hz'],f['p90_hz']-f['p10_hz'],t['spectral_flatness_active_mean'],t['power_above_4000hz_ratio']])
        anchors=np.array([vector(d) for d in refdesc.values()]);center=np.median(anchors,axis=0)
        scale=np.maximum(np.percentile(anchors,75,axis=0)-np.percentile(anchors,25,axis=0),[10,20,.001,.005])
        descriptors=[]
        for s in scores:
            relevant=[r for r in self.rows if r['stage']=='references' and r['reference_id']==s['reference_id'] and r['nfe']==s['nfe']]
            distances=[]
            for r in relevant:
                d=self.descriptor(r);distance=float(np.mean(abs((vector(d)-center)/scale)))
                distances.append(distance);descriptors.append(dict(id=r['id'],descriptor=d,pool_distance=distance))
            s['wer']=s['word_edits']/s['reference_words'];s['texture_distance_median']=float(np.median(distances))
        confirmed=[s for s in scores if s.get('confirmation_included')]
        selected=min(confirmed,key=lambda s:(s['wer'],s['texture_distance_median'],s['nfe']))
        write(self.root,'reference-comparison.json',dict(scores=scores,descriptors=descriptors,selected=selected,
            selection_policy='three phrases for finalists; minimize auxiliary WER then median distance to treated-reference texture; provisional technical selection, requires owner listening',
            fixed_generated_frames=durations,owner_approval=False))
        return next(r for r in self.refs if r['id']==selected['reference_id']),selected['nfe']

    def pronunciations(self, ref, nfe):
        from training.f5_refine_text import pronunciation_cases
        cases=pronunciation_cases();write(self.root,'pronunciation-cases.json',cases)
        for i,c in enumerate(cases):
            row=self.synth(f'pronunciation-{i:02d}','pronunciation',c['id'],c['id'],c['display_text'],ref,nfe,tts_text=c['tts_text'])
            self.transcribe(row)

    def alignment(self):
        import numpy as np
        text='Não apague o arquivo. Primeiro verifique a última linha.'
        generated,timing=self.sample(self.control,text,16)
        outputs={};receipt=[]
        for name,legacy,context in [('legacy',True,0),('aligned',False,0),('context32',False,32)]:
            decode_start=time.perf_counter()
            values,t=self.decode(generated,timing,legacy=legacy,context=context)
            # Same generation shared; independent decode latency is also reported.
            t['decode_seconds']=time.perf_counter()-decode_start
            t['comparison_timing']='diagnostic only: shared generation; total includes previous decodes and CPU ASR; exclude from performance summaries'
            row=self.record('alignment-'+name,'alignment',name,'consonant_onset',text,text,values,t)
            self.transcribe(row);outputs[name]=values;receipt.append(row)
        a,b=outputs['aligned'],outputs['context32'];assert len(a)==len(b)
        write(self.root,'alignment.json',dict(comparisons=receipt,same_generated_mel=True,
            first100ms_aligned_vs_context_rms=float(np.sqrt(np.mean((a[:2400]-b[:2400])**2))),
            after400ms_max_abs_difference=float(np.max(abs(a[9600:]-b[9600:]))),
            legacy_extra_samples=len(outputs['legacy'])-len(b),
            tail_policy='no generated tail is removed; only exact known conditioning prefix'))
        del generated

    def blocks(self,case,ref,nfe,tts_text):
        import numpy as np
        import soundfile as sf
        from benchmark import pcm_metrics
        pieces=linguistic_chunks(tts_text,max_chars=180,max_words=32)
        assert ''.join(pieces)==tts_text
        started=time.perf_counter();chunks=[];arrivals=[];blockrows=[]
        for i,piece in enumerate(pieces):
            row=self.synth(f"selected-{case['id']}-block{i}",'blocks','sentence blocks',case['id'],piece,ref,nfe)
            values=sf.read(row['path'],dtype='float32')[0];chunks.append(values);blockrows.append(row)
            arrivals.append(dict(ms=(time.perf_counter()-started)*1000,bytes=len(values)*2))
        total=time.perf_counter()-started
        values=np.concatenate(chunks)
        row=self.record('selected-'+case['id'],'controlled','selected reference + TTS lexicon + aligned context + blocks',case['id'],case['text'],tts_text,values,
            dict(total_seconds=total,reference_id=ref['id'],nfe=nfe,seed=SEED,blocks=len(pieces),
                latency_scope='instrumented local sentence FIFO: synthesis plus persistence/readback/metrics; excludes HTTP and model/reference load'),
            block_ids=[r['id'] for r in blockrows],boundaries=[dict(sample_jump=float(abs(b[0]-a[-1])),previous100ms_rms=float(np.sqrt(np.mean(a[-2400:]**2))),next100ms_rms=float(np.sqrt(np.mean(b[:2400]**2)))) for a,b in zip(chunks,chunks[1:])],arrivals=arrivals)
        if row['peak']<=1:
            row.update(pcm_metrics(np.rint(values*32767).astype('<i2').tobytes(),arrivals,ev.RATE));write(self.root,'results.json',self.rows)
        self.transcribe(row)
        return row

    def controlled(self,ref,nfe):
        from training.f5_refine_text import normalize_text
        for case in controlled_cases():
            for steps in (16,8):
                row=self.synth(f"original-{case['id']}-{steps}",'controlled',f'original trained inference NFE{steps}',case['id'],case['text'],self.control,steps,legacy=True)
                self.transcribe(row)
            normalized=normalize_text(case['text'])
            self.blocks(case,ref,nfe,normalized['tts_text'])
        # A second seed probes sensitivity beyond one fortunate sample.
        row=self.synth('selected-repeat','stability','second seed','ordinary',SHORT,ref,nfe,seed=SEED+1)
        self.transcribe(row)

    def duration_diagnostic(self):
        case=next(c for c in controlled_cases() if c['id']=='long')
        frames=self.duration(self.control,case['text'])-int(self.conditioning(self.control).shape[1])
        control=self.synth('long-budget100','duration','100 percent budget, corrected context decoder',
            'long',case['text'],self.control,16,generated_frames=frames)
        self.transcribe(control)
        row=self.synth('long-budget90','duration','90 percent neural duration budget, complete text',
            'long',case['text'],self.control,16,generated_frames=round(frames*.9))
        self.transcribe(row)
        write(self.root,'duration-diagnostic.json',dict(id=row['id'],original_generated_frames=frames,
            tested_generated_frames=round(frames*.9),complete_text=True,
            interpretation='diagnostic only; no audio crop and not selected as default; ASR and owner must check missing words'))

    def run(self):
        ref,nfe=self.grid();self.pronunciations(ref,nfe);self.alignment();self.controlled(ref,nfe);self.duration_diagnostic()
        for row in json.loads((self.root/'qwen/results.json').read_text()):self.transcribe(row)
        write(self.root,'gpu-complete.json',dict(checkpoint_sha256_after=ev.digest(CHECKPOINT),
            no_weight_updates=True,selected_reference=ref['id'],selected_nfe=nfe,
            synthesis_files=len(self.rows),asr_files=len(self.asr_rows),owner_approval=False,
            additional_training_started=False,all_gpu_experiments_complete=True))
        assert ev.digest(CHECKPOINT)==BEST_SHA
        del self.model,self.v;self.cond.clear();gc.collect();self.torch.cuda.empty_cache();self.torch.cuda.synchronize()


def main():
    os.umask(0o077)
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','baseline','gpu']);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();root=args.root.resolve(strict=True)
    if any((p/'.git').exists() for p in (root,*root.parents)):raise ValueError('private results must be outside Git')
    if args.action=='preflight':preflight(root)
    elif args.action=='baseline':baseline(root)
    else:
        auth=json.loads((root/'gpu-window-authorized.json').read_text())
        assert auth['owner_authorized'] and auth['production_stopped'] and auth['status']=='active'
        assert json.loads((root/'preflight.json').read_text())['checkpoint_sha256']==BEST_SHA
        if (root/'results.json').exists():raise FileExistsError('preserve earlier synthesis evidence; choose another run root')
        assert len(json.loads((root/'qwen/results.json').read_text()))==3
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            raise RuntimeError('GPU occupied before F5 load; refusing overlap')
        Experiment(root).run()


if __name__=='__main__':main()
