#!/usr/bin/env python3
"""CPU descriptors of private F5 samples; no generation, model load or effects.

Texture descriptors reuse training.reconstruct.texture_metrics. YIN operates
on 24kHz analysis copies and uses an energy mask, not a voicing classifier.
These estimates cannot establish speaker identity, robotic-effect fidelity,
intelligibility, owner approval or a training stop decision.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

RATE=24000
FRAME=1024
HOP=256
METHOD=dict(version='aegis.f5.acoustics.v1',analysis_rate=RATE,frame_length=FRAME,hop_length=HOP,
    yin_fmin_hz=60.0,yin_fmax_hz=500.0,yin_trough_threshold=0.1,
    active_rms_absolute_dbfs=-55.0,active_rms_relative_to_peak_db=-30.0,
    resample='scipy resample_poly Kaiser beta10, analysis only; no exported waveform changes')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.chmod(0o600)
    temp.replace(path)


def active_f0(values,rate=RATE):
    """Energy-gated YIN pitch candidates, including potentially unvoiced frames."""
    import librosa
    import numpy as np
    if rate!=RATE:raise ValueError('F0 analysis must use the common24k rate')
    y=np.asarray(values,dtype=np.float64)
    if y.ndim!=1 or len(y)<FRAME or not np.isfinite(y).all():raise ValueError('finite mono audio of at least one analysis frame required')
    f0=librosa.yin(y,fmin=METHOD['yin_fmin_hz'],fmax=METHOD['yin_fmax_hz'],sr=rate,
        frame_length=FRAME,hop_length=HOP,center=True,trough_threshold=METHOD['yin_trough_threshold'])
    rms=librosa.feature.rms(y=y,frame_length=FRAME,hop_length=HOP,center=True)[0]
    if rms.shape!=f0.shape:raise ValueError('YIN/RMS analysis grids differ')
    threshold=max(10**(METHOD['active_rms_absolute_dbfs']/20),float(rms.max(initial=0))*10**(METHOD['active_rms_relative_to_peak_db']/20))
    active=(rms>threshold)&np.isfinite(f0)
    selected=f0[active]
    metrics=dict(method='YIN; energy mask only, not a voiced/unvoiced classifier',
        median_hz=float(np.median(selected)) if selected.size else None,
        p10_hz=float(np.percentile(selected,10)) if selected.size else None,
        p90_hz=float(np.percentile(selected,90)) if selected.size else None,
        active_frames=int(active.sum()),total_frames=len(f0),active_rms_threshold=threshold,
        active_frame_fraction=float(active.mean()))
    return metrics,np.arange(len(f0))*HOP/rate,np.where(active,f0,np.nan)


def describe(path,cache,expected_sha256=None):
    """Read-only, hash-bound WAV analysis; cache depends on waveform and method."""
    import numpy as np
    import soundfile as sf
    from training.prepare_dataset import resample_pcm
    from training.reconstruct import magnitude,texture_metrics
    path=Path(path).resolve(strict=True)
    actual=digest(path)
    if expected_sha256 and actual!=expected_sha256:raise ValueError(f'WAV hash mismatch: {path}')
    method_key=hashlib.sha256(json.dumps(METHOD,sort_keys=True).encode()).hexdigest()[:12]
    prefix=cache/(actual+'-'+method_key)
    metadata=prefix.with_suffix('.json');arrays=prefix.with_suffix('.npz')
    if metadata.is_file() and arrays.is_file():
        result=json.loads(metadata.read_text())
        if result.get('audio_sha256')!=actual or result.get('method')!=METHOD:raise ValueError('descriptor cache mismatch')
        if digest(arrays)!=result.get('analysis_arrays_sha256'):raise ValueError('descriptor analysis-array cache changed')
        result['audio_path']=str(path)
        return result
    y,rate=sf.read(path,dtype='float64',always_2d=True)
    if y.shape[1]!=1 or not y.size or not np.isfinite(y).all():raise ValueError('finite nonempty mono WAV required')
    y=y[:,0]
    analysis=resample_pcm(y,rate,RATE)
    pitch,times,contour=active_f0(analysis)
    power=magnitude(analysis,RATE)**2
    mean_power=power.mean(axis=1)
    frequencies=np.fft.rfftfreq(FRAME,1/RATE)
    np.savez_compressed(arrays,times_seconds=times,yin_active_f0_hz=contour,
        frequencies_hz=frequencies,mean_spectral_power_db=10*np.log10(np.maximum(mean_power,1e-20)))
    arrays.chmod(0o600)
    result=dict(audio_path=str(path),audio_sha256=actual,audio_seconds=len(y)/rate,
        source_sample_rate=rate,source_subtype=sf.info(path).subtype,method=METHOD,
        native_texture=texture_metrics(y,rate),common24k_texture=texture_metrics(analysis,RATE),
        yin_active_f0=pitch,analysis_arrays=str(arrays),analysis_arrays_sha256=digest(arrays),
        samples_outside_unit_range=int(np.sum(np.abs(y)>1)))
    write_json(metadata,result)
    return result


def label_order(label):
    match=re.search(r'epoch-(\d+)',label)
    if label=='base':return (-1,label)
    return (int(match.group(1)),label) if match else (100000,label)


def discover(root,cases,nfe,repetition,reference_id):
    """Use completed sample receipts; never inspect temporary generated WAVs."""
    items=[];pending=[]
    train=[json.loads(s) for s in (root/'prepared/train.jsonl').read_text().splitlines() if s.strip()]
    reference=next((r for r in train if r['id']==reference_id),None)
    if reference is None:raise ValueError('conditioning reference must be an existing training record')
    for label,path,key in [('original-treated',reference['source_audio_filepath'],'source_audio_sha256'),
                           ('converted-24k',reference['audio_filepath'],'sha256')]:
        items.append(dict(role='conditioning_reference',group=reference_id,label=label,path=path,sha256=reference[key]))
    prompt_vocos=root/'roundtrip'/reference_id/'C-vocos-24000.wav'
    if prompt_vocos.is_file():
        items.append(dict(role='conditioning_reference',group=reference_id,label='vocos-reconstruction',path=str(prompt_vocos)))
    else:pending.append('conditioning prompt Vocos reconstruction unavailable; no new reconstruction performed')
    for receipt in sorted((root/'roundtrip').glob('*/results.json')):
        row=json.loads(receipt.read_text());identifier=row['id']
        for label,path,sha in [('original-treated',row['original_path'],row['original_sha256']),
                               ('converted-24k',row['converted']['path'],row['converted']['sha256']),
                               ('vocos-reconstruction',row['vocos']['path'],row['vocos']['sha256'])]:
            items.append(dict(role='acoustic_roundtrip',group=identifier,label=label,path=path,sha256=sha))
    for receipt in sorted((root/'evaluations').glob('*/*.json')):
        if receipt.name in ('reference.json','load.json','heldout-test.json','results.json'):continue
        row=json.loads(receipt.read_text())
        if not isinstance(row,dict) or row.get('case') not in cases or row.get('nfe')!=nfe or row.get('repetition')!=repetition:continue
        items.append(dict(role='synthesis',group=row['case'],case=row['case'],label=row['label'],nfe=nfe,
            repetition=repetition,seed=row.get('seed'),path=row['path'],sha256=row['sha256']))
    return items,pending


def deltas(anchor,candidate):
    result={k:candidate['common24k_texture'][k]-anchor['common24k_texture'][k] for k in
        ('rms_dbfs','spectral_centroid_hz','spectral_flatness_active_mean','power_above_4000hz_ratio','power_above_8000hz_ratio')}
    a=anchor['yin_active_f0']['median_hz'];b=candidate['yin_active_f0']['median_hz']
    result['yin_active_median_hz']=None if a is None or b is None else b-a
    return result


def comparisons(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['role'],r['group'],r.get('seed'))].append(r)
    results=[]
    for (role,group,seed),records in groups.items():
        if role=='synthesis':
            ordered=sorted(records,key=lambda r:label_order(r['label']))
            anchor=ordered[0]
        else:
            anchor=next((r for r in records if r['label']=='converted-24k'),records[0])
        for candidate in records:
            if candidate is anchor:continue
            results.append(dict(role=role,group=group,seed=seed,anchor=anchor['label'],candidate=candidate['label'],
                candidate_minus_anchor=deltas(anchor['descriptor'],candidate['descriptor']),
                scope='same text/seed for synthesis; descriptors only, no phoneme alignment or speaker-identity claim'))
    return results


def plot(rows,output,cases,reference_id):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.22})
    panels=[('conditioning_reference',reference_id),*[('synthesis',case) for case in cases]]
    fig,axes=plt.subplots(len(panels),1,figsize=(11,3.3*len(panels)),squeeze=False,constrained_layout=True)
    for ax,(role,group) in zip(axes[:,0],panels):
        selected=sorted([r for r in rows if r['role']==role and r['group']==group],key=lambda r:label_order(r['label']))
        for r in selected:
            with np.load(r['descriptor']['analysis_arrays'],allow_pickle=False) as data:
                ax.plot(data['times_seconds'],data['yin_active_f0_hz'],label=r['label'],linewidth=.8,alpha=.8)
        ax.set(title=group,xlabel='Tempo de cada amostra (s); sem alinhamento de fonemas',ylabel='Candidatos YIN ativos (Hz)',ylim=(METHOD['yin_fmin_hz']-5,METHOD['yin_fmax_hz']+5))
        if selected:ax.legend(fontsize=8,ncol=3)
        else:ax.text(.5,.5,'Amostras pendentes',ha='center',transform=ax.transAxes)
    fig.suptitle('AegisVoice v2 — F0 estimada; máscara de energia não classifica vozeamento')
    fig.savefig(output/'f0.png',dpi=140);plt.close(fig)
    roundtrip=next((r['group'] for r in rows if r['role']=='acoustic_roundtrip'),None)
    panels=[('conditioning_reference',reference_id)]
    if roundtrip:panels.append(('acoustic_roundtrip',roundtrip))
    panels.extend(('synthesis',case) for case in cases)
    fig,axes=plt.subplots(len(panels),1,figsize=(11,3.0*len(panels)),squeeze=False,constrained_layout=True)
    for ax,(role,group) in zip(axes[:,0],panels):
        selected=sorted([r for r in rows if r['role']==role and r['group']==group],key=lambda r:label_order(r['label']))
        for r in selected:
            with np.load(r['descriptor']['analysis_arrays'],allow_pickle=False) as data:
                ax.plot(data['frequencies_hz'],data['mean_spectral_power_db'],label=r['label'],linewidth=.8,alpha=.8)
        ax.set(title=group,xlabel='Frequência (Hz); análise comum24k',ylabel='Potência média STFT (dB re1)',xlim=(0,RATE/2),ylim=(-120,0))
        if selected:ax.legend(fontsize=8,ncol=3)
        else:ax.text(.5,.5,'Amostras pendentes',ha='center',transform=ax.transAxes)
    fig.suptitle('AegisVoice v2 — espectro médio sem normalização de ganho')
    fig.savefig(output/'spectrum.png',dpi=140);plt.close(fig)


def run(args):
    os.umask(0o077)
    for key,value in [('CUDA_VISIBLE_DEVICES',''),('OMP_NUM_THREADS','2'),('MKL_NUM_THREADS','2'),('NUMBA_NUM_THREADS','2')]:os.environ[key]=value
    root=args.run_root.resolve(strict=True)
    output=(args.output or root/'acoustics').resolve()
    if root!=output and root not in output.parents:raise ValueError('output must be under private run root')
    if any((p/'.git').exists() for p in (output,*output.parents)):raise ValueError('private audio descriptors cannot be saved in Git')
    output.mkdir(mode=0o700,parents=True,exist_ok=True);output.chmod(0o700)
    cache=output/'cache';cache.mkdir(mode=0o700,exist_ok=True)
    cases=args.case or ['novel_short','aegis_command']
    items,pending=discover(root,cases,args.nfe,args.repetition,args.reference_id)
    rows=[]
    for item in items:
        descriptor=describe(item['path'],cache,item.get('sha256'))
        rows.append(dict(**item,descriptor=descriptor))
        print(json.dumps(dict(role=item['role'],group=item['group'],label=item['label'],
            f0_median_hz=descriptor['yin_active_f0']['median_hz'],rms_dbfs=descriptor['common24k_texture']['rms_dbfs'])),flush=True)
    plot(rows,output,cases,args.reference_id)
    result=dict(schema='aegis.f5.acoustics.v1',generated_utc=datetime.now(timezone.utc).isoformat(),
        method=METHOD,device='cpu',reference_id=args.reference_id,cases=cases,nfe=args.nfe,repetition=args.repetition,
        records=rows,comparisons=comparisons(rows),pending=pending,owner_perceptual_approval=False,
        interpretation=[
            'YIN gives pitch candidates on energy-active frames, including unvoiced sounds; synthetic voices/effects may cause octave and tracking errors.',
            'Native-rate texture metrics are retained; comparisons and plots use analysis copies at24k to control analysis-window and bandwidth differences.',
            'Different texts change spectral/F0 summaries. Conditioning prompt is context; direct longitudinal comparisons hold case, NFE, repetition and seed fixed.',
            'Spectral centroid, flatness, band power, RMS and F0 do not prove speaker identity, intelligibility, effect preservation or owner approval.',
            'Descriptors are auxiliary monitoring, never an independent training stop criterion. No gain, effects, trimming or new Vocos reconstruction applied.'])
    write_json(output/'results.json',result)
    write_json(output/'manifest-sha256.json',{p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='manifest-sha256.json'})
    for p in output.iterdir():
        if p.is_file():p.chmod(0o600)
    print(json.dumps(dict(output=str(output/'results.json'),records=len(rows),pending=pending)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,default=Path('/var/tmp/aegisvoice-v2-training-20261102'))
    p.add_argument('--output',type=Path)
    p.add_argument('--reference-id',default='aegis_ptbr_0586')
    p.add_argument('--case',action='append')
    p.add_argument('--nfe',type=int,default=16)
    p.add_argument('--repetition',type=int,default=0)
    run(p.parse_args())

if __name__=='__main__':main()
