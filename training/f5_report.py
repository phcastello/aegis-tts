#!/usr/bin/env python3
"""Aggregate private F5 evidence and export CPU-only loss/performance plots.

Never loads checkpoints, touches CUDA, controls services or assigns perceptual
approval. Missing results remain explicitly pending in report.json. The output
is private and offline; it contains paths and source artifact hashes.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def load(path,default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def jsonl(path):
    if not path.is_file():return []
    lines=path.read_text().splitlines()
    result=[]
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:result.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(lines)-1:break  # concurrent writer may have an unfinished last line
            raise
    return result


def finite(value):
    return isinstance(value,(int,float)) and math.isfinite(value)


def stats(values):
    values=[float(x) for x in values if finite(x)]
    if not values:return None
    return dict(count=len(values),minimum=min(values),median=statistics.median(values),maximum=max(values),mean=statistics.mean(values))


def duration(start,end):
    if not start or not end:return None
    return (datetime.fromisoformat(end.replace('Z','+00:00'))-datetime.fromisoformat(start.replace('Z','+00:00'))).total_seconds()


def operation_windows(root):
    """Aggregate actual guarded outages; earlier restoration cannot close a resume."""
    directories=[root/'operations',*sorted(root.glob('operations-resume-*'))]
    windows=[]
    for directory in directories:
        if not directory.is_dir():continue
        record=load(directory/'gpu-window.json',{})
        # A prepared snapshot with no GPU window has not interrupted production.
        if not record:continue
        windows.append(dict(directory=str(directory),window=record,
            baseline=load(directory/'baseline.json'),restored=load(directory/'restored.json'),
            downtime_seconds=duration(record.get('stop_started_utc'),record.get('restored_utc')),
            gpu_experiment_seconds=duration(record.get('stopped_utc'),record.get('gpu_work_ended_utc'))))
    windows.sort(key=lambda item:(item['window'].get('stop_started_utc') or item['window'].get('started_utc') or '',item['directory']))
    latest=windows[-1] if windows else {}
    closed=[w for w in windows if w['downtime_seconds'] is not None]
    complete=bool(windows) and len(closed)==len(windows)
    verified=complete and all(w['window'].get('status')=='production_restored' and w['restored'] is not None for w in windows)
    completed_downtime=sum(w['downtime_seconds'] for w in closed)
    gpu_times=[w['gpu_experiment_seconds'] for w in windows if w['gpu_experiment_seconds'] is not None]
    intermediate=[dict(directory=w['directory'],restored_utc=w['window'].get('restored_utc'),
        downtime_seconds=w['downtime_seconds'],preceding_experiment_error=w['window'].get('error'),
        description='Intermediate production restoration before a later guarded GPU window; not the final production state')
        for w in windows[:-1] if w['restored'] is not None]
    return dict(window=latest.get('window',{}),windows=windows,baseline=load(root/'operations/baseline.json'),
        restored=latest.get('restored'),production_restoration_verified=verified,
        downtime_seconds=completed_downtime if complete else None,
        downtime_completed_windows_seconds=completed_downtime,
        downtime_in_progress=bool(windows) and not complete,
        gpu_experiment_seconds=sum(gpu_times) if windows and len(gpu_times)==len(windows) else None,
        gpu_completed_windows_seconds=sum(gpu_times),intermediate_restorations=intermediate)


def private_output(root,output):
    if output.is_symlink():raise ValueError('report output may not be a symlink')
    output=output.resolve()
    if root!=output and root not in output.parents:raise ValueError('output must be under private run root')
    if any((p/'.git').exists() for p in (output,*output.parents)):raise ValueError('private report cannot be written in Git')
    output.mkdir(parents=True,mode=0o700,exist_ok=True)
    output.chmod(0o700)
    return output


def collect(root):
    events=jsonl(root/'events.jsonl')
    initial_validation=next((e['validation'] for e in events if e.get('event')=='initial_validation' and isinstance(e.get('validation'),dict)),None)
    updates=[e for e in events if e.get('event')=='update' and e.get('epoch',0)>0]
    history=load(root/'loss-history.json')
    if history is None:
        history=[dict(epoch=e['epoch'],update=e['update'],train_loss=e.get('train_loss'),validation_loss=e['validation']['loss'],standard_error=e['validation'].get('standard_error'),phase=e.get('phase')) for e in events if e.get('event')=='epoch']
    inference=[]
    for p in sorted((root/'evaluations').glob('*/results.json')):
        rows=load(p,[])
        if isinstance(rows,list):inference.extend(rows)
    grouped=defaultdict(list)
    for r in inference:grouped[(r.get('label'),r.get('nfe'))].append(r)
    aggregate=[]
    for (label,nfe),rows in sorted(grouped.items(),key=lambda x:str(x[0])):
        entry=dict(label=label,nfe=nfe,count=len(rows),cases=sorted(set(r.get('case','') for r in rows)))
        for key in ('ttfa_ms','first_pcm_ms','first_useful_pcm_available_ms','onset_audio_offset_ms',
                    'estimated_first_audible_playback_ms','total_seconds','rtf','resident_allocated_mib',
                    'peak_allocated_mib','peak_reserved_mib'):
            entry[key]=stats([r.get(key) for r in rows])
        aggregate.append(entry)
    checkpoints=load(root/'evaluations/checkpoint-manifest.json',[])
    operation=operation_windows(root)
    training=load(root/'training-result.json')
    package_manifests=list((root/'listening').glob('manifest-sha256.json'))
    best=load(root/'best.json')
    repository=Path(__file__).resolve().parents[1]
    code_paths=[*sorted((repository/'training').glob('f5_*.py')),
        repository/'training/av2_ops.py',repository/'training/reconstruct.py',
        repository/'training/prepare_dataset.py',repository/'benchmarks/architecture_compare/benchmark.py',
        repository/'benchmarks/architecture_compare/f5_probe.py']
    pending=[]
    for name,present in [('training_result',training is not None),('final_cuda_evaluations',bool(checkpoints)),
                         ('all_gpu_phases_closed',(root/'gpu-all-phases-complete.json').is_file()),
                         ('production_restoration_verified',operation['production_restoration_verified']),
                         ('listening_package',bool(package_manifests) and (root/'listening/index.html').is_file()),('owner_perceptual_review',False)]:
        if not present:pending.append(name)
    result=dict(schema='aegisvoice.v2.training.report.v1',generated_utc=datetime.now(timezone.utc).isoformat(),
        private_root=str(root),dataset=load(root/'prepared/audit.json'),cpu_preflight=load(root/'cpu-check.json'),
        training=training,best_checkpoint=best,initial_validation=initial_validation,loss_history=history,
        pipeline_stages=jsonl(root/'pipeline.jsonl'),
        training_update_statistics={k:stats([r.get(k) for r in updates]) for k in ('seconds','loss','gradient_norm','audio_seconds_per_second','peak_allocated_mib','peak_reserved_mib')},
        optimizer_updates_logged=len(updates),benchmark=[load(p) for p in sorted(root.glob('benchmark-*.json'))],
        inference=aggregate,checkpoint_manifest=checkpoints,
        heldout_test={p.parent.name:load(p) for p in sorted((root/'evaluations').glob('*/heldout-test.json'))},
        block_streaming=load(root/'evaluations/block-streaming.json'),asr=load(root/'asr/results.json'),
        asr_settings=load(root/'asr/settings.json'),
        asr_checkpoint_receipts={p.name:load(p) for p in sorted((root/'asr').glob('pilot-epoch-*.json'))},
        asr_checkpoint_receipts_scope='unpooled diagnostic snapshots with their own settings/scoring; canonical asr records use asr_settings and the plain scorer without gateway normalization',
        code_provenance_scope='repository script contents at report generation; per-ASR audio/settings hashes remain in ASR records',
        code_provenance=[dict(path=str(p.relative_to(repository)),sha256=sha256(p)) for p in code_paths if p.is_file()],
        acoustics=load(root/'acoustics/results.json'),
        qwen_standard_proxy=load(root/'report/qwen-standard-proxy.json'),
        browser_playback=load(root/'listening/browser-playback-receipt.json'),
        tempo_comparison_plan=load(root/'tempo-comparison-plan.json'),
        operation=operation,
        listening_manifests=[dict(path=str(p),sha256=sha256(p)) for p in package_manifests],
        package_archives=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(root.glob('*.zip'))],
        owner_perceptual_approval=False,pending=pending,
        interpretation={'loss':'fixed held-out validation estimates CFM objective, not perceptual identity',
            'asr':'auxiliary pronunciation/intelligibility indicator; not human approval',
            'latency':'F5 ttfa_ms/first_pcm_ms measure completed native waveform availability; estimated_first_audible_playback_ms adds digital activity offset assuming immediate FIFO playback, excluding client/device delay',
            'qwen_standard_proxy':'offline shared windowed proxy over preserved Qwen PCM and actual arrival schedules; Qwen includes HTTP/DSP service context while F5 native generation does not',
            'qwen':'operational single-phrase HTTP baseline: first received block with any abs(PCM16)>32, not playback onset; F5 onset proxy instead requires two10ms frames with peak>100 and RMS>30; use matched measurement methods before direct comparison'})
    return result,updates,events


def plots(output,result,updates,events):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.22,'figure.dpi':140})
    history=result['loss_history']
    fig,axes=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
    if updates:
        axes[0].plot([r['update'] for r in updates],[r['loss'] for r in updates],alpha=.3,linewidth=.6,label='CFM por atualização')
        window=min(25,len(updates))
        means=[statistics.mean(r['loss'] for r in updates[max(0,i-window+1):i+1]) for i in range(len(updates))]
        axes[0].plot([r['update'] for r in updates],means,linewidth=1.2,label=f'Média móvel ({window})')
        axes[0].legend()
    else:axes[0].text(.5,.5,'Treinamento ainda sem atualizações registradas',ha='center',transform=axes[0].transAxes)
    axes[0].set(xlabel='Atualização do otimizador',ylabel='Loss de treino CFM')
    validation_points=list(history)
    initial=result.get('initial_validation') or {}
    if finite(initial.get('loss')):
        validation_points.insert(0,dict(epoch=0,validation_loss=initial['loss']))
    if validation_points:
        axes[1].plot([r['epoch'] for r in validation_points],[r['validation_loss'] for r in validation_points],marker='o',markersize=3,label='Validação fixa (época 0 = base)')
        valid=[r for r in history if finite(r.get('train_loss'))]
        if valid:axes[1].plot([r['epoch'] for r in valid],[r['train_loss'] for r in valid],marker='.',label='Média de treino')
        best=result.get('best_checkpoint') or {}
        if best.get('epoch') is not None:axes[1].axvline(best['epoch'],linestyle='--',alpha=.5,label='Melhor checkpoint')
        axes[1].legend()
    else:axes[1].text(.5,.5,'Validação por época pendente',ha='center',transform=axes[1].transAxes)
    axes[1].set(xlabel='Época',ylabel='Loss CFM')
    fig.suptitle('AegisVoice v2 — treino e validação (objetivo CFM)')
    fig.savefig(output/'loss.png');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,6),constrained_layout=True)
    for ax,key,label in [(axes[0],'seconds','Segundos por atualização'),(axes[1],'peak_allocated_mib','Pico de VRAM alocada (MiB)')]:
        if updates:ax.plot([r['update'] for r in updates],[r.get(key,float('nan')) for r in updates],linewidth=.8)
        else:ax.text(.5,.5,'Medições pendentes',ha='center',transform=ax.transAxes)
        ax.set(xlabel='Atualização do otimizador',ylabel=label)
    fig.suptitle('AegisVoice v2 — desempenho real do treinamento')
    fig.savefig(output/'training-performance.png');plt.close(fig)


def generate(root,output,hash_checkpoints=True):
    os.umask(0o077)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    root=root.resolve(strict=True)
    output=private_output(root,output)
    result,updates,events=collect(root)
    source_paths=[root/'events.jsonl',root/'pipeline.jsonl',root/'loss-history.json',root/'training-result.json',root/'best.json',
                  root/'prepared/audit.json',root/'cpu-check.json',root/'operations/baseline.json',root/'operations/restored.json',
                  root/'operations/gpu-window.json',root/'evaluations/checkpoint-manifest.json',root/'evaluations/block-streaming.json',root/'asr/results.json']
    source_paths.extend([root/'acoustics/results.json',root/'asr/settings.json'])
    source_paths.extend((root/'asr').glob('pilot-epoch-*.json'))
    source_paths.extend([root/'report/qwen-standard-proxy.json',root/'listening/browser-playback-receipt.json',
                         root/'recompute_qwen_pcm.py',root/'tempo-comparison-plan.json'])
    source_paths.extend(root.glob('benchmark-*.json'))
    for directory in [root/'operations',*sorted(root.glob('operations-resume-*'))]:
        source_paths.extend(directory/name for name in ('baseline.json','restored.json','gpu-window.json'))
    source_paths.extend((root/'evaluations').glob('*/results.json'))
    source_paths.extend((root/'evaluations').glob('*/heldout-test.json'))
    if hash_checkpoints:source_paths.extend(p for p in (root/'checkpoints').glob('*') if p.suffix in ('.pt','.safetensors') and p.is_file())
    result['source_artifacts']=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(set(source_paths)) if p.is_file()]
    plots(output,result,updates,events)
    target=output/'report.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    manifest={p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='manifest-sha256.json'}
    (output/'manifest-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for p in output.iterdir():
        if p.is_file():p.chmod(0o600)
    print(json.dumps({'report':str(target),'loss_plot':str(output/'loss.png'),'pending':result['pending']},ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,default=Path('/var/tmp/aegisvoice-v2-training-20261102'))
    p.add_argument('--output',type=Path)
    p.add_argument('--skip-checkpoint-hashes',action='store_true',help='for intermediate monitoring only; final report should hash saved checkpoints')
    args=p.parse_args()
    generate(args.run_root,args.output or args.run_root/'report',not args.skip_checkpoint_hashes)

if __name__=='__main__':main()
