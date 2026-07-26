"""Focused, stronger hard-tune comparisons based on the round-2 favourites."""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import graillon_lab as lab
import graillon_round2 as round2


def classify_against_base(rec, pcm, base_pcm):
    if base_pcm is None:
        return 'baseline'
    if not pcm or (max(pcm) == 0 and min(pcm) == 0):
        return 'empty_output'
    diff=lab.rms_difference(pcm,base_pcm)
    rec['rms_difference_from_mask_baseline']=diff
    if diff == 0:
        return 'identical_to_mask_baseline'
    return 'near_identical_to_mask_baseline' if diff is not None and diff < .001 else 'materially_different'


def summary(records, path):
    lines=[f"# Graillon {path.parent.name.capitalize()} — formantes R253/R255",'',
           '| ID | máscara | variante | wet/dry | smooth | inertia | p51 | Δ RMS base | pico | status |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for rec in records:
        s=rec.get('semantic_controls',{})
        lines.append('| {id} | {mask} | {variant} | {wet}/{dry} | {smooth} | {inertia} | {formant} | {diff} | {peak} | {status} |'.format(
            id=rec['id'], mask=rec.get('mask','—'), variant=rec.get('variant','—'),
            wet=s.get('wet_mix','—'), dry=s.get('dry_mix','—'), smooth=s.get('smooth','—'), inertia=s.get('inertia','—'),
            formant=s.get('formant_shift','—'), diff=rec.get('rms_difference_from_mask_baseline','—'),
            peak=rec.get('true_peak_db','—'), status=rec.get('classification','failed')))
    path.write_text('\n'.join(lines)+'\n',encoding='utf8')


def run(args):
    c=lab.config(args)
    inv,_=lab.inspect(c)
    data=lab.read_yaml(c['AEGIS_GRAILLON_PRESETS_CONFIG'])
    round_name=data.get('round')
    if round_name not in ('round3','round4'):
        raise lab.LabError('configuração não é a rodada 3/4')
    source=Path(c['AEGIS_GRAILLON_INPUT']); lab.wav_info(source,c)
    out=Path(c['AEGIS_GRAILLON_OUTPUT_DIR']); out.mkdir(parents=True,exist_ok=True)
    root=Path(c['AEGIS_GRAILLON_REPORT_DIR'])/round_name
    run_dir=root/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); run_dir.mkdir(parents=True)
    calibration=json.loads((lab.LAB/'graillon-formant-shift.json').read_text(encoding='utf8'))
    presets=[round2.materialize(p,data,'G3') for p in data['presets'] if p.get('enabled',True)]
    records=[]; baselines={}
    for pre in presets:
        try:
            pdir=run_dir/pre['id']; pdir.mkdir()
            rec,pcm=round2.plugin_record(c,pre,inv,out,source,pdir)
            rec['mask']=pre['mask']; rec['variant']=pre['variant']
            base=baselines.get(pre['mask'])
            rec['classification']=classify_against_base(rec,pcm,base)
            if pre['variant']=='baseline': baselines[pre['mask']]=pcm
            records.append(rec)
        except Exception as exc:
            records.append({'id':pre['id'],'family':pre['family'],'mask':pre['mask'],'variant':pre['variant'],
                            'success':False,'render_success':False,'error':str(exc),'classification':'failed'})
    manifest={'round':round_name,'source':str(source),'executed_at':datetime.now(timezone.utc).isoformat(),
              'base_engine':'G3','formant_shift_calibration':calibration,
              'sample_rate':24000,'channels':1,'sample_format':'s16le','presets':records}
    shutil.copy2(source,out/'original.wav')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    fields=['id','mask','variant','duration_seconds','integrated_lufs','true_peak_db','rms_db','rms_difference_from_mask_baseline','classification','render_success','error']
    with (out/'manifest.csv').open('w',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fields); writer.writeheader(); writer.writerows([{k:x.get(k) for k in fields} for x in records])
    (out/'controls-effective.json').write_text(json.dumps({x['id']:x.get('lv2_controls',{}) for x in records},indent=2),encoding='utf8')
    (out/'listening.html').write_text(round2.listening(records).replace('rodada 2',f'rodada {round_name[-1]}'),encoding='utf8')
    (out/'listening-blind.html').write_text(round2.listening(records,True).replace('rodada 2',f'rodada {round_name[-1]}'),encoding='utf8')
    (out/'blind-map.json').write_text(json.dumps({f'Amostra {n:02d}':x['id'] for n,x in enumerate(records,1)},ensure_ascii=False,indent=2),encoding='utf8')
    root.mkdir(parents=True,exist_ok=True); summary(records,root/'summary.md')
    (run_dir/'controls-effective.json').write_text((out/'controls-effective.json').read_text(),encoding='utf8')
    print(json.dumps({'rendered':sum(x.get('success',False) for x in records),'failed':sum(not x.get('success',False) for x in records),'output':str(out),'report':str(root)},ensure_ascii=False))
    return 0 if records else 2
