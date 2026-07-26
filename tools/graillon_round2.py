"""Round 2 controlled Graillon renders; imported by :mod:`graillon_lab`."""
from __future__ import annotations
import csv, hashlib, html, json, shutil
from datetime import datetime, timezone
from pathlib import Path

import graillon_lab as lab

NOTE_NAMES={'c':'C','c_sharp':'C#','d':'D','d_sharp':'D#','e':'E','f':'F','f_sharp':'F#','g':'G','g_sharp':'G#','a':'A','a_sharp':'A#','b':'B'}
ENGINES={'G2':1.0,'G3':0.5,'I2':0.0}

def engine_inventory(c):
    """Persist only mappings supported by the installed bundle.

    Graillon 2 Init is a named official preset at 1.0; the binary's ordered
    labels identify the two remaining engine positions.  R1 then confirms
    their observable effect; this is not a claim about a physical scale.
    """
    binary=next(Path(c['AEGIS_GRAILLON_BUNDLE']).glob('*.so'))
    strings=lab.run(['strings',str(binary)]).stdout
    required=('Inner2','Graillon3','Graillon2')
    if not all(x in strings for x in required):
        raise lab.LabError('nomes de pitch engine não encontrados no binário instalado')
    payload={'symbol':'p52','engines':ENGINES,'source':'official preset Graillon 2 Init (1.0) + labels Inner2/Graillon3/Graillon2 in installed plugin binary','verified':True,
             'notes':'R210-R212 are the controlled numeric verification; values are normalized plugin positions.'}
    (lab.LAB/'graillon-pitch-engines.json').write_text(json.dumps(payload,indent=2),encoding='utf8')
    return payload

def _replace_base_engine(value, base):
    return ENGINES[base] if value == '__base_engine__' else value

def materialize(pre, data, base_engine):
    pre=json.loads(json.dumps(pre))
    pre['controls']={k:_replace_base_engine(v,base_engine) for k,v in (pre.get('controls') or {}).items()}
    pre['round2']=True
    if pre['id'] in ('R210','R211','R212'):
        pre['pitch_engine_name']=pre['id'].replace('R21','G') if pre['id']!='R212' else 'I2'
        # IDs map explicitly, avoiding names inferred from numeric port index.
        pre['pitch_engine_name']={'R210':'G2','R211':'G3','R212':'I2'}[pre['id']]
    else:
        pre['pitch_engine_name']=base_engine if 'pitch_engine' in pre['controls'] else None
    allowed=[n for n in lab.NOTES if pre['controls'].get('allow_'+n)==1.0]
    pre['allowed_notes']=allowed
    return pre

def normalize(c, src, dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    p=lab.run([c['AEGIS_GRAILLON_FFMPEG_BIN'],'-y','-v','error','-i',str(src),'-af','loudnorm=I=-16:TP=-1:LRA=11','-ar',c['AEGIS_GRAILLON_SAMPLE_RATE'],'-ac','1','-c:a','pcm_s16le',str(dst)])
    if p.returncode: raise lab.LabError('normalização falhou: '+p.stderr.strip())

def original_record(c, pre, out, source):
    pcm=lab.pcm_from_wav(source,c); rate=int(c['AEGIS_GRAILLON_SAMPLE_RATE'])
    raw=out/'raw'/f"preset-{pre['id']}.wav"; lab.write_wav(raw,pcm,rate)
    comp=out/'compensated'/raw.name; comp.parent.mkdir(exist_ok=True); shutil.copy2(raw,comp)
    norm=out/'normalized'/raw.name; normalize(c,raw,norm)
    rec={'id':pre['id'],'name':pre['name'],'family':pre['family'],'description':pre['description'],'base_state':'source','semantic_controls':{},'lv2_controls':{},'pitch_engine_name':None,'pitch_engine_value':None,'allowed_notes':[], 'render_success':True,'success':True,'output_raw':'raw/'+raw.name,'output_normalized':'normalized/'+norm.name,'raw_file':raw.name,'normalized_file':norm.name,'sha256':lab.sha_file(raw),'output_sha256':lab.sha_file(raw),'duration_seconds':len(pcm)/2/rate,'audio_duration_seconds':len(pcm)/2/rate,'latency_frames':0,'latency_ms':0,'xrun_count':0,'rms_difference_from_bypass':None}
    rec.update(lab.audio_metrics(raw,c)); return rec,pcm

def plugin_record(c, pre, inv, out, source, run_dir):
    rec,pcm=lab.render_one(c,pre,inv,out,source,run_dir)
    oldraw=out/'raw'/rec['raw_file']; raw=out/'raw'/f"preset-{pre['id']}.wav"; raw.parent.mkdir(exist_ok=True)
    shutil.move(oldraw,raw)
    # render_one's historical layout also creates this compensated sibling in
    # raw/.  Round 2 owns a separate compensated/ directory, so remove only
    # that known, just-created duplicate.
    old_compensated=out/'raw'/rec['compensated_file']
    if old_compensated.exists(): old_compensated.unlink()
    oldnorm=out/'normalized'/rec['normalized_file']
    if oldnorm.exists(): oldnorm.unlink()
    comp=out/'compensated'/raw.name;comp.parent.mkdir(exist_ok=True);shutil.copy2(raw,comp)
    norm=out/'normalized'/raw.name;normalize(c,raw,norm)
    rec.update({'family':pre['family'],'render_success':True,'output_raw':'raw/'+raw.name,'output_normalized':'normalized/'+norm.name,'raw_file':raw.name,'normalized_file':norm.name,'sha256':lab.sha_file(raw),'duration_seconds':rec['audio_duration_seconds'],'latency_frames':rec.pop('plugin_latency_frames',None),'latency_ms':rec.pop('plugin_latency_ms',None),'pitch_engine_name':pre['pitch_engine_name'],'pitch_engine_value':pre['controls'].get('pitch_engine'),'allowed_notes':pre['allowed_notes']})
    return rec,pcm

def classify(rec, pcm, bypass_pcm):
    if not rec.get('success') or not pcm: return 'failed'
    if max(pcm,default=0)==0 and min(pcm,default=0)==0: return 'empty_output'
    if bypass_pcm is None: return 'not_compared'
    diff=lab.rms_difference(pcm,bypass_pcm);rec['rms_difference_from_bypass']=diff
    if diff == 0: return 'identical_to_bypass'
    return 'near_identical_to_bypass' if diff is not None and diff<.001 else 'materially_different'

def control_effect(c, out, records, left, right):
    """Classify a *single-control* A/B pair without using the bypass.

    Formant availability is a property of the corresponding port, not merely
    of either render being different from dry audio.
    """
    byid={x['id']:x for x in records}
    a,b=byid.get(left),byid.get(right)
    if not a or not b or not a.get('success') or not b.get('success'):
        return 'unknown'
    pa=out/'raw'/a['raw_file']; pb=out/'raw'/b['raw_file']
    if not pa.exists() or not pb.exists(): return 'unknown'
    aa=lab.pcm_from_wav(pa,c); bb=lab.pcm_from_wav(pb,c)
    if aa==bb: return 'no_effect'
    return 'no_effect' if (lab.rms_difference(aa,bb) or 0)<.001 else 'available'

def ratings(key):
    labels=('semelhança com GLaDOS','força do hard-tune','inteligibilidade','identidade da Giseli','naturalidade mínima','artefatos interessantes','artefatos ruins','fadiga auditiva')
    return ''.join(f'<label>{html.escape(label)} <input type=number min=1 max=5 name="{key}.{i}" onchange="save(this)"></label>' for i,label in enumerate(labels))+f'<label>observações <textarea name="{key}.notes" onchange="save(this)"></textarea></label>'

def listening(records, blind=False):
    cards=[]
    good=[x for x in records if x.get('success')]
    for n,p in enumerate(good,1):
        title=f'Amostra {n:02d}' if blind else f"{p['id']} — {p['description']}"
        details='' if blind else f"<p>Família: {html.escape(p['family'])}; engine: {html.escape(str(p.get('pitch_engine_name') or '—'))}; notas: {html.escape(', '.join(NOTE_NAMES[x] for x in p.get('allowed_notes',[])) or '—')}</p><pre>{html.escape(json.dumps(p.get('semantic_controls',{}),ensure_ascii=False,sort_keys=True))}</pre>"
        cards.append(f'<article><h2>{html.escape(title)}</h2>{details}<audio controls preload="none" src="{html.escape(p["output_normalized"])}"></audio>{ratings(p["id"] if not blind else f"sample-{n:02d}")}</article>')
    return '<!doctype html><meta charset=utf-8><title>Graillon round 2</title><style>body{font:15px system-ui;margin:2rem;max-width:980px}article{border-top:1px solid #ccc;padding:1rem 0}pre{white-space:pre-wrap}label{display:inline-block;margin:.25rem}textarea{display:block;width:95%;height:4em}</style><h1>Graillon — rodada 2'+(' cega' if blind else '')+'</h1><p>Original <audio controls src="original.wav"></audio></p><button onclick="exportRatings()">Exportar notas JSON</button>'+''.join(cards)+'<script>function save(x){localStorage.setItem("graillon-round2-"+x.name,x.value)}function exportRatings(){let x={};document.querySelectorAll("[name]").forEach(e=>x[e.name]=e.value);let a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(x,null,2)],{type:"application/json"}));a.download="ratings.json";a.click()}</script>'

def report_summary(records, engine, path):
    lines=['# Graillon Round 2','',f'Base engine: **{engine}**','', '| ID | família | engine | dry/wet | smooth | inertia | formant | notas | latência | clipping | RMS bypass | status |','|---|---|---|---|---|---|---|---|---|---|---|---|']
    for x in records:
        s=x.get('semantic_controls',{}); formant='off' if s.get('formant_enabled')==0 else ('on' if s.get('formant_enabled')==1 else 'base')
        notes=','.join(NOTE_NAMES[n] for n in x.get('allowed_notes',[])) or '—'; wet=f"{s.get('wet_mix','—')}/{s.get('dry_mix','—')}"
        clip=(x.get('true_peak_db') is not None and x['true_peak_db']>=-0.1)
        lines.append(f"| {x['id']} | {x.get('family','')} | {x.get('pitch_engine_name') or '—'} | {wet} | {s.get('smooth','—')} | {s.get('inertia','—')} | {formant} | {notes} | {x.get('latency_frames','—')} | {clip} | {x.get('rms_difference_from_bypass','—')} | {x.get('classification',x.get('error','failed'))} |")
    path.write_text('\n'.join(lines)+'\n',encoding='utf8')

def run(args):
    c=lab.config(args); inv,_=lab.inspect(c); data=lab.read_yaml(c['AEGIS_GRAILLON_PRESETS_CONFIG'])
    engines=engine_inventory(c); requested=getattr(args,'base_engine',None) or __import__('os').environ.get('AEGIS_GRAILLON_ROUND2_BASE_ENGINE')
    official=lab.base_controls(c,'official-hard-tune')['controls']['p52']; official_name=next((k for k,v in ENGINES.items() if v==official),'G3')
    base=requested or official_name
    if base not in engines['engines']: raise lab.LabError('base engine não confirmado: '+str(base))
    source=Path(c['AEGIS_GRAILLON_INPUT']);lab.wav_info(source,c);out=Path(c['AEGIS_GRAILLON_OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    root=Path(c['AEGIS_GRAILLON_REPORT_DIR'])/'round2';run_dir=root/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');run_dir.mkdir(parents=True)
    requested_preset=getattr(args,'preset',None)
    presets=[materialize(p,data,base) for p in data['presets'] if p.get('enabled',True) and (not requested_preset or p['id']==requested_preset)]
    if not presets: raise lab.LabError('preset round2 não encontrado')
    records=[];bypass_pcm=None
    for pre in presets:
        try:
            pdir=run_dir/pre['id'];pdir.mkdir()
            rec,pcm=original_record(c,pre,out,source) if pre['id']=='R200' else plugin_record(c,pre,inv,out,source,pdir)
            if pre['id']=='R201': bypass_pcm=pcm
            rec['classification']=classify(rec,pcm,bypass_pcm)
            records.append(rec)
        except Exception as exc: records.append({'id':pre['id'],'name':pre.get('name'),'family':pre.get('family'),'description':pre.get('description'),'success':False,'render_success':False,'error':str(exc),'classification':'failed'})
    # R240/R242 differ only in p63; R241/R242 differ only in p16.
    formant={'p16':control_effect(c,out,records,'R241','R242'),
             'p63':control_effect(c,out,records,'R240','R242')}
    manifest={'round':'round2','source':str(source),'executed_at':datetime.now(timezone.utc).isoformat(),'base_state':'round2-hard-tune-base','base_engine':base,'pitch_engines':engines,'formant_controls':{'p16':formant['p16'],'p51':'unknown (physical scale undocumented)','p63':formant['p63']},'sample_rate':24000,'channels':1,'sample_format':'s16le','presets':records}
    shutil.copy2(source,out/'original.wav')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    fields=['id','family','pitch_engine_name','pitch_engine_value','duration_seconds','integrated_lufs','true_peak_db','rms_db','rms_difference_from_bypass','latency_frames','latency_ms','xrun_count','classification','render_success','error']
    with (out/'manifest.csv').open('w',newline='',encoding='utf8') as f: w=csv.DictWriter(f,fields);w.writeheader();w.writerows([{k:x.get(k) for k in fields} for x in records])
    (out/'controls-effective.json').write_text(json.dumps({x['id']:x.get('lv2_controls',{}) for x in records},indent=2),encoding='utf8')
    (out/'listening.html').write_text(listening(records),encoding='utf8');(out/'listening-blind.html').write_text(listening(records,True),encoding='utf8')
    (out/'blind-map.json').write_text(json.dumps({f'Amostra {n:02d}':x['id'] for n,x in enumerate([x for x in records if x.get('success')],1)},ensure_ascii=False,indent=2),encoding='utf8')
    root.mkdir(parents=True,exist_ok=True);report_summary(records,base,root/'summary.md')
    (run_dir/'controls-effective.json').write_text((out/'controls-effective.json').read_text(),encoding='utf8')
    print(json.dumps({'rendered':sum(x.get('success',False) for x in records),'failed':sum(not x.get('success',False) for x in records),'output':str(out),'report':str(root),'base_engine':base},ensure_ascii=False))
    return 0 if records else 2
