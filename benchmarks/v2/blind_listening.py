#!/usr/bin/env python3
"""Build blinded local pairs by symlink, preserving original WAV bytes once."""
from __future__ import annotations
import argparse
import array
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import sys
import wave

CRITERIA=['Clareza e palavras completas','Pronúncia e acentuação PT-BR','Identidade em relação às âncoras','Timbre metálico/robótico','Estabilidade da voz','Sibilantes e consoantes','Prosódia e pausas','Artefatos ou junções','Preferência geral']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def level(path):
    with wave.open(str(path),'rb') as w:
        if w.getsampwidth()!=2 or w.getnchannels()!=1:raise ValueError('listening input must be final mono PCM16 WAV')
        rate=w.getframerate();pcm=w.readframes(w.getnframes())
    a=array.array('h');a.frombytes(pcm)
    if sys.byteorder!='little':a.byteswap()
    frame=round(rate*.02);active=[]
    for start in range(0,len(a)-frame+1,frame):
        rms=math.sqrt(sum(x*x for x in a[start:start+frame])/frame)/32768
        if rms>=10**(-45/20):active.append(rms*rms)
    rms=math.sqrt(sum(active)/len(active)) if active else 0
    return dict(sample_rate=rate,duration_seconds=len(a)/rate,active_rms=rms,
        active_rms_dbfs=20*math.log10(rms) if rms else None,peak=max((abs(x) for x in a),default=0)/32768)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pairs',required=True);p.add_argument('--output',required=True);p.add_argument('--key',required=True);p.add_argument('--seed',type=int,default=40733133);a=p.parse_args()
    os.umask(0o077);out=Path(a.output).resolve();key=Path(a.key).resolve()
    if key==out or out in key.parents:raise ValueError('answer key must stay outside listening directory')
    if key.exists():raise ValueError('answer key exists')
    data=json.loads(Path(a.pairs).read_text());out.mkdir(parents=True,exist_ok=False,mode=0o700);(out/'clips').mkdir(mode=0o700)
    rng=random.Random(a.seed);mapping=[];reviews=[];articles=[];linked={}
    def link(path):
        path=Path(path).resolve(strict=True)
        if path not in linked:
            name='clips/clip-'+str(len(linked)+1).zfill(3)+'.wav';os.symlink(path,out/name);linked[path]=name
        return linked[path]
    for index,pair in enumerate(data['pairs'],1):
        if not pair.get('same_text_confirmed'):raise ValueError('pair must declare same_text_confirmed after checking generation manifests')
        sides=[pair['a'],pair['b']];rng.shuffle(sides);measurements=[level(s['path']) for s in sides]
        target=min((m['active_rms'] for m in measurements if m['active_rms']>0),default=0)
        clips=[];labels={}
        for label,side,m in zip(('A','B'),sides,measurements):
            src=link(side['path']);gain=min(1,target/m['active_rms']) if m['active_rms']>0 else 1
            clips.append(f'<div><b>{label}</b><audio controls preload="none" data-gain="{gain:.9f}" src="{html.escape(src)}"></audio></div>')
            labels[label]=dict(**side,sha256=sha(side['path']),level=m,listening_attenuation=gain)
        anchors=[]
        for n,anchor in enumerate(pair.get('anchors',[]),1):
            anchors.append(f'<div>Âncora real {n}<audio controls preload="none" src="{html.escape(link(anchor["path"]))}"></audio></div>')
        pid='pair-'+str(index).zfill(2)
        fields=''.join(f'<label>{html.escape(c)} <select name="{pid}:{i}"><option value="">não avaliado</option><option>A</option><option>B</option><option>empate</option><option>inconclusivo</option></select></label>' for i,c in enumerate(CRITERIA))
        articles.append(f'<article><h2>Par {index}</h2><p>{html.escape(pair["text"])}</p><div class="pair">'+''.join(clips)+'</div>'+''.join(anchors)+f'<div class="notes">{fields}</div><textarea name="{pid}:notes" placeholder="Palavras, posição do problema, atributo da identidade ou artefato observado"></textarea></article>')
        mapping.append(dict(pair=pid,case=pair['case'],text=pair['text'],labels=labels,anchors=pair.get('anchors',[])))
        reviews.append(dict(pair=pid,case=pair['case'],criteria={c:None for c in CRITERIA},notes='',same_text_confirmed=True))
    content='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Aegis — escuta pareada</title><style>body{font:16px system-ui;max-width:1000px;margin:2em auto}article{border:1px solid #bbb;padding:1em;margin:1em 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:1em}audio{display:block;width:100%}.notes{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:1em 0}textarea{width:98%;min-height:5em}button{padding:.7em}</style><h1>Aegis — escuta pareada</h1><p>Os rótulos A/B foram embaralhados. Compare com as âncoras reais, ouça a frase inteira e registre falhas específicas. A chave fica separada. Identidade e caráter robótico não foram julgados automaticamente.</p><p>Os WAVs originais são reutilizados por links, sem regravação. <label><input id="matched" type="checkbox" checked> Atenuar o mais alto para igualar RMS dos quadros ativos dentro de cada par</label>. Isso é auxílio de volume, não LUFS nem igualdade perceptiva comprovada. Escute também no volume nativo e ajuste o dispositivo se necessário.</p>'''+''.join(articles)+'''<button onclick="download()">Salvar ficha JSON</button><script>function gain(){document.querySelectorAll('audio[data-gain]').forEach(x=>x.volume=document.querySelector('#matched').checked?Number(x.dataset.gain):1)}document.querySelector('#matched').onchange=gain;gain();function download(){let answers={created:new Date().toISOString(),volume_match:document.querySelector('#matched').checked,answers:{}};document.querySelectorAll('select,textarea').forEach(x=>answers.answers[x.name]=x.value);let u=URL.createObjectURL(new Blob([JSON.stringify(answers,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=u;a.download='ficha-escuta.json';a.click();URL.revokeObjectURL(u)}</script></html>'''
    (out/'index.html').write_text(content);(out/'review-template.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2)+'\n')
    key.parent.mkdir(parents=True,exist_ok=True,mode=0o700);key.write_text(json.dumps(dict(seed=a.seed,pairs_source_sha256=sha(a.pairs),pairs=mapping),ensure_ascii=False,indent=2)+'\n');key.chmod(0o600)
    print(json.dumps(dict(output=str(out),pairs=len(mapping),unique_original_wavs=len(linked),audio_bytes_copied=0,key=str(key)),ensure_ascii=False))
if __name__=='__main__':main()
