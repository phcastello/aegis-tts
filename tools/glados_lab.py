#!/usr/bin/env python3
"""Offline, reproducible comparison lab for streaming-safe Aegis voice DSP."""
from __future__ import annotations
import argparse, csv, hashlib, html, json, math, shutil, subprocess, sys, time, wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATE, CHANNELS = 24000, 1

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def command(args, **kw): return subprocess.run(args, text=True, capture_output=True, check=False, **kw)
def tool_info():
    out={"available":{},"versions":{},"filters":[]}
    for name, args in (("ffmpeg",["ffmpeg","-version"]),("ffprobe",["ffprobe","-version"])):
        p=command(args); out["available"][name]=p.returncode==0; out["versions"][name]=(p.stdout.splitlines() or [p.stderr])[0] if p.returncode==0 else None
    if out["available"]["ffmpeg"]:
        p=command(["ffmpeg","-filters"])
        # Rows look like: " TSC acompressor A->A ...".  The filter name is
        # the second whitespace-separated field, not the prose description.
        out["filters"]=[x.split()[1] for x in p.stdout.splitlines() if len(x.split())>=4 and x.lstrip().startswith(("T","."))]
    return out
def load_yaml(path):
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf8"))["presets"]
    except ImportError as e: raise RuntimeError("PyYAML é necessário para ler os presets (pip install PyYAML)") from e
def interpolate(graph, params):
    class Values(dict):
        def __missing__(self,k): raise ValueError(f"Parâmetro ausente no filtergraph: {k}")
    # Older hand-authored presets used `,type=t`; FFmpeg options are colon
    # separated, so canonicalize it before validation/execution.
    expanded=graph.format_map(Values(params)).replace(",type=t", ":type=t")
    # chorus requires in_gain:out_gain:delays:decays:speeds:depths.  The
    # curated shorthand historically omitted its final (very small) depth.
    import re
    def chorus_depth(match):
        parts=match.group(1).split(":")
        return "chorus="+match.group(1)+(":0.10" if len(parts)==5 else "")
    return re.sub(r"chorus=([^,;]+)",chorus_depth,expanded)
def needed_filters(graph):
    # Names occur at start or after comma/semicolon; labels and options are excluded.
    import re
    names=set(re.findall(r"(?:^|[;,])(?:\[[^]]+\])*([a-z][a-z0-9_]*)=", graph))
    # `type=t` is an aphaser option in older graph syntax, not a filter.
    return names-{"type"}
def probe(path):
    p=command(["ffprobe","-v","error","-show_entries","format=duration:stream=sample_rate,channels,sample_fmt","-of","json",str(path)])
    if p.returncode: raise RuntimeError(p.stderr.strip())
    d=json.loads(p.stdout); s=d["streams"][0]
    return {"duration_seconds":float(d["format"].get("duration",0)),"sample_rate":int(s["sample_rate"]),"channels":int(s["channels"]),"sample_fmt":s.get("sample_fmt")}
def metrics(path):
    info=probe(path); p=command(["ffmpeg","-hide_banner","-i",str(path),"-af","ebur128=peak=true,astats=metadata=1:reset=0","-f","null","-"])
    import re
    text=p.stderr; lufs=re.findall(r"\bI:\s*(-?[\d.]+) LUFS",text); peak=re.findall(r"Peak level dB:\s*(-?[\d.]+)",text); rms=re.findall(r"RMS level dB:\s*(-?[\d.]+)",text); dc=re.findall(r"DC offset:\s*(-?[\d.eE+]+)",text)
    info.update(integrated_lufs=float(lufs[-1]) if lufs else None, true_peak_db=float(peak[-1]) if peak else None, rms_db=float(rms[-1]) if rms else None, dc_offset=float(dc[-1]) if dc else None)
    info["clipping_detected"]=(info["true_peak_db"] is not None and info["true_peak_db"] >= -0.1)
    return info
def normalize(src,dst):
    # Lab-only one-pass loudness matching; never used in the production graph.
    p=command(["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(src),"-af","loudnorm=I=-16:TP=-1:LRA=11","-ar",str(RATE),"-ac","1","-c:a","pcm_s16le",str(dst)])
    if p.returncode: raise RuntimeError(p.stderr)
def render(source, target, graph):
    args=["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(source),"-af",graph,"-ar",str(RATE),"-ac","1","-c:a","pcm_s16le",str(target)]
    t=time.monotonic(); p=command(args); return args,time.monotonic()-t,p
def html_page(manifest, blind=False):
    rows=[]; presets=[p for p in manifest["presets"] if p.get("success")]
    import random; order=presets[:]; random.Random(manifest["source_sha256"]).shuffle(order)
    for n,p in enumerate(order,1):
        title=f"Sample {n:02d}" if blind else f"{p['id']} — {p['name']}"
        details="" if blind else f"<p>{html.escape(p['description'])}<br><small>{html.escape(json.dumps(p.get('parameters',{}), ensure_ascii=False))}</small></p>"
        rows.append(f'<article data-family="{p["family"]}"><h2>{title}</h2>{details}<audio controls preload="none" src="{p["file"]}"></audio> <button onclick="this.previousElementSibling.play()">Play</button> <button onclick="this.previousElementSibling.currentTime=0;this.previousElementSibling.play()">Replay</button><p><button onclick="ab(this,\'{p["file"]}\')">Alternar original/preset</button></p>{ratings(p["id"] if not blind else f"sample-{n:02d}")}</article>')
    opts="" if blind else '<label>Família <select onchange="document.querySelectorAll(\'article\').forEach(x=>x.hidden=this.value&&x.dataset.family!=this.value)"><option value="">todas</option>'+''.join(f'<option>{x}</option>' for x in sorted({p['family'] for p in presets}))+'</select></label>'
    return """<!doctype html><meta charset=utf-8><title>Aegis GLaDOS listening lab</title><style>body{font:15px system-ui;margin:2rem;max-width:900px}article{border-top:1px solid #ccc;padding:1rem 0}label{display:inline-block;margin:.25rem}input{width:3em}textarea{display:block;width:95%;height:4em}</style><h1>"""+("Comparação cega" if blind else "Laboratório Aegis / GLaDOS")+"</h1>"+opts+"<p>Original: <audio controls src=\"original.wav\"></audio></p><button onclick=\"exportRatings()\">Exportar notas JSON</button>"+''.join(rows)+"""<script>const fields=['glados','intelligibility','identity','pleasantness','robotic','fatigue'];function ab(b,f){let a=b.parentElement.parentElement.querySelector('audio');a.src=a.src.endsWith('original.wav')?f:'original.wav';a.play()}function save(el){localStorage.setItem('glados-ratings-'+el.name,el.value)}function exportRatings(){let o={};document.querySelectorAll('[name]').forEach(x=>o[x.name]=x.value);let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(o,null,2)],{type:'application/json'}));a.download='ratings.json';a.click()}</script>"""
def ratings(key):
    labels=[('glados','estética GLaDOS'),('intelligibility','inteligibilidade'),('identity','identidade Giseli'),('pleasantness','agradabilidade'),('robotic','textura robótica'),('fatigue','fadiga auditiva')]
    return ''.join(f'<label>{t} <input type=number min=1 max=5 name="{key}.{n}" onchange="save(this)" value=""></label>' for n,t in labels)+f'<label>observações<textarea name="{key}.notes" onchange="save(this)"></textarea></label>'
def write_outputs(out, manifest):
    out.mkdir(parents=True,exist_ok=True); (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf8")
    keys=["id","family","file","success","error","render_seconds","duration_seconds","integrated_lufs","true_peak_db","rms_db","dc_offset","clipping_detected","streaming_safe"]
    with (out/"manifest.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows([{k:p.get(k) for k in keys} for p in manifest["presets"]])
    (out/"listening.html").write_text(html_page(manifest),encoding="utf8"); (out/"listening-blind.html").write_text(html_page(manifest,True),encoding="utf8")
    good=[p for p in manifest["presets"] if p.get("success")]; order=good[:]
    import random; random.Random(manifest["source_sha256"]).shuffle(order)
    (out/"blind-map.json").write_text(json.dumps({f"Sample {i:02d}":p["id"] for i,p in enumerate(order,1)},indent=2),encoding="utf8")
def run(args):
    source=Path(args.input).resolve(); out=Path(args.output_dir).resolve(); info=tool_info()
    if not source.is_file(): raise SystemExit(f"Entrada inexistente: {source}")
    if not all(info["available"].values()): raise SystemExit("FFmpeg e FFprobe são obrigatórios; diagnóstico: "+json.dumps(info["available"]))
    si=probe(source)
    if (si["sample_rate"],si["channels"]) != (RATE,CHANNELS): raise SystemExit("A entrada precisa ser WAV 24 kHz mono.")
    presets=load_yaml(Path(args.config)); available=set(info["filters"]); records=[]; out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,out/"original.wav")
    for pre in presets:
        record={**pre,"file":f"preset-{pre['id']}.wav","success":False,"error":None}; graph=interpolate(pre["filtergraph"],pre["parameters"]); record["filtergraph"]=graph
        missing=sorted(needed_filters(graph)-available)
        if not pre.get("enabled",True) or missing: record["error"]="disabled" if not pre.get("enabled",True) else "missing filters: "+", ".join(missing); records.append(record); continue
        raw=out/("diagnostic-"+record["file"]); cmd,elapsed,p=render(source,raw,graph); record.update(ffmpeg_command=cmd,render_seconds=round(elapsed,4))
        if p.returncode: record["error"]=p.stderr[-2000:]; records.append(record); continue
        try:
            normalize(raw,out/record["file"]); record.update(metrics(out/record["file"])); record["output_sha256"]=sha(out/record["file"]); record["success"]=True
        except Exception as exc: record["error"]=str(exc)
        records.append(record)
    manifest={"source":str(source),"source_sha256":sha(source),"sample_rate":RATE,"channels":CHANNELS,"executed_at":datetime.now(timezone.utc).isoformat(),"ffmpeg":info,"presets":records}; write_outputs(out,manifest)
    print(json.dumps({"successful":sum(x["success"] for x in records),"skipped_or_failed":sum(not x["success"] for x in records),"output":str(out)},ensure_ascii=False))
def refine(args):
    ratings=json.loads(Path(args.ratings).read_text(encoding="utf8")); prefix={}
    for k,v in ratings.items():
        if '.' in k and k.split('.',1)[1] != 'notes': prefix.setdefault(k.split('.',1)[0],[]).append(float(v or 0))
    mapping_path=Path(args.ratings).with_name("blind-map.json")
    mapping=json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    top=[mapping.get(x.replace("sample-","Sample ").title(),x) for x in sorted(prefix,key=lambda k:sum(prefix[k])/len(prefix[k]),reverse=True)[:args.top]]
    originals={p["id"]:p for p in load_yaml(Path(args.config))}; variations=[]
    for ident in top:
        if ident not in originals: continue
        base=originals[ident]
        for n,(field,delta) in enumerate((("wet_mix",.05),("wet_mix",-.05),("pitch_semitones",.3),("pitch_semitones",-.3)),1):
            p=json.loads(json.dumps(base)); p["id"]=f"R-{ident}-{n:02d}"; p["name"]=base["name"]+f" refinement {n:02d}"; p["description"]+=f" Variação de {field}."; q=p["parameters"]; q[field]=round(float(q.get(field,0))+delta,3)
            if field=="pitch_semitones": q["pitch_factor"]=round(2**(q[field]/12),6); q["sample_rate_intermediate"]=round(RATE*q["pitch_factor"])
            if field=="wet_mix": q["wet_mix"]=max(.05,min(.5,q[field])); q["dry_mix"]=round(1-q["wet_mix"],3)
            variations.append(p)
    destination=Path(args.config).with_name("glados-refined-presets.yaml")
    try:
        import yaml; destination.write_text(yaml.safe_dump({"presets":variations},allow_unicode=True,sort_keys=False),encoding="utf8")
    except ImportError: raise RuntimeError("PyYAML é necessário")
    print(json.dumps({"top":top,"variations":len(variations),"config":str(destination),"next_command":f"python tools/glados_lab.py --config {destination}"},ensure_ascii=False))
def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="action"); r=sub.add_parser("refine"); r.add_argument("--ratings",required=True); r.add_argument("--top",type=int,default=4)
    ap.add_argument("--input",default="samples/glados-lab/source.wav"); ap.add_argument("--output-dir",default="samples/glados-lab/results"); ap.add_argument("--config",default=str(ROOT/"config/glados-presets.yaml")); ns=ap.parse_args()
    if ns.action=="refine": refine(ns)
    else: run(ns)
if __name__=="__main__": main()
