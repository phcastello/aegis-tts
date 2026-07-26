from __future__ import annotations
import csv, json, random
from pathlib import Path
def _relative(output, root): return Path(output).resolve().relative_to(root.resolve()).as_posix()
def write_reports(cfg, sources, candidates):
    out=cfg.reports; out.mkdir(parents=True,exist_ok=True); root=cfg.root
    complete=[x for x in candidates if x.get("status")=="complete"]
    rows=[]
    for x in sources:
        if x.get("status")=="complete": rows.append((x["sentence_id"],"Chatterbox "+x["profile"],x["path"]))
    for x in complete: rows.append((x["sentence_id"],x["engine"].upper()+" "+x["preset"],x["output_mastered_path"]))
    grouped={}
    for sentence,label,path in rows: grouped.setdefault(sentence,[]).append((label,path))
    blocks=[]
    for sentence, entries in grouped.items(): blocks.append("<h2>"+sentence+"</h2>"+"".join(f'<p>{label}<br><audio controls src="../../{_relative(path,root)}"></audio></p>' for label,path in entries))
    form="".join(f'<label>{x} <input type="number" min="1" max="5" data-score="{x}"></label><br>' for x in ["não parece humana","identidade de sistema","inteligibilidade","pronúncia pt-BR","neutralidade","conforto","consistência","personalidade própria","fadiga espectral","monotonia cansativa","artefatos interessantes","artefatos ruins"])
    script="<script>function save(){localStorage.setItem('aegis-lab-notes',JSON.stringify([...document.querySelectorAll('[data-score]')].map(x=>[x.dataset.score,x.value])))} function exportNotes(){let a=document.createElement('a');a.href='data:application/json,'+encodeURIComponent(localStorage.getItem('aegis-lab-notes')||'[]');a.download='aegis-listening-notes.json';a.click()}</script>"
    (out/"listening.html").write_text("<meta charset=utf-8><h1>Aegis robotic voice — identified listening</h1>"+form+"<button onclick=save()>Salvar notas</button><button onclick=exportNotes()>Exportar JSON</button>"+"\n".join(blocks)+script,encoding="utf-8")
    rng=random.Random(40733133); blind=complete[:]; rng.shuffle(blind); mapping=[]; blind_blocks=[]
    for i,x in enumerate(blind,1):
        name=f"candidate-{i:03d}"; mapping.append({"blind":name,"engine":x["engine"],"preset":x["preset"],"path":x["output_mastered_path"]}); blind_blocks.append(f'<p>{name}<br><audio controls src="../../{_relative(x["output_mastered_path"],root)}"></audio></p>')
    (out/"listening-blind.html").write_text("<meta charset=utf-8><h1>Aegis robotic voice — blind listening</h1>"+form+"<button onclick=save()>Salvar notas</button><button onclick=exportNotes()>Exportar JSON</button>"+"".join(blind_blocks)+script,encoding="utf-8")
    (cfg.results/"blind-map.private.json").write_text(json.dumps(mapping,indent=2),encoding="utf-8")
    (out/"manifest.json").write_text(json.dumps({"sources":sources,"candidates":candidates},ensure_ascii=False,indent=2),encoding="utf-8")
    with (out/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
        all_rows=sources+candidates; keys=sorted({k for r in all_rows for k in r}); w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(all_rows)
    (out/"latency.html").write_text("<meta charset=utf-8><h1>Latency</h1><pre>"+(cfg.results/"latency.json").read_text() if (cfg.results/"latency.json").exists() else "<h1>Latency unavailable</h1>",encoding="utf-8")
    (out/"summary.md").write_text("# Aegis robotic voice lab\n\nNo automatic winner is selected. Human listening is required; higher fatigue, monotony, and bad-artifact scores are worse.\n",encoding="utf-8")
