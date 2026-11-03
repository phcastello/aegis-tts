"""Freeze a clean FastPitch view of the public corpus without editing originals."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from training.fastpitch_experiment import normalize, digest, atomic_json, rows, audit_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-root",type=Path,default=Path("/var/tmp/aegis-v2-work/portuguese"))
    p.add_argument("--old-features",type=Path,default=Path("/var/tmp/aegis-v2-work/fastpitch/portuguese-features"))
    p.add_argument("--input",type=Path,default=Path("/var/tmp/aegis-v2-work/portuguese/fastpitch-input-v1"))
    p.add_argument("--features",type=Path,default=Path("/var/tmp/aegis-v2-work/fastpitch/portuguese-features-clean-v1"))
    a=p.parse_args();a.input.mkdir(parents=True,exist_ok=True);a.features.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((a.corpus_root/"novelty.json").read_text())["source_manifests"]
    for name,expected in frozen.items():
        if digest(a.corpus_root/"prepared"/(name+".jsonl"))!=expected:
            raise ValueError(f"Original public {name} manifest changed")
    original={name:rows(a.corpus_root/"prepared"/(name+".jsonl")) for name in ("train","validation")}
    groups=collections.defaultdict(list)
    for name,items in original.items():
        for index,r in enumerate(items):groups[r["id"]].append((name,index,r))
    quarantine={"sample-47","sample-70","sample-1430","sample-828"}
    duplicates=[]
    for ident,items in groups.items():
        if len(items)>1:
            signatures={(normalize(r["text"]),r["sha256"]) for _,_,r in items}
            conflict=len(signatures)>1 or len({name for name,_,_ in items})>1
            if conflict:quarantine.add(ident)
            duplicates.append(dict(id=ident,count=len(items),conflicting=conflict,
                                   rows=[dict(split=name,index=index,text_sha256=hashlib.sha256(r["text"].encode()).hexdigest(),
                                              audio_sha256=r["sha256"]) for name,index,r in items]))
    selection={};seen=set();removed=[]
    for name,items in original.items():
        kept=[]
        for r in items:
            if r["id"] in quarantine:
                removed.append(dict(split=name,id=r["id"],reason="quarantined ambiguous/conflicting source ID"));continue
            if r["id"] in seen:
                removed.append(dict(split=name,id=r["id"],reason="exact duplicate id/text/audio; retain one"));continue
            seen.add(r["id"]);kept.append(r)
        selection[name]=kept
        path=a.input/(name+".jsonl")
        encoded="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in kept)
        if path.exists() and path.read_text()!=encoded:raise ValueError(f"Immutable snapshot differs: {path}")
        if not path.exists():path.write_text(encoded)
        path.chmod(0o444)
    receipt=dict(policy="quarantine all conflicting IDs and sample828; deduplicate identical id/text/audio; original manifests and splits unchanged",
                 originals=frozen,duplicates=duplicates,quarantine_ids=sorted(quarantine),removed=removed,
                 splits={name:dict(count=len(items),seconds=sum(r["duration"] for r in items),sha256=digest(a.input/(name+".jsonl"))) for name,items in selection.items()},
                 source_cache_status="Original parallel feature view is superseded: adjacent duplicate IDs could overwrite a previously verified cache object")
    atomic_json(a.input/"selection-receipt.json",receipt)
    dest=a.features/"features";dest.mkdir(exist_ok=True)
    linked=0
    for items in selection.values():
        for r in items:
            source=a.old_features/"features"/(r["id"]+".pt")
            target=dest/source.name
            if source.is_symlink():raise ValueError(f"Unexpected symlink: {source}")
            if source.exists() and not target.exists():os.link(source,target);linked+=1
    runner=Path(__file__).with_name("fastpitch_experiment.py")
    command=[sys.executable,str(runner),"prepare","--manifests",str(a.input/"train.jsonl"),str(a.input/"validation.jsonl"),
             "--output",str(a.features),"--workers","4","--pitch-method","world"]
    print(json.dumps(dict(event="clean_selection",splits=receipt["splits"],quarantine=sorted(quarantine),linked=linked,command=command)),flush=True)
    subprocess.run(command,check=True)
    feature_config=json.loads((a.features/"features.json").read_text())
    feature_config["clean_selection"]=dict(path=str(a.input/"selection-receipt.json"),sha256=digest(a.input/"selection-receipt.json"))
    atomic_json(a.features/"features.json",feature_config)
    # Independently verify all selected cached objects, original audio and text,
    # before declaring the new view qualified. Training repeats this audit.
    audit_features(feature_config,rows(a.features/"train.jsonl")+rows(a.features/"validation.jsonl"),a.features)
    final=dict(status="qualified",features=str(a.features),input_snapshot=str(a.input),
               source_originals=frozen,feature_manifests={n:digest(a.features/(n+".jsonl")) for n in selection},
               feature_config_sha256=digest(a.features/"features.json"),selection_receipt_sha256=digest(a.input/"selection-receipt.json"),
               startup_audit_sha256=digest(a.features/"startup-data-audit.json"))
    atomic_json(a.features/"qualification.json",final)
    for name in ("train.jsonl","validation.jsonl","features.json"):(a.features/name).chmod(0o444)
    print(json.dumps(dict(event="qualified",**final)),flush=True)


if __name__=="__main__":main()
