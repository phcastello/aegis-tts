"""GPU-queue-only FastPitch microbatch measurements on real 10-18s samples."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,path)


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--python",default="/var/tmp/aegis-v2-evaluation-20261102/envs/f5/bin/python")
    p.add_argument("--runner",default="/home/phc/aegis-v2-lab/training/fastpitch_experiment.py")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--seed",type=int,default=20261103)
    p.add_argument("--min-seconds",type=float,default=10.)
    p.add_argument("--max-seconds",type=float,default=18.)
    p.add_argument("--train-count",type=int,default=48)
    p.add_argument("--selection",choices=["random","longest"],default="random")
    p.add_argument("--configs",default="1:4,2:2,4:1,8:1",help="Comma-separated microbatch:accumulation configurations")
    p.add_argument("--stop-after-first-success",action="store_true",help="For longest-case fitting, try fallbacks only after measured OOM")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data=a.output/"subset";data.mkdir(exist_ok=True)
    selected={}
    for name,count in (("train",a.train_count),("validation",1)):
        available=[r for r in read_rows(a.features/(name+".jsonl")) if a.min_seconds<=r["frames"]*256/24000<=a.max_seconds]
        if a.selection=="longest":available.sort(key=lambda r:(-r["frames"],r["id"]))
        else:random.Random(a.seed).shuffle(available)
        if len(available)<count:raise ValueError(f"Insufficient selected samples for {name}")
        selected[name]=available[:count]
        (data/(name+".jsonl")).write_text("".join(json.dumps(r)+"\n" for r in selected[name]))
    features=json.loads((a.features/"features.json").read_text())
    features["resource_probe"]=dict(source_manifests={n:sha(a.features/(n+".jsonl")) for n in selected},
                                     selection=dict(method=a.selection,seed=a.seed,min_seconds=a.min_seconds,max_seconds=a.max_seconds,
                                                    train_count=a.train_count,validation_count=1,final_test_opened=False))
    write(data/"features.json",features)
    configurations=[tuple(map(int,item.split(":"))) for item in a.configs.split(",")]
    if any(len(c)!=2 or min(c)<1 for c in configurations):raise ValueError("Invalid microbatch:accumulation configuration")
    plan=dict(seed=a.seed,runner_sha256=sha(a.runner),configs=configurations,
              selected_samples={n:[dict(id=r["id"],frames=r["frames"],feature_sha256=r.get("feature_sha256")) for r in rs] for n,rs in selected.items()},
              method="3 updates each; initial fixed val1; update1 warmup excluded; updates2/3 cost/sample and seconds audio/sec; samples/update recorded explicitly",
              precision="BF16 with FP32 cumulative duration boundaries and pitch sums",
              estimated_additional_disk_peak_bytes=1800000000,
              checkpoint_policy="collect state hashes/step evidence then remove only this benchmark run's .pt/.tmp before next configuration")
    write(a.output/"plan.json",plan)
    if not a.execute:
        print(json.dumps(dict(event="plan",path=str(a.output/"plan.json"))));return
    parent=Path(f"/proc/{os.getppid()}/cmdline").read_bytes().decode().replace("\x00"," ")
    if "v2_gpu_worker" not in parent:
        raise RuntimeError("Execute only as a job of the private v2_gpu_worker queue")
    child=[None]
    def stop(sig,frame):
        if child[0] is not None and child[0].poll() is None:
            child[0].send_signal(signal.SIGINT)
            child[0].wait(timeout=120)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
    results=[]
    for batch,accum in configurations:
        name=f"batch{batch}-accum{accum}";run=a.output/name
        if (run/"result.json").exists():
            results.append(json.loads((run/"result.json").read_text()));continue
        run.mkdir(exist_ok=True)
        if any(run.glob("*.pt")):raise RuntimeError(f"Partial probe found: inspect {run} before rerun")
        command=[a.python,a.runner,"train","--features",str(data),"--output",str(run),"--device","cuda",
                 "--precision","bf16","--batch-size",str(batch),"--accumulation",str(accum),
                 "--max-steps","3","--validate-every","1000","--validation-limit","1","--seed",str(a.seed)]
        began=time.monotonic()
        with open(run/"stdout.log","w") as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                  env={**os.environ,"OMP_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"1","NUMBA_NUM_THREADS":"1"})
            child[0]=proc
            write(a.output/"progress.json",dict(configuration=name,pid=proc.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status="running"))
            code=proc.wait()
        child[0]=None
        lines=(run/"stdout.log").read_text().splitlines()
        events=[]
        for line in lines:
            try:events.append(json.loads(line))
            except (ValueError,TypeError):pass
        steps=[r for r in events if r.get("event")=="step"]
        stable=steps[1:]
        oom=any("out of memory" in s.lower() or "cuda error: out of memory" in s.lower() for s in lines)
        result=dict(configuration=name,command=command,exit_code=code,oom=oom,total_wall_seconds=time.monotonic()-began,
                    steps=steps,initial_validation=[e for e in events if e.get("event")=="initial_validation"],
                    successful_updates=len(steps),samples_per_update=batch*accum,
                    weight_update_proven=bool(steps and all(s["weight_delta"]>0 for s in steps)),
                    warmed_seconds_per_sample_median=statistics.median(s["seconds"]/s["samples"] for s in stable) if stable else None,
                    warmed_audio_seconds_per_wall_second=statistics.median(s["audio_seconds"]/s["seconds"] for s in stable) if stable else None,
                    peak_torch_allocated_bytes=max([s["vram_peak_bytes"] for s in steps],default=0),
                    checkpoint_hashes={p.name:sha(p) for p in run.glob("*.pt") if p.name in ("last.pt","best.pt")},
                    log_sha256=sha(run/"stdout.log"))
        write(run/"result.json",result)
        # Every candidate is an explicitly disposable engineering probe; source
        # features/audio remain shared. Restrict deletions to exact run directory.
        removed=[]
        for path in list(run.glob("*.pt"))+list(run.glob("*.pt.tmp")):
            if path.is_symlink() or path.resolve().parent!=run.resolve():raise RuntimeError("Unsafe cleanup path")
            removed.append(dict(path=str(path),bytes=path.stat().st_size,inode=path.stat().st_ino,uid=path.stat().st_uid))
            path.unlink()
        write(run/"cleanup.json",dict(removed=removed,free_bytes_after=shutil.disk_usage(run).free))
        results.append(result);write(a.output/"results.json",results)
        print(json.dumps(dict(event="resource_result",configuration=name,exit_code=code,oom=oom,
                               cost_per_sample=result["warmed_seconds_per_sample_median"],peak_bytes=result["peak_torch_allocated_bytes"])),flush=True)
        if code and not oom:
            raise RuntimeError(f"Non-memory probe failure in {name}; inspect log before continuing")
        if code==0 and len(steps)==3 and a.stop_after_first_success:break
        if oom and not a.stop_after_first_success:break
    usable=[r for r in results if r["exit_code"]==0 and r["successful_updates"]==3 and r["weight_update_proven"]]
    winner=min(usable,key=lambda r:r["warmed_seconds_per_sample_median"]) if usable else None
    write(a.output/"decision.json",dict(proposed_configuration=winner["configuration"] if winner else None,
             rationale="Lowest measured warmed cost per sample among complete real updates; verify variance and longest public sample memory before long training",
             resource_results_sha256=sha(a.output/"results.json"),not_a_quality_or_inference_ranking=True))
    write(a.output/"progress.json",dict(status="complete",configurations=len(results)))


if __name__=="__main__":main()
