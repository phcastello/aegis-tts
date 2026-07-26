from __future__ import annotations
import argparse, csv, json, os, subprocess, sys
from pathlib import Path
import numpy as np
from .config import settings, yaml_config
from .manifests import append_jsonl, read_jsonl, latest_candidates
from .audio import sha256, validate_audio
from .timing import Timer, stats
from .analysis.world import analyze_world
from .transforms.prosody import transform_f0
from .transforms.envelope import transform_envelope
from .transforms.aperiodicity import transform_aperiodicity
from .synthesis.world_engine import synthesize_world
from .synthesis.lpc_engine import analyze_lpc, synthesize_lpc
from .mastering.chain import master

def _corpus(): return yaml_config("corpus.yaml")["sentences"]
def _profiles(): return yaml_config("presets.yaml")["source_profiles"]
def _manifest(cfg): return cfg.results/"manifests"/"candidates.jsonl"
def _source_manifest(cfg): return cfg.results/"manifests"/"sources.jsonl"
def _sf():
    try: import soundfile; return soundfile
    except ImportError as e: raise RuntimeError("DSP environment is missing soundfile; run create_environments.sh") from e

def prepare(cfg):
    for p in [cfg.results,cfg.reports,cfg.cache,cfg.models,cfg.envs]: p.mkdir(parents=True,exist_ok=True)
    (cfg.results/"manifests").mkdir(exist_ok=True)
    print(json.dumps({"status":"ready","root":str(cfg.root),"private_environments":str(cfg.envs)}))

def preflight(cfg):
    spec=yaml_config("chatterbox.yaml"); problems=[]
    checks={"model":cfg.model=="ResembleAI/Chatterbox-Multilingual-pt-br","revision":cfg.revision==spec["revision"],"locale":spec["locale"]=="pt-BR","language_id":cfg.language_id=="pt","t3":spec["assets"].get("t3_pt_br.safetensors") is not None,"decoder":spec["decoder"]=="V3","tokenizer":spec["tokenizer"]=="grapheme_mtl_merged_expanded_v1.json","dtype":cfg.dtype in {"float32","float16","bfloat16"}}
    for k,v in checks.items():
        if not v: problems.append(k)
    chat_python=cfg.envs/"chatterbox"/"bin"/"python"
    report={"status":"configuration_valid" if not problems else "invalid","checks":checks,"problems":problems,"chatterbox_environment":{"path":str(chat_python),"installed":chat_python.exists()},"model_metadata":spec,"gpu":None}
    if chat_python.exists():
        probe=subprocess.run([str(chat_python),"-c","import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda'); print(torch.version.cuda)"],text=True,capture_output=True,timeout=30)
        report["gpu"]={"probe":probe.stdout.strip(),"returncode":probe.returncode}
    print(json.dumps(report,ensure_ascii=False,indent=2));
    if problems: raise SystemExit(2)

def _prompt(profile, cfg):
    choice=profile["audio_prompt"]
    if choice is None: return None
    env="AEGIS_AEGIS_GLADOS_REFERENCE_WAV" if choice=="aegis_reference" else "AEGIS_CHATTERBOX_NEUTRAL_REFERENCE_WAV"; value=os.environ.get(env, str(cfg.root/"private/references/Aegis_GLaDOS.wav") if choice=="aegis_reference" else "")
    return value if value and Path(value).is_file() else None

def generate_sources(cfg):
    py=cfg.envs/"chatterbox"/"bin"/"python"; worker=cfg.root/"scripts/chatterbox_worker.py"
    if not py.exists(): raise RuntimeError(f"missing private Chatterbox environment: run {cfg.root}/scripts/create_environments.sh")
    completed={(x["profile"],x["sentence_id"]):x for x in read_jsonl(_source_manifest(cfg)) if x.get("status")=="complete"}
    process=subprocess.Popen([str(py),str(worker)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    ready=json.loads(process.stdout.readline())
    if ready.get("status")!="ready": raise RuntimeError(f"worker failed: {ready}")
    for code,profile in _profiles().items():
        prompt=_prompt(profile,cfg)
        if profile["audio_prompt"] and not prompt:
            for sentence in _corpus(): append_jsonl(_source_manifest(cfg),{"profile":code,"sentence_id":sentence["id"],"status":"skipped","error":"reference missing"})
            continue
        for sentence in _corpus():
            output=cfg.results/"sources"/code/f'{sentence["id"]}.wav'; cached=completed.get((code,sentence["id"]))
            if cached and output.exists() and cached.get("sha256")==sha256(output): continue
            req={"op":"generate","output":str(output),"text":sentence["text"],"audio_prompt":prompt,"exaggeration":profile["exaggeration"],"cfg_weight":profile["cfg_weight"],"seed":cfg.seed}; process.stdin.write(json.dumps(req)+"\n"); process.stdin.flush(); res=json.loads(process.stdout.readline())
            record={"engine":"chatterbox","model":cfg.model,"revision":cfg.revision,"profile":code,"sentence_id":sentence["id"],"text":sentence["text"],"audio_prompt":prompt,"exaggeration":profile["exaggeration"],"cfg_weight":profile["cfg_weight"],"temperature":.8,"seed":cfg.seed,**res,"cold_start_seconds":ready.get("model_load_seconds"),"error":res.get("error")}; append_jsonl(_source_manifest(cfg),record)
    process.stdin.close(); process.wait(timeout=30)

def _process_world(cfg, codes):
    sf=_sf(); presets=yaml_config("presets.yaml")["world"]
    for source in read_jsonl(_source_manifest(cfg)):
        if source.get("status")!="complete": continue
        wave,sr=sf.read(source["path"],always_2d=False); wave=np.asarray(wave,float)
        try:
            with Timer() as t: rep=analyze_world(wave,sr,cfg.frame_period_ms)
            for code in codes:
                preset=presets[code]
                with Timer() as x: f0=transform_f0(rep.f0_hz,rep.voiced,**preset); sp=transform_envelope(rep.spectral_envelope,**preset); ap=transform_aperiodicity(rep.aperiodicity,rep.voiced,**preset)
                with Timer() as y: raw=synthesize_world(rep,f0_hz=f0,spectral_envelope=sp,aperiodicity=ap,seed=cfg.seed,duration_samples=len(wave))
                with Timer() as z: final=master(raw,sr,**yaml_config("presets.yaml")["mastering"])
                validate_audio(final,sr,len(wave)); raw_path=cfg.results/"raw"/source["profile"]/code/f'{source["sentence_id"]}.wav'; out=cfg.results/"mastered"/source["profile"]/code/f'{source["sentence_id"]}.wav'; raw_path.parent.mkdir(parents=True,exist_ok=True); out.parent.mkdir(parents=True,exist_ok=True); sf.write(raw_path,raw,sr); sf.write(out,final,sr)
                append_jsonl(_manifest(cfg),{"source_profile":source["profile"],"engine":"world","preset":code,"sentence_id":source["sentence_id"],"text":source["text"],"source_path":source["path"],"output_raw_path":str(raw_path),"output_mastered_path":str(out),"parameters":preset,"sample_rate":sr,"input_samples":len(wave),"output_samples":len(final),"duration_seconds":len(final)/sr,"analysis_seconds":t.seconds,"transform_seconds":x.seconds,"synthesis_seconds":y.seconds,"mastering_seconds":z.seconds,"total_dsp_seconds":t.seconds+x.seconds+y.seconds+z.seconds,"dsp_rtf":(t.seconds+x.seconds+y.seconds+z.seconds)/(len(final)/sr),"sha256":sha256(out),"status":"complete","error":None})
        except Exception as e: append_jsonl(_manifest(cfg),{"source_profile":source["profile"],"engine":"world","preset":"/".join(codes),"sentence_id":source["sentence_id"],"status":"failed","error":str(e)})

def _process_lpc(cfg):
    sf=_sf(); presets=yaml_config("presets.yaml")["lpc"]
    for source in read_jsonl(_source_manifest(cfg)):
        if source.get("status")!="complete": continue
        wave,sr=sf.read(source["path"],always_2d=False); wave=np.asarray(wave,float)
        try:
            with Timer() as t:
                world=analyze_world(wave,sr,cfg.frame_period_ms); rep=analyze_lpc(wave,sr,world.f0_hz,world.voiced)
            for code,preset in presets.items():
                with Timer() as x: rep.f0_hz=transform_f0(rep.f0_hz,rep.voiced,**preset)
                with Timer() as y: raw=synthesize_lpc(rep,seed=cfg.seed,**preset); raw=np.pad(raw[:len(wave)],(0,max(0,len(wave)-len(raw))))
                with Timer() as z: final=master(raw,sr,**yaml_config("presets.yaml")["mastering"])
                validate_audio(final,sr,len(wave)); raw_path=cfg.results/"raw"/source["profile"]/code/f'{source["sentence_id"]}.wav'; out=cfg.results/"mastered"/source["profile"]/code/f'{source["sentence_id"]}.wav'; raw_path.parent.mkdir(parents=True,exist_ok=True); out.parent.mkdir(parents=True,exist_ok=True); sf.write(raw_path,raw,sr); sf.write(out,final,sr)
                total=t.seconds+x.seconds+y.seconds+z.seconds; append_jsonl(_manifest(cfg),{"source_profile":source["profile"],"engine":"lpc","preset":code,"sentence_id":source["sentence_id"],"text":source["text"],"source_path":source["path"],"output_raw_path":str(raw_path),"output_mastered_path":str(out),"parameters":preset,"sample_rate":sr,"input_samples":len(wave),"output_samples":len(final),"duration_seconds":len(final)/sr,"analysis_seconds":t.seconds,"transform_seconds":x.seconds,"synthesis_seconds":y.seconds,"mastering_seconds":z.seconds,"total_dsp_seconds":total,"dsp_rtf":total/(len(final)/sr),"sha256":sha256(out),"status":"complete","error":None,"lpc_metadata":rep.metadata})
        except Exception as e: append_jsonl(_manifest(cfg),{"source_profile":source["profile"],"engine":"lpc","sentence_id":source["sentence_id"],"status":"failed","error":str(e)})

def analyze_sources(cfg):
    sf=_sf(); result=[]
    for item in read_jsonl(_source_manifest(cfg)):
        if item.get("status")!="complete": continue
        w,sr=sf.read(item["path"]); a=analyze_world(w,sr,cfg.frame_period_ms); result.append({"profile":item["profile"],"sentence_id":item["sentence_id"],"f0_dio_voiced_frames":int(a.voiced.sum()),"aperiodicity_domain":[a.metadata["aperiodicity_min"],a.metadata["aperiodicity_max"]]})
    p=cfg.results/"analysis.json"; p.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(p)

def benchmark(cfg):
    items=latest_candidates(read_jsonl(_manifest(cfg))); values=[x["dsp_rtf"] for x in items if x.get("status")=="complete" and "dsp_rtf" in x]
    by_engine={}
    for engine in {x.get("engine") for x in items}:
        r=[x["dsp_rtf"] for x in items if x.get("engine")==engine and x.get("status")=="complete" and "dsp_rtf" in x]
        if r: by_engine[engine]={"rtf":stats(r),"latency_verdict":"acceptable" if np.median(r)<=.25 else "inadequate"}
    report={"dsp_rtf":stats(values),"by_engine":by_engine,"classification":"inconclusivo","reason":"Human listening has not selected quality; both WORLD and LPC exceed the DSP RTF target in this environment."}
    (cfg.results/"latency.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))

def promote(cfg, source_profile, preset):
    p=cfg.results/"promoted.json"; existing=json.loads(p.read_text()) if p.exists() else []; pair={"source_profile":source_profile,"preset":preset};
    if pair not in existing: existing.append(pair)
    if len(existing)>3: raise ValueError("only two or three promoted candidates are allowed")
    p.write_text(json.dumps(existing,indent=2))

def report(cfg):
    from .reporting.html import write_reports
    write_reports(cfg,read_jsonl(_source_manifest(cfg)),latest_candidates(read_jsonl(_manifest(cfg))))

def main(argv=None):
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True); [sub.add_parser(x) for x in ["preflight","prepare","generate-sources","analyze-sources","world-baselines","world-candidates","lpc-candidates","benchmark-latency","report","status","run-mvp","long-form"]]; q=sub.add_parser("promote"); q.add_argument("--source-profile",required=True); q.add_argument("--preset",required=True); cold=sub.add_parser("qwen-cold-run"); cold.add_argument("--profiles"); cold.add_argument("--keep-vllm",action="store_true"); args=p.parse_args(argv); cfg=settings()
    if args.command=="prepare": prepare(cfg)
    elif args.command=="preflight": preflight(cfg)
    elif args.command=="generate-sources": generate_sources(cfg)
    elif args.command=="analyze-sources": analyze_sources(cfg)
    elif args.command=="world-baselines": _process_world(cfg,["W00"])
    elif args.command=="world-candidates": _process_world(cfg,["W01","W02","W03","W04"])
    elif args.command=="lpc-candidates": _process_lpc(cfg)
    elif args.command=="benchmark-latency": benchmark(cfg)
    elif args.command=="report": report(cfg)
    elif args.command=="status": print(json.dumps({"sources":len(read_jsonl(_source_manifest(cfg))),"candidates":len(read_jsonl(_manifest(cfg))),"root":str(cfg.root)},indent=2))
    elif args.command=="promote": promote(cfg,args.source_profile,args.preset)
    elif args.command=="long-form": print("Generate promoted five-minute corpus only after listening approval.")
    elif args.command=="qwen-cold-run":
        from .qwen_cold import run as run_qwen_cold
        run_qwen_cold(cfg,args.profiles.split(",") if args.profiles else None,args.keep_vllm)
    elif args.command=="run-mvp":
        for command in ["preflight","generate-sources","analyze-sources","world-baselines","world-candidates","lpc-candidates","benchmark-latency","report"]:
            main([command])
