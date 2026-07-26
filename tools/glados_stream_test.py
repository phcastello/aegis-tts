#!/usr/bin/env python3
"""Validate one streaming-safe graph through FFmpeg stdin/stdout PCM pipes."""
from __future__ import annotations
import argparse, json, os, queue, resource, subprocess, threading, time, wave
from pathlib import Path
from glados_lab import RATE, load_yaml, interpolate, probe

def pcm_from_wav(path):
    with wave.open(str(path),"rb") as w:
        if (w.getframerate(),w.getnchannels(),w.getsampwidth()) != (RATE,1,2): raise ValueError("WAV must be 24 kHz mono s16")
        return w.readframes(w.getnframes())
def write_wav(path, pcm):
    with wave.open(str(path),"wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(pcm)
def run_case(pcm, graph, chunk, jitter_ms, variable=False):
    cmd=["ffmpeg","-hide_banner","-loglevel","error","-f","s16le","-ar",str(RATE),"-ac","1","-i","pipe:0","-af",graph,"-f","s16le","-ar",str(RATE),"-ac","1","pipe:1"]
    cpu_before=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    output=bytearray(); first=[None]
    def reader():
        while True:
            b=proc.stdout.read(4096)
            if not b: break
            if first[0] is None: first[0]=time.monotonic()
            output.extend(b)
    thread=threading.Thread(target=reader,daemon=True); started=time.monotonic(); thread.start()
    try:
        pos=0; sizes=[4096,8192,6144] if variable else [chunk]; i=0
        while pos<len(pcm):
            n=sizes[i%len(sizes)]; proc.stdin.write(pcm[pos:pos+n]); proc.stdin.flush(); pos+=n; i+=1
            if jitter_ms: time.sleep((i % (jitter_ms+1))/1000.0)
        proc.stdin.close(); code=proc.wait(timeout=30); thread.join(5)
    except BaseException:
        proc.terminate(); proc.wait(timeout=5); raise
    stderr=proc.stderr.read().decode("utf8","replace")
    cpu_after=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
    return {"chunk_bytes":"variable" if variable else chunk,"jitter_ms":jitter_ms,"first_audio_ms":None if first[0] is None else round((first[0]-started)*1000,2),"input_bytes":len(pcm),"output_bytes":len(output),"duration_seconds":round(len(output)/2/RATE,4),"tail_ms":round((len(output)-len(pcm))/2/RATE*1000,2),"cpu_seconds":round(cpu_after-cpu_before,4),"exit_code":code,"stderr":stderr[-1000:],"continuous":len(output)%2==0,"possible_truncation":len(output)==0},bytes(output)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",default="samples/glados-lab/source.wav"); ap.add_argument("--preset",required=True); ap.add_argument("--config",default="config/glados-presets.yaml"); ap.add_argument("--output-dir",default="samples/glados-lab/results"); ap.add_argument("--offline",help="WAV diagnóstico offline (por padrão diagnostic-preset-ID.wav)"); ns=ap.parse_args()
    presets={p["id"]:p for p in load_yaml(Path(ns.config))}; p=presets[ns.preset]
    if not p.get("streaming_safe"): raise SystemExit("Preset not marked streaming_safe")
    pcm=pcm_from_wav(ns.input); graph=interpolate(p["filtergraph"],p["parameters"]); out=Path(ns.output_dir); out.mkdir(parents=True,exist_ok=True)
    cases=[]
    offline=Path(ns.offline) if ns.offline else out/("diagnostic-preset-"+p["id"]+".wav")
    offline_pcm=pcm_from_wav(offline) if offline.exists() else None
    for chunk,jitter,var in ((4096,0,False),(8192,100,False),(4096,300,True)):
        result,data=run_case(pcm,graph,chunk,jitter,var); name=f"stream-{p['id']}-{result['chunk_bytes']}-{jitter}.wav"; write_wav(out/name,data); result["file"]=name; cases.append(result)
        if offline_pcm is not None: result["offline_byte_delta"]=len(data)-len(offline_pcm); result["matches_offline_bytes"]=data==offline_pcm
    report={"preset":p["id"],"graph":graph,"offline_reference":str(offline) if offline_pcm is not None else None,"cases":cases,"note":"FFmpeg buffering affects first_audio_ms; inspect output WAVs for clicks/gaps. Byte equality is strict and may differ for filters with timing tails."}
    target=Path("reports/glados-lab/streaming-"+p["id"]+".json"); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(report,indent=2),encoding="utf8"); print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__": main()
