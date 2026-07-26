#!/usr/bin/env python3
"""Private JSON-lines worker. It never opens a network listener."""
from __future__ import annotations
from pathlib import Path
import contextlib, json, sys, traceback
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from aegis_robotic_voice.config import settings
from aegis_robotic_voice.source.chatterbox_ptbr import RegionalChatterbox
from aegis_robotic_voice.audio import sha256

def main():
    import torch, soundfile as sf
    from time import perf_counter
    cfg=settings(); started=perf_counter()
    # Third-party Chatterbox/Perth modules print progress messages. stdout is a
    # strict JSONL protocol, so keep all library chatter on stderr.
    with contextlib.redirect_stdout(sys.stderr): model=RegionalChatterbox.load(cfg)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    cold_load_seconds=perf_counter()-started
    print(json.dumps({"status":"ready","identity":model.identity,"model_load_seconds":cold_load_seconds,"peak_vram_mib":(torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0)}),flush=True)
    for raw in sys.stdin:
        try:
            request=json.loads(raw); assert request.get("op")=="generate"; output=Path(request["output"]).resolve()
            allowed=cfg.results.resolve()
            if allowed not in output.parents: raise ValueError("output path is outside results")
            output.parent.mkdir(parents=True,exist_ok=True)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            started=perf_counter()
            with contextlib.redirect_stdout(sys.stderr): wav=model.generate(request["text"],audio_prompt=request.get("audio_prompt"),exaggeration=float(request["exaggeration"]),cfg_weight=float(request["cfg_weight"]),seed=int(request["seed"]))
            if torch.cuda.is_available(): torch.cuda.synchronize()
            elapsed=perf_counter()-started; a=wav.squeeze().detach().cpu().numpy(); sf.write(output,a,model.model.sr)
            print(json.dumps({"status":"complete","path":str(output),"sample_rate":model.model.sr,"duration_seconds":len(a)/model.model.sr,"warm_generation_seconds":elapsed,"rtf":elapsed/(len(a)/model.model.sr),"peak_vram_mib":(torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0),"sha256":sha256(output)}),flush=True)
        except Exception as e: print(json.dumps({"status":"failed","error":str(e),"traceback":traceback.format_exc(limit=2)}),flush=True)
if __name__ == "__main__": main()
