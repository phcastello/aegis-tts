#!/usr/bin/env python3
from pathlib import Path
import os, sys
ROOT=Path(__file__).resolve().parents[1]
# Keep the documented root-level command while guaranteeing that CPU DSP never
# falls back to the global Python interpreter.
DSP_COMMANDS={"analyze-sources","world-baselines","world-candidates","lpc-candidates","benchmark-latency","report","run-mvp","long-form","qwen-cold-run"}
if len(sys.argv)>1 and sys.argv[1] in DSP_COMMANDS:
    dsp=ROOT/"private/environments/dsp/bin/python"
    dsp_prefix=(ROOT/"private/environments/dsp").resolve()
    if dsp.exists() and Path(sys.prefix).resolve()!=dsp_prefix:
        os.execv(str(dsp),[str(dsp),str(Path(__file__).resolve()),*sys.argv[1:]])
sys.path.insert(0,str(ROOT/"src"))
from aegis_robotic_voice.cli import main
if __name__ == "__main__": main()
