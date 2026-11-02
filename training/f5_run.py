"""Run every authorized CUDA phase before the operation guard restores Qwen."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);args=p.parse_args()
    root=args.run_root;os.umask(0o077)
    def stage(name,command,allow_failure=False):
        start=time.monotonic()
        with (root/(name+'.log')).open('ab') as f:
            code=subprocess.call(command,stdout=f,stderr=subprocess.STDOUT)
        with (root/'pipeline.jsonl').open('a') as f:
            f.write(json.dumps({'stage':name,'command':command,'returncode':code,'seconds':time.monotonic()-start})+'\n')
        if code and not allow_failure:raise subprocess.CalledProcessError(code,command)
        return code
    base=[sys.executable,'-m','training.f5_engine','--run-root',str(root),'--cuda-authorized']
    # Two real effective-batch configurations, with identical batch1/mixed precision.
    for accumulation in (4,8):
        code=stage(f'benchmark-bf16-a{accumulation}',base+['--action','benchmark','--accumulation',str(accumulation)],allow_failure=True)
        if code:
            text=(root/f'benchmark-bf16-a{accumulation}.log').read_text()
            if 'out of memory' in text.lower():
                # Checkpointed batch1 is already the smallest complete-utterance
                # microbatch. No clip is silently cropped/relabelled to fit.
                raise RuntimeError('checkpointed batch1 OOM; complete-utterance training not viable under tested configuration; see real failure log')
            raise RuntimeError('real CUDA benchmark failed; inspect preserved log')
    stage('train',base+['--action','train','--accumulation','8'])
    stage('final-inference',[sys.executable,'-m','training.f5_evaluate','--run-root',str(root),'--action','final'])
    # CPU ASR, packaging and plots run after all CUDA operations and production
    # restoration; keeping the outage free of unnecessary CPU postprocessing.
    (root/'gpu-all-phases-complete.json').write_text(json.dumps({'completed':True,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}))
if __name__=='__main__':main()
