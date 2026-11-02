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
        for attempt in range(3):
            start=time.monotonic()
            with (root/(name+'.log')).open('ab') as f:
                code=subprocess.call(command,stdout=f,stderr=subprocess.STDOUT)
            with (root/'pipeline.jsonl').open('a') as f:
                f.write(json.dumps({'stage':name,'attempt':attempt,'command':command,'returncode':code,'seconds':time.monotonic()-start})+'\n')
            if not code or allow_failure:return code
            text=(root/(name+'.log')).read_text()[-20000:]
            recoverable=any(kind in text for kind in ('ModuleNotFoundError:', 'ImportError:', 'NameError:', 'TypeError:', 'FileNotFoundError:'))
            if not recoverable or attempt>=2:raise subprocess.CalledProcessError(code,command)
            # A failed child has released CUDA. Keep the exclusive guard while
            # the supervising agent repairs a code/import issue. This grace is
            # ONLY for fault recovery, never a training duration limit.
            control=root/'recovery-action.json';control.unlink(missing_ok=True)
            needed=root/'recovery-needed.json'
            needed.write_text(json.dumps({'stage':name,'attempt':attempt,'returncode':code,'log':str(root/(name+'.log')),'status':'awaiting_agent_repair','fault_recovery_grace_seconds':900}))
            until=time.monotonic()+900
            while time.monotonic()<until:
                if control.exists():
                    action=json.loads(control.read_text()).get('action')
                    control.unlink()
                    if action=='retry':break
                    raise RuntimeError('agent ended unrecoverable experiment')
                time.sleep(2)
            else:raise RuntimeError('fault recovery unavailable; restore production')
            needed.unlink(missing_ok=True)
            if 'training.f5_engine' in command and 'train' in command and (root/'checkpoints/last.pt').exists() and '--resume' not in command:
                command=command+['--resume']
    base=[sys.executable,'-m','training.f5_engine','--run-root',str(root),'--cuda-authorized']
    # Two real effective-batch configurations, with identical batch1/mixed precision.
    for accumulation in (() if (root/'checkpoints/last.pt').exists() else (4,8)):
        code=stage(f'benchmark-bf16-a{accumulation}',base+['--action','benchmark','--accumulation',str(accumulation)],allow_failure=True)
        if code:
            text=(root/f'benchmark-bf16-a{accumulation}.log').read_text()
            if 'out of memory' in text.lower():
                # Checkpointed batch1 is already the smallest complete-utterance
                # microbatch. No clip is silently cropped/relabelled to fit.
                raise RuntimeError('checkpointed batch1 OOM; complete-utterance training not viable under tested configuration; see real failure log')
            raise RuntimeError('real CUDA benchmark failed; inspect preserved log')
    resume=(root/'checkpoints/last.pt').exists()
    if resume:
        stage('recovered-pilot-synthesis',[sys.executable,'-m','training.f5_evaluate','--run-root',str(root),'--action','checkpoint','--checkpoint',str(root/'checkpoints/pilot.safetensors'),'--label','epoch-003-recovered'])
    stage('train-resumed' if resume else 'train',base+['--action','train','--accumulation','8']+(['--resume'] if resume else []))
    stage('final-inference',[sys.executable,'-m','training.f5_evaluate','--run-root',str(root),'--action','final'])
    # CPU ASR, packaging and plots run after all CUDA operations and production
    # restoration; keeping the outage free of unnecessary CPU postprocessing.
    (root/'gpu-all-phases-complete.json').write_text(json.dumps({'completed':True,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}))
if __name__=='__main__':main()
