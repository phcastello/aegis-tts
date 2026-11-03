"""Reproduce filesystem-check cost with stdlib only; no model or CUDA imports."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import types

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/var/tmp/aegis-v2-work')


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',type=Path,default=ROOT/'voxcpm-gateway/step400-original-nfc/runtime.private.json')
 p.add_argument('--output',type=Path,default=ROOT/'voxcpm-gateway/preparation/cache-guard-cost-cpu.json')
 p.add_argument('--repetitions',type=int,default=20)
 a=p.parse_args();os.nice(19);sys.path.insert(0,str(REPO/'gateway'))
 from voxcpm_runtime import VoxConfig,NativeSession,CacheGuard
 if 'torch' in sys.modules:raise RuntimeError('Use a fresh stdlib-only process')
 fake=types.ModuleType('torch');fake.__path__=[]
 fake.cuda=types.SimpleNamespace(memory_allocated=lambda:0,memory_reserved=lambda:0,
  max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)
 mods={'torch':fake}
 for name in ('torch._dynamo','torch._inductor'):
  mods[name]=types.ModuleType(name);mods[name].__path__=[]
 mods['torch._dynamo.utils']=types.ModuleType('torch._dynamo.utils');mods['torch._dynamo.utils'].counters={'stats':{'unique_graphs':0}}
 mods['torch._inductor.metrics']=types.ModuleType('torch._inductor.metrics');mods['torch._inductor.metrics'].generated_kernel_count=0
 sys.modules.update(mods)
 cfg=VoxConfig.read(a.config);session=NativeSession(cfg);session.torch=fake;guard=CacheGuard(cfg);session.cache_guard=guard
 startup=time.perf_counter();guard.start();guard.check_now();startup=time.perf_counter()-startup
 rows=[]
 try:
  for _ in range(a.repetitions):
   before=time.perf_counter();session.check_budget();session.observe();session.check_budget();session.observe()
   rows.append(time.perf_counter()-before)
  snapshot=guard.snapshot()
 finally:guard.close()
 result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256=hashlib.sha256((REPO/'gateway/voxcpm_runtime.py').read_bytes()).hexdigest(),
  config=str(a.config),method='same two checks+two observations with periodic snapshot, zero-cost GPU allocator stubs, real /proc',
  synchronized_startup_checks_seconds=startup,snapshot_pattern_seconds=rows,
  p50_seconds=statistics.median(rows),max_seconds=max(rows),guard_snapshot=snapshot,
  actual_torch_imported=False,model_constructed=False,gpu_used=False,niceness=os.nice(0),
  limitation='CPU overhead only; does not establish HTTP TTFA improvement. ASR/training may run concurrently.')
 a.output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 a.output.write_text(json.dumps(result,indent=2)+'\n');a.output.chmod(0o600)
 print(json.dumps({k:result[k] for k in ['p50_seconds','max_seconds','synchronized_startup_checks_seconds','actual_torch_imported']}))


if __name__=='__main__':main()
