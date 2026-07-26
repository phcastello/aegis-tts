from __future__ import annotations
from time import perf_counter
from dataclasses import dataclass
import numpy as np
@dataclass
class Timer:
    started: float = 0.0
    def __enter__(self): self.started = perf_counter(); return self
    def __exit__(self, *_): self.seconds = perf_counter() - self.started
def stats(values):
    v=np.asarray(values, dtype=float)
    return {"min":float(v.min()),"median":float(np.median(v)),"p90":float(np.percentile(v,90)),"max":float(v.max()),"mean":float(v.mean()),"stddev":float(v.std())} if len(v) else {}
