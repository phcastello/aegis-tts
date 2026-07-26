from __future__ import annotations
import numpy as np
def voiced_regions(mask: np.ndarray):
    starts=np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int))==1); ends=np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int))==-1)
    return list(zip(starts, ends))
