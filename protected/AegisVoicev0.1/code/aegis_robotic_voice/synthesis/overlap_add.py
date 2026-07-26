from __future__ import annotations
import numpy as np
def overlap_add(frames, hop, length):
    out=np.zeros(length); norm=np.zeros(length)
    for i,frame in enumerate(frames):
        a=i*hop; b=min(length,a+len(frame)); out[a:b]+=frame[:b-a]; norm[a:b]+=1
    return out/np.maximum(norm,1)
