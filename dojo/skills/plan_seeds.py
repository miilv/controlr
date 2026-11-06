"""Deterministic but distinct IK branch banks for successive attempts."""
import numpy as np


def ik_seed_bank(q0, hint, limits, count, attempt):
    bounds=np.asarray(limits,float)
    rng=np.random.default_rng(attempt)
    seeds=rng.uniform(bounds[:,0],bounds[:,1],size=(1,count,len(bounds)))
    if hint is not None:
        local=max(1,count//4)
        seeds[:,:local]=np.asarray(hint)+.04*rng.normal(size=(1,local,len(bounds)))
        seeds[:,0]=hint
    else:
        seeds[:,0]=q0
    return np.clip(seeds,bounds[:,0],bounds[:,1])
