"""Bounded proposal-only escape from shallow initial proxy contact."""
from __future__ import annotations

import time
import numpy as np


def repair_coordinates(q0, limits, evaluate, penetrations, budget_s=3.):
    """Return a monotone escape, never a partial executable motion.

    No initially clear sphere/obstacle pair may develop contact; no pair's
    penetration may increase (10 micrometres numerical slack). Reject self-contact and starts
    deeper than 15 mm. The caller still validates the entire joined trajectory.
    """
    started=time.perf_counter();q0=np.asarray(q0,float);q=q0.copy()
    limits=np.asarray(limits);low=np.maximum(limits[:,0],q0-.3);high=np.minimum(limits[:,1],q0+.3)
    depth=penetrations(q);initial=float(np.max(depth,initial=0.))
    info={"initial_depth_mm":round(initial*1000,5),"depths_mm":[round(initial*1000,5)]}
    if initial>.015:
        return None,info|{"reason":"start_depth_limit"}
    path=[q.copy()]
    for iteration in range(12):
        if time.perf_counter()-started>=budget_s:
            return None,info|{"reason":"repair_budget"}
        metrics=evaluate(q)
        if metrics.get("self_collision",0)>0:
            return None,info|{"reason":"start_self_collision"}
        if metrics.get("scene_collision",0)<=0:
            return np.asarray(path),info|{"reason":"repaired","iterations":iteration}
        gradient=np.zeros(len(q))
        for j in range(len(q)):
            plus=q.copy();minus=q.copy()
            plus[j]=min(high[j],q[j]+.005);minus[j]=max(low[j],q[j]-.005)
            span=plus[j]-minus[j]
            if span>0:
                gradient[j]=(evaluate(plus).get("scene_collision",0)-evaluate(minus).get("scene_collision",0))/span
        norm=np.max(np.abs(gradient))
        if not np.isfinite(norm) or norm<1e-10:
            break
        direction=-.03*gradient/norm
        moved=False
        for scale in (1.,.5,.25):
            candidate=np.clip(q+scale*direction,low,high)
            candidate_depth=penetrations(candidate)
            if np.any(candidate_depth>depth+1e-5):
                continue
            other=evaluate(candidate)
            if other.get("self_collision",0)>0 or other.get("scene_collision",0)>=metrics["scene_collision"]:
                continue
            q=candidate;depth=candidate_depth;path.append(q.copy());moved=True
            info["depths_mm"].append(round(float(np.max(depth,initial=0.))*1000,5))
            break
        if not moved:
            break
    return None,info|{"reason":"no_monotone_escape"}


def check_escape_prefix(path, evaluate, penetrations):
    """A CPU-proposed retract must also respect shallow GPU contact escape."""
    previous=None
    for q in path:
        depth=penetrations(q)
        if float(np.max(depth,initial=0.))>.015:
            return False,"prefix_depth_limit"
        if previous is not None and np.any(depth>previous+1e-5):
            return False,"prefix_worsens_contact"
        if evaluate(q).get("self_collision",0)>0:
            return False,"prefix_self_collision"
        previous=depth
    return True,"checked_prefix"
