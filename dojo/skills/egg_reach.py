"""Bounded egg endpoint search; all retained poses still need normal path checks."""
from __future__ import annotations

from contextlib import contextmanager
import json

import numpy as np

from dojo.skills.geometry import inv_T
from dojo.skills.kinematics import JOINT_LIMITS_MEASURED
from dojo.skills.skills import PLACE_REACH


@contextmanager
def practical_limits(kin):
    old = kin.limits
    limits = JOINT_LIMITS_MEASURED.copy()
    limits[3] = (-1.6, 1.6)
    limits[5] = (-3.0, 3.0)
    kin.limits = np.column_stack((np.maximum(old[:, 0], limits[:, 0]),
                                  np.minimum(old[:, 1], limits[:, 1])))
    kin.__dict__.pop('_ik_cache', None)
    try:
        yield
    finally:
        kin.limits = old
        kin.__dict__.pop('_ik_cache', None)


def screen(sk, arm, candidates, poses, phase):
    """Keep candidates with IK at every endpoint, reporting bounded-search failures.

    This is an endpoint screen, never a collision or path feasibility certificate.
    ``poses`` yields each candidate's ordered TCP targets (grasp/pre/lift or drop/hover).
    """
    kept, rows = [], []
    for candidate in candidates:
        q = sk.ex.q[arm]
        row = {'distance_m': [], 'margin_rad': [], 'failed_at': None}
        for i, T in enumerate(poses(candidate)):
            distance = float(np.linalg.norm(T[:2, 3] - sk.kin.base(arm)[:2, 3]))
            row['distance_m'].append(round(distance, 5))
            if distance > PLACE_REACH:
                row['failed_at'] = i
                row['reason'] = 'reach'
                break
            r = sk.kin.ik(arm, T, q, n_random=1)
            if not r.ok:
                row.update(failed_at=i, reason='ik', residual_m=r.pos_err)
                break
            q = r.q
            row['margin_rad'].append(np.round(np.minimum(q - sk.kin.limits[:, 0],
                                                        sk.kin.limits[:, 1] - q), 5).tolist())
        if row['failed_at'] is None:
            kept.append(candidate)
        rows.append(row)
    sk.log('[egg-reach] ' + json.dumps(dict(phase=phase, arm=arm, total=len(rows), kept=len(kept), rows=rows)))
    return kept


def grasp_candidates(sk, lab, arm, grasps, lift_z):
    def poses(g):
        lift = g.T.copy()
        lift[2, 3] = max(float(g.T[2, 3]) + .06, lift_z or 0.)
        return (g.T, g.pre(), lift)
    return screen(sk, arm, grasps, poses, 'pick:' + lab)


def drop_candidates(sk, arm, candidates, pre_h=.05):
    h = sk.held[arm]
    rel = inv_T(h.T_tcp_obj)
    def poses(c):
        tcp = c[-1] @ rel
        pre = tcp.copy()
        pre[2, 3] += pre_h
        return (tcp, pre)
    return screen(sk, arm, candidates, poses, 'drop:' + h.label)
