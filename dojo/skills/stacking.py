"""Stack N objects (blocks, bowls): choose the site and the arms, then run the jobs.

Site options, cheapest first:
* an existing object stays as the base: every other object must be pickable by some arm and that
  arm must reach the site at the stack's final height (pre-place pose, top-down);
* a site in the region both arms reach: one object is moved there first.
Arms alternate when possible so ``Skills.run_jobs`` can pick with one arm while the other places.
"""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np

from dojo.skills.kinematics import ARMS
from dojo.skills.targets import table_rest
from dojo.skills.skills import Job, Skills

MID_SITES = ((0.0, -0.17), (0.0, -0.12), (0.0, -0.22), (-0.06, -0.17), (0.06, -0.17))


COMFORT_REACH = 0.42          # horizontal distance from the arm base beyond which grasps get tilted / extended
MAX_REACH = 0.52              # no stack plan may need a pick farther than this


def _assign(order, cand, reach=None) -> list[str]:
    """Arms per job: alternate (so one arm picks while the other places) only between arms that
    reach the object comfortably; otherwise the nearer arm (far tilted grasps slipped in the twin)."""
    arms, prev = [], None
    for l in order:
        opts = list(cand[l])
        if reach is not None:
            comfy = [a for a in opts if reach[(l, a)] <= COMFORT_REACH]
            if comfy:
                opts = comfy
            else:
                opts = sorted(opts, key=lambda a: reach[(l, a)])[:1]
        pick = next((a for a in opts if a != prev), opts[0])
        arms.append(pick)
        prev = pick
    return arms


def plan_stack(sk: Skills, labels: list[str], top_h: float, on_top: Callable, yaw_sym: int,
               tol_xy: float = 0.008, pick_method: str = "auto", place_kw: dict | None = None,
               mid_sites=MID_SITES) -> list[Job]:
    """``on_top(below, label) -> target fn``; ``top_h``: TCP height of the last pre-place pose."""
    w = sk.world

    def arms_for(label, site_xy) -> list[str]:
        return [a for a in ARMS if sk.reach_down(a, (site_xy[0], site_xy[1], top_h), tilts=(0.0,))
                and sk.can_pick(label, a, pick_method)]

    reach = {(l, a): float(np.linalg.norm(w.obj(l).center_w[:2] - sk.kin.base(a)[:2, 3])) for l in labels for a in ARMS}
    best = None
    for base in labels:
        site = w.obj(base).center_w[:2]
        others = [l for l in labels if l != base]
        cand = {l: arms_for(l, site) for l in others}
        if all(cand[l] for l in others):
            for order in itertools.permutations(others):
                arms = _assign(order, cand, reach)
                if any(reach[(l, a)] > MAX_REACH for l, a in zip(order, arms)):
                    continue
                cost = sum(float(np.linalg.norm(w.obj(l).center_w[:2] - site)) for l in others)
                cost += sum(max(0.0, reach[(l, a)] - COMFORT_REACH) * 3 for l, a in zip(order, arms))
                cost += 0.15 * sum(a == b for a, b in zip(arms[:-1], arms[1:]))
                if best is None or cost < best[0]:
                    best = (cost, base, None, list(zip(order, arms)))
    # a shared site in the middle always competes (an existing base far to one side forces the
    # other arm into 0.5-0.6 m tilted picks that stall mid-lift: twin, stack_blocks v3)
    for xy in mid_sites:
        cand = {l: arms_for(l, xy) for l in labels}
        if not all(cand[l] for l in labels):
            continue
        for order in itertools.permutations(labels):
            arms = _assign(order, cand, reach)
            if any(reach[(l, a)] > MAX_REACH for l, a in zip(order, arms)):
                continue
            cost = sum(float(np.linalg.norm(w.obj(l).center_w[:2] - np.asarray(xy))) for l in labels) + 0.3
            cost += sum(max(0.0, reach[(l, a)] - COMFORT_REACH) * 3 for l, a in zip(order, arms))
            if best is None or cost < best[0]:
                best = (cost, None, xy, list(zip(order, arms)))
    if best is None:
        return []
    _, base, xy, seq = best
    jobs, below = [], base
    stacked = [base] if base else []
    for l, a in seq:
        if below is None:
            alts = [lambda s, l=l, xy2=xy2: table_rest(s, l, xy2) for xy2 in mid_sites if tuple(xy2) != tuple(xy)]
            jobs.append(Job(l, a, lambda s, l=l: table_rest(s, l, xy), yaw_sym=yaw_sym, tol_xy=0.05, bank=False,
                            required=False, method=pick_method, place_kw=dict(place_kw or {}), alt_targets=alts))
        else:
            jobs.append(Job(l, a, on_top(below, l), yaw_sym=yaw_sym, ignore=tuple(stacked), tol_xy=tol_xy,
                            method=pick_method, place_kw=dict(place_kw or {})))
        below = l
        stacked.append(l)
    return jobs
