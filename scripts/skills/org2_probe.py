"""ORG2 dev probe (CPU, no Isaac): organize-table planning feasibility on settled layouts.

    SIMLAB_GPU=0 scripts/simlab/rdpy.sh scripts/skills/org2_probe.py LAYOUT_DIR 100000 100001 ... [--attr k=v ...]

A kinematic world (``KinematicEnv``) built from each settled record, the GT provider, the arrange route's obstacles.
Reports, per layout: the keyboard's push stroke sequence (simulated kinematically: each planned stroke moves the
keyboard by its step) under the skill attributes given, and the figurine's transfer screen. Dev only: it reads the
layout's true poses; nothing here is a policy input."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from dojo.simlab import paths
from dojo.skills import library as L
from dojo.skills.env import KinematicEnv
from dojo.skills.skills import GTProvider, SkillResult, Skills
from dojo.skills.world import AssetDB


def rows_of(lay: dict) -> list[dict]:
    rows = []
    for typ in ("Geometry", "Rigid", "Articulation"):
        for _cat, recs in lay.get(typ, {}).items():
            for r in recs:
                rows.append(dict(label=r["label"], category=r["category"], category_idx=r["category_idx"], type=typ,
                                 pos=r["default_pos"], quat=r["default_ori"], scale=r.get("scale", [1, 1, 1])))
    return rows


def build(lay: dict, attrs: dict) -> tuple[KinematicEnv, Skills]:
    e = KinematicEnv(rows_of(lay))
    sk = Skills(e, GTProvider(e, AssetDB(paths.OBJECTS)))
    for k, v in attrs.items():
        setattr(sk, k, v)
    return e, sk


def move_row(e: KinematicEnv, label: str, dxy) -> None:
    o = e._obj(label)
    o["pos"] = o["pos"] + np.r_[np.asarray(dxy, float), 0.0]


def probe_push(e, sk, label="keyboard", support="frame", max_strokes=12) -> dict:
    """Simulate ``push_to``'s stroke choice kinematically (no physics): plan, move by the planned step, repeat."""
    out = {"strokes": [], "end": None, "why": None}
    w = sk.refresh()
    if getattr(sk, "push_plan", "off") != "off" or getattr(sk, "push_contact", "off") != "off":
        from dojo.skills import push_plan as PP
        keep = [(w.obj("cube_cushion").center_w[:2], 0.5 * float(np.max(w.obj("garage").asset.extents[:2])))]
        goal, diag = PP.clear_goal(sk, label, support, keep)
        out["goal"] = diag
        pc = getattr(sk, "push_contact", "off")
        fn = (PP.pair_stroke if pc == "pair" else (lambda s_, o_, l_, u_, st_: L._push_stroke(s_, o_, l_, u_, st_)))
        t0 = time.perf_counter()
        for _ in range(max_strokes):
            w = sk.refresh()
            o = w.obj(label)
            if float(np.linalg.norm(goal - o.center_w[:2])) <= 0.02:
                break
            axes = PP.support_axes(w.obj(support)) if pc == "pair" else PP.body_axes(o)
            seq = PP.plan_sequence(sk, label, goal, axes, fn)
            if seq is None:
                out["why"] = f"no sequence at {np.round(o.center_w[:2], 3).tolist()}"
                break
            u, st, arm = seq["strokes"][0]
            out["strokes"].append([arm, np.round(u, 2).tolist(), round(st * 1000), seq["order"]])
            move_row(e, label, np.asarray(u) * st)
        out["s"] = round(time.perf_counter() - t0, 1)
        w = sk.refresh()
        out["end"] = np.round(w.obj(label).center_w[:2], 3).tolist()
        out["covers"] = PP.covers(w.obj(label), w.obj(support))
        return out
    tgt = w.obj(support).center_w[:2]
    for _ in range(max_strokes):
        w = sk.refresh()
        o = w.obj(label)
        d = tgt - o.center_w[:2]
        if float(np.linalg.norm(d)) <= 0.012:
            break
        axes = []
        for k in range(2):
            a = o.R[:, k].copy()
            a[2] = 0
            a /= np.linalg.norm(a)
            axes.append((abs(float(a[:2] @ d)), k, a[:2] * np.sign(float(a[:2] @ d))))
        axes.sort(key=lambda x: -x[0])
        done = False
        for comp, k, u in axes:
            if comp < 0.006:
                continue
            step = min(0.08, comp + 0.003)
            p = L._push_stroke(sk, o, label, u, step)
            if p is not None:
                out["strokes"].append([p.arm, k, round(step * 1000)])
                move_row(e, label, u * step)
                done = True
                break
        if not done:
            out["why"] = f"no stroke at {np.round(o.center_w[:2], 3).tolist()}"
            break
    out["end"] = np.round(sk.refresh().obj(label).center_w[:2], 3).tolist()
    return out


def probe_place(sk, label="garage", support="cube_cushion", place_kw=None) -> dict:
    """The figurine's transfer screen as ``arrange`` plans it (container obstacles, ``on_targets``), with the
    alarm / mouse already moved off the table when ``--after`` asks."""
    L.container_obstacle(sk, "drawer")
    L.container_obstacle(sk, support)
    t0 = time.perf_counter()
    tg = L.on_targets(sk, label, support)
    r = sk.plan_transfer(label, tg, ignore=(support,), place_kw=place_kw or {"pre_h": 0.04})
    res = {"targets": len(tg), "s": round(time.perf_counter() - t0, 1)}
    if isinstance(r, SkillResult):
        res.update(status=r.status, note=r.note[:300])
    else:
        res.update(status="plan", arm=r[0].arm, grasp=r[0].grasp.note)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("layouts")
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--attr", action="append", default=[])
    ap.add_argument("--what", default="push,place")
    a = ap.parse_args()
    attrs = dict(x.split("=", 1) for x in a.attr)
    for i in a.ids:
        p = Path(a.layouts) / f"{i}.settled.json"
        if not p.exists():
            p = Path(a.layouts) / f"{i}.json"
        lay = json.loads(p.read_text())
        rec = {"id": i, "garage_idx": lay["Rigid"]["garage"][0]["category_idx"],
               "keyboard_idx": lay["Rigid"]["keyboard"][0]["category_idx"]}
        for what in a.what.split(","):
            e, sk = build(lay, attrs)
            if what == "push":
                rec["push"] = probe_push(e, sk)
            elif what == "place":
                rec["place"] = probe_place(sk)
        print(json.dumps(rec, default=str), flush=True)


if __name__ == "__main__":
    sys.exit(main())
