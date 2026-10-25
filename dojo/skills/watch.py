"""Watch reads: what a skill that waits on the scene (a scripted opponent, a demonstrator) asks the provider while it
holds still. One dict shape for both sources: perception's ``watch_read(frame, region, vocabulary, ...)`` at eval
(integration-2 wires it behind the runtime provider) and ``gt_watch_read`` here for G3 (dev-only ground truth).

    region {"kind": "cells", "board": {"asset", "pos", "quat"}, "pieces": {"own": key, "opponent": key}}
        -> "cells": [{"k", "occupant": "own" | "opponent" | None, "conf", "pos"}]
    region {"kind": "container", "container": {"asset", "pos", "quat"}, "candidates": [keys]}
        -> "inside": [{"asset", "pos", "conf"}], "outside": [...]
    always: "t_ms", "visible_frac", "occluded", "stable_vs_prior", "new"

An occluded read (``occluded`` or ``visible_frac`` < 0.5) counts toward neither arrival nor stability.
"""

from __future__ import annotations

import time

import numpy as np

from dojo.skills.geometry import quat_to_R


def _cells_of(board_obj) -> list[tuple[int, np.ndarray]]:
    from dojo.skills.interactive import board_cells
    return [(k, p) for k, (_, p) in enumerate(board_cells(board_obj))]


def region_cells(board) -> dict:
    return {"asset": board.asset.key, "pos": np.asarray(board.pos, float).tolist(),
            "quat": np.asarray(board.quat, float).tolist()}


def gt_watch_read(objs: dict, region: dict, prior: dict | None = None, step: int = 0, cell_tol: float = 0.025,
                  dz=(0.0, 0.03)) -> dict:
    """The watch read from GT objects ({label: ObjectState}); see the module doc for the shapes."""
    t0 = time.perf_counter()
    out = {"visible_frac": 1.0, "occluded": False, "step": step}
    if region["kind"] == "cells":
        b = next((o for o in objs.values() if o.asset.key == region["board"]["asset"]), None)
        cells = []
        if b is not None:
            own, opp = region["pieces"]["own"], region["pieces"]["opponent"]
            pcs = [o for o in objs.values() if o.asset.key in (own, opp)]
            for k, c in _cells_of(b):
                occ, pos = None, None
                for o in pcs:
                    h = float(o.pos[2] - b.pos[2])
                    if dz[0] < h < dz[1] and float(np.linalg.norm(o.pos[:2] - c[:2])) <= cell_tol:
                        occ, pos = ("own" if o.asset.key == own else "opponent"), o.pos
                        break
                cells.append({"k": k, "occupant": occ, "conf": 1.0, "pos": None if pos is None else np.round(pos, 4).tolist()})
        out["cells"] = cells
        prev = {c["k"]: c for c in (prior or {}).get("cells", [])}
        out["new"] = [c["k"] for c in cells if c["occupant"] and (prev.get(c["k"]) or {}).get("occupant") != c["occupant"]]
        out["stable_vs_prior"] = bool(prior) and not out["new"] and all(
            c["pos"] is None or prev.get(c["k"], {}).get("pos") is None
            or float(np.linalg.norm(np.subtract(c["pos"], prev[c["k"]]["pos"]))) < 5e-4 for c in cells)
    else:
        C = region["container"]
        R = quat_to_R(np.asarray(C["quat"], float))
        cont = next((o for o in objs.values() if o.asset.key == C["asset"]
                     and float(np.linalg.norm(o.pos - np.asarray(C["pos"], float))) < 0.05), None)
        mn, mx = (cont.asset.bbox_min, cont.asset.bbox_max) if cont is not None else (np.full(3, -0.1), np.full(3, 0.1))
        inside, outside = [], []
        for o in objs.values():
            if o.asset.key not in region["candidates"] or o is cont:
                continue
            P = (o.pos - np.asarray(C["pos"], float)) @ R
            ins = bool(mn[0] <= P[0] <= mx[0] and mn[1] <= P[1] <= mx[1] and P[2] > mn[2])
            (inside if ins else outside).append({"asset": o.asset.key, "pos": np.round(o.pos, 4).tolist(), "conf": 1.0})
        out["inside"], out["outside"] = inside, outside
        prev_in = {(d["asset"]) for d in (prior or {}).get("inside", [])}
        out["new"] = [d["asset"] for d in inside if d["asset"] not in prev_in]
        # stable: every item has a prior item of the same asset within 0.5 mm (identical twins share an asset key)
        prev = (prior or {}).get("inside", []) + (prior or {}).get("outside", [])

        def near(d):
            return any(p["asset"] == d["asset"] and float(np.linalg.norm(np.subtract(d["pos"], p["pos"]))) < 5e-4
                       for p in prev)
        out["stable_vs_prior"] = bool(prior) and all(near(d) for d in inside + outside)
    out["t_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out


def usable(read: dict) -> bool:
    return not read.get("occluded") and float(read.get("visible_frac", 1.0)) >= 0.5


class ArrivalOrder:
    """Arrival order of objects into a container from successive container reads (perception-R&D's helper, same API):
    an asset counts once it has been inside for ``n_consecutive`` usable reads in a row."""

    def __init__(self, n_consecutive: int = 4, expected: int | None = None):
        self.n = int(n_consecutive)
        self.expected = expected
        self.order: list[str] = []
        self.count: dict[str, int] = {}

    def update(self, read: dict) -> dict:
        if usable(read):
            inside = {d["asset"] for d in read.get("inside", [])}
            for a in list(self.count) + list(inside):
                self.count[a] = self.count.get(a, 0) + 1 if a in inside else 0
            for a in sorted(inside, key=lambda a: -self.count.get(a, 0)):
                if self.count.get(a, 0) >= self.n and a not in self.order:
                    self.order.append(a)
        pending = [a for a, c in self.count.items() if 0 < c < self.n and a not in self.order]
        done = self.expected is not None and len(self.order) >= self.expected
        return {"order": list(self.order), "pending": pending, "done": done}


def watch(sk, region: dict, prior: dict | None = None) -> dict:
    """The provider's watch read (perception at eval, GT in G3). A provider without one: the same read computed from
    its refreshed world (the runtime's belief, with whatever its watcher wrote into it)."""
    f = getattr(sk.provider, "watch_read", None)
    if f is not None:
        return f(region, prior)
    w = sk.refresh()
    return gt_watch_read(w.objects, region, prior, step=int(w.step))


__all__ = ["gt_watch_read", "ArrivalOrder", "watch", "usable", "region_cells"]
