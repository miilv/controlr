"""A physically consistent twin load from the belief (``RuntimeConfig.twin_load = "consistent"``).

The twin starts every verification from the provider's at-rest estimate. A belief a few mm off puts stacked pieces
into each other (or a hair above), PhysX pops them apart during the load's settle, and the twin's start state is no
longer the scene (2026-10-26: none_ok in 1/25 mode-A episodes vs 9/25 stack and 18/25 tower in mode B). Before the
layout is built, each movable object is made consistent with what it rests on:

* **level**: a body axis within ``level_deg`` of vertical is made exactly vertical (rotation about the object's
  bbox centre): a resting box or block sits flat on a flat support;
* **seat**: in order of height, an object whose bottom is within ``snap_mm`` of the highest support top below it (the
  table, or an object whose footprint overlaps its own) is put exactly on that top. Larger gaps or overlaps (nested
  bowls, a piece leaning on another, an object held off the table) are left as they are.

Tops and bottoms come from the asset mesh when the catalogue has it, else from the bbox. Footprints are world AABBs.
"""

from __future__ import annotations

import numpy as np

from dojo.skills.geometry import quat_to_R
from dojo.skills.kinematics import TABLE_Z
from dojo.skills.world import table_contains_xy


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix."""
    t = float(np.trace(R))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def _axis_angle_R(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _points(asset) -> np.ndarray:
    """Object-frame points that bound the asset: mesh vertices when known, else the 8 bbox corners."""
    if asset.mesh is not None and len(asset.mesh[0]):
        return np.asarray(asset.mesh[0], float)
    mn, mx = asset.bbox_min, asset.bbox_max
    return np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])], float)


def level(R: np.ndarray, level_deg: float) -> tuple[np.ndarray, float]:
    """(R with its most vertical body axis made exactly vertical, the correction in deg) when that axis is within
    ``level_deg`` of vertical; else (R, 0)."""
    z = np.array([0.0, 0.0, 1.0])
    best = None
    for i in range(3):
        for sgn in (1.0, -1.0):
            ax = sgn * R[:, i]
            ang = float(np.degrees(np.arccos(np.clip(ax @ z, -1.0, 1.0))))
            if best is None or ang < best[0]:
                best = (ang, ax)
    ang, ax = best
    if ang < 1e-6 or ang > level_deg:
        return R, 0.0
    return _axis_angle_R(np.cross(ax, z), np.radians(ang)) @ R, ang


def consistent(objects: list[dict], assets: dict, fixed: set | None = None, snap_mm: float = 6.0,
               level_deg: float = 3.0, min_overlap_m: float = 0.002) -> tuple[list[dict], dict]:
    """``objects`` (est_objects dicts: label, pos, quat wxyz, ...) made consistent; ``assets`` = {label: AssetInfo}
    (labels without one, and ``fixed`` labels, are supports only). Returns (new dicts, stats)."""
    fixed = set(fixed or ())
    st = {"leveled": [], "seated": []}
    items = []
    for o in objects:
        a = assets.get(o.get("label"))
        R = quat_to_R(np.asarray(o["quat"], float))
        p = np.asarray(o["pos"], float).copy()
        movable = a is not None and o.get("label") not in fixed
        if movable:
            R2, ang = level(R, level_deg)
            if ang > 0:
                c = R @ np.asarray(a.center, float) + p          # rotate about the bbox centre
                p = c - R2 @ np.asarray(a.center, float)
                R = R2
                st["leveled"].append((o["label"], round(ang, 2)))
        pts = (R @ _points(a).T).T + p if a is not None else None
        items.append({"o": o, "a": a, "R": R, "p": p, "pts": pts, "movable": movable,
                      "leveled": bool(st["leveled"]) and st["leveled"][-1][0] == o.get("label")})
    done = []
    for it in sorted(items, key=lambda it: float(it["pts"][:, 2].min()) if it["pts"] is not None else 1e9):
        if it["pts"] is None:
            continue
        lo, hi = it["pts"].min(0), it["pts"].max(0)
        if it["movable"]:
            sup = TABLE_Z if table_contains_xy((lo + hi) / 2) else None
            for d in done:
                dlo, dhi = d["pts"].min(0), d["pts"].max(0)
                ov = min(hi[0], dhi[0]) - max(lo[0], dlo[0]), min(hi[1], dhi[1]) - max(lo[1], dlo[1])
                if ov[0] > min_overlap_m and ov[1] > min_overlap_m and dhi[2] <= lo[2] + snap_mm / 1000.0:
                    sup = float(dhi[2]) if sup is None else max(sup, float(dhi[2]))
            if sup is not None and 1e-5 < abs(lo[2] - sup) <= snap_mm / 1000.0:
                dz = sup - float(lo[2])
                it["p"] = it["p"] + np.array([0.0, 0.0, dz])
                it["pts"] = it["pts"] + np.array([0.0, 0.0, dz])
                st["seated"].append((it["o"]["label"], round(dz * 1000, 1)))
        done.append(it)
    out = []
    for it in items:
        o2 = dict(it["o"])
        o2["pos"] = [float(v) for v in it["p"]]
        if it["leveled"]:
            o2["quat"] = [float(v) for v in _R_to_quat(it["R"])]
        out.append(o2)
    return out, st


__all__ = ["consistent", "level"]
