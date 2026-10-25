"""Hybrid H2: the agent says WHAT and roughly WHERE, B1's render-and-compare says exactly where.

For every object of the agent's answer that is new or moved by the agent (no B1 object of the same category within
``same_mm``), or that lacks an orientation, fit the catalogue shape locally in the current frames (head + wrists,
robot self-masked) from the agent's xy: Nelder-Mead over (x, y, yaw) from several yaw starts (the agent's yaw and its
flip when given), over the candidate assets of the object's category in the task's vocabulary. Accept the best fit when
its silhouette loss is below ``max_loss`` and it stays within ``max_shift_mm`` of the agent's point; else keep the
agent's pose. Worker side (needs the perception extra + catalogue + GL or the CPU silhouettes).
"""
from __future__ import annotations

import math

import numpy as np


def _cat(o: dict) -> str | None:
    a = str(o.get("asset") or "").split("/")
    return a[1] if len(a) == 3 else o.get("category")


def _extents(key: str):
    from ..render import load_asset

    try:
        return sorted(load_asset(key).info["extents"])
    except Exception:
        return None


def _candidates(o: dict, b1_objs: list[dict], task: str | None, k_max: int = 3) -> list[str]:
    if o.get("asset"):
        return [o["asset"]]
    keys = _candidates_all(o, b1_objs, task)
    size = o.get("size")
    if size and len(size) == 3 and len(keys) > k_max:  # the agent's measured size ranks the shapes
        sz = sorted(float(v) for v in size)

        def d(k):
            e = _extents(k)
            return 9.0 if e is None else sum(abs(a - b) / max(b, 1e-3) for a, b in zip(sz, e))
        keys = sorted(keys, key=d)
    return keys[:k_max]


def _candidates_all(o: dict, b1_objs: list[dict], task: str | None) -> list[str]:
    cat = _cat(o)
    lab = str(o.get("label") or "")
    keys = []
    for b in b1_objs:  # B1's own reading of the same label / category
        if b.get("asset") and (b.get("label") == lab or (cat and _cat(b) == cat)) and b["asset"] not in keys:
            keys.append(b["asset"])
    if task:
        from ..inverse_layout import label_groups

        for g in label_groups(task):
            for k in g["candidates"]:
                c = k.split("/")[1]
                if (cat and c == cat) or (lab and (lab in g["labels"] or c in lab.lower())):
                    if k not in keys:
                        keys.append(k)
    return keys


def seeded_refine(images: dict, state: dict, cams: dict, task: str | None, objects: list[dict], b1_objs: list[dict],
                  bgs: dict, same_mm: float = 8.0, max_shift_mm: float = 40.0, max_loss: float = 0.5,
                  n_yaw: int = 8, orient_loss: float = 0.75, orient_shift_mm: float = 15.0) -> tuple[list[dict], list[dict]]:
    """-> (objects with refined poses where accepted, per-object info)."""
    from ..cameras import R_to_quat_wxyz, pose_to_T
    from ..inverse_layout import EstimatorConfig, _views_subset, refine as nm_refine, shape_models
    from ..segment import segment_views

    cfg = EstimatorConfig()
    views = _views_subset(segment_views(images, cams, state, bgs, thr=cfg.fg_thr), cfg)
    out, infos = [dict(o) for o in objects], []

    def todo(o):
        """New or moved by the agent: no B1 object (any label) within same_mm. Orientation-less objects that sit on
        a B1 object take that object's asset and pose (the agent re-labelled it, it did not move it)."""
        for b in b1_objs:
            if math.hypot(b["pos"][0] - o["pos"][0], b["pos"][1] - o["pos"][1]) * 1000 <= same_mm:
                if not o.get("quat") and b.get("quat"):
                    o.update({k: b[k] for k in ("pos", "quat", "asset", "size", "top_z") if b.get(k) is not None})
                return False
        return True

    shapes = {}

    def sm_of(o):
        k = o.get("asset")
        if not k or not o.get("quat"):
            return None
        try:
            T = pose_to_T(o["pos"], o["quat"])
            from ..refine import _shape_for

            return _shape_for(k, T[:3, :3], None), T
        except Exception:
            return None

    for i, o in enumerate(out):
        if not todo(o):
            continue
        info = {"i": i, "label": o.get("label"), "accepted": False}
        others = {}
        for name, v in views.items():
            m = np.zeros((v.cam.h, v.cam.w), np.uint8)
            for j, o2 in enumerate(out):
                if j == i:
                    continue
                s = shapes.get(j) or sm_of(o2)
                if s is not None:
                    shapes[j] = s
                    m |= s[0].sil(v.cam, s[1])
            others[name] = m
        x0, y0 = float(o["pos"][0]), float(o["pos"][1])
        best = None
        for key in _candidates(o, b1_objs, task):
            try:
                sms = shape_models(key)
            except Exception as e:
                info.setdefault("errors", []).append(f"{key}: {type(e).__name__}")
                continue
            for sm in sms:
                per = sm.yaw_period
                if o.get("quat"):
                    w, _, _, z = [float(v) for v in o["quat"]]
                    y_world = 2 * math.atan2(z, w)
                    yaws = [y_world, y_world + math.pi] if per > math.pi + 1e-6 else [y_world]
                else:
                    yaws = [k * per / n_yaw for k in range(n_yaw)] if per > 0.2 else [0.0]
                # the seed is the object's centre; the shape's origin can sit off it
                for yaw in yaws:
                    c_off = sm.pose(0, 0, yaw)[:3, :3] @ ((sm.asset.V.min(0) + sm.asset.V.max(0)) / 2)
                    try:
                        x, y, yy, loss, _ = nm_refine(sm, x0 - c_off[0], y0 - c_off[1], yaw, views, others, xy_step=0.006,
                                                      yaw_step=0.15, maxiter=150, fix_yaw=per < 0.2)
                    except Exception as e:
                        info.setdefault("errors", []).append(f"{key}: {type(e).__name__}: {e}")
                        continue
                    if best is None or loss < best[0]:
                        best = (loss, key, sm, x, y, yy)
        if best is not None:
            loss, key, sm, x, y, yy = best
            T = sm.pose(x, y, yy)
            c = T[:3, :3] @ ((sm.asset.V.min(0) + sm.asset.V.max(0)) / 2) + T[:3, 3]
            shift = math.hypot(c[0] - x0, c[1] - y0) * 1000
            info.update(loss=round(float(loss), 3), key=key, shift_mm=round(shift, 1))
            if loss < max_loss and shift < max_shift_mm:
                V = sm.asset.V @ sm.R0.T
                o.update(pos=[round(float(v), 5) for v in T[:3, 3]], quat=[round(float(v), 6) for v in R_to_quat_wxyz(T[:3, :3])],
                         asset=key, category=key.split("/")[1], size=np.round(V.max(0) - V.min(0), 4).tolist(),
                         top_z=round(float((sm.asset.V @ T[:3, :3].T + T[:3, 3])[:, 2].max()), 4),
                         confidence=round(float(max(0.05, min(0.99, 1 - loss))), 3), source="refined")
                shapes[i] = (sm, T)
                info["accepted"] = True
            elif loss < orient_loss and shift < orient_shift_mm and not o.get("quat"):
                # a poor fit that agrees with the agent's point: take its shape and yaw, keep the agent's xy
                c_off = T[:3, :3] @ ((sm.asset.V.min(0) + sm.asset.V.max(0)) / 2)
                T2 = sm.pose(x0 - c_off[0], y0 - c_off[1], yy)
                V = sm.asset.V @ sm.R0.T
                o.update(pos=[round(float(v), 5) for v in T2[:3, 3]], quat=[round(float(v), 6) for v in R_to_quat_wxyz(T2[:3, :3])],
                         asset=key, category=key.split("/")[1], size=np.round(V.max(0) - V.min(0), 4).tolist(),
                         confidence=round(float(min(0.5, 1 - loss)), 3), source="refined")
                shapes[i] = (sm, T2)
                info["accepted"] = "orientation_only"
        infos.append(info)
    return out, infos
