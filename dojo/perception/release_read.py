"""The pose of an object right after the robot let it go (stream W): render-and-compare in 6 DoF around the released
pose, in every camera that sees it (the releasing arm's wrist, the head, the other wrist).

    from dojo.perception.release_read import release_read
    r = release_read(frame, "Rigid/mug/00004", released_pose7, "left", others=[(key, pose7), ...])
    r["ok"], r["pose7"], r["moved_mm"], r["turned_deg"], r["ious"]

Today the belief keeps the PLANNED place pose after a release (``post_place`` re-fits it only as a resting object:
x, y, yaw on its support). An object that does not come to rest the way it was put down is off by that: a mug let go
on a rack hook swings 14-38 degrees about the hook and settles 13-22 mm away (stream C's replays of hang_mugs); the
grasp-time metric then counts it as a neighbour "off" or a phantom at the next events.

Method (``held.held_pose``'s machinery without the gripper): views where enough of the predicted silhouette is visible;
robot pixels (FK, both arms) and known objects IN FRONT of the released one (z-buffer at the prior) are unknown; a
two-class Lab colour model per view learnt in the frame around the prior's silhouette (eroded by the expected release
error); the cost is the weighted soft-IoU loss over views; starts swing the object about horizontal axes through its
centre and shift it; Nelder-Mead in 6 DoF from the best starts, then the colour models are re-learnt at the fit and a
polish runs. ``ok``: the best view's IoU >= ``ok_iou`` and every used view's >= ``min_view_iou``, the releasing wrist among
the views or at least two views, the fit beats the prior's cost by ``min_gain``, inside the search range. A not-ok read leaves the prior (``pose7`` = the prior; ``fit_pose7`` = the rejected fit).
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .cameras import R_to_quat_wxyz, cameras_from_state, pose_to_T
from .held import Silhouettes, View, _iou, _roi_for, apply_delta, robot_unknown
from .render import gl, load_asset, silhouette, transform

CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
DEFAULTS = {
    "max_shift_mm": 50.0,          # search range around the released pose
    "max_turn_deg": 75.0,
    "prior_err_mm": 12.0,          # expected release error: the first object colour model is learnt this far inside
    "ring_mm": 8.0,                # background ring beyond it
    "ok_iou": 0.75,                # the best view's IoU at the fit ...
    "min_view_iou": 0.70,          # ... and every used view's (a view that disagrees vetoes: hang_mugs 34 -> 40 mm)
    "min_gain": 0.03,              # the fit's cost must beat the prior's by this much to move the object
    "min_vis_px": 150,             # visible silhouette pixels for a wrist view (the head: 40)
    "min_vis_frac": 0.25,
    "swing_deg": (-40.0, -20.0, 20.0, 40.0),   # starts: turns about world x / y through the centre
    "shift_mm": (-12.0, 12.0),                 # starts: shifts along world x / y
    "w_head": 0.6,
}


def _pose7(T) -> list:
    return [float(v) for v in T[:3, 3]] + [float(v) for v in R_to_quat_wxyz(T[:3, :3])]


def _front_occluders(cam, a, T, others, margin: float = 0.01) -> np.ndarray:
    """Pixels where a known other object lies in front of the object at ``T`` (depth test on GL depth renders; without
    GL, the others' full silhouettes)."""
    m = np.zeros((cam.h, cam.w), np.uint8)
    if not others:
        return m
    g = gl()
    if g is None:                                   # no depth: the others' silhouettes over the object's
        s_obj = silhouette(cam, transform(a.Vs, T), a.Fs)
        for k2, T2 in others:
            a2 = load_asset(k2)
            m |= silhouette(cam, transform(a2.Vs, T2), a2.Fs) & s_obj
        return m
    d_obj = g.render(cam, [(a.key, a.Vg, a.Fg, T)], depth=True)
    for k2, T2 in others:
        a2 = load_asset(k2)
        d2 = g.render(cam, [(k2, a2.Vg, a2.Fg, T2)], depth=True)
        # only over the object: elsewhere another object is background (a mousemat under a mouse is the ring the
        # background colour model learns from)
        m |= ((d2 > 0) & (d_obj > 0) & (d2 < d_obj - margin)).astype(np.uint8)
    return m


def release_read(frame: dict, key: str, prior_pose7, arm: str, others=(), cfg: dict | None = None) -> dict:
    """See the module doc. ``frame`` = {"images": {cam: HxWx3 uint8}, "state": official state}; ``others`` = [(key,
    pose7)] of the other believed objects (occluders when in front)."""
    t0 = time.time()
    c = {**DEFAULTS, **(cfg or {})}
    images = {k: np.ascontiguousarray(np.asarray(v, np.uint8)[..., :3]) for k, v in frame["images"].items()}
    state = {k: np.asarray(v, float).reshape(-1) for k, v in frame["state"].items()}
    T0 = pose_to_T(prior_pose7[:3], prior_pose7[3:])
    sil = Silhouettes(key)
    a = sil.a
    cams = cameras_from_state(state)
    oth = [(k2, pose_to_T(p2[:3], p2[3:])) for k2, p2 in others or ()]
    max_shift, max_turn = c["max_shift_mm"] / 1000, math.radians(c["max_turn_deg"])
    g_arm = state.get(f"{arm}_ee_joint_state")
    g_arm = 1.0 if g_arm is None else float(g_arm[0])
    other = "right" if arm == "left" else "left"
    g_oth = state.get(f"{other}_ee_joint_state")
    g_oth = None if g_oth is None else float(g_oth[0])
    bb = np.array([[x, y, z] for x in (a.info["bbox_min"][0], a.info["bbox_max"][0])
                   for y in (a.info["bbox_min"][1], a.info["bbox_max"][1])
                   for z in (a.info["bbox_min"][2], a.info["bbox_max"][2])])
    views, diag = [], {}
    for name in CAMS:
        if name not in images or name not in cams:
            continue
        cam = cams[name]
        role = "own" if name == f"cam_{arm}_wrist" else ("head" if name == "cam_head" else "observer")
        S = silhouette(cam, transform(a.Vs, T0), a.Fs)
        n = int(S.sum())
        if n < 30:
            diag[name] = {"used": False, "why": "not in view"}
            continue
        rob = robot_unknown(cam, state, arm, g_arm, g_arm, g_oth)
        occ = rob | cv2.dilate(_front_occluders(cam, a, T0, oth), np.ones((5, 5), np.uint8))
        vis = float((S & (1 - occ)).sum()) / n
        need = c["min_vis_px"] * (40.0 / 150.0 if role == "head" else 1.0)
        if vis < c["min_vis_frac"] or n * vis < need:
            diag[name] = {"used": False, "why": "occluded", "visible_frac": round(vis, 3), "px": n}
            continue
        roi = _roi_for(cam, transform(bb, T0), max_shift + 0.01)
        if roi is None:
            diag[name] = {"used": False, "why": "roi"}
            continue
        side = roi[2] - roi[0]
        size = int(np.clip(side * 3, 96, 320))
        views.append(View(name, role, cam, images[name], (1 - occ).astype(np.uint8), roi, size))
        diag[name] = {"used": True, "role": role, "visible_frac": round(vis, 3), "px": n, "roi_px": side}
    base = {"views": [v.name for v in views], "view_diag": diag}
    if not views:
        return {**base, "ok": False, "why": "no_views", "pose7": list(map(float, prior_pose7)),
                "time_s": round(time.time() - t0, 3)}
    w = np.array([c["w_head"] if v.role == "head" else 1.0 for v in views])
    ctr = (np.asarray(a.info["bbox_min"]) + np.asarray(a.info["bbox_max"])) / 2
    cw = T0[:3, :3] @ ctr + T0[:3, 3]                 # rotations about the body's centre (world axes)

    def cost(d, detail=False):
        T = apply_delta(T0, cw, d)
        tot, ious = 0.0, {}
        for v, wv in zip(views, w):
            if wv <= 0:
                continue
            iou = _iou(sil.render(v.cam, T), v)
            ious[v.name] = round(iou, 4)
            tot += wv * (1.0 - iou)
        tot /= max(float(w.sum()), 1e-9)
        sh, tr = float(np.linalg.norm(d[3:6])), float(np.linalg.norm(d[:3]))
        tot += 10.0 * (max(0.0, sh - max_shift) / 0.001 + max(0.0, tr - max_turn) / 0.01)
        return (tot, ious) if detail else tot

    def models(x, e_m, band_m) -> int:
        T = apply_delta(T0, cw, x)
        n_ok = 0
        for v in views:
            depth = float(np.median(v.cam.project(transform(a.Vs[::max(1, len(a.Vs) // 200)], T))[1]))
            px_m = v.cam.K[0, 0] / max(depth, 1e-3)
            n_ok += v.fit_models(sil.render(v.cam, T), e_m * px_m, max(6.0, band_m * px_m))
        return n_ok

    from scipy.optimize import minimize

    deg, mm = math.radians(1.0), 0.001

    def nm(x0, steps, iters):
        simplex = [np.asarray(x0, float)]
        for i, s in enumerate(steps):
            e = np.zeros(6)
            e[i] = s
            simplex.append(simplex[0] + e)
        r = minimize(cost, x0, method="Nelder-Mead",
                     options={"initial_simplex": np.array(simplex), "maxiter": iters, "xatol": 1e-5, "fatol": 1e-5})
        return r.x, float(r.fun)

    if models(np.zeros(6), 0.35 * c["prior_err_mm"] / 1000, c["ring_mm"] / 1000) == 0:
        return {**base, "ok": False, "why": "no_models", "pose7": list(map(float, prior_pose7)),
                "time_s": round(time.time() - t0, 3)}
    f_prior = cost(np.zeros(6))
    starts = [np.zeros(6)]
    for s in c["swing_deg"]:
        for ax in (0, 1):
            d = np.zeros(6)
            d[ax] = s * deg
            starts.append(d)
    for s in c["shift_mm"]:
        for ax in (3, 4):
            d = np.zeros(6)
            d[ax] = s * mm
            starts.append(d)
    scored = sorted(((cost(s), i, s) for i, s in enumerate(starts)), key=lambda t: t[:2])
    best = None
    for f0, _, s in scored[:3]:
        x, f = nm(s, [4 * deg, 4 * deg, 4 * deg, 3 * mm, 3 * mm, 3 * mm], 400)
        if best is None or f < best[1]:
            best = (x, f)
    models(best[0], 0.002, c["ring_mm"] / 1000)                 # re-learnt around the fit
    f_b = cost(best[0])
    x, f = nm(best[0], [1.5 * deg, 1.5 * deg, 1.5 * deg, 1 * mm, 1 * mm, 1 * mm], 400)
    if f > f_b:
        x, f = best[0], f_b
    f, ious = cost(x, detail=True)
    T = apply_delta(T0, cw, x)
    # the prior's cost under the re-learnt models (the same colour models for both)
    f0, ious0 = cost(np.zeros(6), detail=True)
    moved = float(np.linalg.norm((T[:3, :3] @ ctr + T[:3, 3]) - cw)) * 1000
    turned = math.degrees(float(np.linalg.norm(x[:3])))
    best_iou = max(ious.values()) if ious else 0.0
    used = [v for v, wv in zip(views, w) if wv > 0]
    why = ""
    if best_iou < c["ok_iou"] or min(ious.values(), default=0.0) < c["min_view_iou"]:
        why = "fit_poor"
    elif len(used) < 2 and not any(v.role == "own" for v in used):
        why = "one_far_view"            # the head alone (~2.4 mm/px) does not pin 6 DoF (5.1 tower: 1 -> 22 mm)
    elif float(np.linalg.norm(x[3:])) > 0.98 * max_shift or float(np.linalg.norm(x[:3])) > 0.98 * max_turn:
        why = "out_of_range"
    elif f > f0 - c["min_gain"]:
        why = "no_gain"
    out = {**base, "ok": why == "", "why": why or None, "ious": ious, "ious_prior": ious0, "cost": round(f, 4),
           "cost_prior": round(f0, 4), "cost_prior_first": round(f_prior, 4), "moved_mm": round(moved, 1),
           "turned_deg": round(turned, 1), "fit_pose7": _pose7(T), "time_s": round(time.time() - t0, 3)}
    out["pose7"] = out["fit_pose7"] if out["ok"] else list(map(float, prior_pose7))
    return out
