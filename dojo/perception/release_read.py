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
from .held import Silhouettes, View, _edge_cost, _iou, _roi_for, apply_delta, robot_unknown
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
    "w_edge": 0.5,                 # outline-to-image-edge term (held_pose's), per view, against 1 - IoU
    "rest_starts": True,           # also search around the asset's resting shapes at the release xy (it fell over)
    "rest_margin": 0.05,           # ... taken only when that fit's cost beats the local fit's by this much
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


def rest_poses(key: str, T0: np.ndarray, ctr: np.ndarray, others=()) -> list[tuple[np.ndarray, str]]:
    """The asset's resting shapes (``inverse_layout.shape_models``) with the body centre at the released xy, at the
    released yaw and +90 degrees, resting on the highest known support under it (``refine.support_z``), else the table."""
    from .cameras import TABLE_Z
    from .inverse_layout import shape_models
    from .refine import support_z

    try:
        sms = shape_models(key)
    except Exception:  # noqa: BLE001 (no shape models for this asset: no resting alternatives)
        return []
    c0 = T0[:3, :3] @ ctr + T0[:3, 3]
    try:
        top = support_z(load_asset(key), _pose7(T0), list(others or ()))
    except Exception:  # noqa: BLE001
        top = TABLE_Z
    out = []
    for sm in sms:
        Rz = T0[:3, :3] @ sm.R0.T
        yaw0 = math.atan2(Rz[1, 0], Rz[0, 0])
        for k, yaw in enumerate((yaw0, yaw0 + math.pi / 2)):
            T = sm.pose(0.0, 0.0, yaw, top - TABLE_Z)
            cc = T[:3, :3] @ ctr + T[:3, 3]
            T[0, 3] += c0[0] - cc[0]
            T[1, 3] += c0[1] - cc[1]
            out.append((T, f"{sm.tag}/{k * 90}"))
    return out


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
    base_T = {"T": T0, "c": cw}                         # the search's base pose (the prior, or a resting alternative)

    def cost(d, detail=False):
        T = apply_delta(base_T["T"], base_T["c"], d)
        tot, ious = 0.0, {}
        for v, wv in zip(views, w):
            if wv <= 0:
                continue
            sv = sil.render(v.cam, T)
            iou = _iou(sv, v)
            ious[v.name] = round(iou, 4)
            ec = _edge_cost(sv, v) if c["w_edge"] > 0 else None
            # the outline against image edges: a gold ring on a black board with yellow grid lines has a colour model
            # that confuses the lines with the ring (iou 0.4-0.5 at a 4 mm prior); its outline still sits on edges
            tot += wv * ((1.0 - iou) + (c["w_edge"] * ec if ec is not None else c["w_edge"] * 0.5))
        tot /= max(float(w.sum()), 1e-9)
        sh, tr = float(np.linalg.norm(d[3:6])), float(np.linalg.norm(d[:3]))
        tot += 10.0 * (max(0.0, sh - max_shift) / 0.001 + max(0.0, tr - max_turn) / 0.01)
        return (tot, ious) if detail else tot

    def models(x, e_m, band_m) -> int:
        T = apply_delta(base_T["T"], base_T["c"], x)
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

    def set_base(T):
        base_T["T"], base_T["c"] = T, T[:3, :3] @ ctr + T[:3, 3]

    if models(np.zeros(6), 0.35 * c["prior_err_mm"] / 1000, c["ring_mm"] / 1000) == 0:
        return {**base, "ok": False, "why": "no_models", "pose7": list(map(float, prior_pose7)),
                "time_s": round(time.time() - t0, 3)}
    f_prior = cost(np.zeros(6))
    starts = [np.zeros(6)]
    for s_ in c["swing_deg"]:
        for ax in (0, 1):
            d = np.zeros(6)
            d[ax] = s_ * deg
            starts.append(d)
    for s_ in c["shift_mm"]:
        for ax in (3, 4):
            d = np.zeros(6)
            d[ax] = s_ * mm
            starts.append(d)
    scored = sorted(((cost(s_), i, s_) for i, s_ in enumerate(starts)), key=lambda t: t[:2])
    cands = []                                          # (cost, base T, x, tag)
    for f_s, _, s_ in scored[:3]:
        x, f = nm(s_, [4 * deg, 4 * deg, 4 * deg, 3 * mm, 3 * mm, 3 * mm], 400)
        cands.append((f, T0, x, "local"))
    rest = []
    if c["rest_starts"]:
        # it may have come to rest another way than it was put down (a mallet / broom lying flat, a pen fallen over):
        # every resting shape of the asset at the release xy, on the support under it
        for T_alt, tag in rest_poses(key, T0, ctr, others):
            set_base(T_alt)
            rest.append((cost(np.zeros(6)), T_alt, tag))
        rest.sort(key=lambda t: t[0])
        for f_a, T_alt, tag in rest[:2]:
            set_base(T_alt)
            x, f = nm(np.zeros(6), [4 * deg, 4 * deg, 4 * deg, 3 * mm, 3 * mm, 3 * mm], 300)
            cands.append((f, T_alt, x, f"rest:{tag}"))
    # a resting alternative must beat the local fit clearly (a mug on its hook fitted as one lying on the table: 19 ->
    # 27 mm at a slightly lower cost under the first colour models)
    loc = min((t for t in cands if t[3] == "local"), key=lambda t: t[0])
    alt = min((t for t in cands if t[3] != "local"), key=lambda t: t[0], default=None)
    f_b, Tb, xb, how = alt if alt is not None and alt[0] < loc[0] - c["rest_margin"] else loc
    set_base(Tb)
    models(xb, 0.002, c["ring_mm"] / 1000)                       # re-learnt around the fit
    f_b = cost(xb)
    x, f = nm(xb, [1.5 * deg, 1.5 * deg, 1.5 * deg, 1 * mm, 1 * mm, 1 * mm], 400)
    if f > f_b:
        x, f = xb, f_b
    f, ious = cost(x, detail=True)
    T = apply_delta(base_T["T"], base_T["c"], x)
    # the prior's cost under the re-learnt models (the same colour models for both)
    set_base(T0)
    f0, ious0 = cost(np.zeros(6), detail=True)
    moved = float(np.linalg.norm((T[:3, :3] @ ctr + T[:3, 3]) - cw)) * 1000
    dR = T0[:3, :3].T @ T[:3, :3]
    turned = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(dR)) - 1) / 2))))
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
           "turned_deg": round(turned, 1), "fit_pose7": _pose7(T), "via": how, "time_s": round(time.time() - t0, 3)}
    out["pose7"] = out["fit_pose7"] if out["ok"] else list(map(float, prior_pose7))
    return out


# ------------------------------------------------------------------------------------------------ the wide search
SEARCH_DEFAULTS = {
    "radius_mm": 150.0,            # head-view search around the release point (fell near: 50-150 mm, M's replays)
    "pre_max_shift_mm": 20.0,      # the pre-grasp check: the object still where it was picked from (a missed grasp)
    "pre_min_prec": 0.6,           # ... its head silhouette there this much static-background foreground
    "pre_min_px": 20,
    "head_min_iou": 0.65,          # a head-only (x, y, yaw) fit must explain its view this well ...
    "blob_min_cover": 0.5,         # ... and cover this much of its blob
    "min_blob_px": 25,
    "max_cands": 4,
    "yaws": 4,
}


def _head_view(frame: dict, bgs: dict, thr: float):
    from .cameras import head_camera
    from .segment import segment_views

    cam = head_camera()
    return segment_views({"cam_head": frame["images"]["cam_head"]}, {"cam_head": cam}, frame["state"], bgs, thr=thr)["cam_head"]


def _head_fit(v, key: str, T_init: np.ndarray, others_mask, yaws: int = 1, iters: int = 60, centre_xy=None):
    """(T, iou, silhouette) of the asset's best resting shape fitted in (x, y, yaw) in the head view
    (``inverse_layout.refine``; ``others_mask``: pixels of other believed objects, background here). ``yaws`` == 1:
    from ``T_init`` itself (its resting shape and height); else every resting shape on the table at ``yaws`` yaws with
    the body centre at ``centre_xy``."""
    from .cameras import TABLE_Z
    from .inverse_layout import refine as nm_refine
    from .inverse_layout import shape_models

    a = load_asset(key)
    ctr = (np.asarray(a.info["bbox_min"]) + np.asarray(a.info["bbox_max"])) / 2
    best = None
    for sm in shape_models(key):
        Rz = T_init[:3, :3] @ sm.R0.T
        yaw0 = math.atan2(Rz[1, 0], Rz[0, 0])
        z_off = 0.0 if yaws > 1 else float(T_init[2, 3] - (TABLE_Z - sm.zmin))
        period = sm.yaw_period if sm.yaw_period < 1.5 * math.pi else 2 * math.pi
        for k in range(yaws):
            yaw = yaw0 + k * period / max(yaws, 1)
            x0, y0 = T_init[0, 3], T_init[1, 3]
            if centre_xy is not None:                   # the origin that puts the body centre on the blob
                Tc = sm.pose(0.0, 0.0, yaw, z_off)
                cc = Tc[:3, :3] @ ctr + Tc[:3, 3]
                x0, y0 = centre_xy[0] - cc[0], centre_xy[1] - cc[1]
            x, y, yw, loss, per = nm_refine(sm, x0, y0, yaw, {"cam_head": v}, others={"cam_head": others_mask},
                                            xy_step=0.004, yaw_step=0.1, maxiter=iters, z_off=z_off)
            iou = float((per or {}).get("cam_head", 0.0))
            if best is None or iou > best[1]:
                best = (sm.pose(x, y, yw, z_off), iou, sm)
        if yaws == 1:
            break                                       # the pre-grasp check keeps the believed resting shape
    if best is None:
        return None
    T, iou, sm = best
    return T, iou, sm.sil(v.cam, T)


def release_search(frame: dict, key: str, prior_pose7, arm: str, others=(), pre_pose7=None, head_bgs=None,
                   head_thr: float = 18.0, cfg: dict | None = None) -> dict:
    """The released object when the local read finds nothing at the release (it fell, bounced, or never left the table):
    (1) the pre-grasp pose ``pre_pose7`` (a missed grasp: still where it was picked from), checked by a head-view
    (x, y, yaw) fit and, when a wrist sees it, the 6-DoF read; (2) a head-view search within ``radius_mm`` of the
    release: static-background foreground (``head_bgs``) that no other believed object explains, blobs nearest
    first, each fitted with the asset's resting shapes; a wrist that sees the fit polishes it (6-DoF read). Returns
    {ok, via ("pre_grasp" | "head_search"), pose7, ious, tried, ...}; not ok keeps the prior."""
    import cv2

    t0 = time.time()
    c = {**SEARCH_DEFAULTS, **(cfg or {})}
    rcfg = {k: v for k, v in (cfg or {}).items() if k in DEFAULTS}
    prior = list(map(float, prior_pose7))
    tried: list[dict] = []
    if head_bgs is None:
        return {"ok": False, "why": "no_head", "pose7": prior, "tried": tried, "time_s": round(time.time() - t0, 3)}
    v = _head_view(frame, head_bgs, head_thr)
    a = load_asset(key)
    oth = [(k2, pose_to_T(p2[:3], p2[3:])) for k2, p2 in others or ()]
    om = np.zeros((v.cam.h, v.cam.w), np.uint8)
    for k2, T2 in oth:
        try:
            a2 = load_asset(k2)
            om |= silhouette(v.cam, transform(a2.Vs, T2), a2.Fs)
        except Exception:  # noqa: BLE001
            continue
    om = cv2.dilate(om, np.ones((5, 5), np.uint8))

    def verify(T: np.ndarray, via: str, iou_h: float, force: bool = False) -> dict | None:
        """A wrist that sees the hypothesis polishes it (6-DoF read, any range rules); else the head fit stands
        (``force``: the head evidence alone is enough, e.g. foreground where the object was grasped from)."""
        rr = release_read(frame, key, _pose7(T), arm, others=others, cfg={**rcfg, "rest_starts": False,
                                                                          "max_shift_mm": 20.0, "max_turn_deg": 30.0})
        if rr.get("ok"):
            return {**rr, "via": via, "head_iou": round(iou_h, 4)}
        if force or (rr.get("why") in ("one_far_view", "no_gain", "no_views") and iou_h >= c["head_min_iou"]):
            return {"ok": True, "via": via, "pose7": _pose7(T), "head_iou": round(iou_h, 4), "ious": {"cam_head": iou_h},
                    "read6_why": rr.get("why")}
        return None

    # (1) the pre-grasp pose: the pose the object had when the jaws closed was a measured one (the pick hover's refine);
    # a missed grasp leaves the object there. Its silhouette is foreground in the head view (``prec``: thin objects such
    # as a mallet are 3-5 head px wide, so the fit's IoU is noisy and the precision decides), and a fit stays near it
    if pre_pose7 is not None:
        Tp = pose_to_T(pre_pose7[:3], pre_pose7[3:])
        sp = silhouette(v.cam, transform(a.Vs, Tp), a.Fs)
        vis = (sp > 0) & (v.known > 0) & (om == 0)
        n_vis = int(vis.sum())
        prec0 = float((vis & (v.fg > 0)).sum()) / max(n_vis, 1)
        f = _head_fit(v, key, Tp, om, yaws=1)
        T, iou = (f[0], f[1]) if f is not None else (Tp, 0.0)
        shift = float(np.hypot(T[0, 3] - Tp[0, 3], T[1, 3] - Tp[1, 3])) * 1000
        prec1 = 0.0
        if f is not None and shift <= c["pre_max_shift_mm"]:
            vis1 = (f[2] > 0) & (v.known > 0) & (om == 0)
            prec1 = float((vis1 & (v.fg > 0)).sum()) / max(int(vis1.sum()), 1) if int(vis1.sum()) >= c["pre_min_px"] else 0.0
        tried.append({"via": "pre_grasp", "head_iou": round(iou, 3), "prec": round(prec0, 3), "prec_fit": round(prec1, 3),
                      "vis_px": n_vis, "shift_mm": round(shift, 1)})
        there = (n_vis >= c["pre_min_px"] and prec0 >= c["pre_min_prec"]) or prec1 >= c["pre_min_prec"]
        if there or (shift <= c["pre_max_shift_mm"] and iou >= c["head_min_iou"]):
            T_use = T if (prec1 >= prec0 and shift <= c["pre_max_shift_mm"]) else Tp
            out = verify(T_use, "pre_grasp", iou, force=there)
            if out is not None:
                return {**out, "pre_prec": round(prec0, 3), "tried": tried, "time_s": round(time.time() - t0, 3)}
    # (2) head-view blobs near the release
    T0 = pose_to_T(prior[:3], prior[3:])
    resid = ((v.fg > 0) & (v.known > 0) & (om == 0)).astype(np.uint8)
    resid = cv2.morphologyEx(resid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(resid, connectivity=8)
    from .cameras import TABLE_Z

    hz = 0.5 * float(np.ptp(a.V[:, 2])) if len(a.V) else 0.02
    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < c["min_blob_px"]:
            continue
        X = v.cam.ray_plane(np.asarray([cents[i]], float), TABLE_Z + min(hz, 0.03))[0]
        if not np.isfinite(X).all():
            continue
        d = float(np.hypot(X[0] - T0[0, 3], X[1] - T0[1, 3])) * 1000
        if d <= c["radius_mm"]:
            blobs.append((d, i, area, X))
    for d, i, area, X in sorted(blobs)[: c["max_cands"]]:
        f = _head_fit(v, key, T0, om, yaws=c["yaws"], centre_xy=(float(X[0]), float(X[1])))
        if f is None:
            continue
        T, iou, s = f
        m = lab == i
        cover = float((m & (s > 0)).sum()) / max(int(m.sum()), 1)
        tried.append({"via": "head_search", "blob_px": area, "from_release_mm": round(d, 1), "head_iou": round(iou, 3),
                      "blob_cover": round(cover, 3)})
        if iou >= c["head_min_iou"] and cover >= c["blob_min_cover"]:
            out = verify(T, "head_search", iou)
            if out is not None:
                return {**out, "tried": tried, "time_s": round(time.time() - t0, 3)}
    return {"ok": False, "why": "not_found", "pose7": prior, "tried": tried, "time_s": round(time.time() - t0, 3)}
