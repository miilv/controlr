"""In-hand pose of a held object: render-and-compare in every camera that sees it, constrained by the gripper.

    from dojo.perception.held import held_pose
    r = held_pose(frame, "left", "Rigid/charger/00000", prior_T_tcp_obj)
    r["ok"], r["T_tcp_obj"], r["sigma"]       # 4x4 in the skills' TCP frame; per-axis sigma (mm / deg)

Inputs are the official observation only: 3 RGB frames, measured joints (FK -> wrist cameras, TCP, finger meshes),
the commanded gripper and the published camera / robot config. The prior is the skills' predicted in-hand transform
(``Held.T_tcp_obj``: object root in the TCP frame; TCP = FK link6 @ +``TCP_OFFSET`` along flange x; x = approach,
y = closing axis).

Method (B1 machinery: ``render`` silhouettes, ``refine.local_foreground``):
- Views: the holding arm's own wrist camera (the object at 6-15 cm: 0.15-0.4 mm per pixel; rigid with the TCP, so
  the arm's own FK error cancels), the head camera and the other arm's wrist camera, each used when enough of the
  object's silhouette is visible (not under the robot, inside the image).
- Robot pixels (both arms from FK; the holding arm's fingers over the range of openings the object allows) are
  unknown: they neither count as object nor as background.
- Foreground per view: a background colour model from a ring around the expected silhouette (``local_foreground``).
- Fit: soft IoU of the rendered silhouette with the foreground, summed over views, in square ROIs at up to
  ``ROI_MAX`` px (sub-pixel through supersampling), 6 DoF about the object's bbox centre, from the prior and a few
  starts in the pad plane (turn about the closing axis, slide along approach / across), then a 6-DoF polish.
  A soft gripper term keeps the object centred between the pads along the closing axis.
- Sigma: the Laplace approximation of the fit (a numerical Hessian of the cost) mapped to the TCP axes, the leading
  point and the insertion axis, times a factor calibrated on our dev captures (``SIGMA_SCALE``).
"""
from __future__ import annotations

import math
import time
from functools import lru_cache

import cv2
import numpy as np

from .cameras import Camera, cameras_from_state, pose_to_T, R_to_quat_wxyz
from .render import gl_roi, load_asset, robot_link_meshes, silhouette, transform, crop_camera
from .robot_model import X5Model

VERSION = "held-pose-1"
TCP_OFFSET = 0.145                 # dojo.skills.kinematics.TCP_OFFSET
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
ROI_MAX = 256                      # px of a view's ROI render (supersampled up to x3 for small objects)
MIN_VIS_PX = 150                   # visible silhouette pixels (native) for a view to count
MIN_VIS_FRAC = 0.25                # visible share of the silhouette
PAD_LEN = 0.030                    # m of each finger (from the tip) that can touch the object
CENTRE_SIGMA_M = 0.0015            # soft gripper term: the contact faces centred on the TCP along the closing axis
SIGMA_SCALE = {"pos": 1.0, "rot": 1.0}   # calibrated on dev captures (held_eval)
OK_IOU = 0.80                      # best view's IoU for an ok fit
OK_MAX_LEAD_SIGMA_MM = 2.0
VIEW_W = {"own": 1.0, "head": 0.6, "observer": 1.0}


# ----------------------------------------------------------------------------------------------- geometry helpers
def _rotvec_R(v) -> np.ndarray:
    v = np.asarray(v, float)
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def tcp_T(arm: str, q, model=None) -> np.ndarray:
    model = model or X5Model.default()
    T6 = model.fk_world(arm, q)["link6"]
    T = T6.copy()
    T[:3, 3] = T6[:3, 3] + T6[:3, 0] * TCP_OFFSET
    return T


def apply_delta(T_prior: np.ndarray, c: np.ndarray, d) -> np.ndarray:
    """``T_prior`` (object in TCP) turned by rotvec d[:3] (rad, TCP axes) about the point ``c`` (TCP frame), then
    moved by d[3:] (m, TCP axes)."""
    R = _rotvec_R(d[:3])
    D = np.eye(4)
    D[:3, :3] = R
    D[:3, 3] = c - R @ c + np.asarray(d[3:6], float)
    return D @ T_prior


@lru_cache(maxsize=1)
def finger_table():
    """Pad geometry vs the normalised opening, in the TCP frame (identical for both arms): for o in [0, 1], the gap
    between the inner pad faces (m), the fingertip x and each finger's points (TCP frame)."""
    model = X5Model.default()
    meshes = robot_link_meshes()
    q = np.zeros(6)
    rows = []
    for o in np.linspace(0.0, 1.0, 51):
        P = model.fk_world("left", q, float(o))
        Tt = P["link6"].copy()
        Tt[:3, 3] += Tt[:3, 0] * TCP_OFFSET
        inv = np.linalg.inv(Tt)
        F = {}
        for ln in ("link7", "link8"):
            V, _ = meshes[ln]
            W = V @ P[ln][:3, :3].T + P[ln][:3, 3]
            F[ln] = W @ inv[:3, :3].T + inv[:3, 3]
        tip = max(F["link7"][:, 0].max(), F["link8"][:, 0].max())
        A = {ln: F[ln][F[ln][:, 0] > tip - PAD_LEN] for ln in F}
        hi, lo = sorted(A, key=lambda ln: -A[ln][:, 1].mean())
        gap = float(A[hi][:, 1].min() - A[lo][:, 1].max())
        rows.append((float(o), gap, float(tip)))
    return np.array(rows)


def opening_for_width(width: float) -> float:
    t = finger_table()
    order = np.argsort(t[:, 1])
    return float(np.clip(np.interp(width, t[order, 1], t[order, 0]), 0.0, 1.0))


def pad_zone(V_tcp: np.ndarray, tip_x: float, half_h: float = 0.012) -> np.ndarray:
    """Object points the pads can touch: within PAD_LEN of the fingertip along x and near the pad's mid-plane."""
    return (V_tcp[:, 0] > tip_x - PAD_LEN) & (V_tcp[:, 0] < tip_x + 0.002) & (np.abs(V_tcp[:, 2]) < half_h)


# ----------------------------------------------------------------------------------------------- views
class View:
    """One camera restricted to a square ROI around the object (resampled to ``size``)."""

    def __init__(self, name, role, cam: Camera, img, fg, known, crop, size):
        self.name, self.role, self.cam_full = name, role, cam
        x0, y0, x1, y1 = crop
        self.crop = crop
        self.size = size
        self.cam = crop_camera(cam, x0, y0, x1, y1, size)
        interp = cv2.INTER_AREA if (x1 - x0) > size else cv2.INTER_LINEAR
        self.fg = cv2.resize(_crop(fg.astype(np.float32), crop), (size, size), interpolation=interp)
        self.known = cv2.resize(_crop(known.astype(np.float32), crop), (size, size), interpolation=interp)
        self.px_scale = (x1 - x0) / size     # native px per ROI px
        self.img = img


def _crop(a: np.ndarray, crop) -> np.ndarray:
    x0, y0, x1, y1 = crop
    h, w = a.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0) + a.shape[2:], a.dtype)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
    return out


class Silhouettes:
    """Silhouette renders of one asset: GL when available (exact mesh), else the CPU fill of the decimated mesh."""

    def __init__(self, key: str):
        self.a = load_asset(key)
        self.key = key

    def render(self, cam: Camera, T: np.ndarray) -> np.ndarray:
        g = gl_roi(cam.w) if cam.w == cam.h else None
        if g is not None:
            return g.render(cam, [(self.key, self.a.Vg, self.a.Fg, T)]).astype(np.float32)
        return silhouette(cam, transform(self.a.Vs, T), self.a.Fs).astype(np.float32)


def robot_unknown(cam: Camera, state: dict, arm: str, o_lo: float, o_hi: float, other_g: float | None,
                  dilate_px: int = 3) -> np.ndarray:
    """Robot pixels: both arms from FK; the holding arm's fingers over openings [o_lo, o_hi], the other arm's fingers
    at its commanded opening (or swept to open when it is commanded closed: it may hold something)."""
    model = X5Model.default()
    meshes = robot_link_meshes()
    m = np.zeros((cam.h, cam.w), np.uint8)
    for a in ("left", "right"):
        q = state.get(f"{a}_arm_joint_state")
        if q is None:
            continue
        q = np.asarray(q, float).reshape(-1)[:6]
        if a == arm:
            opens = np.linspace(o_lo, o_hi, max(2, int(math.ceil((o_hi - o_lo) / 0.05)) + 1))
        else:
            g = 1.0 if other_g is None else float(other_g)
            opens = [g] if g >= 0.95 else np.linspace(g, 1.0, 6)
        base = model.fk_world(a, q, float(opens[0]))
        for ln, (V, F) in meshes.items():
            if ln in base and ln not in ("link7", "link8"):
                silhouette(cam, transform(V, base[ln]), F, out=m)
        for o in opens:
            P = model.fk_world(a, q, float(o))
            for ln in ("link7", "link8"):
                if ln in meshes:
                    V, F = meshes[ln]
                    silhouette(cam, transform(V, P[ln]), F, out=m)
    if dilate_px > 0:
        m = cv2.dilate(m, np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8))
    return m


def _roi_for(cam: Camera, P_world: np.ndarray, margin_m: float):
    """Square native-pixel ROI around projected points, grown by ``margin_m`` at their depth."""
    uv, z = cam.project(P_world)
    if (z <= 0.02).any():
        return None
    zc = float(np.median(z))
    pad = margin_m * cam.K[0, 0] / max(zc, 1e-3) + 8
    (x0, y0), (x1, y1) = uv.min(0) - pad, uv.max(0) + pad
    if x1 < 0 or y1 < 0 or x0 >= cam.w or y0 >= cam.h:
        return None
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = int(math.floor(cx - side / 2)), int(math.floor(cy - side / 2))
    s = int(math.ceil(side))
    return (x0, y0, x0 + s, y0 + s)


def _views(images, state, arm, sil: Silhouettes, T_tcp, T_prior, cams, o_lo, o_hi, other_g, max_shift_m,
           others=None, exclude_cams=()):
    from .refine import local_foreground

    a = sil.a
    T_w = T_tcp @ T_prior
    Vw = transform(a.Vs, T_w)
    bb = np.array([[x, y, z] for x in (a.info["bbox_min"][0], a.info["bbox_max"][0])
                   for y in (a.info["bbox_min"][1], a.info["bbox_max"][1])
                   for z in (a.info["bbox_min"][2], a.info["bbox_max"][2])])
    bbw = transform(bb, T_w)
    out, diag = [], {}
    for name in CAMS:
        if name in exclude_cams or name not in images or name not in cams:
            continue
        cam = cams[name]
        role = "own" if name == f"cam_{arm}_wrist" else ("head" if name == "cam_head" else "observer")
        S = silhouette(cam, Vw, a.Fs)
        n = int(S.sum())
        if n < 30:
            diag[name] = {"used": False, "why": "not in view"}
            continue
        rob = robot_unknown(cam, state, arm, o_lo, o_hi, other_g)
        vis = float((S & (1 - rob)).sum()) / n
        if vis < MIN_VIS_FRAC or n * vis < MIN_VIS_PX * (0.15 if role == "head" else 1.0):
            diag[name] = {"used": False, "why": "occluded", "visible_frac": round(vis, 3), "px": n}
            continue
        excl = None
        if others:
            excl = np.zeros_like(S)
            for k2, p2 in others:
                a2 = load_asset(k2)
                excl |= silhouette(cam, transform(a2.Vs, pose_to_T(p2[:3], p2[3:])), a2.Fs)
        # the colour model's ring sits outside the expected silhouette grown by the search range
        grow_px = int(max_shift_m * cam.K[0, 0] / max(float(np.median(cam.project(bbw)[1])), 1e-3))
        Sg = cv2.dilate(S, np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8)) if grow_px > 0 else S
        fg, known = local_foreground(images[name], cam, Sg, rob, grow=0.5, exclude=excl)
        known = known & (1 - rob)
        roi = _roi_for(cam, bbw, max_shift_m)
        if roi is None:
            diag[name] = {"used": False, "why": "roi"}
            continue
        side = roi[2] - roi[0]
        size = int(np.clip(side * 3, 96, ROI_MAX))
        size = min(size, ROI_MAX)
        v = View(name, role, cam, images[name], fg, known, roi, size)
        out.append(v)
        diag[name] = {"used": True, "role": role, "visible_frac": round(vis, 3), "px": n, "roi_px": side,
                      "roi_size": size}
    return out, diag


def _iou(s: np.ndarray, v: View) -> float:
    k = v.known
    inter = float((s * v.fg * k).sum())
    union = float(((s + v.fg - s * v.fg) * k).sum())
    return inter / union if union > 0 else 0.0


# ----------------------------------------------------------------------------------------------- the estimator
def held_pose(frame: dict, arm: str, asset_key: str, prior_T_tcp_obj, cfg: dict | None = None) -> dict:
    """See the module doc. ``frame``: {"images": {cam: HxWx3 uint8}, "state": {k: array}} (a decoded service frame)."""
    t0 = time.time()
    cfg = dict(cfg or {})
    images = {k: np.ascontiguousarray(np.asarray(v, np.uint8)[..., :3]) for k, v in frame["images"].items()}
    state = {k: np.asarray(v, float).reshape(-1) for k, v in frame["state"].items()}
    T_prior = np.asarray(prior_T_tcp_obj, float).reshape(4, 4)
    max_shift = float(cfg.get("max_shift_mm", 10.0)) / 1000
    max_turn = math.radians(float(cfg.get("max_turn_deg", 15.0)))
    model = X5Model.default()
    q = state[f"{arm}_arm_joint_state"][:6]
    T_tcp = tcp_T(arm, q, model)
    sil = Silhouettes(asset_key)
    a = sil.a
    cams = cameras_from_state({k: v for k, v in state.items()}, model)
    # the object between the pads: its width there sets the finger opening; unknown band over +-3 mm
    Vt = transform(a.V, T_prior)
    tbl = finger_table()
    tip_x = float(np.interp(0.5, tbl[:, 0], tbl[:, 2]))
    zone = pad_zone(Vt, tip_x)
    width = float(np.ptp(Vt[zone, 1])) if zone.sum() >= 3 else float(np.ptp(Vt[:, 1]))
    o_est = opening_for_width(width)
    o_lo, o_hi = opening_for_width(max(0.0, width - 0.006)), opening_for_width(width + 0.006)
    other_g = state.get(f"{'right' if arm == 'left' else 'left'}_ee_joint_state")
    other_g = None if other_g is None else float(other_g[0])
    views, diag = _views(images, state, arm, sil, T_tcp, T_prior, cams, o_lo, o_hi, other_g, max_shift,
                         others=cfg.get("others"), exclude_cams=tuple(cfg.get("exclude_cams", ())))
    lead_obj, axis_obj = _lead_axis(a, T_prior, cfg)
    base = {"version": VERSION, "views": [v.name for v in views], "view_diag": diag, "opening_est": round(o_est, 3),
            "width_mm": round(width * 1000, 2)}
    if not views:
        return _fail(T_prior, T_tcp, "no_views", base, t0, lead_obj, axis_obj)
    c = (np.asarray(a.info["bbox_min"]) + np.asarray(a.info["bbox_max"])) / 2
    c_tcp = T_prior[:3, :3] @ c + T_prior[:3, 3]
    # pad-zone points for the gripper term (a subsample)
    Vp = a.V[zone] if zone.sum() >= 3 else None
    if Vp is not None and len(Vp) > 400:
        Vp = Vp[np.random.default_rng(0).choice(len(Vp), 400, replace=False)]
    w_views = np.array([VIEW_W[v.role] for v in views])

    def cost(d, detail=False):
        T = apply_delta(T_prior, c_tcp, d)
        Tw = T_tcp @ T
        tot, ious = 0.0, {}
        for v, w in zip(views, w_views):
            s = sil.render(v.cam, Tw)
            iou = _iou(s, v)
            ious[v.name] = iou
            tot += w * (1.0 - iou)
        tot /= float(w_views.sum())
        pen = 0.0
        if Vp is not None:
            P = transform(Vp, T)
            mid = 0.5 * (P[:, 1].max() + P[:, 1].min())
            pen = 0.5 * (mid / CENTRE_SIGMA_M) ** 2 * 0.002
        # beyond the search range: a wall
        sh = float(np.linalg.norm(d[3:6]))
        tr = float(np.linalg.norm(d[:3]))
        wall = 10.0 * (max(0.0, sh - max_shift) / 0.001 + max(0.0, tr - max_turn) / 0.01)
        f = tot + pen + wall
        return (f, ious) if detail else f

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

    # 1) pad-plane starts: turn about the closing axis (TCP y), slide along approach (x) / across (z)
    starts = [np.zeros(6)]
    for ry in (-6, 6):
        starts.append(np.array([0, ry * deg, 0, 0, 0, 0], float))
    for dx, dz in ((3, 0), (-3, 0), (0, 3), (0, -3)):
        starts.append(np.array([0, 0, 0, dx * mm, 0, dz * mm], float))
    scored = sorted(((cost(s), s) for s in starts), key=lambda t: t[0])
    best = None
    for f0, s in scored[:3]:
        x, f = nm(s, [3 * deg, 3 * deg, 3 * deg, 2 * mm, 1 * mm, 2 * mm], 400)
        if best is None or f < best[1]:
            best = (x, f)
    # 2) polish with small steps
    x, f = nm(best[0], [0.5 * deg, 0.5 * deg, 0.5 * deg, 0.3 * mm, 0.3 * mm, 0.3 * mm], 400)
    if f > best[1]:
        x, f = best
    f, ious = cost(x, detail=True)
    T_est = apply_delta(T_prior, c_tcp, x)
    sig = _sigma(cost, x, T_prior, c_tcp, lead_obj, axis_obj, f)
    out = _result(T_est, T_prior, T_tcp, lead_obj, axis_obj, sig, ious, f, base, t0)
    best_iou = max(ious.values()) if ious else 0.0
    why = ""
    if best_iou < OK_IOU:
        why = "fit_poor"
    elif float(np.linalg.norm(x[3:])) > 0.98 * max_shift or float(np.linalg.norm(x[:3])) > 0.98 * max_turn:
        why = "out_of_range"
    elif max(sig["lead_mm"]) > OK_MAX_LEAD_SIGMA_MM:
        why = "ambiguous"
    out["ok"] = why == ""
    out["why"] = why
    if not out["ok"]:
        out["fit_T_tcp_obj"] = out["T_tcp_obj"]
        out["T_tcp_obj"] = T_prior.tolist()
        out["pose7"] = _pose7(T_tcp @ T_prior)
    return out


def _lead_axis(a, T_prior, cfg):
    """Insertion geometry in the object frame: given (cfg lead_obj / axis_obj), else the end of the mesh pointing
    down at the prior (world -z through the TCP) and the principal mesh axis nearest that direction."""
    if cfg.get("lead_obj") is not None and cfg.get("axis_obj") is not None:
        ax = np.asarray(cfg["axis_obj"], float)
        return np.asarray(cfg["lead_obj"], float), ax / np.linalg.norm(ax)
    V = a.V
    down_obj = T_prior[:3, :3].T @ np.array([1.0, 0.0, 0.0])    # approach direction (TCP +x) in the object frame
    if cfg.get("axis_obj") is not None:
        ax = np.asarray(cfg["axis_obj"], float)
    else:
        C = np.cov((V - V.mean(0)).T)
        w, E = np.linalg.eigh(C)
        ax = E[:, int(np.argmax(np.abs(E.T @ down_obj)))]
        if ax @ down_obj < 0:
            ax = -ax
    ax = ax / np.linalg.norm(ax)
    if cfg.get("lead_obj") is not None:
        return np.asarray(cfg["lead_obj"], float), ax
    s = V @ ax
    lead = V[s >= s.max() - 0.001].mean(0)
    return lead, ax


def _pose7(T) -> list:
    return [float(v) for v in T[:3, 3]] + [float(v) for v in R_to_quat_wxyz(T[:3, :3])]


def _sigma(cost, x, T_prior, c_tcp, lead_obj, axis_obj, f0) -> dict:
    """Laplace approximation: cov = 2 f0 / n_eff * H^-1 of the cost in (rad, m), mapped to TCP pos / rot, the leading
    point and the axis; times SIGMA_SCALE (calibrated)."""
    h = np.array([0.4, 0.4, 0.4, 0.3, 0.3, 0.3]) * np.array([math.radians(1)] * 3 + [0.001] * 3)
    n = 6
    H = np.zeros((n, n))
    fx = cost(x)
    E = np.eye(n)
    fp = [cost(x + h[i] * E[i]) for i in range(n)]
    fm = [cost(x - h[i] * E[i]) for i in range(n)]
    for i in range(n):
        H[i, i] = (fp[i] - 2 * fx + fm[i]) / h[i] ** 2
    for i in range(n):
        for j in range(i + 1, n):
            fpp = cost(x + h[i] * E[i] + h[j] * E[j])
            H[i, j] = H[j, i] = (fpp - fp[i] - fp[j] + fx) / (h[i] * h[j])
    # a non-smooth cost can give a non-PD Hessian: floor the eigenvalues
    w, U = np.linalg.eigh((H + H.T) / 2)
    w = np.maximum(w, 1e-3 * max(float(w.max()), 1e-9))
    s2 = max(fx, 1e-4) * 1e-3     # residual-scaled noise level (calibrated via SIGMA_SCALE)
    cov = U @ np.diag(s2 / w) @ U.T
    # Jacobians of the outputs w.r.t. d (numerical, tiny steps)
    T0 = apply_delta(T_prior, c_tcp, x)

    def outputs(T):
        lead = T[:3, :3] @ lead_obj + T[:3, 3]
        ax = T[:3, :3] @ axis_obj
        return np.r_[T[:3, 3], lead, ax]
    o0 = outputs(T0)
    J = np.zeros((9, 6))
    for i in range(6):
        e = np.zeros(6)
        e[i] = 1e-6
        J[:, i] = (outputs(apply_delta(T_prior, c_tcp, x + e)) - o0) / 1e-6
    C = J @ cov @ J.T
    pos = np.sqrt(np.maximum(np.diag(C)[:3], 0)) * 1000 * SIGMA_SCALE["pos"]
    lead = np.sqrt(np.maximum(np.diag(C)[3:6], 0)) * 1000 * SIGMA_SCALE["pos"]
    rot = np.degrees(np.sqrt(np.maximum(np.diag(cov)[:3], 0))) * SIGMA_SCALE["rot"]
    Ca = C[6:9, 6:9]
    axis_deg = float(np.degrees(math.sqrt(max(float(np.trace(Ca)), 0.0)))) * SIGMA_SCALE["rot"]
    return {"pos_mm": pos.tolist(), "rot_deg": rot.tolist(), "lead_mm": lead.tolist(), "axis_deg": axis_deg,
            "hessian_eig": w.tolist()}


def _result(T_est, T_prior, T_tcp, lead_obj, axis_obj, sig, ious, f, base, t0) -> dict:
    lead_e = T_est[:3, :3] @ lead_obj + T_est[:3, 3]
    lead_p = T_prior[:3, :3] @ lead_obj + T_prior[:3, 3]
    dR = T_est[:3, :3] @ T_prior[:3, :3].T
    turn = math.degrees(math.acos(float(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return {**base, "ok": True, "why": "", "T_tcp_obj": T_est.tolist(), "pose7": _pose7(T_tcp @ T_est),
            "lead_obj": lead_obj.tolist(), "axis_obj": axis_obj.tolist(),
            "lead_tcp": lead_e.tolist(), "axis_tcp": (T_est[:3, :3] @ axis_obj).tolist(),
            "sigma": {k: v for k, v in sig.items() if k != "hessian_eig"}, "hessian_eig": sig.get("hessian_eig"),
            "shift_mm": float(np.linalg.norm(T_est[:3, 3] - T_prior[:3, 3]) * 1000), "turn_deg": turn,
            "lead_shift_mm": ((lead_e - lead_p) * 1000).tolist(),
            "iou": {k: round(v, 4) for k, v in ious.items()}, "residual": round(float(f), 5),
            "t_ms": round((time.time() - t0) * 1000, 1)}


def _fail(T_prior, T_tcp, why, base, t0, lead_obj, axis_obj) -> dict:
    lead = T_prior[:3, :3] @ lead_obj + T_prior[:3, 3]
    return {**base, "ok": False, "why": why, "T_tcp_obj": T_prior.tolist(), "pose7": _pose7(T_tcp @ T_prior),
            "lead_obj": lead_obj.tolist(), "axis_obj": axis_obj.tolist(), "lead_tcp": lead.tolist(),
            "axis_tcp": (T_prior[:3, :3] @ axis_obj).tolist(), "sigma": None, "shift_mm": 0.0, "turn_deg": 0.0,
            "lead_shift_mm": [0.0, 0.0, 0.0], "iou": {}, "residual": None, "t_ms": round((time.time() - t0) * 1000, 1)}
