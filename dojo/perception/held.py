"""In-hand pose of a held object: render-and-compare in every camera that sees it, constrained by the gripper.

    from dojo.perception.held import held_pose
    r = held_pose(frame, "left", "Rigid/charger/00000", prior_T_tcp_obj)
    r["ok"], r["T_tcp_obj"], r["sigma"]       # 4x4 in the skills' TCP frame; per-axis sigma (mm / deg)

Inputs are the official observation only: 3 RGB frames, measured joints (FK -> wrist cameras, TCP, finger meshes),
the commanded gripper and the published camera / robot config. The prior is the skills' predicted in-hand transform
(``Held.T_tcp_obj``: object root in the TCP frame; TCP = FK link6 @ +``TCP_OFFSET`` along flange x; x = approach,
y = closing axis).

Method (B1 machinery: ``render`` silhouettes and crop cameras):
- Views: the holding arm's own wrist camera (the object at 6-15 cm: 0.15-0.4 mm per pixel; rigid with the TCP, so
  the arm's own FK error cancels), the head camera and the other arm's wrist camera, each used when enough of the
  object's silhouette is visible (not under the robot, inside the image).
- Robot pixels (both arms from FK; the holding arm's fingers over the range of openings the object allows) are
  unknown: they neither count as object nor as background.
- Foreground per view: P(object | colour) from two Lab histograms learnt in the frame itself: the object inside the
  expected silhouette (eroded by the prior's error), the background in a ring outside it; re-learnt around the fit.
  (A one-class table model, ``refine.local_foreground``, missed a light-wood block on the wood table.)
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
ROI_MAX = 320                      # px of a view's ROI render (supersampled up to x3 for small objects)
MIN_VIS_PX = 150                   # visible silhouette pixels (native) for a view to count
MIN_VIS_FRAC = 0.25                # visible share of the silhouette
PAD_LEN = 0.030                    # m of each finger (from the tip) that can touch the object
CENTRE_SIGMA_M = 0.0015            # soft gripper term: the contact faces centred on the TCP along the closing axis
# raw sigma -> calibrated sigma, per geometry class ("spin": a continuous symmetry axis, i.e. tubes, pens, coins; else
# "rigid"); fitted 2-fold by layout on our dev frames (held_eval report: calibration), held-out coverage in the report
SIGMA_SCALE = {"rigid": {"pos": 1.21, "rot": 1.31}, "spin": {"pos": 1.62, "rot": 1.61}}
SIGMA_DF = 0.002                   # cost rise below which poses are indistinguishable (calibrated with SIGMA_SCALE)
SIGMA_SAMPLES = 70
OK_IOU = 0.70                      # best view's IoU for an ok fit (a gross mismatch; precision is the sigma's job)
DROP_VIEW_IOU = 0.60               # a view below this IoU at the first fit is dropped when another view is good
OK_MAX_LEAD_SIGMA_MM = 1.0        # calibrated 1-sigma of the leading point ACROSS the insertion axis
OK_MAX_AXIS_SIGMA_DEG = 2.0        # calibrated 1-sigma of the insertion axis direction
VIEW_W = {"own": 1.0, "head": 0.6, "observer": 1.0}
HIST_BITS = 4                      # Lab histogram: 16 levels per channel
PRIOR_ERR_M = 0.004                # expected prior error: the object model is learnt this far inside the prior's outline
RING_M = 0.006                     # background ring width beyond that
FINGER_BAND_M = 0.0025             # the holding fingers are masked over openings for widths +- this
EDGE_CAP_PX = 8.0                  # outline-to-edge distance cap (ROI px)
W_EDGE = 0.5                       # weight of the outline term against 1 - IoU
W_CREASE = 0.3                     # weight of the crease term (inner mesh edges vs image edges)
W_SHADE = 0.0                      # weight of the shading term (flat-shaded render vs image, NCC inside the silhouette)
CREASE_DEG = 40.0
W_PRIOR = 0.01                     # cost units per sigma^2 of the gripper / off-pad priors
OFFPAD_ROT_SIGMA = math.radians(1.5)
OFFPAD_Y_SIGMA_M = 0.0008
OBJ_MODEL_ERODE = 0.35             # the first object colour model: the prior's silhouette eroded by this x prior error


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


@lru_cache(maxsize=16)
def surface_points(key: str, n: int = 6000) -> np.ndarray:
    """Area-weighted samples of the asset's surface (object frame): a box's flat sides have vertices only at their
    edges, so vertex-based pad tests missed the faces the pads actually touch."""
    a = load_asset(key)
    V, F = a.V, np.asarray(a.F, np.int64)
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(F), n, p=area / max(area.sum(), 1e-12))
    r1, r2 = rng.random((2, n))
    sq = np.sqrt(r1)
    return (1 - sq)[:, None] * tri[idx, 0] + (sq * (1 - r2))[:, None] * tri[idx, 1] + (sq * r2)[:, None] * tri[idx, 2]


def pad_width(P_obj: np.ndarray, T_tcp_obj) -> tuple[float, np.ndarray]:
    """Width of the object between the pads (m) and the pad-zone mask of the points: their extent along the closing
    axis where the pads can touch; the whole extent when nothing lies in the zone."""
    Pt = transform(P_obj, np.asarray(T_tcp_obj, float))
    tbl = finger_table()
    tip_x = float(np.interp(0.5, tbl[:, 0], tbl[:, 2]))
    zone = pad_zone(Pt, tip_x)
    if zone.sum() >= 20:
        return float(np.ptp(Pt[zone, 1])), zone
    return float(np.ptp(Pt[:, 1])), zone


def est_opening(key: str, T_tcp_obj) -> float:
    """The normalised finger opening at which the pads touch an object held at ``T_tcp_obj``."""
    return opening_for_width(pad_width(surface_points(key), T_tcp_obj)[0])


def pad_zone(V_tcp: np.ndarray, tip_x: float, half_h: float = 0.012) -> np.ndarray:
    """Object points the pads can touch: within PAD_LEN of the fingertip along x and near the pad's mid-plane."""
    return (V_tcp[:, 0] > tip_x - PAD_LEN) & (V_tcp[:, 0] < tip_x + 0.002) & (np.abs(V_tcp[:, 2]) < half_h)


@lru_cache(maxsize=16)
def crease_points(key: str, deg: float = None):
    """Points sampled every ~1 mm along the mesh's crease edges (dihedral angle > ``CREASE_DEG``) with both adjacent
    face normals: inner edges between faces (a cube's top / front faces) that the outline misses. In the object frame."""
    deg = CREASE_DEG if deg is None else deg
    a = load_asset(key)
    V, F = a.Vs, np.asarray(a.Fs, np.int64)
    N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    nn = np.linalg.norm(N, axis=1)
    ok = nn > 1e-12
    N[ok] /= nn[ok, None]
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fi = np.tile(np.arange(len(F)), 3)
    Es = np.sort(E, axis=1)
    order = np.lexsort((Es[:, 1], Es[:, 0]))
    Es, fi = Es[order], fi[order]
    same = np.all(Es[1:] == Es[:-1], axis=1)
    i = np.nonzero(same)[0]
    f1, f2 = fi[i], fi[i + 1]
    good = ok[f1] & ok[f2]
    cosang = np.einsum("ij,ij->i", N[f1], N[f2])
    sel = good & (cosang < math.cos(math.radians(deg)))
    e, f1, f2 = Es[i][sel], f1[sel], f2[sel]
    P, N1, N2 = [], [], []
    for (u, v), a1, a2 in zip(e, f1, f2):
        L = float(np.linalg.norm(V[v] - V[u]))
        n = max(2, int(math.ceil(L / 0.001)) + 1)
        t = np.linspace(0.0, 1.0, n)[1:-1, None] if n > 2 else np.array([[0.5]])
        P.append(V[u] + t * (V[v] - V[u]))
        N1.append(np.repeat(N[a1][None], len(t), 0))
        N2.append(np.repeat(N[a2][None], len(t), 0))
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(P), np.concatenate(N1), np.concatenate(N2)


def _crease_cost(Pw, N1w, N2w, v) -> float | None:
    """Mean truncated distance (cap units) from the visible crease points (both faces toward the camera) to image
    edges in view ``v`` (bilinear lookup in the edge distance map); None when fewer than 20 points count."""
    if len(Pw) == 0:
        return None
    c = v.cam.T_wc[:3, 3]
    d = Pw - c
    vis = (np.einsum("ij,ij->i", N1w, d) < 0) & (np.einsum("ij,ij->i", N2w, d) < 0)
    if vis.sum() < 20:
        return None
    uv, z = v.cam.project(Pw[vis])
    s = v.size
    inb = (z > 0.01) & (uv[:, 0] >= 1) & (uv[:, 1] >= 1) & (uv[:, 0] < s - 2) & (uv[:, 1] < s - 2)
    uv = uv[inb]
    if len(uv) < 20:
        return None
    x, y = uv[:, 0] - 0.5, uv[:, 1] - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    ok = v.outline_ok[y0, x0]
    if ok.sum() < 20:
        return None
    D = v.edge_D
    val = (D[y0, x0] * (1 - fx) * (1 - fy) + D[y0, x0 + 1] * fx * (1 - fy) + D[y0 + 1, x0] * (1 - fx) * fy
           + D[y0 + 1, x0 + 1] * fx * fy)
    return float(val[ok].mean()) / EDGE_CAP_PX


# ----------------------------------------------------------------------------------------------- views
class View:
    """One camera restricted to a square ROI around the object (resampled to ``size``), with a two-class colour model
    (object vs background, Lab histograms) learnt from the frame around a silhouette (``fit_models``)."""

    def __init__(self, name, role, cam: Camera, img, known, crop, size):
        self.name, self.role, self.cam_full = name, role, cam
        x0, y0, x1, y1 = crop
        self.crop = crop
        self.size = size
        self.cam = crop_camera(cam, x0, y0, x1, y1, size)
        interp = cv2.INTER_AREA if (x1 - x0) > size else cv2.INTER_LINEAR
        self.known = cv2.resize(_crop(known.astype(np.float32), crop), (size, size), interpolation=interp)
        self.known = (self.known > 0.5).astype(np.float32)
        self.rgb = cv2.resize(_crop(img, crop), (size, size), interpolation=interp)
        self.gray = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lab = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2LAB)
        q = (lab.astype(np.int32) >> (8 - HIST_BITS))
        self.bins = (q[..., 0] << (2 * HIST_BITS)) | (q[..., 1] << HIST_BITS) | q[..., 2]
        self.px_scale = (x1 - x0) / size     # native px per ROI px
        self.img = img
        self.fg = np.zeros((size, size), np.float32)
        self.domain = np.ones((size, size), np.float32)
        self.model_px = (0, 0)
        # image edges (Canny on L, a*, b*: a white part on light wood differs in b*, not in grey level), robot edges
        # removed; distance to the nearest edge, truncated (the outline term)
        from .refine import image_edges
        E = image_edges(self.rgb)
        kn = cv2.erode((self.known > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8))
        E[kn == 0] = 0
        self.edge_D = np.minimum(cv2.distanceTransform((E == 0).astype(np.uint8), cv2.DIST_L2, 3),
                                 EDGE_CAP_PX).astype(np.float32)
        self.outline_ok = kn.astype(bool)
        self.outline_ok[:3, :] = self.outline_ok[-3:, :] = False
        self.outline_ok[:, :3] = self.outline_ok[:, -3:] = False

    def fit_models(self, S: np.ndarray, e_in: float, band: float) -> bool:
        """Object model from the silhouette ``S`` (ROI px) eroded by ``e_in`` px, background model from the ring
        ``e_in`` .. ``e_in + band`` px outside it (robot / out-of-image pixels excluded); ``fg`` = P(object | colour)."""
        Sb = (S > 0.5).astype(np.uint8)
        din = cv2.distanceTransform(Sb, cv2.DIST_L2, 3)
        dout = cv2.distanceTransform(1 - Sb, cv2.DIST_L2, 3)
        k = self.known > 0.5
        e = float(e_in)
        if din.max() > 0 and ((din > e) & k).sum() < 0.25 * max(1, (Sb.astype(bool) & k).sum()):
            e = 0.5 * float(din.max())                 # a thin object: keep its core
        fgm = (din > e) & k
        bgm = (dout > e) & (dout <= e + band) & k
        # the IoU is scored only near the expected outline: far background of a similar colour (a light wall behind a
        # white tube seen from the side) had P(object) ~ 0.5 and inflated the union (observer views dropped at IoU 0.3)
        self.domain = (dout <= e + band).astype(np.float32)
        nf, nb = int(fgm.sum()), int(bgm.sum())
        self.model_px = (nf, nb)
        if nf < 30 or nb < 30:
            return False
        nbins = 1 << (3 * HIST_BITS)
        hf = np.bincount(self.bins[fgm], minlength=nbins).astype(np.float64) / nf
        hb = np.bincount(self.bins[bgm], minlength=nbins).astype(np.float64) / nb
        p = (hf + 1e-4) / (hf + hb + 2e-4)
        self.fg = p[self.bins].astype(np.float32)
        return True


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

    def shaded(self, cam: Camera, T: np.ndarray) -> np.ndarray:
        """Flat-shaded render (painter's order, CPU): grey = the facet's normal toward a light from above and the
        camera; 0 outside. Only the ORDER of facet brightness matters (it is compared by NCC): a cube's lit top and
        darker sides, i.e. the inner shading edges the outline misses (the turn about the closing axis)."""
        V = transform(self.a.Vs, T)
        F = np.asarray(self.a.Fs)
        N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        c = cam.T_wc[:3, 3]
        ctr = V[F].mean(1)
        to_cam = c - ctr
        dist = np.linalg.norm(to_cam, axis=1)
        to_cam /= np.maximum(dist[:, None], 1e-12)
        front = np.einsum("ij,ij->i", N, to_cam) > 0
        L = np.array([0.0, 0.0, 1.0])
        shade = 0.25 + 0.6 * np.clip(N @ L, 0, 1) + 0.35 * np.clip(np.einsum("ij,ij->i", N, to_cam), 0, 1)
        uv, z = cam.project(V)
        img = np.zeros((cam.h, cam.w), np.float32)
        P = ((uv[F] - 0.5) * 16).round().astype(np.int32)
        idx = np.nonzero(front & (z[F] > 0.01).all(1))[0]
        for i in idx[np.argsort(-dist[idx])]:
            cv2.fillConvexPoly(img, P[i], float(shade[i]), lineType=cv2.LINE_8, shift=4)
        return img


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
        known = (1 - rob).astype(np.uint8)
        for k2, T2 in others or ():            # known objects (and the other arm's held one): unknown pixels
            a2 = load_asset(k2)
            known &= 1 - cv2.dilate(silhouette(cam, transform(a2.Vs, T2), a2.Fs), np.ones((5, 5), np.uint8))
        roi = _roi_for(cam, bbw, max_shift_m + 0.01)
        if roi is None:
            diag[name] = {"used": False, "why": "roi"}
            continue
        side = roi[2] - roi[0]
        size = int(np.clip(side * 3, 96, ROI_MAX))
        v = View(name, role, cam, images[name], known, roi, size)
        out.append(v)
        diag[name] = {"used": True, "role": role, "visible_frac": round(vis, 3), "px": n, "roi_px": side,
                      "roi_size": size}
    return out, diag


def _iou(s: np.ndarray, v: View) -> float:
    k = v.known * v.domain
    inter = float((s * v.fg * k).sum())
    union = float(((s + v.fg - s * v.fg) * k).sum())
    return inter / union if union > 0 else 0.0


_K3 = np.ones((3, 3), np.uint8)


def _edge_cost(s: np.ndarray, v: View) -> float | None:
    """Mean truncated distance (in units of the cap) from the silhouette's visible outline to image edges."""
    sb = (s > 0.5).astype(np.uint8)
    O = (sb > 0) & (cv2.erode(sb, _K3) == 0) & v.outline_ok
    n = int(O.sum())
    if n < 20:
        return None
    return float(v.edge_D[O].mean()) / EDGE_CAP_PX


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
    # the object between the pads: its width there sets the finger opening; the fingers over +-FINGER_BAND_M are unknown
    Ps = surface_points(asset_key)
    width, zone = pad_width(Ps, T_prior)
    o_est = opening_for_width(width)
    o_lo, o_hi = opening_for_width(max(0.0, width - FINGER_BAND_M)), opening_for_width(width + FINGER_BAND_M)
    other_g = state.get(f"{'right' if arm == 'left' else 'left'}_ee_joint_state")
    other_g = None if other_g is None else float(other_g[0])
    others = []
    for o in cfg.get("others") or ():
        k2, p2 = (o["key"], o["pose7"]) if isinstance(o, dict) else (o[0], o[1])
        others.append((k2, pose_to_T(p2[:3], p2[3:])))
    oh = cfg.get("other_held")
    if oh:
        oq = state.get(f"{oh['arm']}_arm_joint_state")
        if oq is not None:
            others.append((oh["key"], tcp_T(oh["arm"], oq[:6], model) @ np.asarray(oh["T_tcp_obj"], float)))
    views, diag = _views(images, state, arm, sil, T_tcp, T_prior, cams, o_lo, o_hi, other_g, max_shift,
                         others=others, exclude_cams=tuple(cfg.get("exclude_cams", ())))
    lead_obj, axis_obj = _lead_axis(a, T_prior, cfg)
    base = {"version": VERSION, "views": [v.name for v in views], "view_diag": diag, "opening_est": round(o_est, 3),
            "width_mm": round(width * 1000, 2)}
    if not views:
        return _fail(T_prior, T_tcp, "no_views", base, t0, lead_obj, axis_obj)
    c = (np.asarray(a.info["bbox_min"]) + np.asarray(a.info["bbox_max"])) / 2
    c_tcp = T_prior[:3, :3] @ c + T_prior[:3, 3]
    # pad-zone points for the gripper term (a subsample)
    Vp = Ps[zone] if zone.sum() >= 20 else None
    if Vp is not None and len(Vp) > 400:
        Vp = Vp[np.random.default_rng(0).choice(len(Vp), 400, replace=False)]
    w_views = np.array([VIEW_W[v.role] for v in views])
    w_shade = [float(cfg.get("w_shade", W_SHADE))]
    Pc, N1c, N2c = crease_points(asset_key) if W_CREASE > 0 else (np.zeros((0, 3)),) * 3
    if len(Pc) > 1500:
        sel = np.random.default_rng(0).choice(len(Pc), 1500, replace=False)
        Pc, N1c, N2c = Pc[sel], N1c[sel], N2c[sel]

    def cost(d, detail=False):
        T = apply_delta(T_prior, c_tcp, d)
        Tw = T_tcp @ T
        tot, ious = 0.0, {}
        Pw = transform(Pc, Tw) if len(Pc) else Pc
        N1w, N2w = (N1c @ Tw[:3, :3].T, N2c @ Tw[:3, :3].T) if len(Pc) else (Pc, Pc)
        for v, w in zip(views, w_views):
            if w <= 0:
                continue
            s = sil.render(v.cam, Tw)
            iou = _iou(s, v)
            ious[v.name] = iou
            ec = _edge_cost(s, v) if W_EDGE > 0 else None
            cc = _crease_cost(Pw, N1w, N2w, v) if len(Pc) else None
            tot += w * ((1.0 - iou) + (W_EDGE * ec if ec is not None else W_EDGE * 0.5)
                        + (W_CREASE * cc if cc is not None else 0.0))
            if w_shade[0] > 0 and v.role != "head":
                sh = sil.shaded(v.cam, Tw)
                m = (sh > 0) & (v.known > 0.5)
                if m.sum() > 200:
                    a_, b_ = sh[m], v.gray[m]
                    if a_.std() > 1e-3 and b_.std() > 1.0:
                        ncc = float(((a_ - a_.mean()) * (b_ - b_.mean())).mean() / (a_.std() * b_.std()))
                        tot += w * w_shade[0] * (1.0 - ncc) / 2
        tot /= max(float(w_views.sum()), 1e-9)
        pen = 0.0
        if Vp is not None:
            P = transform(Vp, T)
            mid = 0.5 * (P[:, 1].max() + P[:, 1].min())
            pen = 0.5 * (mid / CENTRE_SIGMA_M) ** 2 * W_PRIOR
        # the pads hold flat faces / a cylinder's sides parallel to them: turns about x / z (and the slide along y)
        # stay near the prior's; a weak Gaussian prior (MAP), the free in-pad DoF (x, z, turn about y) have none
        pen += 0.5 * W_PRIOR * ((d[0] / OFFPAD_ROT_SIGMA) ** 2 + (d[2] / OFFPAD_ROT_SIGMA) ** 2
                                + (d[4] / OFFPAD_Y_SIGMA_M) ** 2)
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

    def models(x, e_m, band_m):
        Tw = T_tcp @ apply_delta(T_prior, c_tcp, x)
        n_ok = 0
        for v in views:
            depth = float(np.median(v.cam.project(transform(a.Vs[::max(1, len(a.Vs) // 200)], Tw))[1]))
            px_m = v.cam.K[0, 0] / max(depth, 1e-3)               # ROI px per metre at the object
            n_ok += v.fit_models(sil.render(v.cam, Tw), e_m * px_m, max(6.0, band_m * px_m))
        return n_ok

    # 1) colour models around the prior; pad-plane starts: turn about the closing axis (TCP y), slide along approach
    #    (x) / across (z); Nelder-Mead in 6 DoF from the best three
    if models(np.zeros(6), OBJ_MODEL_ERODE * float(cfg.get("prior_err_mm", PRIOR_ERR_M * 1000)) / 1000, RING_M) == 0:
        return _fail(T_prior, T_tcp, "no_views", base, t0, lead_obj, axis_obj)
    # a grid over the in-pad DoF: the turn about the closing axis and the slide along the approach trade off in the
    # own wrist view (a flat valley of the cost), so seeds span it rather than a few axis-aligned steps
    starts = [np.array([0, ry * deg, 0, dx * mm, 0, dz * mm], float)
              for ry in (-6, -3, 0, 3, 6) for dx in (-3, 0, 3) for dz in (-2, 0, 2)]
    scored = sorted(((cost(s), i, s) for i, s in enumerate(starts)), key=lambda t: t[:2])
    best = None
    free = (1, 3, 5)                                    # the in-pad DoF: turn about y, slide along x / z

    def cost3(p):
        d = np.zeros(6)
        d[list(free)] = p
        return cost(d)
    for f0, _, s in scored[:3]:
        r3 = minimize(cost3, s[list(free)], method="Nelder-Mead",
                      options={"initial_simplex": np.array([s[list(free)]] + [s[list(free)] + e for e in
                                                            np.diag([3 * deg, 2 * mm, 2 * mm])]),
                               "maxiter": 200, "xatol": 1e-5, "fatol": 1e-5})
        x0 = np.zeros(6)
        x0[list(free)] = r3.x
        x, f = nm(x0, [1.5 * deg, 1.5 * deg, 1.5 * deg, 1 * mm, 0.5 * mm, 1 * mm], 400)
        if best is None or f < best[1]:
            best = (x, f)
    # 2) colour models re-learnt around the fit (now within a few px), then a 6-DoF polish with small steps. A view
    # whose silhouette does not match at the fit (a far head view with a same-colour receptacle behind the object)
    # is dropped when a good view remains: its edges pulled the fit off
    models(best[0], 0.0008, RING_M)
    _, ious_b = cost(best[0], detail=True)
    if len(views) > 1 and max(ious_b.values()) >= OK_IOU:
        for i, v in enumerate(views):
            if ious_b.get(v.name, 0.0) < DROP_VIEW_IOU:
                w_views[i] = 0.0
                diag[v.name]["dropped"] = round(ious_b.get(v.name, 0.0), 3)
    f_b = cost(best[0])
    x, f = nm(best[0], [1.0 * deg, 1.0 * deg, 1.0 * deg, 0.6 * mm, 0.4 * mm, 0.6 * mm], 500)
    if f > f_b:
        x, f = best[0], f_b
    models(x, 0.0005, RING_M)
    x2, f2 = nm(x, [0.4 * deg, 0.4 * deg, 0.4 * deg, 0.2 * mm, 0.2 * mm, 0.2 * mm], 300)
    if f2 <= cost(x):
        x = x2
    f, ious = cost(x, detail=True)
    T_est = apply_delta(T_prior, c_tcp, x)
    sp = spin_axis(asset_key)
    sig = _sigma(cost, x, T_prior, c_tcp, lead_obj, axis_obj, f,
                 spin_tcp=None if sp is None else apply_delta(T_prior, c_tcp, x)[:3, :3] @ sp,
                 scale=SIGMA_SCALE["spin" if sp is not None else "rigid"])
    base["geometry_class"] = "spin" if sp is not None else "rigid"
    out = _result(T_est, T_prior, T_tcp, lead_obj, axis_obj, sig, ious, f, base, t0)
    best_iou = max(ious.values()) if ious else 0.0
    why = ""
    if best_iou < OK_IOU:
        why = "fit_poor"
    elif float(np.linalg.norm(x[3:])) > 0.98 * max_shift or float(np.linalg.norm(x[:3])) > 0.98 * max_turn:
        why = "out_of_range"
    elif sig["lead_lat_mm"] > OK_MAX_LEAD_SIGMA_MM or sig["axis_deg"] > OK_MAX_AXIS_SIGMA_DEG:
        why = "ambiguous"
    out["ok"] = why == ""
    out["why"] = why
    if cfg.get("debug"):
        out["_debug"] = {"views": views, "sil": sil, "T_tcp": T_tcp}
    if not out["ok"]:
        out["fit_T_tcp_obj"] = out["T_tcp_obj"]
        out["T_tcp_obj"] = T_prior.tolist()
        out["pose7"] = _pose7(T_tcp @ T_prior)
    return out


def _lead_axis(a, T_prior, cfg):
    """Insertion geometry in the object frame: given (cfg lead_obj / axis_obj), or from the place target
    (cfg T_obj_target: the axis = world down there, the leading point = the lowest mesh points there), else the end of
    the mesh along the approach (TCP +x) at the prior and the principal mesh axis nearest that direction."""
    if cfg.get("T_obj_target") is not None and cfg.get("lead_obj") is None:
        from .held_data import lead_axis_from_target
        return lead_axis_from_target(a.V, np.asarray(cfg["T_obj_target"], float).reshape(4, 4))
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


@lru_cache(maxsize=32)
def spin_axis(key: str):
    """The object-frame axis (through the bbox centre) about which the mesh is continuously symmetric (a tube, a pen, a
    coin), or None. A turn about it changes no image, so it is left out of the uncertainty."""
    from .inverse_layout import world_z_symmetry

    a = load_asset(key)
    c = (np.asarray(a.info["bbox_min"]) + np.asarray(a.info["bbox_max"])) / 2
    V = a.Vs - c
    for k in range(3):
        z = np.zeros(3)
        z[k] = 1.0
        x = np.zeros(3)
        x[(k + 1) % 3] = 1.0
        R = np.stack([x, np.cross(z, x), z])                 # rows: the object axis k becomes z
        if world_z_symmetry(V @ R.T, np.asarray(a.Fs)) >= 36:
            return z
    return None


def _sigma(cost, x, T_prior, c_tcp, lead_obj, axis_obj, f0, spin_tcp=None, scale=None) -> dict:
    """Uncertainty of the fit: a full quadratic model of the cost fitted to ``SIGMA_SAMPLES`` random poses around the
    optimum (robust to the IoU cost's pixel steps, unlike finite differences), its Hessian H (eigenvalues floored),
    cov = 2 * SIGMA_DF * H^-1 (the poses whose cost rises by less than SIGMA_DF are indistinguishable), mapped to the
    TCP position / rotation, the leading point and the insertion axis; times SIGMA_SCALE (both calibrated on dev
    captures, ``held_eval``)."""
    S = np.array([math.radians(1.0)] * 3 + [0.0006, 0.0004, 0.0006])     # sampling half-widths per DoF
    # the sampled subspace: all 6 DoF, minus the turn about a symmetry axis (no image changes, the outputs neither)
    B = np.eye(6)
    if spin_tcp is not None:
        v = np.r_[np.asarray(spin_tcp, float) / math.radians(1.0), np.zeros(3)]   # normalised coordinates
        v /= np.linalg.norm(v)
        Q, _ = np.linalg.qr(np.c_[v, np.eye(6)])
        B = Q[:, 1:6]
    m = B.shape[1]
    rng = np.random.default_rng(1)
    U = rng.uniform(-1.0, 1.0, (SIGMA_SAMPLES, m))
    fs = np.array([cost(x + (B @ u) * S) for u in U])
    iu = np.triu_indices(m)
    A = np.c_[np.ones(len(U)), U, 0.5 * np.array([np.outer(u, u)[iu] * np.where(iu[0] == iu[1], 1.0, 2.0) for u in U])]
    coef, *_ = np.linalg.lstsq(A, fs, rcond=None)
    Hn = np.zeros((m, m))
    Hn[iu] = coef[1 + m:]
    Hn = Hn + np.triu(Hn, 1).T
    w, V = np.linalg.eigh(Hn)
    w = np.maximum(w, 1e-2 * max(float(w.max()), 1e-6))
    cov_n = B @ (V @ np.diag(2.0 * SIGMA_DF / w) @ V.T) @ B.T
    cov = cov_n * np.outer(S, S)
    def outputs(T):
        lead = T[:3, :3] @ lead_obj + T[:3, 3]
        ax = T[:3, :3] @ axis_obj
        return np.r_[T[:3, 3], lead, ax]
    o0 = outputs(apply_delta(T_prior, c_tcp, x))
    J = np.zeros((9, 6))
    for i in range(6):
        e = np.zeros(6)
        e[i] = 1e-6
        J[:, i] = (outputs(apply_delta(T_prior, c_tcp, x + e)) - o0) / 1e-6
    C = J @ cov @ J.T
    a = o0[6:9] / max(float(np.linalg.norm(o0[6:9])), 1e-12)
    Pl = np.eye(3) - np.outer(a, a)                       # across the insertion axis at the leading point
    Cl = C[3:6, 3:6]
    kp, kr = (scale or SIGMA_SCALE["rigid"])["pos"], (scale or SIGMA_SCALE["rigid"])["rot"]
    lat = math.sqrt(max(float(np.trace(Pl @ Cl @ Pl)), 0.0)) * 1000 * kp
    along = math.sqrt(max(float(a @ Cl @ a), 0.0)) * 1000 * kp
    pos = np.sqrt(np.maximum(np.diag(C)[:3], 0)) * 1000 * kp
    lead = np.sqrt(np.maximum(np.diag(C)[3:6], 0)) * 1000 * kp
    rot = np.degrees(np.sqrt(np.maximum(np.diag(cov)[:3], 0))) * kr
    axis_deg = float(np.degrees(math.sqrt(max(float(np.trace(C[6:9, 6:9])), 0.0)))) * kr
    return {"pos_mm": pos.tolist(), "rot_deg": rot.tolist(), "lead_mm": lead.tolist(), "lead_lat_mm": lat,
            "lead_along_mm": along, "axis_deg": axis_deg, "hessian_eig": w.tolist()}


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
