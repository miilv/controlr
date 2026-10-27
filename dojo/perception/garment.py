"""A garment lying flat on the table, read from the head camera: which garment, where, which way it faces, and its
named points (sleeves, chest, hems, shoulders: the asset's ``passive.functional`` particle ids).

A garment is cloth: no rigid pose fits its mesh once it has fallen. But it falls the same way every time from the same
spawn shape, so each asset's settled shape is one flat template (its particles and named points in its root frame,
from the twin: ``scripts/perception/garment_frames.py``). The read:

1. foreground: colour distance to the head background (the empty table at the home pose), robot pixels removed;
2. the foreground blob nearest the table's front middle, back-projected onto the plane the cloth lies in;
3. per template: the yaw (+- 45 deg about the spawn's) and shift that best overlap the template's footprint with the
   blob's on a 6 mm grid (IoU), coarse then fine; the best template wins;
4. the named points = the template's, moved by that yaw and shift.

Same dict shape as the twin's GT read (``RoboDojoRunner.garment_gt`` without ``points``), so the fold skill reads it
through ``provider.garment`` either way. Valid for the flat garment at the start: plan the folds from this one read.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .cameras import TABLE_Z, Camera, head_camera

CLOTH_Z = TABLE_Z + 0.006          # the plane the flat cloth's top surface lies in (particles 4-10 mm up)
GRID = 0.006                       # m: occupancy cells for the fit
TEMPLATES = Path(__file__).resolve().parent / "data" / "garment_templates.json"


def load_templates(path: str | Path | None = None) -> dict:
    """{asset key: {"xy": (N, 2) root-frame footprint samples, "kp": {name: (3,) root frame}}}."""
    p = Path(path or TEMPLATES)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {k: {"xy": np.asarray(v["xy"], float) / 1000.0, "kp": {n: np.asarray(x, float) / 1000.0
                                                                   for n, x in v["kp"].items()}}
            for k, v in raw.items()}


def foreground(img: np.ndarray, bg: np.ndarray | None, thr: float = 22.0, robot: np.ndarray | None = None) -> np.ndarray:
    """uint8 mask of pixels unlike the background (colour distance, light-weighted), minus the robot."""
    import cv2

    if bg is not None:
        from .segment import color_distance
        m = (color_distance(img, bg) > thr).astype(np.uint8)
    else:
        # no background: unlike the table's dominant colour (the median of the lower half)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref = np.median(lab[img.shape[0] // 2:].reshape(-1, 3), axis=0)
        m = (np.linalg.norm(lab - ref, axis=2) > thr).astype(np.uint8)
    if robot is not None:
        m[robot > 0] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def blob_points(mask: np.ndarray, cam: Camera, near_xy=(0.0, -0.12), max_dist: float = 0.25,
                region=((-0.45, 0.45), (-0.45, 0.25))) -> np.ndarray:
    """World xy (on ``CLOTH_Z``) of the connected foreground blob whose table-plane centroid is nearest ``near_xy``
    (within ``max_dist``) and inside ``region``; empty when none."""
    import cv2

    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    best, best_d = None, max_dist
    for i in range(1, n):
        vv, uu = np.nonzero(lab == i)
        if len(vv) < 400:
            continue
        P = cam.ray_plane(np.c_[uu + 0.5, vv + 0.5], CLOTH_Z)[:, :2]
        P = P[np.isfinite(P).all(1)]
        P = P[(P[:, 0] > region[0][0]) & (P[:, 0] < region[0][1]) & (P[:, 1] > region[1][0]) & (P[:, 1] < region[1][1])]
        if len(P) < 400:
            continue
        d = float(np.linalg.norm(P.mean(0) - np.asarray(near_xy, float)))
        if d < best_d:
            best, best_d = P, d
    return best if best is not None else np.zeros((0, 2))


def _cells(P: np.ndarray) -> set:
    return set(map(tuple, np.floor(P / GRID).astype(np.int64).tolist()))


def _iou(obs: set, P: np.ndarray) -> float:
    c = _cells(P)
    inter = len(c & obs)
    return inter / max(1, len(c | obs))


def _rot(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def fit_template(obs_xy: np.ndarray, tpl: dict, yaw0: float = 0.0, yaw_span: float = math.radians(45)) -> dict:
    """The yaw (rad) and shift that put the template's footprint on the observed one: coarse yaw grid with centroid
    alignment, then a local search over shift and yaw. {"yaw", "t", "iou"}."""
    obs = _cells(obs_xy)
    c_o = obs_xy.mean(0)
    T = tpl["xy"][:: max(1, len(tpl["xy"]) // 2500)]
    c_t = T.mean(0)
    best = (-1.0, 0.0, np.zeros(2))
    for th in np.arange(yaw0 - yaw_span, yaw0 + yaw_span + 1e-9, math.radians(3)):
        R = _rot(th)
        t = c_o - R @ c_t
        s = _iou(obs, T @ R.T + t)
        if s > best[0]:
            best = (s, th, t)
    s, th, t = best
    for step_t, step_th in ((0.012, math.radians(2)), (0.006, math.radians(1)), (0.003, math.radians(0.5))):
        improved = True
        while improved:
            improved = False
            for dth in (-step_th, 0.0, step_th):
                for dx in (-step_t, 0.0, step_t):
                    for dy in (-step_t, 0.0, step_t):
                        if dth == dx == dy == 0.0:
                            continue
                        R = _rot(th + dth)
                        t2 = t + np.array([dx, dy])
                        s2 = _iou(obs, T @ R.T + t2)
                        if s2 > s + 1e-6:
                            s, th, t, improved = s2, th + dth, t2, True
    return {"yaw": float(th), "t": t, "iou": float(s)}


def read_garment(img: np.ndarray, templates: dict, cam: Camera | None = None, bg: np.ndarray | None = None,
                 robot: np.ndarray | None = None, assets: list[str] | None = None, near_xy=(0.0, -0.12)) -> dict | None:
    """The garment read from one head frame: {"asset", "pos", "quat", "keypoints", "iou", "n_px"} or None."""
    cam = cam or head_camera()
    m = foreground(img, bg, robot=robot)
    P = blob_points(m, cam, near_xy=near_xy)
    if len(P) < 400:
        return None
    best = None
    for key, tpl in templates.items():
        if assets and key not in assets:
            continue
        f = fit_template(P, tpl)
        if best is None or f["iou"] > best[1]["iou"]:
            best = (key, f)
    if best is None:
        return None
    key, f = best
    R2 = _rot(f["yaw"])
    kp = {}
    for n, x in templates[key]["kp"].items():
        xy = R2 @ np.asarray(x[:2], float) + f["t"]
        kp[n] = [float(xy[0]), float(xy[1]), float(TABLE_Z + 0.008)]
    half = f["yaw"] / 2
    return {"asset": key, "pos": [float(f["t"][0]), float(f["t"][1]), float(TABLE_Z)],
            "quat": [math.cos(half), 0.0, 0.0, math.sin(half)], "keypoints": kp, "iou": f["iou"], "n_px": int(len(P))}


def build_templates(frames_root: str | Path, out: str | Path | None = None, n: int = 2500) -> dict:
    """Templates from ``garment_frames.py``'s ``templates/*.npz`` (root frame, metres) -> the JSON this module loads
    (mm, footprint subsampled to ``n`` points, named points)."""
    root = Path(frames_root).expanduser() / "templates"
    res = {}
    for p in sorted(root.glob("*.npz")):
        z = np.load(p)
        P = np.asarray(z["points"], float)
        idx = np.random.default_rng(0).choice(len(P), size=min(n, len(P)), replace=False)
        key = p.stem.replace("_", "/", 2)
        res[key] = {"xy": np.round(P[idx, :2] * 1000).astype(int).tolist(),
                    "kp": {str(k): np.round(np.asarray(v, float) * 1000, 1).tolist()
                           for k, v in zip(z["kp_names"], z["kp"])}}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res))
    return res
