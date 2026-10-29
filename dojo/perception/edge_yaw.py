"""Yaw of a four-fold symmetric box (a cube) from the image edges along its outline.

A 4-5 cm cube's silhouette barely constrains its yaw: B1's silhouette fit left align_blocks cubes 17.8 and 35 deg off
(5.1 camera 100021 / 100024); the pushes planned on those yaws left the row 30 deg off while the belief read it square.
The cube's outline in the image is sharp, though, and it changes with the yaw: here each yaw in the 90 deg period is
scored by the image gradient magnitude along the rendered outline band (other objects and the robot masked), summed
over the views. The yaw changes only for a clear, distinct peak: at least ``EDGE_GAIN`` better than the current yaw and
``EDGE_PEAK`` better than the best yaw at least ``RIVAL_DEG`` away. Nothing is keyed on a task or an asset: any shape
whose resting orientation has four-fold symmetry (``ShapeModel.sym == 4``)."""
from __future__ import annotations

import cv2
import numpy as np

STEP_DEG = 3.0       # coarse yaw grid over the period
FINE_DEG = 0.5       # ... refined around the best
BAND_PX = 1          # outline band half-width (px)
SHIFT_M = (-0.002, 0.0, 0.002)  # xy jitter (m) tried with every yaw: the fit's xy error
EDGE_GAIN = 0.05     # the best yaw must beat the current one by this share of its score
EDGE_PEAK = 0.05     # ... and the best yaw at least RIVAL_DEG away by this share
RIVAL_DEG = 20.0
MIN_BAND_PX = 40     # a view with fewer usable outline pixels is skipped


def gradient(img: np.ndarray) -> np.ndarray:
    """Gradient magnitude of the LAB lightness (float32), lightly smoothed."""
    L = cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    L = cv2.GaussianBlur(L, (3, 3), 0)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _band(sil: np.ndarray) -> np.ndarray:
    k = np.ones((2 * BAND_PX + 1, 2 * BAND_PX + 1), np.uint8)
    return (cv2.dilate(sil, k) & (1 - cv2.erode(sil, k))).astype(bool)


def score_yaws(sm, x: float, y: float, z_off: float, yaws, views: dict, others: dict, grads: dict) -> np.ndarray:
    """Mean gradient along the outline band per yaw, pooled over the views (usable pixels: known, not robot, not
    another object's)."""
    num = np.zeros(len(yaws))
    den = np.zeros(len(yaws))
    for name, v in views.items():
        G = grads.get(name)
        if G is None:
            continue
        usable = (np.asarray(v.known) > 0) & (np.asarray(v.robot) == 0)
        if name in others:
            usable &= np.asarray(others[name]) == 0
        for i, yaw in enumerate(yaws):
            b = _band(sm.sil(v.cam, sm.pose(x, y, float(yaw), z_off))) & usable
            n = int(b.sum())
            if n < MIN_BAND_PX:
                continue
            num[i] += float(G[b].sum())
            den[i] += n
    return np.where(den > 0, num / np.maximum(den, 1), 0.0)


def edge_yaw(sm, x: float, y: float, yaw: float, z_off: float, views: dict, others: dict, grads: dict) -> dict | None:
    """{"yaw", "from", "score", "score_cur", "rival", "changed"} for a four-fold symmetric shape, else None."""
    if getattr(sm, "sym", 0) != 4:
        return None
    period = np.pi / 2
    coarse = yaw + np.radians(np.arange(-45.0, 45.0, STEP_DEG))
    shifts = [(dx, dy) for dx in SHIFT_M for dy in SHIFT_M]
    S = np.array([score_yaws(sm, x + dx, y + dy, z_off, coarse, views, others, grads) for dx, dy in shifts])
    if not S.any():
        return None
    s = S.max(0)                                        # per yaw, its best xy jitter
    i = int(np.argmax(s))
    dx, dy = shifts[int(np.argmax(S[:, i]))]
    fine = coarse[i] + np.radians(np.arange(-STEP_DEG, STEP_DEG + 1e-9, FINE_DEG))
    sf = score_yaws(sm, x + dx, y + dy, z_off, fine, views, others, grads)
    j = int(np.argmax(sf))
    best, s_best = float(fine[j]), float(sf[j])
    s_cur = float(max(score_yaws(sm, x + ddx, y + ddy, z_off, [yaw], views, others, grads)[0] for ddx, ddy in shifts))
    d = np.degrees(np.abs((coarse - best + period / 2) % period - period / 2))
    far = s[d >= RIVAL_DEG]
    rival = float(far.max()) if far.size else 0.0
    changed = bool(s_best >= (1 + EDGE_GAIN) * s_cur and s_best >= (1 + EDGE_PEAK) * rival)
    out_yaw = float((best - yaw + period / 2) % period - period / 2 + yaw) if changed else float(yaw)
    return {"yaw": out_yaw, "from": float(yaw), "score": round(s_best, 2), "score_cur": round(s_cur, 2),
            "rival": round(rival, 2), "changed": changed,
            "delta_deg": round(float(np.degrees(out_yaw - yaw)), 1)}
