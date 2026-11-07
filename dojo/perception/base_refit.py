"""Base-footprint xy refit for any object resting on a base (stream ORG2; default off, ``EstimatorConfig.base_refit``).

A pose fit to an object's whole silhouette can sit off its functional origin when the silhouette is dominated by parts
above the base (outstretched appendages, a part hidden by a neighbour, a shadowed side). What rests on the table is
the object's lowest mesh band: its base. This refit slides that band in xy under the current pose (orientation and
height kept) and scores, per view, the foreground inside the projected band against a ring just outside it, on the
camera-facing side only (the far edge is behind the object itself). Views are fused by their numbers of usable points;
robot pixels are unknown (a gripper next to the object is never read as the object's base). A shift is proposed only
for a unique peak inside the search window that beats the current pose by ``MARGIN``.

Geometry comes from the asset catalogue (any key), images and masks from ``segment.segment_views``; no task, label or
layout knowledge.
"""
from __future__ import annotations

import math

import numpy as np

BAND = 0.012          # m: the lowest band of the mesh taken as the base
STEP = 0.003          # m: surface sampling of that band
RING = 0.008          # m: the outside ring's distance from the band
SEARCH = 0.03         # m: half-size of the xy search window
GRID = 0.002          # m: its step
MARGIN = 0.08         # score gain over the current pose needed to propose a shift
UNIQUE = 0.006        # m: a competing peak farther than this must score MARGIN / 2 below the best
MIN_POINTS = 12       # usable band points (over all views) for a reading


def base_band(V: np.ndarray, F: np.ndarray, band: float = BAND, step: float = STEP) -> np.ndarray:
    """Object-frame surface samples within ``band`` of the mesh's lowest point (z up in the object frame)."""
    from dojo.skills.insertion import _surface_samples
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    z0 = float(V[:, 2].min())
    F = F[V[F, 2].min(axis=1) < z0 + band]              # only faces that reach into the band
    P = _surface_samples(V, F, step=step)
    P = P[P[:, 2] < z0 + band]
    # one point per ``step`` voxel (a dense mesh puts a sample on every small face: 3e5 points for a 14 cm base)
    _, keep = np.unique(np.floor((P - P.min(0)) / step).astype(np.int64), axis=0, return_index=True)
    return P[np.sort(keep)]


def _ring(P: np.ndarray, ring: float) -> np.ndarray:
    """``P`` (world, the band) pushed out horizontally from the band's xy centre by ``ring``."""
    c = P[:, :2].mean(0)
    d = P[:, :2] - c
    n = np.linalg.norm(d, axis=1, keepdims=True)
    out = P.copy()
    out[:, :2] = P[:, :2] + d / np.maximum(n, 1e-9) * ring
    return out


def _lookup(mask: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    val = np.zeros(len(uv), float)
    val[ok] = mask[v[ok], u[ok]]
    return val, ok


def score_at(views: dict, P_in: np.ndarray, P_out: np.ndarray) -> tuple[float, int]:
    """Fused score (mean fg inside minus mean fg on the camera-facing outer ring) and the usable point count."""
    num, n_tot = 0.0, 0
    c = P_in[:, :2].mean(0)
    for vm in views.values():
        cam = vm.cam
        cxy = cam.center[:2]
        toward = (cxy - c) / max(float(np.linalg.norm(cxy - c)), 1e-9)
        front = ((P_out[:, :2] - c) @ toward) > 0
        uv_i, z_i = cam.project(P_in)
        uv_o, z_o = cam.project(P_out[front])
        fi, oki = _lookup(vm.fg, uv_i)
        ki, _ = _lookup(vm.known, uv_i)
        fo, oko = _lookup(vm.fg, uv_o)
        ko, _ = _lookup(vm.known, uv_o)
        mi = oki & (ki > 0) & (z_i > 0)
        mo = oko & (ko > 0) & (z_o > 0)
        if mi.sum() < 3 or mo.sum() < 3:
            continue
        n = int(mi.sum() + mo.sum())
        num += n * (float(fi[mi].mean()) - float(fo[mo].mean()))
        n_tot += n
    return (num / n_tot if n_tot else 0.0), n_tot


def refit_xy(views: dict, T_obj: np.ndarray, band_pts: np.ndarray, search: float = SEARCH, grid: float = GRID,
             ring: float = RING) -> dict:
    """The base refit (module doc) of an object at world pose ``T_obj`` (4x4) with object-frame band points.
    Returns {"ok", "dxy" (world m), "score0", "score", "points", "why"}; ``ok`` False = keep the pose."""
    T_obj = np.asarray(T_obj, float)
    P0 = band_pts @ T_obj[:3, :3].T + T_obj[:3, 3]
    s0, n0 = score_at(views, P0, _ring(P0, ring))
    if n0 < MIN_POINTS:
        return {"ok": False, "dxy": [0.0, 0.0], "score0": s0, "score": s0, "points": n0, "why": "too few points"}
    offs = np.arange(-search, search + 1e-9, grid)
    best, best_s, scores = None, -9.0, []
    for dx in offs:
        for dy in offs:
            P = P0 + np.array([dx, dy, 0.0])
            s, n = score_at(views, P, _ring(P, ring))
            if n < MIN_POINTS:
                continue
            scores.append((s, dx, dy))
            if s > best_s:
                best, best_s = (dx, dy), s
    out = {"ok": False, "dxy": [0.0, 0.0], "score0": round(s0, 4), "score": round(best_s, 4), "points": n0}
    if best is None:
        out["why"] = "no usable offsets"
        return out
    if max(abs(best[0]), abs(best[1])) >= search - 1e-9:
        out["why"] = "peak on the window edge"
        return out
    if best_s - s0 < MARGIN:
        out["why"] = "gain below margin"
        return out
    rival = max((s for s, dx, dy in scores if math.hypot(dx - best[0], dy - best[1]) > UNIQUE), default=-9.0)
    if best_s - rival < MARGIN / 2:
        out["why"] = "ambiguous peak"
        return out
    out.update(ok=True, dxy=[round(float(best[0]), 4), round(float(best[1]), 4)], why="")
    return out
