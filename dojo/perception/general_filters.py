"""Task-free RGB multiview proposal checks; geometry is estimated, never GT.

No measured depth is present in the recorded panel. Bounds inferred from matched
silhouettes are a support/size proxy, not a physical-contact measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping
import time

import numpy as np

from .cameras import Camera
from .general_masks import catalogue_labels, mask_iou, normalize


FILTERS = ('multiview', 'support', 'size', 'agreement', 'score')


@dataclass(frozen=True)
class FilterConfig:
    ray_gap_m: float = .04
    reprojection_coverage: float = .25
    min_baseline_m: float = .05
    min_ray_sine: float = .05
    support_tolerance_m: float = .025
    table_z_m: float = .765
    support_xy_overlap: float = .1
    size_min_factor: float = .5
    size_max_factor: float = 2.
    agreement_iou: float = .1
    minimum_score: float = .5
    mask_samples: int = 128


@dataclass
class Geometry:
    low: np.ndarray
    high: np.ndarray
    partner: tuple[str, int]
    coverage: float
    ray_gap_m: float


def pixels(mask, n=128):
    y, x = np.nonzero(mask)
    if not len(x):
        return np.empty((0, 2))
    take = np.linspace(0, len(x)-1, min(n, len(x)), dtype=int)
    return np.c_[x[take], y[take]]


def center(mask):
    y, x = np.nonzero(mask)
    return np.array([x.mean(), y.mean()])


def at_depth(camera, uv, point):
    rays = camera.rays(uv)
    forward = camera.T_wc[:3, 2]
    depth = float((point-camera.center) @ forward)
    denom = rays @ forward
    return camera.center + rays * (depth/denom)[:, None]


def coverage(points, camera, mask):
    uv, depth = camera.project(points)
    finite = np.isfinite(uv).all(1) & (depth > 0)
    xy = np.rint(np.where(finite[:, None], uv, -1)).astype(int)
    valid = finite & (xy[:, 0] >= 0) & (xy[:, 0] < camera.w) & (xy[:, 1] >= 0) & (xy[:, 1] < camera.h)
    hits = np.zeros(len(points), bool)
    hits[valid] = np.asarray(mask, bool)[xy[valid, 1], xy[valid, 0]]
    return float(hits.mean()) if len(hits) else 0.


def pair_geometry(a, ca, b, cb, cfg=FilterConfig()):
    if np.linalg.norm(ca.center-cb.center) < cfg.min_baseline_m:
        return None
    da, db = ca.rays(center(a.mask)[None])[0], cb.rays(center(b.mask)[None])[0]
    if np.linalg.norm(np.cross(da, db)) < cfg.min_ray_sine:
        return None
    t = np.linalg.lstsq(np.c_[da, -db], cb.center-ca.center, rcond=None)[0]
    if (t <= 0).any():
        return None
    pa, pb = ca.center+t[0]*da, cb.center+t[1]*db
    gap = float(np.linalg.norm(pa-pb))
    if gap > cfg.ray_gap_m:
        return None
    point = (pa+pb)/2
    xa = at_depth(ca, pixels(a.mask, cfg.mask_samples), point)
    xb = at_depth(cb, pixels(b.mask, cfg.mask_samples), point)
    cov = min(coverage(xa, cb, b.mask), coverage(xb, ca, a.mask))
    if cov < cfg.reprojection_coverage:
        return None
    cloud = np.r_[xa, xb]
    # These are planar backprojection bounds at centroid depth, not a depth sensor.
    return cloud.min(0), cloud.max(0), cov, gap


def geometries(proposals: Mapping, cameras: Mapping, cfg=FilterConfig()):
    out = {view: [None]*len(ds) for view, ds in proposals.items()}
    sampled = {v: [pixels(d.mask, cfg.mask_samples) for d in ds] for v, ds in proposals.items()}
    rays = {v: cameras[v].rays(np.array([center(d.mask) for d in ds]).reshape(-1, 2))
            for v, ds in proposals.items()}
    for va, vb in combinations(sorted(proposals), 2):
        ca, cb = cameras[va], cameras[vb]
        if np.linalg.norm(ca.center-cb.center) < cfg.min_baseline_m:
            continue
        da, db = rays[va][:, None], rays[vb][None, :]
        w = ca.center-cb.center
        dot = (da*db).sum(2)
        denom = 1-dot**2
        safe = np.maximum(denom, 1e-12)
        d, e = (da*w).sum(2), (db*w).sum(2)
        ta, tb = (dot*e-d)/safe, (e-dot*d)/safe
        pa, pb = ca.center+ta[:, :, None]*da, cb.center+tb[:, :, None]*db
        gaps = np.linalg.norm(pa-pb, axis=2)
        valid = (denom >= cfg.min_ray_sine**2) & (ta > 0) & (tb > 0) & (gaps <= cfg.ray_gap_m)
        # Vectorized epipolar geometry rejects impossible pairs, independent of labels.
        for ia, ib in np.argwhere(valid):
            point = (pa[ia, ib]+pb[ia, ib])/2
            xa = at_depth(ca, sampled[va][ia], point)
            xb = at_depth(cb, sampled[vb][ib], point)
            cov = min(coverage(xa, cb, proposals[vb][ib].mask), coverage(xb, ca, proposals[va][ia].mask))
            if cov < cfg.reprojection_coverage:
                continue
            cloud = np.r_[xa, xb]
            low, high, gap = cloud.min(0), cloud.max(0), float(gaps[ia, ib])
            for view, i, partner in [(va, ia, (vb, ib)), (vb, ib, (va, ia))]:
                old = out[view][i]
                if old is None or (cov, -gap) > (old.coverage, -old.ray_gap_m):
                    out[view][i] = Geometry(low, high, partner, cov, gap)
    return out


def supported(g, others, cfg=FilterConfig()):
    if g is None:
        return False
    if abs(g.low[2]-cfg.table_z_m) <= cfg.support_tolerance_m:
        return True
    for other in others:
        if other is None or other is g or other.low[2] >= g.low[2]:
            continue
        overlap = np.maximum(0, np.minimum(g.high[:2], other.high[:2])-np.maximum(g.low[:2], other.low[:2]))
        area = np.prod(np.maximum(g.high[:2]-g.low[:2], 1e-6))
        if (np.prod(overlap)/area >= cfg.support_xy_overlap and
                abs(g.low[2]-other.high[2]) <= cfg.support_tolerance_m):
            return True
    return False


def size_ranges(index):
    """Every asset contributes; only global category/name/extents fields are read."""
    labels = catalogue_labels(index)
    by_category = {}
    for key, row in index.items():
        if not isinstance(key, str) or key.count('/') != 2 or not isinstance(row, Mapping):
            continue
        v = np.asarray(row.get('extents', []), float)
        if v.shape == (3,) and np.isfinite(v).all() and (v > 0).all():
            by_category.setdefault(key.split('/')[1], []).append(np.sort(v))
    all_sizes = [v for vs in by_category.values() for v in vs]
    if not all_sizes:
        raise ValueError('catalogue has no valid extents')
    envelope = lambda vs: (np.min(vs, axis=0), np.max(vs, axis=0))
    out = {'*': envelope(all_sizes)}
    for label, cats in labels.items():
        values = [v for cat in cats for v in by_category.get(cat, [])]
        if values:
            out[label] = envelope(values)
    return out


def plausible(g, phrase, ranges, cfg=FilterConfig()):
    if g is None:
        return False
    words = normalize(phrase).split()
    # Whole-catalogue lexical lookup only; never a task-specific candidate subset.
    matches = [label for label in ranges if label != '*' and
               any(words[i:i+len(label.split())] == label.split() for i in range(len(words)))]
    if matches:
        low = np.min([ranges[k][0] for k in matches], axis=0)
        high = np.max([ranges[k][1] for k in matches], axis=0)
    else:
        low, high = ranges['*']  # Unknown/generic label uses the entire catalogue.
    extent = np.sort(g.high-g.low)
    return bool((extent >= cfg.size_min_factor*low).all() and (extent <= cfg.size_max_factor*high).all())


def evidence(proposals, cameras, base, index, cfg=FilterConfig()):
    """Independent flags; no filter silently depends on another filter passing."""
    t = time.monotonic()
    geom = geometries(proposals, cameras, cfg)
    timings = {'geometry_s': time.monotonic()-t}
    t = time.monotonic()
    ranges = size_ranges(index)
    timings['ranges_s'] = time.monotonic()-t
    flags = {}
    for view, ds in proposals.items():
        timings[view] = {k: 0. for k in FILTERS}
        flags[view] = []
        for i, d in enumerate(ds):
            g = geom[view][i]
            row = {'multiview': g is not None}
            for name, check in [
                ('support', lambda: supported(g, geom[view], cfg)),
                ('size', lambda: plausible(g, getattr(d, 'prompt', ''), ranges, cfg)),
                ('agreement', lambda: any(mask_iou(d.mask, b.mask) >= cfg.agreement_iou for b in base[view])),
                ('score', lambda: d.score >= cfg.minimum_score),
            ]:
                t = time.monotonic()
                row[name] = check()
                timings[view][name] += time.monotonic()-t
            flags[view].append(row)
    return flags, geom, timings


def enabled(bits):
    return tuple(f for i, f in enumerate(FILTERS) if bits & (1 << i))


def select(ds, flags, bits):
    checks = enabled(bits)
    return [d for d, f in zip(ds, flags) if all(f[k] for k in checks)]
