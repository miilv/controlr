"""Watch reads: what the scene's own actors changed in a region (watch read (b), Skills-6 / integration-2 contract).

    watch_read(frame, region, vocabulary=None, bgs=None, cfg=None, prior=None) -> dict
        frame  = {"images": {cam: RGB}, "state": {...}, "instruction": str}   (official observation, any arm pose)
        region = {"kind": "cells", "board": {"asset", "pos", "quat"}, "pieces": {"own": asset, "opponent": asset}}
               | {"kind": "container", "container": {"asset", "pos", "quat"}, "candidates": [asset keys]}
        -> {"t_ms", "visible_frac", "occluded", "stable_vs_prior", "new": [...],
            cells:     "cells": [{"k", "occupant": "own" | "opponent" | None, "conf", "pos", "scores"}]
            container: "inside": [{"asset", "label", "pos", "conf"}], "outside": [...]}
    ArrivalOrder(n_consecutive=4, expected=None).update(read) -> {"order", "pending", "done"}   (caller side, pure python)

Cells: the board asset's public ``passive.support`` frames ``cell/0..8`` (centre in the board frame, radius 32 mm).
A cell holds a piece when the bright (gold) pixels inside its disc match a piece silhouette better than the empty
board does; the kind is the piece model whose silhouette fits them best (ring vs cross: same colour, other shape), with
a yaw sweep for 4-fold pieces. Container: inverse layout over the task vocabulary (``estimate``), then each candidate
object's root inside the container's footprint polygon and above its bottom. GT is never read here.
"""
from __future__ import annotations

import math
import time

import numpy as np

CELL_IN_M = 0.025          # a piece within this of a cell centre occupies it (Skills-6 spec)
STABLE_M = 0.0005          # contents within 0.5 mm of the prior read = stable
OCCLUDED_BELOW = 0.5       # visible_frac under this: the read must not count toward arrival / stability


def _T(p: dict) -> np.ndarray:
    from .cameras import pose_to_T

    return pose_to_T(p["pos"], p.get("quat") or [1, 0, 0, 0])


def board_cells(board: dict) -> list[dict]:
    """World centres of the board's cells (catalogue ``support`` frames, from the asset's public metadata)."""
    from .render import load_asset

    sup = load_asset(board["asset"]).info.get("support") or {}
    T = _T(board)
    out = []
    for name in sorted(sup, key=lambda n: int(n.split("/")[-1]) if n.split("/")[-1].isdigit() else 99):
        if not name.startswith("cell/"):
            continue
        c = sup[name]["center"][0]
        p = T @ np.array([c[0], c[1], c[2], 1.0])
        out.append({"k": int(name.split("/")[-1]), "pos": p[:3], "radius": float(sup[name]["radius"][0])})
    return out


def container_footprint(container: dict) -> tuple[np.ndarray, float, float]:
    """(convex footprint polygon xy (N, 2), bottom z, top z) of the container mesh at its pose."""
    import cv2

    from .render import load_asset

    a = load_asset(container["asset"])
    T = _T(container)
    W = a.V @ T[:3, :3].T + T[:3, 3]
    hull = cv2.convexHull(np.round(W[:, :2] * 1e4).astype(np.int32)).reshape(-1, 2) / 1e4
    return hull, float(W[:, 2].min()), float(W[:, 2].max())


def inside_polygon(xy, poly: np.ndarray) -> bool:
    import cv2

    return cv2.pointPolygonTest(np.round(poly * 1e4).astype(np.int32).reshape(-1, 1, 2),
                                (float(xy[0]) * 1e4, float(xy[1]) * 1e4), False) >= 0


def _robot_masks(cams: dict, state: dict) -> dict:
    from .render import robot_mask

    return {c: robot_mask(cam, state, dilate_px=2) for c, cam in cams.items()}


def _visible_frac(cams: dict, rmasks: dict, pts: np.ndarray) -> float:
    """Share of the region's sample points that the head camera sees unoccluded by any robot."""
    cam = cams["cam_head"]
    uv, z = cam.project(pts)
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.w) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.h)
    if not ok.any():
        return 0.0
    m = rmasks["cam_head"]
    vis = m[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] == 0
    return float(vis.sum()) / len(pts)


def _bright(img: np.ndarray) -> np.ndarray:
    """Gold piece pixels vs the glossy black board: Lab lightness well above the board's."""
    import cv2

    L = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    return L > 90.0


def _piece_shapes(asset: str):
    from .inverse_layout import shape_models

    return shape_models(asset)


def _read_cells(images, cams, rmasks, region: dict) -> list[dict]:
    from .render import silhouette, load_asset, transform

    cells = board_cells(region["board"])
    pieces = region.get("pieces") or {}
    out = []
    bright = {c: _bright(images[c]) for c in images if c in cams}
    for cell in cells:
        scores = {}
        best_piece = None
        for kind, asset in pieces.items():
            a = load_asset(asset)
            best = None
            for sm in _piece_shapes(asset):
                yaws = [0.0] if sm.sym >= 36 else list(np.linspace(0, sm.yaw_period, 6, endpoint=False))
                for yaw in yaws:
                    T = sm.pose(0, 0, yaw)
                    T[:3, 3] = [cell["pos"][0], cell["pos"][1], cell["pos"][2] - sm.zmin]
                    num = den = 0.0
                    for c, cam in cams.items():
                        if c not in bright:
                            continue
                        S = silhouette(cam, transform(a.Vs, T), a.Fs) > 0
                        S &= rmasks[c] == 0
                        if S.sum() < 15:
                            continue
                        ys, xs = np.nonzero(S)
                        pad = 6
                        y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + pad + 1, max(xs.min() - pad, 0), xs.max() + pad + 1
                        Sb, Bb = S[y0:y1, x0:x1], bright[c][y0:y1, x0:x1] & (rmasks[c][y0:y1, x0:x1] == 0)
                        inter = float((Sb & Bb).sum())
                        union = float((Sb | Bb).sum())
                        num += inter
                        den += union
                    iou = num / den if den else 0.0
                    if best is None or iou > best[0]:
                        best = (iou, yaw, T)
            if best is not None:
                scores[kind] = round(best[0], 3)
                if best_piece is None or best[0] > best_piece[1]:
                    best_piece = (kind, best[0], best[2])
        occupant, conf, pos = None, 0.0, None
        if best_piece is not None:
            ranked = sorted(scores.values(), reverse=True)
            margin = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
            if best_piece[1] >= 0.35:
                occupant, pos = best_piece[0], [round(float(v), 4) for v in best_piece[2][:3, 3]]
                conf = round(float(min(1.0, best_piece[1] + margin)), 3)
            else:
                conf = round(float(1.0 - best_piece[1]), 3)
        out.append({"k": cell["k"], "occupant": occupant, "conf": conf, "pos": pos, "scores": scores,
                    "cell_pos": [round(float(v), 4) for v in cell["pos"]]})
    return out


def _read_container(frame, cams, region: dict, vocabulary: str, bgs, cfg) -> tuple[list, list]:
    from .inverse_layout import estimate

    rec = estimate(frame["images"], frame["state"], vocabulary, frame.get("instruction", ""), bgs=bgs, cfg=cfg, cams=cams)
    poly, zb, zt = container_footprint(region["container"])
    cand = set(region.get("candidates") or [])
    inside, outside = [], []
    for typ in ("Rigid", "Geometry", "Articulation"):
        for cat, lst in (rec.get(typ) or {}).items():
            for r in lst:
                key = f"{typ}/{cat}/{int(r['category_idx']):05d}"
                if cand and key not in cand:
                    continue
                p = r["default_pos"]
                e = {"asset": key, "label": r["label"], "pos": [round(float(v), 4) for v in p],
                     "conf": round(float((r.get("_perception") or {}).get("confidence", 0.5)), 3)}
                (inside if inside_polygon(p[:2], poly) and p[2] > zb else outside).append(e)
    return inside, outside


def watch_read(frame: dict, region: dict, vocabulary: str | None = None, bgs=None, cfg=None, prior: dict | None = None) -> dict:
    from .cameras import cameras_from_state

    t0 = time.time()
    state = {k: np.asarray(v, float) for k, v in frame["state"].items()}
    cams = {c: cam for c, cam in cameras_from_state(state).items() if c in frame["images"]}
    rmasks = _robot_masks(cams, state)
    out: dict = {"kind": region["kind"]}
    if region["kind"] == "cells":
        cells = board_cells(region["board"])
        pts = np.array([c["pos"] + np.array([dx, dy, 0.0]) for c in cells for dx in (-0.015, 0, 0.015) for dy in (-0.015, 0, 0.015)])
        out["visible_frac"] = round(_visible_frac(cams, rmasks, pts), 3)
        out["cells"] = _read_cells(frame["images"], cams, rmasks, region)
        cur = {c["k"]: c for c in out["cells"] if c["occupant"]}
        if prior and prior.get("cells") is not None:
            prev = {c["k"]: c for c in prior["cells"] if c.get("occupant")}
            out["new"] = [c for k, c in cur.items() if k not in prev]
            out["stable_vs_prior"] = set(cur) == set(prev) and all(
                cur[k]["occupant"] == prev[k]["occupant"] and math.dist(cur[k]["pos"][:2], prev[k]["pos"][:2]) <= STABLE_M
                for k in cur)
    else:
        poly, zb, zt = container_footprint(region["container"])
        xs = np.linspace(poly[:, 0].min(), poly[:, 0].max(), 8)
        ys = np.linspace(poly[:, 1].min(), poly[:, 1].max(), 8)
        pts = np.array([[x, y, zt] for x in xs for y in ys if inside_polygon((x, y), poly)])
        out["visible_frac"] = round(_visible_frac(cams, rmasks, pts), 3) if len(pts) else 0.0
        out["inside"], out["outside"] = _read_container(frame, cams, region, vocabulary, bgs, cfg)
        if prior and prior.get("inside") is not None:
            prev = {e["asset"]: e for e in prior["inside"]}
            cur = {e["asset"]: e for e in out["inside"]}
            out["new"] = [e for k, e in cur.items() if k not in prev]
            out["stable_vs_prior"] = set(cur) == set(prev) and all(
                math.dist(cur[k]["pos"], prev[k]["pos"]) <= STABLE_M for k in cur)
    out.setdefault("new", [])
    out.setdefault("stable_vs_prior", False)
    out["occluded"] = out["visible_frac"] < OCCLUDED_BELOW
    out["t_ms"] = round(1000 * (time.time() - t0), 1)
    return out


class ArrivalOrder:
    """Arrival order from repeated container reads: an object has arrived after ``n_consecutive`` non-occluded reads
    with it inside; ``done`` once ``expected`` objects arrived and the contents stayed the same for ``n_consecutive``
    non-occluded reads."""

    def __init__(self, n_consecutive: int = 4, expected: int | None = None):
        self.n, self.expected = n_consecutive, expected
        self.streak: dict[str, int] = {}
        self.order: list[str] = []
        self.same = 0
        self._last: set | None = None

    def update(self, read: dict) -> dict:
        if not read.get("occluded"):
            now = {e["asset"] for e in read.get("inside") or []}
            for k in list(self.streak):
                if k not in now:
                    self.streak[k] = 0
            for k in now:
                self.streak[k] = self.streak.get(k, 0) + 1
                if self.streak[k] >= self.n and k not in self.order:
                    self.order.append(k)
            self.same = self.same + 1 if now == self._last else 1
            self._last = now
        pending = [k for k, s in self.streak.items() if 0 < s < self.n and k not in self.order]
        done = self.expected is not None and len(self.order) >= self.expected and self.same >= self.n
        return {"order": list(self.order), "pending": pending, "done": bool(done)}
