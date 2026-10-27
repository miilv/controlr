"""Watch reads: what the scene's own actors changed in a region (watch read (b), Skills-6 / integration-2 contract).

    watch_read(frame, region, vocabulary=None, bgs=None, cfg=None, prior=None) -> dict
        frame  = {"images": {cam: RGB}, "state": {...}, "instruction": str}   (official observation, any arm pose)
        region = {"kind": "cells", "board": {"asset", "pos", "quat"}, "pieces": {"own": asset, "opponent": asset}}
               | {"kind": "container", "container": {"asset", "pos", "quat"}, "candidates": [asset keys]}
        -> {"t_ms", "visible_frac", "occluded", "stable_vs_prior", "new": [...],
            cells:     "cells": [{"k", "occupant": "own" | "opponent" | None, "conf", "pos", "scores"[, "occluded"]}]
            container: "inside": [{"asset", "label", "pos", "conf"}], "outside": [...]}
    ArrivalOrder(n_consecutive=4, expected=None).update(read) -> {"order", "pending", "done"}   (caller side, pure python)

Cells: the board asset's public ``passive.support`` frames ``cell/0..8`` (centre in the board frame, radius 32 mm).
A cell holds a piece when the gold pixels inside its disc (coloured: the glossy black board and the white opponent arm
are neutral; the board's gold grid lines masked out) overlap a piece silhouette well enough; the kind is the piece model
whose silhouette fits them best over a small offset grid (ring vs cross: same colour, other shape), with a yaw sweep for
4-fold pieces. A cell an arm covers is not read: the prior read's occupant stands, flagged ``occluded``. Container: inverse layout over the task vocabulary (``estimate``), then each candidate
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


PIECE_CHROMA = 3.0         # a piece pixel is gold: coloured, where the glossy black board is neutral (chroma 0). Not
                           # lightness: a cross in shade is L 14-58 (chroma 4-25), and the opponent's white arm passing
                           # over the board is light but neutral (a lightness test read it as rings in 3 cells)
LINE_PX = 2                # ... and off the board's gold grid lines (half width in px around the projected lines)
OCCUPIED_IOU = 0.35        # a cell holds a piece when a piece silhouette overlaps its piece pixels this well
SEARCH_M = (-0.008, -0.004, 0.0, 0.004, 0.008)   # piece offsets from the cell centre tried (a placed piece sits 0-5 mm off)
OCCLUDER_L = 60.0          # light neutral pixels on the board are an arm (the opponent's), not a piece ...
OCCLUDED_CELL = 0.2        # ... and a cell whose disc is this much covered by them is not read (the prior's read kept)
KIND_MARGIN = 0.06         # ring vs cross closer than this: the kind is not read from this frame (prior's kind kept)


def _occluder_mask(img: np.ndarray) -> np.ndarray:
    """Light neutral pixels: neither the black board nor gold (a piece, the grid lines). On the board that is the
    opponent's white arm passing over a cell (not in our robot mask): such a cell is not read."""
    import cv2

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
    return ((lab[..., 0] > OCCLUDER_L) & (chroma <= PIECE_CHROMA)).astype(np.uint8)


def _piece_mask(img: np.ndarray, cam, cells: list[dict], robot: np.ndarray) -> np.ndarray:
    """Pixels that are neither the black board nor its gold grid lines (nor the robot): the pieces. An absolute
    gold threshold (L > 90) lost a piece in shade (5.1 ttt 100005: the opponent's cross in cell 2, L median 21 but
    chroma ~10 on a neutral board, scored 0.05) and counted the grid lines inside each cell's bounding box against
    every piece."""
    import cv2

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
    m = (chroma > PIECE_CHROMA).astype(np.uint8)
    m[robot > 0] = 0
    lines = np.zeros_like(m)
    by_k = {c["k"]: c["pos"] for c in cells}
    if len(by_k) == 9:
        P = np.array([by_k[k] for k in range(9)]).reshape(3, 3, 3)
        segs = []
        for r in range(3):          # boundaries between columns / rows, as long as the grid
            for c in range(2):
                a, b = P[r, c], P[r, c + 1]
                mid, d = 0.5 * (a + b), b - a
                n = np.array([-d[1], d[0], 0.0])
                segs.append((mid + 0.5 * n, mid - 0.5 * n))
                a2, b2 = P[c, r], P[c + 1, r]
                mid, d = 0.5 * (a2 + b2), b2 - a2
                n = np.array([-d[1], d[0], 0.0])
                segs.append((mid + 0.5 * n, mid - 0.5 * n))
        for a, b in segs:
            uv, z = cam.project(np.array([a, b]))
            if (z > 0).all():
                cv2.line(lines, tuple(int(round(v)) for v in uv[0]), tuple(int(round(v)) for v in uv[1]), 1, 2 * LINE_PX + 1)
    m[lines > 0] = 0
    return m


def _read_cells(images, cams, rmasks, region: dict, prior: dict | None = None) -> list[dict]:
    """Per cell: the piece kind whose silhouette (best of a small offset grid and, for crosses, a yaw sweep) best overlaps
    the cell's piece pixels inside its disc, when that overlap reaches ``OCCUPIED_IOU``. When the two kinds score
    within ``KIND_MARGIN`` the prior read's kind for that cell is kept (a settled piece does not change kind: 5.1 ttt
    100000 cell 8 alternated x / o on every other read)."""
    import cv2

    from .render import load_asset, silhouette, transform

    cells = board_cells(region["board"])
    pieces = region.get("pieces") or {}
    prev = {c["k"]: c.get("occupant") for c in ((prior or {}).get("cells") or [])}
    masks = {c: _piece_mask(images[c], cam, cells, rmasks[c]) for c, cam in cams.items() if c in images}
    occl = {c: _occluder_mask(images[c]) for c in masks}
    out = []
    for cell in cells:
        scores, best_by = {}, {}
        disc = {}
        for c, cam in cams.items():
            if c not in masks:
                continue
            th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            ring = np.c_[cell["pos"][0] + cell["radius"] * np.cos(th), cell["pos"][1] + cell["radius"] * np.sin(th),
                         np.full(48, cell["pos"][2] + 0.005)]
            uv, z = cam.project(ring)
            if (z <= 0).any():
                continue
            dm = np.zeros(masks[c].shape, np.uint8)
            cv2.fillPoly(dm, [np.round(uv - 0.5).astype(np.int32)], 1)
            dm &= (rmasks[c] == 0).astype(np.uint8)
            if dm.sum() >= 30 and float((occl[c] & dm).sum()) < OCCLUDED_CELL * float(dm.sum()):
                disc[c] = dm
        for kind, asset in pieces.items():
            a = load_asset(asset)
            best = None
            for sm in _piece_shapes(asset):
                yaws = [0.0] if sm.sym >= 36 else list(np.linspace(0, sm.yaw_period, 6, endpoint=False))
                for yaw in yaws:
                    for dx in SEARCH_M:
                        for dy in SEARCH_M:
                            T = sm.pose(0, 0, yaw)
                            T[:3, 3] = [cell["pos"][0] + dx, cell["pos"][1] + dy, cell["pos"][2] - sm.zmin]
                            num = den = 0.0
                            for c, dm in disc.items():
                                S = (silhouette(cams[c], transform(a.Vs, T), a.Fs) > 0) & (dm > 0)
                                P = (masks[c] > 0) & (dm > 0)
                                num += float((S & P).sum())
                                den += float((S | P).sum())
                            iou = num / den if den else 0.0
                            if best is None or iou > best[0]:
                                best = (iou, yaw, T)
            if best is not None:
                scores[kind] = round(best[0], 3)
                best_by[kind] = best
        occupant, conf, pos = None, 0.0, None
        if not disc:
            # no view sees this cell clear (an arm over it): the prior read's occupant stands, flagged
            pc = next((c for c in ((prior or {}).get("cells") or []) if c.get("k") == cell["k"]), None) or {}
            out.append({"k": cell["k"], "occupant": pc.get("occupant"), "conf": 0.0, "pos": pc.get("pos"),
                        "scores": {}, "occluded": True, "cell_pos": [round(float(v), 4) for v in cell["pos"]]})
            continue
        if best_by:
            ranked = sorted(best_by.items(), key=lambda kv: -kv[1][0])
            top_kind, top = ranked[0]
            margin = top[0] - (ranked[1][1][0] if len(ranked) > 1 else 0.0)
            if top[0] >= OCCUPIED_IOU:
                if margin < KIND_MARGIN and prev.get(cell["k"]) in best_by:
                    top_kind, top = prev[cell["k"]], best_by[prev[cell["k"]]]
                occupant, pos = top_kind, [round(float(v), 4) for v in top[2][:3, 3]]
                conf = round(float(min(1.0, top[0] + margin)), 3)
            else:
                conf = round(float(1.0 - top[0]), 3)
        out.append({"k": cell["k"], "occupant": occupant, "conf": conf, "pos": pos, "scores": scores,
                    "cell_pos": [round(float(v), 4) for v in cell["pos"]]})
    return out


GONE_FG = 0.35             # a source spot whose own pixels are under this share foreground: the object left it
BASE_MIN_FG = 0.4          # ... if it read at least this at the first read (else its start pose is off: unreadable;
                           # 0.5 left a watch ring, table showing through its silhouette, unread: imitate 100017)
SOURCE_MIN_PX = 30         # fewer visible pixels (our robot in front): the spot cannot be read


def read_sources(frame: dict, cams: dict, sources: list[dict], bgs, cfg) -> list[dict]:
    """Per source object (its episode-start pose: {"asset", "label", "pos", "quat"}), is it still where it was? The
    share of its own head-view silhouette (pixels where it is the front-most source, our robot and unknown pixels
    excluded) that is still foreground. An object a scripted actor took away leaves background behind (an arm in front
    of it reads as foreground: "still there", never a false departure). No inverse layout: a moved object is outside
    every generator region, so the estimate never proposes it in its new place."""
    from .cameras import pose_to_T
    from .inverse_layout import shape_models
    from .render import gl
    from .segment import segment_views

    bgs = dict(bgs or {})
    if "cam_head_noactor" in bgs:
        # a scene background with a scripted actor at home (the demonstrator's arm): read against the scene without
        # it. Against the actor, a spot the arm left read foreground for good and its object never went (imitate
        # 100020: read last instead of third); now the arm reads foreground (present) wherever it is, as elsewhere
        bgs["cam_head"] = bgs["cam_head_noactor"]
    v = segment_views({"cam_head": frame["images"]["cam_head"]}, {"cam_head": cams["cam_head"]},
                      {k: np.asarray(x) for k, x in frame["state"].items()}, bgs, thr=getattr(cfg, "fg_thr", 18.0))["cam_head"]
    g = gl()
    Ts, sms = [], []
    for o in sources:
        try:
            sms.append(shape_models(o["asset"])[0])
        except Exception:  # noqa: BLE001 (not in the catalogue)
            sms.append(None)
        Ts.append(pose_to_T(np.asarray(o["pos"], float), np.asarray(o.get("quat") or [1, 0, 0, 0], float)))
    if g is not None:
        depth = [np.full(v.fg.shape, np.inf) if sm is None else
                 np.where((d := g.render(v.cam, [(sm.key, sm.asset.Vg, sm.asset.Fg, T)], depth=True)) > 0, d, np.inf)
                 for sm, T in zip(sms, Ts)]
        front = np.argmin(np.stack(depth), axis=0) if depth else None
        has = np.isfinite(np.stack(depth)).any(axis=0) if depth else None
    # our robot's pixels are unreadable; a shadow on an empty spot (the demonstrator's arm above it) is not the object:
    # ``fg`` already excludes shadow-like pixels, so they count as background here
    ok_px = v.robot == 0
    out = []
    for i, (o, sm, T) in enumerate(zip(sources, sms, Ts)):
        if sm is None:
            out.append({"asset": o["asset"], "label": o.get("label"), "fg": None, "n_px": 0})
            continue
        own = (has & (front == i)) if g is not None else sm.sil(v.cam, T).astype(bool)
        m = own & ok_px
        n = int(m.sum())
        fgf = float((v.fg.astype(bool) & m).sum() / n) if n else None
        out.append({"asset": o["asset"], "label": o.get("label"), "fg": None if fgf is None else round(fgf, 3), "n_px": n})
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
    """Container regions with ``sources`` (every mover's episode-start pose) read departures (``read_sources``): the
    full-scene estimate per read took 23.7 s and never saw an object inside the demonstrator's basket (a moved object is
    outside every generator region; percrnd watch sequences 100002: 0 of 5 reads with objects inside)."""
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
        out["cells"] = _read_cells(frame["images"], cams, rmasks, region, prior)
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
        if region.get("sources"):
            # what left its place: the demonstrator moves one object at a time into its container, so the order of
            # departures is the order of arrivals; a departure is final (the arm in front of an empty spot reads as
            # "still there", so a departed object stays departed)
            src = read_sources(frame, cams, region["sources"], bgs, cfg)
            gone = {e["label"] for e in (prior or {}).get("gone") or []}
            # each spot's own baseline: the first read where it was readable, or the source's own episode-start fit
            # confidence (``conf``: the demonstrator may have taken it before the first read). A source whose start
            # pose is off shows little foreground from the start; it is not "gone", it is unreadable (< BASE_MIN_FG)
            base = dict((prior or {}).get("src_base") or {})
            conf0 = {o.get("label"): float(o.get("conf", 0.0)) for o in region["sources"]}
            for e in src:
                if e["label"] not in base and e["fg"] is not None and e["n_px"] >= SOURCE_MIN_PX:
                    base[e["label"]] = max(e["fg"], conf0.get(e["label"], 0.0))
            out["src_base"] = base
            out["sources"] = src
            out["gone"] = [{"asset": e["asset"], "label": e["label"]} for e in src
                           if e["label"] in gone or (e["fg"] is not None and e["n_px"] >= SOURCE_MIN_PX
                                                     and base.get(e["label"], 0.0) >= BASE_MIN_FG
                                                     and e["fg"] < min(GONE_FG, 0.5 * base[e["label"]]))]
            out["inside"] = [{"asset": e["asset"], "label": e["label"], "pos": None, "conf": 1.0} for e in out["gone"]]
            out["outside"] = []
        else:
            out["inside"], out["outside"] = _read_container(frame, cams, region, vocabulary, bgs, cfg)
        if prior and prior.get("inside") is not None:
            prev = {e["asset"]: e for e in prior["inside"]}
            cur = {e["asset"]: e for e in out["inside"]}
            out["new"] = [e for k, e in cur.items() if k not in prev]
            out["stable_vs_prior"] = set(cur) == set(prev) and all(
                cur[k]["pos"] is None or prev[k]["pos"] is None or math.dist(cur[k]["pos"], prev[k]["pos"]) <= STABLE_M
                for k in cur)
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
