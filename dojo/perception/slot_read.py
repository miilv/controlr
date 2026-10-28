"""The pose of an object pinned in a support slot, read from one wrist frame at its pick hover (stream C).

deposit_coin's coin stands edge-on in a small stand. Perception places it from the stand (``support_slot_pose``),
so a stand yaw read wrong at frame 0 turns the coin with it. On the 5.1 camera rows (c4b_coin_off 100002) the error
was 48 deg, and the edge grasp closed across the disc and lifted nothing. The coin's own wrist refine cannot fix it:
the disc is 2 mm thick, IoU 0.25-0.36 even from the GT pose. Nor can a refine of the stand alone: its top is a
table-coloured insert in a thin white rim, and a free yaw fit lands 50 deg off at IoU 0.8.

This fits the base and its slot child together. Their union silhouette is fitted to the frame's own foreground,
with the holes filled. The foreground uses the table colour model of ``refine.local_foreground``, so the insert that
looks like the table is a hole inside the rim. The search is a grid over the base's yaw period and xy within
``radius_m``, then a local polish of the best few. The margin between the best fit and the best fit 90 deg away
says whether the yaw is determined at all.

    out = slot_read(base_key, base_pose7, child_key, placement, wrist_rgb, proprio, arm)
    out["ok"], out["base_pose"], out["child_pose"], out["iou"], out["yaw_margin"]
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .cameras import R_to_quat_wxyz, quat_wxyz_to_R, wrist_camera

MIN_HULL_IOU = 0.75    # the fitted pair's silhouette vs the convex hull of the foreground next to it (offline: 0.6 let 14 wrong reads of 200 through, 0.75 one)
MIN_MARGIN = 0.04      # the fit's IoU this much above the best fit >= 45 deg away (else the yaw is not determined)
HULL_MIN_PX = 15       # foreground specks smaller than this stay out of that hull
HOLE_MAX_FRAC = 0.5    # a hole in the foreground up to this share of the expected union area is filled
CLOSE_FRAC = 0.25      # the closing kernel before the hole fill, as a share of the expected pair's size (sqrt area)
BORDER_PX = 12         # the fitted pair this close to the image border counts as cut by it


def _rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _fill_holes(fg: np.ndarray, max_area: int) -> np.ndarray:
    """Background regions enclosed by foreground (not reachable from the image border), up to ``max_area`` px."""
    inv = (fg == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = fg.copy()
    h, w = fg.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h or area > max_area:
            continue
        out[lab == i] = 1
    return out


class _Pair:
    """Base + child in its slot, posed by the base's (x, y, yaw) on its prior's height and tilt."""

    def __init__(self, base_key, base_pose7, child_key, placement):
        from .inverse_layout import support_slot_pose
        from .render import load_asset

        self.a_base, self.a_child = load_asset(base_key), load_asset(child_key)
        self.hb, self.hc = _hull(self.a_base.Vs), _hull(self.a_child.Vs)
        self.base_key, self.child_key, self.placement = base_key, child_key, placement
        self.slot_pose = support_slot_pose
        R0 = quat_wxyz_to_R(base_pose7[3:])
        self.yaw0 = math.atan2(R0[1, 0], R0[0, 0])
        self.R_rest = _rotz(-self.yaw0) @ R0         # the base's rest orientation without its yaw
        self.z = float(base_pose7[2])
        # the child in the base frame (slot poses are rigid in the base frame)
        p, q = support_slot_pose(base_key, base_pose7[:3], base_pose7[3:], child_key, placement)
        Tb = np.eye(4)
        Tb[:3, :3], Tb[:3, 3] = R0, np.asarray(base_pose7[:3], float)
        Tc = np.eye(4)
        Tc[:3, :3], Tc[:3, 3] = quat_wxyz_to_R(q), np.asarray(p, float)
        self.T_bc = np.linalg.inv(Tb) @ Tc

    def poses(self, x, y, yaw):
        Tb = np.eye(4)
        Tb[:3, :3], Tb[:3, 3] = _rotz(yaw) @ self.R_rest, (x, y, self.z)
        return Tb, Tb @ self.T_bc

    def sil(self, cam, x, y, yaw, scale=1.0):
        from .render import silhouette, transform

        Tb, Tc = self.poses(x, y, yaw)
        m = silhouette(cam, transform(self.hb[0], Tb), self.hb[1], scale=scale)
        return silhouette(cam, transform(self.hc[0], Tc), self.hc[1], scale=scale, out=m)


def _hull(V: np.ndarray):
    """The convex hull of a mesh's vertices (vertices, faces): the same silhouette for a convex outline (the coin, the
    stand's block, with the slot filled like the foreground's holes), at a few hundred faces instead of 17 k."""
    from scipy.spatial import ConvexHull

    h = ConvexHull(np.asarray(V, float))
    return np.asarray(V, float), np.asarray(h.simplices, np.int64)


def _period(base_key) -> float:
    from .inverse_layout import shape_models

    try:
        return float(shape_models(base_key)[0].yaw_period)
    except Exception:  # noqa: BLE001
        return 2 * math.pi


def slot_read(base_key: str, base_pose7, child_key: str, placement: dict, wrist_frame: np.ndarray, proprio: dict,
              arm: str, radius_m: float = 0.018, yaw_step_deg: float = 10.0, xy_step_m: float = 0.006,
              empty_hand: bool = True, others=None, polish_top: int = 3, debug: dict | None = None) -> dict:
    from .render import load_asset, robot_mask, silhouette, transform
    from .refine import local_foreground

    t0 = time.time()
    cam = wrist_camera(arm, np.asarray(proprio[f"{arm}_arm_joint_state"], dtype=float))
    pair = _Pair(base_key, list(map(float, base_pose7)), child_key, placement)
    x0, y0 = float(base_pose7[0]), float(base_pose7[1])
    period = _period(base_key)
    yaws = [pair.yaw0 + d for d in np.arange(0.0, period, math.radians(yaw_step_deg))]
    # the region every candidate can occupy: its union, so the table ring is sampled outside all of them
    reach = np.zeros((cam.h, cam.w), np.uint8)
    for yw in yaws[:: max(1, len(yaws) // 6)]:
        for dx, dy in ((-radius_m, -radius_m), (radius_m, radius_m), (-radius_m, radius_m), (radius_m, -radius_m), (0, 0)):
            reach |= pair.sil(cam, x0 + dx, y0 + dy, yw)
    if reach.sum() < 30:
        return {"ok": False, "why": "out of view", "time_s": round(time.time() - t0, 3)}
    rob = robot_mask(cam, proprio, dilate_px=3, closed_jaws=not empty_hand)
    excl = None
    if others:
        excl = np.zeros_like(reach)
        for key, pose in others:
            from .cameras import pose_to_T
            ao = load_asset(key)
            excl |= silhouette(cam, transform(ao.Vs, pose_to_T(pose[:3], pose[3:])), ao.Fs)
        excl = cv2.dilate(excl, np.ones((7, 7), np.uint8))
    fg, known = local_foreground(wrist_frame, cam, reach, rob, exclude=excl)
    if excl is not None:
        known &= 1 - excl                          # another known object's pixels are neither the pair nor table
    exp_area = int(pair.sil(cam, x0, y0, pair.yaw0).sum())
    # the base's rim can be broken where it is in shadow (5.1 RTX: the far side of the stand's white rim), so the insert
    # is not enclosed: close gaps up to ~1/4 of the pair's size first, then fill what is enclosed
    k = int(max(5, min(31, CLOSE_FRAC * math.sqrt(max(exp_area, 1))))) | 1
    fgf = _fill_holes(cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)),
                      max(50, int(HOLE_MAX_FRAC * exp_area)))
    # score only where a candidate can be (plus a margin): foreground elsewhere in the ROI (shadows, the table's grain,
    # another object) says nothing about this pair and only lowers every candidate's IoU alike
    known &= cv2.dilate(reach, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    ys, xs = np.nonzero(known)
    if len(xs) == 0:
        return {"ok": False, "why": "no known pixels", "time_s": round(time.time() - t0, 3)}
    roi = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    F, K = fgf[roi].astype(bool), known[roi].astype(bool)

    def iou(x, y, yw):
        s = pair.sil(cam, x, y, yw)[roi].astype(bool)
        inter = (s & F & K).sum()
        uni = ((s | F) & K).sum()
        vis = (s & K).sum() / max(s.sum(), 1)
        return (inter / uni if uni else 0.0), vis

    grid = []
    offs = np.arange(-radius_m, radius_m + 1e-9, xy_step_m)
    for yw in yaws:
        for dx in offs:
            for dy in offs:
                if dx * dx + dy * dy > radius_m * radius_m * 1.01:
                    continue
                v, _ = iou(x0 + dx, y0 + dy, yw)
                grid.append((v, x0 + dx, y0 + dy, yw))
    grid.sort(key=lambda r: -r[0])

    def polish(x, y, yw):
        best = (iou(x, y, yw)[0], x, y, yw)
        st_xy, st_yw = xy_step_m / 2, math.radians(yaw_step_deg) / 2
        for _ in range(4):
            improved = True
            while improved:
                improved = False
                for ddx, ddy, ddw in ((st_xy, 0, 0), (-st_xy, 0, 0), (0, st_xy, 0), (0, -st_xy, 0), (0, 0, st_yw), (0, 0, -st_yw)):
                    c = (best[1] + ddx, best[2] + ddy, best[3] + ddw)
                    v = iou(*c)[0]
                    if v > best[0] + 1e-4:
                        best, improved = (v, *c), True
            st_xy, st_yw = st_xy / 2, st_yw / 2
        return best

    seen, cands = [], []
    for r in grid:
        if len(cands) >= polish_top:
            break
        if any(abs(r[1] - s[1]) < xy_step_m and abs((r[3] - s[3] + period / 2) % period - period / 2) < math.radians(15)
               for s in seen):
            continue
        seen.append(r)
        cands.append(polish(r[1], r[2], r[3]))
    best = max(cands, key=lambda r: r[0])
    v, x, y, yw = best
    # the yaw margin: the best grid fit at least 45 deg (mod period) away from the winner
    def ydist(a, b):
        d = (a - b) % period
        return min(d, period - d)
    alt = max((r[0] for r in grid if ydist(r[3], yw) >= math.radians(45)), default=0.0)
    _, vis = iou(x, y, yw)
    Tb, Tc = pair.poses(x, y, yw)
    # the pair cut by the image border: the fit slides along the cut and turns with it (5.1-proxy deposit_coin 100002, a
    # stand in the top corner of a place hover's view, read 35-40 deg off at iou 0.52)
    uv = np.vstack([cam.project(transform(pair.hb[0], Tb))[0], cam.project(transform(pair.hc[0], Tc))[0]])
    cut = bool((uv[:, 0] < BORDER_PX).any() or (uv[:, 1] < BORDER_PX).any() or (uv[:, 0] > cam.w - BORDER_PX).any()
               or (uv[:, 1] > cam.h - BORDER_PX).any())
    # acceptance by shape: the convex hull of the foreground next to the fitted pair against its silhouette. 5.1 RTX
    # shows the stand's rim only at its ends and the coin's face reflects the table, so the foreground covers about a
    # third of the pair (IoU 0.2-0.3 at the right pose); its hull is the pair's outline. Nothing there: no hull
    S = pair.sil(cam, x, y, yw)
    near = cv2.dilate(S, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) & known & fgf
    n_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats(near, connectivity=8)
    keep = np.isin(lab_c, [i for i in range(1, n_c) if st_c[i, cv2.CC_STAT_AREA] >= HULL_MIN_PX])
    H = np.zeros_like(S)
    pts = np.column_stack(np.nonzero(keep)[::-1]).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillConvexPoly(H, cv2.convexHull(pts), 1)
    Sb, Hb, Kb = S.astype(bool), H.astype(bool), known.astype(bool)
    hull_iou = float((Sb & Hb & Kb).sum() / max(((Sb | Hb) & Kb).sum(), 1))
    if debug is not None:                      # dev: the masks behind the read
        debug.update(fg=fg, fgf=fgf, known=known, sil=S, hull=H, roi=roi)
    ok = bool(hull_iou >= MIN_HULL_IOU and v - alt >= MIN_MARGIN and vis >= 0.5 and not cut)
    why = None if ok else ("shape" if hull_iou < MIN_HULL_IOU else "yaw not determined" if v - alt < MIN_MARGIN
                           else "occluded" if vis < 0.5 else "truncated")
    return {"ok": ok, "why": why, "iou": round(float(v), 3), "hull_iou": round(hull_iou, 3),
            "yaw_margin": round(float(v - alt), 3),
            "visible_frac": round(float(vis), 3),
            "base_pose": [float(t) for t in Tb[:3, 3]] + [float(t) for t in R_to_quat_wxyz(Tb[:3, :3])],
            "child_pose": [float(t) for t in Tc[:3, 3]] + [float(t) for t in R_to_quat_wxyz(Tc[:3, :3])],
            "moved_mm": round(1000 * math.hypot(x - x0, y - y0), 1),
            "turned_deg": round(math.degrees(ydist(yw, pair.yaw0)), 1),
            "n_grid": len(grid), "time_s": round(time.time() - t0, 3)}


SLOT_CHILD_RADIUS_M = 0.04  # the believed child this close (xy) to its slot on the believed base is still in it


def slot_read_for(frame: dict, task: str, label: str, objects: list[dict], arm: str, cfg: dict | None = None) -> dict:
    """``slot_read`` for ``label`` at its pick hover when the task config pins it in a support slot of another believed
    object and the belief still has it in that slot. ``frame`` = {"images", "state"}; ``objects`` = the belief
    [{label, key, pos, quat}]. Returns {"applicable": False, "why"} or {"applicable": True, "base_label", ...}."""
    from .inverse_layout import support_slot_pose, task_map
    from .slots import support_slot

    d = ((task_map().get(task) or {}).get("labels") or {}).get(label) or {}
    sl = support_slot(d.get("placement") or {})
    if sl is None:
        return {"applicable": False, "why": "not slot-pinned"}
    by = {o["label"]: o for o in objects}
    base, child = by.get(sl[0]), by.get(label)
    if base is None or child is None:
        return {"applicable": False, "why": f"no {sl[0]} in the belief"}
    sp = support_slot_pose(base["key"], base["pos"], base["quat"], child["key"], d["placement"])
    if sp is None:
        return {"applicable": False, "why": "no slot pose"}
    if math.hypot(float(child["pos"][0]) - float(sp[0][0]), float(child["pos"][1]) - float(sp[0][1])) > SLOT_CHILD_RADIUS_M:
        return {"applicable": False, "why": "not in its slot"}
    img = frame["images"].get(f"cam_{arm}_wrist")
    if img is None:
        return {"applicable": False, "why": "no wrist image"}
    c = np.asarray(base["pos"][:2], float)
    others = [(o["key"], list(o["pos"]) + list(o["quat"])) for o in objects
              if o["label"] not in (label, sl[0]) and float(np.linalg.norm(np.asarray(o["pos"][:2], float) - c)) < 0.25]
    kw = {k: v for k, v in (cfg or {}).items() if k in ("radius_m", "yaw_step_deg", "xy_step_m", "empty_hand")}
    out = slot_read(base["key"], list(base["pos"]) + list(base["quat"]), child["key"], d["placement"], img,
                    frame["state"], arm, others=others or None, **kw)
    return {"applicable": True, "base_label": sl[0], **out}
