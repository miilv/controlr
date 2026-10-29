"""State reads the scene's own actors cause (no robot contact).

- ``actor_change(head_before, head_after, vocabulary)``: the generic read behind the runtime's watcher. It watches every
  STANDING movable object at a public config slot of the instruction-derived vocabulary (``fixed_slot_labels``, a
  ``straight*`` place tag, type Rigid) and reports the one whose head-view box changed most, when it beats the next by a
  margin: a standing object that something other than our arms knocked over. No task name: over today's catalogue only
  make_kong has such objects (its 14 hand tiles and 4 push tiles; the opponent knocks a push tile), so every other
  vocabulary has nothing to watch (single-fixed-system audit, journal 2026-10-28-legality-generic-watch).
- ``kong_discard``: the same read over the four push tiles only (the earlier make_kong read; kept for its tests).
- ``laid_pose(head_before, head_after, key, pose7)``: where a knocked-over object now lies. The runtime first lays it
  face up where it stood (``RuntimePolicy._lay_face_up``); in make_kong's dev renders the knocked push tile lies 48 mm
  from its slot (toward the robot, turned ~180 deg) in 25 of 25 layouts, so a read cropped at the slot saw a standing
  neighbour's back. Searched in the head view around the laid pose: the silhouette that best matches the pixels that
  changed since frame 0 AND are now tile-face white. Images and the laid (config-slot) pose only.

Which hand tiles match a discard is NOT read here, and never from the config's ``same_index_as_label`` ties (that is
the task's answer): ``library.declare_kong`` matches glyphs read from the images only (``dojo.perception.identity_marks``).
"""
from __future__ import annotations

import cv2
import numpy as np

from .cameras import TABLE_Z, head_camera

PUSH_LABELS = ("mahjong5_0", "mahjong6_0", "mahjong7_0", "mahjong8_0")  # the opponent's standing tiles (config slots)


BOX_H, BOX_HALF_W = 0.065, 0.025   # the box a standing object's change is read in (m): a tile's height / half width
MARGIN = 5.0                       # mean Lab distance the most-changed box must beat the next one by


def standing_slots(vocabulary: str) -> dict:
    """{label: (x, y, place_tag)} of the vocabulary's standing movable objects at public config slots."""
    from .inverse_layout import fixed_slot_labels, task_map
    labels = (task_map().get(vocabulary) or {}).get("labels") or {}
    try:
        slots = fixed_slot_labels(vocabulary)
    except KeyError:
        return {}
    return {lab: v for lab, v in slots.items() if str(v[2] or "").startswith("straight")
            and (labels.get(lab) or {}).get("type", "Rigid") == "Rigid"}


def actor_change(head_before: np.ndarray, head_after: np.ndarray, vocabulary: str) -> dict:
    """The generic watch read: ``supported`` (the vocabulary has >= 2 standing slot objects), ``knocked`` (the label whose
    box changed most, or None), per-label ``scores``, ``margin``, ``confident`` (margin > MARGIN)."""
    slots = standing_slots(vocabulary)
    if len(slots) < 2:
        return {"supported": False, "knocked": None, "scores": {}}
    return {"supported": True, **_changed(head_before, head_after, slots, sorted(slots))}


def kong_discard(head_before: np.ndarray, head_after: np.ndarray, task: str = "make_kong",
                 tile_h: float = BOX_H, half_w: float = BOX_HALF_W) -> dict:
    from .inverse_layout import fixed_slot_labels
    return _changed(head_before, head_after, fixed_slot_labels(task), PUSH_LABELS, tile_h, half_w)


def _changed(head_before, head_after, slots: dict, labels, tile_h: float = BOX_H, half_w: float = BOX_HALF_W) -> dict:
    cam = head_camera()
    a = cv2.cvtColor(head_before, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(head_after, cv2.COLOR_RGB2LAB).astype(np.float32)
    scores = {}
    for lab in labels:
        if lab not in slots:
            continue
        x, y, _ = slots[lab]
        pts = np.array([[x + dx, y, TABLE_Z + dz] for dx in (-half_w, half_w) for dz in (0.0, tile_h)])
        uv, _ = cam.project(pts)
        x0, y0 = np.floor(uv.min(0)).astype(int)
        x1, y1 = np.ceil(uv.max(0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(cam.w, x1 + 1), min(cam.h, y1 + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        d = a[y0:y1, x0:x1] - b[y0:y1, x0:x1]
        d[..., 0] *= 0.5
        scores[lab] = float(np.sqrt((d ** 2).sum(-1)).mean())
    if not scores:
        return {"knocked": None, "scores": {}}
    best = max(scores, key=scores.get)
    rest = sorted(v for k, v in scores.items() if k != best)
    margin = scores[best] - (rest[-1] if rest else 0.0)
    return {"knocked": best, "scores": {k: round(v, 2) for k, v in scores.items()},
            "margin": round(margin, 2), "confident": bool(margin > MARGIN)}


# laid_pose: the search window around the laid pose (both directions: no fall direction assumed), its grids, and the
# "new tile face" pixels (changed since frame 0 by more than LAID_CHANGE in Lab, L weighted 0.5, and now white:
# HSV value > LAID_WHITE_V, saturation < LAID_WHITE_S). Chosen on make_kong dev renders (100000-04 / 100030-49).
LAID_WINDOW = 0.07            # m, +- in x and y
LAID_COARSE = 0.01            # m
LAID_FINE = 0.0025            # m, +- LAID_COARSE around the coarse best
LAID_YAWS = (0.0, 90.0)       # deg about world z, on top of the laid orientation (a box's silhouette repeats at 180)
LAID_CHANGE = 12.0
LAID_WHITE_V, LAID_WHITE_S = 150, 70
LAID_MIN_IOU = 0.4            # below: not found (the laid pose stays); dev located tiles scored 0.59-0.84
LAID_REACH = 0.03             # m: the match counts new face pixels within the window + this (a tile's half length)


def new_face_mask(head_before: np.ndarray, head_after: np.ndarray, change: float = LAID_CHANGE,
                  white_v: int = LAID_WHITE_V, white_s: int = LAID_WHITE_S) -> np.ndarray:
    """Pixels that changed since ``head_before`` and are now tile-face white (a knocked tile's face; the slot it left
    shows the table or another tile's back, not white)."""
    a = cv2.cvtColor(head_before, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(head_after, cv2.COLOR_RGB2LAB).astype(np.float32)
    d = a - b
    d[..., 0] *= 0.5
    hsv = cv2.cvtColor(head_after, cv2.COLOR_RGB2HSV)
    return (np.sqrt((d ** 2).sum(-1)) > change) & (hsv[..., 2] > white_v) & (hsv[..., 1] < white_s)


def _rot_z(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


def laid_pose(head_before: np.ndarray, head_after: np.ndarray, key: str, pose7, cam=None, shape=None,
              window: float = LAID_WINDOW, min_iou: float = LAID_MIN_IOU) -> dict:
    """Where the object ``key`` laid at ``pose7`` (x, y, z, qw, qx, qy, qz: lying where it stood) actually lies in the
    head view: (x, y) on a coarse then a fine grid within ``window`` of it and a turn about world z (``LAID_YAWS``),
    scored by the IoU of its head silhouette with ``new_face_mask``. Returns {"ok", "pos", "quat", "iou", "shift_mm",
    "yaw_deg", "why"}; not ok (the laid pose stays) below ``min_iou`` or with no new face pixels. ``shape``: the shape
    model (``.sil(cam, T)``; default ``refine._shape_for(key, R, None)``)."""
    from .cameras import R_to_quat_wxyz, pose_to_T

    cam = cam or head_camera()
    T0 = pose_to_T(np.asarray(pose7[:3], float), np.asarray(pose7[3:7], float))
    if shape is None:
        from .refine import _shape_for
        shape = _shape_for(key, T0[:3, :3], None)
    # the new face pixels near the laid pose only (the search window + a tile's reach, on the table plane): an arm
    # passing elsewhere in the view (the opponent's, mid-push) is not part of the match
    r = window + LAID_REACH
    corners = np.array([[T0[0, 3] + sx * r, T0[1, 3] + sy * r, T0[2, 3]] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    uv, z = cam.project(corners)
    roi = np.zeros(head_after.shape[:2], np.uint8)
    if np.all(np.asarray(z) > 0):
        cv2.fillConvexPoly(roi, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
    W = new_face_mask(head_before, head_after) & (roi > 0)
    n_w = int(W.sum())
    if n_w == 0:
        return {"ok": False, "why": "no new face pixels"}

    def score(dx, dy, yaw):
        T = T0.copy()
        T[:3, :3] = _rot_z(yaw) @ T0[:3, :3]
        T[:2, 3] = T0[:2, 3] + (dx, dy)
        S = np.asarray(shape.sil(cam, T)) > 0
        inter = int((S & W).sum())
        return (inter / float(int(S.sum()) + n_w - inter) if inter else 0.0), T

    best = (-1.0, None, 0.0, 0.0, 0.0)
    g = np.arange(-window, window + 1e-9, LAID_COARSE)
    for yaw in LAID_YAWS:
        for dx in g:
            for dy in g:
                iou, T = score(dx, dy, yaw)
                if iou > best[0]:
                    best = (iou, T, dx, dy, yaw)
    _, _, cx, cy, yaw = best
    f = np.arange(-LAID_COARSE, LAID_COARSE + 1e-9, LAID_FINE)
    for dx in cx + f:
        for dy in cy + f:
            iou, T = score(dx, dy, yaw)
            if iou > best[0]:
                best = (iou, T, dx, dy, yaw)
    iou, T, dx, dy, yaw = best
    out = {"iou": round(float(iou), 3), "shift_mm": round(1000 * float(np.hypot(dx, dy)), 1), "yaw_deg": float(yaw)}
    if T is None or iou < min_iou:
        return {**out, "ok": False, "why": f"best iou {iou:.2f} < {min_iou}"}
    return {**out, "ok": True, "pos": [float(v) for v in T[:3, 3]], "quat": [float(v) for v in R_to_quat_wxyz(T[:3, :3])],
            "why": None}
