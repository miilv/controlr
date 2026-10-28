"""State reads the scene's own actors cause (no robot contact).

- ``actor_change(head_before, head_after, vocabulary)``: the generic read behind the runtime's watcher. It watches every
  STANDING movable object at a public config slot of the instruction-derived vocabulary (``fixed_slot_labels``, a
  ``straight*`` place tag, type Rigid) and reports the one whose head-view box changed most, when it beats the next by a
  margin: a standing object that something other than our arms knocked over. No task name: over today's catalogue only
  make_kong has such objects (its 14 hand tiles and 4 push tiles; the opponent knocks a push tile), so every other
  vocabulary has nothing to watch (single-fixed-system audit, journal 2026-10-28-legality-generic-watch).
- ``kong_discard``: the same read over the four push tiles only (the earlier make_kong read; kept for its tests).

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
