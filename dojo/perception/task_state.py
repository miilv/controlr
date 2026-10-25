"""Task-specific state reads that the checkers need beyond poses.

- ``kong_discard(head_before, head_after)``: make_kong. The four "push" tiles mahjong5_0..mahjong8_0 stand at
  config slots and are declared ``same_index_as_label`` with the hand triples mahjong0..3, so no glyph
  identity is needed: the triple to expose is the one whose push tile the Franka knocked over. Compares the
  head view at the four slots before and after the discard; returns the knocked label, its triple and the
  per-slot change scores (mean Lab distance in a box around each standing tile).
"""
from __future__ import annotations

import cv2
import numpy as np

from .cameras import TABLE_Z, head_camera

PUSH_TO_TRIPLE = {"mahjong5_0": "mahjong0", "mahjong6_0": "mahjong1", "mahjong7_0": "mahjong2", "mahjong8_0": "mahjong3"}


def kong_discard(head_before: np.ndarray, head_after: np.ndarray, task: str = "make_kong",
                 tile_h: float = 0.065, half_w: float = 0.025) -> dict:
    from .inverse_layout import fixed_slot_labels

    slots = fixed_slot_labels(task)
    cam = head_camera()
    a = cv2.cvtColor(head_before, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(head_after, cv2.COLOR_RGB2LAB).astype(np.float32)
    scores = {}
    for lab in PUSH_TO_TRIPLE:
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
        return {"knocked": None, "triple": None, "scores": {}}
    best = max(scores, key=scores.get)
    rest = sorted(v for k, v in scores.items() if k != best)
    margin = scores[best] - (rest[-1] if rest else 0.0)
    return {"knocked": best, "triple": PUSH_TO_TRIPLE[best], "scores": {k: round(v, 2) for k, v in scores.items()},
            "margin": round(margin, 2), "confident": bool(margin > 5.0)}
