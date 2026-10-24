"""Debug overlays: foreground (blue tint), unknown pixels (grey), predicted silhouettes (green), optional
reference silhouettes (red; dev GT for failure analysis only). One row of three views, 2x zoom optional."""
from __future__ import annotations

import cv2
import numpy as np

from .cameras import pose_to_T
from .render import load_asset, silhouette, transform


def _contours(img, mask, color):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, color, 1)


def record_objects(rec: dict):
    for typ in ("Rigid", "Geometry", "Articulation", "Dynamic"):
        for cat, lst in (rec.get(typ) or {}).items():
            for r in lst:
                yield f"{typ}/{cat}/{int(r['category_idx']):05d}", pose_to_T(r["default_pos"], r["default_ori"]), r


def overlay(views: dict, rec: dict | None = None, ref_objects: list | None = None, zoom: int = 1) -> np.ndarray:
    """views: name -> ViewMasks. rec: layout record (green). ref_objects: [(key, T)] (red)."""
    tiles = []
    for name in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        if name not in views:
            continue
        v = views[name]
        img = cv2.cvtColor(v.img, cv2.COLOR_RGB2BGR).copy()
        unk = v.known == 0
        img[unk] = (0.5 * img[unk] + 64).astype(np.uint8)
        fg = v.fg > 0
        img[fg] = (0.6 * img[fg] + [100, 40, 0]).astype(np.uint8)
        if zoom > 1:
            img = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        for objs, col in (([(k, T) for k, T, _ in record_objects(rec)] if rec else [], (0, 255, 0)), (ref_objects or [], (0, 0, 255))):
            for key, T in objs:
                try:
                    a = load_asset(key)
                except Exception:
                    continue
                m = silhouette(v.cam, transform(a.Vs, T), a.Fs)
                if zoom > 1:
                    m = cv2.resize(m, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
                _contours(img, m, col)
        cv2.putText(img, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(img)
    return np.concatenate(tiles, axis=1) if tiles else np.zeros((10, 10, 3), np.uint8)
