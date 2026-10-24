"""Offline colour calibration: catalogue mean texture colour -> colour observed in the renderer.

Isaac's lighting and tone mapping make objects brighter and less saturated than their texture means
(stack_blocks: nearest-of-5 identity 84-90 % uncalibrated, 100 % with this map on held-out layouts at GT
poses). The map is a property of the renderer, not of a layout: an affine 4x3 matrix ``M`` with
``observed ~= [ref, 1] @ M``, fitted by least squares on dev layouts where the asset and pose are known.

2-fold, like the backgrounds: ``fold<k>`` holds the map fitted on the OTHER fold's episodes, so an
evaluated episode never contributes to its own calibration.

    python -m dojo.perception.color_calib --d0 ~/dojo-data/d0 --out <catalog>/color_calib.json
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np


def collect(root: Path, tasks=None, max_per_task: int = 25) -> list[dict]:
    """(fold, ref colour, observed colour) per GT object, frame 0, all views. DEV-ONLY (uses GT poses)."""
    from . import d0
    from .cameras import cameras_from_state, pose_to_T
    from .eval_g1 import fold_backgrounds
    from .inverse_layout import observed_color, shape_models
    from .render import load_asset
    from .segment import segment_views

    rows = []
    tnames = tasks or sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "logs")
    for t in tnames:
        for ep in d0.episodes(root, t)[:max_per_task]:
            obs = d0.load_obs(ep, 0)
            gt = d0.load_gt(ep, 0)
            views = segment_views(obs["images"], cameras_from_state(obs["state"]), obs["state"], fold_backgrounds(ep))
            for o in gt["objects"]:
                if o.get("clutter") or o["category"] in ("camera_stand", "dustbin") or o.get("pos") is None:
                    continue
                key = f"{o['type']}/{o['category']}/{o['category_idx']:05d}"
                try:
                    sm = shape_models(key)[0]
                except Exception:
                    continue
                col, n = observed_color(sm, pose_to_T(o["pos"], o["quat"]), views)
                if col is None or n < 30:
                    continue
                rows.append({"fold": zlib.crc32(ep.name.encode()) % 2, "key": key, "obs": col.tolist(),
                             "ref": load_asset(key).info["mean_color"], "n": n})
    return rows


def fit(rows: list[dict]) -> np.ndarray:
    A = np.array([np.r_[r["ref"], 1.0] for r in rows])
    B = np.array([r["obs"] for r in rows])
    w = np.sqrt(np.log1p([r["n"] for r in rows]))[:, None]
    M, *_ = np.linalg.lstsq(A * w, B * w, rcond=None)
    return M


def apply(M, rgb) -> np.ndarray:
    return np.clip(np.r_[np.asarray(rgb, dtype=float), 1.0] @ np.asarray(M), 0.0, 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="*")
    a = ap.parse_args(argv)
    rows = collect(a.d0, a.tasks)
    out = {"n": len(rows), "note": "fold<k> = fitted on the other fold; observed ~= [ref, 1] @ M"}
    for k in (0, 1):
        tr = [r for r in rows if r["fold"] != k]
        out[f"fold{k}"] = fit(tr).tolist()
        out[f"fold{k}_n"] = len(tr)
    out["all"] = fit(rows).tolist()
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k.endswith("_n") or k == "n"}))


if __name__ == "__main__":
    main()
