"""Active re-view experiment on synthetic episodes: does one requested wrist look fix flips and identity?

1. estimate frame 0;
2. for every object whose record carries a ``review`` request, render the scene (GT poses, textured
   catalogue meshes) from the requested wrist camera over the table-plane-warped background;
3. re-estimate with those views added; score both against GT.

    python -m dojo.perception.review_exp --d0 <synthetic root> --bg <bg dir> --out <dir> [--tasks ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cameras import Camera, head_camera, pose_to_T
from .eval_g1 import score_record
from .render import SceneRenderer, catalog_root
from .segment import load_backgrounds, table_homography_bg


def render_requested(gt_objects, cam: Camera, bg_head, ren: SceneRenderer, root: Path) -> np.ndarray:
    bg, _ = table_homography_bg(bg_head, head_camera(), cam)
    items = []
    for i, o in enumerate(gt_objects):
        key = f"{o['type']}/{o['category']}/{o['category_idx']:05d}"
        g = np.load(root / key / "geom.npz")
        items.append({"id": 10 + i, "key": key, "V": g["vertices"], "F": g["faces"], "T": pose_to_T(o["pos"], o["quat"]),
                      "glb": root / key / ("visual.glb" if (root / key / "visual.glb").exists() else "visual_lo.glb")})
    _, ids, col = ren.render(cam, items, rgb=True, ambient=0.5)
    img = bg.copy()
    m = ids >= 10
    img[m] = col[m]
    return img


def run_episode(ep: Path, bgs, ren, root):
    from . import d0
    from .inverse_layout import estimate

    obs = d0.load_obs(ep, 0)
    rec0 = estimate(obs["images"], obs["state"], obs["task"])
    reqs = []
    for typ in ("Rigid", "Geometry"):
        for lst in (rec0.get(typ) or {}).values():
            for r in lst:
                rv = r.get("_perception", {}).get("review")
                if rv:
                    reqs.append((r["label"], rv))
    gt = d0.load_gt(ep, 0)  # DEV-ONLY: scoring + rendering the synthetic re-view
    rows0, _, _ = score_record(rec0, gt, obs["task"], {"episode": ep.name, "stage": "before"})
    if not reqs:
        return rows0, rows0, 0
    gts = [o for o in gt["objects"] if not o.get("clutter")]
    from .cameras import WRIST, intrinsics

    K = intrinsics(WRIST["focal_length"], WRIST["horizontal_aperture"])
    extra = []
    for lab, rv in reqs:
        cam = Camera(f"review_{lab}", K, np.asarray(rv["T_world_cam_cv"]))
        extra.append((cam.name, cam, render_requested(gts, cam, bgs["cam_head"], ren, root)))
    rec1 = estimate(obs["images"], obs["state"], obs["task"], extra_views=extra)
    rows1, _, _ = score_record(rec1, gt, obs["task"], {"episode": ep.name, "stage": "after"})
    return rows0, rows1, len(reqs)


def main(argv=None):
    from . import d0

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="*", default=["pack_objects_into_box_random", "put_bottles_into_dustbin", "push_T", "build_tower"])
    ap.add_argument("--bg", type=Path, default=None)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    bgs = load_backgrounds(a.bg)
    ren = SceneRenderer()
    root = catalog_root()
    before, after, n_req = [], [], 0
    for t in a.tasks:
        for ep in d0.episodes(None, t):
            r0, r1, n = run_episode(ep, bgs, ren, root)
            before += r0
            after += r1
            n_req += n
            print(t, ep.name, "requests", n, flush=True)
    (a.out / "before.jsonl").write_text("\n".join(json.dumps(r) for r in before))
    (a.out / "after.jsonl").write_text("\n".join(json.dumps(r) for r in after))

    def stats(rows, sel=None):
        rs = [r for r in rows if sel is None or sel(r)]
        f = [r for r in rs if r.get("found")]
        return {"n": len(rs), "found": len(f), "id_acc": round(float(np.mean([r["id_ok"] for r in rs])) if rs else float("nan"), 3),
                "rot_gt20": sum(1 for r in f if r["rot_err_deg"] > 20), "xy_gt15": sum(1 for r in f if r["xy_err_mm"] > 15)}

    key = lambda r: (r["task"], r["episode"], r["label"])  # noqa: E731
    asked = {key(r) for r in before if r.get("has_review")}
    summary = {"n_requests": n_req,
               "all_before": stats(before), "all_after": stats(after),
               "asked_before": stats(before, lambda r: key(r) in asked), "asked_after": stats(after, lambda r: key(r) in asked)}
    by_cat = {}
    for c in sorted({r["category"] for r in before}):
        by_cat[c] = {"before": stats(before, lambda r: r["category"] == c), "after": stats(after, lambda r: r["category"] == c)}
    summary["by_category"] = by_cat
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
