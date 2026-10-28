"""DEV-ONLY E3, continuous stage: refine B1's frame-0 poses by the multi-view DINOv2 feature metric
(``featmetric.refine``: Nelder-Mead over x, y, yaw on the feature distance, head + wrists), for objects whose own
distance exceeds --gate-d (or all with --gate-d 0). eval_g1 before / after on the same records.

    python scripts/perception/mv/fm_refine_eval.py --out DIR [--gate-d 0.4] [--iters 40]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

CALIB = {"d0_51": "color_calib_51.json", "d0_ext": "color_calib.json", "d0_coin": "color_calib.json"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", nargs="+", default=["~/dojo-data/d0_51", "~/dojo-data/d0_ext"])
    ap.add_argument("--b1dump", default="~/dojo-data/mv/b1/dump")
    ap.add_argument("--gate-d", type=float, default=0.4)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--min-gain", type=float, default=0.02, help="keep the refined pose only if the distance drops this much")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    from dojo.perception import d0
    from dojo.perception.cameras import R_to_quat_wxyz, pose_to_T
    from dojo.perception.featmetric import FeatureMetric, refine
    from dojo.perception.mv3d_eval import SET_ALIAS, _b1_record, _score, scene_list, scene_views
    from dojo.perception.render import robot_mask

    fm = FeatureMetric("dinov2_b")
    b1 = Path(a.b1dump).expanduser()
    eps = scene_list([os.path.expanduser(r) for r in a.roots])
    if a.limit:
        eps = eps[:a.limit]
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    tot = Counter()
    with open(out / "frames.jsonl", "w") as fh:
        for ep in eps:
            root_name, task, epn = ep.parent.parent.name, ep.parent.name, ep.name
            t0 = time.time()
            meta = json.loads((b1 / {v: k for k, v in SET_ALIAS.items()}.get(root_name, root_name) / task / epn / "meta.json").read_text())
            calib = json.loads((Path(os.path.expanduser("~/dojo-perception-data/catalog")) / CALIB.get(root_name, "color_calib.json")).read_text())
            fm.set_color(calib.get(f"fold{zlib.crc32(epn.encode()) % 2}"))
            vs = scene_views(ep, "home")
            views = [{"name": v["name"], "img": v["img"], "cam": v["cam"], "robot": robot_mask(v["cam"], v["state"], dilate_px=2)} for v in vs]
            rec0 = _b1_record(meta)
            rec1 = copy.deepcopy(rec0)
            log = []
            for typ in ("Rigid", "Geometry", "Articulation"):
                for cat, lst in (rec0.get(typ) or {}).items():
                    for j, r in enumerate(lst):
                        key = f"{typ}/{cat}/{int(r['category_idx']):05d}"
                        T0 = pose_to_T(r["default_pos"], r["default_ori"])
                        try:
                            if a.gate_d > 0:
                                dd = float(fm.score(key, [T0], views)[0][0])
                                if not np.isfinite(dd) or dd <= a.gate_d:
                                    continue
                            T1, d_start, d_end = refine(fm, key, T0, views, iters=a.iters)
                        except Exception as e:  # an asset outside the catalogue, a view-less object
                            log.append({"label": r.get("label"), "error": f"{type(e).__name__}: {e}"})
                            continue
                        moved = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3])) * 1000
                        keep = np.isfinite(d_end) and d_end < d_start - a.min_gain
                        if keep:
                            e = rec1[typ][cat][j]
                            e["default_pos"] = [float(v) for v in T1[:3, 3]]
                            e["default_ori"] = [float(v) for v in R_to_quat_wxyz(T1[:3, :3])]
                        log.append({"label": r.get("label"), "key": key, "d0": round(d_start, 4), "d1": round(d_end, 4),
                                    "moved_mm": round(moved, 1), "kept": bool(keep)})
            gt = d0.load_gt(ep, 0)  # DEV-ONLY, after the refinement
            m = {"set": root_name, "task": task, "episode": epn, "frame": 0}
            c0, _ = _score(rec0, gt, task, m)
            c1, _ = _score(rec1, gt, task, m)
            changes = [{"label": k[0], "before": c0.get(k), "after": c1.get(k)} for k in c0 if c0.get(k) != c1.get(k)]
            for c in changes:
                tot[(c["before"], c["after"])] += 1
            tot["refined"] += len(log)
            tot["kept"] += sum(1 for x in log if x.get("kept"))
            row = {**m, "log": log, "changes": changes, "s": round(time.time() - t0, 1)}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(task, epn, changes, len(log), row["s"], flush=True)
    print(dict(tot))


if __name__ == "__main__":
    main()
