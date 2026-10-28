"""DEV-ONLY: does the doubt signal's feature distance sharpen at grasp time? Pre-grasp frames of stream C's per-step
sequences (``seqev/*.npz``: head + both wrists a few steps before a grasp, plus frame 0, with the GT rows).

The grasped object = the GT object nearest the grasping arm's flange. Its right pose (GT) against wrong ones (GT
shifted 15 / 30 mm along x and y, turned 90 / 180 deg) is scored by ``featmetric`` on:
- frame 0's head + wrists (GT at frame 0);
- the pre-grasp head alone;
- the pre-grasp grasping wrist alone;
- the pre-grasp head + both wrists.
The output is the AUC (right below wrong) per view set and per perturbation.

    python scripts/perception/mv/doubt_grasp_eval.py --bundles ~/dojo-data/mv/seqev --out OUT.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import zlib
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from scipy.spatial.transform import Rotation

    from dojo.perception.cameras import cameras_from_state, pose_to_T
    from dojo.perception.featmetric import FeatureMetric
    from dojo.perception.render import load_asset, robot_mask
    from dojo.skills.artroot import row_to_asset

    fm = FeatureMetric("dinov2_b")
    cal = json.loads((Path(os.path.expanduser("~/dojo-perception-data/catalog")) / "color_calib.json").read_text())
    out = open(Path(a.out).expanduser(), "w")
    for f in sorted(Path(a.bundles).expanduser().glob("*.npz")):
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        fm.set_color(cal.get(f"fold{zlib.crc32(str(meta['layout_id']).encode()) % 2}"))
        st = {k: np.asarray(v, float) for k, v in json.loads(str(z["state"])).items()}
        st0 = {k: np.asarray(v, float) for k, v in json.loads(str(z["state0"])).items()}
        gt, gt0 = json.loads(str(z["gt"])), json.loads(str(z["gt0"]))
        arm = meta["arm"]
        flange = np.asarray(st[f"{arm}_ee_pose"][:3], float)
        rows = [row_to_asset(o) for o in gt if o.get("pos") is not None and o.get("category") != "camera_stand"]
        if not rows:
            continue
        o = min(rows, key=lambda r: np.linalg.norm(np.asarray(r["pos"]) - flange))
        key = f"{o['type']}/{o['category']}/{int(o['category_idx']):05d}"
        try:
            asset = load_asset(key)
        except Exception:
            continue
        o0 = next((row_to_asset(r) for r in gt0 if r.get("inst_name") == o.get("inst_name") and r.get("pos") is not None), None)

        def poses(row):
            T = pose_to_T(row["pos"], row["quat"])
            c = T[:3, :3] @ ((asset.V.min(0) + asset.V.max(0)) / 2) + T[:3, 3]
            out_ = {"right": T}
            for dmm in (15, 30):
                for ax in (0, 1):
                    T2 = T.copy()
                    T2[ax, 3] += dmm / 1000
                    out_[f"shift{dmm}_{'xy'[ax]}"] = T2
            for deg in (90, 180):
                Rz = Rotation.from_euler("z", np.deg2rad(deg)).as_matrix()
                T2 = np.eye(4)
                T2[:3, :3] = Rz @ T[:3, :3]
                T2[:3, 3] = Rz @ (T[:3, 3] - c) + c
                out_[f"yaw{deg}"] = T2
            return out_

        def views(prefix, state, names):
            cams = cameras_from_state(state)
            return [{"name": n, "img": z[f"{prefix}{n}"], "cam": cams[n], "robot": robot_mask(cams[n], state, dilate_px=2)}
                    for n in names if f"{prefix}{n}" in z.files and n in cams]

        sets = {"pre_head": views("img_", st, ["cam_head"]),
                "pre_wrist": views("img_", st, [f"cam_{arm}_wrist"]),
                "pre_all": views("img_", st, ["cam_head", "cam_left_wrist", "cam_right_wrist"])}
        if o0 is not None:
            sets["frame0_all"] = views("img0_", st0, ["cam_head", "cam_left_wrist", "cam_right_wrist"])
        rec = {"bundle": f.name, "task": meta["task"], "layout_id": meta["layout_id"], "arm": arm, "key": key,
               "dist_mm": round(float(np.linalg.norm(np.asarray(o["pos"]) - flange)) * 1000, 1), "d": {}}
        for name, vv in sets.items():
            P = poses(o0 if name == "frame0_all" else o)
            rec["d"][name] = {k: (None if not np.isfinite(x) else round(float(x), 4))
                              for k, x in zip(P, [fm.score(key, [T], vv)[0][0] for T in P.values()])}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f.name, key, rec["dist_mm"], {k: v.get("right") for k, v in rec["d"].items()}, flush=True)


if __name__ == "__main__":
    main()
