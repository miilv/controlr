"""Dev-only stacking-toy hover attachment and pre-release peg error from saved camera actions + physics replay.

Run in rdpy (SIMLAB_GPU=0), --root DIR with reports/*dev_pbr*report.json.gz,
actions/*dev_pbr*actions.json.gz and replay/pbr_<id>.json.gz (--every 1).
GT never enters a controller. Identity is nearest same-asset GT at the hover, then fixed through release.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dojo.perception.grasp_metric import grip_events, pose_T
from dojo.skills.kinematics import X5Kin
from dojo.skills.pegs import peg_geometry, through_hole
from dojo.skills.world import AssetDB


def audit(root):
    assets, kin, rows = AssetDB(), X5Kin(), []
    for rp in sorted((root / "reports").glob("*dev_pbr*report.json.gz")):
        lid = int(rp.name.split("_report")[0].split("_")[-1])
        if lid not in {*range(100000, 100005), *range(100030, 100050)}:
            raise ValueError("dev ids only")
        report = json.load(gzip.open(rp))
        replay_path = root / f"replay/pbr_{lid}.json.gz"
        if not replay_path.exists():
            continue
        replay = json.load(gzip.open(replay_path))
        acts = json.load(gzip.open(root / "actions" / rp.name.replace("report", "actions")))["actions"]
        grips = grip_events(acts)
        for ev in report["events"]:
            if ev.get("kind") != "consume" or ev.get("role") != "held":
                continue
            step, arm = ev["step"], ev["arm"]
            frame = replay["frames"].get(str(step))
            releases = [e for e in grips if e["arm"] == arm and e["step"] > step]
            if not frame or not releases or releases[0]["kind"] != "release":
                continue
            release = releases[0]["step"] - 1
            rf = replay["frames"].get(str(release))
            if rf is None:
                continue
            typ, cat, idx = ev["key"].split("/")
            asset = assets.get(typ, cat, int(idx), mesh=True)
            hole = through_hole(SimpleNamespace(asset=asset))[0]
            h = np.asarray(ev["T_tcp_obj7"])
            H = pose_T(h[:3], h[3:])
            tcp = kin.tcp_from_flange(pose_T(frame["flange"][arm][:3], frame["flange"][arm][3:]))
            pred = tcp @ H
            candidates = [o for o in frame["objects"] if o["category"] == cat and int(o["category_idx"]) == int(idx)]
            gt = min(candidates, key=lambda o: np.linalg.norm(np.asarray(o["pos"]) - pred[:3, 3]))
            G = pose_T(gt["pos"], gt["quat"])
            true_h = np.linalg.inv(tcp) @ G
            actual = (true_h @ np.r_[hole, 1])[:3] * 1000
            predicted = (H @ np.r_[hole, 1])[:3] * 1000
            delta = actual - predicted
            released = next(o for o in rf["objects"] if o["label"] == gt["label"])
            R = pose_T(released["pos"], released["quat"])
            base = next(o for o in rf["objects"] if o["category"] == "stack_base")
            basset = assets.get(base["type"], base["category"], base["category_idx"], mesh=True)
            bt = pose_T(base["pos"], base["quat"])
            b = SimpleNamespace(asset=basset, T=bt)
            geos = [peg_geometry(b, tag) for tag in basset.support if tag.startswith("pole/")]
            target = pose_T(ev["pose7_used"][:3], ev["pose7_used"][3:]) @ np.r_[hole, 1]
            peg = min(geos, key=lambda p: np.linalg.norm(p["p"][:2] - target[:2]))
            xy = (R @ np.r_[hole, 1])[:2] - peg["p"][:2]
            release_tcp = kin.tcp_from_flange(pose_T(rf["flange"][arm][:3], rf["flange"][arm][3:]))
            release_delta = (np.linalg.inv(release_tcp) @ R @ np.r_[hole, 1])[:3] * 1000 - predicted
            low = (asset.mesh[0] @ R[:3, :3].T + R[:3, 3])[:, 2].min()
            rows.append(dict(id=lid, label=ev["label"], gt_label=gt["label"], hover_step=step, release_step=release,
                             peg=peg["tag"], hover_hole_tcp_mm=actual.round(3).tolist(),
                             predicted_hole_tcp_mm=predicted.round(3).tolist(),
                             hover_hole_delta_tcp_mm=delta.round(3).tolist(),
                             hover_hole_error_mm=round(float(np.linalg.norm(delta)), 3),
                             release_xy_mm=round(float(np.linalg.norm(xy) * 1000), 3),
                             release_hole_error_mm=round(float(np.linalg.norm(release_delta)), 3),
                             release_hole_delta_tcp_mm=release_delta.round(3).tolist(),
                             release_bottom_below_top_mm=round(float(peg["top"] - low) * 1000, 3)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(audit(a.root), indent=2))
