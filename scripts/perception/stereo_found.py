"""DEV: does the plane-sweep height map find the objects B1's colour pipeline misses (sweep_blocks cubes)?

    DOJO_GL=0 python scripts/perception/stereo_found.py --runs DIR [DIR ...] --task sweep_blocks --out OUT

Input: agentic-perceiver call workspaces saved by integ runs (``agentic/call_*/<stamp>/``: cam_*.png at frame 0, arms
at home, ``calib0.json``, B1's record ``b1.json``; some also ``*_scan.png`` + ``calib_scan.json``). The layout id
comes from matching ``b1.json`` to the runs' reports. GT: our own settled layout records (scoring only).
Per GT object: B1 found it (a B1 object of the same category within ``--tol`` mm in xy), and the sweep's verdict on
its head silhouette (as in stereo_coin.py).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dojo.perception.cameras import Camera  # noqa: E402
from dojo.perception.render import robot_mask  # noqa: E402
from dojo.perception.stereo import REGION, View, plane_sweep, scaled_camera, table_region_mask  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stereo_coin import objects_of, sil, verdict  # noqa: E402

HOME = {"left_arm_joint_state": np.zeros(6), "right_arm_joint_state": np.zeros(6),
        "left_ee_joint_state": np.array([1.0]), "right_ee_joint_state": np.array([1.0])}


def cams_from_calib(p: Path) -> dict:
    d = json.loads(p.read_text())["calib"]
    return {c: Camera(c, np.asarray(v["K"], float), np.asarray(v["T_world_cam"], float)) for c, v in d.items()}


def scan_state() -> dict:
    from dojo.perception.scan import eye_target, pose_for, scan_state as ss

    return ss({a: pose_for(a, *eye_target(a)) for a in ("left", "right")})


def load_views(ws: Path, suffix: str, calib: str, state: dict) -> tuple[View, list[View]]:
    cams = cams_from_calib(ws / calib)
    vs = {}
    for c, cam in cams.items():
        p = ws / f"{c}{suffix}.png"
        if not p.exists():
            continue
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        vs[c] = View(img, cam, (1 - robot_mask(cam, state, dilate_px=4)).astype(np.uint8))
    ref = vs.pop("cam_head")
    ref.valid = ref.valid & table_region_mask(ref.cam, REGION)
    return ref, list(vs.values())


def b1_objects(b1: dict) -> list[dict]:
    return [{"key": o["asset"], "label": o["label"], "cat": o["category"], "pos": o["pos"], "quat": o["quat"]}
            for o in b1.get("objects", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=20.0)
    ap.add_argument("--sweep", default='{"agg":5,"fill":true,"min_blob":30,"lr":true}')
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    kw = json.loads(a.sweep)
    layouts = Path(os.path.expanduser(f"~/dojo-data/simlab/layouts/{a.task}"))
    rows, done = [], set()
    for run in map(Path, a.runs):
        reps = {}
        for rp in sorted(run.glob(f"{a.task}_*_report.json.gz")):
            lid = int(rp.name.split("_report")[0].rsplit("_", 1)[1])
            reps[lid] = json.load(gzip.open(rp))
        for ws in sorted(run.glob("agentic/call_*/*/")):
            if not (ws / "b1.json").exists() or not (ws / "calib0.json").exists():
                continue
            b1 = json.loads((ws / "b1.json").read_text())
            bo = b1_objects(b1)
            lid = None
            for i, rep in reps.items():  # the report whose frame-0 record has this B1 object
                ini = objects_of(rep["records"].get("initial", {}))
                if bo and any(np.linalg.norm(np.subtract(o["pos"], bo[0]["pos"])) < 1e-3 for o in ini):
                    lid = i
            if lid is None or lid in done or not (100000 <= lid <= 100009 or 100020 <= lid <= 100049):
                continue
            done.add(lid)
            gts = objects_of(json.loads((layouts / f"{lid}.settled.json").read_text()))  # DEV-ONLY GT
            sets = {"home": ("", "calib0.json", HOME)}
            if (ws / "calib_scan.json").exists():
                sets["scan"] = ("_scan", "calib_scan.json", scan_state())
            for sname, (suf, cal, st) in sets.items():
                ref, srcs = load_views(ws, suf, cal, st)
                sw = plane_sweep(ref, srcs, scale=1.0, **kw)
                cam = scaled_camera(ref.cam, 1.0)
                vis = ref.img.copy()
                vis[sw.on_table > 0] = (0.5 * vis[sw.on_table > 0] + (0, 80, 0)).astype(np.uint8)
                vis[sw.proud > 0] = (255, 255, 0)
                for g in gts:
                    near = [o for o in bo if o["cat"] == g["cat"]
                            and np.hypot(*(np.subtract(o["pos"], g["pos"])[:2])) * 1000 <= a.tol]
                    r = {"id": lid, "sweep": sname, "label": g["label"], "cat": g["cat"], "b1_found": bool(near),
                         **verdict(sw, sil(cam, g))}
                    rows.append(r)
                    print(json.dumps(r), flush=True)
                    cnt, _ = cv2.findContours(sil(cam, g).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, cnt, -1, (0, 255, 0) if near else (255, 0, 0), 1)
                cv2.imwrite(str(out / f"{a.task}_{lid}_{sname}.png"), np.concatenate([ref.img, vis], 1)[..., ::-1])
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
