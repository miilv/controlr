"""Offline evaluation of ``dojo.perception.base_refit`` on recorded camera runs (dev only; GT for scoring only).

    DOJO_GL=0 python scripts/perception/base_refit_eval.py DATA_DIR > rows.jsonl

DATA_DIR holds ``runs/<run>/{*_report.json.gz, *_episodes.jsonl, *_look_frames/frame0_*}``, ``layouts/<id>.settled.json``
(the episodes' own settled records), ``bg/<calib set>/`` (the frozen calibration backgrounds) and ``cat/<Type>/<cat>/<idx>/
geom.npz`` (catalogue meshes). For every Rigid object of the first planning-time belief snapshot: its origin xy error
against the layout before and after a proposed refit, the refit's reading, and the write verdict (harmful = adds more
than 2 mm, helpful = removes more than 2 mm; EXPERIMENTS adopt rule)."""
import glob
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    root = Path(sys.argv[1])
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import cv2
    from dojo.perception import base_refit as BR
    from dojo.perception.cameras import cameras_from_state, pose_to_T
    from dojo.perception.segment import load_backgrounds, segment_views
    bg_dir = next(p for p in (root / "bg").iterdir() if p.is_dir())
    bgs = load_backgrounds(bg_dir)
    bands = {}
    for rd in sorted((root / "runs").iterdir()):
        rep = json.load(gzip.open(next(rd.glob("*_report.json.gz"))))
        ep = json.loads(next(rd.glob("*_episodes.jsonl")).read_text().splitlines()[-1])
        lid = int(os.path.basename(ep["layout"]).split(".")[0])
        lay = json.loads((root / "layouts" / os.path.basename(ep["layout"])).read_text())
        lf = next(rd.glob("*_look_frames"))
        state = json.loads((lf / "frame0_state.json").read_text())
        imgs = {}
        for c in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            p = lf / f"frame0_{c}.png"
            if p.exists():
                imgs[c] = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        cams = cameras_from_state(state)
        views = segment_views(imgs, {k: cams[k] for k in imgs}, state, bgs)
        snap = rep["dev_belief_probe"]["snaps"][0]["objects"]
        gt = {r["label"]: r for _, recs in lay.get("Rigid", {}).items() for r in recs}
        for lab, g in gt.items():
            b = snap.get(lab)
            if b is None:
                continue
            key = b["key"]
            gp = root / "cat" / key / "geom.npz"
            if not gp.exists():
                continue
            if key not in bands:
                z = np.load(gp)
                bands[key] = BR.base_band(z["vertices"], z["faces"])
            T = pose_to_T(b["pos"], b["quat"])
            r = BR.refit_xy(views, T, bands[key])
            e0 = float(np.hypot(b["pos"][0] - g["default_pos"][0], b["pos"][1] - g["default_pos"][1]))
            p1 = np.asarray(b["pos"][:2]) + (np.asarray(r["dxy"]) if r["ok"] else 0.0)
            e1 = float(np.hypot(p1[0] - g["default_pos"][0], p1[1] - g["default_pos"][1]))
            verdict = "none" if not r["ok"] else ("harmful" if e1 > e0 + 0.002 else "helpful" if e1 < e0 - 0.002 else "neutral")
            print(json.dumps({"run": rd.name, "id": lid, "label": lab, "key": key, "key_ok": key == f"Rigid/{g['category']}/{int(g['category_idx']):05d}",
                              "err0_mm": round(1000 * e0, 1), "err1_mm": round(1000 * e1, 1), "write": verdict, **r}), flush=True)


if __name__ == "__main__":
    main()
