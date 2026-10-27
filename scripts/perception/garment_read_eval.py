"""Score the camera garment read (``dojo.perception.garment``) against the twin's GT on rendered frame 0s
(``garment_frames.py`` output): per layout the asset picked, the yaw error and each named point's xy error.

    python scripts/perception/garment_read_eval.py ~/dojo-data/gennew/garment_frames [--bg BG_DIR] [--templates JSON]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--bg", default=None, help="dir with cam_head.png (the empty table at the home pose)")
    ap.add_argument("--templates", default=None, help="templates JSON (default: built from <root>/templates)")
    ap.add_argument("--tasks", default="fold_clothes,fold_clothes_random")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    import cv2

    from dojo.perception import garment as GR
    root = Path(a.root).expanduser()
    tpl = GR.load_templates(a.templates) if a.templates else \
        GR.load_templates_dict(GR.build_templates(root))
    bg = None
    if a.bg:
        bg = cv2.cvtColor(cv2.imread(str(Path(a.bg).expanduser() / "cam_head.png")), cv2.COLOR_BGR2RGB)
    rows = []
    for task in a.tasks.split(","):
        for d in sorted((root / task).glob("1*")):
            if not (d / "cam_head.png").exists():
                continue
            img = cv2.cvtColor(cv2.imread(str(d / "cam_head.png")), cv2.COLOR_BGR2RGB)
            gt = np.load(d / "gt.npz")
            meta = json.loads((d / "meta.json").read_text())
            r = GR.read_garment(img, tpl, bg=bg)
            kp_gt = {str(k): np.asarray(v, float) for k, v in zip(gt["kp_names"], gt["kp"])}
            q = np.asarray(gt["quat"], float)
            yaw_gt = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            if r is None:
                rows.append({"task": task, "seed": d.name, "asset": meta.get("asset"), "read": None})
                print(f"{task} {d.name}: no read", flush=True)
                continue
            q2 = r["quat"]
            yaw = 2 * math.atan2(q2[3], q2[0])
            err = {k: float(np.linalg.norm(np.asarray(r["keypoints"][k])[:2] - kp_gt[k][:2]))
                   for k in kp_gt if k in r["keypoints"]}
            row = {"task": task, "seed": d.name, "asset": meta.get("asset"), "read_asset": r["asset"],
                   "iou": round(r["iou"], 3), "yaw_err_deg": round(math.degrees((yaw - yaw_gt + math.pi) % (2 * math.pi) - math.pi), 2),
                   "kp_err_mm": {k: round(v * 1000, 1) for k, v in err.items()},
                   "kp_err_max_mm": round(1000 * max(err.values()), 1), "kp_err_med_mm": round(1000 * float(np.median(list(err.values()))), 1)}
            rows.append(row)
            print(f"{task} {d.name}: {meta.get('asset')} -> {r['asset']} iou {r['iou']:.2f} yaw err {row['yaw_err_deg']:+.1f} deg, "
                  f"points med {row['kp_err_med_mm']:.0f} max {row['kp_err_max_mm']:.0f} mm", flush=True)
    ok = [r for r in rows if r.get("read_asset")]
    if ok:
        print(f"n {len(rows)} read {len(ok)} asset right {sum(r['asset'] == r['read_asset'] for r in ok)}; "
              f"point error median {np.median([r['kp_err_med_mm'] for r in ok]):.0f} mm, worst point per layout median "
              f"{np.median([r['kp_err_max_mm'] for r in ok]):.0f} / max {max(r['kp_err_max_mm'] for r in ok):.0f} mm; "
              f"|yaw err| median {np.median([abs(r['yaw_err_deg']) for r in ok]):.1f} deg", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
