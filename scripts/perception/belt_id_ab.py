"""Belt read A/B on recorded belt frames: the same frames read with our identity (off) and with the embedding pick
(``EstimatorConfig.embed_model``), scored like Perception-2's belt_id.py / Memory's belt_g1d.py: the GT riders in view
vs the reported riders, Hungarian on xy within 10 cm; identity = the asset key. GT is read only to score.

    belt_id_ab.py ROOT OUT.jsonl [--ids ...] [--every 3] [--offset 0] [--model dinov2_s] [--weight 0.25]
    belt_id_ab.py --summary OUT.jsonl
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def score(rec, vis):
    from scipy.optimize import linear_sum_assignment

    est = [(cat, r) for cat, lst in (rec.get("Rigid") or {}).items() for r in lst]
    rows = []
    if vis and est:
        C = np.array([[np.hypot(r["default_pos"][0] - o["pos"][0], r["default_pos"][1] - o["pos"][1]) for _, r in est] for o in vis])
        for i, k in zip(*linear_sum_assignment(C)):
            if C[i, k] < 0.10:
                cat, r = est[k]
                p = r.get("_perception") or {}
                rows.append({"gt": f"Rigid/{vis[i]['category']}/{int(vis[i]['category_idx']):05d}",
                             "est": f"Rigid/{cat}/{int(r['category_idx']):05d}", "xy_mm": round(1000 * C[i, k], 1),
                             "conf": p.get("confidence"), "blob_px": p.get("blob_px"), "embed": p.get("embed")})
    return rows, len(est) - len(rows)


def summary(path):
    rows = [json.loads(line) for line in open(path)]
    for arm in [k for k in ("off", "on", "merge") if k in rows[0]]:
        c, n, conf, fp, xy, t = collections.Counter(), collections.Counter(), collections.Counter(), 0, [], []
        for r in rows:
            fp += r[arm]["fp"]
            t.append(r[arm]["s"])
            for m in r[arm]["matched"]:
                k = m["gt"].split("/")[1]
                n[k] += 1
                xy.append(m["xy_mm"])
                if m["gt"] == m["est"]:
                    c[k] += 1
                else:
                    conf[(k, m["est"].split("/")[1])] += 1
        tot = sum(n.values())
        print(f"{arm}: identity right {sum(c.values())}/{tot} ({sum(c.values()) / max(1, tot):.2f}); "
              + ", ".join(f"{k} {c[k]}/{n[k]}" for k in sorted(n))
              + f"; phantoms {fp} over {len(rows)} reads; xy median {np.median(xy) if xy else float('nan'):.1f} mm; "
              f"s/read median {np.median(t):.2f}")
        print("   confusions:", ", ".join(f"{g}->{p} {v}" for (g, p), v in conf.most_common(8)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, nargs="?")
    ap.add_argument("out", type=Path)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--model", default="dinov2_s")
    ap.add_argument("--weight", type=float, default=0.25)
    ap.add_argument("--crop", default="composite")
    ap.add_argument("--arms", nargs="*", default=["off", "merge"], help="off | on (embed_merge off) | merge (the flag's default)")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(argv)
    if a.summary:
        return summary(a.out)

    import cv2

    from dojo.perception.belt import belt_read
    from dojo.perception.cameras import cameras_from_state
    from dojo.perception.inverse_layout import EstimatorConfig, task_map
    from dojo.perception.segment import load_backgrounds, with_scene

    task = a.root.name
    fd = Path(os.environ["DOJO_BG_FOLDS"]) / "fold0"
    keys = sorted({k for d in task_map()[task]["labels"].values() if d.get("type") in ("Dynamic", "Geometry") for k in d["candidates"]})
    bgs = with_scene(load_backgrounds(fd), fd, keys)
    cal = os.environ.get("DOJO_COLOR_CALIB")
    off = EstimatorConfig(use_wrists=True, **({"color_M": json.loads(Path(cal).read_text())["fold0"]} if cal else {}))
    on = dataclasses.replace(off, embed_model=a.model, embed_weight=a.weight, embed_crop=a.crop)
    cfgs = {"off": off, "on": dataclasses.replace(on, embed_merge=False), "merge": dataclasses.replace(on, embed_merge=True)}
    open(a.out, "w").close()
    for d in sorted(a.root.iterdir()):
        if not d.is_dir() or not d.name.isdigit() or (a.ids and d.name not in a.ids):
            continue
        for fj in sorted(d.glob("frame_W*.json"), key=lambda p: int(p.stem[7:]))[a.offset::a.every]:
            j = json.loads(fj.read_text())
            tag = j["frame"]
            if not all((d / f"frame_{tag}_{c}.png").exists() for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")):
                continue
            imgs = {c: cv2.cvtColor(cv2.imread(str(d / f"frame_{tag}_{c}.png")), cv2.COLOR_BGR2RGB)
                    for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
            st = {k: np.asarray(v, float) for k, v in j["obs"]["state"].items()}
            cam = cameras_from_state(st)["cam_head"]
            vis = []
            for o in j["gt"]["objects"]:
                if o["type"] != "Rigid" or o.get("clutter") or not o.get("pos"):
                    continue
                uv, z = cam.project(np.asarray([o["pos"]], float))
                if z[0] > 0 and 15 <= uv[0, 0] < cam.w - 15 and 15 <= uv[0, 1] < cam.h - 15:
                    vis.append(o)
            row = {"layout": d.name, "step": j["step"], "n_vis": len(vis)}
            for arm in a.arms:
                cfg = cfgs[arm]
                t0 = time.time()
                rec = belt_read(imgs, st, task, j["obs"]["instruction"], bgs=bgs, cfg=cfg)
                m, fp = score(rec, vis)
                row[arm] = {"s": round(time.time() - t0, 2), "matched": m, "fp": fp, "n_est": len(m) + fp}
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("layout", "step", "n_vis")} | {
                arm: [(x["gt"].split("/")[1], x["est"].split("/")[1]) for x in row[arm]["matched"]] for arm in a.arms}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
