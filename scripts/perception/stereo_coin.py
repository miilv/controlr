"""DEV: does the plane-sweep on-table test reject B1's deposit_coin phantoms, and does it keep the true coin / stand?

    DOJO_GL=0 python scripts/perception/stereo_coin.py --out DIR [--b1] [--ids 100000 ...]

Frames: the proxy integ runs' saved frames (frame 0 at home, the scan pose), layouts 100000-09. GT: our own settled
layout records (read only for scoring). B1 predictions: the runs' own records (``initial`` = frame 0, ``scan``) and,
with ``--b1``, current B1 re-run on frame 0. Per predicted / GT object: its silhouette in the head view, the share
of pixels the sweep decides (``known``), and of those on the table plane / proud. Verdict: ``rejected`` when
>= 50 % of the decided pixels are on the table and < 20 % proud, ``confirmed`` when >= 30 % proud, else ``undecided``
(``unseen`` when < 30 % of the silhouette is decided).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dojo.perception.cameras import cameras_from_state, pose_to_T  # noqa: E402
from dojo.perception.render import load_asset, robot_mask, silhouette, transform  # noqa: E402
from dojo.perception.stereo import REGION, View, plane_sweep, scaled_camera, table_region_mask  # noqa: E402

RUNS = Path(os.path.expanduser("~/dojo-data/integ/runs"))
LAYOUTS = Path(os.path.expanduser("~/dojo-data/simlab/layouts/deposit_coin"))
SRC = {i: ("S79_deposit_coin_main_a2" if i < 100005 else "S79_deposit_coin_main_b") for i in range(100000, 100010)}
LABEL_TOL_M = 0.03


def load_set(d: Path, tag: str):
    st = json.loads((d / f"{tag}_state.json").read_text())
    st = {k: np.asarray(v, float) for k, v in st.items()}
    imgs = {c: cv2.cvtColor(cv2.imread(str(d / f"{tag}_{c}.png")), cv2.COLOR_BGR2RGB)
            for c in ("cam_head", "cam_left_wrist", "cam_right_wrist") if (d / f"{tag}_{c}.png").exists()}
    return imgs, st


def views(imgs, st, region=REGION):
    cams = cameras_from_state(st)
    out = {}
    for c, img in imgs.items():
        rob = robot_mask(cams[c], st, dilate_px=4)
        out[c] = View(img, cams[c], (1 - rob).astype(np.uint8))
    out["cam_head"].valid = out["cam_head"].valid & table_region_mask(out["cam_head"].cam, region)
    return out


def objects_of(rec: dict) -> list[dict]:
    out = []
    for typ in ("Rigid", "Geometry", "Articulation"):
        for cat, lst in (rec.get(typ) or {}).items():
            for r in lst:
                if cat == "camera_stand":
                    continue
                out.append({"key": f"{typ}/{cat}/{int(r['category_idx']):05d}", "label": r.get("label") or cat, "cat": cat,
                            "pos": r["default_pos"], "quat": r["default_ori"]})
    return out


def sil(cam, o) -> np.ndarray:
    a = load_asset(o["key"])
    return silhouette(cam, transform(a.Vs, pose_to_T(o["pos"], o["quat"])), a.Fs) > 0


def pred_heights(cam, o) -> np.ndarray:
    """Height above the table of the object's own surface per head pixel (NaN off the object), at its pose."""
    from dojo.perception.stereo_eval import gt_height_map

    typ, cat, idx = o["key"].split("/")
    h, oid, _, _ = gt_height_map(cam, [{"type": typ, "category": cat, "category_idx": int(idx), "pos": o["pos"],
                                        "quat": o["quat"]}])
    return np.where(oid == 0, h, np.nan)


def verdict(sw, m: np.ndarray, ph: np.ndarray | None = None, tol: float = 0.01) -> dict:
    n = int(m.sum())
    if n == 0:
        return {"px": 0, "verdict": "offscreen"}
    kn = m & (sw.known > 0)
    k = int(kn.sum())
    on = float((sw.on_table[kn] > 0).mean()) if k else 0.0
    pr = float((sw.proud[kn] > 0).mean()) if k else 0.0
    pr_any = float((sw.proud[m] > 0).mean())
    if k < 0.3 * n:
        v = "unseen" if pr_any < 0.1 else "confirmed"
    elif pr >= 0.3:
        v = "confirmed"
    elif on >= 0.5 and pr < 0.2:
        v = "rejected"
    else:
        v = "undecided"
    out = {"px": n, "known": round(k / n, 2), "on_table": round(on, 2), "proud": round(pr, 2), "verdict": v}
    if ph is not None:  # height check: measured minus the object's own predicted surface height
        both = np.isfinite(ph) & np.isfinite(sw.height) & m
        if both.sum() >= 8:
            d = float(np.median(sw.height[both] - ph[both]))
            out.update({"dh_mm": round(d * 1000, 1), "h_px": int(both.sum()), "h_mismatch": bool(abs(d) > tol)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", nargs="*", type=int, default=list(range(100000, 100010)))
    ap.add_argument("--b1", action="store_true")
    ap.add_argument("--sweep", default='{"agg":5,"fill":true,"min_blob":30,"lr":true}')
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--b1-cache", nargs="*", default=[], help="dirs with b1_<id>.json from an earlier --b1 run")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    kw = json.loads(a.sweep)
    rows = []
    for lid in a.ids:
        run = RUNS / SRC[lid]
        d = run / f"deposit_coin_perception_verify-bundle_{lid}_look_frames"
        rep = json.load(gzip.open(run / f"deposit_coin_perception_verify-bundle_{lid}_report.json.gz"))
        preds = {k: objects_of(rep["records"][k]) for k in ("initial", "scan") if k in rep["records"]}
        i0, s0 = load_set(d, "frame0")
        cached = [Path(c).expanduser() / f"b1_{lid}.json" for c in a.b1_cache]
        cached = [c for c in cached if c.exists()]
        if cached:
            preds["b1_now"] = objects_of(json.loads(cached[0].read_text()))
        elif a.b1:
            from dojo.perception.inverse_layout import EstimatorConfig, estimate
            from dojo.perception.segment import load_backgrounds

            fold = zlib.crc32(str(lid).encode()) % 2
            fd = Path(os.path.expanduser("~/dojo-perception-data/bg_folds")) / f"fold{fold}"
            bgs = load_backgrounds(fd)
            m = fd / "cam_head_clean_unseen.png"
            if m.exists():
                bgs["cam_head_clean_unseen"] = (cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            cal = json.loads(Path(os.path.expanduser("~/dojo-perception-data/catalog/color_calib.json")).read_text())
            t0 = time.time()
            rec = estimate(i0, s0, "deposit_coin", rep.get("instruction") or "", bgs=bgs,
                           cfg=EstimatorConfig(use_wrists=True, color_M=cal[f"fold{fold}"]))
            preds["b1_now"] = objects_of(rec)
            (out / f"b1_{lid}.json").write_text(json.dumps({k: v for k, v in rec.items() if k in ("Rigid", "Geometry", "Articulation")}))
            print(lid, "B1", round(time.time() - t0, 1), "s", flush=True)
        iS, sS = load_set(d, "scan")
        v0, vS = views(i0, s0), views(iS, sS)
        sweeps = {
            "home": (v0["cam_head"], [v0["cam_left_wrist"], v0["cam_right_wrist"]]),
            "scan": (vS["cam_head"], [vS["cam_left_wrist"], vS["cam_right_wrist"]]),
            "home+scanwrists": (v0["cam_head"], [v0["cam_left_wrist"], v0["cam_right_wrist"], vS["cam_left_wrist"], vS["cam_right_wrist"]]),
        }
        layout = json.loads((LAYOUTS / f"{lid}.settled.json").read_text())  # DEV-ONLY GT
        gts = objects_of(layout)
        for sname, (ref, srcs) in sweeps.items():
            sw = plane_sweep(ref, srcs, scale=a.scale, **kw)
            cam = scaled_camera(ref.cam, a.scale)
            for src, objs in [("gt", gts)] + list(preds.items()):
                for o in objs:
                    g = next((x for x in gts if x["cat"] == o["cat"]), None)
                    err = float(np.linalg.norm(np.subtract(o["pos"], g["pos"]))) if g else None
                    r = {"id": lid, "sweep": sname, "src": src, "label": o["label"], "cat": o["cat"],
                         "err_mm": None if err is None else round(err * 1000, 1),
                         "phantom": bool(src != "gt" and (err is None or err > LABEL_TOL_M)),
                         **verdict(sw, sil(cam, o), pred_heights(cam, o))}
                    rows.append(r)
                    print(json.dumps(r), flush=True)
            if sname in ("home", "scan"):
                small = cv2.resize(ref.img, (cam.w, cam.h))
                vis = small.copy()
                vis[sw.on_table > 0] = (0.5 * vis[sw.on_table > 0] + (0, 80, 0)).astype(np.uint8)
                vis[sw.proud > 0] = (255, 255, 0)
                for src, objs in preds.items():
                    for o in objs:
                        cnt, _ = cv2.findContours(sil(cam, o).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(vis, cnt, -1, (255, 0, 0) if src == "initial" else (255, 0, 255), 1)
                cv2.imwrite(str(out / f"coin_{lid}_{sname}.png"), np.concatenate([small, vis], 1)[..., ::-1])
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
