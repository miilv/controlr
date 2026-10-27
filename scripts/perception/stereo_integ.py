"""DEV: the plane-sweep height map on the integ runs' saved frames (frame 0 at home, the scan pose, the look pose),
against our own settled layout records, with B1's own records from the same episodes.

    DOJO_GL=0 python scripts/perception/stereo_integ.py --out OUT [--tasks ...] [--workers 4]

Per (layout, view set):
- GT objects: top-surface height error (|n_z| > 0.9 pixels, median of estimate - GT) and coverage;
- B1 objects (``records.initial`` for frame 0, ``records.scan`` for the scan set): the height CHECK, measured minus
  B1's own predicted surface height over its silhouette (median, pixels with an estimate), flagged at > ``--flag-mm``;
  truth = GT minus B1's predicted height over the same pixels; a flag is right when |truth| > ``--flag-mm`` too;
- candidates: proud blobs (>= ``--blob`` px) that no B1 silhouette explains: real (it overlaps a GT object) or
  phantom; and the GT objects B1 missed (no B1 object of the category within ``--tol`` mm) that a candidate covers.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOTS = [os.path.expanduser("~/dojo-data/integ/runs")]
ALLOWED = set(range(100000, 100010)) | set(range(100020, 100050))


def find_sets(tasks=None) -> list[tuple[str, int, Path, Path]]:
    """(task, id, look_frames dir, report) for the newest run of each (task, id) that saved scan frames."""
    best = {}
    for root in ROOTS:
        for d in Path(root).glob("*/*_look_frames"):
            if not (d / "scan_cam_head.png").exists() or not (d / "frame0_cam_head.png").exists():
                continue
            name = d.name[: -len("_look_frames")]
            lid = int(name.rsplit("_", 1)[1])
            task = name.split("_perception")[0]
            if lid not in ALLOWED or (tasks and task not in tasks):
                continue
            rep = d.parent / f"{name}_report.json.gz"
            if not rep.exists():
                continue
            mt = d.stat().st_mtime
            if (task, lid) not in best or mt > best[(task, lid)][0]:
                best[(task, lid)] = (mt, d, rep)
    return [(t, i, d, r) for (t, i), (_, d, r) in sorted(best.items())]


def gt_objects(task: str, lid: int) -> list[dict]:
    from stereo_coin import objects_of

    p = Path(os.path.expanduser(f"~/dojo-data/simlab/layouts/{task}/{lid}.settled.json"))
    if not p.exists():
        p = p.with_name(f"{lid}.json")
    return objects_of(json.loads(p.read_text()))


def as_gt(objs: list[dict]) -> list[dict]:
    out = []
    for o in objs:
        typ, cat, idx = o["key"].split("/")
        out.append({"type": typ, "category": cat, "category_idx": int(idx), "pos": o["pos"], "quat": o["quat"],
                    "label": o["label"]})
    return out


def run_one(args) -> list[dict]:
    try:
        return _run_one(args)
    except Exception as e:  # keep the batch going
        import traceback

        return [{"task": args[0], "id": args[1], "kind": "error", "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}]


def _run_one(args) -> list[dict]:
    task, lid, d, rep_p, kw, flag_mm, blob, tol, out = args
    cv2.setNumThreads(2)
    from stereo_coin import load_set, objects_of, views

    from dojo.perception.render import robot_mask
    from dojo.perception.stereo import plane_sweep, scaled_camera
    from dojo.perception.stereo_eval import gt_height_map

    rep = json.load(gzip.open(rep_p))
    recs = rep.get("records", {})
    gts = gt_objects(task, lid)  # DEV-ONLY
    sets = {"home": "frame0", "scan": "scan"}
    if (d / "look_cam_head.png").exists():
        sets["look"] = "look"
    loaded = {k: load_set(d, tag) for k, tag in sets.items()}
    vw = {k: views(*v) for k, v in loaded.items()}
    combos = {k: (vw[k]["cam_head"], [vw[k]["cam_left_wrist"], vw[k]["cam_right_wrist"]]) for k in vw}
    extra = [vw[k][c] for k in ("scan", "look") if k in vw for c in ("cam_left_wrist", "cam_right_wrist")]
    combos["home+all_wrists"] = (vw["home"]["cam_head"], [vw["home"]["cam_left_wrist"], vw["home"]["cam_right_wrist"]] + extra)
    rows = []
    for cname, (ref, srcs) in combos.items():
        sw = plane_sweep(ref, srcs, scale=1.0, **kw)
        cam = scaled_camera(ref.cam, 1.0)
        st = loaded["scan" if cname == "scan" else ("look" if cname == "look" else "home")][1]
        rob = robot_mask(cam, st, dilate_px=2)
        gtl = as_gt(gts)
        gt_h, gt_id, missing, gt_nz = gt_height_map(cam, gtl, rob)
        b1 = objects_of(recs.get("scan" if cname == "scan" else "initial", {}))
        b1l = as_gt(b1)
        pr_h, pr_id, _, _ = gt_height_map(cam, b1l, rob)
        base = {"task": task, "id": lid, "set": cname, "missing_meshes": missing, "t_s": sw.timings["total_s"]}
        reg = np.isfinite(gt_h) & (sw.valid > 0)
        rows.append({**base, "kind": "frame", "coverage": round(float(((sw.known > 0) & reg).sum() / max(reg.sum(), 1)), 3)})
        # GT objects
        for i, g in enumerate(gtl):
            vis = gt_id == i
            if vis.sum() < 8:
                continue
            est = vis & np.isfinite(sw.height)
            top = est & (gt_nz > 0.9) & (gt_h > 0.003)
            near = [o for o in b1 if o["cat"] == g["category"] and np.hypot(*(np.subtract(o["pos"], g["pos"])[:2])) * 1000 <= tol]
            r = {**base, "kind": "gt", "label": g["label"], "cat": g["category"], "px": int(vis.sum()),
                 "known": round(float((sw.known[vis] > 0).mean()), 3), "b1_found": bool(near),
                 "proud": round(float((sw.proud[vis] > 0).mean()), 3), "gt_top_mm": round(float(np.nanmax(gt_h[vis])) * 1000, 1)}
            if top.sum() >= 8:
                r["top_err_mm"] = round(float(np.median(sw.height[top] - gt_h[top])) * 1000, 1)
            rows.append(r)
        # B1 objects: the height check
        for i, o in enumerate(b1l):
            m = pr_id == i
            if m.sum() < 8:
                continue
            both = m & np.isfinite(sw.height)
            r = {**base, "kind": "b1", "label": o["label"], "cat": o["category"], "px": int(m.sum()), "h_px": int(both.sum())}
            tr = m & np.isfinite(gt_h)
            if tr.sum() >= 8:
                r["truth_mm"] = round(float(np.median(gt_h[tr] - pr_h[tr])) * 1000, 1)
            if both.sum() >= 8:
                r["dh_mm"] = round(float(np.median(sw.height[both] - pr_h[both])) * 1000, 1)
                r["flag"] = bool(abs(r["dh_mm"]) > flag_mm)
                r["truth_same_px_mm"] = round(float(np.nanmedian((gt_h - pr_h)[both])) * 1000, 1)
            rows.append(r)
        # candidates: proud blobs no B1 silhouette explains
        expl = cv2.dilate((pr_id >= 0).astype(np.uint8), np.ones((11, 11), np.uint8))
        n, lab, stt, _ = cv2.connectedComponentsWithStats((sw.proud > 0).astype(np.uint8))
        gt_any = cv2.dilate((gt_id >= 0).astype(np.uint8), np.ones((5, 5), np.uint8))
        for j in range(1, n):
            a = int(stt[j, cv2.CC_STAT_AREA])
            cm = lab == j
            if a < blob or (expl[cm] > 0).mean() > 0.5:
                continue
            hit = np.unique(gt_id[cm & (gt_id >= 0)])
            rows.append({**base, "kind": "cand", "area": a, "real": bool((gt_any[cm] > 0).mean() > 0.3),
                         "gt_hit": [gtl[k]["label"] for k in hit], "uv": [int(np.mean(np.nonzero(cm)[1])), int(np.mean(np.nonzero(cm)[0]))]})
        if out is not None and cname in ("home", "scan"):
            vis = ref.img.copy()
            vis[sw.on_table > 0] = (0.5 * vis[sw.on_table > 0] + (0, 80, 0)).astype(np.uint8)
            vis[sw.proud > 0] = (255, 255, 0)
            cnt, _ = cv2.findContours((pr_id >= 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnt, -1, (255, 0, 255), 1)
            cv2.imwrite(str(Path(out) / f"{task}_{lid}_{cname}.png"), np.concatenate([ref.img, vis], 1)[..., ::-1])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sweep", default='{"agg":5,"fill":true,"min_blob":30,"lr":true}')
    ap.add_argument("--flag-mm", type=float, default=10.0)
    ap.add_argument("--blob", type=int, default=30)
    ap.add_argument("--tol", type=float, default=20.0)
    ap.add_argument("--viz", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(t, i, d, r, json.loads(a.sweep), a.flag_mm, a.blob, a.tol, str(out) if a.viz else None) for t, i, d, r in find_sets(a.tasks)]
    print(len(jobs), "layouts", flush=True)
    with open(out / "rows.jsonl", "w") as fh, ProcessPoolExecutor(a.workers) as ex:
        for j, rows in zip(jobs, ex.map(run_one, jobs)):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(j[0], j[1], len(rows), flush=True)


if __name__ == "__main__":
    main()
