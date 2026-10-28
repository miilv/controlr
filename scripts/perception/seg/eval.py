"""Twin-trained segmenter, component evaluation on held-out frames it never saw. DEV ONLY (reads GT).

    # 1. GT instance codes of the evaluation frames (CPU: Mesa llvmpipe through EGL, see CPU_GL_ENV)
    python scripts/perception/seg/eval.py gt --out ~/dojo-data/segtrain/eval_gt [--sets d0_51,d0_ext] [--frames 0]
    # 2. predictions + per-object rows (GPU: the model in this process)
    python scripts/perception/seg/eval.py pred --gt ~/dojo-data/segtrain/eval_gt --weights W --classes classes.json \\
        --out rows.jsonl [--b1 ~/dojo-data/sam/b1/objects.jsonl]
    # 3. tables
    python scripts/perception/seg/eval.py table rows.jsonl

Frames: d0_51 (genuine Isaac 5.1) and d0_ext (Isaac 6.0 proxy), ids 100000-09 and 100020-24 only (the held-out
evaluation ids: never 100010-19, never 100025+, so no training draw's id either), frame 0 by default, all three cameras.
GT masks: every GT object (task objects, clutter, fixtures) rendered from its GT pose with the catalogue mesh and the
GT camera, with the arms (FK) as occluders (``seg_twin.gt_code_map``); an object counts when >= 30 px are visible.

Per GT object: the best one-to-one prediction (greedy IoU), its IoU, class (catalogue asset) and category agreement,
whether its asset was in the training classes, and B1's frame-0 failure class on the same object (``--b1``:
the SAM bake-off's B1 run, eval_g1's scorer). Per frame: predictions on nothing (IoU < 0.3 with every GT mask and the
arms) = phantoms.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

HOME = Path.home()
SETS = {"d0_51": HOME / "dojo-data/d0_51", "d0_ext": HOME / "dojo-data/d0_ext"}
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
# CPU OpenGL (no GPU process): Mesa's llvmpipe through EGL (device 1 of the Mesa vendor on compute4)
CPU_GL_ENV = {"LIBGL_ALWAYS_SOFTWARE": "1", "EGL_DEVICE_ID": "1", "PYOPENGL_PLATFORM": "egl",
              "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json", "LP_NUM_THREADS": "2"}
MIN_VIS = 30


def held_out(i: int) -> bool:
    return 100000 <= i <= 100009 or 100020 <= i <= 100024


def eval_frames(sets, frames=(0,), tasks=None):
    out = []
    for s in sets:
        root = SETS[s]
        for td in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs"):
            if tasks and td.name not in tasks:
                continue
            for ep in sorted(td.iterdir()):
                if not (ep.is_dir() and ep.name.isdigit() and held_out(int(ep.name))):
                    continue
                for k in frames:
                    if (ep / f"frame_{k}.json").exists():
                        out.append((s, td.name, ep.name, k))
    return out


def gt_job(job, out: str):
    from PIL import Image

    from dojo.perception.seg_twin import gt_camera, gt_code_map

    s, task, ep, k = job
    d = SETS[s] / task / ep
    rec = json.loads((d / f"frame_{k}.json").read_text())
    st = {kk: np.asarray(v, float) for kk, v in rec["obs"]["state"].items()}
    od = Path(out) / s / task / ep
    od.mkdir(parents=True, exist_ok=True)
    meta = {"set": s, "task": task, "episode": ep, "frame": k, "cams": {}}
    for cam in CAMS:
        if not (d / f"frame_{k}_{cam}.png").exists():
            continue
        codes, ok = gt_code_map(gt_camera(rec["gt"], cam), rec["gt"]["objects"], st)
        Image.fromarray(codes).save(od / f"gt_{k}_{cam}.png")
        meta["cams"][cam] = {"rendered": ok}
    (od / f"gt_{k}.json").write_text(json.dumps(meta))
    return job


def cmd_gt(a):
    for kk, v in CPU_GL_ENV.items():
        os.environ.setdefault(kk, v)
    jobs = eval_frames(a.sets.split(","), tuple(int(x) for x in a.frames.split(",")))
    jobs = [j for j in jobs if a.force or not (Path(a.out) / j[0] / j[1] / j[2] / f"gt_{j[3]}.json").exists()]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, j in enumerate(ex.map(gt_job, jobs, [a.out] * len(jobs), chunksize=2)):
            if i % 50 == 0:
                print(f"[gt] {i}/{len(jobs)} {time.time() - t0:.0f}s {j}", flush=True)
    print(f"[gt] {len(jobs)} frames in {time.time() - t0:.0f}s", flush=True)


def _b1_index(path: str | None) -> dict:
    """(set, task, episode) -> [B1 row] (frame 0)."""
    from dojo.perception.eval_g1 import TOLERANCE, failure_class

    idx = defaultdict(list)
    if not path:
        return idx
    for line in open(os.path.expanduser(path)):
        r = json.loads(line)
        if r.get("frame", 0) != 0:
            continue
        r["b1_class"] = failure_class(r, TOLERANCE.get(r["task"], (None, None, "")))
        idx[(r["set"], r["task"], str(r["episode"]))].append(r)
    return idx


def _b1_for(rows: list[dict], o: dict) -> dict | None:
    """The B1 row of GT object ``o`` (same category / asset index, nearest GT position within 5 mm)."""
    from dojo.skills.artroot import row_to_asset

    p = np.asarray(row_to_asset(o)["pos"], float)
    best = None
    for r in rows:
        if r.get("category") != o.get("category") or r.get("gt_idx") != o.get("category_idx"):
            continue
        d = float(np.linalg.norm(np.asarray(r["gt_pos"], float) - p))
        if d < 0.005 and (best is None or d < best[0]):
            best = (d, r)
    return best[1] if best else None


def cmd_pred(a):
    from PIL import Image

    from dojo.perception import seg_twin as st
    from dojo.perception.seg_twin import Det, asset_key, category_of, instances, iou, match_greedy

    classes = json.load(open(os.path.expanduser(a.classes)))["names"]
    seen = set(classes)
    M = st._SegModel(os.path.expanduser(a.weights), classes, conf=a.conf)
    b1 = _b1_index(a.b1)
    gt_root = Path(a.gt).expanduser()
    metas = sorted(gt_root.glob("*/*/*/gt_*.json"))
    out = open(os.path.expanduser(a.out), "w")
    dump = open(os.path.expanduser(a.dump), "w") if a.dump else None
    t_det = []
    for mp in metas:
        meta = json.loads(mp.read_text())
        s, task, ep, k = meta["set"], meta["task"], meta["episode"], meta["frame"]
        d = SETS[s] / task / ep
        rec = json.loads((d / f"frame_{k}.json").read_text())
        objs = rec["gt"]["objects"]
        b1rows = b1.get((s, task, ep), []) if k == 0 else []
        for cam in meta["cams"]:
            img = np.asarray(Image.open(d / f"frame_{k}_{cam}.png").convert("RGB"))
            codes = np.asarray(Image.open(mp.parent / f"gt_{k}_{cam}.png")).astype(np.int64)
            t0 = time.time()
            dets = [Det(m.astype(np.uint8), key, sc, tuple(b)) for m, key, sc, b in M.detect(img, a.conf)]
            t_det.append(time.time() - t0)
            if dump is not None:   # for other streams (V's multi-view selector): class, score, box, mask polygon
                rows_d = []
                for p_ in dets:
                    poly = st.mask_polygon(p_.mask)
                    rows_d.append({"key": p_.key, "score": round(p_.score, 4), "box": [round(v, 1) for v in p_.box],
                                   "px": int(p_.mask.sum()), "poly": None if poly is None else poly.round(1).tolist()})
                dump.write(json.dumps({"set": s, "task": task, "episode": ep, "frame": k, "cam": cam, "dets": rows_d}) + "\n")
            gt = instances(codes, MIN_VIS)
            m = match_greedy(gt, dets, thr=0.0)
            robot = codes == st.ROBOT_CODE
            for gi, gm in gt.items():
                o = objs[gi]
                key = asset_key(o)
                row = {"set": s, "task": task, "episode": ep, "frame": k, "cam": cam, "gi": gi, "label": o.get("label"),
                       "key": key, "clutter": bool(o.get("clutter")), "type": o["type"], "px": int(gm.sum()),
                       "seen_class": key in seen}
                if gi in m:
                    pi, v = m[gi]
                    p = dets[pi]
                    row.update(iou=round(v, 4), pred_key=p.key, score=round(p.score, 3), class_ok=p.key == key,
                               cat_ok=category_of(p.key) == category_of(key))
                else:
                    row.update(iou=0.0, pred_key=None, score=None, class_ok=False, cat_ok=False)
                # best IoU of any prediction of the right class (the proposer looks masks up by class)
                same = [iou(gm, p.mask) for p in dets if p.key == key]
                row["iou_same_class"] = round(max(same), 4) if same else 0.0
                if b1rows and cam == "cam_head" and not o.get("clutter"):
                    r = _b1_for(b1rows, o)
                    if r is not None:
                        row["b1_class"] = r["b1_class"]
                out.write(json.dumps(row) + "\n")
            # phantoms: predictions on no GT object and not on the arms
            allm = list(gt.values())
            ph = 0
            for p in dets:
                if max([iou(p.mask, g) for g in allm] + [0.0]) < 0.3 and (p.mask & robot).sum() < 0.5 * p.mask.sum():
                    ph += 1
            out.write(json.dumps({"set": s, "task": task, "episode": ep, "frame": k, "cam": cam, "frame_row": True,
                                  "n_pred": len(dets), "n_gt": len(gt), "phantoms": ph}) + "\n")
        out.flush()
    print(f"[pred] {len(metas)} frames; detect {np.median(t_det) * 1000:.0f} ms median, "
          f"{np.percentile(t_det, 90) * 1000:.0f} ms p90; {M.info()}", flush=True)


def cmd_dump(a):
    """Detections on frame-0 head images of every allowed scene (ids 100000-09, 100020-49), for other streams."""
    from PIL import Image

    from dojo.perception import seg_twin as st
    from dojo.perception.seg_twin import category_of

    classes = json.load(open(os.path.expanduser(a.classes)))["names"]
    M = st._SegModel(os.path.expanduser(a.weights), classes, conf=a.conf)
    out = open(os.path.expanduser(a.out), "w")
    n = 0
    for s in a.sets.split(","):
        root = SETS[s]
        for td in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs"):
            for ep in sorted(td.iterdir()):
                if not (ep.is_dir() and ep.name.isdigit()):
                    continue
                i = int(ep.name)
                if not (100000 <= i <= 100009 or 100020 <= i <= 100049) or not (ep / "frame_0_cam_head.png").exists():
                    continue
                img = np.asarray(Image.open(ep / "frame_0_cam_head.png").convert("RGB"))
                dets = []
                for m, key, sc, b in M.detect(img, a.conf):
                    poly = st.mask_polygon(m)
                    dets.append({"key": key, "category": category_of(key), "bbox_xyxy": [round(v, 1) for v in b],
                                 "score": round(sc, 4), "px": int(m.sum()),
                                 "poly": None if poly is None else poly.round(1).tolist()})
                out.write(json.dumps({"set": s, "task": td.name, "episode": ep.name, "frame": 0, "cam": "cam_head",
                                      "train_id_range": i >= 100025, "dets": dets}) + "\n")
                n += 1
    print(f"[dump] {n} scenes -> {a.out}", flush=True)


def cmd_table(a):
    rows = [json.loads(line) for line in open(os.path.expanduser(a.rows))]
    objs = [r for r in rows if not r.get("frame_row")]
    frs = [r for r in rows if r.get("frame_row")]

    def stats(rs):
        n = len(rs)
        if not n:
            return "0 | - | - | - | -"
        rec = sum(r["iou"] >= 0.5 for r in rs)
        miou = np.mean([r["iou"] for r in rs])
        cls = sum(r["class_ok"] and r["iou"] >= 0.5 for r in rs)
        cat = sum(r["cat_ok"] and r["iou"] >= 0.5 for r in rs)
        return f"{n} | {rec / n:.2f} | {miou:.2f} | {cls / n:.2f} | {cat / n:.2f}"

    print("| subset | n | recall@0.5 | mean IoU | class right & IoU>=0.5 | category right & IoU>=0.5 |")
    print("|---|---|---|---|---|---|")
    for name, f in [("all objects, all cams", lambda r: True),
                    ("task objects, head", lambda r: not r["clutter"] and r["cam"] == "cam_head"),
                    ("task objects, wrists", lambda r: not r["clutter"] and r["cam"] != "cam_head"),
                    ("clutter, head", lambda r: r["clutter"] and r["cam"] == "cam_head"),
                    ("seen asset, head", lambda r: r["seen_class"] and r["cam"] == "cam_head"),
                    ("unseen asset, head", lambda r: not r["seen_class"] and r["cam"] == "cam_head")]:
        print(f"| {name} | {stats([r for r in objs if f(r)])} |")
    print("\nBy view and object size (sqrt of visible px; research report E2 bins), task objects:\n")
    print("| view | size px | n | recall@0.5 | mean IoU | class right & IoU>=0.5 | category right & IoU>=0.5 |")
    print("|---|---|---|---|---|---|---|")
    bins = [(0, 16, "<16"), (16, 32, "16-32"), (32, 64, "32-64"), (64, 1e9, "64+")]
    for cam in CAMS:
        for lo, hi, name in bins:
            rs = [r for r in objs if r["cam"] == cam and not r["clutter"] and lo <= r["px"] ** 0.5 < hi]
            if rs:
                print(f"| {cam} | {name} | {stats(rs)} |")
    for s in sorted({r["set"] for r in objs}):
        print(f"| {s}, task objects, head | {stats([r for r in objs if r['set'] == s and not r['clutter'] and r['cam'] == 'cam_head'])} |")
    print("\nB1's frame-0 class of the same object (head view, task objects):\n")
    print("| B1 class | n | recall@0.5 | mean IoU | class right & IoU>=0.5 | category right & IoU>=0.5 |")
    print("|---|---|---|---|---|---|")
    for c in ("ok", "identity", "overshoot", "gross", "missed"):
        print(f"| {c} | {stats([r for r in objs if r.get('b1_class') == c])} |")
    print("\nPer task (task objects, head):\n")
    print("| task | n | recall@0.5 | mean IoU | class right & IoU>=0.5 | category right & IoU>=0.5 |")
    print("|---|---|---|---|---|---|")
    for t in sorted({r["task"] for r in objs}):
        print(f"| {t} | {stats([r for r in objs if r['task'] == t and not r['clutter'] and r['cam'] == 'cam_head'])} |")
    for cam in CAMS:
        fs = [r for r in frs if r["cam"] == cam]
        if fs:
            print(f"\n{cam}: {len(fs)} frames, predictions {sum(r['n_pred'] for r in fs)}, phantoms "
                  f"{sum(r['phantoms'] for r in fs)} ({np.mean([r['phantoms'] for r in fs]):.2f} per frame)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gt")
    g.add_argument("--out", required=True)
    g.add_argument("--sets", default="d0_51,d0_ext")
    g.add_argument("--frames", default="0")
    g.add_argument("--workers", type=int, default=6)
    g.add_argument("--force", action="store_true")
    p = sub.add_parser("pred")
    p.add_argument("--gt", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--classes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--b1", default=str(HOME / "dojo-data/sam/b1/objects.jsonl"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--dump", default=None, help="also write every detection (class, score, box, polygon) here")
    dd = sub.add_parser("dump")
    dd.add_argument("--weights", required=True)
    dd.add_argument("--classes", required=True)
    dd.add_argument("--out", required=True)
    dd.add_argument("--sets", default="d0_51,d0_ext")
    dd.add_argument("--conf", type=float, default=0.25)
    t = sub.add_parser("table")
    t.add_argument("rows")
    a = ap.parse_args(argv)
    {"gt": cmd_gt, "pred": cmd_pred, "dump": cmd_dump, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    main()
