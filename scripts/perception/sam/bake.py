"""SAM bake-off driver: B1 (or a SAM variant of it) on D0 frame 0, scored with eval_g1's scorer, plus the dumps the mask
metrics need (head foreground, GT visible silhouettes, B1's predicted silhouettes). DEV ONLY (reads GT after the
estimate).

    python scripts/perception/sam/bake.py run --out DIR [--cfg '{"sam_fg": "union"}'] [--sets d0_51,d0_ext,coin]
        [--tasks ...] [--workers 8] [--dump]
    python scripts/perception/sam/bake.py summary DIR [DIR_B]     # per task table (+ paired vs DIR_B)

Sets (ids 100000-09 and 100020-49 only; never 100010-19, never >= 100100):
  d0_51 = ~/dojo-data/d0_51 (genuine 5.1; bg_folds_51, color_calib_51), d0_ext = ~/dojo-data/d0_ext (6.0 proxy;
  bg_folds, color_calib), coin = ~/dojo-data/pmodels/d0_coin (6.0 proxy, deposit_coin, rendered by the perceiver stream).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HOME = Path.home()
SETS = {
    "d0_51": (HOME / "dojo-data/d0_51", HOME / "dojo-perception-data/bg_folds_51", HOME / "dojo-perception-data/catalog/color_calib_51.json"),
    "d0_ext": (HOME / "dojo-data/d0_ext", HOME / "dojo-perception-data/bg_folds", HOME / "dojo-perception-data/catalog/color_calib.json"),
    "coin": (HOME / "dojo-data/pmodels/d0_coin", HOME / "dojo-perception-data/bg_folds", HOME / "dojo-perception-data/catalog/color_calib.json"),
    # store_laptop_and_headphones 100000-04: Integration's genuine-5.1 frame 0 + our settled records (frames_from_run.py)
    "laptop": (HOME / "dojo-data/sam/laptop51", HOME / "dojo-perception-data/bg_folds_51", HOME / "dojo-perception-data/catalog/color_calib_51.json"),
    "sweep": (HOME / "dojo-data/sam/sweep51", HOME / "dojo-perception-data/bg_folds_51", HOME / "dojo-perception-data/catalog/color_calib_51.json"),
    # numbered-series tasks (sweep_blocks 100000-09 + 100020-24, classify_objects 100000-09): 6.0 proxy (render_series.sh)
    "series": (HOME / "dojo-data/sam/d0_series", HOME / "dojo-perception-data/bg_folds", HOME / "dojo-perception-data/catalog/color_calib.json"),
}


def allowed(i: int) -> bool:
    return 100000 <= i <= 100009 or 100020 <= i <= 100049


def frames(sets, tasks=None):
    out = []
    for s in sets:
        root = SETS[s][0]
        for td in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "logs"):
            if tasks and td.name not in tasks:
                continue
            for ep in sorted(td.iterdir()):
                if ep.is_dir() and ep.name.isdigit() and allowed(int(ep.name)) and (ep / "frame_0.json").exists():
                    out.append((s, str(ep)))
    return out


def _match(rec, gt, task):
    """eval_g1.score_record's matching (Hungarian on 3-D distance within interchangeable label groups), returning
    (gt_index -> pred) and the unmatched preds."""
    from scipy.optimize import linear_sum_assignment

    from dojo.perception.eval_g1 import flatten_pred
    from dojo.perception.inverse_layout import label_groups

    from dojo.perception.eval_g1 import gt_label, series_labels

    groups = label_groups(task)
    lab2group = {lab: gi for gi, g in enumerate(groups) for lab in g["labels"]}
    series = series_labels(task)
    gts = [(i, {**o, "label": gt_label(o, lab2group, series)}) for i, o in enumerate(gt["objects"])
           if gt_label(o, lab2group, series) is not None and not o.get("clutter") and o.get("pos") is not None]
    preds = [{**p, "label": gt_label(p, lab2group, series)} for p in flatten_pred(rec) if gt_label(p, lab2group, series) is not None]
    m, used = {}, set()
    for gi in range(len(groups)):
        G = [(i, o) for i, o in gts if lab2group[o["label"]] == gi]
        P = [(j, p) for j, p in enumerate(preds) if lab2group[p["label"]] == gi]
        if not G or not P:
            continue
        C = np.array([[np.linalg.norm(np.asarray(p["default_pos"]) - np.asarray(o["pos"])) for _, p in P] for _, o in G])
        r, c = linear_sum_assignment(C)
        for a, b in zip(r, c):
            m[G[a][0]] = P[b][1]
            used.add(P[b][0])
    return m, [p for j, p in enumerate(preds) if j not in used], preds


def _render_ids(cam, items):
    from dojo.perception.render import SceneRenderer

    global _SR
    try:
        _SR
    except NameError:
        _SR = SceneRenderer(cam.w, cam.h)
    _, ids, _ = _SR.render(cam, items)
    return ids


def run_frame(job):
    set_name, ep, cfg_over, out_root, dump = job
    import cv2

    from dojo.perception import d0
    from dojo.perception.cameras import cameras_from_state, pose_to_T
    from dojo.perception.eval_g1 import fold_backgrounds, score_record
    from dojo.perception.inverse_layout import EstimatorConfig, estimate
    from dojo.perception.render import load_asset, robot_items
    from dojo.perception.segment import segment_views

    root, folds, calib = SETS[set_name]
    os.environ["DOJO_BG_FOLDS"] = str(folds)
    ep = Path(ep)
    obs = d0.load_obs(ep, 0)
    kw = {"color_M": json.loads(calib.read_text())[f"fold{zlib.crc32(ep.name.encode()) % 2}"]}
    kw.update(cfg_over or {})
    cfg = EstimatorConfig(use_wrists=True, **kw)
    bgs = fold_backgrounds(ep)
    t0 = time.time()
    try:
        rec = estimate(obs["images"], obs["state"], obs["task"], obs["instruction"], bgs=bgs, cfg=cfg)
        err = None
    except Exception as e:
        import traceback

        rec, err = {}, f"{type(e).__name__}: {e} | {traceback.format_exc()[-600:]}"
    rt = time.time() - t0
    gt = d0.load_gt(ep, 0)  # DEV-ONLY, after the estimate
    task = obs["task"]
    meta = {"set": set_name, "task": task, "episode": ep.name, "frame": 0}
    rows, n_gt, n_pred = score_record(rec, gt, task, meta)
    match, unmatched, preds = _match(rec, gt, task) if rec else ({}, [], [])
    # phantoms: predictions matched to nothing, or whose position is > 40 mm from every GT object (clutter included)
    allpos = np.array([o["pos"] for o in gt["objects"] if o.get("pos") is not None]) if gt["objects"] else np.zeros((0, 3))

    def far(p):
        if not len(allpos):
            return True
        return float(np.min(np.linalg.norm(allpos[:, :2] - np.asarray(p["default_pos"])[:2], axis=1))) > 0.04

    phantoms = len(unmatched) + sum(1 for p in match.values() if far(p))
    fr = {**meta, "runtime_s": rt, "error": err, "n_gt": n_gt, "n_pred": n_pred, "phantoms": phantoms,
          "phantom_unmatched": len(unmatched), "timings": rec.get("_perception", {}).get("timings"),
          "sam": rec.get("_perception", {}).get("sam")}
    if dump:
        d = Path(out_root) / "dump" / set_name / task / ep.name
        d.mkdir(parents=True, exist_ok=True)
        cams = cameras_from_state(obs["state"])
        cam = cams["cam_head"]
        v = segment_views({"cam_head": obs["images"]["cam_head"]}, {"cam_head": cam}, obs["state"], bgs, thr=cfg.fg_thr)["cam_head"]
        cv2.imwrite(str(d / "fg.png"), v.fg * 255)
        cv2.imwrite(str(d / "known.png"), v.known * 255)
        items, objs = [], []
        for i, o in enumerate(gt["objects"]):
            key = f"{o['type']}/{o['category']}/{o['category_idx']:05d}"
            ok = o.get("pos") is not None
            if ok:
                try:
                    a = load_asset(key)
                    items.append({"id": i + 1, "key": key, "V": a.Vg, "F": a.Fg, "T": pose_to_T(o["pos"], o["quat"])})
                except Exception:
                    ok = False
            objs.append({"i": i, "label": o.get("label"), "key": key, "clutter": bool(o.get("clutter")), "rendered": ok,
                         "pos": o.get("pos")})
        ids = _render_ids(cam, items + robot_items(obs["state"]))
        ids_gt = np.where(ids < 1000, ids, 0).astype(np.uint16)
        cv2.imwrite(str(d / "gt_ids.png"), ids_gt)
        # B1's predicted silhouettes (unoccluded, each alone), for its points and its fit masks
        pr = []
        for gi_, p in match.items():
            pr.append((gi_, p))
        pr += [(-1, p) for p in unmatched]
        pmeta = []
        for gi_, p in pr:
            key = f"{p['type']}/{p['category']}/{p['category_idx']:05d}"
            try:
                a = load_asset(key)
                pid = _render_ids(cam, [{"id": 1, "key": key, "V": a.Vg, "F": a.Fg, "T": pose_to_T(p["default_pos"], p["default_ori"])}])
                ys, xs = np.nonzero(pid == 1)
                uv = [float(xs.mean() + 0.5), float(ys.mean() + 0.5)] if len(xs) else None
                area = int(len(xs))
            except Exception:
                uv, area = None, 0
            pmeta.append({"gt_i": gi_, "label": p.get("label"), "key": key, "pos": p["default_pos"], "ori": p["default_ori"],
                          "uv": uv, "area": area,
                          "far": bool(far(p))})
        (d / "meta.json").write_text(json.dumps({**meta, "instruction": obs["instruction"], "objects": objs, "preds": pmeta}))
    return {"frame": fr, "objects": rows}


def cmd_run(a):
    sets = a.sets.split(",")
    jobs = [(s, ep, json.loads(a.cfg) if a.cfg else None, str(a.out), a.dump) for s, ep in frames(sets, a.tasks)]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps({"cfg": a.cfg, "sets": sets, "tasks": a.tasks, "n_frames": len(jobs),
                                                   "sam_addr": os.environ.get("DOJO_SAM_ADDR", ""), "argv": sys.argv}))
    print(len(jobs), "frames", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex, open(a.out / "objects.jsonl", "w") as fo, \
            open(a.out / "frames.jsonl", "w") as ff:
        for i, res in enumerate(ex.map(run_frame, jobs)):
            ff.write(json.dumps(res["frame"]) + "\n")
            for r in res["objects"]:
                fo.write(json.dumps(r) + "\n")
            if res["frame"].get("error"):
                print("ERR", res["frame"]["task"], res["frame"]["episode"], res["frame"]["error"][:300], flush=True)
            if i % 20 == 0:
                print(i, res["frame"]["task"], res["frame"]["episode"], round(res["frame"]["runtime_s"], 1),
                      round(time.time() - t0), "s", flush=True)
    print("done", round(time.time() - t0), "s")


def _load(dirs):
    """One run dir, or several joined by commas (a baseline split over set lists)."""
    rows, frs = [], []
    for d in str(dirs).split(","):
        d = Path(d)
        rows += [json.loads(x) for x in open(d / "objects.jsonl") if x.strip()]
        frs += [json.loads(x) for x in open(d / "frames.jsonl") if x.strip()]
    return rows, frs


def task_table(rows, frs):
    from collections import defaultdict

    from dojo.perception.eval_g1 import TOLERANCE, failure_class

    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    ph = defaultdict(int)
    nfr = defaultdict(int)
    rt = defaultdict(list)
    for f in frs:
        ph[f["task"]] += f.get("phantoms", 0)
        nfr[f["task"]] += 1
        rt[f["task"]].append(f["runtime_s"])
    out = {}
    for t, rs in sorted(by.items()):
        tol = TOLERANCE.get(t, (None, None, ""))
        c = {"ok": 0, "identity": 0, "overshoot": 0, "gross": 0, "missed": 0}
        for r in rs:
            c[failure_class(r, tol)] += 1
        within = c["ok"] + c["identity"]  # pose within tolerance (failure_class: 15 mm where the task has none)
        out[t] = {"n": len(rs), "frames": nfr[t], "within": within, **c, "phantoms": ph[t],
                  "rt_med": float(np.median(rt[t])) if rt[t] else None}
    return out


from collections import defaultdict  # noqa: E402


def cmd_summary(a):
    A = task_table(*_load(a.dir))
    B = task_table(*_load(a.base)) if a.base else None
    for t in a.exclude or ():
        A.pop(t, None)
        if B:
            B.pop(t, None)
    keys = ("within", "gross", "missed", "identity", "phantoms")
    hdr = "| task | frames | n | " + " | ".join(keys) + " | s/frame |"
    print(hdr)
    print("|" + "---|" * (len(keys) + 4))
    tot = {k: 0 for k in keys + ("n", "frames")}
    totb = {k: 0 for k in keys}
    for t, d in A.items():
        cells = []
        for k in keys:
            if B and t in B:
                cells.append(f"{B[t][k]} -> {d[k]}")
                totb[k] += B[t][k]
            else:
                cells.append(str(d[k]))
            tot[k] += d[k]
        tot["n"] += d["n"]
        tot["frames"] += d["frames"]
        rt = f"{B[t]['rt_med']:.1f} -> {d['rt_med']:.1f}" if B and t in B else f"{d['rt_med']:.1f}"
        print(f"| {t} | {d['frames']} | {d['n']} | " + " | ".join(cells) + f" | {rt} |")
    cells = [f"{totb[k]} -> {tot[k]}" if B else str(tot[k]) for k in keys]
    print(f"| **all** | {tot['frames']} | {tot['n']} | " + " | ".join(cells) + " | |")
    if B:
        fx, br = paired_changes(a.dir, a.base, a.exclude or ())
        print(f"\npaired objects: within-tol fixed {len(fx)}, broken {len(br)}")
        for name, lst in (("fixed", fx), ("broken", br)):
            by = defaultdict(int)
            for t in lst:
                by[t[0]] += 1
            print(f"  {name}: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))


def paired_changes(dir_a, dir_b, exclude=()):
    """Objects (task, episode, label, GT position) within tolerance in one run and not in the other."""
    from dojo.perception.eval_g1 import TOLERANCE, failure_class

    def ok(rows):
        out = {}
        for r in rows:
            if r["task"] in exclude:
                continue
            k = (r["task"], r["episode"], r["label"], tuple(round(x, 4) for x in r["gt_pos"]))
            out[k] = failure_class(r, TOLERANCE.get(r["task"], (None, None, ""))) in ("ok", "identity")
        return out

    A, B = ok(_load(dir_a)[0]), ok(_load(dir_b)[0])
    keys = set(A) & set(B)
    return sorted(k for k in keys if A[k] and not B[k]), sorted(k for k in keys if B[k] and not A[k])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--cfg", default="")
    r.add_argument("--sets", default="d0_51,d0_ext,coin")
    r.add_argument("--tasks", nargs="*")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--dump", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("dir")
    s.add_argument("base", nargs="?")
    s.add_argument("--exclude", nargs="*", help="tasks left out of the table")
    a = ap.parse_args(argv)
    {"run": cmd_run, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
