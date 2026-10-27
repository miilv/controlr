"""SAM bake-off, component metrics: SAM masks vs B1's static-background foreground, against GT visible silhouettes.
DEV ONLY (reads the GT silhouettes bake.py dumped). Talks to a running ``python -m dojo.perception.sam_masks serve``.

    DOJO_SAM_ADDR=SOCK python scripts/perception/sam/masks.py run DUMP_DIR OUT.jsonl [--models sam2.1_small,sam2.1_large,sam3.1]
    python scripts/perception/sam/masks.py table OUT.jsonl

Per GT target object (visible >= 30 px in the head view):
  b1_blob: IoU of the B1 foreground component that overlaps it most; b1_cov: share of it that is foreground.
  <model>@gt / @b1: a click at the object's most interior pixel (oracle) / at B1's matched prediction's projected
  centre (what the runtime has). SAM 2.1: top-score mask ("top") and best of 3 ("best3"). "fit": REAL_RSI's
  proposer, i.e. clicks at B1's centre + 4 offsets, the mask whose IoU with B1's rendered silhouette is highest.
  text: best IoU among SAM 3.1's text masks for the scene vocabulary.
Per frame: text masks on nothing (IoU < 0.3 with every GT object incl. clutter: phantom candidates), on clutter; B1
foreground and text-union pixels outside every GT silhouette (+3 px): shadows / table.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOTS = {"d0_51": Path.home() / "dojo-data/d0_51", "d0_ext": Path.home() / "dojo-data/d0_ext",
         "coin": Path.home() / "dojo-data/pmodels/d0_coin", "laptop": Path.home() / "dojo-data/sam/laptop51",
         "sweep": Path.home() / "dojo-data/sam/sweep51", "series": Path.home() / "dojo-data/sam/d0_series"}


def iou(a, b):
    a, b = a > 0, b > 0
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def interior_point(m):
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    v, u = np.unravel_index(int(np.argmax(d)), d.shape)
    return [float(u) + 0.5, float(v) + 0.5]


def best_blob_iou(fg, gt):
    n, lab = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
    ov = np.bincount(lab[gt > 0].ravel(), minlength=n)
    ov[0] = 0
    if ov.max() == 0:
        return 0.0
    return iou(lab == int(np.argmax(ov)), gt)


def run(a):
    from dojo.perception import sam_masks
    from dojo.perception.inverse_layout import label_groups, restrict_by_instruction

    c = sam_masks.client()
    assert c is not None, "no SAM server (DOJO_SAM_ADDR)"
    models = a.models.split(",")
    dirs = sorted(Path(a.dump).glob("*/*/*/meta.json"))
    if a.only:
        dirs = [p for p in dirs if any(o in str(p) for o in a.only)]
    out = open(a.out, "w")
    t0 = time.time()
    for k, mp in enumerate(dirs):
        d = mp.parent
        meta = json.loads(mp.read_text())
        img = cv2.cvtColor(cv2.imread(str(ROOTS[meta["set"]] / meta["task"] / meta["episode"] / "frame_0_cam_head.png")), cv2.COLOR_BGR2RGB)
        ids = cv2.imread(str(d / "gt_ids.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        fg = (cv2.imread(str(d / "fg.png"), 0) > 0).astype(np.uint8)
        known = (cv2.imread(str(d / "known.png"), 0) > 0).astype(np.uint8)
        objs = meta["objects"]
        has_clutter = any(o["clutter"] for o in objs)
        groups = restrict_by_instruction(label_groups(meta["task"]), meta["instruction"])
        timing = {}
        t1 = time.time()
        voc = sam_masks.vocabulary(groups, meta["instruction"])
        tm = c.text(img, voc, thr=0.4) if voc else []
        lim = 0.08 * img.shape[0] * img.shape[1]
        tm = [m for m in tm if 0 < int(m.mask.sum()) <= lim]
        timing["text"] = time.time() - t1
        anyobj = (ids > 0).astype(np.uint8)
        near_obj = cv2.dilate(anyobj, np.ones((7, 7), np.uint8))
        text_union = np.zeros_like(fg)
        for m in tm:
            text_union |= m.mask
        frame = {"kind": "frame", "set": meta["set"], "task": meta["task"], "episode": meta["episode"], "vocab": voc,
                 "n_text": len(tm), "has_clutter": has_clutter,
                 "b1_false_fg_px": int((fg & (1 - near_obj)).sum()), "text_false_px": int((text_union & known & (1 - near_obj)).sum()),
                 "b1_far_preds": sum(1 for p in meta["preds"] if p["far"])}
        on_nothing = on_clutter = 0
        for m in tm:
            best, bi = 0.0, -1
            for o in objs:
                g = ids == o["i"] + 1
                if g.any():
                    v = iou(m.mask, g)
                    if v > best:
                        best, bi = v, o["i"]
            if best < 0.3:
                on_nothing += 1
            elif objs[bi]["clutter"]:
                on_clutter += 1
        frame.update(text_on_nothing=on_nothing, text_on_clutter=on_clutter)
        rows = []
        for o in objs:
            if o["clutter"] or not o["rendered"]:
                continue
            g = (ids == o["i"] + 1).astype(np.uint8)
            vis = int(g.sum())
            if vis < 30:
                continue
            pred = next((p for p in meta["preds"] if p["gt_i"] == o["i"]), None)
            r = {"kind": "obj", "set": meta["set"], "task": meta["task"], "episode": meta["episode"], "label": o["label"],
                 "key": o["key"], "vis_px": vis, "has_clutter": has_clutter,
                 "b1_cov": float((fg & g).sum()) / vis, "b1_blob": best_blob_iou(fg, g),
                 "b1_found": pred is not None and pred["uv"] is not None,
                 "b1_pred_far_mm": None,
                 "text": max([iou(m.mask, g) for m in tm], default=0.0)}
            if pred is not None and pred.get("pos") is not None and o.get("pos") is not None:
                r["b1_pred_far_mm"] = 1000 * float(np.linalg.norm(np.asarray(pred["pos"][:2]) - np.asarray(o["pos"][:2])))
            pts = {"gt": interior_point(g)}
            if r["b1_found"]:
                pts["b1"] = pred["uv"]
            for mdl in models:
                t1 = time.time()
                for pn, uv in pts.items():
                    ms = c.points(img, np.array([uv]), model=mdl)
                    if not ms:
                        continue
                    top = max(ms, key=lambda m: m.score)
                    r[f"{mdl}@{pn}_top"] = iou(top.mask, g)
                    r[f"{mdl}@{pn}_best3"] = max(iou(m.mask, g) for m in ms)
                if r["b1_found"] and pred["area"] > 0:
                    # REAL_RSI proposer: centre + 4 offsets, every mask scored by agreement with B1's model silhouette
                    rad = 0.5 * np.sqrt(pred["area"] / np.pi)
                    u, v = pred["uv"]
                    uvs = np.array([[u, v], [u + rad, v], [u - rad, v], [u, v + rad], [u, v - rad]])
                    ms = c.points(img, uvs, model=mdl)
                    psil = _pred_sil(d, pred)
                    if ms and psil is not None:
                        sel = max(ms, key=lambda m: iou(m.mask, psil))
                        r[f"{mdl}@fit"] = iou(sel.mask, g)
                        r["b1_fit"] = iou(psil, g)
                timing[mdl] = timing.get(mdl, 0.0) + time.time() - t1
            rows.append(r)
        frame["timing"] = {k_: round(v, 3) for k_, v in timing.items()}
        out.write(json.dumps(frame) + "\n")
        for r in rows:
            out.write(json.dumps(r) + "\n")
        out.flush()
        if k % 20 == 0:
            print(k, len(dirs), meta["task"], meta["episode"], round(time.time() - t0), "s", flush=True)
    print("server", c.info())


_PSIL: dict = {}


def _pred_sil(d, pred):
    """B1's predicted silhouette in the head view (rendered from its pose; cached per dump dir)."""
    from dojo.perception.cameras import cameras_from_state, pose_to_T
    from dojo.perception import d0
    from dojo.perception.render import load_asset, silhouette, transform

    meta = json.loads((d / "meta.json").read_text())
    ep = ROOTS[meta["set"]] / meta["task"] / meta["episode"]
    cam = cameras_from_state(d0.load_obs(ep, 0)["state"])["cam_head"]
    a = load_asset(pred["key"])
    key = (str(d), pred["key"], tuple(pred["pos"]))
    if key not in _PSIL:
        _PSIL[key] = silhouette(cam, transform(a.Vs, pose_to_T(pred["pos"], pred["ori"])), a.Fs)
    return _PSIL[key]


_LABS: dict = {}


def _scored(task, label):
    """Only objects the pose scorer scores (eval_g1.gt_label): labels of the task's label groups or members of a
    numbered series; fixtures with no label group (the camera stand) are left out."""
    if task not in _LABS:
        from dojo.perception.eval_g1 import series_labels
        from dojo.perception.inverse_layout import label_groups

        _LABS[task] = ({lab for g in label_groups(task) for lab in g["labels"]}, series_labels(task))
    from dojo.perception.eval_g1 import gt_label

    labs, series = _LABS[task]
    return gt_label({"label": label}, labs, series) is not None


def table(a):
    rows = [json.loads(x) for x in open(a.out)]
    objs = [r for r in rows if r["kind"] == "obj" and _scored(r["task"], r["label"])]
    frs = [r for r in rows if r["kind"] == "frame"]
    cols = sorted({k for r in objs for k in r if "@" in k or k in ("text", "b1_blob", "b1_fit")},
                  key=lambda k: (k.split("@")[0], k))
    classes = {
        "all": lambda r: True,
        "(a) B1 low fg (cov < 0.5) or missed": lambda r: r["b1_cov"] < 0.5 or not r["b1_found"],
        "(a'') sweep_blocks cubes": lambda r: r["task"] == "sweep_blocks" and r["label"].startswith("cube"),
        "(a') B1 missed": lambda r: not r["b1_found"],
        "(b-ish) B1 found >= 15 mm off": lambda r: r["b1_found"] and (r["b1_pred_far_mm"] or 0) >= 15,
        "(c) deposit_coin": lambda r: r["task"] == "deposit_coin",
        "(d) clutter frames": lambda r: r["has_clutter"],
    }
    print("| class | n | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for name, f in classes.items():
        rs = [r for r in objs if f(r)]
        cells = []
        for k in cols:
            v = [r[k] for r in rs if k in r]
            cells.append(f"{np.mean(v):.2f} / {np.mean(np.array(v) >= 0.5):.2f} ({len(v)})" if v else "-")
        print(f"| {name} | {len(rs)} | " + " | ".join(cells) + " |")
    print("\ncells: mean IoU / recall@0.5 (n)\n")
    by = defaultdict(list)
    for f in frs:
        by["coin" if f["task"] == "deposit_coin" else ("clutter" if f["has_clutter"] else "other")].append(f)
    print("| frames | n | B1 fg px off-object (median / mean) | text px off-object | text masks | on nothing | on clutter |")
    print("|---|---|---|---|---|---|---|")
    for k, fs in sorted(by.items()):
        print(f"| {k} | {len(fs)} | {np.median([f['b1_false_fg_px'] for f in fs]):.0f} / {np.mean([f['b1_false_fg_px'] for f in fs]):.0f} | "
              f"{np.median([f['text_false_px'] for f in fs]):.0f} / {np.mean([f['text_false_px'] for f in fs]):.0f} | "
              f"{sum(f['n_text'] for f in fs)} | {sum(f['text_on_nothing'] for f in fs)} | {sum(f['text_on_clutter'] for f in fs)} |")
    tt = defaultdict(list)
    for f in frs:
        for k, v in (f.get("timing") or {}).items():
            tt[k].append(v)
    print("\nper-frame SAM time (s, all prompts of the frame):", {k: round(float(np.median(v)), 3) for k, v in tt.items()})


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("dump")
    r.add_argument("out")
    r.add_argument("--models", default="sam2.1_small,sam2.1_large,sam3.1")
    r.add_argument("--only", nargs="*", help="dump paths containing one of these substrings")
    t = sub.add_parser("table")
    t.add_argument("out")
    a = ap.parse_args(argv)
    {"run": run, "table": table}[a.cmd](a)


if __name__ == "__main__":
    main()
