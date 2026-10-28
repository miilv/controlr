"""DEV-ONLY E3 arm 4 (research R3): candidate fusion + multi-view feature-metric selection on B1's frame-0 records.

For each B1 object the learned height check flags (|dh| > --flag-mm on VGGT's fused head height; or every object with
--all), candidate poses come from:
- B1's own pose and its yaw turns;
- swaps: the positions of the other B1 objects of the same label group;
- gaps: raised regions of the learned height map no B1 object explains (``mv3d_unexplained``);
- B1's own head foreground blobs that no B1 object explains.
Every non-B1 candidate is fitted at its seed by B1's render-and-compare (``mv3d._fit_at``, silhouette loss <= --max-loss).
Candidates within --nms-mm of each other merge (3D NMS, the lowest silhouette loss kept). The DINOv2 feature metric
(``featmetric.FeatureMetric``, head + wrists, all candidates on the same patches) picks one, and the object switches to
it when it beats B1's pose by --margin. eval_g1 before / after on the same records (GT read after).

    python scripts/perception/mv/e3_fusion_eval.py --pred ~/dojo-data/pbake-mv/vggt --out DIR [--flag-mm 10] [--all]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

CALIB = {"d0_51": "color_calib_51.json", "d0_ext": "color_calib.json", "d0_coin": "color_calib.json"}


def run_scene(f: Path, fm, b1dump: Path, a) -> dict:
    import cv2

    from dojo.perception import d0, mv3d
    from dojo.perception.cameras import R_to_quat_wxyz, TABLE_Z, cameras_from_state, pose_to_T
    from dojo.perception.eval_g1 import fold_backgrounds
    from dojo.perception.featmetric import yaw_turns
    from dojo.perception.inverse_layout import EstimatorConfig, _views_subset, blobs, label_groups
    from dojo.perception.mv3d_eval import BG_FOLDS, SET_ALIAS, _b1_record, _score, learned_height_full, root_dir, scene_views
    from dojo.perception.render import load_asset, robot_mask
    from dojo.perception.segment import segment_views
    from dojo.perception.stereo import REGION, check_heights, table_region_mask

    root_name, task, epn = f.name.split("__")[:3]
    ep = root_dir(root_name) / task / epn
    os.environ["DOJO_BG_FOLDS"] = os.path.expanduser(f"~/dojo-perception-data/{BG_FOLDS.get(root_name, 'bg_folds')}")
    meta = json.loads((b1dump / {v: k for k, v in SET_ALIAS.items()}.get(root_name, root_name) / task / epn / "meta.json").read_text())
    calib = json.loads((Path(os.path.expanduser("~/dojo-perception-data/catalog")) / CALIB.get(root_name, "color_calib.json")).read_text())
    fm.set_color(calib.get(f"fold{zlib.crc32(epn.encode()) % 2}"))
    t0 = time.time()
    z = dict(np.load(f))
    vs = scene_views(ep, "home")
    h_l, _ = learned_height_full(z, vs, "poly")
    obs = d0.load_obs(ep, 0)
    cams = cameras_from_state(obs["state"])
    head = cams["cam_head"]
    rob = robot_mask(head, obs["state"], dilate_px=4)
    valid = (table_region_mask(head, REGION) > 0) & (rob == 0)
    rec0 = _b1_record(meta)
    rec_chk = copy.deepcopy(rec0)
    check_heights(rec_chk, head, mv3d.height_sweep_like(h_l, valid), key="mv3d", blob_px=60)
    unexpl = (rec_chk.get("_perception") or {}).get("mv3d_unexplained", [])
    bgs = fold_backgrounds(ep)
    views = _views_subset(segment_views(obs["images"], cams, obs["state"], bgs, thr=EstimatorConfig().fg_thr), EstimatorConfig(use_wrists=True))
    fviews = [{"name": v["name"], "img": v["img"], "cam": v["cam"], "robot": robot_mask(v["cam"], v["state"], dilate_px=2)} for v in vs]
    groups = label_groups(task)
    lab2g = {lab: gi for gi, g in enumerate(groups) for lab in g["labels"]}
    entries = [(typ, cat, j, r) for typ in ("Rigid", "Geometry", "Articulation") for cat, lst in (rec0.get(typ) or {}).items()
               for j, r in enumerate(lst)]
    # seeds: every B1 head foreground blob and every raised region of the learned map (explained or not: a swap's
    # right place is another object's blob), each with its measured top height
    def top_at(mask):
        hs = h_l[(mask > 0) & np.isfinite(h_l)]
        return float(np.percentile(hs, 90)) if hs.size >= 10 else None

    def seed(src, mask, uv):
        th = top_at(mask)
        X = head.ray_plane(np.array([[uv[0] + 0.5, uv[1] + 0.5]]), TABLE_Z + (th or 0.01) / 2)[0]
        return (src, float(X[0]), float(X[1]), th) if np.isfinite(X).all() else None

    fg_seeds = [seed("fg", b.mask, b.centroid) for b in blobs(views["cam_head"], "cam_head") if b.area >= a.min_blob]
    raised = (valid & np.isfinite(h_l) & (np.nan_to_num(h_l) > 0.012)).astype(np.uint8)
    n, lab, stt, cen = cv2.connectedComponentsWithStats(raised)
    gap_seeds = [seed("raised", (lab == i).astype(np.uint8), cen[i]) for i in range(1, n) if stt[i, cv2.CC_STAT_AREA] >= a.min_blob]
    fg_seeds = [x for x in fg_seeds if x]
    gap_seeds = [x for x in gap_seeds if x]
    del unexpl
    det_seeds = []  # detector candidates (stream S): (key or category, seed)
    for dt in (a.dets_by_scene or {}).get((root_name, task, epn), []):
        if dt.get("score", 1.0) < a.det_thr:
            continue
        m = np.zeros(h_l.shape, np.uint8)
        if dt.get("poly"):
            cv2.fillPoly(m, [np.round(np.asarray(dt["poly"], float)).astype(np.int32)], 1)
        else:
            x0, y0, x1, y1 = [int(round(v)) for v in dt["bbox_xyxy"]]
            m[max(y0, 0):y1, max(x0, 0):x1] = 1
        ys, xs = np.nonzero(m)
        if not len(xs):
            continue
        sd = seed("det", m, (xs.mean(), ys.mean()))
        if sd:
            det_seeds.append((dt.get("key"), dt.get("category"), sd))
    log = []
    rec1 = copy.deepcopy(rec0)
    taken = []
    for typ, cat, j, r in entries:
        st = (rec_chk[typ][cat][j].get("_perception") or {}).get("mv3d") or {}
        dh = st.get("dh_mm")
        key = f"{typ}/{cat}/{int(r['category_idx']):05d}"
        try:
            asset = load_asset(key)
        except Exception:
            continue
        T0 = pose_to_T(r["default_pos"], r["default_ori"])
        d_b1 = None
        if not a.all:  # gate: the height check flags it, or B1's own pose explains the views poorly in appearance
            flag = dh is not None and abs(dh) > a.flag_mm
            if not flag and a.gate_d < 9:
                d_b1 = float(fm.score(key, [T0], fviews)[0][0])
                flag = np.isfinite(d_b1) and d_b1 > a.gate_d
            if not flag:
                continue
        ctr0 = T0[:3, :3] @ ((asset.V.min(0) + asset.V.max(0)) / 2) + T0[:3, 3]
        cands = [{"src": "b1", "T": T0, "sil": None}]
        if a.yaw:
            cands += [{"src": "b1_yaw", "T": T, "sil": None} for T in yaw_turns(T0, 4, ctr0)[1:]]
        gi = lab2g.get(r.get("label"))
        swap_seeds = [("swap", float(r2["default_pos"][0]), float(r2["default_pos"][1]), None) for _, _, _, r2 in entries
                      if r2 is not r and gi is not None and lab2g.get(r2.get("label")) == gi]
        mine = [sd for k_, c_, sd in det_seeds if k_ == key or (k_ is None and c_ == cat)
                or (gi is not None and k_ in groups[gi]["candidates"])]
        seeds = [sd for sd in mine + swap_seeds + gap_seeds + fg_seeds if sd[0] in a.seed_srcs]
        # seeds closer than nms to each other are one seed (the first kept: swaps, then raised, then foreground)
        uniq = []
        for sd in seeds:
            if np.hypot(sd[1] - ctr0[0], sd[2] - ctr0[1]) < 0.012:
                continue
            if any(np.hypot(sd[1] - u[1], sd[2] - u[2]) < a.nms_mm / 1000 for u in uniq):
                continue
            uniq.append(sd)
        # other B1 objects' silhouettes out of the foreground: but in a swap the right place IS another (wrong) object's
        others = None if a.no_others else mv3d._others_masks(rec0, views, skip=r)
        for src, x, y, th in uniq[:a.max_seeds]:
            fit = mv3d._fit_at([key], x, y, views, others, n_yaw=a.n_yaw, top_h=th)
            if fit is None or fit[0] > a.max_loss:
                continue
            loss, _, sm, fx, fy, fyaw, zo = fit
            cands.append({"src": src, "T": sm.pose(fx, fy, fyaw, zo), "sil": round(float(loss), 3)})
        # 3D NMS on the centres (B1's own candidates are always kept)
        kept = []
        for c in sorted(cands, key=lambda c: (c["src"] not in ("b1", "b1_yaw"), c["sil"] if c["sil"] is not None else 0)):
            cc = c["T"][:3, :3] @ ((asset.V.min(0) + asset.V.max(0)) / 2) + c["T"][:3, 3]
            if c["src"] not in ("b1", "b1_yaw") and any(np.linalg.norm(cc - k["ctr"]) < a.nms_mm / 1000 for k in kept):
                continue
            c["ctr"] = cc
            kept.append(c)
        if len(kept) <= 1:
            continue
        d, _ = (fm.score_each if a.each else fm.score)(key, [c["T"] for c in kept], fviews)
        if not np.isfinite(d[0]):
            continue
        k = int(np.nanargmin(d))
        ent = {"label": r.get("label"), "key": key, "dh_mm": dh, "d_b1_gate": d_b1, "n_cands": len(kept),
               "typ": typ, "cat": cat, "j": j,
               "cands": [{"src": c["src"], "sil": c["sil"], "d": None if not np.isfinite(dd) else round(float(dd), 4),
                          "xy": [round(float(c["ctr"][0]), 4), round(float(c["ctr"][1]), 4)],
                          "pos": [round(float(v), 5) for v in c["T"][:3, 3]],
                          "ori": [round(float(v), 6) for v in R_to_quat_wxyz(c["T"][:3, :3])]} for c, dd in zip(kept, d)]}
        if k != 0 and d[k] < d[0] - a.margin:
            cbest = kept[k]
            if any(np.linalg.norm(cbest["ctr"][:2] - t) < 0.015 for t in taken):
                ent["blocked"] = True
            else:
                e = rec1[typ][cat][j]
                e["default_pos"] = [float(v) for v in cbest["T"][:3, 3]]
                e["default_ori"] = [float(v) for v in R_to_quat_wxyz(cbest["T"][:3, :3])]
                taken.append(cbest["ctr"][:2])
                ent["switched"] = cbest["src"]
        log.append(ent)
    gt = d0.load_gt(ep, 0)  # DEV-ONLY, after the selection
    m = {"set": root_name, "task": task, "episode": epn, "frame": 0}
    c0, p0 = _score(rec0, gt, task, m)
    c1, p1 = _score(rec1, gt, task, m)
    changes = [{"label": kk[0], "before": c0.get(kk), "after": c1.get(kk)} for kk in c0 if c0.get(kk) != c1.get(kk)]
    return {**m, "log": log, "changes": changes, "phantoms": [p0, p1], "s": round(time.time() - t0, 1),
            "seeds": {"gap": len(gap_seeds), "fg": len(fg_seeds)}}


def main(argv=None):
    from collections import Counter

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--b1dump", default="~/dojo-data/pbake-mv/c4/sam/b1/dump")
    ap.add_argument("--flag-mm", type=float, default=10.0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-loss", type=float, default=0.6)
    ap.add_argument("--nms-mm", type=float, default=15.0)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--n-yaw", type=int, default=8)
    ap.add_argument("--gate-d", type=float, default=0.45, help="also refine objects whose own feature distance exceeds this (9 = off)")
    ap.add_argument("--min-blob", type=int, default=60)
    ap.add_argument("--max-seeds", type=int, default=12)
    ap.add_argument("--no-yaw", dest="yaw", action="store_false", help="no yaw turns of B1's own pose as candidates")
    ap.add_argument("--dets", default="", help="detector candidates (stream S jsonl: set, task, episode, dets[key, bbox_xyxy, score, poly])")
    ap.add_argument("--det-thr", type=float, default=0.25)
    ap.add_argument("--seed-srcs", nargs="+", default=["det", "swap", "raised", "fg"])
    ap.add_argument("--no-others", action="store_true", help="fit seeds against the whole foreground (swaps)")
    ap.add_argument("--model", default="dinov2_b")
    ap.add_argument("--union", dest="each", action="store_false", help="all candidates on one shared crop (default: each its own)")
    ap.add_argument("--match", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    from dojo.perception.featmetric import FeatureMetric

    fm = FeatureMetric(a.model)
    a.dets_by_scene = {}
    if a.dets:
        for line in open(os.path.expanduser(a.dets)):
            r = json.loads(line)
            if r.get("train_id_range"):
                continue
            a.dets_by_scene[(r["set"], r["task"], str(r["episode"]))] = r.get("dets", [])
    files = sorted(Path(a.pred).expanduser().glob("pred/*__home.npz"))
    if a.match:
        files = [f for f in files if any(mm in f.name for mm in a.match)]
    if a.limit:
        files = files[:a.limit]
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    tot = Counter()
    fmode = "w"
    if a.resume and (out / "frames.jsonl").exists():
        done = {f"{r['set']}__{r['task']}__{r['episode']}" for r in map(json.loads, open(out / "frames.jsonl")) if "error" not in r}
        files = [f for f in files if "__".join(f.name.split("__")[:3]) not in done]
        fmode = "a"
    with open(out / "frames.jsonl", fmode) as fh:
        for f in files:
            try:
                r = run_scene(f, fm, Path(a.b1dump).expanduser(), a)
            except Exception as e:
                import traceback

                r = {"file": f.name, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1000:]}
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if "error" in r:
                print("ERROR", r["file"], r["error"], r["tb"], flush=True)
                continue
            for c in r["changes"]:
                tot[(c["before"], c["after"])] += 1
            tot["phantoms_before"] += r["phantoms"][0]
            tot["phantoms_after"] += r["phantoms"][1]
            tot["tried"] += len(r["log"])
            tot["switched"] += sum(1 for e in r["log"] if e.get("switched"))
            print(r["task"], r["episode"], r["changes"], [(e["label"], e.get("switched")) for e in r["log"] if e.get("switched")], r["s"], flush=True)
    print(dict(tot))


if __name__ == "__main__":
    main()
