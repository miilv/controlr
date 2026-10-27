"""Perception bake-off, identity: learned embeddings vs our shape / colour identity on the SAME head-view crops.

    identity_bake.py ROOT OUT.jsonl [--ids 100000 ...] [--every 3] [--offset 0] [--models dinov2_b ...]
        [--emb-out EMB.npz] [--viz DIR]

ROOT is a recorded frame dir of one task (``<ROOT>/<layout>/frame_W<step>.json`` + ``_cam_*.png``, e.g. the 5.1 belt
frames). Per frame: the belt read's segmentation and proposals (``belt.belt_read``'s own steps), then per head blob:
- our identity: the best hypothesis' key (what ``belt_read`` reports; ``reported`` = its confidence >= MIN_RIDER_CONF)
  and the best loss per proposed key;
- learned identity: the blob crop (masked on grey, or raw) against renders of EVERY candidate asset of the riders'
  group (``embed_id.Gallery``: each place orientation x 8 yaws, aligned to the crop or tight), per model: per-key
  max cosine;
- truth (scorer only): the GT object whose GT-pose silhouette covers the most of the blob (>= 30 %).
Writes one row per blob, and the observed crops' embeddings (``--emb-out``) for gallery (kNN) arms.
GT is read only to label rows.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=["dinov2_s", "dinov2_b", "siglip2_b", "clip_l"])
    ap.add_argument("--n-yaw", type=int, default=8)
    ap.add_argument("--emb-out", type=Path)
    ap.add_argument("--viz", type=Path)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--min-area", type=int, default=40, help="smaller blobs get our identity only (no crop scores)")
    a = ap.parse_args(argv)

    from dojo.perception.belt import MIN_RIDER_CONF, rider_groups
    from dojo.perception.cameras import TABLE_Z, cameras_from_state, pose_to_T
    from dojo.perception.embed_id import Embedder, Gallery, composite, crop, key_scores, masked, square_box
    from dojo.perception.inverse_layout import (EstimatorConfig, _geometry_groups, _placement_prior, _propose,
                                                _views_subset, blobs, label_groups, task_map)
    from dojo.perception.render import load_asset, silhouette, transform
    from dojo.perception.segment import load_backgrounds, segment_views, with_scene

    task = a.root.name
    fd = Path(os.environ["DOJO_BG_FOLDS"]) / "fold0"
    tm = task_map()[task]
    fixture_keys = sorted({k for d in tm["labels"].values() if d.get("type") in ("Dynamic", "Geometry") for k in d["candidates"]})
    bgs = with_scene(load_backgrounds(fd), fd, fixture_keys)
    cal = os.environ.get("DOJO_COLOR_CALIB")
    color_M = json.loads(Path(cal).read_text())["fold0"] if cal else None
    cfg0 = EstimatorConfig(use_wrists=True, **({"color_M": color_M} if color_M is not None else {}))
    cfg = dataclasses.replace(cfg0, max_candidates_per_blob=10, refine_per_blob=6, refine_per_group=6)
    groups = label_groups(task)
    gis = rider_groups(groups, tm["labels"])
    cand = sorted({k for gi in gis for k in groups[gi]["candidates"]})
    print(f"task {task}: rider groups {gis}, {len(cand)} candidates", flush=True)
    z0 = TABLE_Z + 0.003  # the belt top
    gal = Gallery(n_yaw=a.n_yaw, z0=z0, lit=True, color_M=color_M)
    t0 = time.time()
    embs = {m: Embedder(m) for m in a.models}
    print(f"models loaded in {time.time() - t0:.1f} s", flush=True)
    emb_rows: dict[str, list] = {f"{m}/{v}": [] for m in a.models for v in ("masked", "raw")}
    emb_meta: list = []
    a.out.parent.mkdir(parents=True, exist_ok=True)
    open(a.out, "w").close()
    nf = 0

    def save_emb():  # after every layout: a stopped run keeps what it embedded
        if a.emb_out is not None and emb_meta:
            np.savez_compressed(a.emb_out, meta=json.dumps(emb_meta), **{k.replace("/", "__"): np.asarray(v, np.float16)
                                                                        for k, v in emb_rows.items() if v})

    for d in sorted(a.root.iterdir()):
        if not d.is_dir() or not d.name.isdigit() or (a.ids and d.name not in a.ids):
            continue
        frames = sorted(d.glob("frame_W*.json"), key=lambda p: int(p.stem[7:]))[a.offset::a.every]
        for fj in frames:
            if a.max_frames and nf >= a.max_frames:
                break
            if int(fj.stem[7:]) < a.min_step:
                continue
            j = json.loads(fj.read_text())
            tag = j["frame"]
            if not all((d / f"frame_{tag}_{c}.png").exists() for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")):
                continue
            nf += 1
            imgs = {c: cv2.cvtColor(cv2.imread(str(d / f"frame_{tag}_{c}.png")), cv2.COLOR_BGR2RGB)
                    for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
            st = {k: np.asarray(v, float) for k, v in j["obs"]["state"].items()}
            cams = cameras_from_state(st)
            cam = cams["cam_head"]
            t_seg = time.time()
            views = _views_subset(segment_views(imgs, cams, st, bgs, thr=cfg.fg_thr), cfg)
            hb = blobs(views["cam_head"], "cam_head")
            pending = [(gi, groups[gi], _placement_prior(groups[gi])) for gi in gis]
            sbg = {gi: _geometry_groups(g, prior) for gi, g, prior in pending}
            hyps = _propose(hb, pending, sbg, views, cfg, groups) if hb else []
            t_b1 = time.time() - t_seg
            per_blob: dict[int, list] = {}
            for h in hyps:
                per_blob.setdefault(h.blob, []).append(h)
            # truth: GT-pose silhouettes of the visible riders
            gts = []
            for o in j["gt"]["objects"]:
                if o["type"] != "Rigid" or o.get("clutter") or not o.get("pos"):
                    continue
                key = f"Rigid/{o['category']}/{int(o['category_idx']):05d}"
                try:
                    A = load_asset(key)
                except Exception:
                    continue
                T = pose_to_T(o["pos"], o["quat"])
                sil = silhouette(cam, transform(A.Vs, T), A.Fs)
                if sil.sum() > 0:
                    gts.append((key, sil, o["pos"]))
            bg_head = bgs.get("cam_head")
            for bi, b in enumerate(hb):
                ov = sorted(((int((s & b.mask).sum()), k) for k, s, _ in gts), reverse=True)
                gt_key = ov[0][1] if ov and ov[0][0] >= 0.3 * b.area else None
                second = ov[1][0] / b.area if len(ov) > 1 else 0.0
                hs = sorted(per_blob.get(bi, []), key=lambda h: h.loss)
                b1 = None
                b1_losses: dict[str, float] = {}
                for h in hs:
                    b1_losses.setdefault(h.key, round(float(h.loss), 4))
                if hs:
                    conf = float(np.clip(1 - hs[0].per_view.get("sil_loss", hs[0].loss), 0, 1))
                    b1 = {"key": hs[0].key, "conf": round(conf, 3), "reported": conf >= MIN_RIDER_CONF,
                          "xy": [round(float(hs[0].T()[0, 3]), 4), round(float(hs[0].T()[1, 3]), 4)]}
                row = {"layout": d.name, "step": j["step"], "blob": bi, "area": int(b.area), "bbox": [int(v) for v in b.bbox],
                       "gt": gt_key, "gt_overlap": round(ov[0][0] / b.area, 3) if ov else 0.0,
                       "second_overlap": round(second, 3), "b1": b1, "b1_losses": b1_losses, "t_b1_frame": round(t_b1, 2)}
                if gt_key is None or b.area < a.min_area:  # scorer-side skip: unlabelled blobs and specks get no crops
                    with open(a.out, "a") as fh:
                        fh.write(json.dumps(row) + "\n")
                    continue
                t1 = time.time()
                box = square_box(b.bbox)
                obs = crop(imgs["cam_head"], box)
                m = crop((b.mask * 255).astype(np.uint8), box, interp=cv2.INTER_NEAREST) > 127
                obs_masked = masked(obs, m)
                bgc = crop(bg_head, box) if bg_head is not None else None
                g_imgs: dict[str, list] = {"ma_lit": [], "ma_unlit": [], "mt_lit": [], "mt_unlit": [], "ca_lit": [], "ca_unlit": []}
                g_keys: dict[str, list] = {k: [] for k in g_imgs}
                for key in cand:
                    for lit in (True, False):
                        sfx = "lit" if lit else "unlit"
                        for v in gal.views(key, cam, box, b.centroid, lit=lit):
                            g_imgs[f"ma_{sfx}"].append(masked(v["rgb"], v["alpha"]))
                            g_keys[f"ma_{sfx}"].append(key)
                            if bgc is not None:
                                g_imgs[f"ca_{sfx}"].append(composite(v["rgb"], v["alpha"], bgc))
                                g_keys[f"ca_{sfx}"].append(key)
                            if "tight_rgb" in v:
                                g_imgs[f"mt_{sfx}"].append(masked(v["tight_rgb"], v["tight_alpha"]))
                                g_keys[f"mt_{sfx}"].append(key)
                t_render = time.time() - t1
                t2 = time.time()
                scores = {}
                for mn, E in embs.items():
                    fo = E([obs_masked, obs])
                    emb_rows[f"{mn}/masked"].append(fo[0])
                    emb_rows[f"{mn}/raw"].append(fo[1])
                    for gv, ims in g_imgs.items():
                        if not ims:
                            continue
                        fg = E(ims)
                        o = fo[1] if gv.startswith("ca") else fo[0]
                        sim = fg @ o
                        ks = key_scores(sim, g_keys[gv], cand)
                        scores[f"{mn}/{gv}"] = {k: round(v, 4) for k, v in sorted(ks.items(), key=lambda kv: -kv[1])}
                t_emb = time.time() - t2
                emb_meta.append({"layout": d.name, "step": j["step"], "blob": bi, "gt": gt_key})
                row.update({"scores": scores, "n_views": {k: len(v) for k, v in g_imgs.items()},
                            "t_render": round(t_render, 2), "t_embed": round(t_emb, 2)})
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                if a.viz is not None and gt_key is not None:
                    a.viz.mkdir(parents=True, exist_ok=True)
                    tiles = [obs, obs_masked]
                    for k in cand:  # the best lit aligned view of each candidate, in candidate order
                        idx = [i for i, kk in enumerate(g_keys["ma_lit"]) if kk == k]
                        if idx:
                            tiles.append(g_imgs["ma_lit"][idx[0]])
                    cv2.imwrite(str(a.viz / f"{d.name}_{j['step']:04d}_b{bi}.jpg"),
                                cv2.cvtColor(np.concatenate([cv2.resize(t, (112, 112)) for t in tiles], axis=1), cv2.COLOR_RGB2BGR))
                best = {k: next(iter(v)) for k, v in scores.items() if v}
                print(f"{d.name} {j['step']} b{bi} gt={gt_key} b1={b1 and b1['key']} "
                      f"dino_b={best.get('dinov2_b/ma_lit')} t={t_b1:.1f}/{t_render:.1f}/{t_emb:.1f}", flush=True)
        save_emb()
    save_emb()
    print(f"done: {nf} frames in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
