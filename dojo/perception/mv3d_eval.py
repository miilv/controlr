"""DEV-ONLY evaluation of learned multi-view depth (``mv3d``) against our own layout ground truth (D0-style frames).

Two stages, so the GPU holds its lock only for inference:

    # GPU (inside the caller's flock): predictions per scene -> OUT/pred/*.npz
    python -m dojo.perception.mv3d_eval infer --model vggt --weights DIR --root ~/dojo-data/d0_51 ... --views home --out OUT
    # CPU: GT surfaces by ray casting, metrics -> OUT/eval_<tag>/frames.jsonl + summary
    python -m dojo.perception.mv3d_eval eval --out OUT --views home --workers 8

Views: ``home`` = frame 0's head + both wrists (what the runtime has at frame 0); ``x`` = plus the wrist views of
frames 1-3 (D0's scripted looks; objects untouched, so a static scene).

Metrics (head view at the model resolution, the definitions of ``stereo_eval.score`` so the numbers compare with the
plane sweep): per object top height (90th percentile over its visible pixels) vs GT, top-facing surface error,
"detected" (>= 30 % of its raised pixels read > 1 cm), phantom blobs, and B1's colour foreground on the bare table
(rejected = reads on the table plane). Plus: per-pixel depth error per view, the sim3 residual of the model's own
cameras against ours, and a top-down fusion of every view (objects the head cannot see).
GT is read only here (``d0.load_gt``), after the prediction.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .cameras import TABLE_Z, Camera, pose_to_T
from .stereo_eval import ALLOWED_IDS

H_FG = 0.012      # "raised": the plane sweep's h_fg
H_TABLE = 0.005   # "on the table"


SET_ALIAS = {"coin": "d0_coin"}  # eval_g1's set name -> the frame root's directory name


def root_dir(name: str) -> Path:
    """A frame root by its directory name: ~/dojo-data/<name>, or a copy under $DOJO_MV_ROOTS (colon-separated)."""
    for base in [os.path.expanduser("~/dojo-data")] + [p for p in os.environ.get("DOJO_MV_ROOTS", "").split(":") if p]:
        d = Path(base) / name
        if d.is_dir():
            return d
    return Path(os.path.expanduser("~/dojo-data")) / name


def scene_list(roots: list[str], tasks: list[str] | None = None, ids: list[int] | None = None) -> list[Path]:
    from . import d0

    out = []
    for root in roots:
        for ep in d0.episodes(Path(root).expanduser()):
            if tasks and ep.parent.name not in tasks:
                continue
            if not ep.name.isdigit():
                continue
            i = int(ep.name)
            if i not in ALLOWED_IDS or (ids and i not in ids):
                continue
            out.append(ep)
    return out


def scene_key(ep: Path) -> str:
    return f"{ep.parent.parent.name}__{ep.parent.name}__{ep.name}"


def scene_views(ep: Path, views: str = "home") -> list[dict]:
    """[{name, frame, img, cam, state}] in order: head first, then the wrists of frame 0, then (``x``) the wrists of
    frames 1-3 when the scripted move succeeded."""
    from . import d0
    from .cameras import cameras_from_state

    out = []
    frames = [0] + ([1, 2, 3] if views == "x" else [])
    for k in frames:
        if not (ep / f"frame_{k}.json").exists():
            continue
        if k > 0:
            mv = json.loads((ep / f"frame_{k}.json").read_text()).get("move") or {}
            if mv.get("ok") is False:  # harness metadata only: the frame equals the previous one
                continue
        obs = d0.load_obs(ep, k)
        cams = cameras_from_state(obs["state"])
        for name in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            if k > 0 and name == "cam_head":
                continue
            if name in obs["images"] and name in cams:
                out.append({"name": name, "frame": k, "img": obs["images"][name], "cam": cams[name], "state": obs["state"]})
    return out


# --------------------------------------------------------------------------------------------
# Stage 1: inference (GPU)
# --------------------------------------------------------------------------------------------

def infer_main(a):
    from . import mv3d

    out = Path(a.out).expanduser()
    (out / "pred").mkdir(parents=True, exist_ok=True)
    eps = scene_list(a.root, a.tasks, a.ids)
    if a.limit:
        eps = eps[:a.limit]
    t0 = time.time()
    if a.model == "vggt":
        run = mv3d.VGGTRunner(a.weights, bf16_trunk=a.bf16_trunk)
    elif a.model == "mast3r":
        run = mv3d.Mast3rRunner(a.weights)
    elif a.model == "mapanything":
        run = mv3d.MapAnythingRunner(a.weights, use_poses=not a.no_poses)
    else:
        raise SystemExit(f"unknown model {a.model}")
    print(f"loaded {a.model} in {time.time() - t0:.1f}s; {len(eps)} scenes", flush=True)
    log = open(out / f"infer_{a.views}.jsonl", "a")
    for ep in eps:
        f = out / "pred" / f"{scene_key(ep)}__{a.views}.npz"
        if f.exists() and not a.force:
            continue
        vs = scene_views(ep, a.views)
        if a.model == "mast3r":
            _infer_mast3r(run, vs, f, a.subsample)
            r = {"timings": run.last_timings}
        else:
            r = run.infer([v["img"] for v in vs], [v["cam"] for v in vs]) if a.model == "mapanything" else run.infer([v["img"] for v in vs])
            np.savez_compressed(f, depth=r["depth"].astype(np.float16), conf=r["conf"].astype(np.float16),
                                T_cw=r["T_cw"], K=r["K"], names=np.array([v["name"] for v in vs]),
                                frames=np.array([v["frame"] for v in vs]))
        row = {"scene": scene_key(ep), "views": a.views, **r["timings"]}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(row, flush=True)
    print(f"done in {time.time() - t0:.1f}s", flush=True)


def _infer_mast3r(run, vs: list[dict], f: Path, subsample: int = 4):
    """Pairs: head <-> every wrist both ways (dense depth for each view: the head's from its wrist partners, a
    wrist's from the head) + every wrist-wrist pair once (matches). Saves the dense depths in the VGGT layout
    (T_cw = OUR cameras, so the eval's "sim3" scale mode = MASt3R's raw metric depth) and every pair's matches."""
    from . import mv3d

    n = len(vs)
    pairs = [(0, j) for j in range(1, n)] + [(j, 0) for j in range(1, n)] + [(i, j) for i in range(1, n) for j in range(i + 1, n)]
    r = run.infer([v["img"] for v in vs], pairs, subsample=subsample)
    h, w = next(iter(r["depth"].values())).shape
    depth = np.zeros((n, h, w), np.float32)
    conf = np.zeros((n, h, w), np.float32)
    for (i, j), d in r["depth"].items():
        if i == 0 or j == 0:  # head <-> wrist pairs only: a view's depth as seen with the head as partner
            c = r["conf"][(i, j)]
            depth[i] += d * c
            conf[i] += c
    depth = depth / np.maximum(conf, 1e-6)
    cnt = np.array([sum(1 for (i, j) in r["depth"] if (i == k and (i == 0 or j == 0))) for k in range(n)], np.float32)
    conf = conf / np.maximum(cnt, 1)[:, None, None]
    T_cw = np.stack([np.linalg.inv(mv3d.model_camera(v["cam"], w).T_wc) for v in vs])
    mi, mj, xy1, xy2, mc = [], [], [], [], []
    for (i, j), (a, b, c) in r["matches"].items():
        if j == 0 and i > 0:  # (wrist, head) duplicates (head, wrist)
            continue
        mi.append(np.full(len(a), i, np.int8))
        mj.append(np.full(len(a), j, np.int8))
        xy1.append(a)
        xy2.append(b)
        mc.append(c)
    np.savez_compressed(f, depth=depth.astype(np.float16), conf=conf.astype(np.float16), T_cw=T_cw, K=np.zeros((n, 3, 3)),
                        names=np.array([v["name"] for v in vs]), frames=np.array([v["frame"] for v in vs]),
                        m_i=np.concatenate(mi), m_j=np.concatenate(mj), m_xy1=np.concatenate(xy1), m_xy2=np.concatenate(xy2),
                        m_conf=np.concatenate(mc))
    run.last_timings = r["timings"]


# --------------------------------------------------------------------------------------------
# Stage 2: GT + metrics (CPU)
# --------------------------------------------------------------------------------------------

def gt_surface(cam: Camera, objects: list[dict], robot: np.ndarray | None = None):
    """GT first surface per pixel: (camera-z depth, height above TABLE_Z, object index (-1 table, -2 robot), |n_z|,
    keys without a mesh). Off the table (and nothing hit): NaN."""
    import trimesh

    from .render import load_asset, silhouette, transform
    from .segment import TABLE_CENTER, TABLE_HALF

    h, w = cam.h, cam.w
    vv, uu = np.mgrid[0:h, 0:w]
    uv = np.c_[uu.ravel() + 0.5, vv.ravel() + 0.5]
    X = cam.ray_plane(uv, TABLE_Z)
    on = np.isfinite(X).all(1) & (np.abs(X[:, 0] - TABLE_CENTER[0]) < TABLE_HALF[0]) & (np.abs(X[:, 1] - TABLE_CENTER[1]) < TABLE_HALF[1])
    Rt = cam.T_wc[:3, :3]
    zc = lambda P: (P - cam.center) @ Rt[:, 2]  # noqa: E731
    depth = np.where(on, zc(np.nan_to_num(X)), np.inf)
    hgt = np.where(on, 0.0, np.nan)
    oid = np.full(h * w, -1, np.int32)
    nz = np.where(on, 1.0, np.nan)
    missing = []
    dirs = cam.rays(uv)
    for i, o in enumerate(objects):
        if o.get("pos") is None:
            continue
        key = f"{o['type']}/{o['category']}/{int(o['category_idx']):05d}"
        try:
            a = load_asset(key)
        except Exception:
            missing.append(key)
            continue
        Vw = transform(a.Vs, pose_to_T(o["pos"], o["quat"]))
        m = silhouette(cam, Vw, a.Fs).ravel() > 0
        if not m.any():
            continue
        m = cv2.dilate(m.reshape(h, w).astype(np.uint8), np.ones((3, 3), np.uint8)).ravel() > 0
        idx = np.flatnonzero(m)
        mesh = trimesh.Trimesh(Vw, a.Fs, process=False)
        loc, ir, it = mesh.ray.intersects_location(np.repeat(cam.center[None], len(idx), 0), dirs[idx], multiple_hits=False)
        if len(ir) == 0:
            continue
        d = zc(loc)
        pix = idx[ir]
        closer = d < depth[pix]
        pix, d, loc, it = pix[closer], d[closer], loc[closer], it[closer]
        nz[pix] = np.abs(mesh.face_normals[it][:, 2])
        depth[pix] = d
        hgt[pix] = loc[:, 2] - TABLE_Z
        oid[pix] = i
    depth = np.where(np.isfinite(depth), depth, np.nan).reshape(h, w)
    hgt, oid, nz = hgt.reshape(h, w), oid.reshape(h, w), nz.reshape(h, w)
    if robot is not None:
        r = robot > 0
        depth[r] = np.nan
        hgt[r] = np.nan
        oid[r] = -2
    return depth.astype(np.float32), hgt.astype(np.float32), oid, nz.astype(np.float32), missing


def gt_topdown(objects: list[dict], td, n_samples: int = 40000):
    """GT top-down max height per cell of ``td``'s grid (table = 0) and the owning object index (-1 table)."""
    import trimesh

    from .render import load_asset, transform

    ny, nx = td.height.shape
    hm = np.zeros(ny * nx)
    own = np.full(ny * nx, -1, np.int32)
    for i, o in enumerate(objects):
        if o.get("pos") is None:
            continue
        key = f"{o['type']}/{o['category']}/{int(o['category_idx']):05d}"
        try:
            a = load_asset(key)
        except Exception:
            continue
        mesh = trimesh.Trimesh(transform(a.Vs, pose_to_T(o["pos"], o["quat"])), a.Fs, process=False)
        P, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=0)
        P = np.r_[P, mesh.vertices]
        ix, iy = td.cell_of(P[:, :2])
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        flat = iy[ok] * nx + ix[ok]
        hz = P[ok, 2] - TABLE_Z
        cur = np.full(ny * nx, -np.inf)
        np.maximum.at(cur, flat, hz)
        win = cur > hm
        hm[win] = cur[win]
        own[win] = i
    return hm.reshape(ny, nx), own.reshape(ny, nx)


def _pred_points(npz: dict, vs: list[dict], scale_mode: str, conf_q: float, robs: list | None = None):
    """Per view: model-resolution camera (ours), metric depth, world points, usable mask; plus the scale info."""
    from . import mv3d
    from .render import robot_mask

    depth = npz["depth"].astype(np.float32)
    conf = npz["conf"].astype(np.float32)
    S, h, w = depth.shape
    cams = [mv3d.model_camera(v["cam"], w) for v in vs]
    s3, _, _, res = mv3d.align_sim3(npz["T_cw"], [c.T_wc for c in cams])
    out = []
    for i, (v, cam) in enumerate(zip(vs, cams)):
        rob = robs[i] if robs is not None else robot_mask(cam, v["state"], dilate_px=2)
        valid = (rob == 0) & np.isfinite(depth[i]) & (depth[i] > 0)
        if scale_mode == "table":
            s, share = mv3d.depth_scale_table(depth[i], cam, valid.astype(np.uint8))
            if not np.isfinite(s):
                s, share = s3, 0.0
            field = s
        elif scale_mode == "poly":
            field, share = mv3d.depth_correct_table(depth[i], cam, valid.astype(np.uint8))
            if not np.isfinite(field).all():
                field, share = s3, 0.0
            s = float(np.median(field))
        else:
            s, share, field = s3, None, s3
        D = depth[i] * field
        thr = np.quantile(conf[i][valid], conf_q) if conf_q > 0 and valid.any() else -np.inf
        use = valid & (conf[i] >= thr)
        out.append({"cam": cam, "depth": D, "points": mv3d.unproject(D, cam), "valid": valid, "use": use, "rob": rob,
                    "scale": s, "scale_share": share, "name": v["name"], "frame": v["frame"], "conf": conf[i]})
    return out, s3, res


def eval_scene(f: Path, views: str, scale_modes=("sim3", "table", "poly"), conf_q: float = 0.0, fuse_cell: float = 0.004,
               viz_dir: Path | None = None) -> dict:
    from . import d0, mv3d
    from .stereo import table_region_mask
    from .stereo_eval import head_bg_fg, score

    root_name, task, epn = f.name.split("__")[:3]
    ep = root_dir(root_name) / task / epn
    z = dict(np.load(f))
    vs = scene_views(ep, views)
    names = [str(n) for n in z["names"]]
    assert names == [v["name"] for v in vs], (names, [v["name"] for v in vs])
    gt = d0.load_gt(ep, 0)  # DEV-ONLY, after the prediction
    objs = gt["objects"]
    stand_idx = [i for i, o in enumerate(objs) if o.get("category") == "camera_stand"]
    row = {"set": root_name, "task": task, "episode": epn, "views": views, "n_views": len(vs)}
    fg = head_bg_fg(ep, vs[0]["img"], vs[0]["state"])
    fg_full = fg
    from .render import robot_mask

    w = z["depth"].shape[2]
    mcams = [mv3d.model_camera(v["cam"], w) for v in vs]
    robs = [robot_mask(c, v["state"], dilate_px=2) for c, v in zip(mcams, vs)]
    gt_cache = [gt_surface(c, objs, r) for c, r in zip(mcams, robs)]  # GT per view, shared by every scale mode
    for mode in scale_modes:
        fg = fg_full
        P, s3, res = _pred_points(z, vs, mode, conf_q, robs)
        row["sim3_res"] = res
        row["sim3_scale"] = round(s3, 5)
        pm = {}
        # --- per view depth error (model resolution)
        per_view = []
        gts = []
        for p, gtv in zip(P, gt_cache):
            gd, gh, gi, gn, missing = gtv
            gts.append((gd, gh, gi, gn))
            ok = p["use"] & np.isfinite(gd) & (table_region_mask(p["cam"]) > 0)
            isobj = (gi >= 0) & ~np.isin(gi, stand_idx)
            e = np.abs(p["depth"] - gd)[ok]
            eo = np.abs(p["depth"] - gd)[ok & isobj]
            ph = p["points"][..., 2] - TABLE_Z
            eh = np.abs(ph - gh)[ok & isobj]
            per_view.append({"name": p["name"], "frame": int(p["frame"]), "scale": round(float(p["scale"]), 5),
                             "scale_share": p["scale_share"],
                             "d_med_mm": round(float(np.median(e)) * 1000, 1) if e.size else None,
                             "d_p90_mm": round(float(np.percentile(e, 90)) * 1000, 1) if e.size else None,
                             "dobj_med_mm": round(float(np.median(eo)) * 1000, 1) if eo.size else None,
                             "dobj_p90_mm": round(float(np.percentile(eo, 90)) * 1000, 1) if eo.size else None,
                             "hobj_med_mm": round(float(np.median(eh)) * 1000, 1) if eh.size else None,
                             "absrel": round(float(np.median(e / gd[ok])), 4) if e.size else None})
        pm["per_view"] = per_view
        # --- head view, the plane sweep's scorer
        hp, (gd, gh, gi, gn) = P[0], gts[0]
        cam = hp["cam"]
        hh_self = hp["points"][..., 2] - TABLE_Z
        reg = table_region_mask(cam) > 0
        if fg is not None:
            fg = cv2.resize(fg, (cam.w, cam.h), interpolation=cv2.INTER_NEAREST)
        small = cv2.resize(vs[0]["img"], (cam.w, cam.h), interpolation=cv2.INTER_AREA)
        for tag, hmap in (("self", hh_self), ("fused", mv3d.height_in(cam, [p["points"] for p in P], [p["use"] for p in P]))):
            known = hp["use"] & reg & np.isfinite(hmap) if tag == "self" else reg & np.isfinite(hmap) & (hp["rob"] == 0)
            height = np.where(known, hmap, np.nan).astype(np.float32)
            sw = SimpleNamespace(height=height, known=known.astype(np.uint8), valid=(reg & (hp["rob"] == 0)).astype(np.uint8),
                                 proud=(known & (hmap > H_FG)).astype(np.uint8),
                                 on_table=(known & (np.abs(hmap) < H_TABLE)).astype(np.uint8))
            gh_s = np.where(gi == -2, np.nan, gh)
            sc = score(sw, gh_s, gi, objs, bg_fg=fg, img=small, gt_nz=gn)
            pm[f"head_{tag}"] = sc
        # --- top-down fusion of all views
        td = mv3d.fuse_topdown([p["points"] for p in P], [p["use"] for p in P], cell=fuse_cell)
        g_hm, g_own = gt_topdown(objs, td)
        pm["topdown"] = _score_topdown(td, g_hm, g_own, objs)
        if viz_dir is not None:
            _viz(viz_dir / f"{f.stem}_{mode}.png", vs, P, gts, td, g_hm)
        row[mode] = pm
    if "m_xy1" in z:
        row["tri"] = _eval_tri(z, vs, objs, fg_full, ep, fuse_cell, viz_dir / f"{f.stem}_tri.png" if viz_dir else None,
                               robs=robs, head_gt=gt_cache[0])
    return row


def _tri_points(z: dict, vs: list[dict], err_px: float = 1.0, min_ang: float = 2.0, conf_min: float = 0.0, robs=None):
    """Triangulate every pair's matches with OUR cameras; keep reprojection <= ``err_px`` (model px), ray angle >=
    ``min_ang`` and pixels off the robot in both views. Returns (points (N, 3), first view index per point, stats)."""
    from . import mv3d
    from .render import robot_mask

    w = z["depth"].shape[2]
    cams = [mv3d.model_camera(v["cam"], w) for v in vs]
    robs = robs if robs is not None else [robot_mask(c, v["state"], dilate_px=2) for c, v in zip(cams, vs)]
    mi, mj = z["m_i"].astype(int), z["m_j"].astype(int)
    xy1, xy2 = z["m_xy1"].astype(np.int64), z["m_xy2"].astype(np.int64)
    mc = z["m_conf"].astype(np.float32)
    Xs, src = [], []
    n_all = 0
    for i, j in sorted(set(zip(mi.tolist(), mj.tolist()))):
        sel = (mi == i) & (mj == j) & (mc >= conf_min)
        a, b = xy1[sel], xy2[sel]
        n_all += len(a)
        X, err, ang = mv3d.triangulate(cams[i], cams[j], a + 0.5, b + 0.5)
        ok = (err <= err_px) & (ang >= min_ang) & (robs[i][a[:, 1], a[:, 0]] == 0) & (robs[j][b[:, 1], b[:, 0]] == 0)
        Xs.append(X[ok])
        src.append(np.full(int(ok.sum()), i, np.int8))
    X = np.concatenate(Xs) if Xs else np.zeros((0, 3))
    return X, np.concatenate(src) if src else np.zeros(0, np.int8), {"matches": n_all, "kept": int(len(X))}, cams, robs


def _eval_tri(z, vs, objs, fg, ep, fuse_cell, viz_path=None, robs=None, head_gt=None) -> dict:
    from . import mv3d
    from .stereo import table_region_mask
    from .stereo_eval import score

    X, src, st, cams, robs = _tri_points(z, vs, robs=robs)
    cam, rob = cams[0], robs[0]
    gd, gh, gi, gn, _ = head_gt if head_gt is not None else gt_surface(cam, objs, rob)
    reg = table_region_mask(cam) > 0
    small = cv2.resize(vs[0]["img"], (cam.w, cam.h), interpolation=cv2.INTER_AREA)
    fgs = cv2.resize(fg, (cam.w, cam.h), interpolation=cv2.INTER_NEAREST) if fg is not None else None
    out = {"per_view": [], "tri_stats": st}
    for tag, P in (("self", X[src == 0]), ("fused", X)):
        hmap = mv3d.height_in(cam, [P.reshape(-1, 1, 3)])
        known = reg & np.isfinite(hmap) & (rob == 0)
        sw = SimpleNamespace(height=np.where(known, hmap, np.nan).astype(np.float32), known=known.astype(np.uint8),
                             valid=(reg & (rob == 0)).astype(np.uint8), proud=(known & (hmap > H_FG)).astype(np.uint8),
                             on_table=(known & (np.abs(hmap) < H_TABLE)).astype(np.uint8))
        out[f"head_{tag}"] = score(sw, np.where(gi == -2, np.nan, gh), gi, objs, bg_fg=fgs, img=small, gt_nz=gn)
    td = mv3d.fuse_topdown([X.reshape(-1, 1, 3)], cell=fuse_cell)
    g_hm, g_own = gt_topdown(objs, td)
    out["topdown"] = _score_topdown(td, g_hm, g_own, objs)
    return out


def _score_topdown(td, g_hm: np.ndarray, g_own: np.ndarray, objs: list[dict], min_cells: int = 4) -> dict:
    ph = td.height
    have = np.isfinite(ph)
    rows = []
    for i, o in enumerate(objs):
        m = g_own == i
        n = int(m.sum())
        if n < min_cells:
            continue
        gtop = float(np.percentile(g_hm[m], 90))
        cov = m & have
        r = {"label": o.get("label"), "category": o.get("category"), "gt_idx": i, "cells": n,
             "cov": round(float(cov.sum() / n), 3), "gt_top_mm": round(gtop * 1000, 1)}
        if cov.sum() >= min_cells:
            d = ph[cov] - g_hm[cov]
            r["top_mm"] = round(float(np.percentile(ph[cov], 90)) * 1000, 1)
            r["top_err_mm"] = round((float(np.percentile(ph[cov], 90)) - float(np.percentile(g_hm[cov], 90))) * 1000, 1)
            r["cell_med_mm"] = round(float(np.median(d)) * 1000, 1)
            r["cell_absmed_mm"] = round(float(np.median(np.abs(d))) * 1000, 1)
            if gtop > 0.015:
                raised = cov & (g_hm > 0.01)
                if raised.sum() >= min_cells:
                    r["found"] = bool((ph[raised] > np.minimum(0.5 * g_hm[raised], 0.01)).mean() >= 0.3)
        rows.append(r)
    # phantoms: raised cells far from every GT object
    near = cv2.dilate((g_own >= 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    bad = have & (ph > 0.015) & ~near
    n, lab, st, _ = cv2.connectedComponentsWithStats(bad.astype(np.uint8))
    ph_blobs = [int(st[j, cv2.CC_STAT_AREA]) for j in range(1, n) if st[j, cv2.CC_STAT_AREA] >= 6]
    tab = have & (g_own < 0) & ~near
    return {"objects": rows, "phantom_blobs": len(ph_blobs), "phantom_cells": int(sum(ph_blobs)),
            "table_cells": int(tab.sum()), "table_absmed_mm": round(float(np.median(np.abs(ph[tab]))) * 1000, 2) if tab.any() else None,
            "table_p90_mm": round(float(np.percentile(np.abs(ph[tab]), 90)) * 1000, 2) if tab.any() else None,
            "coverage": round(float((have & (g_own >= -1)).mean()), 3)}


def _viz(path: Path, vs, P, gts, td, g_hm):
    def cmap(hm, vmax=0.15):
        v = np.clip(np.nan_to_num(hm, nan=-1) / vmax, 0, 1)
        c = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        c[~np.isfinite(hm)] = 40
        return c

    rows = []
    for v, p, (gd, gh, gi, gn) in list(zip(vs, P, gts))[:3]:
        img = cv2.resize(v["img"], (p["cam"].w, p["cam"].h))[..., ::-1]
        ph = np.where(p["valid"], p["points"][..., 2] - TABLE_Z, np.nan)
        err = np.abs(ph - gh)
        e = cv2.applyColorMap((np.clip(np.nan_to_num(err, nan=0) / 0.02, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        rows.append(np.concatenate([img, cmap(ph), cmap(gh), e], 1))
    top = np.concatenate(rows, 0)
    a = cv2.resize(cmap(td.height[::-1]), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(cmap(np.where(g_hm > -1, g_hm, np.nan)[::-1]), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    bot = np.concatenate([a, b], 1)
    W = top.shape[1]
    bot = cv2.resize(bot, (W, int(bot.shape[0] * W / bot.shape[1])))
    cv2.imwrite(str(path), np.concatenate([top, bot], 0))


def _job(args):
    cv2.setNumThreads(1)
    f, views, kw = args
    try:
        return eval_scene(Path(f), views, **kw)
    except Exception as e:  # keep going
        import traceback

        return {"file": Path(f).name, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}


def eval_main(a):
    from concurrent.futures import ProcessPoolExecutor

    os.environ.setdefault("DOJO_GL", "0")
    out = Path(a.out).expanduser()
    files = sorted((out / "pred").glob(f"*__{a.views}.npz"))
    if a.limit:
        files = files[:a.limit]
    tag = a.tag or f"{a.views}_q{a.conf_q}"
    od = out / f"eval_{tag}"
    od.mkdir(exist_ok=True)
    vd = od / "viz"
    vd.mkdir(exist_ok=True)
    kw = {"conf_q": a.conf_q, "scale_modes": tuple(a.modes.split(","))}
    jobs = [(str(f), a.views, {**kw, "viz_dir": vd if i < a.viz else None}) for i, f in enumerate(files)]
    with open(od / "frames.jsonl", "w") as fh, ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(_job, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if "error" in r:
                print("ERROR", r["file"], r["error"], r.get("tb"), flush=True)
            else:
                print(r["task"], r["episode"], r["set"], json.dumps(r["sim3_res"])[:120], flush=True)


# --------------------------------------------------------------------------------------------
# The height check of B1's own records: learned depth vs the plane sweep (paired, same frames)
# --------------------------------------------------------------------------------------------

def _b1_record(meta: dict) -> dict:
    """A layout record from the B1 dump's predictions (key, pos, ori, label)."""
    rec: dict = {}
    for p in meta.get("preds", []):
        typ, cat, idx = p["key"].split("/")
        rec.setdefault(typ, {}).setdefault(cat, []).append(
            {"label": p.get("label"), "category": cat, "category_idx": int(idx), "default_pos": list(p["pos"]),
             "default_ori": list(p["ori"]), "_gt_i": p.get("gt_i")})
    return rec


def learned_height_full(z: dict, vs: list[dict], mode: str = "poly"):
    """The learned fused height map of the head view at FULL resolution (nearest upsampling of the model-resolution
    map) + the head's robot mask there."""
    from . import mv3d

    P, _, _ = _pred_points(z, vs, mode, 0.0)
    cam_m = P[0]["cam"]
    hm = mv3d.height_in(cam_m, [p["points"] for p in P], [p["use"] for p in P])
    hm = np.where(P[0]["rob"] == 0, hm, np.nan).astype(np.float32)
    head = vs[0]["cam"]
    return cv2.resize(hm, (head.w, head.h), interpolation=cv2.INTER_NEAREST), P


def height_sources(h_learned: np.ndarray, sweep, gate_mm: float = 10.0) -> dict:
    """Height maps (full-res head) to check against: the plane sweep, the learned map, and two hybrids: the sweep
    wherever it measured a height (``hyb_any``), or only where it agrees with the learned map within ``gate_mm``
    (``hyb_gate``: the sweep refines, the learned map vetoes its vertical-face / repeated-texture mistakes)."""
    hs = np.where(sweep.known > 0, sweep.height, np.nan).astype(np.float32)
    hl = h_learned
    any_ = np.where(np.isfinite(hs), hs, hl)
    with np.errstate(invalid="ignore"):
        agree = np.isfinite(hs) & np.isfinite(hl) & (np.abs(hs - hl) < gate_mm / 1000)
    gate = np.where(agree, hs, hl)
    return {"sweep": None, "mv3d": hl, "hyb_any": any_, "hyb_gate": gate}


def _as_sweep(h: np.ndarray, valid: np.ndarray) -> SimpleNamespace:
    known = valid & np.isfinite(h)
    with np.errstate(invalid="ignore"):
        return SimpleNamespace(height=np.where(known, h, np.nan).astype(np.float32), known=known.astype(np.uint8),
                               proud=(known & (h > H_FG)).astype(np.uint8), on_table=(known & (np.abs(h) < H_TABLE)).astype(np.uint8))


def check_scene(f: Path, views: str, b1dump: Path, mode: str = "poly", sweep: bool = True, dh_mm: float = 4.0) -> dict:
    """Per B1 object (the dump's frame-0 record): truth = median(GT - B1's predicted surface) over its head pixels;
    each source's ``stereo.check_heights`` dh: the plane sweep (exactly ``stereo.check_record``'s sweep), the learned
    fused map, and the hybrids (``height_sources``). Full-resolution head camera for all."""
    import copy

    from . import d0
    from .render import load_asset, robot_mask
    from .stereo import REGION, View, check_heights, object_heights, plane_sweep, table_region_mask

    root_name, task, epn = f.name.split("__")[:3]
    ep = root_dir(root_name) / task / epn
    mf = b1dump / {v: k for k, v in SET_ALIAS.items()}.get(root_name, root_name) / task / epn / "meta.json"
    if not mf.exists():
        return {"file": f.name, "error": "no B1 dump"}
    meta = json.loads(mf.read_text())
    z = dict(np.load(f))
    vs = scene_views(ep, views)
    t0 = time.time()
    h_l, P = learned_height_full(z, vs, mode)
    t_l = time.time() - t0
    cam = vs[0]["cam"]
    rob = robot_mask(cam, vs[0]["state"], dilate_px=4)
    valid = (table_region_mask(cam, REGION) > 0) & (rob == 0)
    timings = {"learned_s": round(t_l, 2)}
    srcs = {}
    if sweep:  # check_record's sweep, on frame 0's own three views
        obs = d0.load_obs(ep, 0)
        from .cameras import cameras_from_state

        cams = cameras_from_state(obs["state"])
        vv = {n: View(im, cams[n], (1 - (robot_mask(cams[n], obs["state"], dilate_px=4) > 0)).astype(np.uint8))
              for n, im in obs["images"].items() if n in cams}
        ref = vv.pop("cam_head")
        ref.valid = ref.valid & table_region_mask(ref.cam, REGION)
        t0 = time.time()
        sw = plane_sweep(ref, list(vv.values()), agg=5, fill=True, min_blob=30, lr=True, scale=1.0)
        timings["sweep_s"] = round(time.time() - t0, 2)
        hsrc = height_sources(h_l, sw)
        srcs["sweep"] = sw
        for k in ("mv3d", "hyb_any", "hyb_gate"):
            srcs[k] = _as_sweep(hsrc[k], valid)
    else:
        srcs["mv3d"] = _as_sweep(h_l, valid)
    rec0 = _b1_record(meta)
    recs, summ = {}, {}
    for k, sw_k in srcs.items():
        recs[k] = copy.deepcopy(rec0)
        summ[k] = check_heights(recs[k], cam, sw_k, dh_mm=dh_mm, key="hcheck")
    gt = d0.load_gt(ep, 0)  # DEV-ONLY, after every check
    gd, gh, gi, gn, _ = gt_surface(cam, gt["objects"], rob)
    ents, items = [], []
    for typ in ("Rigid", "Geometry", "Articulation"):
        for cat, lst in (rec0.get(typ) or {}).items():
            for j, r in enumerate(lst):
                k = f"{typ}/{cat}/{int(r['category_idx']):05d}"
                try:
                    a = load_asset(k)
                except Exception:
                    continue
                ents.append((typ, cat, j, r))
                items.append((k, a.Vs, a.Fs, pose_to_T(r["default_pos"], r["default_ori"])))
    ph, pid = object_heights(cam, items)
    rows = []
    for i, (typ, cat, j, r) in enumerate(ents):
        m = (pid == i) & np.isfinite(gh)
        row = {"label": r["label"], "key": f"{typ}/{cat}/{int(r['category_idx']):05d}", "gt_i": r.get("_gt_i"), "px": int(m.sum()),
               "pos": [round(float(x), 4) for x in r["default_pos"]]}
        if m.sum() >= 8:
            row["truth_dh_mm"] = round(float(np.median(gh[m] - ph[m])) * 1000, 1)
        for k in recs:
            pk = (recs[k][typ][cat][j].get("_perception") or {}).get("hcheck") or {}
            row[f"{k}_dh_mm"], row[f"{k}_mismatch"] = pk.get("dh_mm"), pk.get("mismatch")
        rows.append(row)
    unex = {}
    for k in recs:  # raised blobs no B1 object explains: distance to the nearest GT object (a miss) or none (a phantom)
        lst = []
        for u in (recs[k].get("_perception") or {}).get("hcheck_unexplained", []):
            d = [np.hypot(o["pos"][0] - u["xy"][0], o["pos"][1] - u["xy"][1]) for o in gt["objects"] if o.get("pos") is not None
                 and o.get("category") != "camera_stand"]
            lst.append({**u, "gt_dist_mm": round(float(min(d)) * 1000, 1) if d else None})
        unex[k] = lst
    return {"set": root_name, "task": task, "episode": epn, "objects": rows, "summary": summ, "timings": timings,
            "unexplained": unex}


BG_FOLDS = {"d0_51": "bg_folds_51", "d0_ext": "bg_folds", "d0_coin": "bg_folds", "d0": "bg_folds"}


def _score(rec: dict, gt: dict, task: str, meta: dict) -> tuple[dict, int]:
    """eval_g1's per-object classes ((label, gt category_idx, rounded gt pos) -> class) and the phantoms (predictions
    matched to nothing or > 40 mm from every GT object), as the SAM bake-off counts them."""
    from .eval_g1 import TOLERANCE, failure_class, flatten_pred, score_record

    rows, _, _ = score_record(rec, gt, task, meta)
    tol = TOLERANCE.get(task, (None, None, ""))
    cls = {(r["label"], r["gt_idx"], tuple(np.round(r["gt_pos"], 4))): failure_class(r, tol) for r in rows}
    allpos = np.array([o["pos"] for o in gt["objects"] if o.get("pos") is not None])
    ph = 0
    for p in flatten_pred(rec):
        if not len(allpos) or float(np.min(np.linalg.norm(allpos[:, :2] - np.asarray(p["default_pos"])[:2], axis=1))) > 0.04:
            ph += 1
    return cls, ph


def propose_scene(f: Path, views: str, b1dump: Path, mode: str = "poly", source: str = "mv3d", **kw) -> dict:
    """B1's dump record before / after ``mv3d.propose_unexplained`` on the learned (or hybrid) head height map,
    both scored by eval_g1 (DEV-ONLY GT, after the proposal)."""
    import copy

    from . import d0, mv3d
    from .cameras import cameras_from_state
    from .eval_g1 import fold_backgrounds
    from .inverse_layout import EstimatorConfig, _views_subset
    from .render import robot_mask
    from .segment import segment_views
    from .stereo import REGION, table_region_mask

    root_name, task, epn = f.name.split("__")[:3]
    ep = root_dir(root_name) / task / epn
    mf = b1dump / {v: k for k, v in SET_ALIAS.items()}.get(root_name, root_name) / task / epn / "meta.json"
    if not mf.exists():
        return {"file": f.name, "error": "no B1 dump"}
    os.environ["DOJO_BG_FOLDS"] = os.path.expanduser(f"~/dojo-perception-data/{BG_FOLDS.get(root_name, 'bg_folds')}")
    meta = json.loads(mf.read_text())
    z = dict(np.load(f))
    vs = scene_views(ep, views)
    t0 = time.time()
    h_l, _ = learned_height_full(z, vs, mode)
    obs = d0.load_obs(ep, 0)
    cams = cameras_from_state(obs["state"])
    cam = cams["cam_head"]
    rob = robot_mask(cam, obs["state"], dilate_px=4)
    valid = (table_region_mask(cam, REGION) > 0) & (rob == 0)
    if source.startswith("hyb"):
        from .stereo import View, plane_sweep

        vv = {n: View(im, cams[n], (1 - (robot_mask(cams[n], obs["state"], dilate_px=4) > 0)).astype(np.uint8))
              for n, im in obs["images"].items() if n in cams}
        ref = vv.pop("cam_head")
        ref.valid = ref.valid & table_region_mask(ref.cam, REGION)
        sw = plane_sweep(ref, list(vv.values()), agg=5, fill=True, min_blob=30, lr=True, scale=1.0)
        h_l = height_sources(h_l, sw)[source]
    bgs = fold_backgrounds(ep)
    vws = _views_subset(segment_views(obs["images"], cams, obs["state"], bgs, thr=EstimatorConfig().fg_thr), EstimatorConfig(use_wrists=True))
    rec0 = _b1_record(meta)
    rec1 = copy.deepcopy(rec0)
    log = mv3d.propose_unexplained(rec1, vws, h_l, task, valid=valid, **kw)
    dt = time.time() - t0
    gt = d0.load_gt(ep, 0)  # DEV-ONLY, after the proposal
    m = {"set": root_name, "task": task, "episode": epn, "frame": 0}
    c0, p0 = _score(rec0, gt, task, m)
    c1, p1 = _score(rec1, gt, task, m)
    changes = [{"label": k[0], "before": c0.get(k), "after": c1.get(k)} for k in c0 if c0.get(k) != c1.get(k)]
    return {**m, "before": dict(__import__("collections").Counter(c0.values())), "after": dict(__import__("collections").Counter(c1.values())),
            "phantoms_before": p0, "phantoms_after": p1, "changes": changes, "log": log, "s": round(dt, 2)}


def _propose_job(args):
    cv2.setNumThreads(1)
    f, views, kw = args
    try:
        return propose_scene(Path(f), views, **kw)
    except Exception as e:
        import traceback

        return {"file": Path(f).name, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}


def propose_main(a):
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor

    os.environ.setdefault("DOJO_GL", "0")
    out = Path(a.out).expanduser()
    files = sorted((out / "pred").glob(f"*__{a.views}.npz"))
    if a.sets:
        files = [f for f in files if f.name.split("__")[0] in a.sets]
    if a.match:
        files = [f for f in files if any(m in f.name for m in a.match)]
    if a.limit:
        files = files[:a.limit]
    od = out / f"propose_{a.views}_{a.source}{a.tag}"
    od.mkdir(exist_ok=True)
    kw = {"b1dump": Path(a.b1dump).expanduser(), "mode": a.mode, "source": a.source, **json.loads(a.kw)}
    tot = Counter()
    fmode = "w"
    if a.resume and (od / "frames.jsonl").exists():  # keep the scenes already done
        done = set()
        for line in open(od / "frames.jsonl"):
            r = json.loads(line)
            if "error" not in r:
                done.add(f"{r['set']}__{r['task']}__{r['episode']}")
        files = [f for f in files if "__".join(f.name.split("__")[:3]) not in done]
        fmode = "a"
    with open(od / "frames.jsonl", fmode) as fh, ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(_propose_job, [(str(f), a.views, kw) for f in files]):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if "error" in r:
                print("ERROR", r.get("file"), r["error"], r.get("tb"), flush=True)
                continue
            for c in r["changes"]:
                tot[(c["before"], c["after"])] += 1
            tot["phantoms_before"] += r["phantoms_before"]
            tot["phantoms_after"] += r["phantoms_after"]
            print(r["task"], r["episode"], r["changes"], [x.get("accepted") for x in r["log"]], r["s"], flush=True)
    print(dict(tot))


def _check_job(args):
    cv2.setNumThreads(1)
    f, views, kw = args
    try:
        return check_scene(Path(f), views, **kw)
    except Exception as e:
        import traceback

        return {"file": Path(f).name, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}


def check_main(a):
    from concurrent.futures import ProcessPoolExecutor

    os.environ.setdefault("DOJO_GL", "0")
    out = Path(a.out).expanduser()
    files = sorted((out / "pred").glob(f"*__{a.views}.npz"))
    if a.limit:
        files = files[:a.limit]
    od = out / f"check_{a.views}_{a.mode}"
    od.mkdir(exist_ok=True)
    kw = {"b1dump": Path(a.b1dump).expanduser(), "mode": a.mode, "sweep": not a.no_sweep}
    with open(od / "frames.jsonl", "w") as fh, ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(_check_job, [(str(f), a.views, kw) for f in files]):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(r.get("task"), r.get("episode"), r.get("error") or (r["summary"], r["timings"]), flush=True)


def check_summary(rows: list[dict], truth_mm: float = 10.0, thrs=(10.0, 15.0)) -> dict:
    """The height check as a detector of B1 objects whose predicted surface is > ``truth_mm`` off the GT surface
    (the plane sweep report's protocol): flagged, precision, recall, false flags on right objects, per source."""
    rows = [r for r in rows if "error" not in r]
    objs = [o for r in rows for o in r["objects"] if "truth_dh_mm" in o]
    bad = np.array([abs(o["truth_dh_mm"]) > truth_mm for o in objs])
    truth = np.array([o["truth_dh_mm"] for o in objs])
    out = {"frames": len(rows), "objects": len(objs), "truth_bad": int(bad.sum())}
    srcs = [k[:-6] for k in (objs[0] if objs else {}) if k.endswith("_dh_mm") and k != "truth_dh_mm"]
    for src in srcs:
        dh = np.array([o.get(f"{src}_dh_mm") if o.get(f"{src}_dh_mm") is not None else np.nan for o in objs], float)
        have = np.isfinite(dh)
        S = {"measured": int(have.sum()), "measured_bad": int((have & bad).sum())}
        for t in thrs:
            fl = have & (np.abs(np.nan_to_num(dh)) > t)
            S[f"{t:g}mm"] = {"flagged": int(fl.sum()), "precision": f"{int((fl & bad).sum())}/{int(fl.sum())}",
                             "recall": f"{int((fl & bad).sum())}/{int(bad.sum())}",
                             "false_flags_on_right": round(float((fl & ~bad).sum() / max((~bad).sum(), 1)), 3)}
        e = np.abs(dh - truth)[have & ~bad]
        S["dh_err_on_right_med_mm"] = _pct(e, 50)
        S["dh_err_on_right_p90_mm"] = _pct(e, 90)
        un = [u for r in rows for u in (r.get("unexplained") or {}).get(src, [])]
        S["unexplained_per_frame"] = round(len(un) / max(len(rows), 1), 2)
        S["unexplained_near_gt_30mm"] = sum(1 for u in un if (u.get("gt_dist_mm") or 1e9) < 30)
        out[src] = S
    return out


def _pct(a, q):
    return round(float(np.percentile(a, q)), 1) if len(a) else None


def _b1_classes(path: str | None) -> dict:
    """(set, task, episode, index into frame 0's gt["objects"]) -> B1's class at frame 0 (eval_g1 objects.jsonl):
    ok / identity / gross / missed. eval_g1's own ``gt_idx`` uses another ordering, so rows are matched to GT objects by
    their GT position."""
    from . import d0

    if not path or not Path(path).expanduser().exists():
        return {}
    out = {}
    gts: dict = {}
    for line in open(Path(path).expanduser()):
        r = json.loads(line)
        c = "missed" if not r["found"] else "identity" if not r["id_ok"] else "gross" if r["xy_err_mm"] > 15 else "ok"
        key = (SET_ALIAS.get(r["set"], r["set"]), r["task"], str(r["episode"]))
        if key not in gts:
            ep = root_dir(key[0]) / key[1] / key[2]
            gts[key] = [o.get("pos") for o in d0.load_gt(ep, 0)["objects"]] if (ep / "frame_0.json").exists() else []
        d = [np.linalg.norm(np.subtract(p, r["gt_pos"])) if p is not None else np.inf for p in gts[key]]
        if d and min(d) < 1e-3:
            out[key + (int(np.argmin(d)),)] = c
    return out


def _stereo_objs(path: str | None) -> dict:
    """(task, episode, label) -> the plane sweep's object row (stereo_eval frames.jsonl, labelled objects only)."""
    if not path or not Path(path).expanduser().exists():
        return {}
    out = {}
    for line in open(Path(path).expanduser()):
        r = json.loads(line)
        for o in r.get("objects", []):
            if o.get("label"):
                out[(r["task"], str(r["episode"]), o["label"])] = o
    return out


def summarize(rows: list[dict], b1: dict | None = None, stereo: dict | None = None, modes=("sim3", "table", "poly", "tri")) -> dict:
    rows = [r for r in rows if "error" not in r]
    out = {"frames": len(rows)}
    sim = [r["sim3_res"] for r in rows if "sim3_res" in r]
    out["sim3"] = {"pos_rms_mm_med": _pct([s["pos_rms_mm"] for s in sim], 50), "rot_max_deg_med": _pct([s["rot_max_deg"] for s in sim], 50),
                   "umeyama_rms_mm_med": _pct([s.get("umeyama_centres_rms_mm", np.nan) for s in sim], 50)}
    for mode in modes:
        if not rows or mode not in rows[0]:
            continue
        M = {}
        pv = [v for r in rows for v in r[mode]["per_view"]]
        for cam in ("head", "wrist"):
            vv = [v for v in pv if (cam == "head") == (v["name"] == "cam_head")]
            M[f"{cam}_depth_med_mm"] = _pct([v["d_med_mm"] for v in vv if v["d_med_mm"] is not None], 50)
            M[f"{cam}_obj_height_med_mm"] = _pct([v["hobj_med_mm"] for v in vv if v["hobj_med_mm"] is not None], 50)
        for tag in ("head_self", "head_fused"):
            objs, keys = [], []
            for r in rows:
                for o in r[mode][tag]["objects"]:
                    if o.get("category") == "camera_stand":
                        continue
                    objs.append(o)
                    keys.append((r["set"], r["task"], r["episode"], o.get("gt_idx")))
            z = np.array([abs(o["z_err_mm"]) for o in objs if "z_err_mm" in o])
            tz = np.array([abs(o["top_err_mm"]) for o in objs if "top_err_mm" in o])
            tall = [o for o in objs if o.get("tall")]
            sh = [r[mode][tag].get("shadow") or {} for r in rows]
            fp = sum(s.get("bg_fp_px", 0) for s in sh)
            tp = sum(s.get("bg_tp_px", 0) for s in sh)
            T = {"objects": len(objs), "obj_measured": int(len(z)), "z_abs_med_mm": _pct(z, 50), "z_abs_p90_mm": _pct(z, 90),
                 "z_within_5mm": round(float((z < 5).mean()), 3) if len(z) else None,
                 "z_within_10mm": round(float((z < 10).mean()), 3) if len(z) else None,
                 "top_n": int(len(tz)), "top_abs_med_mm": _pct(tz, 50), "top_abs_p90_mm": _pct(tz, 90),
                 "top_within_5mm": round(float((tz < 5).mean()), 3) if len(tz) else None,
                 "tall_detected": f"{sum(bool(o.get('detected')) for o in tall)}/{len(tall)}",
                 "phantom_blobs_per_frame": round(float(np.mean([len(r[mode][tag]["phantoms"]) for r in rows])), 2),
                 "bgFP_rejected": round(sum(s.get("bg_fp_on_table", 0) for s in sh) / max(fp, 1), 3),
                 "bgFP_confirmed": round(sum(s.get("bg_fp_proud", 0) for s in sh) / max(fp, 1), 3),
                 "bgTP_wrongly_rejected": round(sum(s.get("bg_tp_on_table", 0) for s in sh) / max(tp, 1), 3)}
            if b1:
                for cls in ("missed", "gross", "identity"):
                    sel = [o for o, k in zip(objs, keys) if b1.get(k) == cls]
                    st = [o for o in sel if o.get("tall")]
                    zz = np.array([abs(o["z_err_mm"]) for o in sel if "z_err_mm" in o])
                    T[f"b1_{cls}"] = (f"n {len(sel)}, tall detected {sum(bool(o.get('detected')) for o in st)}/{len(st)}, "
                                      f"|z| med {_pct(zz, 50)} p90 {_pct(zz, 90)}")
            if stereo:
                pair = []
                for r in rows:
                    for o in r[mode][tag]["objects"]:
                        so = stereo.get((r["task"], r["episode"], o.get("label")))
                        if so and "z_err_mm" in so and "z_err_mm" in o:
                            pair.append((abs(so["z_err_mm"]), abs(o["z_err_mm"]), abs(so.get("top_err_mm", np.nan)), abs(o.get("top_err_mm", np.nan))))
                if pair:
                    P = np.array(pair)
                    tp_ = P[np.isfinite(P[:, 2]) & np.isfinite(P[:, 3])]
                    T["vs_stereo_z"] = (f"n {len(P)}: stereo |z| med {_pct(P[:, 0], 50)} p90 {_pct(P[:, 0], 90)} vs this "
                                        f"{_pct(P[:, 1], 50)} / {_pct(P[:, 1], 90)}")
                    if len(tp_):
                        T["vs_stereo_top"] = (f"n {len(tp_)}: stereo |top| med {_pct(tp_[:, 2], 50)} p90 {_pct(tp_[:, 2], 90)} vs this "
                                              f"{_pct(tp_[:, 3], 50)} / {_pct(tp_[:, 3], 90)}")
            M[tag] = T
        tdo, tdk = [], []
        for r in rows:
            for o in r[mode]["topdown"]["objects"]:
                if o.get("category") == "camera_stand":
                    continue
                tdo.append(o)
                tdk.append((r["set"], r["task"], r["episode"], o.get("gt_idx")))
        te = np.array([abs(o["top_err_mm"]) for o in tdo if "top_err_mm" in o])
        ce = np.array([o["cell_absmed_mm"] for o in tdo if "cell_absmed_mm" in o])
        tall = [o for o in tdo if "found" in o]
        TD = {"objects": len(tdo), "cov_mean": round(float(np.mean([o["cov"] for o in tdo])), 3) if tdo else None,
              "top_abs_med_mm": _pct(te, 50), "top_abs_p90_mm": _pct(te, 90),
              "top_within_5mm": round(float((te < 5).mean()), 3) if len(te) else None,
              "cell_absmed_med_mm": _pct(ce, 50), "found": f"{sum(o['found'] for o in tall)}/{len(tall)}",
              "phantom_blobs_per_frame": round(float(np.mean([r[mode]["topdown"]["phantom_blobs"] for r in rows])), 2),
              "table_p90_mm_med": _pct([r[mode]["topdown"]["table_p90_mm"] for r in rows if r[mode]["topdown"]["table_p90_mm"] is not None], 50)}
        if b1:
            for cls in ("missed", "gross", "identity"):
                sel = [o for o, k in zip(tdo, tdk) if b1.get(k) == cls]
                st = [o for o in sel if "found" in o]
                zz = np.array([abs(o["top_err_mm"]) for o in sel if "top_err_mm" in o])
                TD[f"b1_{cls}"] = f"n {len(sel)}, found {sum(o['found'] for o in st)}/{len(st)}, |top| med {_pct(zz, 50)} p90 {_pct(zz, 90)}"
        M["topdown"] = TD
        out[mode] = M
    return out


def check_summary_main(a):
    od = Path(a.out).expanduser() / a.tag
    rows = [json.loads(line) for line in open(od / "frames.jsonl")]
    if a.sets:
        rows = [r for r in rows if r.get("set") in a.sets]
    summ = check_summary(rows)
    (od / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def summary_main(a):
    out = Path(a.out).expanduser()
    od = out / f"eval_{a.tag}"
    rows = [json.loads(line) for line in open(od / "frames.jsonl")]
    if a.sets:
        rows = [r for r in rows if r.get("set") in a.sets]
    summ = summarize(rows, _b1_classes(a.b1), _stereo_objs(a.stereo))
    inf = out / f"infer_{rows[0]['views']}.jsonl" if rows else None
    if inf and inf.exists():
        t = [json.loads(line) for line in open(inf)]
        summ["infer"] = {"s_med": _pct([x["infer_s"] for x in t], 50), "s_p90": _pct([x["infer_s"] for x in t], 90),
                         "vram_peak_gb_max": max(x["vram_peak_gb"] for x in t), "views_med": _pct([x["n_views"] for x in t], 50)}
    (od / f"summary{'_' + '_'.join(a.sets) if a.sets else ''}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("infer")
    p.add_argument("--model", default="vggt")
    p.add_argument("--weights", required=True)
    p.add_argument("--root", nargs="+", required=True)
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--ids", nargs="*", type=int)
    p.add_argument("--views", default="home", choices=["home", "x"])
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--subsample", type=int, default=4, help="MASt3R: match seed grid step (px)")
    p.add_argument("--no-poses", action="store_true", help="MapAnything: intrinsics only, no camera poses")
    p.add_argument("--bf16-trunk", action="store_true", help="VGGT: aggregator weights in bf16 (about half the VRAM)")
    e = sub.add_parser("eval")
    e.add_argument("--out", required=True)
    e.add_argument("--views", default="home")
    e.add_argument("--conf-q", type=float, default=0.0)
    e.add_argument("--tag", default="")
    e.add_argument("--workers", type=int, default=6)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--viz", type=int, default=6)
    e.add_argument("--modes", default="sim3,table,poly", help="depth scale modes: sim3 (the model's cameras), table, poly")
    sm = sub.add_parser("summary")
    sm.add_argument("--out", required=True)
    sm.add_argument("--tag", required=True)
    sm.add_argument("--b1", default="~/dojo-data/pbake-mv/b1_objects.jsonl")
    sm.add_argument("--stereo", default="")
    sm.add_argument("--sets", nargs="*")
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    c.add_argument("--views", default="home")
    c.add_argument("--mode", default="poly")
    c.add_argument("--b1dump", default="~/dojo-data/pbake-mv/c4/sam/b1/dump")
    c.add_argument("--no-sweep", action="store_true")
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--limit", type=int, default=0)
    pp = sub.add_parser("propose")
    pp.add_argument("--out", required=True)
    pp.add_argument("--views", default="home")
    pp.add_argument("--mode", default="poly")
    pp.add_argument("--source", default="mv3d", choices=["mv3d", "hyb_any", "hyb_gate"])
    pp.add_argument("--b1dump", default="~/dojo-data/pbake-mv/c4/sam/b1/dump")
    pp.add_argument("--kw", default="{}", help="propose_unexplained kwargs (JSON)")
    pp.add_argument("--tag", default="")
    pp.add_argument("--sets", nargs="*")
    pp.add_argument("--match", nargs="*", help="scene name substrings")
    pp.add_argument("--resume", action="store_true", help="append, skipping scenes already in frames.jsonl")
    pp.add_argument("--workers", type=int, default=4)
    pp.add_argument("--limit", type=int, default=0)
    cs = sub.add_parser("check-summary")
    cs.add_argument("--out", required=True)
    cs.add_argument("--tag", default="check_home_poly")
    cs.add_argument("--sets", nargs="*")
    a = ap.parse_args(argv)
    {"infer": infer_main, "eval": eval_main, "summary": summary_main, "check": check_main,
     "check-summary": check_summary_main, "propose": propose_main}[a.cmd](a)


if __name__ == "__main__":
    main()
