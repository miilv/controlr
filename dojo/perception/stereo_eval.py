"""DEV-ONLY evaluation of ``stereo.plane_sweep`` against our own layout ground truth (D0-style frames).

    python -m dojo.perception.stereo_eval --root ~/dojo-data/d0_51 --tasks stack_blocks ... --out DIR [--ids ...]

Per frame: the GT height of the first surface each head pixel sees (ray cast into the GT meshes, table = 0),
then
- coverage: share of on-table head pixels the sweep can decide (another view sees them, textured);
- per object: top-surface height, GT vs estimate, as the 90th percentile over the object's visible head pixels
  (``z_err_mm`` = estimate - GT);
- foreground: pixels with GT height > ``fg_h`` vs ``proud`` (recall / precision over decided pixels), per object
  "detected" when >= 30 % of its decided pixels are proud, and phantom blobs (proud components touching no object).
GT is read only here (``d0.load_gt``), after the sweep.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .cameras import TABLE_Z, Camera, pose_to_T
from .segment import TABLE_CENTER, TABLE_HALF

ALLOWED_IDS = set(range(100000, 100010)) | set(range(100020, 100050))


def gt_height_map(cam: Camera, objects: list[dict], robot: np.ndarray | None = None):
    """(height above TABLE_Z per pixel, NaN = off-table / robot; object index per pixel, -1 = table / none;
    keys of objects without a catalogue mesh)."""
    import trimesh

    from .render import load_asset, silhouette, transform

    h, w = cam.h, cam.w
    vv, uu = np.mgrid[0:h, 0:w]
    uv = np.c_[uu.ravel() + 0.5, vv.ravel() + 0.5]
    X = cam.ray_plane(uv, TABLE_Z)
    on = np.isfinite(X).all(1) & (np.abs(X[:, 0] - TABLE_CENTER[0]) < TABLE_HALF[0]) & (np.abs(X[:, 1] - TABLE_CENTER[1]) < TABLE_HALF[1])
    depth = np.where(on, np.linalg.norm(np.nan_to_num(X) - cam.center, axis=1), np.inf)
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
        T = pose_to_T(o["pos"], o["quat"])
        Vw = transform(a.Vs, T)
        m = silhouette(cam, Vw, a.Fs).ravel() > 0
        if not m.any():
            continue
        m = cv2.dilate(m.reshape(h, w).astype(np.uint8), np.ones((3, 3), np.uint8)).ravel() > 0
        idx = np.flatnonzero(m)
        mesh = trimesh.Trimesh(Vw, a.Fs, process=False)
        loc, ir, it = mesh.ray.intersects_location(np.repeat(cam.center[None], len(idx), 0), dirs[idx], multiple_hits=False)
        if len(ir) == 0:
            continue
        d = np.linalg.norm(loc - cam.center, axis=1)
        pix = idx[ir]
        closer = d < depth[pix]
        pix, d, loc, it = pix[closer], d[closer], loc[closer], it[closer]
        nz[pix] = np.abs(mesh.face_normals[it][:, 2])
        depth[pix] = d
        hgt[pix] = loc[:, 2] - TABLE_Z
        oid[pix] = i
    hgt = hgt.reshape(h, w)
    oid = oid.reshape(h, w)
    if robot is not None:
        hgt[robot > 0] = np.nan
        oid[robot > 0] = -2
    return hgt.astype(np.float32), oid, missing, nz.reshape(h, w).astype(np.float32)


def _lab(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)


def score(sw, gt_h: np.ndarray, gt_id: np.ndarray, objects: list[dict], fg_h: float = 0.01, min_px: int = 8,
          bg_fg: np.ndarray | None = None, img: np.ndarray | None = None, gt_nz: np.ndarray | None = None) -> dict:
    region = np.isfinite(gt_h) & ((sw.valid > 0) if sw.valid is not None else True)
    known = (sw.known > 0) & region
    table_px = region
    gt_fg = known & (gt_h > fg_h)
    pred = (sw.proud > 0) & region
    tp = int((pred & gt_fg).sum())
    hv = np.isfinite(sw.height) & np.isfinite(gt_h)
    err = np.abs(sw.height - gt_h)[hv]
    objs = []
    for i, o in enumerate(objects):
        vis = gt_id == i
        nvis = int(vis.sum())
        if nvis < min_px:
            continue
        gt_top = float(np.percentile(gt_h[vis], 90))
        kn = vis & (sw.known > 0)
        est = vis & np.isfinite(sw.height)
        row = {"label": o.get("label"), "category": o.get("category"), "gt_idx": i, "clutter": bool(o.get("clutter")), "px": nvis,
               "px_known": int(kn.sum()), "px_est": int(est.sum()), "gt_top_mm": round(gt_top * 1000, 1)}
        if est.sum() >= min_px:
            z = float(np.percentile(sw.height[est], 90))
            gt_cov = float(np.percentile(gt_h[est], 90))  # the GT top over the same (measured) pixels
            row.update({"est_top_mm": round(z * 1000, 1), "z_err_mm": round((z - gt_cov) * 1000, 1),
                        "z_err_all_mm": round((z - gt_top) * 1000, 1), "gt_top_cov_mm": round(gt_cov * 1000, 1),
                        "px_err_med_mm": round(float(np.median(np.abs(sw.height[est] - gt_h[est]))) * 1000, 1)})
        if kn.sum() >= min_px:
            raised = kn & (gt_h > fg_h)
            row["proud_frac"] = round(float((sw.proud[kn] > 0).mean()), 3)
            row["detected"] = bool(raised.sum() >= min_px and (sw.proud[raised] > 0).mean() >= 0.3)
            row["tall"] = bool(gt_top > 0.02)
        if gt_nz is not None:  # horizontal top-facing surfaces (|n_z| > 0.9): what horizontal planes model exactly
            topm = est & (gt_nz > 0.9) & (gt_h > 0.003)
            if topm.sum() >= min_px:
                row["top_err_mm"] = round(float(np.median(sw.height[topm] - gt_h[topm])) * 1000, 1)
                row["top_px"] = int(topm.sum())
        if img is not None:  # colour contrast of the object against the table ring around it (Lab, L halved)
            ring = cv2.dilate(vis.astype(np.uint8), np.ones((9, 9), np.uint8)) & ~cv2.dilate(vis.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            ring = ring.astype(bool) & (gt_id == -1)
            if ring.sum() >= min_px:
                L = _lab(img)
                d = np.median(L[vis], 0) - np.median(L[ring], 0)
                row["contrast"] = round(float(np.sqrt((0.5 * d[0]) ** 2 + d[1] ** 2 + d[2] ** 2)), 1)
        if bg_fg is not None:
            row["bg_frac"] = round(float((bg_fg[vis] > 0).mean()), 3)
            if kn.sum() >= min_px:
                row["bg_frac_known"] = round(float((bg_fg[kn] > 0).mean()), 3)
        objs.append(row)
    shadow = {}
    if bg_fg is not None:  # B1's colour foreground on the table (shadows, reflections): does the sweep reject it?
        bfp = (bg_fg > 0) & known & (gt_h <= 0.003) & (cv2.dilate((gt_id >= 0).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)
        btp = (bg_fg > 0) & known & (gt_h > fg_h)
        shadow = {"bg_fp_px": int(bfp.sum()), "bg_fp_on_table": int((bfp & (sw.on_table > 0)).sum()),
                  "bg_fp_proud": int((bfp & (sw.proud > 0)).sum()), "bg_tp_px": int(btp.sum()),
                  "bg_tp_on_table": int((btp & (sw.on_table > 0)).sum()), "bg_tp_proud": int((btp & (sw.proud > 0)).sum())}
    # phantom blobs: proud components that touch no GT object pixel (objects dilated by 5 px)
    obj_d = cv2.dilate((gt_id >= 0).astype(np.uint8), np.ones((11, 11), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((pred & ~(gt_id == -2)).astype(np.uint8))
    phantoms = []
    for j in range(1, n):
        a = int(stats[j, cv2.CC_STAT_AREA])
        if a < min_px * 4:
            continue
        if not obj_d[lab == j].any():
            ys, xs = np.nonzero(lab == j)
            phantoms.append({"area_px": a, "uv": [int(xs.mean()), int(ys.mean())]})
    return {
        "coverage": round(float(known.sum() / max(table_px.sum(), 1)), 4),
        "fg_recall": round(tp / max(int(gt_fg.sum()), 1), 4), "fg_precision": round(tp / max(int(pred.sum()), 1), 4),
        "n_fg_gt": int(gt_fg.sum()), "n_pred": int(pred.sum()),
        "h_err_med_mm": round(float(np.median(err)) * 1000, 2) if err.size else None,
        "h_err_p90_mm": round(float(np.percentile(err, 90)) * 1000, 2) if err.size else None,
        "h_within_5mm": round(float((err < 0.005).mean()), 4) if err.size else None,
        "h_px": int(err.size), "objects": objs, "phantoms": phantoms, "shadow": shadow,
    }


def viz(path: Path, img: np.ndarray, sw, gt_h: np.ndarray):
    s = sw.scale
    small = cv2.resize(img, (sw.height.shape[1], sw.height.shape[0]), interpolation=cv2.INTER_AREA)

    def cmap(hm):
        v = np.clip(np.nan_to_num(hm, nan=-1) / 0.15, 0, 1)
        c = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
        c[~np.isfinite(hm)] = 40
        return c

    pr = np.zeros_like(small)
    pr[sw.on_table > 0] = (60, 90, 60)
    pr[sw.proud > 0] = (255, 255, 0)
    top = np.concatenate([small, cmap(sw.height)], 1)
    bot = np.concatenate([pr, cmap(gt_h)], 1)
    out = np.concatenate([top, bot], 0)
    _ = s
    cv2.imwrite(str(path), out[..., ::-1])


def bg_dir_for(root: Path) -> Path | None:
    import os

    base = Path(os.path.expanduser("~/dojo-perception-data"))
    d = base / ("bg_folds_51" if root.name == "d0_51" else "bg_folds")
    return d if d.exists() else None


def head_bg_fg(ep: Path, img: np.ndarray, state: dict) -> np.ndarray | None:
    """B1's head-view colour foreground (2-fold background: built from the other half of the episodes)."""
    import zlib

    from .cameras import head_camera
    from .segment import load_backgrounds, segment_views

    d = bg_dir_for(ep.parent.parent)
    if d is None:
        return None
    fd = d / f"fold{zlib.crc32(ep.name.encode()) % 2}"
    bgs = load_backgrounds(fd)
    m = fd / "cam_head_clean_unseen.png"
    if m.exists():
        bgs["cam_head_clean_unseen"] = (cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    v = segment_views({"cam_head": img}, {"cam_head": head_camera()}, state, bgs)["cam_head"]
    return v.fg


def run_frame(ep: Path, k=0, scale: float = 0.5, out: Path | None = None, sweep_kw: dict | None = None,
              extra_views: str = "") -> dict:
    from . import d0
    from .render import robot_mask
    from .stereo import home_views, plane_sweep, scaled_camera

    obs = d0.load_obs(ep, k)
    t0 = time.time()
    ref, srcs = home_views(obs["images"], obs["state"])
    if extra_views:  # other frames of the same episode as extra source views (objects static: look frames)
        from .cameras import cameras_from_state
        from .stereo import View

        for f in extra_views.split(","):
            if not (ep / f"frame_{f}.json").exists():
                continue
            o2 = d0.load_obs(ep, f if not f.isdigit() else int(f))
            cams = cameras_from_state(o2["state"])
            for name, img in o2["images"].items():
                if name == "cam_head":
                    continue
                rob = robot_mask(cams[name], o2["state"], dilate_px=4)
                srcs.append(View(img, cams[name], (1 - rob).astype(np.uint8)))
    t_mask = time.time() - t0
    sw = plane_sweep(ref, srcs, scale=scale, **(sweep_kw or {}))
    gt = d0.load_gt(ep, k)  # DEV-ONLY, after the sweep
    cam = scaled_camera(ref.cam, scale)
    rob = robot_mask(cam, obs["state"], dilate_px=2)
    gt_h, gt_id, missing, gt_nz = gt_height_map(cam, gt["objects"], rob)
    fg = head_bg_fg(ep, obs["images"]["cam_head"], obs["state"])
    if fg is not None and scale != 1.0:
        fg = cv2.resize(fg, (cam.w, cam.h), interpolation=cv2.INTER_NEAREST)
    small = cv2.resize(obs["images"]["cam_head"], (cam.w, cam.h), interpolation=cv2.INTER_AREA)
    sc = score(sw, gt_h, gt_id, gt["objects"], bg_fg=fg, img=small, gt_nz=gt_nz)
    sc.update({"task": ep.parent.name, "episode": ep.name, "frame": k, "timings": {**sw.timings, "masks_s": round(t_mask, 3)},
               "missing_meshes": missing, "n_src": len(srcs)})
    if out is not None:
        viz(out / f"{ep.parent.name}_{ep.name}_{k}{'_x' if extra_views else ''}.png", obs["images"]["cam_head"], sw, gt_h)
    return sc


def _job(args):
    import cv2 as _cv

    _cv.setNumThreads(2)
    ep, scale, out, kw, extra = args
    try:
        return run_frame(Path(ep), 0, scale, out, kw, extra)
    except Exception as e:  # keep the eval going
        return {"task": Path(ep).parent.name, "episode": Path(ep).name, "error": f"{type(e).__name__}: {e}"}


def _found_stats(objs: list[dict], low_contrast: float = 15.0) -> dict:
    """Objects inside the sweep's coverage with GT height > 1 cm: found by B1's colour foreground (>= 30 % of its
    decided pixels), by the sweep (``detected``), by either; the same for low-contrast objects."""
    cov = [o for o in objs if o.get("tall") is not None and "bg_frac_known" in o and o.get("detected") is not None
           and o["gt_top_mm"] > 10]
    out = {}
    for tag, sel in (("", cov), ("lowc_", [o for o in cov if o.get("contrast", 99) < low_contrast])):
        bg = sum(o["bg_frac_known"] >= 0.3 for o in sel)
        st = sum(bool(o["detected"]) for o in sel)
        un = sum(o["bg_frac_known"] >= 0.3 or bool(o["detected"]) for o in sel)
        out[tag + "found_bg_st_union"] = f"{bg}/{st}/{un} of {len(sel)}"
    return out


def _shadow_stats(rows: list[dict]) -> dict:
    sh = [r.get("shadow") or {} for r in rows]
    fp = sum(s.get("bg_fp_px", 0) for s in sh)
    tp = sum(s.get("bg_tp_px", 0) for s in sh)
    if not fp and not tp:
        return {}
    return {"bgFP_px": fp, "bgFP_rejected": round(sum(s.get("bg_fp_on_table", 0) for s in sh) / max(fp, 1), 3),
            "bgFP_confirmed": round(sum(s.get("bg_fp_proud", 0) for s in sh) / max(fp, 1), 3),
            "bgTP_wrongly_rejected": round(sum(s.get("bg_tp_on_table", 0) for s in sh) / max(tp, 1), 3)}


def summarize(rows: list[dict]) -> dict:
    out = {}
    tasks = sorted({r["task"] for r in rows})
    for t in tasks + ["ALL"]:
        rs = [r for r in rows if (t == "ALL" or r["task"] == t) and "error" not in r]
        if not rs:
            continue
        objs = [o for r in rs for o in r["objects"] if o.get("category") != "camera_stand"]
        z = np.array([abs(o["z_err_mm"]) for o in objs if "z_err_mm" in o])
        zs = np.array([o["z_err_mm"] for o in objs if "z_err_mm" in o])
        tall = [o for o in objs if o.get("tall") is not None and o["tall"]]
        tz = np.array([abs(o["top_err_mm"]) for o in objs if "top_err_mm" in o])
        full = [r for r in rs if not r.get("missing_meshes")]  # GT complete: precision / phantoms are meaningful
        n_gt = sum(r["n_fg_gt"] for r in full)
        tp = sum(r["fg_recall"] * r["n_fg_gt"] for r in full)
        n_pred = sum(r["n_pred"] for r in full)
        out[t] = {"frames": len(rs), "objects": len(objs), "obj_seen": int(len(z)),
                  "coverage": round(float(np.mean([r["coverage"] for r in rs])), 3),
                  "z_abs_med_mm": round(float(np.median(z)), 1) if len(z) else None,
                  "z_abs_p90_mm": round(float(np.percentile(z, 90)), 1) if len(z) else None,
                  "z_bias_med_mm": round(float(np.median(zs)), 1) if len(z) else None,
                  "z_within_5mm": round(float((z < 5).mean()), 3) if len(z) else None,
                  "z_within_10mm": round(float((z < 10).mean()), 3) if len(z) else None,
                  "top_n": int(len(tz)), "top_abs_med_mm": round(float(np.median(tz)), 1) if len(tz) else None,
                  "top_abs_p90_mm": round(float(np.percentile(tz, 90)), 1) if len(tz) else None,
                  "top_within_5mm": round(float((tz < 5).mean()), 3) if len(tz) else None,
                  "fg_recall_px": round(tp / max(n_gt, 1), 3), "fg_precision_px": round(tp / max(n_pred, 1), 3),
                  "tall_detected": f"{sum(o['detected'] for o in tall)}/{len(tall)}",
                  "phantom_blobs_per_frame": round(float(np.mean([len(r["phantoms"]) for r in full])), 2) if full else None,
                  "frames_gt_complete": len(full),
                  **_found_stats(objs), **_shadow_stats(full),
                  "s_per_frame": round(float(np.mean([r["timings"]["total_s"] + r["timings"]["masks_s"] for r in rs])), 2)}
    return out


def to_markdown(summ: dict) -> str:
    cols = ["frames", "objects", "obj_seen", "coverage", "z_abs_med_mm", "z_abs_p90_mm", "z_bias_med_mm", "z_within_5mm",
            "z_within_10mm", "top_n", "top_abs_med_mm", "top_abs_p90_mm", "top_within_5mm", "fg_recall_px", "fg_precision_px", "tall_detected", "phantom_blobs_per_frame",
            "found_bg_st_union", "lowc_found_bg_st_union", "bgFP_px", "bgFP_rejected", "bgFP_confirmed", "bgTP_wrongly_rejected",
            "s_per_frame"]
    lines = ["| task | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for t, r in summ.items():
        lines.append(f"| {t} | " + " | ".join(str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None):
    import os

    os.environ.setdefault("DOJO_GL", "0")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--ids", nargs="*", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--sweep", default="{}", help="plane_sweep kwargs (JSON)")
    ap.add_argument("--extra", default="", help="extra source frames, e.g. 1,2,3 or L1")
    ap.add_argument("--viz", type=int, default=6, help="write a picture for the first N frames")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    from concurrent.futures import ProcessPoolExecutor

    from . import d0

    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for root in a.root:
        for ep in d0.episodes(Path(root).expanduser()):
            if a.tasks and ep.parent.name not in a.tasks:
                continue
            if not ep.name.isdigit():
                continue
            i = int(ep.name)
            if i not in ALLOWED_IDS or (a.ids and i not in a.ids):
                continue
            jobs.append((str(ep), a.scale, out if len(jobs) < a.viz else None, json.loads(a.sweep), a.extra))
    rows = []
    with open(out / "frames.jsonl", "w") as fh, ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(_job, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
            if "error" in r:
                print(r["task"], r["episode"], "ERROR", r["error"], flush=True)
                continue
            print(f"{r['task']} {r['episode']} cov {r['coverage']:.2f} fgR {r['fg_recall']:.2f} fgP {r['fg_precision']:.2f} "
                  f"herr {r['h_err_med_mm']} n_obj_z {sum('z_err_mm' in o for o in r['objects'])} "
                  f"ph {len(r['phantoms'])} t {r['timings']['total_s']}s", flush=True)
    summ = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    (out / "summary.md").write_text(to_markdown(summ) + "\n")
    print(to_markdown(summ))


if __name__ == "__main__":
    main()
