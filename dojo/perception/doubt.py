"""Per-object doubt about B1's record, without an LLM: does the scene look and stand the way the record says?

Two independent measurements per record object (both from the mv3d GPU server, ``DOJO_MV3D_ADDR``):

- ``d``: the multi-view DINOv2 feature distance of the object rendered at its recorded pose and composited over the
  observed crops, against the observed crops (head + wrists; ``featmetric``). A right pose looks like the image; a swap,
  a wrong blob or a flip does not.
- ``dh_mm``: the learned height check (``stereo.check_heights`` on VGGT's fused head height map): the recorded object's
  surface vs the measured one.

``doubt`` = d > ``feat_d`` or |dh| > ``dh_mm`` (mode ``or``), or both (``and``). On 1016 frame-0 dev objects (95
gross / overshoot; d alone: AUC 0.91): ``or`` at 0.6 / 15 mm flags 67 of 95 wrong objects at 7.7 % false flags on right
ones, at 0.55 / 15 mm 73 at 10 %; d > 0.6 alone 60 at 2.4 % (experiments/2026-10-28-e3-featmetric).

Written per object as ``_perception.b1_doubt`` = {"d", "dh_mm", "doubt", "why"} and per record as
``_perception.b1_doubt`` = {"labels", "thr", "s"}. Consumers (each behind its own flag): the agentic gate
(``agentic.record.needs_agent`` / ``j1.risk``) and the look planner (``runtime.estimate.flagged``,
``look.look_plan``). For other streams: ``doubted(record)`` -> [(label, world xyz, info)].
"""
from __future__ import annotations

import time

import numpy as np


def _entries(record: dict):
    for typ in ("Rigid", "Geometry", "Articulation"):
        for cat, lst in (record.get(typ) or {}).items():
            for r in lst:
                yield typ, cat, r


def annotate(record: dict, d: dict, dh: dict, feat_d: float = 0.6, dh_mm: float = 15.0, mode: str = "or") -> list[str]:
    """Write the per-object doubt from measured feature distances ``d`` and height mismatches ``dh`` (both keyed by
    ``id(entry)``; a missing value = not measured). Returns the doubted labels."""
    out = []
    for typ, cat, r in _entries(record):
        p = r.setdefault("_perception", {})
        if p.get("place_tag") == "fixed":
            continue
        dv, hv = d.get(id(r)), dh.get(id(r))
        a = dv is not None and np.isfinite(dv) and dv > feat_d
        b = hv is not None and np.isfinite(hv) and abs(hv) > dh_mm
        doubt = (a or b) if mode == "or" else (a and b)
        why = (["appearance"] if a else []) + (["height"] if b else [])
        p["b1_doubt"] = {"d": None if dv is None or not np.isfinite(dv) else round(float(dv), 4),
                         "dh_mm": None if hv is None or not np.isfinite(hv) else round(float(hv), 1),
                         "doubt": bool(doubt), "why": why}
        if doubt:
            out.append(r.get("label"))
    return out


def b1_doubt(record: dict, images: dict, cams: dict, robot_masks: dict, heights: np.ndarray | None = None,
             feat_d: float = 0.6, dh_mm: float = 15.0, mode: str = "or", color_M=None, cl=None) -> dict:
    """Measure and annotate (see the module doc). ``heights``: the learned head height map when the caller already has
    it. Without a server: {"skipped": ...} and no annotation."""
    from . import mv3d
    from .cameras import pose_to_T
    from .stereo import check_heights

    t0 = time.time()
    cl = cl or mv3d.client()
    if cl is None:
        return {"skipped": f"no mv3d server at ${mv3d.ENV_ADDR}"}
    dh: dict = {}
    if heights is None:
        r = mv3d.learned_head_height(images, cams, robot_masks, cl=cl)
        heights = None if r is None else r[0]
    if heights is not None:
        check_heights(record, cams["cam_head"], mv3d.height_sweep_like(heights, np.isfinite(heights)), key="mv3d")
        for _, _, r in _entries(record):
            v = ((r.get("_perception") or {}).get("mv3d") or {}).get("dh_mm")
            if v is not None:
                dh[id(r)] = float(v)
    ents = [(typ, cat, r) for typ, cat, r in _entries(record)]
    objs = [(f"{typ}/{cat}/{int(r['category_idx']):05d}", pose_to_T(r["default_pos"], r["default_ori"]).tolist())
            for typ, cat, r in ents]
    names = [n for n in ("cam_head", "cam_left_wrist", "cam_right_wrist") if n in images and n in cams]
    d: dict = {}
    if objs and names:
        fd = cl.featdist([images[n] for n in names], [cams[n] for n in names],
                         [(robot_masks or {}).get(n) for n in names], objs, color_M)
        for (_, _, r), v in zip(ents, fd["d"]):
            if v is not None:
                d[id(r)] = float(v)
    labels = annotate(record, d, dh, feat_d, dh_mm, mode)
    summ = {"labels": labels, "thr": {"feat_d": feat_d, "dh_mm": dh_mm, "mode": mode}, "s": round(time.time() - t0, 3)}
    record.setdefault("_perception", {})["b1_doubt"] = summ
    return summ


def doubted(record: dict) -> list[tuple[str | None, list, dict]]:
    """[(label, world xyz, the object's b1_doubt)] of the doubted objects of an annotated record (empty otherwise)."""
    return [(r.get("label"), [float(v) for v in r["default_pos"]], (r.get("_perception") or {})["b1_doubt"])
            for _, _, r in _entries(record) if ((r.get("_perception") or {}).get("b1_doubt") or {}).get("doubt")]


def mark_for_look(record: dict) -> dict:
    """A copy of ``record`` whose doubted objects carry ``_perception.doubt_look`` (``look.look_plan`` gives them a
    targeted look first)."""
    import copy

    out = copy.deepcopy(record)
    for _, _, r in _entries(out):
        p = r.get("_perception") or {}
        if (p.get("b1_doubt") or {}).get("doubt"):
            p["doubt_look"] = True
    return out


def refine_doubted(record: dict, images: dict, cams: dict, robot_masks: dict, min_d: float = 0.45, min_gain: float = 0.05,
                   color_M=None, iters: int = 40, cl=None, b1_views: dict | None = None, w_sil: float = 0.0) -> dict:
    """The continuous feature-metric stage (research R3, ``featmetric.refine`` on the mv3d server) for the Rigid
    objects whose measured feature distance (``_perception.b1_doubt.d``, from ``b1_doubt``) exceeds ``min_d``: a
    refined pose is kept when it lowers the distance by >= ``min_gain``. Frame 0, 207 dev scenes, d > 0.45: gross -9,
    within tolerance +9 net, 0 right objects broken into gross (5 into overshoot). Annotates
    ``_perception.fm_refine`` per object. Returns a summary ({"skipped"} without a server)."""
    from . import mv3d
    from .cameras import R_to_quat_wxyz, pose_to_T

    t0 = time.time()
    cl = cl or mv3d.client()
    if cl is None:
        return {"skipped": f"no mv3d server at ${mv3d.ENV_ADDR}"}
    todo = []
    for typ, cat, r in _entries(record):
        d = ((r.get("_perception") or {}).get("b1_doubt") or {}).get("d")
        if typ == "Rigid" and d is not None and d > min_d:
            todo.append((f"{typ}/{cat}/{int(r['category_idx']):05d}", r))
    names = [n for n in ("cam_head", "cam_left_wrist", "cam_right_wrist") if n in images and n in cams]
    if not todo or not names:
        return {"refined": 0, "kept": 0}
    if w_sil > 0 and b1_views:
        return _refine_joint(record, todo, images, cams, robot_masks, names, min_gain, color_M, iters, cl, b1_views, w_sil, t0)
    out = cl.fmrefine([images[n] for n in names], [cams[n] for n in names], [(robot_masks or {}).get(n) for n in names],
                      [(k, pose_to_T(r["default_pos"], r["default_ori"])) for k, r in todo], color_M, iters)
    kept = []
    for (k, r), T1, a, b in zip(todo, out["poses"], out["d0"], out["d1"]):
        keep = a is not None and b is not None and b < a - min_gain
        T1 = np.asarray(T1, float)
        info = {"d0": a, "d1": b, "kept": bool(keep),
                "moved_mm": round(float(np.linalg.norm(T1[:3, 3] - np.asarray(r["default_pos"], float))) * 1000, 1)}
        if keep:
            r["default_pos"] = [round(float(v), 6) for v in T1[:3, 3]]
            r["default_ori"] = [round(float(v), 6) for v in R_to_quat_wxyz(T1[:3, :3])]
            kept.append(r.get("label"))
        r.setdefault("_perception", {})["fm_refine"] = info
    summ = {"refined": len(todo), "kept": kept, "s": round(time.time() - t0, 3)}
    record.setdefault("_perception", {})["fm_refine"] = summ
    return summ


def _refine_joint(record, todo, images, cams, robot_masks, names, min_gain, color_M, iters, cl, b1_views, w_sil, t0) -> dict:
    """The joint objective: the server's feature distance (one fixed window at the start pose) + ``w_sil`` x B1's
    silhouette loss in B1's own views, Nelder-Mead in this process. Frame 0, 207 dev scenes, w_sil 1.0: gross -7,
    within +8, 0 right objects broken (feature distance alone: gross -9, +9, 5 into overshoot)."""
    from . import mv3d
    from .cameras import R_to_quat_wxyz, pose_to_T
    from .featmetric import refine_with
    from .inverse_layout import make_roi_views, roi_loss
    from .refine import _shape_for
    from .render import load_asset

    imgs = [images[n] for n in names]
    cs = [cams[n] for n in names]
    robs = [(robot_masks or {}).get(n) for n in names]
    kept = []
    for k, r in todo:
        T0 = pose_to_T(r["default_pos"], r["default_ori"])
        a = load_asset(k)
        centre = T0[:3, :3] @ ((a.V.min(0) + a.V.max(0)) / 2) + T0[:3, 3]
        sm = _shape_for(k, T0[:3, :3], None)
        rvs = make_roi_views(sm, T0, b1_views, mv3d._others_masks(record, b1_views, skip=r))
        extra = (lambda T, sm=sm, rvs=rvs: roi_loss(sm, T, rvs)[0]) if rvs else None

        def dist(T, k=k, T0=T0):
            v = cl.featdist(imgs, cs, robs, [(k, T)], color_M, box_pose=T0)["d"][0]
            return 1.0 if v is None else v
        T1, f0, f1 = refine_with(dist, T0, centre, iters=iters, extra=extra, w_extra=w_sil)
        keep = f1 < f0 - min_gain
        info = {"f0": round(f0, 4), "f1": round(f1, 4), "kept": bool(keep), "w_sil": w_sil,
                "moved_mm": round(float(np.linalg.norm(T1[:3, 3] - T0[:3, 3])) * 1000, 1)}
        if keep:
            r["default_pos"] = [round(float(v), 6) for v in T1[:3, 3]]
            r["default_ori"] = [round(float(v), 6) for v in R_to_quat_wxyz(T1[:3, :3])]
            kept.append(r.get("label"))
        r.setdefault("_perception", {})["fm_refine"] = info
    summ = {"refined": len(todo), "kept": kept, "w_sil": w_sil, "s": round(time.time() - t0, 3)}
    record.setdefault("_perception", {})["fm_refine"] = summ
    return summ
