"""Single-object refinement from one wrist frame, e.g. at the pre-grasp hover (~10-15 cm above the target).

    from dojo.perception.refine import refine
    out = refine("Rigid/block/00012", wrist_rgb, proprio, prior_pose, arm="left")
    out["pose"]      # [x, y, z, qw, qx, qy, qz], env frame (same convention as the layout record)
    out["residual"]  # 1 - IoU of the refined silhouette with the wrist foreground (lower is better)

- The wrist camera comes from FK on ``proprio[f"{arm}_arm_joint_state"]`` (no simulator query).
- Foreground (``seg="local"``, default): a table colour model sampled from a ring around the prior's
  projection in the same frame (k-means in Lab); pixels near the prior far from every table cluster are
  foreground; the robot (FK silhouette from the same proprio) is unknown. No background images needed.
  ``seg="warped"``: the head background warped through the table plane (``DOJO_BG``), which suffers from
  view-dependent sheen at close range.
- The prior is decomposed into the asset's resting place frame + yaw about world z + a height offset
  (kept: stacked targets keep their height). (x, y, yaw) are refined by Nelder-Mead from a few starts
  (prior, +-15 mm, the 180-degree flip when the shape is not symmetric); the lowest silhouette loss wins.
- ``others``: [(key, pose7)] of nearby known objects; their pixels are removed from the foreground, and a
  result within 0.7 x footprint of one of them is reported not-ok (``why`` = "neighbour:<key>").
- ``held``: (key, pose7) or a list of them: objects in the gripper (at a place hover the carried block is
  between the camera and the support): masked as occluders. Without it, place-time refines of the support
  mostly fail (integration: 15/44 ok, IoU ~0.24).
- ``refine_identity(obj_ids, ...)``: the same for several candidate assets (shape swaps such as
  build_tower's board / prism / pillar): the lowest residual wins, with a margin.
- Returns also ``iou``, ``visible_frac`` (share of the silhouette not under the robot / out of view),
  ``moved_mm`` / ``turned_deg`` vs the prior, ``ok`` (iou >= 0.6 and visible_frac >= 0.5), ``time_s``.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .cameras import TABLE_Z, R_to_quat_wxyz, pose_to_T, wrist_camera
from .inverse_layout import make_roi_views, refine as nm_refine, roi_loss, shape_by_tag, shape_models
from .segment import load_backgrounds, segment_virtual

_BGS = None


def _backgrounds(bgs):
    global _BGS
    if bgs is not None:
        return bgs
    if _BGS is None:
        _BGS = load_backgrounds()
    return _BGS


def local_foreground(img: np.ndarray, cam, sil: np.ndarray, robot: np.ndarray, grow: float = 0.8,
                     k: int = 5, thr: float = 3.0, exclude: np.ndarray | None = None):
    """Foreground near an expected object from the frame itself: a table colour model (k-means in Lab, with a
    per-cluster spread) sampled from a ring around the expected silhouette, then pixels inside the ROI far
    from every table cluster. Immune to the view-dependent sheen that breaks warped backgrounds at close
    range. Returns (fg, known) uint8 masks over the full image (known = the ROI minus the robot).
    ``exclude``: pixels that are not table either (other known objects, e.g. a light-wood board next to a white
    pillar: sampled as "table", it made the pillar's lit face background; 5.1 build_tower post-place iou 0.03-0.2)."""
    import cv2

    from .segment import ViewMasks  # noqa: F401  (same mask conventions)

    h, w = sil.shape
    ys, xs = np.nonzero(sil)
    known = np.zeros((h, w), np.uint8)
    fg = np.zeros((h, w), np.uint8)
    if len(xs) == 0:
        return fg, known
    size = max(np.ptp(xs), np.ptp(ys)) + 1
    pad = int(grow * size) + 10
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    roi = (slice(y0, y1), slice(x0, x1))
    near = cv2.dilate(sil, np.ones((2 * (pad // 3) + 1, 2 * (pad // 3) + 1), np.uint8))
    ring = (near[roi] == 0) & (robot[roi] == 0)
    if exclude is not None:
        ring &= exclude[roi] == 0
    lab = cv2.cvtColor(img[roi], cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 0.5
    samples = lab[ring]
    if len(samples) < 50:
        return fg, known
    if len(samples) > 4000:
        samples = samples[np.random.default_rng(0).choice(len(samples), 4000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(samples, min(k, len(samples)), None, crit, 2, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    spread = np.array([max(np.sqrt(((samples[labels == i] - centers[i]) ** 2).sum(1)).mean(), 3.0)
                       if (labels == i).any() else 3.0 for i in range(len(centers))])
    d = np.sqrt(((lab[..., None, :] - centers[None, None]) ** 2).sum(-1)) / spread[None, None]
    f = (d.min(-1) > thr).astype(np.uint8)
    f = cv2.morphologyEx(f, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    f = cv2.morphologyEx(f, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    known[roi] = 1
    known &= 1 - robot
    fg[roi] = f
    fg &= known
    return fg, known


def frame_foreground(img: np.ndarray, robot: np.ndarray, k: int = 6, thr: float = 3.0, n_samples: int = 6000) -> np.ndarray:
    """Foreground over the whole frame: a table/room colour model (k-means in Lab) from random non-robot pixels
    (objects are a small share of a wrist view), then pixels far from every cluster."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 0.5
    pool = lab[robot == 0]
    if len(pool) < 100:
        return np.zeros(robot.shape, np.uint8)
    samples = pool[np.random.default_rng(0).choice(len(pool), min(n_samples, len(pool)), replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(samples, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    spread = np.array([max(np.sqrt(((samples[labels == i] - centers[i]) ** 2).sum(1)).mean(), 3.0)
                       if (labels == i).any() else 3.0 for i in range(len(centers))])
    d = np.min(np.stack([np.sqrt(((lab - centers[i]) ** 2).sum(-1)) / spread[i] for i in range(len(centers))]), axis=0)
    f = ((d > thr) & (robot == 0)).astype(np.uint8)
    f = cv2.morphologyEx(f, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(f, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _shape_for(key: str, R: np.ndarray, tag: str | None):
    """The place frame whose resting up-direction matches the prior rotation best (or the given tag)."""
    sms = shape_models(key)
    if tag:
        try:
            return shape_by_tag(key, tag)  # includes fixed-pose (zlim / qpos) shapes estimate() created
        except StopIteration:
            pass
    up = R.T @ np.array([0, 0, 1.0])
    return max(sms, key=lambda s: float((s.R0.T @ np.array([0, 0, 1.0])) @ up))


def refine(obj_id: str, wrist_frame: np.ndarray, proprio: dict, prior_pose, arm: str = "left",
           tag: str | None = None, bgs: dict | None = None, others: list | None = None, iters: int = 60,
           xy_starts: float = 0.015, seg: str = "local", held=None, search: bool = True,
           support_snap: bool = True, edge_fallback: bool = True) -> dict:
    if support_snap and others:
        # a prior floating above every known support under it (a carried-height belief after a release; 5.1 stack
        # 100007 s142: 56 mm up, iou 0.0): also refine from the prior lowered onto that support. The local masks are
        # built around the prior's projection, so this is a second full refine, not a second start
        from .render import load_asset

        a_t = load_asset(obj_id)
        Tp = pose_to_T(prior_pose[:3], prior_pose[3:])
        lo = float((a_t.Vs @ Tp[:3, :3].T + Tp[:3, 3])[:, 2].min())
        drop = lo - support_z(a_t, prior_pose, others)
        if drop > SUPPORT_FLOAT_M:
            kw = dict(tag=tag, bgs=bgs, others=others, iters=iters, xy_starts=xy_starts, seg=seg, held=held,
                      search=search, support_snap=False, edge_fallback=edge_fallback)
            out_a = refine(obj_id, wrist_frame, proprio, prior_pose, arm, **kw)
            prior_b = [float(prior_pose[0]), float(prior_pose[1]), float(prior_pose[2]) - drop] + list(map(float, prior_pose[3:]))
            out_b = refine(obj_id, wrist_frame, proprio, prior_b, arm, **kw)
            if out_b["ok"] and out_b["iou"] >= out_a["iou"] + SUPPORT_GAIN_IOU:
                return {**out_b, "z_dropped_mm": round(1000 * drop, 1), "iou_at_prior_height": out_a["iou"]}
            return {**out_a, "support_tried": {"z_dropped_mm": round(1000 * drop, 1), "iou": out_b["iou"], "ok": out_b["ok"]}}
    t0 = time.time()
    bgs = _backgrounds(bgs) if seg != "local" else bgs
    cam = wrist_camera(arm, np.asarray(proprio[f"{arm}_arm_joint_state"], dtype=float))
    T_prior = pose_to_T(prior_pose[:3], prior_pose[3:])
    if seg == "local":  # table colour model from the frame itself, around the prior's projection
        from .render import load_asset, robot_mask, silhouette, transform
        from .segment import ViewMasks

        a = load_asset(obj_id)
        sil = silhouette(cam, transform(a.Vs, T_prior), a.Fs)
        rob = robot_mask(cam, proprio, dilate_px=3)
        # a held object (e.g. the block carried to a place) sits between the camera and the target: treat its
        # silhouette as an occluder, so it neither trains the table colour model nor counts as foreground
        for hk, hp in ([held] if held and isinstance(held[0], str) else (held or [])):
            ha = load_asset(hk)
            hm = silhouette(cam, transform(ha.Vs, pose_to_T(hp[:3], hp[3:])), ha.Fs)
            rob |= cv2.dilate(hm, np.ones((9, 9), np.uint8))
        excl = None
        if others:  # known objects other than the target's supports are not table: keep them out of its colour model
            excl = np.zeros_like(sil)
            for key, pose in others:
                ao = load_asset(key)
                if _under(ao, pose, a, prior_pose):
                    continue
                excl |= silhouette(cam, transform(ao.Vs, pose_to_T(pose[:3], pose[3:])), ao.Fs)
            excl = cv2.dilate(excl, np.ones((7, 7), np.uint8))
        fgm, kn = local_foreground(wrist_frame, cam, sil, rob, exclude=excl)
        v = ViewMasks(cam, wrist_frame, fgm, kn, np.zeros(fgm.shape, np.float32), rob)
    else:
        v = segment_virtual(wrist_frame, cam, bgs, state=proprio)
    if others:
        from .render import load_asset, silhouette, transform

        a_t = load_asset(obj_id)
        keep = None
        for key, pose in others:
            a = load_asset(key)
            m = silhouette(cam, transform(a.Vs, pose_to_T(pose[:3], pose[3:])), a.Fs)
            if _under(a, pose, a_t, prior_pose):
                # a support under the target: where the target is expected, its pixels are the target's (a stack seen
                # from above); elsewhere they are the support's (5.1 tower: a plank slid 36-44 mm along the same-colour
                # board under it). Keep the expected silhouette + a prior-error margin
                if keep is None:
                    keep = cv2.dilate(silhouette(cam, transform(a_t.Vs, T_prior), a_t.Fs).astype(np.uint8),
                                      np.ones((2 * SUPPORT_KEEP_PX + 1, 2 * SUPPORT_KEEP_PX + 1), np.uint8))
                m = m & (keep == 0)
            v.fg &= 1 - m
    views = {cam.name: v}
    T0 = pose_to_T(prior_pose[:3], prior_pose[3:])
    sm = _shape_for(obj_id, T0[:3, :3], tag)
    trunc = truncated_frac(cam, sm.asset.V @ T0[:3, :3].T + T0[:3, 3])
    Rz = T0[:3, :3] @ sm.R0.T
    yaw0 = float(np.arctan2(Rz[1, 0], Rz[0, 0]))
    z_off = float(T0[2, 3] - (TABLE_Z - sm.zmin))  # keep the prior's height (stacked targets)
    if trunc > TRUNCATED_MAX:
        # the object is larger than the wrist view (a 24-37 cm tower board under a 15 cm hover): a free fit slides
        # along the cut (5.1 build_tower: 83-122 mm "corrections" of the boards). Fit only what the view constrains:
        # the offset across the long axis and the yaw; the position along the long axis stays the prior's
        out = _partial_fit(sm, z_off, T0, yaw0, v)
        base = {"place_tag": sm.tag, "truncated_frac": round(trunc, 3), "camera": cam.name}
        if out is None or not out["ok"]:
            return {"pose": list(map(float, prior_pose)), "residual": 1.0, "iou": 0.0 if out is None else out["iou"],
                    "visible_frac": round(1 - trunc, 3), "moved_mm": 0.0, "turned_deg": 0.0, "ok": False,
                    "why": "truncated", "time_s": round(time.time() - t0, 3), **base}
        T = out["T"]
        return {"pose": np.r_[T[:3, 3], R_to_quat_wxyz(T[:3, :3])].tolist(), "residual": out["loss"], "iou": out["iou"],
                "visible_frac": round(1 - trunc, 3), "moved_mm": round(float(np.linalg.norm(T[:2, 3] - T0[:2, 3]) * 1000), 1),
                "turned_deg": round(float(np.degrees(abs(out["dyaw"]))), 1), "ok": True,
                "partial": {"unobserved_axis": out["along_axis"], "end_seen": out["end_seen"],
                            "note": "position along this axis kept from the prior (truncated view)"},
                "time_s": round(time.time() - t0, 3), **base}

    def pose_T(x, y, yaw):
        T = sm.pose(x, y, yaw)
        T[2, 3] += z_off
        return T

    starts = [(T0[0, 3], T0[1, 3], yaw0)]
    for dx, dy in (((xy_starts, 0), (-xy_starts, 0), (0, xy_starts), (0, -xy_starts)) if xy_starts > 0 else ()):
        starts.append((T0[0, 3] + dx, T0[1, 3] + dy, yaw0))
    if sm.yaw_period > 1.5 * np.pi:
        starts.append((T0[0, 3], T0[1, 3], yaw0 + np.pi))

    best = None
    for x, y, yaw in starts:
        # nm_refine poses objects on the table; a height offset is applied through a shifted shape copy
        x2, y2, yaw2, loss, per = nm_refine(_ZShift(sm, z_off), x, y, yaw, views, maxiter=iters,
                                            xy_step=0.004, yaw_step=0.08)
        if best is None or loss < best[3]:
            best = (x2, y2, yaw2, loss, per)
    x, y, yaw, loss, per = best
    T = pose_T(x, y, yaw)
    rvs = make_roi_views(sm, T, views)
    vis = 0.0
    if rvs:
        s = sm.sil(cam, T)
        vis = float((s & v.known).sum() / max(s.sum(), 1))
    iou = float(next(iter(per.values()), 0.0)) if per else 0.0
    other_pose, tried = None, []
    if iou < EMPTY_FIT_IOU and vis >= 0.5:
        # the object is in view but its prior's resting pose fits nothing: it may rest another way (a bottle lying where
        # the belief has it upright: conveyor 5.1, 8 of 9 wrist looks at iou 0.0). Try the asset's other resting shapes
        # at the same place, on the same support
        for sm2 in shape_models(obj_id):
            if sm2.tag == sm.tag:
                continue
            z2 = z_off                          # the same support under it: its own rest height on top of it
            b2 = None
            for yw in (yaw0, yaw0 + np.pi / 2):
                r2 = nm_refine(_ZShift(sm2, z2), T0[0, 3], T0[1, 3], yw, views, maxiter=iters, xy_step=0.004,
                               yaw_step=0.08)
                if b2 is None or r2[3] < b2[3]:
                    b2 = r2
            iou2 = float(next(iter(b2[4].values()), 0.0)) if b2[4] else 0.0
            tried.append((sm2.tag, round(iou2, 4)))
            if iou2 > max(iou, OTHER_POSE_MIN_IOU) and (other_pose is None or iou2 > other_pose["iou"]):
                other_pose = {"tag": sm2.tag, "iou": iou2, "fit": b2, "sm": sm2, "z_off": z2}
        if other_pose is not None:
            sm, z_off = other_pose["sm"], other_pose["z_off"]
            x, y, yaw, loss, per = other_pose["fit"]
            iou = other_pose["iou"]
            T = pose_T(x, y, yaw)
            s = sm.sil(cam, T)
            vis = float((s & v.known).sum() / max(s.sum(), 1))
    d_rot = np.degrees(np.arccos(np.clip((np.trace(T0[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
    ok = bool(iou >= 0.6 and vis >= 0.5)
    searched = None
    if not ok and search and seg == "local":
        searched = _wide_search(obj_id, sm, z_off, wrist_frame, cam, proprio, rob, T0, iters)
        if searched is not None and searched.get("ok"):
            x, y, yaw, loss, per, iou, vis = (searched[k] for k in ("x", "y", "yaw", "loss", "per", "iou", "vis"))
            T = pose_T(x, y, yaw)
            ok = True
            d_rot = np.degrees(np.arccos(np.clip((np.trace(T0[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
    slid = None
    if ok:  # a fit that slid out of the view: the IoU only counts the part still inside (5.1 tower: a plank on a
        # same-colour board slid 36-44 mm toward the image edge at iou 0.7-0.8)
        trunc_fit = truncated_frac(cam, sm.asset.V @ T[:3, :3].T + T[:3, 3])
        if trunc_fit > trunc + TRUNC_SLIDE_MAX:
            ok, slid = False, f"slid out of view ({trunc:.2f} -> {trunc_fit:.2f} outside)"
    edge = None
    stacked = bool(others) and any(_under(load_asset(k2), p2, load_asset(obj_id), prior_pose) for k2, p2 in others) \
        if seg == "local" else False
    if ok and stacked and edge_fallback:
        # on a same-colour support the colour fit can be ok and still wrong (5.1 tower: the plank on block7 10-40 mm
        # off at iou 0.8): a tight edge fit that disagrees by > 5 mm wins
        e2 = _edge_fit(sm, z_off, wrist_frame, cam, rob, T0, yaw0, iters)
        if e2 is not None and e2["ok"] and e2["cost_px"] <= EDGE_OVERRIDE_COST_PX and \
                np.hypot(e2["x"] - T[0, 3], e2["y"] - T[1, 3]) > 0.005:
            edge = e2
            T, x, y, yaw = e2["T"], e2["x"], e2["y"], e2["yaw"]
            loss, per = roi_loss(sm, T, make_roi_views(sm, T, views))
            iou = float(next(iter(per.values()), 0.0)) if per else 0.0
            d_rot = np.degrees(np.arccos(np.clip((np.trace(T0[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
    if not ok and seg == "local" and edge_fallback:
        # the colour model cannot separate the target from a same-colour support (5.1 tower: a white pillar on the
        # light-wood board, iou 0.03-0.2): fit the silhouette outline to image edges instead
        edge = _edge_fit(sm, z_off, wrist_frame, cam, rob, T0, yaw0, iters)
        if edge is not None and edge["ok"]:
            s_e = sm.sil(cam, edge["T"])
            vis_e = float((s_e & v.known).sum() / max(s_e.sum(), 1))
            if vis_e >= EDGE_MIN_VIS:
                T = edge["T"]
                x, y, yaw = edge["x"], edge["y"], edge["yaw"]
                loss, per = roi_loss(sm, T, make_roi_views(sm, T, views))
                iou = float(next(iter(per.values()), 0.0)) if per else 0.0
                vis = vis_e
                ok = True
                d_rot = np.degrees(np.arccos(np.clip((np.trace(T0[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
            else:  # "ok" with nothing in view (Memory, conveyor 100004 s260: ok at iou 0.0, vis 0.0, 18.8 mm)
                edge = dict(edge, ok=False, why=f"edge fit out of view (vis {vis_e:.2f})")
    yaw_read = None
    from .top_feature import read_yaw, top_feature

    if top_feature(sm) is not None:
        # a round outline with a cut in its top (insert_key's lock): the outline fit's yaw is arbitrary, the keyhole
        # seen from above gives it (axis + bulb side); at the fit when it is ok, else at the prior's place
        xr, yr = (x, y) if ok else (T0[0, 3], T0[1, 3])
        rob_m = rob if seg == "local" else v.robot
        yaw_read = read_yaw(wrist_frame, cam, sm, xr, yr, z_off, rob_m)
        if yaw_read is not None and yaw_read.get("ok"):
            yaw = yaw_read["yaw"]
            if yaw_read.get("sign_ok"):
                # the opening, both ends resolved, places the object too: 0.2-0.7 mm on 25 5.1 wrist hovers where the
                # outline fit of the round lock was 0.3-6.9 mm off
                x, y = yaw_read["x"], yaw_read["y"]
            if ok:
                T = pose_T(x, y, yaw)
            else:  # the rejected fit stays rejected; the prior keeps its place and takes the read yaw
                Tp = pose_T(T0[0, 3], T0[1, 3], yaw)
                Tp[2, 3] = T0[2, 3]
                prior_pose = np.r_[Tp[:3, 3], R_to_quat_wxyz(Tp[:3, :3])].tolist()
            d_rot = np.degrees(np.arccos(np.clip((np.trace(T0[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
    out = {"pose": np.r_[T[:3, 3], R_to_quat_wxyz(T[:3, :3])].tolist(), "residual": round(float(loss), 4),
            "iou": round(iou, 4), "visible_frac": round(vis, 3), "place_tag": sm.tag,
            "moved_mm": round(float(np.linalg.norm(T[:2, 3] - T0[:2, 3]) * 1000), 1), "turned_deg": round(float(d_rot), 1),
            "ok": ok, "time_s": round(time.time() - t0, 3), "camera": cam.name}
    if tried:
        out["other_poses_tried"] = tried
    if other_pose is not None:
        out["place_tag"] = sm.tag
        out["other_pose"] = {"tag": sm.tag, "iou": round(other_pose["iou"], 4)}
    if yaw_read is not None:
        out["yaw_read"] = yaw_read
        if yaw_read.get("ok"):
            out["yaw_source"] = "top_feature" if yaw_read.get("sign_ok") else "top_feature_axis"
    if slid and not ok:
        out["why"] = slid
    if edge is not None:
        out["edge"] = {k: edge[k] for k in ("ok", "cost_px", "coverage", "n_px", "moved_mm")}
        if edge["ok"]:
            out["via"] = "edges"
    # what the pixels look like at the fit and at the prior (identity of what is there: ``identity.rank``)
    for tag_, T_ in (("fit", T), ("prior", T0)):
        rgb, n_fg, n_sil = _obs_color(sm, T_, v)
        out[f"obs_{tag_}"] = {"rgb": None if rgb is None else [round(float(c), 4) for c in rgb], "fg_px": n_fg,
                              "sil_px": n_sil}
    if others and out["ok"]:
        from .render import load_asset

        a_t = load_asset(obj_id)
        for k2, p2 in others:  # a known object resting on the target covers its top: from above, only side
            if _rests_on(load_asset(k2), p2, a_t, prior_pose):  # strips are left (5.1 stack: middle block 8 mm off)
                out["ok"] = False
                out["why"] = f"covered:{k2}"
                break
    if others and out["ok"]:  # landed on a known neighbour: not this object (a large shift can grab one)
        from .render import load_asset

        fp = float(np.min(load_asset(obj_id).extents[:2]))
        for k2, p2 in others:
            try:
                fp2 = float(np.min(load_asset(k2).extents[:2]))
            except Exception:
                fp2 = fp
            h = 0.5 * min(float(load_asset(obj_id).extents[2]), float(load_asset(k2).extents[2]) if k2 else 1.0)
            if (np.hypot(out["pose"][0] - p2[0], out["pose"][1] - p2[1]) < 0.7 * min(fp, fp2)
                    and abs(out["pose"][2] - p2[2]) < h):  # same level only: the blocks of a stack share xy
                out["ok"] = False
                out["why"] = f"neighbour:{k2}"
                break
    if not out["ok"]:
        # a rejected fit never travels as the pose: on D0 wrist frames the rejected fits 40-110 mm away were mostly an
        # identical neighbour (pillars, blocks) at IoU 0.7-0.96; a caller that skips ``ok`` would adopt it
        out["fit_pose"], out["fit_moved_mm"] = out["pose"], out["moved_mm"]
        out["pose"], out["moved_mm"], out["turned_deg"] = list(map(float, prior_pose)), 0.0, 0.0
    if searched is not None and others:  # a suggestion on a known neighbour is that neighbour, not this object
        from .render import load_asset

        fp = float(np.min(load_asset(obj_id).extents[:2]))
        for k2, p2 in others:
            if np.hypot(searched["suggest_xy"][0] - p2[0], searched["suggest_xy"][1] - p2[1]) < 0.7 * fp:
                out["suggest_dropped"] = f"neighbour:{k2}"
                searched = None
                break
    if searched is not None:  # where the target's best match in the wrist frame is (re-hover there if not ok)
        out["searched"] = True
        out["suggest_xy"] = searched["suggest_xy"]
        out["suggest_shift_mm"] = round(float(np.hypot(searched["suggest_xy"][0] - T0[0, 3],
                                                       searched["suggest_xy"][1] - T0[1, 3]) * 1000), 1)
        out["suggest_score"] = searched["score"]
    return out


EMPTY_FIT_IOU = 0.2       # the prior's resting shape fitting this little of an object in view: try its other ones ...
OTHER_POSE_MIN_IOU = 0.6  # ... and take one that fits at least this well
EDGE_MIN_VIS = 0.05       # an edge fit counts only for an object in view at its pose (else it locked onto other
# edges: Memory's conveyor "ok" at vis 0.0; D0 val_refine's right edge fits of half-hidden blocks are all above this)
EDGE_MAX_COST_PX = 1.0  # mean outline-to-edge distance (px, truncated at EDGE_CAP_PX) of an accepted edge fit (5.1 tower
# + stack, priors displaced 15 / 40 mm: right fits 0.04-0.9 px; the one wrong fit, on a neighbouring pillar, 1.37 px)
EDGE_MIN_COVERAGE = 0.85
EDGE_OVERRIDE_COST_PX = 0.5  # an edge fit this tight overrides an ok colour fit it disagrees with  # share of the outline within 2 px of an image edge
EDGE_CAP_PX = 10.0
EDGE_MAX_SHIFT_M = 0.03


def image_edges(img: np.ndarray) -> np.ndarray:
    """Canny on lightness and on both chroma channels: a white pillar on a light-wood board differs in b*, not in
    grey level (grey-only Canny found no edge between them)."""
    lab = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2LAB), (5, 5), 0)
    E = cv2.Canny(lab[..., 0], 40, 120)
    for c in (1, 2):
        E |= cv2.Canny(lab[..., c], EDGE_CHROMA_LO, 3 * EDGE_CHROMA_LO)
    return E


EDGE_CHROMA_LO = 6


def _edge_fit(sm, z_off, img, cam, rob, T0, yaw0, iters: int = 80) -> dict | None:
    """Fit the silhouette OUTLINE to image edges (truncated chamfer), from the prior. Used when the colour
    foreground fails: a target on a same-colour support. Edges on the robot are ignored; outline pixels on the robot
    or the image border do not count. Accepted when the outline lies on edges (cost, coverage) within 30 mm."""
    from scipy.optimize import minimize

    E = image_edges(img)
    robd = cv2.dilate(rob.astype(np.uint8), np.ones((7, 7), np.uint8))
    E[robd > 0] = 0
    D = np.minimum(cv2.distanceTransform((E == 0).astype(np.uint8), cv2.DIST_L2, 3), EDGE_CAP_PX)
    h, w = D.shape
    border = np.zeros((h, w), np.uint8)
    border[3:-3, 3:-3] = 1
    k3 = np.ones((3, 3), np.uint8)

    def outline(x, y, yaw):
        T = sm.pose(x, y, yaw)
        T[2, 3] += z_off
        S = sm.sil(cam, T).astype(np.uint8)
        B = (S > 0) & (cv2.erode(S, k3) == 0) & (robd == 0) & (border > 0)
        return T, B

    def cost(p):
        _, B = outline(*p)
        n = int(B.sum())
        return EDGE_CAP_PX if n < 60 else float(D[B].mean())

    x0 = np.array([T0[0, 3], T0[1, 3], yaw0])
    best = None
    for dx, dy in ((0, 0), (0.008, 0), (-0.008, 0), (0, 0.008), (0, -0.008)):
        s0 = x0 + np.array([dx, dy, 0.0])
        simplex = [s0, s0 + [0.004, 0, 0], s0 + [0, 0.004, 0], s0 + [0, 0, 0.05]]
        r = minimize(cost, s0, method="Nelder-Mead",
                     options={"initial_simplex": np.array(simplex), "maxiter": iters, "xatol": 2e-4, "fatol": 1e-3})
        if best is None or r.fun < best.fun:
            best = r
    x, y, yaw = (float(v) for v in best.x)
    T, B = outline(x, y, yaw)
    n = int(B.sum())
    if n < 60:
        return None
    c = float(D[B].mean())
    cov = float((D[B] <= 2.0).mean())
    moved = float(np.hypot(x - T0[0, 3], y - T0[1, 3]))
    ok = bool(c <= EDGE_MAX_COST_PX and cov >= EDGE_MIN_COVERAGE and moved <= EDGE_MAX_SHIFT_M)
    return {"ok": ok, "T": T, "x": x, "y": y, "yaw": yaw, "cost_px": round(c, 2), "coverage": round(cov, 3), "n_px": n,
            "moved_mm": round(1000 * moved, 1)}


def _partial_fit(sm, z_off, T0, yaw0, v, s_lim: float = 0.03, iters: int = 60) -> dict | None:
    """A truncated object (longer than the view; box-like: tower boards): fit the offset ACROSS its long horizontal
    axis and the yaw; the position along the long axis stays the prior's. The box is clipped to the stretch of the
    long axis the camera sees (the far ends project behind it), and scored by IoU against the colour foreground in
    the image around its expected silhouette. ok at IoU >= 0.6."""
    from scipy.optimize import minimize

    from .render import silhouette

    V = sm.asset.V
    c0 = (V.max(0) + V.min(0)) / 2
    half = (V.max(0) - V.min(0)) / 2
    R = T0[:3, :3]
    horiz = [i for i in range(3) if abs(R[2, i]) < 0.7]
    if not horiz:
        return None
    il = max(horiz, key=lambda i: half[i])
    u = R[:2, il] / max(np.linalg.norm(R[:2, il]), 1e-9)
    n = np.array([-u[1], u[0]])
    cam = v.cam
    ts = np.arange(-half[il], half[il] + 1e-9, 0.01)
    P = np.array([T0[:3, :3] @ (c0 + np.eye(3)[il] * t) + T0[:3, 3] for t in ts])
    uv, z = cam.project(P)
    inside = (z > 0.05) & (uv[:, 0] > -40) & (uv[:, 0] < cam.w + 40) & (uv[:, 1] > -40) & (uv[:, 1] < cam.h + 40)
    if inside.sum() < 2:
        return None
    seen = (z > 0.05) & (uv[:, 0] > 10) & (uv[:, 0] < cam.w - 10) & (uv[:, 1] > 10) & (uv[:, 1] < cam.h - 10)
    end_seen = bool(seen.any() and (ts[seen].min() <= ts[0] + 0.011 or ts[seen].max() >= ts[-1] - 0.011))
    t0_, t1_ = ts[inside].min() - 0.02, ts[inside].max() + 0.02
    lo, hi = c0 - half, c0 + half
    lo[il], hi[il] = max(lo[il], c0[il] + t0_), min(hi[il], c0[il] + t1_)
    C = np.array([[x, y, zz] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for zz in (lo[2], hi[2])])
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    Rz0 = R.copy()

    def world(p):
        a = p[1]
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = Rz @ Rz0
        T[:3, 3] = T0[:3, 3] + np.r_[n * p[0] + (u * p[2] if len(p) > 2 else 0.0), 0.0]
        return T, C @ T[:3, :3].T + T[:3, 3]

    S0 = silhouette(cam, world(np.zeros(2))[1], F)
    if S0.sum() < 200:
        return None
    region = cv2.dilate(S0.astype(np.uint8), np.ones((61, 61), np.uint8)) & v.known
    fg = v.fg & region

    def iou(p):
        S = silhouette(cam, world(p)[1], F) & region
        inter = float((S & fg).sum())
        union = float(((S | fg) & region).sum())
        return inter / union if union > 0 else 0.0

    def f(p):
        pen = sum(10.0 * (abs(p[i]) - s_lim) for i in (0, 2) if i < len(p) and abs(p[i]) > s_lim)
        return 1.0 - iou(p) + pen

    # the position along the axis is NOT fitted even when an end is in view: the end strip is a small part of the
    # IoU and the fits stopped 10-20 mm off (5.1 tower, priors displaced 15 mm); the head view checks it
    x0 = np.zeros(2)
    r = minimize(f, x0, method="Nelder-Mead",
                 options={"initial_simplex": np.array([x0, [0.004, 0], [0, 0.03]]), "maxiter": iters,
                          "xatol": 2e-4, "fatol": 1e-4})
    T, _ = world(r.x)
    q = iou(r.x)
    return {"ok": bool(q >= 0.6 and abs(r.x[0]) <= s_lim), "T": T, "loss": round(1 - q, 4), "iou": round(q, 4),
            "dyaw": float(r.x[1]), "end_seen": end_seen, "along_axis": [round(float(u[0]), 4), round(float(u[1]), 4)]}


def _obs_color(sm, T: np.ndarray, v, erode: int = 2) -> tuple:
    """(mean RGB 0..1 | None, foreground px, silhouette px) inside the eroded silhouette of ``sm`` at ``T`` in the
    view: known foreground pixels only (others and the robot are masked out of the foreground already)."""
    s = sm.sil(v.cam, T).astype(np.uint8)
    if erode:
        s = cv2.erode(s, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    m = (s > 0) & (v.known > 0) & (v.fg > 0)
    n_sil = int(((s > 0) & (v.known > 0)).sum())
    if m.sum() < 10:
        return None, int(m.sum()), n_sil
    return v.img[m].astype(float).mean(0) / 255.0, int(m.sum()), n_sil


def _under(a_other, pose_other, a_target, pose_target, tol: float = 0.004) -> bool:
    """``other`` lies under ``target``: its top at or below the target's bottom (+ tol) and their footprints overlap."""
    To, Tt = pose_to_T(pose_other[:3], pose_other[3:]), pose_to_T(pose_target[:3], pose_target[3:])
    Wo = a_other.Vs @ To[:3, :3].T + To[:3, 3]
    Wt = a_target.Vs @ Tt[:3, :3].T + Tt[:3, 3]
    if Wo[:, 2].max() > Wt[:, 2].min() + tol:
        return False
    lo = np.maximum(Wo[:, :2].min(0), Wt[:, :2].min(0))
    hi = np.minimum(Wo[:, :2].max(0), Wt[:, :2].max(0))
    return bool((hi > lo).all())


SUPPORT_KEEP_PX = 15  # around the target's expected silhouette a support's pixels stay foreground (prior error margin)
SUPPORT_FLOAT_M = 0.008  # prior's bottom this far above the highest known support under it: floating
SUPPORT_GAIN_IOU = 0.15  # the support-height fit must beat the prior-height fit by this much to be taken


def support_z(asset, pose, others) -> float:
    """Top of the highest known support under ``pose``'s footprint (others' tops at or below its bottom + 6 mm,
    footprints overlapping by >= 30 %), else the table."""
    from .render import load_asset

    T = pose_to_T(pose[:3], pose[3:])
    W = asset.Vs @ T[:3, :3].T + T[:3, 3]
    lo = float(W[:, 2].min())
    top = TABLE_Z
    for k2, p2 in others:
        try:
            a2 = load_asset(k2)
        except Exception:
            continue
        T2 = pose_to_T(p2[:3], p2[3:])
        W2 = a2.Vs @ T2[:3, :3].T + T2[:3, 3]
        hi2 = float(W2[:, 2].max())
        if hi2 > lo + 0.006:
            continue
        a_lo = np.maximum(W[:, :2].min(0), W2[:, :2].min(0))
        a_hi = np.minimum(W[:, :2].max(0), W2[:, :2].max(0))
        inter = float(np.prod(np.clip(a_hi - a_lo, 0, None)))
        small = min(float(np.prod(np.ptp(W[:, :2], 0))), float(np.prod(np.ptp(W2[:, :2], 0))))
        if inter >= 0.3 * max(small, 1e-9):
            top = max(top, hi2)
    return top


def _rests_on(a_top, pose_top, a_base, pose_base, tol: float = 0.006, min_overlap: float = 0.3) -> bool:
    """``top`` rests on ``base``: its bottom within ``tol`` of the base's top and their footprints overlap by at least
    ``min_overlap`` of the smaller one (a block hovering in the jaws above the base does not count)."""
    Tt, Tb = pose_to_T(pose_top[:3], pose_top[3:]), pose_to_T(pose_base[:3], pose_base[3:])
    Wt = a_top.Vs @ Tt[:3, :3].T + Tt[:3, 3]
    Wb = a_base.Vs @ Tb[:3, :3].T + Tb[:3, 3]
    if abs(Wt[:, 2].min() - Wb[:, 2].max()) > tol:
        return False
    lo = np.maximum(Wt[:, :2].min(0), Wb[:, :2].min(0))
    hi = np.minimum(Wt[:, :2].max(0), Wb[:, :2].max(0))
    inter = float(np.prod(np.clip(hi - lo, 0, None)))
    small = min(float(np.prod(np.ptp(Wt[:, :2], 0))), float(np.prod(np.ptp(Wb[:, :2], 0))))
    return inter >= min_overlap * max(small, 1e-9)


TRUNC_SLIDE_MAX = 0.1  # a fit may leave at most this much more of the object outside the view than the prior did
TRUNCATED_MAX = 0.3  # more of the object outside the wrist image than this: translation along the cut is unobservable


def truncated_frac(cam, V_world: np.ndarray) -> float:
    """Share of the object's vertices that project outside the image (or behind the camera)."""
    uv, z = cam.project(V_world)
    inside = (z > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.w) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.h)
    return float(1.0 - inside.mean()) if len(uv) else 1.0


def _wide_search(obj_id, sm, z_off, img, cam, proprio, rob, T0, iters):
    """Whole-frame search for the target when the local refine fails: table colours from the full frame
    (k-means, robot excluded), foreground blobs scored by expected size and colour, best blob's ray on the
    plane at the object's mid-height -> suggested xy; a refine from there may succeed (then ``ok``)."""
    from .render import load_asset, robot_mask, silhouette, transform

    if rob is None:
        rob = robot_mask(cam, proprio, dilate_px=3)
    a = load_asset(obj_id)
    sil0 = silhouette(cam, transform(a.Vs, T0), a.Fs)
    exp_area = max(int(sil0.sum()), 30)
    fg = frame_foreground(img, rob)
    # grid over +-7 cm around the prior at the prior's yaw: silhouette IoU with the frame foreground, minus a
    # mild distance penalty (neighbours of the same look are the usual confusion)
    Rz = T0[:3, :3] @ sm.R0.T
    yaw0 = float(np.arctan2(Rz[1, 0], Rz[0, 0]))
    best = None
    for dx in np.arange(-0.07, 0.0701, 0.01):
        for dy in np.arange(-0.07, 0.0701, 0.01):
            Tg = sm.pose(T0[0, 3] + dx, T0[1, 3] + dy, yaw0)
            Tg[2, 3] += z_off
            s_g = sm.sil(cam, Tg)
            n_s = int(s_g.sum())
            if n_s < 0.25 * exp_area:
                continue
            known = s_g & (1 - rob)
            if known.sum() < 0.5 * n_s:
                continue
            inter = int((known & fg).sum())
            union = int(((s_g | fg) & (1 - rob) & cv2.dilate(s_g, np.ones((25, 25), np.uint8))).sum())
            score = inter / max(union, 1) - 1.0 * float(np.hypot(dx, dy))
            if best is None or score > best[0]:
                best = (score, (T0[0, 3] + dx, T0[1, 3] + dy))
    if best is None:
        return None
    X = np.array([best[1][0], best[1][1]])
    res = {"suggest_xy": [round(float(X[0]), 4), round(float(X[1]), 4)], "score": round(float(best[0]), 3)}
    T1 = sm.pose(X[0], X[1], yaw0)
    T1[2, 3] += z_off
    sil1 = silhouette(cam, transform(a.Vs, T1), a.Fs)
    fg1, kn1 = local_foreground(img, cam, sil1, rob)
    from .segment import ViewMasks

    v = ViewMasks(cam, img, fg1, kn1, np.zeros(fg1.shape, np.float32), rob)
    x, y, yaw, loss, per = nm_refine(_ZShift(sm, z_off), X[0], X[1], yaw0, {cam.name: v}, maxiter=iters, xy_step=0.004, yaw_step=0.08)
    T2 = sm.pose(x, y, yaw)
    T2[2, 3] += z_off
    s2 = sm.sil(cam, T2)
    vis = float((s2 & kn1).sum() / max(s2.sum(), 1))
    iou = float(next(iter(per.values()), 0.0)) if per else 0.0
    res.update({"x": x, "y": y, "yaw": yaw, "loss": loss, "per": per, "iou": iou, "vis": vis, "ok": bool(iou >= 0.6 and vis >= 0.5)})
    return res


def look_alike(current: str, rep: str, color_tol: float = 0.02, ext_tol: float = 0.003) -> str:
    """The catalogue asset with ``rep``'s shape (sorted extents within ``ext_tol``) and ``current``'s look (mean
    texture colour within ``color_tol``), else ``rep``. Shape classes are represented by one key each; deciding the
    SIZE at the hover must not swap the texture too (5.1 stack 100006: 35 mm texture #4 "corrected" to 40 mm texture
    #1)."""
    if rep == current:
        return rep
    try:
        idx = _catalog_index()
        cur, r = idx.get(current), idx.get(rep)
        if cur is None or r is None:
            return rep
        cat = rep.rsplit("/", 1)[0]
        e_r = np.sort(np.asarray(r["extents"], float))
        hits = [k for k, v in idx.items()
                if k.rsplit("/", 1)[0] == cat and np.abs(np.sort(np.asarray(v["extents"], float)) - e_r).max() <= ext_tol
                and np.abs(np.asarray(v["mean_color"], float) - np.asarray(cur["mean_color"], float)).max() <= color_tol]
        if not hits:
            return rep
        n = int(current.rsplit("/", 1)[1])  # the same series: catalogue neighbours of the current asset first
        return min(hits, key=lambda k: abs(int(k.rsplit("/", 1)[1]) - n))
    except Exception:
        return rep


_IDX: dict | None = None


def _catalog_index() -> dict:
    global _IDX
    if _IDX is None:
        import json

        from .render import catalog_root
        _IDX = json.loads((catalog_root() / "index.json").read_text())
    return _IDX


IDENTITY_CHANGE_MIN_IOU = 0.9  # a hover identity CHANGE needs a fit this good (D0 6.0 + 5.1 stack sizes: 115/115 right;
# below it 24 % wrong). 5.1 build_tower s0L9: a pillar (00002) became the prism (00003) at iou 0.67, turned 91 deg:
# from above their footprints are 38 x 62 vs 37.5 x 61 mm, and the pillar then went in standing on its 62 mm end


def refine_identity(obj_ids: list[str], wrist_frame: np.ndarray, proprio: dict, prior_pose, arm: str = "left",
                    margin: float = 0.05, max_full: int = 6, keep_look: bool = True,
                    change_min_iou: float = IDENTITY_CHANGE_MIN_IOU, **kw) -> dict:
    """Identity check at the hover: ``refine`` every candidate asset at the prior (e.g. the label group's
    candidates, or the record's ``label_rival``) and keep the lowest residual. Adds ``obj_id``,
    ``identity_scores`` {id: residual} and ``identity_margin`` (runner-up minus best residual);
    ``identity_ok`` is false when the margin is below ``margin`` (look again or keep the prior identity).
    ``keep_look``: the first id is the current belief; every other candidate is replaced by its same-shape asset
    with the current one's texture (``look_alike``), so a silhouette decision changes the size only."""
    ids = list(dict.fromkeys(obj_ids))
    if keep_look and len(ids) > 1:
        ids = list(dict.fromkeys([ids[0]] + [look_alike(ids[0], k) for k in ids[1:]]))
    current = ids[0]  # the belief's identity
    if len(ids) > max_full:  # screen with a short single-start fit, then the full refine on the best few
        quick = {k: refine(k, wrist_frame, proprio, prior_pose, arm=arm, **{**kw, "iters": 15, "xy_starts": 0.0,
                                                                            "search": False}) for k in ids}
        ids = [k for k, _ in sorted(quick.items(), key=lambda kv: kv[1]["residual"])[:max_full]]
        ids = ids if current in ids else [current] + ids[:max_full - 1]
    outs = {k: refine(k, wrist_frame, proprio, prior_pose, arm=arm, **kw) for k in ids}
    ranked = sorted(outs.items(), key=lambda kv: kv[1]["residual"])
    best_id, best = ranked[0]
    m = (ranked[1][1]["residual"] - best["residual"]) if len(ranked) > 1 else 1.0
    scores = {k: v["residual"] for k, v in outs.items()}
    if best_id != current and float(best.get("iou") or 0.0) < change_min_iou:
        # not enough to change what we believe: report the current identity's own fit (its pose, its ok)
        return {**outs[current], "obj_id": current, "identity_scores": scores, "identity_margin": round(float(m), 4),
                "identity_ok": False, "identity_rival": best_id,
                "identity_why": f"change to {best_id} needs iou >= {change_min_iou} (got {best.get('iou')})"}
    return {**best, "obj_id": best_id, "identity_scores": scores, "identity_margin": round(float(m), 4),
            "identity_ok": bool(m >= margin and best["ok"])}


class _ZShift:
    """A ShapeModel view whose poses are lifted by ``dz`` (keeps a stacked prior's height during refine)."""

    def __init__(self, sm, dz: float):
        self._sm, self._dz = sm, dz

    def __getattr__(self, k):
        return getattr(self._sm, k)

    def pose(self, x, y, yaw, z_off: float = 0.0):
        return self._sm.pose(x, y, yaw, z_off + self._dz)

    def sil(self, cam, T, out=None):
        return self._sm.sil(cam, T, out)

    def bbox_world(self, T):
        return self._sm.bbox_world(T)


def refine_record_object(rec_obj: dict, wrist_frame, proprio, arm: str, **kw) -> dict:
    """Convenience: refine a layout-record entry (``category``, ``category_idx``, ``default_pos/ori``)."""
    typ = rec_obj.get("type", "Rigid")
    key = f"{typ}/{rec_obj['category']}/{int(rec_obj['category_idx']):05d}"
    prior = list(rec_obj["default_pos"]) + list(rec_obj["default_ori"])
    tag = rec_obj.get("_perception", {}).get("place_tag")
    return refine(key, wrist_frame, proprio, prior, arm=arm, tag=tag if tag != "fixed" else None, **kw)

