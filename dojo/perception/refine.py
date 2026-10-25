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
- ``others``: [(key, pose7)] of nearby known objects; their pixels are removed from the foreground.
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
from .inverse_layout import make_roi_views, refine as nm_refine, shape_models
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
                     k: int = 5, thr: float = 3.0):
    """Foreground near an expected object from the frame itself: a table colour model (k-means in Lab, with a
    per-cluster spread) sampled from a ring around the expected silhouette, then pixels inside the ROI far
    from every table cluster. Immune to the view-dependent sheen that breaks warped backgrounds at close
    range. Returns (fg, known) uint8 masks over the full image (known = the ROI minus the robot)."""
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
        m = [s for s in sms if s.tag == tag]
        if m:
            return m[0]
    up = R.T @ np.array([0, 0, 1.0])
    return max(sms, key=lambda s: float((s.R0.T @ np.array([0, 0, 1.0])) @ up))


def refine(obj_id: str, wrist_frame: np.ndarray, proprio: dict, prior_pose, arm: str = "left",
           tag: str | None = None, bgs: dict | None = None, others: list | None = None, iters: int = 60,
           xy_starts: float = 0.015, seg: str = "local", held=None, search: bool = True) -> dict:
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
        fgm, kn = local_foreground(wrist_frame, cam, sil, rob)
        v = ViewMasks(cam, wrist_frame, fgm, kn, np.zeros(fgm.shape, np.float32), rob)
    else:
        v = segment_virtual(wrist_frame, cam, bgs, state=proprio)
    if others:
        from .render import load_asset, silhouette, transform

        for key, pose in others:
            a = load_asset(key)
            m = silhouette(cam, transform(a.Vs, pose_to_T(pose[:3], pose[3:])), a.Fs)
            v.fg &= 1 - m
    views = {cam.name: v}
    T0 = pose_to_T(prior_pose[:3], prior_pose[3:])
    sm = _shape_for(obj_id, T0[:3, :3], tag)
    Rz = T0[:3, :3] @ sm.R0.T
    yaw0 = float(np.arctan2(Rz[1, 0], Rz[0, 0]))
    z_off = float(T0[2, 3] - (TABLE_Z - sm.zmin))  # keep the prior's height (stacked targets)

    def pose_T(x, y, yaw):
        T = sm.pose(x, y, yaw)
        T[2, 3] += z_off
        return T

    starts = [(T0[0, 3], T0[1, 3], yaw0)]
    for dx, dy in ((xy_starts, 0), (-xy_starts, 0), (0, xy_starts), (0, -xy_starts)):
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
    out = {"pose": np.r_[T[:3, 3], R_to_quat_wxyz(T[:3, :3])].tolist(), "residual": round(float(loss), 4),
            "iou": round(iou, 4), "visible_frac": round(vis, 3), "place_tag": sm.tag,
            "moved_mm": round(float(np.linalg.norm(T[:2, 3] - T0[:2, 3]) * 1000), 1), "turned_deg": round(float(d_rot), 1),
            "ok": ok, "time_s": round(time.time() - t0, 3), "camera": cam.name}
    if searched is not None:  # where the target's best match in the wrist frame is (re-hover there if not ok)
        out["searched"] = True
        out["suggest_xy"] = searched["suggest_xy"]
        out["suggest_shift_mm"] = round(float(np.hypot(searched["suggest_xy"][0] - T0[0, 3],
                                                       searched["suggest_xy"][1] - T0[1, 3]) * 1000), 1)
        out["suggest_score"] = searched["score"]
    return out


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


def refine_identity(obj_ids: list[str], wrist_frame: np.ndarray, proprio: dict, prior_pose, arm: str = "left",
                    margin: float = 0.05, **kw) -> dict:
    """Identity check at the hover: ``refine`` every candidate asset at the prior (e.g. the label group's
    candidates, or the record's ``label_rival``) and keep the lowest residual. Adds ``obj_id``,
    ``identity_scores`` {id: residual} and ``identity_margin`` (runner-up minus best residual);
    ``identity_ok`` is false when the margin is below ``margin`` (look again or keep the prior identity)."""
    outs = {k: refine(k, wrist_frame, proprio, prior_pose, arm=arm, **kw) for k in dict.fromkeys(obj_ids)}
    ranked = sorted(outs.items(), key=lambda kv: kv[1]["residual"])
    best_id, best = ranked[0]
    m = (ranked[1][1]["residual"] - best["residual"]) if len(ranked) > 1 else 1.0
    return {**best, "obj_id": best_id, "identity_scores": {k: v["residual"] for k, v in outs.items()},
            "identity_margin": round(float(m), 4), "identity_ok": bool(m >= margin and best["ok"])}


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

