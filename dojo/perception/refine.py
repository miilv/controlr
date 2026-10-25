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
- Returns also ``iou``, ``visible_frac`` (share of the silhouette not under the robot / out of view),
  ``moved_mm`` / ``turned_deg`` vs the prior, ``ok`` (iou >= 0.6 and visible_frac >= 0.5), ``time_s``.
"""
from __future__ import annotations

import time

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
           xy_starts: float = 0.015, seg: str = "local") -> dict:
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
    return {"pose": np.r_[T[:3, 3], R_to_quat_wxyz(T[:3, :3])].tolist(), "residual": round(float(loss), 4),
            "iou": round(iou, 4), "visible_frac": round(vis, 3), "place_tag": sm.tag,
            "moved_mm": round(float(np.linalg.norm(T[:2, 3] - T0[:2, 3]) * 1000), 1), "turned_deg": round(float(d_rot), 1),
            "ok": bool(iou >= 0.6 and vis >= 0.5), "time_s": round(time.time() - t0, 3), "camera": cam.name}


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

