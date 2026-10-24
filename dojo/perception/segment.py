"""Foreground segmentation against a static background, with the robot masked out from FK.

The eval scene's lighting, table, floor, room and head camera are fixed (no domain randomisation), and
both arms start at the same home pose. So a per-camera background image at the home pose is a strong
model: the per-pixel median of many initial frames of *different* layouts (objects move, the scene
does not). It is built offline from our own dev frames (observations only, no ground truth).

After the arms move, a wrist camera sees the table from a new pose: its background is synthesised by
warping the head background through the table-plane homography (pixels whose ray misses the table
or hits the home-pose robot in the head view are "unknown"). Pixels under the current robot (FK
silhouette, dilated) are unknown in every view.

Masks are uint8: ``fg`` 1 = foreground; ``known`` 1 = the pixel can be scored.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .cameras import TABLE_Z, Camera, head_camera
from .render import robot_mask

BG_DIRS = (os.environ.get("DOJO_BG", ""), "/root/dojo-data/perception/bg", os.path.expanduser("~/dojo-perception-data/bg"))
TABLE_HALF = (0.7, 0.55)  # table scale (1.4, 1.1) at default_pos (0, -0.05)
TABLE_CENTER = (0.0, -0.05)
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def bg_root() -> Path | None:
    for d in BG_DIRS:
        if d and Path(d, "cam_head.png").exists():
            return Path(d)
    return None


def build_background(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of RGB frames (uint8 HxWx3)."""
    st = np.stack(frames).astype(np.uint8)
    return np.median(st, axis=0).astype(np.uint8)


def load_backgrounds(root: Path | None = None) -> dict[str, np.ndarray]:
    root = root or bg_root()
    if root is None:
        return {}
    out = {}
    for c in CAMS:
        p = root / f"{c}.png"
        if p.exists():
            out[c] = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    return out


def table_homography_bg(head_bg: np.ndarray, head_cam: Camera, cam: Camera, z: float = TABLE_Z):
    """Warp the head background onto ``cam`` through the plane z. Returns (bg, valid) where valid marks
    pixels whose ray hits the table top inside its extent."""
    h, w = cam.h, cam.w
    vv, uu = np.mgrid[0:h, 0:w]
    uv = np.c_[uu.ravel() + 0.5, vv.ravel() + 0.5]
    X = cam.ray_plane(uv, z)
    inside = np.isfinite(X).all(axis=1)
    inside &= (np.abs(X[:, 0] - TABLE_CENTER[0]) < TABLE_HALF[0] - 0.01) & (np.abs(X[:, 1] - TABLE_CENTER[1]) < TABLE_HALF[1] - 0.01)
    Xs = np.where(inside[:, None], X, 0.0)
    huv, hz = head_cam.project(Xs)
    mapx = (huv[:, 0] - 0.5).reshape(h, w).astype(np.float32)
    mapy = (huv[:, 1] - 0.5).reshape(h, w).astype(np.float32)
    bg = cv2.remap(head_bg, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = inside.reshape(h, w) & (mapx >= 0) & (mapy >= 0) & (mapx < head_cam.w - 1) & (mapy < head_cam.h - 1)
    return bg, valid.astype(np.uint8)


def color_distance(img: np.ndarray, bg: np.ndarray, w_l: float = 0.5) -> np.ndarray:
    """Lab distance with lightness down-weighted (shadows change L more than a/b)."""
    a = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB).astype(np.float32)
    d = a - b
    return np.sqrt((w_l * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def shadow_like(img: np.ndarray, bg: np.ndarray, ratio=(0.35, 0.97), chroma_tol: float = 0.025) -> np.ndarray:
    """Pixels darker than the background with the same chromaticity (cast shadows). They are marked
    unknown, not background: a dark object side can look the same, and should not count against a fit."""
    a = img.astype(np.float32) + 1.0
    b = bg.astype(np.float32) + 1.0
    sa, sb = a.sum(-1), b.sum(-1)
    r = sa / sb
    dc = np.abs(a / sa[..., None] - b / sb[..., None]).max(-1)
    return ((r > ratio[0]) & (r < ratio[1]) & (dc < chroma_tol)).astype(np.uint8)


@dataclass
class ViewMasks:
    cam: Camera
    img: np.ndarray
    fg: np.ndarray  # uint8 0/1
    known: np.ndarray  # uint8 0/1
    dist: np.ndarray  # float32 colour distance to background
    robot: np.ndarray  # uint8 0/1 (current robot silhouette, dilated)


HOME = np.zeros(6)


def is_home(state: dict, tol: float = 0.02) -> dict[str, bool]:
    return {arm: (state.get(f"{arm}_arm_joint_state") is not None
                  and float(np.max(np.abs(np.asarray(state[f"{arm}_arm_joint_state"]) - HOME))) < tol)
            for arm in ("left", "right")}


def segment_views(images: dict[str, np.ndarray], cams: dict[str, Camera], state: dict,
                  bgs: dict[str, np.ndarray], thr: float = 18.0, robot_dilate: int = 3,
                  home_state: dict | None = None, shadow: bool = True) -> dict[str, ViewMasks]:
    """Foreground/known masks for every camera."""
    home = is_home(state)
    home_state = home_state or {"left_arm_joint_state": HOME, "right_arm_joint_state": HOME,
                                "left_ee_joint_state": [1.0], "right_ee_joint_state": [1.0]}
    out = {}
    head_cam = cams.get("cam_head") or head_camera()
    for name, img in images.items():
        cam = cams[name]
        rob = robot_mask(cam, state, dilate_px=robot_dilate)
        known = np.ones((cam.h, cam.w), np.uint8)
        arm = "left" if "left" in name else ("right" if "right" in name else None)
        bg = None
        if name == "cam_head" and name in bgs:
            bg = bgs[name]
            if not all(home.values()):  # the home-pose robot is baked into the head background
                known &= 1 - robot_mask(cam, home_state, dilate_px=robot_dilate)
        elif arm is not None and home[arm] and name in bgs:
            bg = bgs[name]
        elif "cam_head" in bgs:
            bg, valid = table_homography_bg(bgs["cam_head"], head_cam, cam)
            hb = robot_mask(head_cam, home_state, dilate_px=robot_dilate)
            # pixels whose table point is hidden by the home robot in the head background are unknown
            h, w = cam.h, cam.w
            vv, uu = np.mgrid[0:h, 0:w]
            X = cam.ray_plane(np.c_[uu.ravel() + 0.5, vv.ravel() + 0.5], TABLE_Z)
            huv, _ = head_cam.project(np.nan_to_num(X))
            hu = np.clip(huv[:, 0].astype(int), 0, head_cam.w - 1)
            hv = np.clip(huv[:, 1].astype(int), 0, head_cam.h - 1)
            known &= valid & (1 - hb[hv, hu].reshape(h, w))
        if bg is None:
            dist = np.zeros((cam.h, cam.w), np.float32)
            fg = np.zeros((cam.h, cam.w), np.uint8)
            known[:] = 0
        else:
            dist = color_distance(img, bg)
            fg = (dist > thr).astype(np.uint8)
            if shadow:
                sh = shadow_like(img, bg)
                fg &= 1 - sh
                known &= 1 - sh
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        known &= 1 - rob
        fg &= known
        out[name] = ViewMasks(cam, img, fg, known, dist.astype(np.float32), rob)
    return out


def main(argv=None):
    """Build per-camera home-pose backgrounds from initial frames of many D0 layouts (observations only).

        python -m dojo.perception.segment --d0 /root/dojo-data/d0 --out /root/dojo-data/perception/bg [--max 80]
    """
    import argparse
    import random

    from . import d0 as d0mod

    ap = argparse.ArgumentParser()
    ap.add_argument("--d0", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max", type=int, default=80)
    ap.add_argument("--exclude-episodes", nargs="*", default=[], help="held-out episodes not to use")
    a = ap.parse_args(argv)
    eps = [e for e in d0mod.episodes(a.d0) if e.name not in set(a.exclude_episodes)]
    random.Random(0).shuffle(eps)
    eps = eps[: a.max]
    a.out.mkdir(parents=True, exist_ok=True)
    for cam in CAMS:
        frames = []
        for e in eps:
            p = e / f"frame_0_{cam}.png"
            if p.exists():
                frames.append(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
        if frames:
            bg = build_background(frames)
            cv2.imwrite(str(a.out / f"{cam}.png"), cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
            print(cam, len(frames), "frames")
    (a.out / "sources.json").write_text(__import__("json").dumps([str(e) for e in eps], indent=0))


if __name__ == "__main__":
    main()
