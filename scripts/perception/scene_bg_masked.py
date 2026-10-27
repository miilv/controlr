"""A scene background from frame 0 of dev layouts whose fixtures move from layout to layout (organize_table's drawer,
monitor, mouse mat): the per-pixel median with every labelled object masked out (its GT silhouette, dilated: dev
layouts only, built offline), so the table keeps the scene's own shading (big white fixtures darken and warm the wood
next to them, which the table background reads as foreground) while every object stays foreground. Written like
belt_bg's (``<dest>/scene/<Type_cat_idx>/``) so ``segment.with_scene`` picks it for that scene.

    python scripts/perception/scene_bg_masked.py --root ROOT/<task> --key Geometry/cube_cushion/00002 --dest BG/fold0 \\
        [--dest ...] [--ids ...] [--frame W0000] [--dilate 6]
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--dest", action="append", required=True)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--frame", default="W0000")
    ap.add_argument("--dilate", type=int, default=6)
    a = ap.parse_args(argv)
    from dojo.perception import d0
    from dojo.perception.cameras import cameras_from_state, pose_to_T
    from dojo.perception.inverse_layout import shape_models
    from dojo.perception.segment import SCENE_DIR, load_backgrounds, masked_median, scene_dir_name

    from dojo.perception.render import robot_mask

    frames = {c: [] for c in CAMS}
    masks = {c: [] for c in CAMS}
    clean_f, clean_m = [], []      # the robot-free head view (cam_head_clean): frame 0 + scan-pose frames, robot masked
    used = []
    k = np.ones((2 * a.dilate + 1, 2 * a.dilate + 1), np.uint8)
    for d in sorted(p for p in Path(a.root).iterdir() if p.is_dir() and p.name.isdigit()):
        if int(d.name) < 100000 or (a.ids and d.name not in a.ids) or not (d / f"frame_{a.frame}.json").exists():
            continue
        for tag in [a.frame] + sorted(p.stem[len("frame_"):] for p in d.glob("frame_A*.json")):
            if not (d / f"frame_{tag}_cam_head.png").exists():
                continue
            ob = d0.load_obs(d, tag)
            hc = cameras_from_state(ob["state"])["cam_head"]
            m = robot_mask(hc, ob["state"], dilate_px=4)
            for o in d0.load_gt(d, tag)["objects"]:
                if o.get("type") in ("Rigid", "Geometry", "Articulation") and o.get("category") != "camera_stand":
                    try:
                        shape_models(f"{o['type']}/{o['category']}/{int(o['category_idx']):05d}")[0].sil(
                            hc, pose_to_T(np.asarray(o["pos"]), np.asarray(o["quat"])), out=m)
                    except Exception:  # noqa: BLE001
                        pass
            clean_f.append(ob["images"]["cam_head"])
            clean_m.append(cv2.dilate(m, k))
        obs = d0.load_obs(d, a.frame)
        gt = d0.load_gt(d, a.frame)
        cams = cameras_from_state(obs["state"])
        for c in CAMS:
            if c not in obs["images"]:
                continue
            m = np.zeros(obs["images"][c].shape[:2], np.uint8)
            for o in gt["objects"]:
                if o.get("type") not in ("Rigid", "Geometry", "Articulation") or o.get("category") == "camera_stand":
                    continue
                try:
                    sm = shape_models(f"{o['type']}/{o['category']}/{int(o['category_idx']):05d}")[0]
                except Exception:  # noqa: BLE001
                    continue
                sm.sil(cams[c], pose_to_T(np.asarray(o["pos"]), np.asarray(o["quat"])), out=m)
            frames[c].append(obs["images"][c])
            masks[c].append(cv2.dilate(m, k))
        used.append(d.name)
    if not frames["cam_head"]:
        raise SystemExit("no frames")
    for dest in a.dest:
        fb = load_backgrounds(Path(dest))
        o = Path(dest) / SCENE_DIR / scene_dir_name(a.key)
        o.mkdir(parents=True, exist_ok=True)
        for c in CAMS:
            if frames[c] and c in fb:
                bg, unseen = masked_median(frames[c], masks[c], fb[c])
                cv2.imwrite(str(o / f"{c}.png"), cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
        if clean_f and "cam_head_clean" in fb:
            bg, unseen = masked_median(clean_f, clean_m, fb["cam_head_clean"])
            cv2.imwrite(str(o / "cam_head_clean.png"), cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(o / "cam_head_clean_unseen.png"), (unseen * 255).astype(np.uint8))
        meta = {"key": a.key, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "layouts": used,
                "frame": a.frame, "clean_frames": len(clean_f), "how": "masked median of frame 0, labelled objects masked by their "
                "dev-layout GT silhouettes; cam_head_clean over frame 0 + scan-pose frames with the robot (FK) masked too "
                "(scripts/perception/scene_bg_masked.py)", "source": "our dev layouts (>= 100000), never eval data"}
        (o / "provenance.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"layouts": used}))


if __name__ == "__main__":
    main()
