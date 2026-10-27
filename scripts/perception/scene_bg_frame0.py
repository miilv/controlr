"""A scene background from frame 0 of dev layouts: the scene's fixtures and a scripted actor at its home pose (the
imitate demonstrator's arm), our arms at home. Per camera the per-pixel median over the layouts' frame-0 images; with
``belt_bg record`` frames at perception's scan pose (``frame_A*``, ``--away-frames``) also the robot-free head view
(``cam_head_clean`` + ``_unseen``: masked median, FK robot masks, as ``belt_bg.build``). Written like belt_bg's
(``<dest>/scene/<Type_cat_idx>/``, ``provenance.json``), so ``segment.with_scene`` picks it for a scene with that asset.

    python scripts/perception/scene_bg_frame0.py --key Geometry/basket/00005 --dest $BG/fold0 --dest $BG/fold1 \\
        --root ~/dojo-data/perception3/imibg51/imitate_sorting_sequence [--root ...] [--ids ...] [--frame W0000]

Dev layouts only (>= 100000, never eval data).
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _rgb(p: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def actor_mask(scene: np.ndarray, table: np.ndarray, thr: float = 25.0, dilate: int = 4) -> np.ndarray:
    """Head pixels a scripted actor covers at its home pose (the imitate demonstrator's arm): where the scene's
    frame-0 median differs from the table background, the connected parts that touch the image's top border (the arm
    reaches in from above; the baskets and objects do not), dilated. Once the actor moves those pixels show the table
    and read foreground against the scene background: departure reads use the scene with these pixels from the table
    (``cam_head_noactor``)."""
    from dojo.perception.segment import color_distance

    d = (color_distance(scene, table) > thr).astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
    top = {int(v) for v in np.unique(lab[0]) if v}
    m = np.isin(lab, list(top)).astype(np.uint8)
    return cv2.dilate(m, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--dest", action="append", required=True)
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--frame", default="W0000")
    ap.add_argument("--actor-thr", type=float, default=25.0)
    a = ap.parse_args(argv)
    from dojo.perception.cameras import head_camera
    from dojo.perception.render import robot_mask
    from dojo.perception.segment import SCENE_DIR, build_background, masked_median, scene_dir_name

    home = {c: [] for c in CAMS}
    clean_f, clean_m, used = [], [], []
    hc = head_camera()
    for root in a.root:
        for d in sorted(p for p in Path(root).iterdir() if p.is_dir() and p.name.isdigit()):
            if int(d.name) < 100000 or (a.ids and d.name not in a.ids):
                continue
            if not all((d / f"frame_{a.frame}_{c}.png").exists() for c in CAMS):
                continue
            for c in CAMS:
                home[c].append(_rgb(d / f"frame_{a.frame}_{c}.png"))
            n_away = 0
            for tag in [a.frame] + sorted(p.stem[len("frame_"):] for p in d.glob("frame_A*.json")):
                if not (d / f"frame_{tag}_cam_head.png").exists():
                    continue
                st = {k: np.asarray(v, float) for k, v in json.loads((d / f"frame_{tag}.json").read_text())["obs"]["state"].items()}
                clean_f.append(_rgb(d / f"frame_{tag}_cam_head.png"))
                clean_m.append(robot_mask(hc, st, dilate_px=4))
                n_away += tag != a.frame
            used.append({"layout": d.name, "root": str(root), "away_frames": n_away})
    if not home["cam_head"]:
        raise SystemExit("no frame-0 images")
    out = {c: build_background(v) for c, v in home.items()}
    unseen = None
    if any(u["away_frames"] for u in used):
        out["cam_head_clean"], unseen = masked_median(clean_f, clean_m, out["cam_head"])
    meta = {"key": a.key, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "layouts": used,
            "frame": a.frame, "home_frames": len(home["cam_head"]), "clean_frames": len(clean_f) if unseen is not None else 0,
            "unseen_px": int(unseen.sum()) if unseen is not None else None,
            "how": "per-pixel median of frame 0; clean = masked median over frame 0 + scan-pose frames "
                   "(scripts/perception/scene_bg_frame0.py)",
            "source": "our dev layouts (>= 100000), never eval data"}
    for dest in a.dest:
        o = Path(dest) / SCENE_DIR / scene_dir_name(a.key)
        o.mkdir(parents=True, exist_ok=True)
        for c, v in out.items():
            cv2.imwrite(str(o / f"{c}.png"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))
        if unseen is not None:
            cv2.imwrite(str(o / "cam_head_clean_unseen.png"), (unseen * 255).astype(np.uint8))
        table = Path(dest) / "cam_head.png"
        if table.exists():
            # the scene without its actor: the actor's home pixels from the fold's table background (departure reads)
            tb = _rgb(table)
            am = actor_mask(out["cam_head"], tb, a.actor_thr).astype(bool)
            na = out["cam_head"].copy()
            na[am] = tb[am]
            cv2.imwrite(str(o / "cam_head_noactor.png"), cv2.cvtColor(na, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(o / "cam_head_actor_mask.png"), (am * 255).astype(np.uint8))
            meta["actor_px"] = int(am.sum())
            (o / "provenance.json").write_text(json.dumps(meta, indent=1))
        (o / "provenance.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "layouts"}))


if __name__ == "__main__":
    main()
