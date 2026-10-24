"""Synthetic D0-like episodes for pipeline tests while Isaac data is unavailable.

Samples a layout from a task's config (labels, candidates, select mode, xlim/ylim, yaw, place tags),
rests each object on the table (z from contact), renders the catalogue's textured meshes with pyrender
over real home-pose background frames, with the home-pose robot as an occluder, and writes the D0
format (``frame_0.json`` with ``obs`` and ``gt``, ``frame_0_<cam>.png``). Lighting and shadows are not
Isaac's: accuracy here is an upper bound for the geometry, not a G1 number.

    python -m dojo.perception.synth --task push_T --n 10 --bg <dir with cam_*.png> --out <d0-like root>
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from .cameras import R_to_quat_wxyz, head_camera, wrist_camera
from .inverse_layout import _placement_prior, label_groups, shape_models
from .render import SceneRenderer, catalog_root, robot_items
from .segment import load_backgrounds

HOME = {"left_arm_joint_state": [0.0] * 6, "right_arm_joint_state": [0.0] * 6,
        "left_ee_joint_state": [1.0], "right_ee_joint_state": [1.0]}


def _ranges(lim):
    if lim is None:
        return [(-0.4, 0.4)]
    if isinstance(lim[0], list):
        return [tuple(x) for x in lim]
    return [tuple(lim)]


def sample_layout(task: str, rng: random.Random) -> list[dict]:
    objs = []
    for g in label_groups(task):
        prior = _placement_prior(g)
        if prior["relative_plane"] == "Ground":
            continue  # fixed ground geometry (dustbin) is not rendered here
        n = len(g["labels"])
        mode = g["select_mode"].get("mode", "unique")
        cands = list(g["candidates"])
        if mode == "same":
            keys = [rng.choice(cands)] * n
        elif mode == "hierarchical":
            cat = rng.choice(g["categories"])
            keys = [f"{g['type']}/{cat['name']}/{i:05d}" for i in rng.sample(cat["index"], n)]
        elif mode == "unique" and len(cands) >= n:
            keys = rng.sample(cands, n)
        else:
            keys = [rng.choice(cands) for _ in range(n)]
        p = g["placement"]
        for lab, key in zip(g["labels"], keys):
            sms = [s for s in shape_models(key) if not prior["place_tag"] or s.tag == prior["place_tag"]]
            sm = rng.choice(sms)
            rd = p.get("rotate_deg", 0) or 0
            if prior["yaw_fixed"]:
                yaw = 0.0
            elif isinstance(rd, list):
                yaw = np.deg2rad(rng.uniform(*rd))
            else:
                yaw = np.deg2rad(rng.uniform(-rd, rd)) if rd else rng.uniform(0, 2 * np.pi)
            r = 0.5 * float(np.linalg.norm(sm.asset.extents[:2]))
            for _ in range(200):
                if prior["fixed_xy"]:
                    x, y = prior["fixed_xy"]
                else:
                    xr = rng.choice(_ranges(p.get("xlim")))
                    x = rng.uniform(*xr)
                    y = rng.uniform(*_ranges(p.get("ylim"))[0])
                if all(np.hypot(x - o["x"], y - o["y"]) > r + o["r"] + 0.01 for o in objs) or prior["fixed_xy"]:
                    break
            objs.append({"label": lab, "key": key, "tag": sm.tag, "x": x, "y": y, "yaw": yaw, "r": r, "T": sm.pose(x, y, yaw)})
    return objs


def render_episode(objs, bgs, renderer: SceneRenderer, root: Path):
    cams = {"cam_head": head_camera(), "cam_left_wrist": wrist_camera("left", np.zeros(6)),
            "cam_right_wrist": wrist_camera("right", np.zeros(6))}
    rob = robot_items(HOME)
    images = {}
    for name, cam in cams.items():
        items = [dict(it, id=1) for it in rob]
        for i, o in enumerate(objs):
            d = root / o["key"]
            g = np.load(d / "geom.npz")
            items.append({"id": 10 + i, "key": o["key"], "V": g["vertices"], "F": g["faces"], "T": o["T"], "glb": d / "visual.glb"})
        _, ids, col = renderer.render(cam, items, rgb=True, ambient=0.5)
        img = bgs[name].copy()
        m = ids >= 10
        img[m] = col[m]
        images[name] = img
    return images


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--bg", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    bgs = load_backgrounds(a.bg)
    root = catalog_root()
    ren = SceneRenderer()
    for i in range(a.n):
        objs = sample_layout(a.task, rng)
        imgs = render_episode(objs, bgs, ren, root)
        d = a.out / a.task / f"{900000 + a.seed * 1000 + i}"
        d.mkdir(parents=True, exist_ok=True)
        for cam, img in imgs.items():
            cv2.imwrite(str(d / f"frame_0_{cam}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        gt = {"objects": [{"inst_name": f"{o['key'].split('/')[1]}_{int(o['key'].split('/')[2])}_{j}", "type": o["key"].split("/")[0],
                           "category": o["key"].split("/")[1], "category_idx": int(o["key"].split("/")[2]), "label": o["label"],
                           "clutter": False, "pos": o["T"][:3, 3].tolist(), "quat": R_to_quat_wxyz(o["T"][:3, :3]).tolist(),
                           "place_tag": o["tag"]} for j, o in enumerate(objs)]}
        rec = {"frame": 0, "obs": {"instruction": "", "state": HOME}, "gt": gt, "move": None, "synthetic": True}
        (d / "frame_0.json").write_text(json.dumps(rec))
        print(d, len(objs))


if __name__ == "__main__":
    main()
