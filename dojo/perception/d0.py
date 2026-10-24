"""Dataset D0 access (written by dojo.simlab.make_d0): ``<root>/<task>/<id>/frame_<k>.json`` + PNGs.

Firewall: ``load_obs`` returns only the official observation (images, state, instruction). Ground truth
is read only by ``load_gt``, which the estimator never imports; ``eval_g1`` is its only user.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

D0_DIRS = (os.environ.get("DOJO_D0", ""), "/root/dojo-data/d0", os.path.expanduser("~/dojo-perception-data/d0"))
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def d0_root() -> Path:
    for d in D0_DIRS:
        if d and Path(d).is_dir():
            return Path(d)
    raise FileNotFoundError("D0 not found; set DOJO_D0")


def episodes(root: Path | None = None, task: str | None = None) -> list[Path]:
    root = root or d0_root()
    tasks = [task] if task else sorted(p.name for p in root.iterdir() if p.is_dir())
    out = []
    for t in tasks:
        td = root / t
        if td.is_dir():
            out += sorted(p for p in td.iterdir() if p.is_dir() and (p / "frame_0.json").exists())
    return out


def frames(ep: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in ep.glob("frame_*.json"))


def _read_rgb(p: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def load_obs(ep: Path, k: int) -> dict:
    """The official observation of frame k: {"images": {cam: RGB}, "state": {...}, "instruction": str}."""
    rec = json.loads((ep / f"frame_{k}.json").read_text())
    obs = rec["obs"]
    images = {cam: _read_rgb(ep / f"frame_{k}_{cam}.png") for cam in CAMS if (ep / f"frame_{k}_{cam}.png").exists()}
    state = {key: np.asarray(v, dtype=float) for key, v in obs["state"].items()}
    # rec["move"] (the scripted move, which names its target object) is dev-harness data: not exposed.
    return {"images": images, "state": state, "instruction": obs.get("instruction", ""), "task": ep.parent.name,
            "episode": ep.name, "frame": k}


def load_gt(ep: Path, k: int) -> dict:
    """DEV-ONLY ground truth. Never call from the estimator."""
    return json.loads((ep / f"frame_{k}.json").read_text())["gt"]
