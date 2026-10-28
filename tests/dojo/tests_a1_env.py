"""Helpers for test_a1 (no import from ``tests.``: CI collects without the repo root on sys.path)."""
import numpy as np

from dojo.skills.env import KinematicEnv
from dojo.skills.world import AssetInfo

STACK = "Stack the three blocks with different textures."
POS = [(0.412, -0.09, 0.78), (-0.26, -0.189, 0.78), (-0.344, -0.1, 0.78)]


def env_and_obs():
    rows = [{"label": f"block_{i}", "category": "block", "category_idx": 5, "pos": p, "quat": [1, 0, 0, 0],
             "half": [0.015] * 3} for i, p in enumerate(POS)]
    env = KinematicEnv(rows, step_lim=550, instruction=STACK)
    img = np.zeros((48, 64, 3), np.uint8)
    obs = {"images": {c: img for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")}, "state": env.proprio(),
           "instruction": STACK, "steps_used": 0, "step_lim": 550, "renderer": "fake"}
    return env, obs


class FakeAssets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        h = np.full(3, 0.015)
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=-h, bbox_max=h)
