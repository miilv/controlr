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


class FakeEpisode:
    def __init__(self, env, task="stack_blocks"):
        self.env, self.task = env, task

    def info(self):
        return {"stable": True, "task": self.task, "layout_id": 1, "instruction": self.env.instruction,
                "step_lim": self.env.step_lim, "reset_s": 0.0, "render_stage_s": 0.0}

    def observe(self):
        img = np.zeros((48, 64, 3), np.uint8)
        return {"images": {c: img for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
                "state": self.env.proprio(), "instruction": self.env.instruction, "steps_used": self.env.steps_used,
                "step_lim": self.env.step_lim, "renderer": "fake"}


def run_episode(policy, ep, max_calls=2000):
    policy.reset(ep.info())
    for _ in range(max_calls):
        d = policy.next(ep.observe())
        if d.get("done"):
            return policy.episode_report()
        for a in d["actions"]:
            if ep.env.step(a):
                return policy.episode_report()
    raise AssertionError("episode did not finish")
