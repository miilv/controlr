"""XPolicyLab model "dojo_replay": serves a recorded trajectory segment by segment (like
``dojo.simlab.proxy_eval.ReplayPolicy``), then an empty chunk. Runs in RoboDojo's policy server env
(numpy only). Trajectory: ``$DOJO_REPLAY_TRAJ`` (or ``traj`` in deploy.yml), ``{"actions": [...],
"segments": [{"steps": n}, ...]}`` with XPolicyLab joint-action dicts."""

from __future__ import annotations

import json
import os

import numpy as np

try:
    from XPolicyLab.model_template import ModelTemplate
except ImportError:          # unit tests (no XPolicyLab)
    ModelTemplate = object


def load_traj(path: str) -> tuple[list[dict], list[int]]:
    with open(path) as f:
        d = json.load(f)
    actions = d["actions"]
    segs = [int(s["steps"]) for s in (d.get("segments") or [{"steps": len(actions)}])]
    return actions, segs


class Model(ModelTemplate):
    def __init__(self, model_cfg=None):
        self.model_cfg = dict(model_cfg or {})
        path = os.environ.get("DOJO_REPLAY_TRAJ") or self.model_cfg.get("traj")
        if not path:
            raise RuntimeError("dojo_replay: set DOJO_REPLAY_TRAJ to a trajectory JSON")
        self.actions, self.segments = load_traj(path)
        self.i = self.k = 0
        print(f"[dojo_replay] {path}: {len(self.actions)} actions in {len(self.segments)} segments", flush=True)

    def reset(self):
        self.i = self.k = 0

    def update_obs(self, obs):
        del obs

    def update_obs_batch(self, obs_list):
        del obs_list

    def get_action(self):
        """The next recorded segment (list of action dicts, float32 arrays); [] once the trajectory is over."""
        if self.k >= len(self.segments) or self.i >= len(self.actions):
            return []
        n = self.segments[self.k]
        chunk = self.actions[self.i:self.i + n]
        self.i, self.k = self.i + n, self.k + 1
        return [{key: np.asarray(v, np.float32) for key, v in a.items()} for a in chunk]

    def get_action_batch(self, env_idx_list=None):
        raise RuntimeError("dojo_replay runs one env (eval_batch: false)")
