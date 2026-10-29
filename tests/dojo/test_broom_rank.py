"""Stream H: ``RuntimeConfig.broom_rank`` / ``Skills.broom_rank`` ("com" = today | "hold": ``sweep.handle_grasps`` ranks
by ``grasps.hold_quality``). The effect is screened in G3; here: the ranking key, the default, plumbing. CPU only."""

from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import grasps as G
from dojo.skills import sweep
from dojo.skills.world import AssetInfo


def _grasp(z_off, score=0.5):
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])     # straight down (x = -z)
    T[2, 3] = z_off
    return G.Grasp("box", T, 0.03, score=score, note=f"box z {z_off}")


@pytest.mark.parametrize("mode, first", [(None, 0.02), ("com", 0.02), ("hold", 0.0)])
def test_rank_by_hold_quality_or_com(monkeypatch, mode, first):
    tool = SimpleNamespace(T=np.eye(4), asset=SimpleNamespace(bbox_min=np.array([-0.1, -0.1, -0.1]),
                                                               bbox_max=np.array([0.1, 0.1, 0.1])))
    gs = [_grasp(0.0), _grasp(0.02)]
    # the head along +z: every grasp is on the handle side (below +0.025 m past the middle)
    monkeypatch.setattr(sweep, "tool_head", lambda t: {"axis": 2, "sign": 1.0})
    monkeypatch.setattr(G, "box_grasps", lambda *a, **k: [_grasp(0.0), _grasp(0.02)])
    monkeypatch.setattr(G, "slice_grasps", lambda *a, **k: [])
    monkeypatch.setattr(G, "antipodal_grasps", lambda *a, **k: [])
    import dojo.skills.library as L
    monkeypatch.setattr(L, "physics_com", lambda t: np.array([0.0, 0.0, 0.02]))        # the CoM at the +20 mm station
    monkeypatch.setattr(G, "hold_quality", lambda t, T: 1.0 + 100 * abs(float(T[2, 3])))  # the centred pinch holds
    sk = SimpleNamespace(kin=None)
    if mode:
        sk.broom_rank = mode
    out = sweep.handle_grasps(sk, tool, "left")
    assert len(out) == len(gs) and float(out[0].T[2, 3]) == pytest.approx(first)


class _Assets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.015), bbox_max=np.full(3, 0.015))


def test_default_validation_and_plumbing():
    assert RuntimeConfig().broom_rank == "com"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", broom_rank="patch"), assets=_Assets())
    from dojo.skills.env import KinematicEnv
    env = KinematicEnv([{"label": "b", "category": "block", "category_idx": 5, "pos": (0.3, -0.1, 0.78),
                         "quat": [1, 0, 0, 0], "half": [0.015] * 3}], step_lim=50, instruction="Stack the blocks.")
    pol = RuntimePolicy(RuntimeConfig(mode="gt", verify="off", broom_rank="hold"), assets=_Assets())
    pol.reset({"task": "x", "instruction": env.instruction, "step_lim": 50, "layout_id": 1})
    obs = {"images": {c: np.zeros((8, 8, 3), np.uint8) for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
           "state": env.proprio(), "instruction": env.instruction, "steps_used": 0, "step_lim": 50,
           "_dev_gt": env.gt_objects()}
    pol.next(obs)
    assert pol._live["sk"].broom_rank == "hold"
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["broom_rank"][0] == RuntimeConfig().broom_rank
