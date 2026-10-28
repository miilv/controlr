"""Stream H: ``RuntimeConfig.nest_height`` / ``Skills.nest_height`` ("bins" = today | "measured": ``bowl_geometry``'s
nest offset from the twin's per-asset table, else the mesh's radial profile). Effect screened in G3; here: the table,
the profile on a synthetic bowl, the default, plumbing. CPU only."""

import math

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills.library import bowl_geometry, bowl_nest_measured, nest_profile_dz
from dojo.skills.world import AssetInfo


def _cone_bowl(key, r_bot=0.03, r_top=0.07, h=0.06, wall=0.004, n=48):
    """A conical shell: outer surface r_bot..r_top over 0..h, the inner one ``wall`` inside it, a flat floor."""
    V, F = [], []
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)

    def ring(r, z):
        base = len(V)
        V.extend([(r * math.cos(t), r * math.sin(t), z) for t in ang])
        return base

    def band(a, b):
        for i in range(n):
            j = (i + 1) % n
            F.append((a + i, a + j, b + j))
            F.append((a + i, b + j, b + i))

    o0, o1 = ring(r_bot, 0.0), ring(r_top, h)
    i0, i1 = ring(r_bot - wall, wall), ring(r_top - wall, h)
    band(o0, o1)
    band(i0, i1)
    band(o1, i1)                                                   # the rim
    c_o, c_i = len(V), len(V) + 1
    V.extend([(0.0, 0.0, 0.0), (0.0, 0.0, wall)])
    for i in range(n):
        j = (i + 1) % n
        F.append((c_o, o0 + j, o0 + i))
        F.append((c_i, i0 + i, i0 + j))
    V = np.asarray(V, float)
    V[:, 2] -= h / 2
    return AssetInfo(key=key, bbox_min=V.min(0), bbox_max=V.max(0), mesh=(V, np.asarray(F, int)))


def test_profile_of_a_cone_is_where_the_walls_meet():
    a = _cone_bowl("Rigid/bowl/09999")
    # the copy's bottom edge (r 0.030) meets the inner cone r(z) = 0.026 + (0.040 / 0.056)(z - 0.004) at z = 9.6 mm
    # (the wall above it clears); the 1 mm height bins round it up
    dz = nest_profile_dz(a)
    assert dz is not None and 0.0095 <= dz <= 0.012


def test_measured_table_wins_then_the_profile_then_bins():
    assert bowl_nest_measured("Rigid/bowl/00002") == pytest.approx(0.0232)
    assert bowl_nest_measured("Rigid/bowl/99999") is None
    a = _cone_bowl("Rigid/bowl/09998")
    o = type("O", (), {"asset": a})()
    bins = bowl_geometry(o)["nest_dz"]
    assert bowl_geometry(o, nest="bins")["nest_dz"] == bins
    assert bowl_geometry(o, nest="measured")["nest_dz"] == pytest.approx(nest_profile_dz(a))
    b = _cone_bowl("Rigid/bowl/00002")                              # a key in the table: the twin's number
    assert bowl_geometry(type("O", (), {"asset": b})(), nest="measured")["nest_dz"] == pytest.approx(0.0232)


class _Assets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.015), bbox_max=np.full(3, 0.015))


def test_default_validation_and_plumbing():
    assert RuntimeConfig().nest_height == "bins"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", nest_height="cad"), assets=_Assets())
    from dojo.skills.env import KinematicEnv
    env = KinematicEnv([{"label": "b", "category": "block", "category_idx": 5, "pos": (0.3, -0.1, 0.78),
                         "quat": [1, 0, 0, 0], "half": [0.015] * 3}], step_lim=50, instruction="Stack the blocks.")
    pol = RuntimePolicy(RuntimeConfig(mode="gt", verify="off", nest_height="measured"), assets=_Assets())
    pol.reset({"task": "x", "instruction": env.instruction, "step_lim": 50, "layout_id": 1})
    obs = {"images": {c: np.zeros((8, 8, 3), np.uint8) for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
           "state": env.proprio(), "instruction": env.instruction, "steps_used": 0, "step_lim": 50,
           "_dev_gt": env.gt_objects()}
    pol.next(obs)
    assert pol._live["sk"].nest_height == "measured"
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["nest_height"][0] == RuntimeConfig().nest_height
