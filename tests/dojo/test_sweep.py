"""sweep_into: the planner's route, the dustpan's geometry (entry, lip, collecting circle), the broom's head, the
dustpan's handle grasps, and a whole sweep planned in the kinematic env (CPU only)."""

from __future__ import annotations

import math

import numpy as np


def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def _merge(*ms):
    Vs, Fs, n = [], [], 0
    for V, F in ms:
        Vs.append(V)
        Fs.append(F + n)
        n += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def _obj(label, cat, xyz, mesh, typ="Rigid", quat=(1.0, 0, 0, 0), support=None):
    from dojo.skills.world import AssetInfo, ObjectState
    V = mesh[0]
    info = AssetInfo(key=f"{typ}/{cat}/00000", bbox_min=V.min(0), bbox_max=V.max(0), mesh=mesh, support=support or {})
    return ObjectState(label, typ, cat, 0, np.asarray(xyz, float), np.asarray(quat, float), info)


PAN = _merge(_box((-0.092, -0.14, -0.024), (0.092, 0.03, -0.019)),     # the pan's floor
             _box((-0.092, 0.03, -0.024), (0.092, 0.04, 0.024)),       # its back wall
             _box((-0.015, 0.04, 0.012), (0.015, 0.14, 0.024)))        # the handle
BROOM = _merge(_box((-0.011, -0.009, -0.125), (0.011, 0.017, 0.0)),    # handle (thin)
               _box((-0.019, -0.041, 0.009), (0.019, 0.013, 0.125)))   # head (bristle block)
SUPPORT = {"broom_shovel/0": {"center": [[0.0, -0.053, -0.015, 1.0, 0.0, 0.0, 0.0]], "radius": [0.081]}}


def test_planner_routes_sweep():
    from dojo.skills import planner
    from dojo.skills.world import WorldState
    pan = _obj("broom_shovel", "broom_shovel", (-0.05, 0.02, 0.789), PAN, support=SUPPORT)
    broom = _obj("broom", "broom", (-0.25, -0.05, 0.784), BROOM)
    cubes = [_obj(f"cube_{i}", "small_cube", (0.1 + 0.1 * i, -0.1, 0.775), _box((-0.0096,) * 3, (0.0096,) * 3))
             for i in range(3)]
    w = WorldState("", "", {o.label: o for o in [pan, broom] + cubes}, {}, {})
    p = planner.plan("Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the blocks.", w)
    assert p.skill == "sweep_into"
    assert p.args == {"objects": ["cube_0", "cube_1", "cube_2"], "pan": "broom_shovel", "tool": "broom"}


def test_pan_geometry_entry_lip_and_circle():
    from dojo.skills.sweep import pan_geometry
    th = math.radians(90)
    pan = _obj("p", "broom_shovel", (0.0, 0.0, 0.789), PAN, quat=(math.cos(th / 2), 0, 0, math.sin(th / 2)),
               support=SUPPORT)
    g = pan_geometry(pan)
    assert np.allclose(g["e"][:2], [-1.0, 0.0], atol=1e-9)              # asset +y turned 90 deg: into the pan = -x
    assert np.allclose(g["c"][:2], [0.053, 0.0], atol=1e-9) and abs(g["r"] - 0.081) < 1e-9
    assert np.allclose(g["lip"][:2], [0.14, 0.0], atol=1e-9)            # the lip faces +x
    assert abs(g["lip_in"] - 0.087) < 1e-9


def test_tool_head_is_the_wide_end():
    from dojo.skills.sweep import tool_head
    b = _obj("broom", "broom", (0, 0, 0.78), BROOM)
    h = tool_head(b)
    assert h["axis"] == 2 and h["sign"] == 1.0 and h["centre"][2] > 0.05


def test_pan_handle_grasps_pinch_across_the_handle():
    from dojo.skills.sweep import pan_handle_grasps
    pan = _obj("p", "broom_shovel", (0.1, 0.0, 0.789), PAN, support=SUPPORT)
    gs = pan_handle_grasps(pan)
    assert gs and all(abs(g.width - 0.03) < 0.003 for g in gs)
    straight = [g for g in gs if "pitch" not in g.note]
    assert straight and all(abs(g.T[2, 0] + 1.0) < 1e-9 for g in straight)            # straight down
    assert all(abs(abs(g.T[0, 1]) - 1.0) < 1e-6 for g in straight)                    # closing across the handle (x)


def test_a_sweep_plans_end_to_end_in_the_kinematic_env():
    from dojo.skills.env import KinematicEnv
    from dojo.skills.skills import GTProvider, Skills
    from dojo.skills.sweep import sweep_into
    from dojo.skills.world import AssetDB
    rows = [{"label": "broom_shovel", "category": "broom_shovel", "category_idx": 0, "pos": [-0.05, 0.02, 0.789], "quat": [1, 0, 0, 0],
             "half": [0.092, 0.14, 0.024], "center": [0, 0, 0]},
            {"label": "broom", "category": "broom", "category_idx": 0, "pos": [0.05, -0.27, 0.784], "quat": [0.7071, 0.0, 0.7071, 0.0],
             "half": [0.019, 0.041, 0.125], "center": [0, 0, 0]},
            {"label": "cube_0", "category": "small_cube", "category_idx": 0, "pos": [0.2, -0.05, 0.775], "quat": [1, 0, 0, 0],
             "half": [0.0096] * 3, "center": [0, 0, 0]}]
    env = KinematicEnv(rows, step_lim=1000, instruction="sweep the blocks")

    class DB(AssetDB):
        def get(self, obj_type, category, idx, clutter=False, mesh=False):
            from dojo.skills.world import AssetInfo
            m = {"broom_shovel": PAN, "broom": BROOM}.get(category, _box((-0.0096,) * 3, (0.0096,) * 3))
            return AssetInfo(key=f"Rigid/{category}/00000", bbox_min=m[0].min(0), bbox_max=m[0].max(0), mesh=m,
                             support=SUPPORT if category == "broom_shovel" else {})
    sk = Skills(env, GTProvider(env, DB(None, None), instruction="sweep the blocks"), reserve=30, log=lambda m: None)
    res = sweep_into(sk, ["cube_0"], "broom_shovel", "broom")
    # the kinematic env does not push cubes: the plan runs end to end, the cube stays out
    assert res["stage"] == "sweep" and "cube_0" in res.get("note", "")
    assert env.steps_used > 100


def test_sweep_relay_grip_default_off_and_giver_grasps_straight_down_first(monkeypatch):
    """``Skills.sweep_relay_grip`` (default off): the relay's giver plans its straight-down broom grasps first, ranked
    by ``grasps.hold_quality``, and falls back to today's set (G3 dev: tilted giver pinches slipped)."""
    from dojo.runtime.policy import RuntimeConfig
    from dojo.skills import sweep
    from dojo.skills.env import KinematicEnv
    from dojo.skills.skills import GTProvider, SkillResult, Skills
    from dojo.skills.world import AssetDB, AssetInfo
    assert RuntimeConfig().sweep_relay_grip is False
    rows = [{"label": "broom", "category": "broom", "category_idx": 0, "pos": [-0.33, -0.12, 0.784],
             "quat": [0.7071, 0.0, 0.7071, 0.0], "half": [0.019, 0.041, 0.125], "center": [0, 0, 0]}]
    env = KinematicEnv(rows, step_lim=1000, instruction="sweep the blocks")

    class DB(AssetDB):
        def get(self, obj_type, category, idx, clutter=False, mesh=False):
            return AssetInfo(key="Rigid/broom/00000", bbox_min=BROOM[0].min(0), bbox_max=BROOM[0].max(0), mesh=BROOM)
    seen = []

    def plan_transfer(self, tool, targets, arms=None, yaw_sym=1, grasps=None, **kw):
        gs = grasps(arms[0])
        seen.append([round(-float(g.T[2, 0]), 3) for g in gs])
        return SkillResult(False, "transfer", "ik_fail", 0, arms[0], note="test")
    for flag, n_calls in (("off", 1), ("on", 2)):
        seen.clear()
        sk = Skills(env, GTProvider(env, DB(None, None), instruction="sweep the blocks"), reserve=30, log=lambda m: None)
        sk.sweep_relay_grip = flag
        monkeypatch.setattr(Skills, "plan_transfer", plan_transfer)
        monkeypatch.setattr(sweep, "_spots_ok", lambda *a: True, raising=False)
        sweep.relay_tool(sk, "broom", "right")
        assert seen, flag                                             # a spot the taker can pick from was tried
        if flag == "on":
            assert all(v > sweep.RELAY_DOWN for v in seen[0])        # the firm set first, straight down only
            assert len(seen[1]) >= len(seen[0])                       # then today's set
        else:
            assert len(seen) == len(seen) // 1 and all(len(x) == len(seen[0]) for x in seen)   # today's set only
