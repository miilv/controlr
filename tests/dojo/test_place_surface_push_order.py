"""Stream H: ``Skills.place_surface`` ("bbox" = today | "mesh": ``library.on_targets`` rests the object on the support
mesh's surface under its whole footprint) and ``Skills.push_order`` ("greedy" = today | "lookahead": ``library.push_to``
pushes the other axis first when its stroke would be blocked once the first is done). Their effect is screened in G3
(physics); here: geometry, the defaults, plumbing. CPU only."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import library
from dojo.skills.kinematics import X5Kin
from dojo.skills.library import footprint_xy, on_targets, push_to, support_z_fn
from dojo.skills.skills import SkillResult
from dojo.skills.world import AssetInfo, ObjectState

TABLE = 0.765


def _box(lo, hi):
    """Triangle mesh (V, F) of an axis-aligned box."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], int)
    return V, F


def _union(*boxes):
    Vs, Fs, n = [], [], 0
    for lo, hi in boxes:
        V, F = _box(lo, hi)
        Vs.append(V)
        Fs.append(F + n)
        n += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def _tray_asset():
    """A drawer-like cabinet: a tray floor 23 mm below the rims, a back block and side rims, open at the front, where
    only a low handle sticks out (organize_table's drawer, simplified)."""
    V, F = _union(((-0.06, -0.17, 0.0), (0.16, 0.15, 0.204)),          # body, its top = the tray floor
                  ((0.16, -0.187, 0.0), (0.1875, 0.164, 0.227)),       # back block
                  ((-0.06, 0.15, 0.0), (0.16, 0.164, 0.227)),          # side rims
                  ((-0.06, -0.187, 0.0), (0.16, -0.17, 0.227)),
                  ((-0.08, -0.03, 0.15), (-0.06, 0.03, 0.164)))        # a handle on the front, 63 mm below the top
    return AssetInfo(key="Geometry/drawer/00000", bbox_min=V.min(0), bbox_max=V.max(0), mesh=(V, F),
                     place={"up": np.array([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])})


def _alarm_asset():
    V, F = _box((-0.04, -0.025, -0.04), (0.04, 0.025, 0.04))
    return AssetInfo(key="Rigid/alarm/00000", bbox_min=V.min(0), bbox_max=V.max(0), mesh=(V, F),
                     place={"up": np.array([1.0, 0, 0, 0])})


def _quat_z(yaw):
    return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


def _sk(objs, **attrs):
    kin = X5Kin()
    logs = []
    sk = SimpleNamespace(world=SimpleNamespace(obj=lambda l: objs[l]), kin=kin, log=logs.append, logs=logs)
    for k, v in attrs.items():
        setattr(sk, k, v)
    return sk


def _scene():
    drawer = ObjectState("drawer", "Geometry", "drawer", 0, np.array([-0.40, 0.10, TABLE]), _quat_z(math.pi / 2),
                         _tray_asset())
    alarm = ObjectState("alarm", "Rigid", "alarm", 0, np.array([-0.30, -0.20, TABLE + 0.04]), _quat_z(0.0),
                        _alarm_asset())
    return {"drawer": drawer, "alarm": alarm}


def test_support_z_reads_the_tray_floor_and_the_rims_in_world():
    objs = _scene()
    z_at = support_z_fn(objs["drawer"])
    # the drawer is turned 90 deg: its local +x (towards the back block) is world +y
    z = z_at(np.array([[-0.40, 0.15], [-0.40, 0.10 + 0.175], [-0.40 + 0.18, 0.15], [-0.48, 0.10 - 0.075], [-0.40, 0.10 - 0.075]]))
    assert z[0] == pytest.approx(TABLE + 0.204, abs=1e-6)            # tray floor
    assert z[1] == pytest.approx(TABLE + 0.227, abs=1e-6)            # back block
    assert z[2] == pytest.approx(TABLE + 0.227, abs=1e-6)            # a side rim (local y -0.18 -> world x +0.18)
    assert np.isinf(z[3])                                            # in front of the body, beside the handle: nothing
    assert z[4] == pytest.approx(TABLE + 0.164, abs=1e-6)            # the handle, 63 mm below the top


def test_footprint_is_the_face_the_object_rests_on():
    o = _scene()["alarm"]
    F = footprint_xy(o, np.eye(3), n=3)
    assert F[:, 0].min() == pytest.approx(-0.04) and F[:, 1].max() == pytest.approx(0.025)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])               # turned 90 deg: 50 mm along x, 80 along y
    F = footprint_xy(o, R, n=3)
    assert F[:, 0].max() == pytest.approx(0.025) and F[:, 1].max() == pytest.approx(0.04)


def test_bbox_is_the_default_and_unchanged():
    objs = _scene()
    a = on_targets(_sk(objs), "alarm", "drawer")
    b = on_targets(_sk(objs, place_surface="bbox"), "alarm", "drawer")
    assert len(a) == len(b) and all(np.allclose(x, y) for x, y in zip(a, b))
    top = objs["drawer"].top_z
    assert all(T[2, 3] == pytest.approx(top + 0.04) for T in a)       # the bbox top (the rims), 23 mm over the floor
    assert RuntimeConfig().place_surface == "bbox" and RuntimeConfig().push_order == "greedy"


def test_mesh_rests_on_the_floor_and_keeps_the_footprint_on_it():
    objs = _scene()
    sk = _sk(objs, place_surface="mesh")
    out = on_targets(sk, "alarm", "drawer")
    assert out and "rest flat" in sk.logs[-1]
    z_at = support_z_fn(objs["drawer"])
    o = objs["alarm"]
    for T in out:
        assert T[2, 3] == pytest.approx(TABLE + 0.204 + 0.04, abs=1e-6)
        R = T[:3, :3]
        c = (T[:3, :3] @ o.asset.center + T[:3, 3])[:2]
        z = z_at(c + footprint_xy(o, R))
        assert np.all(np.isfinite(z)) and float(z.max() - z.min()) < 1e-6
    # the bbox targets reach over the open front (the alarm's front over nothing); the mesh ones do not
    bbox = on_targets(_sk(objs), "alarm", "drawer")
    front = [T for T in bbox if not np.all(np.isfinite(z_at((T[:3, :3] @ o.asset.center + T[:3, 3])[:2]
                                                                 + footprint_xy(o, T[:3, :3]))))]
    assert front


def test_mesh_falls_back_to_the_bbox_targets_when_nothing_rests_flat():
    objs = _scene()
    V, F = _box((-0.025, -0.025, 0.0), (0.025, 0.025, 0.002))         # a 50 mm cushion under a 80 x 50 mm object
    objs["drawer"] = ObjectState("drawer", "Geometry", "cube_cushion", 2, np.array([-0.15, 0.10, TABLE]), _quat_z(0.0),
                                 AssetInfo(key="Geometry/cube_cushion/00002", bbox_min=V.min(0), bbox_max=V.max(0),
                                           mesh=(V, F), place={"up": np.array([1.0, 0, 0, 0])}))
    sk = _sk(objs, place_surface="mesh")
    out = on_targets(sk, "alarm", "drawer")
    ref = on_targets(_sk(objs), "alarm", "drawer")
    assert "bbox targets" in sk.logs[-1]
    assert len(out) == len(ref) and all(np.allclose(x, y) for x, y in zip(out, ref))


# ------------------------------------------------------------------ push_order


class _PushSk:
    """Just enough of ``Skills`` for ``push_to``: a stroke moves the object by its step; ``blocked(start_xy)`` says
    where the fingers cannot go down (a drawer's front)."""

    def __init__(self, obj, push_order=None):
        self.o = obj
        self.ex = SimpleNamespace(steps_used=0)
        self.logs = []
        if push_order:
            self.push_order = push_order

    def refresh(self):
        return SimpleNamespace(obj=lambda l: self.o)

    def parallel(self, plans, name=""):
        p = plans[0]
        self.o.pos = self.o.pos + np.r_[p.u * p.step, 0.0]
        self.ex.steps_used += 30

    def confirm_moved(self, *a, **k):
        return None

    def _record(self, r):
        return r

    def log(self, m):
        self.logs.append(m)


def _fake_stroke(blocked):
    def stroke(sk, o, label, u, step, centre=None):
        c = o.center_w[:2] if centre is None else np.asarray(centre, float)
        half_u = float(np.abs(o.R.T @ np.r_[u, 0.0]) @ (o.asset.extents / 2))
        start = c - u * (half_u + 0.02)
        return None if blocked(start) else SimpleNamespace(arm="left", u=u, step=step)
    return stroke


def _keyboard():
    return ObjectState("keyboard", "Rigid", "keyboard", 1, np.array([-0.05, -0.165, 0.776]), _quat_z(0.0),
                       AssetInfo(key="Rigid/keyboard/00001", bbox_min=np.array([-0.1636, -0.08, -0.0111]),
                                 bbox_max=np.array([0.1636, 0.08, 0.0111])))


def _blocked_far_left(start):      # the sideways stroke's start next to something by the frame (organize_table
    return start[0] < -0.19 and start[1] > -0.09     # 100002: the alarm on the drawer's front edge)


@pytest.mark.parametrize("order, ok", [(None, False), ("greedy", False), ("lookahead", True)])
def test_lookahead_pushes_the_blocked_axis_first(monkeypatch, order, ok):
    monkeypatch.setattr(library, "_push_stroke", _fake_stroke(_blocked_far_left))
    sk = _PushSk(_keyboard(), order)
    r = push_to(sk, "keyboard", (0.10, 0.0))
    assert isinstance(r, SkillResult) and r.ok is ok
    if ok:
        assert np.linalg.norm(sk.o.center_w[:2] - [0.10, 0.0]) <= 0.012
        assert any("axis 0 first" in m for m in sk.logs)
    else:
        assert r.status == "ik_fail" and "axis 0" in r.note


def test_lookahead_keeps_the_greedy_order_when_nothing_blocks(monkeypatch):
    monkeypatch.setattr(library, "_push_stroke", _fake_stroke(lambda s: False))
    a, b = _PushSk(_keyboard(), "greedy"), _PushSk(_keyboard(), "lookahead")
    ra, rb = push_to(a, "keyboard", (0.10, 0.0)), push_to(b, "keyboard", (0.10, 0.0))
    assert ra.ok and rb.ok and np.allclose(a.o.pos, b.o.pos) and not b.logs


def test_options_are_validated_and_reach_the_skills():
    class _Assets:
        def get(self, typ, cat, idx, clutter=False, mesh=False):
            return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.015), bbox_max=np.full(3, 0.015))

    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", place_surface="cad"), assets=_Assets())
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", push_order="random"), assets=_Assets())
    from dojo.skills.env import KinematicEnv
    env = KinematicEnv([{"label": "b", "category": "block", "category_idx": 5, "pos": (0.3, -0.1, 0.78),
                         "quat": [1, 0, 0, 0], "half": [0.015] * 3}], step_lim=50, instruction="Stack the blocks.")
    pol = RuntimePolicy(RuntimeConfig(mode="gt", verify="off", place_surface="mesh", push_order="lookahead"),
                        assets=_Assets())
    pol.reset({"task": "x", "instruction": env.instruction, "step_lim": 50, "layout_id": 1})
    obs = {"images": {c: np.zeros((8, 8, 3), np.uint8) for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
           "state": env.proprio(), "instruction": env.instruction, "steps_used": 0, "step_lim": 50,
           "_dev_gt": env.gt_objects()}
    pol.next(obs)
    sk = pol._live["sk"]
    assert sk.place_surface == "mesh" and sk.push_order == "lookahead"


def test_g3_skill_attrs_are_the_runtime_switches():
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["place_surface"][0] == RuntimeConfig().place_surface
    assert SKILL_ATTRS["push_order"][0] == RuntimeConfig().push_order


# ------------------------------------------------------------------ arrange: the bbox targets when the flat ones do not plan


class _ArrangeSk:
    def __init__(self, objs, surface):
        self.objs = objs
        self.kin = X5Kin()
        self.held = {"left": None, "right": None}
        self.notes = {}
        self.logs = []
        self.place_surface = surface
        self.world = SimpleNamespace(obj=lambda l: self.objs[l], notes=self.notes)

    def refresh(self):
        return self.world

    def bank(self):
        pass

    def log(self, m):
        self.logs.append(m)


@pytest.mark.parametrize("surface, first, calls", [("mesh", "ik_fail", 2), ("mesh", "place_miss", 1),
                                                    ("bbox", "ik_fail", 1)])
def test_arrange_falls_back_to_the_bbox_targets(monkeypatch, surface, first, calls):
    objs = _scene()
    seen = []

    def fake_place(sk, label, targets, **kw):
        seen.append([T[2, 3] for T in targets])
        st = first if len(seen) == 1 else "ok"
        return SkillResult(st == "ok", "place_oriented", st, 0 if st == "ik_fail" else 40, "left")

    monkeypatch.setattr(library, "place_oriented", fake_place)
    monkeypatch.setattr(library, "container_obstacle", lambda sk, label: None)
    sk = _ArrangeSk(objs, surface)
    library.arrange(sk, [("place", "alarm", "drawer")])
    assert len(seen) == calls
    top = objs["drawer"].top_z + 0.04
    if surface == "mesh":
        assert all(z == pytest.approx(TABLE + 0.204 + 0.04) for z in seen[0])      # the flat poses first
    if calls == 2:
        assert all(z == pytest.approx(top) for z in seen[1])                        # then today's
        assert any("the bbox ones" in m for m in sk.logs)
