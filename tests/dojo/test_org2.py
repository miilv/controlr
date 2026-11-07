"""Stream ORG2 switches (all default off): planned two-point pushes (``dojo.skills.push_plan``), the "clear" push goal,
arrange retry / push-first order, torso grasps, stand offsets and the axis-first retreat (``dojo.skills.torso``).
CPU only: ``KinematicEnv`` / ``SimpleNamespace`` stand-ins, no Isaac, no LLM."""
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import library
from dojo.skills import push_plan as PP
from dojo.skills import torso
from dojo.skills.g3 import SKILL_ATTRS
from dojo.skills.kinematics import TIP_BEYOND_TCP

SWITCHES = {"push_contact": ("off", "pair"), "push_plan": ("off", "sequence"), "arrange_retry": ("off", "once"),
            "grasp_torso": ("off", "on"), "stand_offset": ("off", "base"), "retreat_axis": ("off", "on")}


@pytest.mark.parametrize("name,values", sorted(SWITCHES.items()))
def test_switch_defaults_off_and_validated(name, values):
    assert getattr(RuntimeConfig(), name) == values[0]
    assert SKILL_ATTRS[name] == values
    RuntimePolicy(RuntimeConfig(**{name: values[1]}))
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(**{name: "bad"}))


def test_extended_enums():
    assert "clear" in SKILL_ATTRS["push_goal"] and SKILL_ATTRS["push_goal"][0] == "centre"
    assert "push_first" in SKILL_ATTRS["arrange_order"] and SKILL_ATTRS["arrange_order"][0] == "near"
    RuntimePolicy(RuntimeConfig(push_goal="clear", arrange_order="push_first"))


def test_skills_defaults_off():
    from dojo.skills.env import KinematicEnv
    from dojo.skills.skills import GTProvider, Skills
    env = KinematicEnv([], step_lim=10)
    sk = Skills(env, GTProvider(env, None))
    for name, values in SWITCHES.items():
        assert getattr(sk, name) == values[0]


def _obj(c, ext, yaw=0.0, top=None, bottom=None, mesh=None):
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    ext = np.asarray(ext, float)
    c = np.asarray(c, float)
    half = np.abs(R) @ (ext / 2)
    return SimpleNamespace(R=R, asset=SimpleNamespace(extents=ext, mesh=mesh, center=np.zeros(3)), center_w=c,
                           pos=c.copy(), aabb=lambda: (c - half, c + half),
                           top_z=c[2] + ext[2] / 2 if top is None else top,
                           bottom_z=c[2] - ext[2] / 2 if bottom is None else bottom)


def test_footprint_and_polygon_distance():
    a = _obj([0.0, 0.0, 0.8], [0.2, 0.1, 0.02])
    fp = PP.footprint(a)
    np.testing.assert_allclose(np.sort(fp[:, 0]), [-0.1, -0.1, 0.1, 0.1])
    b = PP.footprint(_obj([0.3, 0.0, 0.8], [0.2, 0.1, 0.02]))
    assert PP.poly_dist(fp, b) == pytest.approx(0.1)
    assert PP.poly_dist(fp, PP.footprint(_obj([0.05, 0.0, 0.8], [0.2, 0.1, 0.02]))) == 0.0
    turned = PP.footprint(a, yaw=np.pi / 2)
    assert np.ptp(turned[:, 1]) == pytest.approx(0.2)


def test_aligned_yaw_squares_to_the_support():
    kb = _obj([0, 0, 0.77], [0.368, 0.152, 0.013], yaw=np.radians(17))
    frame = _obj([0.1, 0, 0.766], [0.24, 0.10, 0.0003])
    assert PP.aligned_yaw(kb, frame) == pytest.approx(0.0, abs=1e-9)
    kb2 = _obj([0, 0, 0.77], [0.368, 0.152, 0.013], yaw=np.radians(80))
    assert PP.aligned_yaw(kb2, frame) == pytest.approx(np.pi / 2)


def _kb_world(extra=()):
    from dojo.skills.env import KinematicEnv
    from dojo.skills.skills import GTProvider, Skills
    rows = [dict(label="kb", category="keyboard", category_idx=0, type="Rigid", pos=[-0.04, -0.18, 0.7715],
                 quat=[1, 0, 0, 0], half=[0.184, 0.076, 0.0065]),
            dict(label="frame", category="frame", category_idx=0, type="Geometry", pos=[0.1, 0.0, 0.7656],
                 quat=[1, 0, 0, 0], half=[0.12, 0.05, 0.00015])] + list(extra)
    env = KinematicEnv(rows, step_lim=1000)
    return env, Skills(env, GTProvider(env, None))


def test_clear_goal_keeps_away_from_a_pending_site():
    env, sk = _kb_world()
    lo, hi, _ = PP.cover_box(sk.world.obj("kb"), sk.world.obj("frame"), 0.0)
    g0, d0 = PP.clear_goal(sk, "kb", "frame", ())
    # no constraint: the middle of the region (the frame's centre)
    np.testing.assert_allclose(g0, [0.1, 0.0], atol=1e-6)
    g1, d1 = PP.clear_goal(sk, "kb", "frame", [((-0.15, 0.1), 0.05)])
    assert g1[0] > g0[0] + 0.02                         # away from the site on the -x side
    assert g1[0] <= 0.1 + hi[0] - PP.GOAL_EDGE + 1e-9   # still inside the cover region, GOAL_EDGE in
    assert d1["keep_clear"] == 1


def test_clear_goal_ignores_flat_markers_and_held_objects():
    stand = dict(label="stand", category="cube_cushion", category_idx=2, type="Geometry", pos=[0.0, 0.01, 0.765],
                 quat=[1, 0, 0, 0], half=[0.025, 0.025, 0.00005])
    env, sk = _kb_world([stand])
    _, d = PP.clear_goal(sk, "kb", "frame", ())
    assert d["others"] == 0


def test_pair_stroke_opens_the_jaws_across_the_stroke():
    calls = []

    class Motion:
        def to_pose(self, arm, q, T, *a, **kw):
            calls.append(("to_pose", T.copy(), kw))
            return SimpleNamespace(ok=False)

        def line(self, *a, **kw):
            return SimpleNamespace(ok=False)

        def joint(self, arm, q0, q1, *a, **kw):
            assert kw.get("check", True)                 # the escape home is always collision-checked
            return SimpleNamespace(ok=False)

    o = _obj([0.0, 0.1, 0.78], [0.4, 0.12, 0.02], top=0.79, bottom=0.77)
    sk = SimpleNamespace(motion=Motion(), kin=SimpleNamespace(base=lambda a: np.eye(4), tcp=lambda a, q: np.eye(4)),
                         ex=SimpleNamespace(q={a: np.zeros(6) for a in ("left", "right")}, g={"left": 1.0, "right": 1.0}),
                         world=SimpleNamespace(home_q={a: np.zeros(6) for a in ("left", "right")}),
                         plan_home=lambda arm: (_ for _ in ()).throw(AssertionError("unchecked home")))
    assert PP.pair_stroke(sk, o, "wide", np.array([1.0, 0.0]), 0.05) is None
    assert calls
    for _, T, kw in calls:
        assert kw["gripper"] == PP.PAIR_GRIP and kw["ignore"] == ("wide",)
        across = abs(float(T[:3, 1] @ np.array([1.0, 0.0, 0.0])))          # closing axis across the stroke
        assert across < 1e-9 if T[2, 0] < -0.999 else across < 0.2          # (tilted: projected, a few deg off)
        tip = T[:3, 3] + T[:3, 0] * TIP_BEYOND_TCP
        assert tip[0] == pytest.approx(-0.2 - PP.STANDOFF)


def test_pair_stroke_plans_in_a_kinematic_world():
    env, sk = _kb_world()
    sk.push_contact = "pair"
    o = sk.world.obj("kb")
    p = PP.pair_stroke(sk, o, "kb", np.array([0.0, 1.0]), 0.05)
    assert p is not None and p.steps > 0


class _Obj(SimpleNamespace):
    @property
    def center_w(self):
        return self.pos


def test_plan_sequence_takes_the_order_whose_every_stroke_plans():
    o = _Obj(pos=np.zeros(3))
    sk = SimpleNamespace(world=SimpleNamespace(obj=lambda lab: o), log=lambda m: None)

    def fn(sk_, ob, lab, u, step):
        # an x stroke plans only while y is still near the start (later its start is in a fixture)
        if abs(u[0]) > 0.5 and ob.pos[1] > 0.05:
            return None
        return SimpleNamespace(arm="left")
    goal = np.array([0.10, 0.18])                       # more to go in y: the greedy order is y first
    seq = PP.plan_sequence(sk, "kb", goal, [np.array([1.0, 0.0]), np.array([0.0, 1.0])], fn)
    assert seq is not None and seq["order"] == [0, 1]   # x first: the y-first order left no x stroke
    assert seq["strokes"][0][0][0] == 1.0
    np.testing.assert_allclose(o.pos, 0.0)              # the planning world is restored
    moved = sum(np.asarray(u) * s for u, s, _ in seq["strokes"])
    assert np.all(np.abs(moved - goal) <= 0.012)
    assert PP.plan_sequence(sk, "kb", goal, [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
                            lambda *a: None) is None


def test_stroke_len_cuts_long_strokes():
    assert PP.stroke_len(0.2, 0.08) == pytest.approx(0.08 * PP.OVERSHOOT)
    assert PP.stroke_len(0.01, 0.08) == pytest.approx(0.013)


def test_push_to_legacy_path_without_switches(monkeypatch):
    sk = SimpleNamespace()
    monkeypatch.setattr(library, "_push_planned", lambda *a, **k: (_ for _ in ()).throw(AssertionError("planned")))
    called = []

    class Stop(Exception):
        pass

    def refresh():
        called.append(1)
        raise Stop
    sk.refresh = refresh
    sk.ex = SimpleNamespace(steps_used=0)
    with pytest.raises(Stop):
        library.push_to(sk, "kb", (0.0, 0.0))
    assert called
    sk.push_contact = "pair"
    with pytest.raises(AssertionError, match="planned"):
        library.push_to(sk, "kb", (0.0, 0.0))


@pytest.mark.parametrize("retry", ["off", "once"])
def test_arrange_retry_once(monkeypatch, retry):
    from dojo.skills.env import KinematicEnv
    from dojo.skills.skills import GTProvider, Skills, SkillResult
    rows = [dict(label=lab, category=cat, category_idx=0, type=typ, pos=[x, -0.2, 0.785], quat=[1, 0, 0, 0],
                 half=[0.02, 0.02, 0.02])
            for lab, cat, typ, x in (("a", "mouse", "Rigid", 0.0), ("pa", "mousemat", "Geometry", 0.1),
                                     ("b", "alarm", "Rigid", -0.1), ("pb", "drawer", "Geometry", -0.2))]
    env = KinematicEnv(rows, step_lim=600)
    sk = Skills(env, GTProvider(env, None), reserve=30)
    sk.arrange_retry = retry
    tries = {"a": 0, "b": 0}
    monkeypatch.setattr(library, "container_obstacle", lambda *a: None)
    monkeypatch.setattr(library, "_on_targets", lambda *a: ([np.eye(4)], False))

    def place(sk_, obj, tg, **kw):
        tries[obj] += 1
        ok = obj == "b" or tries[obj] >= 2               # "a" fails the first time only
        return SkillResult(ok, "place", "ok" if ok else "ik_fail", 0, None)
    monkeypatch.setattr(library, "place_oriented", place)
    r = library.arrange(sk, [("place", "a", "pa"), ("place", "b", "pb")])
    if retry == "off":
        assert not r["ok"] and tries == {"a": 1, "b": 1}
    else:
        assert r["ok"] and tries == {"a": 2, "b": 1}


def test_arrange_push_first_and_keep_clear(monkeypatch):
    from dojo.skills.env import KinematicEnv
    from dojo.skills.skills import GTProvider, Skills, SkillResult
    rows = [dict(label="kb", category="keyboard", category_idx=0, type="Rigid", pos=[0.0, -0.18, 0.77],
                 quat=[1, 0, 0, 0], half=[0.18, 0.07, 0.006]),
            dict(label="frame", category="frame", category_idx=0, type="Geometry", pos=[0.1, 0.0, 0.766],
                 quat=[1, 0, 0, 0], half=[0.12, 0.05, 0.0002]),
            dict(label="fig", category="garage", category_idx=0, type="Rigid", pos=[-0.3, -0.1, 0.86],
                 quat=[1, 0, 0, 0], half=[0.05, 0.05, 0.095]),
            dict(label="stand", category="cube_cushion", category_idx=0, type="Geometry", pos=[-0.15, 0.1, 0.765],
                 quat=[1, 0, 0, 0], half=[0.025, 0.025, 0.00005])]
    env = KinematicEnv(rows, step_lim=600)
    sk = Skills(env, GTProvider(env, None), reserve=30)
    sk.arrange_order, sk.push_goal = "push_first", "clear"
    order, kws = [], []
    monkeypatch.setattr(library, "container_obstacle", lambda *a: None)
    monkeypatch.setattr(library, "_on_targets", lambda *a: ([np.eye(4)], False))
    monkeypatch.setattr(library, "place_oriented", lambda sk_, obj, tg, **kw: order.append(obj) or
                        SkillResult(True, "place", "ok", 0, None))
    monkeypatch.setattr(library, "push_to", lambda sk_, obj, tgt, **kw: order.append(obj) or kws.append(kw) or
                        SkillResult(True, "push", "ok", 0, None))
    library.arrange(sk, [("place", "fig", "stand"), ("push", "kb", "frame")])
    assert order == ["kb", "fig"]
    assert kws[0]["support"] == "frame"
    (c, r), = kws[0]["keep_clear"]
    np.testing.assert_allclose(c, [-0.15, 0.1])
    assert r == pytest.approx(0.05)


def _figurine_mesh():
    """A base disc (r 50 mm, 15 mm tall), a torso 40 x 50 mm up to 150 mm, an arm bar 180 mm long at 100-120 mm."""
    pts = []
    for z in np.linspace(0.0, 0.015, 4):
        for t in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            pts.append([0.05 * np.cos(t), 0.05 * np.sin(t), z])
    for z in np.linspace(0.015, 0.15, 28):
        for x in (-0.02, 0.02):
            for y in np.linspace(-0.025, 0.025, 6):
                pts.append([x, y, z])
        for y in (-0.025, 0.025):
            for x in np.linspace(-0.02, 0.02, 5):
                pts.append([x, y, z])
    for z in np.linspace(0.10, 0.12, 5):
        for x in np.linspace(-0.09, 0.09, 37):
            for y in (-0.01, 0.01):
                pts.append([x, y, z])
    V = np.asarray(pts) - np.array([0.0, 0.0, 0.075])
    return V, np.zeros((1, 3), int)


def _figurine(xy=(-0.2, -0.15)):
    V, F = _figurine_mesh()
    pos = np.array([xy[0], xy[1], 0.765 + 0.075])
    return SimpleNamespace(R=np.eye(3), pos=pos, asset=SimpleNamespace(mesh=(V, F), extents=np.array([0.18, 0.1, 0.15]),
                                                                       center=np.zeros(3)),
                           center_w=pos.copy(), top_z=pos[2] + 0.075, bottom_z=pos[2] - 0.075)


def test_long_axis_is_the_arm_line():
    o = _figurine()
    V = o.asset.mesh[0] @ o.R.T + o.pos
    a = torso.long_axis(V, V[:, 2].min() + 0.045, V[:, 2].max())
    assert abs(a[0]) > 0.95


def test_torso_grasps_close_across_the_body_from_the_arm_side():
    from dojo.skills.kinematics import X5Kin
    kin = X5Kin()
    o = _figurine()
    gs = torso.torso_grasps(o, kin, "left")
    assert gs and all(g.method == "torso" for g in gs)
    base = kin.base("left")[:2, 3]
    for g in gs:
        assert 0.012 <= g.width <= 0.084
        closing = g.T[:3, 1]
        assert abs(closing[2]) < 1e-9
        ang = np.degrees(np.arccos(min(1.0, abs(float(closing[0])))))
        assert ang == pytest.approx(90.0, abs=1) or ang == pytest.approx(45.0, abs=1)
        assert float(g.T[:2, 0] @ (g.T[:2, 3] - base)) > 0          # approaches away from the arm's base
        assert g.T[2, 0] <= 1e-9                                      # level or downward
    # no grasp at the arm bar's height closes along the 180 mm bar
    assert all(g.width < 0.18 for g in gs)


def test_torso_needs_a_mesh():
    o = _figurine()
    o.asset.mesh = None
    from dojo.skills.kinematics import X5Kin
    assert torso.torso_grasps(o, X5Kin(), "left") == []


def test_base_radius_and_stand_points():
    o = _figurine((0.0, 0.0))
    assert torso.base_radius(o) == pytest.approx(0.05, abs=1e-6)
    stand = _obj([-0.15, 0.1, 0.765], [0.05, 0.05, 0.0001])
    drawer = _obj([-0.35, 0.15, 0.88], [0.35, 0.27, 0.227])
    objs = {"fig": o, "stand": stand, "drawer": drawer}
    sk = SimpleNamespace(world=SimpleNamespace(obj=lambda l: objs[l], objects=objs))
    pts = torso.stand_points(sk, "fig", "stand")
    assert len(pts) == 16
    d = [float(np.linalg.norm(p - np.array([-0.15, 0.1]))) for p in pts]
    assert max(d) == pytest.approx(0.025, abs=1e-6) and min(d) == pytest.approx(0.0125, abs=1e-6)
    assert pts[0][0] > -0.15                                          # the first one moves away from the drawer


@pytest.mark.parametrize("mode", ["off", "on"])
def test_retreat_axis_backs_out_level_then_rises(mode):
    from dojo.skills.motion import MoveResult, Segment
    from dojo.skills.skills import Skills
    sk = object.__new__(Skills)
    sk.retreat_axis = mode
    goals = []

    def line(arm, q0, T, *a, **kw):
        goals.append((kw.get("name", a[1] if len(a) > 1 else ""), np.asarray(T, float).copy()))
        return MoveResult("reached", arm, [Segment(np.stack([np.zeros(6), np.ones(6) * 0.01]), 0.03)])
    sk.motion = SimpleNamespace(line=line)
    T_side = np.eye(4)
    T_side[:3, :3] = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])   # approach +x (level)
    T_side[:3, 3] = [0.1, 0.0, 0.85]
    h = SimpleNamespace(label="fig", T_tcp_obj=np.eye(4), box=None)
    t = sk._place_tail("left", np.zeros(6), T_side, h, release_gap=0.0, retreat=0.05, ignore=("fig",),
                       descend_rad=0.02, open_g=1.0, settle=1, open_steps=2, pre_open_hold=1)
    names = [g[0] for g in goals]
    retreat = next(T for n, T in goals if n == "place_retreat")
    if mode == "on":
        assert retreat[2, 3] == pytest.approx(0.85) and "place_rise" in names
        rise = next(T for n, T in goals if n == "place_rise")
        assert rise[2, 3] == pytest.approx(0.89) and rise[0, 3] == pytest.approx(retreat[0, 3])
    else:
        assert retreat[2, 3] == pytest.approx(0.87) and "place_rise" not in names
    assert t.steps > 0


def test_escape_home_is_checked_and_rises_first_when_needed():
    from dojo.skills.env import KinematicEnv
    from dojo.skills.motion import MoveResult, Segment
    from dojo.skills.skills import GTProvider, Skills
    env = KinematicEnv([], step_lim=100)
    sk = Skills(env, GTProvider(env, None))
    seen = []
    seg = [Segment(np.stack([np.zeros(6), np.ones(6) * 0.01]), 0.03)]

    def joint(arm, q0, q1, *a, **kw):
        seen.append(("joint", kw.get("check", True), float(np.asarray(q0)[0])))
        ok = len([s for s in seen if s[0] == "line"]) > 0       # only after a rise
        return MoveResult("reached" if ok else "collision", arm, seg if ok else [])

    def line(arm, q0, T, *a, **kw):
        seen.append(("line", kw.get("check", True), float(T[2, 3])))
        return MoveResult("reached", arm, seg)
    sk.motion = SimpleNamespace(joint=joint, line=line)
    p = PP.escape_home(sk, "left")
    assert p is not None
    assert all(c for _, c, _ in seen)                                   # every leg checked
    assert [k for k, _, _ in seen] == ["joint", "line", "joint"]
    sk.motion = SimpleNamespace(joint=lambda *a, **k: MoveResult("collision", "left", []),
                                line=lambda *a, **k: MoveResult("collision", "left", []))
    assert PP.escape_home(sk, "left") is None
