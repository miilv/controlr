"""Legality, "For Ilia" items of robot-harness's verification notes (P1S 2026-10-27, G1F 2026-10-28): the agentic
perceiver's episode budget, episode-scoped worker state, and the fake J2 server's diagnostics. Each test drives the
real function with fakes: no Isaac, no GPU, no LLM (FakeChat), no catalogue.

- P1S I1-1: a spent budget is a hard stop (no B1 -> a fail-closed reply, never an agent call); the count is keyed by
  the server's opaque ``episode["id"]`` and seeded from its ``agent_calls``; every output says ``agent_call``.
- G1F-12 / G2-11: the B1 worker's IK seed cache (``_Q_LAST``) never crosses an episode id.
- G1F-13: ``bridge/fake_server.py`` reports executed-prefix ``grip_change_while_moving`` / ``max_step_dq``.
- G1F-2 (worker half): every perceive output names its effective B1 data (``data``); ``J1_REQUIRE_DATA=1`` refuses
  fallback roots.
"""
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from dojo.perception import j1_adapter as J
from dojo.perception.agentic import j1
from dojo.perception.agentic.llm import FakeChat


def _schema():
    sys.path.insert(0, str(Path(j1.__file__).resolve().parent / "vendor"))
    try:
        import j1_schema  # the strict server's own validator, vendored byte-identical
    finally:
        sys.path.pop(0)
    return j1_schema


def _calib():
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    T_h = np.eye(4)
    T_h[:3, :3] = [[1, 0, 0], [0, -c, s], [0, -s, -c]]
    T_h[:3, 3] = [0, -0.41, 1.308]
    return {"cam_head": {"K": [[288.1325, 0, 320], [0, 288.1325, 240], [0, 0, 1]], "T_world_cam": T_h.tolist()}}


FLANGE = {"left": [-0.3, -0.2, 0.9, 1, 0, 0, 0], "right": [0.3, -0.2, 0.9, 1, 0, 0, 0]}
CFG = {"gate": "budget", "max_per_episode": 2, "refine": False, "model": "fake", "track": "0"}


def _frames():
    return {"cam_head": np.full((480, 640, 3), 90, np.uint8)}


def _b1():
    return {"objects": [{"label": "block_0", "category": "block", "asset": "Rigid/block/00007", "pos": [0.05, -0.12, 0.7825],
                         "quat": [1, 0, 0, 0], "size": [0.035] * 3, "confidence": 0.9, "source": "head", "residual": 0.05}],
            "regions": [], "needs_look": [], "want_match": {"block": "block_0"}}


def _fresh_worker():
    """What a worker restarted after a timeout has: no episode state."""
    j1._EPISODE.clear()
    j1._EPISODE.update(agent_calls=0, id=None)


def _answer():
    return FakeChat(['{"answer": {"objects": [{"keep": 0}]}}'])


# ------------------------------------------------------------------ P1S I1-1: the budget is a hard stop

def test_adapters_name_the_episode_parameter():
    # robot-harness's host passes episode= only to a callable that NAMES it (a bare **kwargs does not count)
    for fn in (j1.perceive, J.perceive):
        assert "episode" in inspect.signature(fn).parameters


@pytest.mark.parametrize("b1_mode", ["off", "fails"])
def test_spent_budget_without_b1_fails_closed_and_never_calls_the_agent(tmp_path, monkeypatch, b1_mode):
    _fresh_worker()
    if b1_mode == "fails":
        def broken(*a, **k):
            raise FileNotFoundError("catalog/index.json")
        monkeypatch.setattr(j1._b1, "_perceive_impl", broken)
    fc = FakeChat(['{"answer": {"objects": [{"label": "block", "pos": [0, 0, 0.78]}]}}'])
    ep = {"id": "ep-0123abcd", "perceive_n": 4, "agent_calls": 2}       # the server counted 2 agent estimates
    out = j1.perceive_impl(_frames(), _calib(), FLANGE, {"left": 1.0}, "Stack the three blocks.", ["block"],
                           prior={"objects": []}, chat=fc, use_b1=(b1_mode != "off"), log_dir=str(tmp_path),
                           episode=ep, **CFG)
    assert fc.sent == []                                              # no agent call past the budget
    assert out["objects"] == [] and out["agent_call"] is False and "result=budget_spent_no_b1" in out["notes"]
    assert out["error"].startswith("agent budget spent and no B1 output")
    assert ("b1_error=FileNotFoundError" in out["notes"]) == (b1_mode == "fails")
    clean, rep = _schema().validate_perceive(out)
    assert rep["ok"] and clean["objects"] == [] and clean["error"] == out["error"]


def test_budget_follows_the_episode_id_not_the_prior(tmp_path):
    _fresh_worker()                                                   # restarted mid-episode, the next call has no prior
    ep = {"id": "ep-aaaa", "perceive_n": 3, "agent_calls": 2}
    fc = FakeChat([])
    out = j1.perceive_impl(_frames(), _calib(), FLANGE, {"left": 1.0}, "Stack the three blocks.", ["block"],
                           prior=None, b1=_b1(), chat=fc, log_dir=str(tmp_path), episode=ep, **CFG)
    assert fc.sent == [] and "result=b1_gated" in out["notes"] and "gate=episode_budget_spent" in out["notes"]
    assert out["agent_call"] is False and j1._EPISODE == {"agent_calls": 2, "id": "ep-aaaa", "tracker": None}
    # G2-9 (deferred): the agent scene's tracker died with the old worker; the gated reply says so
    out = j1.perceive_impl(_frames(), _calib(), FLANGE, {"left": 1.0}, "Stack.", ["block"], prior={"objects": []},
                           b1=_b1(), chat=FakeChat([]), log_dir=str(tmp_path), episode=dict(ep, perceive_n=4),
                           **dict(CFG, track="1"))
    assert "tracker=lost" in out["notes"]
    # the same episode: the larger of the worker's and the server's count stands
    j1._EPISODE["agent_calls"] = 3
    j1.perceive_impl(_frames(), _calib(), FLANGE, {"left": 1.0}, "Stack.", ["block"], prior={"objects": []}, b1=_b1(),
                     chat=FakeChat([]), log_dir=str(tmp_path), episode=dict(ep, perceive_n=4, agent_calls=1), **CFG)
    assert j1._EPISODE["agent_calls"] == 3
    # a new episode id: its first call gets the agent, counted, and says so
    fc = _answer()
    out = j1.perceive_impl(_frames(), _calib(), FLANGE, {"left": 1.0}, "Stack the three blocks.", ["block"], prior=None,
                           b1=_b1(), chat=fc, log_dir=str(tmp_path), episode={"id": "ep-bbbb", "perceive_n": 1,
                                                                              "agent_calls": 0}, **CFG)
    assert len(fc.sent) == 1 and out["agent_call"] is True and "gate=first_call" in out["notes"]
    assert j1._EPISODE["id"] == "ep-bbbb" and j1._EPISODE["agent_calls"] == 1


def test_without_an_episode_the_old_gate_is_unchanged(tmp_path):
    _fresh_worker()
    out = j1.perceive_impl(_frames(), _calib(), FLANGE, {"left": 1.0}, "Stack.", ["block"], prior=None, b1=_b1(),
                           chat=_answer(), log_dir=str(tmp_path), **CFG)
    assert out["agent_call"] is True and "gate=first_call" in out["notes"] and j1._EPISODE["agent_calls"] == 1
    out = j1.perceive_impl(_frames(), _calib(), FLANGE, {"left": 1.0}, "Stack.", ["block"], prior=out, b1=_b1(),
                           chat=FakeChat([]), log_dir=str(tmp_path), gate_thr=0.99, **{k: v for k, v in CFG.items()})
    assert out["agent_call"] is False and "result=b1_gated" in out["notes"]


# ------------------------------------------------------------------ G1F-12 / G2-11: no IK seed across episodes

def test_ik_seed_cache_never_crosses_an_episode(monkeypatch):
    seeds = []

    def branching_ik(arm, pose, q0=None):                             # two IK branches fit the flange: the seed picks
        seeds.append(None if q0 is None else float(q0[0]))
        return np.full(6, -1.0 if q0 is None else float(q0[0])), 0.0, 0.0
    monkeypatch.setattr(J, "ik_flange", branching_ik)
    J._Q_LAST.clear()
    assert J.episode_begin({"id": "ep-1", "perceive_n": 1, "agent_calls": 0})
    J._state({"left": FLANGE["left"]}, {"left": 1.0})
    J._Q_LAST["left"] = np.full(6, 1.0)                               # episode 1 ends on the other branch
    assert not J.episode_begin({"id": "ep-1", "perceive_n": 2})
    st, _ = J._state({"left": FLANGE["left"]}, {"left": 1.0})
    assert seeds[-1] == 1.0 and st["left_arm_joint_state"][0] == 1.0  # within the episode: seeded as before
    assert J.episode_begin({"id": "ep-2", "perceive_n": 1})
    st, _ = J._state({"left": FLANGE["left"]}, {"left": 1.0})
    assert seeds[-1] is None and st["left_arm_joint_state"][0] == -1.0  # a new episode: independent of the last one
    assert not J.episode_begin(None) and not J.episode_begin({"perceive_n": 3})   # no id: nothing changes


def test_perceive_impl_starts_the_episode_before_anything(monkeypatch):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    J._Q_LAST["left"] = np.full(6, 1.0)
    J._EPISODE_ID[0] = "ep-old"
    out = J._perceive_impl({}, {}, FLANGE, {}, "Do a dance.", episode={"id": "ep-new", "perceive_n": 1, "agent_calls": 0})
    assert out["error"] == "unknown task for this instruction" and J._Q_LAST == {} and J._EPISODE_ID[0] == "ep-new"


# ------------------------------------------------------------------ G1F-2: effective B1 data, strict refusal

def test_data_identity_reports_effective_paths_and_refuses_fallbacks(tmp_path, monkeypatch):
    cat, bg, x5 = tmp_path / "cat", tmp_path / "bg", tmp_path / "x5"
    for d in (cat, bg, x5):
        d.mkdir()
    (cat / "index.json").write_text("{}")
    (bg / "cam_head.png").write_bytes(b"png")
    (x5 / "X5A.urdf").write_text("<robot/>")
    (tmp_path / "cc.json").write_text('{"all": []}')
    monkeypatch.setenv("DOJO_CATALOG", str(cat))
    monkeypatch.setenv("DOJO_COLOR_CALIB", str(tmp_path / "cc.json"))
    from types import SimpleNamespace                                 # the loaders' own resolvers, without cv2 / scipy

    monkeypatch.setitem(sys.modules, "dojo.perception.render", SimpleNamespace(catalog_root=lambda: cat))
    seg = SimpleNamespace(bg_root=lambda: bg)
    monkeypatch.setitem(sys.modules, "dojo.perception.segment", seg)
    rm = SimpleNamespace(X5Model=SimpleNamespace(default=lambda: SimpleNamespace(urdf_path=x5 / "X5A.urdf")))
    monkeypatch.setitem(sys.modules, "dojo.perception.robot_model", rm)
    monkeypatch.setenv("DOJO_X5_DIR", str(tmp_path / "elsewhere"))     # configured, but the loader found another root
    monkeypatch.setenv("DOJO_BG", str(bg))
    J._DATA.clear()
    d = J.data_identity()
    assert d["catalog"]["path"] == str(cat) and d["catalog"]["fallback"] is False and len(d["catalog"]["sha256"]) == 64
    assert d["bg"]["fallback"] is False and d["color_calib"]["sha256"] and d["id"].startswith("b1d-")
    assert d["x5"]["path"] == str(x5) and d["x5"]["fallback"] is True
    monkeypatch.setenv("J1_REQUIRE_DATA", "0")
    J.require_data()                                                  # report only
    monkeypatch.setenv("J1_REQUIRE_DATA", "1")
    with pytest.raises(RuntimeError, match="x5"):
        J.require_data()
    with pytest.raises(RuntimeError, match="J1_REQUIRE_DATA"):        # the agentic route refuses too (not a B1 failure
        j1.perceive_impl(_frames(), _calib(), FLANGE, {}, "Stack.", None, b1=_b1(), chat=FakeChat([]),  # it runs without)
                         log_dir=str(tmp_path), **CFG)
    monkeypatch.setenv("DOJO_X5_DIR", str(x5))
    J._DATA.clear()
    J.require_data()                                                  # every root as configured: no refusal
    J._DATA.clear()


# ------------------------------------------------------------------ G1F-13: the fake J2 server's diagnostics

def test_fake_server_reports_executed_prefix_diagnostics(tmp_path):
    from dojo.bridge.fake_server import SCENES, FakeStrictServer
    from dojo.skills.env import KinematicEnv

    instr, rows = SCENES["stack_blocks"]
    srv = FakeStrictServer("fake-diag", tmp_path, [lambda: KinematicEnv(rows, step_lim=200, instruction=instr)],
                           success_fn=lambda env: env.steps_used >= 1)  # the latch ends the episode after entry 0
    srv.cmd_reset({})
    q = {a: np.asarray(srv.hold_q[a], float) for a in ("left", "right")}
    g = {a: float(srv.hold_g[a]) for a in ("left", "right")}

    def entry(dq, grip):
        return {f"{a}_{k}": v for a in ("left", "right")
                for k, v in (("arm_joint_state", (q[a] + dq).tolist()), ("ee_joint_state", [grip]))}
    acts = [entry(0.0, g["left"]), entry(0.05, 0.0 if g["left"] > 0.5 else 1.0)]   # entry 1: grip change while moving
    rep = srv.cmd_joint_steps({"actions": acts, "reserve": 0})
    assert rep["executed"] == 1 and rep["refused"] is None
    assert rep["grip_change_while_moving"] == {"left": 0, "right": 0} and rep["max_step_dq"] == {"left": 0.0, "right": 0.0}
    assert rep["requested_grip_change_while_moving"] == {"left": 1, "right": 1}
    assert rep["requested_max_step_dq"]["left"] == pytest.approx(0.05, abs=1e-5)
