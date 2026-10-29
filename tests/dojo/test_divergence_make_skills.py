"""A divergence found while building fresh ``Skills`` (their first world read: ``provider.refresh`` -> resync) is a
re-plan, never an escaped exception (camera sweep_blocks_random 100005 / 07 / 09 ended as "error")."""

from __future__ import annotations

from types import SimpleNamespace

from dojo.runtime.belief import Divergence
from dojo.runtime.policy import MAKE_SKILLS_TRIES, RuntimeConfig, RuntimePolicy
from dojo.runtime.supervisor import Decision


class _Sk:
    def __init__(self, name):
        self.name, self.history, self.finished = name, [], 0

    def refresh(self):
        return SimpleNamespace(step=0, notes={}, steps_left=500)

    def finish(self):
        self.finished += 1
        return SimpleNamespace(status="ok")


class _Sup:
    """Run the skill until it succeeds, at most ``n`` decisions."""

    def __init__(self, n=4):
        self.n = n

    def decide(self, instruction, world, events):
        self.n -= 1
        return Decision("run", skill="x", params={}) if self.n >= 0 else Decision("finish", reason="done")


def _ctx(make_skills, skill):
    events = []
    return SimpleNamespace(bridge=SimpleNamespace(steps_used=7), events=events, instruction="", sk=_Sk("first"),
                           make_skills=make_skills, library=SimpleNamespace(skill_fn=lambda name: skill)), events


def _policy(sup):
    pol = RuntimePolicy(RuntimeConfig())
    pol.supervisor = sup
    pol.bridge = SimpleNamespace(steps_used=7)
    pol.report = {"decisions": [], "skill": None}
    return pol


def test_a_divergent_world_read_while_making_skills_is_a_replan():
    calls = {"n": 0}

    def make_skills(prev):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Divergence(["cube_2"], {})
        return _Sk(f"fresh{calls['n']}")

    tries = {"n": 0}

    def skill(sk):
        tries["n"] += 1
        if tries["n"] == 1:
            raise Divergence(["cube_2"], {})          # the skill's own divergence: today's re-plan path
        return {"ok": True}

    ctx, events = _ctx(make_skills, skill)
    res = _policy(_Sup())._supervise(ctx)
    assert res is not None and res["ok"]
    assert [e.get("at") for e in events if e["kind"] == "divergence_replan"] == [None, "make_skills"]


def test_every_fresh_skills_diverging_stops_safely():
    def make_skills(prev):
        raise Divergence(["broom"], {})

    def skill(sk):
        return {"ok": False, "stage": "sweep"}

    ctx, events = _ctx(make_skills, skill)
    res = _policy(_Sup())._supervise(ctx)
    assert res == {"ok": False, "stage": "sweep"}                       # what was done is kept
    assert sum(e.get("at") == "make_skills" for e in events) == MAKE_SKILLS_TRIES
    assert any(e["kind"] == "divergence_stop" for e in events) and ctx.sk.finished == 1
