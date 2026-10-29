"""Stream H: ``RuntimeConfig.ported_pour_pace`` / ``Skills.ported_pour_pace`` ("v1" = the vendored pour_bottle |
"lean" / "rush": ``pour_bottle_lean``, a marked copy whose tilt-up, untilt and drain are paced for three pours in
pour_by_language's 800 steps; ``planner.POUR_PACE``). The liquid's behaviour is screened in G3; here: the copy with its
defaults is the vendored skill, the paces cut steps, the planner / pour_order pass them, the default, validation. CPU
only (the adapter tests' kinematic env)."""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import planner, pour_order
from dojo.skills.ported import TIER_A, ported

from test_pour_aim import _Assets, _world
from test_port_adapter import gt_skills, rows_bottle

INSTR = "Pour the liquid from the bottle into the cup."
SKILLS_DIR = Path(planner.__file__).parent / "ported" / "skills"


def _pour(skill, args):
    env, sk = gt_skills(rows_bottle(), INSTR, step_lim=800)
    r = ported(sk, skill, args)
    run = sk.ported_runs[-1]
    return r, run, json.loads(run["details"])


def test_the_copy_with_its_defaults_is_the_vendored_skill():
    r0, run0, d0 = _pour("pour_bottle", {})
    r1, run1, d1 = _pour("pour_bottle_lean", {})
    assert r0["verified"] and r1["verified"] and r0["steps"] == r1["steps"]
    assert [(c["cmd"], c["steps"]) for c in run0["calls_tail"]] == [(c["cmd"], c["steps"]) for c in run1["calls_tail"]]
    assert [p["phi"] for p in d0["pour"]] == [p["phi"] for p in d1["pour"]]


def test_every_change_in_the_copy_is_marked():
    old = (SKILLS_DIR / "pour_bottle" / "skill_robodojo.py").read_text().splitlines()[2:]     # its 2-line attribution
    new = (SKILLS_DIR / "pour_bottle_lean" / "skill_robodojo.py").read_text().splitlines()
    head = next(i for i, ln in enumerate(new) if ln.startswith('"""'))                       # the H header above it
    sm = difflib.SequenceMatcher(a=old, b=new[head:], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or j1 == 0:                                       # the docstring's first line names the copy
            continue
        assert any("H:" in ln or "# H" in ln for ln in new[head + j1:head + j2]), new[head + j1:head + j2]


@pytest.mark.parametrize("pace", ["lean", "rush"])
def test_paces_cut_the_steps_and_still_run_the_whole_profile(pace):
    r0, _, _ = _pour("pour_bottle", {})
    r, _, d = _pour("pour_bottle_lean", dict(planner.POUR_PACE[pace]))
    assert r["verified"] and d["pour_stop"] == "profile end"
    phis = [p["phi"] for p in d["pour"]]
    assert phis[0] == 90 and phis[-1] == 150                               # straight to 90, then the flow angles
    assert r["steps"] <= r0["steps"] - 100                                 # kinematic env: 380 -> 269 / 257


@pytest.mark.parametrize("aim, pace, skill, back", [(None, None, "pour_bottle", None), ("b10", "v1", "pour_bottle", 0.010),
                                                    ("b10", "lean", "pour_bottle_lean", 0.010),
                                                    ("v1", "rush", "pour_bottle_lean", None)])
def test_planner_passes_the_pace(aim, pace, skill, back):
    w = _world()
    opts = {k: v for k, v in (("pour_aim", aim), ("pour_pace", pace)) if v is not None}
    p = planner.ported_plan(w.instruction, w, opts=opts)
    assert p.args["skill"] == skill and skill in TIER_A
    assert p.args["args"].get("spout_back") == back
    lean = planner.POUR_PACE.get(pace or "v1")
    assert {k: v for k, v in p.args["args"].items() if k != "spout_back"} == (lean or {})


def test_pour_order_pours_with_the_pace(monkeypatch):
    calls = []

    def fake(sk, skill, args, reserve=None):
        calls.append((skill, args))
        return {"ok": True, "status": "ok", "verified": True, "steps": 10}
    import dojo.skills.ported as ported_mod
    monkeypatch.setattr(ported_mod, "ported", fake)

    class Sk:
        ported_pour_aim, ported_pour_pace = "b10", "lean"

        def refresh(self):
            return None

        def log(self, m):
            pass
    assert pour_order.pour_in_order(Sk(), [["b0", "w0"], ["b1", "w1"]])["ok"]
    assert [c[0] for c in calls] == ["pour_bottle_lean"] * 2
    assert calls[0][1]["spout_back"] == 0.010 and calls[0][1]["rush_to"] == 90 and calls[1][1]["container"] == "@b1"


def test_default_and_validation():
    assert RuntimeConfig().ported_pour_pace == "v1" and planner.POUR_PACE["v1"] is None
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", ported_pour_pace="fast"), assets=_Assets())
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["ported_pour_pace"][0] == RuntimeConfig().ported_pour_pace
