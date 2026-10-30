"""Stream H: ``RuntimeConfig.ported_pour_untilt`` / ``Skills.ported_pour_untilt`` ("direct" = today | "staged"):
pour_p1 retraces its reached profile poses back to upright at slow speed before the carry back. Kinematic env, CPU."""

import json

from dojo.runtime.policy import RuntimeConfig
from dojo.skills import planner
from dojo.skills.ported import ported
from test_port_adapter import SCENES, belief_skills, rows_pour


def _run(untilt):
    instr = SCENES["pour"][1]
    record = [r for r in rows_pour() if not r["label"].startswith("sphere")]
    env0, sk = belief_skills(record, rows_pour(), instr, step_lim=600)
    p = planner.plan(instr, sk.refresh(), ported={"pour": "full", "pour_untilt": untilt})
    assert p.args["skill"] == "pour_p1"
    ported(sk, **p.args)
    return p.args, json.loads(sk.ported_runs[-1]["details"])


def test_staged_untilt_retraces_the_profile():
    args, d = _run("staged")
    assert args["args"].get("untilt") == "staged"
    back = d["untilt_staged"]
    assert back and back[-1]["phi"] == 0 and all(b["move"]["moved"] for b in back)
    phis = [b["phi"] for b in back]
    assert phis == sorted(phis, reverse=True)
    assert d["final"]["cup_tilt"] <= 15.0


def test_direct_is_today():
    args, d = _run("direct")
    assert "untilt" not in args["args"] and "untilt_staged" not in d
    assert RuntimeConfig().ported_pour_untilt == "direct"
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["ported_pour_untilt"] == ("direct", "staged")
