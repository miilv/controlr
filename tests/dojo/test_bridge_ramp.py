"""ramp_jumps: plans that start from the MEASURED joints (which lag the command after contact) would make the strict v2
shim refuse the whole joint_steps list (a > 0.2 rad jump from hold_q, the last executed entry). The backend inserts
velocity-respecting ramp entries instead, counted against the budget. Found in the D-C19 rehearsal (fill_egg_holder,
fill_pen_holder, insert_key). Synthetic: the shim's rule re-stated below, plus the fake strict server."""

import numpy as np
import pytest
from test_bridge import bench, serve  # noqa: F401  (bench: the short-path tmp fixture)

from dojo.bridge.backend import ACTION_KEYS, ARM_KEYS, RAMP_DQ, SERVER_MAX_DQ, ramp_jumps


def shim_jump_ok(hold_q: dict, actions: list[dict], max_dq: float = SERVER_MAX_DQ) -> int | None:
    """robot-harness 7fac1253 rd_shim_strict_v2/deploy.py validate_joint_steps, the jump rule: per arm and joint,
    |q_i - q_(i-1)| <= max_dq (+1e-6 float32 slack), q_(-1) = hold_q. -> the first refused index, or None."""
    prev = {k: np.asarray(hold_q[k], np.float64) for k in ARM_KEYS}
    for i, a in enumerate(actions):
        q = {k: np.asarray(np.asarray(a[k], np.float32), np.float64) for k in ARM_KEYS}
        if any(np.max(np.abs(q[k] - prev[k])) > max_dq + 1e-6 for k in ARM_KEYS):
            return i
        prev = q
    return None


def _a(lq, rq, lg=1.0, rg=1.0):
    return {"left_arm_joint_state": np.asarray(lq, np.float32), "left_ee_joint_state": np.array([lg], np.float32),
            "right_arm_joint_state": np.asarray(rq, np.float32), "right_ee_joint_state": np.array([rg], np.float32)}


def test_ramp_makes_any_plan_legal_and_keeps_every_entry():
    rng = np.random.default_rng(7)
    hold = _a(rng.normal(size=6), rng.normal(size=6))
    plan, q = [], {k: np.asarray(hold[k], np.float64) for k in ARM_KEYS}
    for i in range(60):
        for k in ARM_KEYS:
            q[k] = q[k] + rng.normal(scale=0.02, size=6) + (rng.uniform(-0.9, 0.9, 6) if i % 17 == 0 else 0)
        plan.append(_a(q["left_arm_joint_state"], q["right_arm_joint_state"], lg=float(i % 2), rg=0.5))
    assert shim_jump_ok(hold, plan) is not None                  # as planned: refused
    out, ramps = ramp_jumps(hold, plan)
    assert shim_jump_ok(hold, out) is None and ramps
    it = iter(out)                                               # every planned entry, in order, untouched
    assert all(any(e is p for e in it) for p in plan)
    assert len(out) == len(plan) + sum(r["n"] for r in ramps)
    for r in ramps:                                              # grippers hold the previous command during a ramp
        j = r["at"] + sum(x["n"] for x in ramps if x["at"] < r["at"])
        prev = hold if r["at"] == 0 else plan[r["at"] - 1]
        for e in out[j:j + r["n"]]:
            assert all(np.array_equal(e[k], prev[k]) for k in ACTION_KEYS if k not in ARM_KEYS)
    assert ramp_jumps(hold, plan[:0]) == ([], []) and ramp_jumps(None, plan[:1]) == (plan[:1], [])


def test_the_backend_ramps_from_hold_q_and_counts_the_steps(bench):  # noqa: F811
    srv, be, envs = serve(bench, sid="r1")
    be.connect()
    be.reset()
    logs = []
    be.log = logs.append
    q0 = {k: np.asarray(be.observe()["state"][k], np.float64) for k in ARM_KEYS}
    ok = [_a(q0["left_arm_joint_state"] + 0.01 * i, q0["right_arm_joint_state"]) for i in range(1, 4)]
    assert be.execute(ok)["steps"] == 3 and be.stats["ramps"] == 0  # small moves: verbatim
    held = {k: np.asarray(srv.hold_q[k.split("_")[0]], np.float64) for k in ARM_KEYS}
    # the next plan starts 0.5 rad off hold_q (planned from lagging measured joints) and jumps 0.45 inside
    start = q0["left_arm_joint_state"] + np.r_[0, 0, 0, 0.5, 0, 0]
    plan = [_a(start, q0["right_arm_joint_state"]), _a(start + 0.01, q0["right_arm_joint_state"]),
            _a(start + np.r_[0, 0.45, 0, 0, 0, 0], q0["right_arm_joint_state"])]
    assert shim_jump_ok(held, plan) == 0
    n_before = len(be.executed)
    r = be.execute(plan)
    sent = be.executed[n_before:]
    assert shim_jump_ok(held, sent) is None and len(sent) == r["steps"]
    assert be.stats["ramps"] == 2 and be.stats["ramp_steps"] == r["steps"] - 3 == 2 + 2
    assert envs[0].steps_used == be.steps_used == len(be.executed)  # no hidden steps
    assert any("ramp at step 3" in m and "from hold_q" in m for m in logs)
    # a huge jump: logged as a warning, still ramped, nothing dropped
    far = _a(np.asarray(sent[-1]["left_arm_joint_state"]) + 2.0, q0["right_arm_joint_state"])
    r = be.execute([far])
    assert r["steps"] == 11 and any(m.startswith("[bridge] WARNING large ramp") for m in logs)
    assert be.stats["ramp_max_rad"] == pytest.approx(2.0, abs=1e-5)


def test_ramp_off_sends_verbatim(bench):  # noqa: F811
    srv, be, envs = serve(bench, sid="r2")
    be.connect()
    be.reset()
    be.ramp = False
    q0 = {k: np.asarray(be.observe()["state"][k], np.float64) for k in ARM_KEYS}
    from dojo.bridge.client import ServerError
    with pytest.raises(ServerError, match="jump at entry 0"):
        be.execute([_a(q0["left_arm_joint_state"] + 0.5, q0["right_arm_joint_state"])])
    assert envs[0].steps_used == 0 and be.stats["ramps"] == 0
    assert RAMP_DQ < SERVER_MAX_DQ
