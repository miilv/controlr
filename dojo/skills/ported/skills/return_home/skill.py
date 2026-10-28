# Vendored unchanged from robot-harness (Mikhail's side) a21dd6f6f2db323bd0fe84f8afd20312cb8dffc6, skills/return_home/skill.py.
# Ported into controlr by stream P (2026-10-28) through dojo/skills/ported/harness_robot.py; logic not modified.
"""return_home (RoboDojo): both arms back to their start pose, grippers open, verified by proprioception.

RoboDojo's back-to-origin check (44 of 53 task files): each arm's link6 within 0.15 m per axis and 20 deg of the pose
recorded at reset, at the same env step as the task predicate. The server's return_home opens the grippers first
(<= 0.25 of range per step, arms still), then moves both arms together on one straight joint line to the start joints
(trapezoidal timing, 0.08 rad/step empty-handed). Only server commands are used.
"""

OPEN_GAP_M = 0.8 * 0.088          # both fingers >= 80 % open (0.044 m each)
TOL_M, TOL_DEG = 0.15, 20.0


def run(robot, open=True, arms=None):
    r = robot.return_home(open=open, arms=arms or ["left", "right"])
    if not r.get("ok"):
        return {"ok": False, "verified": False, "error": r.get("error"), "details": {"return_home": r}}
    o = robot.obs(cameras=[], objects=False)
    if not o.get("ok"):
        return {"ok": False, "verified": False, "error": o.get("error"), "details": {"return_home": r}}
    checks = {}
    for arm in (arms or ["left", "right"]):
        e = o["eef"][arm]
        checks[arm] = {"dpos_max_m": e["home_dpos_max_m"], "ddeg": e["home_ddeg"], "gap_m": e["gap_m"],
                       "pose_ok": e["home_dpos_max_m"] <= TOL_M and e["home_ddeg"] <= TOL_DEG,
                       "open_ok": (e["gap_m"] >= OPEN_GAP_M) if open else True}
    verified = all(c["pose_ok"] and c["open_ok"] for c in checks.values())
    return {"ok": True, "verified": verified, "steps": r.get("steps"), "ended": o.get("ended"),
            "details": {"checks": checks, "move_steps": r.get("move_steps"), "grip_steps": r.get("grip_steps"),
                        "steps_left": o.get("steps_left")}}
