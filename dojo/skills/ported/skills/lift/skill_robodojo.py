# Vendored unchanged from robot-harness (Mikhail's side) a21dd6f6f2db323bd0fe84f8afd20312cb8dffc6, skills/lift/skill_robodojo.py.
# Ported into controlr by stream P (2026-10-28) through dojo/skills/ported/harness_robot.py; logic not modified.
"""lift (RoboDojo): raise an object that one arm already holds until it is `by` m above a reference height.

lift(robot, arm="auto", by=0.10, obj=None, z_ref=None, margin=0.03, tcp_ref=None, reserve=None)
  arm     left | right | auto (auto = the one arm whose gripper is commanded closed with fingers stopped > 3 mm apart)
  by      required rise in m above the reference
  obj     object instance name; with --privileged poses the rise is measured on the object's root (mode "object"),
          otherwise on the TCP (mode "tcp")
  z_ref   reference height of the object root (default: its height now); pick_up passes the height at episode start
  tcp_ref reference TCP height for mode "tcp" (default: the TCP height now)
  margin  extra target rise in m (the benchmark latch needs a strict >, and a held object can sag a few mm)
  reserve env steps the lift must leave for a later return_home (default: the server's --reserve, 20); pass 0 when
          the lift itself is the goal and the task has no back-to-origin rule (general_pickup)
Moves (only server commands): move_to straight up keeping the tool orientation; if that is refused (IK / workspace),
the same height pulled 6 or 12 cm toward the arm's base; then the same three with the margin cut to 1 cm.
Verified = after the move the arm still holds (finger joints > 3 mm while commanded closed) AND the measured rise
(object root, or TCP without poses) >= by. A latch that ends the episode during the lift is reported as episode_ended.
"""
import numpy as np

BASES = {"left": np.array([-0.30, -0.45]), "right": np.array([0.30, -0.45])}
HELD_Q = 0.003


def _held(e):
    return bool(e and min(e.get("finger_q") or [0.0]) > HELD_Q and float(e.get("grip_cmd", 1.0)) < 0.5)


def _measure(robot, arm, obj, z_ref, tcp_ref):
    o = robot.obs(cameras=[])
    e = (o.get("eef") or {}).get(arm) or {}
    objs = o.get("objects") or {}
    if obj and obj in objs:
        rise, mode = float(objs[obj]["pos"][2]) - float(z_ref), "object"
    else:
        rise, mode = float(e.get("pos", [0, 0, tcp_ref])[2]) - float(tcp_ref), "tcp"
    return o, e, rise, mode


def run(robot, arm="auto", by=0.10, obj=None, z_ref=None, margin=0.03, tcp_ref=None, reserve=None):
    by, margin = float(by), float(margin)
    o = robot.obs(cameras=[])
    if not o.get("ok"):
        return {"ok": False, "verified": False, "error": o.get("error")}
    eef = o.get("eef") or {}
    if arm == "auto":
        holding = [a for a in ("left", "right") if _held(eef.get(a))]
        if len(holding) != 1:
            return {"ok": False, "verified": False, "error": f"cannot tell which arm holds the object (holding: {holding})",
                    "hint": "pass arm=left|right"}
        arm = holding[0]
    e = eef.get(arm) or {}
    if not _held(e):
        return {"ok": False, "verified": False, "arm": arm,
                "error": "arm is not holding anything (gripper not commanded closed or finger joints <= 3 mm)",
                "hint": "run pick first"}
    objs = o.get("objects") or {}
    tcp = np.asarray(e["pos"], float)
    if obj is not None and obj not in objs:
        obj = None                       # tier B or unknown name: fall back to the TCP
    if obj is not None:
        oz = float(objs[obj]["pos"][2])
        z_ref = oz if z_ref is None else float(z_ref)
        done = oz - z_ref
    else:
        tcp_ref = float(tcp[2]) if tcp_ref is None else float(tcp_ref)
        done = float(tcp[2]) - tcp_ref
    tries = []
    toward = BASES[arm] - tcp[:2]
    toward = toward / (np.linalg.norm(toward) + 1e-9)
    if done < by + 0.5 * margin:
        for m_ in (margin, min(margin, 0.01)):
            dz = by + m_ - done
            moved = False
            for pull in (0.0, 0.06, 0.12):
                tgt = tcp + np.array([toward[0] * pull, toward[1] * pull, dz])
                kw = {} if reserve is None else {"reserve": int(reserve)}
                r = robot.move_to(arm=arm, pos=tgt.tolist(), **kw)
                tries.append({"pull": pull, "dz": round(dz, 4), "moved": r.get("moved"), "reached": r.get("reached"),
                              "steps": r.get("steps"), "plan_fail_reason": r.get("plan_fail_reason"),
                              "residual_m": r.get("residual_m")})
                if r.get("moved") or r.get("ended"):
                    moved = True
                    break
                if r.get("plan_fail_reason") in ("budget", "lease_too_short"):
                    break                  # no other pull/height fits the remaining steps either
            if moved:
                break
    if z_ref is None:
        z_ref = 0.0
    if tcp_ref is None:
        tcp_ref = float(tcp[2])
    o, e, rise, mode = _measure(robot, arm, obj, z_ref, tcp_ref)
    held = _held(e)
    verified = bool(held and rise >= by)
    rep = {"ok": True, "verified": verified, "arm": arm, "object": obj,
           "details": {"mode": mode, "rise_m": round(rise, 4), "by": by, "held": held, "moves": tries,
                       "episode_ended": bool(o.get("ended"))}}
    if not verified:
        rep["ok"] = bool(tries and any(t["moved"] for t in tries))
        rep["error"] = ("object slipped (fingers closed on nothing)" if not held else
                        f"rise {rise:.3f} m < {by:.3f} m" + ("" if tries and any(t["moved"] for t in tries) else " (no lift move was feasible)"))
    return rep
