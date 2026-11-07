"""deposit_coin's contact sequence (COIN2): default-off switches for a thin disc standing in a fixed holder.

Each switch is a ``Skills`` attribute, mirrored by a ``RuntimeConfig`` field and a G3 ``--skill-attr`` (first value =
default = today's behaviour). They act only inside ``precision.disc_into_slot`` / ``plan_disc_handover``:

* ``disc_align`` "walls": before the pick, the picking hand straddles the holder's two walls with its pads, closes on
  them and reads the strict-v2 official jaw value (the applied gripper control). That value follows one finger, so it
  grows with the pads' yaw error to the walls (V-shaped, ``WALL_B``) and with the hand's offset along the closing axis
  (signed, ``WALL_C``). Three squeezes at yaw -delta / 0 / +delta (more while the V is not bracketed) give the signed
  yaw error and the offset; the coin and the holder are rewritten together (the coin stands in the holder's slot, whose
  plane is the holder's long axis). A fixed holder (a Geometry object) cannot move under the squeeze. No reading (strict
  v1, G3 without ``DOJO_GRIP_SOURCE=v2``) or a fit that does not explain the readings leaves the belief as it is.
* ``disc_snap`` "on": the pick descends with the pads open around the walls, closes on them, and rises with the jaws
  closing until the fingertips clear the walls' top (``SNAP_CLEAR`` over it): the pads then close onto the coin there,
  wherever the real top is, holding ~3 mm more of the coin than the default grasp (fingertips 3 mm over the top).
* ``disc_lift`` "shallow": while the hold is shallow the lift goes in ``LIFT_STEP_M`` per env step and the carry is
  slow (``CARRY_RAD``).
* ``disc_seat`` "table": after the lift the hand moves the coin ``SEAT_OFF`` along its own plane, clear of the holder,
  and lowers it onto the table until the table has pushed it ``SEAT_DEPTH`` (same-arm) / ``SEAT_DEPTH_GIVER`` (an
  in-air handover's giver) below the TCP: a grip across the coin's widest part instead of its top. The planned in-hand
  pose moves with it.
* ``disc_release`` "staged": the insertion asks for the in-hand read at the pre-insert hover and lets go in stages
  (``RELEASE_STAGES``), the fingertips just over the slot.
* ``disc_reserve`` "on": a pick (and its retry) starts only with ``RESERVE_SAME`` / ``RESERVE_CROSS`` env steps left.

Simulator constants here were re-measured in our twin (journal 2026-11-07-dojo-coin2): readings are deterministic.
"""
from __future__ import annotations

import math

import numpy as np

from dojo.skills.executor import ArmPlan, BudgetExceeded, EpisodeEnded
from dojo.skills.geometry import R_to_quat, T_from
from dojo.skills.kinematics import TABLE_Z, gap_to_grip

TIP = 0.0126                  # m: fingertips beyond the TCP along the approach (finger model)

# --- disc_align "walls" (twin, coin stand 32.1 x 11.1 x 17.1 mm, fingertips 5 mm under its top, 6 identical reads)
WALL_IN = 0.005               # m: fingertips this far under the holder's top while squeezing its walls
WALL_GAP = 0.020              # m: pad gap around the 11.1 mm walls before the squeeze
WALL_A = 0.0347               # official value, pads square to the walls and centred
WALL_B = 0.00163              # per degree of yaw error (|psi|: 0.0423-0.0429 at 5 deg, 0.0512-0.0516 at 10 deg)
WALL_B_RANGE = (0.0010, 0.0030)  # slopes the fit accepts (0.0016-0.0021 seen across layouts / arms)
WALL_C = -0.0164              # per mm of TCP offset along its +y (closing) axis: one finger's travel
WALL_DELTA = 10.0             # deg between squeezes
WALL_MAX_PROBES = 4
WALL_FIT_TOL = 0.0015         # max residual of the V fit (official units); worse: no write
WALL_MAX_SHIFT_MM = 4.0       # a larger fitted offset is not written (the walls were not where the fit says)
WALL_MAX_YAW = 25.0           # deg searched
WALL_MIN_WRITE_DEG = 0.5      # smaller corrections are not written (the slot leaves the coin +-1.4 deg)
WALL_SETTLE = 1               # still steps after a squeeze's close before the read
WALL_CLOSE_STEPS = 3

# --- disc_snap "on"
SNAP_CLEAR = 0.0007           # m: fingertips over the holder's top when the pads close onto the coin
SNAP_IN = 0.005               # m: fingertips under the holder's top before the rise
SNAP_GAP = 0.020              # m: pad gap at the descent (around the walls)
SNAP_RISE_RAD = 0.01          # rad/step of the rise from the walls to the snap height

# --- disc_lift "shallow"
LIFT_STEP_M = 0.004           # m of lift per env step
CARRY_RAD = 0.04              # rad/step of the carry while the hold is shallow

# --- disc_seat "table"
SEAT_OFF = 0.042              # m along the coin's plane from the holder before the press
SEAT_STEP_M = 0.006           # m per env step of that move
SEAT_DEPTH = 0.010            # m: coin centre below the TCP after the press (same-arm insertion)
SEAT_DEPTH_GIVER = 0.015      # m: the giver of an in-air handover (the coin must stick out past its fingertips)
SEAT_PRESS = 0.002            # m: commanded past the expected table contact
SEAT_UP = 0.04                # m: lift after the press

# --- disc_release "staged"
RELEASE_STAGES = ((0.05, 2), (0.2, 2))   # (gripper value, steps) before the full open

# --- disc_reserve "on"
RESERVE_SAME = 80             # env steps a same-arm pick needs left (pick + carry + insert + finish)
RESERVE_CROSS = 150           # ... a pick that needs an in-air handover


def on(sk, name: str, value: str) -> bool:
    return getattr(sk, name, "off") == value


# ============================================================================ disc_align "walls"

def wall_fit(reads: dict, A: float = WALL_A, C: float = WALL_C, max_yaw: float = WALL_MAX_YAW,
             b_range=WALL_B_RANGE):
    """Fit official readings {theta_deg: value} (theta = the hand turned from its first yaw) to a V:
    ``value = k + b * |theta - apex|``. ``apex`` is the turn that squares the pads to the walls (the first yaw's error
    is ``e = -apex``); ``k`` = A + C * d_mm gives the TCP's offset along its closing axis. The slope is fitted too (twin:
    0.0016-0.0021 per degree across layouts and arms). Returns {"e", "apex", "d_mm", "b", "resid", "bracketed",
    "low_side"}. ``bracketed``: readings on both branches (on one branch apex and k trade off exactly)."""
    th = np.array(sorted(reads), float)
    r = np.array([reads[t] for t in sorted(reads)], float)
    k_min = int(np.argmin(r))
    best = None
    for apex in np.arange(-max_yaw, max_yaw + 1e-9, 0.05):
        x = np.abs(th - apex)
        M = np.stack([np.ones_like(x), x], axis=1)
        (k, b), *_ = np.linalg.lstsq(M, r, rcond=None)
        if not (b_range[0] <= b <= b_range[1]):
            continue
        res = float(np.sqrt(np.mean((M @ np.array([k, b]) - r) ** 2)))
        if best is None or res < best[0] - 1e-12:
            best = (res, float(apex), float(k), float(b))
    if best is None:
        return {"e": None, "apex": None, "d_mm": None, "b": None, "resid": None, "bracketed": False,
                "low_side": -1 if k_min == 0 else 1}
    res, apex, k, b = best
    bracketed = bool(np.any(th < apex - 0.3) and np.any(th > apex + 0.3)) and len(th) >= 3
    low = None
    if not bracketed:
        low = -1 if k_min == 0 else (1 if k_min == len(r) - 1 else (-1 if r[0] < r[-1] else 1))
    return {"e": round(-apex, 2), "apex": round(apex, 2), "d_mm": round((k - A) / C, 2), "b": round(b, 5),
            "resid": res, "bracketed": bracketed, "low_side": low}


def _rot_z_about(T, c, deg: float):
    a = math.radians(deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    out = np.asarray(T, float).copy()
    out[:3, :3] = Rz @ out[:3, :3]
    out[:3, 3] = c + Rz @ (out[:3, 3] - c)
    return out


def official_read(sk, arm: str) -> float | None:
    st = sk.env.proprio()
    v = st.get(f"{arm}_ee_joint_state_official")
    if v is None:
        return None
    return float(np.asarray(v, float).reshape(-1)[0])


def wall_frame(sk, coin, hold) -> np.ndarray | None:
    """TCP pose that squeezes ``hold``'s walls: straight down over the coin's centre, closing along the coin's believed
    normal, fingertips ``WALL_IN`` under the holder's top."""
    from dojo.skills.grasps import _frame
    from dojo.skills.precision import disc_axis
    k = disc_axis(coin)
    if k is None:
        return None
    n = np.array(coin.R[:, k], float)
    n[2] = 0.0
    if np.linalg.norm(n) < 0.5:
        return None                                    # lying flat
    n /= np.linalg.norm(n)
    R = _frame(np.array([0.0, 0.0, -1.0]), n)
    c = coin.center_w
    return T_from(np.array([c[0], c[1], hold.top_z - WALL_IN + TIP]), R)


def wall_probe(sk, obj: str, hold: str, arm: str) -> dict:
    """``disc_align`` "walls": squeeze ``hold``'s walls at several yaws, fit the readings, and rewrite ``obj`` and
    ``hold`` (``provider.set_pose``, writer "wall_probe"). Returns a log dict (``ok``, readings, fit, write)."""
    w = sk.refresh()
    coin, h = w.obj(obj), w.obj(hold)
    out = {"kind": "wall_probe", "label": obj, "holder": hold, "arm": arm, "ok": False, "reads": {}}
    T0 = wall_frame(sk, coin, h)
    if T0 is None:
        out["why"] = "no standing disc"
        return out
    c = coin.center_w.copy()
    go = gap_to_grip(WALL_GAP)
    T_pre = T0.copy()
    T_pre[2, 3] += 0.04
    to = sk.motion.to_pose(arm, sk.ex.q[arm], T_pre, 0.14, "wall_pre", gripper=go, ignore=(obj,))
    if not to.ok:
        out["why"] = f"pre {to.status}"
        return out
    dn = sk.motion.line(arm, to.q_end, T0, 0.05, "wall_down", gripper=go, ignore=(obj,))
    if not dn.ok:
        out["why"] = f"down {dn.status}"
        return out
    steps0 = sk.env.steps_used
    try:
        sk.ex.run({arm: ArmPlan(arm, sk.ex.q[arm].copy(), sk.ex.g[arm]).grip(go, 2, "open").move(to).move(dn)},
                  name=f"wall probe {hold} down")
        thetas = [0.0, WALL_DELTA, -WALL_DELTA]
        fit = None
        while thetas and len(out["reads"]) < WALL_MAX_PROBES:
            th = thetas.pop(0)
            T = _rot_z_about(T0, c, th)
            mv = sk.motion.line(arm, sk.ex.q[arm], T, 0.05, "wall_turn", gripper=go, check=False)
            if not mv.ok:
                out["why"] = f"turn {th:+.0f}: {mv.status}"
                break
            sk.ex.run({arm: ArmPlan(arm, sk.ex.q[arm].copy(), go).move(mv).grip(0.0, WALL_CLOSE_STEPS, "squeeze")
                       .hold(WALL_SETTLE)}, name=f"wall probe {th:+.0f}")
            v = official_read(sk, arm)
            sk.ex.run({arm: ArmPlan(arm, sk.ex.q[arm].copy(), 0.0).grip(go, 1, "open")}, name="wall probe open")
            if v is None:
                out["why"] = "no official jaw reading"
                break
            out["reads"][round(th, 1)] = round(v, 4)
            if not thetas:
                fit = wall_fit(out["reads"])
                if not fit["bracketed"] and len(out["reads"]) < WALL_MAX_PROBES:
                    far = (min if fit["low_side"] < 0 else max)(out["reads"])
                    thetas.append(far + fit["low_side"] * WALL_DELTA)
        up = sk.motion.line(arm, sk.ex.q[arm], _rot_z_about(T_pre, c, 0.0), 0.10, "wall_up", gripper=go, check=False)
        if up.ok:
            sk.ex.run({arm: ArmPlan(arm, sk.ex.q[arm].copy(), go).move(up)}, name="wall probe up")
    except (BudgetExceeded, EpisodeEnded) as e:
        out["why"] = f"budget: {e}"
        out["steps"] = int(sk.env.steps_used - steps0)
        return out
    out["steps"] = int(sk.env.steps_used - steps0)
    if len(out["reads"]) < 3:
        out.setdefault("why", "too few reads")
        return out
    fit = wall_fit(out["reads"])
    out["fit"] = fit
    if not fit["bracketed"] or fit["resid"] > WALL_FIT_TOL:
        out["why"] = "fit not bracketed" if not fit["bracketed"] else f"residual {fit['resid']:.4f}"
        return out
    if abs(float(fit["d_mm"])) > WALL_MAX_SHIFT_MM:
        fit = {**fit, "d_mm": 0.0}
        out["d_dropped"] = True
    out["ok"] = True
    apex = float(fit["apex"])
    # the true holder: the believed one turned by the apex about the vertical through the coin's centre, then moved
    # against the hand's closing-axis offset (the TCP sat d_mm along its +y from the walls' centre line)
    T_sq = _rot_z_about(T0, c, apex)
    shift = -T_sq[:3, 1] * (float(fit["d_mm"]) / 1000.0)
    shift[2] = 0.0
    if abs(apex) < WALL_MIN_WRITE_DEG and float(np.linalg.norm(shift)) < 0.0005:
        out["write"] = "none (square)"
        return out
    writes = {}
    setp = getattr(sk.provider, "set_pose", None)
    for lab, o in ((obj, coin), (hold, h)):
        Tn = _rot_z_about(o.T, c, apex)
        Tn[:3, 3] += shift
        writes[lab] = [round(float(x), 5) for x in np.r_[Tn[:3, 3], R_to_quat(Tn[:3, :3])]]
        if setp is not None:
            setp(lab, Tn[:3, 3], R_to_quat(Tn[:3, :3]), writer="wall_probe")
    out["write"] = writes if setp is not None else "provider has no set_pose"
    sk.refresh()
    return out


# ============================================================================ disc_snap / disc_lift (plan_pick kwargs)

def pick_kw(sk, base: dict) -> dict:
    """``plan_pick`` keyword arguments for a disc's pick out of its holder under the switches (``base`` = today's)."""
    kw = dict(base)
    if on(sk, "disc_snap", "on"):
        kw.update(snap_in=SNAP_IN + SNAP_CLEAR, snap_rad=SNAP_RISE_RAD, open_g=gap_to_grip(SNAP_GAP))
    if on(sk, "disc_lift", "shallow"):
        kw["lift_step"] = LIFT_STEP_M
    return kw


def edge_floor(sk, holder_top: float, default: float) -> float:
    """The lowest fingertip height of a disc's edge grasp: today ``default`` (the holder's top + 3 mm); with
    ``disc_snap`` the holder's top + ``SNAP_CLEAR``."""
    return holder_top + SNAP_CLEAR if on(sk, "disc_snap", "on") else default


def place_kw(sk, base: dict) -> dict:
    kw = dict(base)
    if on(sk, "disc_lift", "shallow"):
        kw["approach_rad"] = min(float(kw.get("approach_rad", CARRY_RAD)), CARRY_RAD)
    if on(sk, "disc_release", "staged"):
        kw["held_read"] = True
        kw["release_stages"] = RELEASE_STAGES
    return kw


# ============================================================================ disc_seat "table"

def seat_post_lift(sk, label: str, holder: str | None, depth: float):
    """``plan_pick``'s ``post_lift`` for ``disc_seat`` "table": a callable (arm, q, T_tcp_obj, obj) -> (moves, new
    T_tcp_obj) | None. It moves the coin ``SEAT_OFF`` along its plane at the lift height (the side farther from the
    other objects, collision-checked with the coin), presses it onto the table, holds, and lifts ``SEAT_UP``."""
    from dojo.skills.motion import FAST, held_box
    from dojo.skills.precision import disc_axis

    def post(arm, q, T_tcp_obj, obj):
        k = disc_axis(obj)
        if k is None:
            return None
        T_tcp = sk.kin.tcp(arm, q)
        T_obj = T_tcp @ T_tcp_obj
        n = T_obj[:3, :3][:, k].copy()
        n[2] = 0.0
        if np.linalg.norm(n) < 0.5:
            return None
        n /= np.linalg.norm(n)
        u = np.cross([0.0, 0.0, 1.0], n)
        r = float(np.delete(obj.asset.extents, k).max()) / 2
        c_loc = obj.asset.center
        # the coin's centre below the TCP along the approach, as planned
        e0 = float((T_tcp_obj[:3, :3] @ c_loc + T_tcp_obj[:3, 3])[0])
        if e0 <= depth + 0.001:
            return None                                       # already as deep as the press would put it
        others = [o for o in sk.world.objects.values() if o.label not in (label, holder) and not o.clutter
                  and o.category != "camera_stand"]
        box = held_box(obj, T_tcp_obj)
        best = None
        for s in (1.0, -1.0):
            p = T_tcp[:3, 3] + s * SEAT_OFF * u
            gap = min((float(np.linalg.norm(o.center_w[:2] - p[:2])) for o in others), default=1.0)
            T1 = T_tcp.copy()
            T1[:3, 3] = p
            mv = sk.motion.line(arm, q, T1, FAST, "seat_side", gripper=0.0, held=box, ignore=(label,),
                                max_step_m=SEAT_STEP_M)
            if not mv.ok:
                continue
            T2 = T1.copy()
            T2[2, 3] = TABLE_Z + r + depth - SEAT_PRESS
            dn = sk.motion.line(arm, mv.q_end, T2, 0.02, "seat_press", gripper=0.0, check=False)
            if not dn.ok:
                continue
            T3 = T2.copy()
            T3[2, 3] += SEAT_UP + SEAT_PRESS
            up = sk.motion.line(arm, dn.q_end, T3, FAST, "seat_up", gripper=0.0, check=False, max_step_m=LIFT_STEP_M)
            if not up.ok:
                continue
            if best is None or gap > best[0]:
                best = (gap, mv, dn, up)
        if best is None:
            return None
        _, mv, dn, up = best
        Tn = np.asarray(T_tcp_obj, float).copy()
        Tn[0, 3] -= (e0 - depth)                              # the table pushed the coin towards the TCP
        return [("move", mv), ("hold", 2), ("move", dn), ("hold", 3), ("grip", 0.0, 2), ("move", up)], Tn

    return post


# ============================================================================ disc_reserve "on"

def reserve_ok(sk, cross: bool) -> tuple[bool, int]:
    """Whether a pick may start: ``RESERVE_CROSS`` / ``RESERVE_SAME`` env steps left (``disc_reserve`` "on"; else
    always)."""
    if not on(sk, "disc_reserve", "on"):
        return True, -1
    left = int(sk.ex.steps_left)
    need = RESERVE_CROSS if cross else RESERVE_SAME
    return left >= need, left


__all__ = ["wall_fit", "wall_probe", "pick_kw", "place_kw", "edge_floor", "seat_post_lift", "reserve_ok"]
