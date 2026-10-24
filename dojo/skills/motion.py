"""Motion layer: straight-line Cartesian paths and joint moves on our own IK (PyRUA-style).

Why not cuRobo (``dojo/simlab/motion.py``): RoboDojo's cuRobo world is only a table cuboid at
z = 0.74 (25 mm below the real top) with no objects, so it buys no collision safety; it needs
the GPU process; and it picks its own IK branch. The criterion is env steps, and the step cost
of a path is set by its joint travel and the per-step joint budget, which both backends share.
``compare_curobo`` in ``g3.py`` measures it.

Step budget (RoboDojo ``eval_env.take_action``): a joint action is linearly interpolated from
the MEASURED joints over 8 of the 10 physics substeps (40 ms); there is no clip, but the X5
actuator caps joint speed at 5 rad/s, i.e. 0.2 rad per env step. Paths are cut so no joint moves
more than ``step_rad`` per env step; ``step_rad`` is set per segment: fast in free space, slow
near contact.

Every plan is a ``MoveResult``: status ``reached`` / ``partial`` / ``ik_fail`` / ``collision``,
the joint waypoints by segment, the exact env-step cost, the TCP pose reached, and reachable
``alternatives`` when the target is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dojo.skills.geometry import T_from, angle_between_R, slerp_R
from dojo.skills.kinematics import TABLE_Z, X5Kin, down_R, tilted_R
from dojo.skills.world import ObjectState, WorldState, table_contains_xy

# step sizes (rad per env step, max over joints). Twin probe (g3 probe, 2026-10-24): landing error right
# after the last step 1 / 2 / 3-5 mm at 0.05 / 0.12 / 0.2 rad per step, < 1 mm one step later, 0.1-0.4 mm
# two steps later; 10 cm vertical lines deviate <= 0.16 mm in xy at any step size. So: fast in free
# space, slow only for the last centimetres before contact, then 1-2 settle steps.
FAST = 0.18
CARRY = 0.14                  # transit while holding an object (with ease)
MEDIUM = 0.10
SLOW = 0.05
EASE = 2
LINE_STEP_M = 0.02
LINE_STEP_DEG = 8.0
MAX_WAYPOINT_JUMP = 0.6        # rad between consecutive line waypoints = branch flip


@dataclass
class Segment:
    q: np.ndarray                 # (K, 6) joint waypoints, q[0] = start
    step_rad: float
    name: str = ""
    ease: int = 0                 # ramp steps at each end (trapezoidal speed profile)

    def n_steps(self) -> int:
        return len(segment_targets(self.q, self.step_rad, self.ease))


@dataclass
class MoveResult:
    status: str                                   # reached | partial | ik_fail | collision
    arm: str
    segments: list[Segment] = field(default_factory=list)
    T_reached: np.ndarray | None = None
    pos_err: float = 0.0
    rot_err: float = 0.0
    note: str = ""
    alternatives: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "reached"

    @property
    def steps(self) -> int:
        return sum(s.n_steps() for s in self.segments)

    @property
    def q_end(self) -> np.ndarray | None:
        return self.segments[-1].q[-1] if self.segments else None

    def then(self, other: "MoveResult") -> "MoveResult":
        if not other.segments:
            return MoveResult(other.status if not other.ok else self.status, self.arm, list(self.segments),
                              other.T_reached if other.T_reached is not None else self.T_reached,
                              other.pos_err, other.rot_err, (self.note + "; " + other.note).strip("; "),
                              other.alternatives)
        return MoveResult(other.status if self.ok else self.status, self.arm, self.segments + other.segments,
                          other.T_reached, other.pos_err, other.rot_err, (self.note + "; " + other.note).strip("; "),
                          other.alternatives)


def segment_steps(q: np.ndarray, step_rad: float, ease: int = 0) -> int:
    return len(segment_targets(q, step_rad, ease))


def segment_targets(q: np.ndarray, step_rad: float, ease: int = 0) -> list[np.ndarray]:
    """Per-env-step joint targets (excluding the start), no joint moving more than ``step_rad``
    in one step. ``ease`` > 0: the per-step advance ramps up over the first ``ease`` steps and down
    over the last ones (a held object slides in the pads on an abrupt 1.5 m/s stop; twin trace
    2026-10-24), measured along the path in the max-joint norm."""
    q = np.asarray(q, float)
    if len(q) < 2:
        return []
    d = np.abs(np.diff(q, axis=0)).max(axis=1)
    keep = np.r_[True, d > 1e-9]
    q, d = q[keep], d[d > 1e-9]
    if len(q) < 2:
        return []
    if ease <= 0:
        out = []
        for a, b, di in zip(q[:-1], q[1:], d):
            n = max(1, math.ceil(di / step_rad - 1e-9))
            for i in range(1, n + 1):
                out.append(a + (b - a) * i / n)
        return out
    s_cum = np.r_[0.0, np.cumsum(d)]
    total = float(s_cum[-1])
    ramp = [step_rad * (k + 1) / (ease + 1) for k in range(ease)]
    # trapezoid: ramp up, cruise, ramp down (shrink the ramps for short moves)
    up = []
    acc = 0.0
    for r in ramp:
        if 2 * (acc + r) > total:
            break
        up.append(r)
        acc += r
    cruise = total - 2 * acc
    n_c = max(0, math.ceil(cruise / step_rad - 1e-9))
    incs = up + ([cruise / n_c] * n_c if n_c else []) + up[::-1]
    if not incs:
        incs = [total]
    out = []
    s = 0.0
    for inc in incs:
        s = min(total, s + inc)
        j = min(int(np.searchsorted(s_cum, s, side="right")) - 1, len(d) - 1)
        t = (s - s_cum[j]) / d[j]
        out.append(q[j] + (q[j + 1] - q[j]) * min(1.0, t))
    out[-1] = q[-1].copy()
    return out


# --------------------------------------------------------------------------- collision model
@dataclass
class HeldBox:
    """A held object as an OBB in the TCP frame (for sweep checks)."""
    T_tcp_obj: np.ndarray         # object root pose in the TCP frame
    center: np.ndarray            # bbox centre, object frame
    half: np.ndarray

    def corners_w(self, T_tcp: np.ndarray, shrink: float = 0.0) -> np.ndarray:
        T = T_tcp @ self.T_tcp_obj
        h = np.maximum(self.half - shrink, 0.0)
        c = np.array([[x, y, z] for x in (-h[0], 0, h[0]) for y in (-h[1], 0, h[1]) for z in (-h[2], 0, h[2])])
        c = c + self.center
        return (T[:3, :3] @ c.T).T + T[:3, 3]


class Collision:
    """Coarse checks: table clearance for the arm and fingers, gripper/arm/held-object points
    against inflated object OBBs. No self-collision model (X5 IK solutions stay near home)."""

    def __init__(self, kin: X5Kin, world: WorldState | None = None, link_margin: float = 0.015,
                 finger_margin: float = 0.002, obj_margin: float = 0.006):
        self.kin = kin
        self.world = world
        self.link_margin = link_margin
        self.finger_margin = finger_margin
        self.obj_margin = obj_margin

    def obstacles(self, ignore=()) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        if self.world is None:
            return []
        out = []
        for o in self.world.obstacles(exclude=set(ignore)):
            c, R, h = o.obb(self.obj_margin)
            out.append((c, R, h, o.label))
        return out

    def touching(self, arm: str, q, gripper: float = 1.0, ignore=(), held: HeldBox | None = None) -> set:
        """Objects the arm already overlaps (within margin) at ``q``: a move away from them must be allowed."""
        out = set()
        obs = self.obstacles(ignore)
        while True:
            ok, why = self.check(arm, q, gripper, ignore=tuple(set(ignore) | out), held=held, table=False,
                                 obstacles=[o for o in obs if o[3] not in out])
            if ok or not why.startswith("hits "):
                return out
            out.add(why[5:])

    def check(self, arm: str, q, gripper: float = 1.0, ignore=(), held: HeldBox | None = None,
              obstacles=None, table: bool = True) -> tuple[bool, str]:
        fr = self.kin.frames(arm, q)
        Tf = fr[-1]
        fingers = self.kin.finger_points(Tf, gripper)
        palm = self.kin.palm_points(Tf)
        links = self.kin.link_points(arm, q, frames=fr)
        if table:
            for pts_, m, what in ((links, self.link_margin, "link"), (palm, self.finger_margin + 0.004, "palm"),
                                  (fingers, self.finger_margin, "fingertip")):
                for p in pts_:
                    if table_contains_xy(p) and TABLE_Z - 0.3 < p[2] < TABLE_Z + m:
                        return False, f"{what} at table z={p[2]:.3f}"
        obs = self.obstacles(ignore) if obstacles is None else obstacles
        if obs:
            pts = np.vstack([links, palm, fingers])
            if held is not None:
                pts = np.vstack([pts, held.corners_w(self.kin.tcp_from_flange(Tf), shrink=0.002)])
            for c, R, h, lab in obs:
                loc = (pts - c) @ R
                inside = np.all(np.abs(loc) <= h, axis=1)
                if inside.any():
                    return False, f"hits {lab}"
        if held is not None and table:
            hc = held.corners_w(self.kin.tcp_from_flange(Tf), shrink=0.002)
            for p in hc:
                if table_contains_xy(p) and TABLE_Z - 0.3 < p[2] < TABLE_Z - 0.003:
                    return False, "held object below the table"
        return True, ""


# --------------------------------------------------------------------------- planner
class Motion:
    def __init__(self, kin: X5Kin | None = None, coll: Collision | None = None):
        self.kin = kin or X5Kin()
        self.coll = coll or Collision(self.kin)

    # ---- IK
    def ik(self, arm: str, T_tcp, q_current, **kw):
        return self.kin.ik(arm, T_tcp, q_current, **kw)

    # ---- straight Cartesian line
    def line(self, arm: str, q0, T_goal, step_rad: float = SLOW, name: str = "line", gripper: float = 1.0,
             ignore=(), held: HeldBox | None = None, check: bool = True, table: bool = True,
             max_step_m: float = LINE_STEP_M, max_step_deg: float = LINE_STEP_DEG) -> MoveResult:
        q0 = np.asarray(q0, float)
        T0 = self.kin.tcp(arm, q0)
        T_goal = np.asarray(T_goal, float)
        dist = float(np.linalg.norm(T_goal[:3, 3] - T0[:3, 3]))
        ang = math.degrees(angle_between_R(T0[:3, :3], T_goal[:3, :3]))
        n = max(1, math.ceil(dist / max_step_m), math.ceil(ang / max_step_deg))
        qs = [q0]
        obs = self.coll.obstacles(ignore) if check else []
        start_touch = self.coll.touching(arm, q0, gripper, ignore, held) if check else set()
        obs_far = [o for o in obs if o[3] not in start_touch]
        q = q0
        for i in range(1, n + 1):
            t = i / n
            T = T_from(T0[:3, 3] + t * (T_goal[:3, 3] - T0[:3, 3]), slerp_R(T0[:3, :3], T_goal[:3, :3], t))
            r = self.kin.solve(arm, T, q)
            if not r.ok:
                r2 = self.kin.solve(arm, T, q, iters=300)
                r = r2 if r2.ok else r
            if not r.ok:
                return self._partial(arm, qs, step_rad, name, "ik_fail", f"line waypoint {i}/{n} unreachable "
                                     f"(err {r.pos_err*1000:.1f} mm / {math.degrees(r.rot_err):.1f} deg)", T_goal)
            if float(np.abs(r.q - q).max()) > MAX_WAYPOINT_JUMP:
                return self._partial(arm, qs, step_rad, name, "ik_fail", f"branch flip at waypoint {i}/{n}", T_goal)
            if check:
                near_start = t * dist < 0.05 and t * ang < 20
                ok, why = self.coll.check(arm, r.q, gripper, ignore=ignore, held=held,
                                          obstacles=obs_far if near_start else obs, table=table)
                if not ok:
                    return self._partial(arm, qs, step_rad, name, "collision", f"waypoint {i}/{n}: {why}", T_goal)
            q = r.q
            qs.append(q)
        Tr = self.kin.tcp(arm, q)
        return MoveResult("reached", arm, [Segment(np.asarray(qs), step_rad, name)], Tr,
                          float(np.linalg.norm(Tr[:3, 3] - T_goal[:3, 3])), angle_between_R(Tr[:3, :3], T_goal[:3, :3]))

    def _partial(self, arm, qs, step_rad, name, status, note, T_goal) -> MoveResult:
        segs = [Segment(np.asarray(qs), step_rad, name)] if len(qs) > 1 else []
        Tr = self.kin.tcp(arm, qs[-1])
        res = MoveResult(status if len(qs) <= 1 else "partial", arm, segs, Tr,
                         float(np.linalg.norm(Tr[:3, 3] - T_goal[:3, 3])), angle_between_R(Tr[:3, :3], T_goal[:3, :3]),
                         f"{status}: {note}")
        if status == "ik_fail":
            res.alternatives = self.alternatives(arm, qs[-1], T_goal)
        return res

    # ---- joint-space move
    def joint(self, arm: str, q0, q1, step_rad: float = FAST, name: str = "joint", gripper: float = 1.0,
              ignore=(), held: HeldBox | None = None, check: bool = True, table: bool = True,
              ease: int = EASE) -> MoveResult:
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        if check:
            obs = self.coll.obstacles(ignore)
            start_touch = self.coll.touching(arm, q0, gripper, ignore, held)
            obs_far = [o for o in obs if o[3] not in start_touch]
            n = max(2, math.ceil(float(np.abs(q1 - q0).max()) / 0.08))
            for t in np.linspace(0, 1, n + 1)[1:]:
                ok, why = self.coll.check(arm, q0 + t * (q1 - q0), gripper, ignore=ignore, held=held,
                                          obstacles=obs_far if t <= 0.3 else obs, table=table)
                if not ok:
                    return MoveResult("collision", arm, [], self.kin.tcp(arm, q0), note=f"joint move at t={t:.2f}: {why}")
        T = self.kin.tcp(arm, q1)
        return MoveResult("reached", arm, [Segment(np.stack([q0, q1]), step_rad, name, ease)], T)

    def to_pose(self, arm: str, q0, T_goal, step_rad: float = FAST, name: str = "move", gripper: float = 1.0,
                ignore=(), held: HeldBox | None = None, check: bool = True, safe_z: float | None = None,
                ease: int = EASE) -> MoveResult:
        """Free-space move to a TCP pose: a straight joint move to the closest IK solution if it is
        collision-free; otherwise up to ``safe_z`` (line), across in joint space, down (line)."""
        r = self.kin.ik(arm, T_goal, q0)
        if not r.ok:
            return MoveResult("ik_fail", arm, [], self.kin.tcp(arm, q0), r.pos_err, r.rot_err,
                              f"target unreachable ({r.note})", self.alternatives(arm, q0, T_goal))
        mv = self.joint(arm, q0, r.q, step_rad, name, gripper, ignore, held, check, ease=ease)
        if mv.ok or not check:
            return mv
        # via a safe height (try a few)
        T0 = self.kin.tcp(arm, q0)
        base_z = max(T0[2, 3], T_goal[2, 3])
        heights = [safe_z] if safe_z is not None else [base_z + 0.06, base_z + 0.03, base_z + 0.10]
        last = mv.note
        for z in heights:
            out = self._via(arm, q0, T0, np.asarray(T_goal, float), z, step_rad, name, gripper, ignore, held, check,
                            ease)
            if out.ok:
                return out
            last = out.note
        return MoveResult("collision", arm, [], T0, note=f"{mv.note}; via failed: {last}")

    def _via(self, arm, q0, T0, T_goal, z, step_rad, name, gripper, ignore, held, check, ease) -> MoveResult:
        up = T0.copy()
        up[2, 3] = max(z, T0[2, 3])
        above = T_goal.copy()
        above[2, 3] = max(z, T_goal[2, 3])
        m1 = self.line(arm, q0, up, step_rad=MEDIUM, name=name + "_up", gripper=gripper, ignore=ignore, held=held)
        if not m1.ok:
            return MoveResult("collision", arm, [], T0, note=f"lift: {m1.note}")
        r2 = self.kin.ik(arm, above, m1.q_end)
        if not r2.ok:
            return MoveResult("ik_fail", arm, [], T0, note=f"via-pose z={z:.3f} unreachable")
        m2 = self.joint(arm, m1.q_end, r2.q, step_rad, name + "_across", gripper, ignore, held, check, ease=ease)
        if not m2.ok:
            m2 = self.line(arm, m1.q_end, above, step_rad=step_rad, name=name + "_across", gripper=gripper,
                           ignore=ignore, held=held, max_step_m=0.04)
            if not m2.ok:
                return MoveResult("collision", arm, [], T0, note=f"across: {m2.note}")
        out = m1.then(m2)
        if above[2, 3] - T_goal[2, 3] > 1e-4:
            out = out.then(self.line(arm, m2.q_end, T_goal, step_rad=MEDIUM, name=name + "_down", gripper=gripper,
                                     ignore=ignore, held=held))
        return out

    # ---- reachable alternatives (fed back when a target is out of reach)
    def alternatives(self, arm: str, q0, T_goal, max_n: int = 3) -> list[dict]:
        out = []
        T_goal = np.asarray(T_goal, float)
        p = T_goal[:3, 3]
        x = T_goal[:3, 0]
        yaw = math.atan2(T_goal[1, 1], T_goal[0, 1])
        heading = math.atan2(p[1] - self.kin.base(arm)[1, 3], p[0] - self.kin.base(arm)[0, 3])
        tries = []
        if x[2] < -0.95:                         # pointing down: tilt the approach outwards
            for pitch in (70, 55, 45):
                tries.append((f"pitch {pitch}", T_from(p, tilted_R(yaw, pitch, heading))))
        for dz in (-0.02, 0.02, -0.04):
            T = T_goal.copy()
            T[2, 3] += dz
            tries.append((f"z {dz:+.2f}", T))
        for note, T in tries:
            r = self.kin.ik(arm, T, q0, n_random=1)
            if r.ok:
                out.append({"note": note, "T": T})
                if len(out) >= max_n:
                    break
        return out

    # ---- helpers
    def down(self, xyz, yaw: float) -> np.ndarray:
        return T_from(xyz, down_R(yaw))


def merge_steps(per_arm: dict[str, list[np.ndarray]], hold: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
    """Zip per-arm step targets; an arm that finishes early holds its last target."""
    n = max((len(v) for v in per_arm.values()), default=0)
    out = []
    last = {a: np.asarray(q, float) for a, q in hold.items()}
    for i in range(n):
        st = {}
        for a in last:
            seq = per_arm.get(a) or []
            if i < len(seq):
                last[a] = seq[i]
            st[a] = last[a]
        out.append(st)
    return out


def min_arm_distance(kin: X5Kin, qL, qR) -> float:
    """Closest pair of sampled points (chain, palm, fingers) of the two arms. The links are ~7-8 cm
    thick, so a clearance threshold must be >= ~0.10 m."""
    fl, fr = kin.flange("left", qL), kin.flange("right", qR)
    pl = np.vstack([kin.link_points("left", qL), kin.palm_points(fl), kin.finger_points(fl)])
    pr = np.vstack([kin.link_points("right", qR), kin.palm_points(fr), kin.finger_points(fr)])
    d = np.linalg.norm(pl[:, None, :] - pr[None, :, :], axis=2)
    return float(d.min())
