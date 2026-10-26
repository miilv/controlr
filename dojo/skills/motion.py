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
    """A held object as an OBB in the TCP frame (for sweep checks); ``points`` (object frame): sparse mesh
    points used instead of the inflated box against obstacles when the shape fills its box poorly."""
    T_tcp_obj: np.ndarray         # object root pose in the TCP frame
    center: np.ndarray            # bbox centre, object frame
    half: np.ndarray
    points: np.ndarray | None = None
    # mesh points (object frame) checked against ``Collision.parts`` obstacles (a rack's thin arms pass between the box
    # corners of a held mug); None = the box corners
    dense: np.ndarray | None = None

    def dense_w(self, T_tcp: np.ndarray) -> np.ndarray:
        T = T_tcp @ self.T_tcp_obj
        return (T[:3, :3] @ self.dense.T).T + T[:3, 3]

    def points_w(self, T_tcp: np.ndarray) -> np.ndarray:
        T = T_tcp @ self.T_tcp_obj
        return (T[:3, :3] @ self.points.T).T + T[:3, 3]

    def corners_w(self, T_tcp: np.ndarray, shrink: float = 0.0) -> np.ndarray:
        T = T_tcp @ self.T_tcp_obj
        h = np.maximum(self.half - shrink, 0.0)
        c = np.array([[x, y, z] for x in (-h[0], 0, h[0]) for y in (-h[1], 0, h[1]) for z in (-h[2], 0, h[2])])
        c = c + self.center
        return (T[:3, :3] @ c.T).T + T[:3, 3]


def link_slabs(lk: dict, thick: float = 0.02, min_pts: int = 50) -> list[tuple[np.ndarray, np.ndarray]]:
    """A link's sampled points (object frame) as boxes, one per ``thick`` slab along its longest bbox axis."""
    P = np.asarray(lk.get("points") if lk.get("points") is not None else [], float).reshape(-1, 3)
    mn0, mx0 = np.asarray(lk["bbox_min"], float), np.asarray(lk["bbox_max"], float)
    if len(P) < min_pts:
        return [(mn0, mx0)]
    k = int(np.argmax(mx0 - mn0))
    edges = np.arange(mn0[k], mx0[k] + thick, thick)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = P[(P[:, k] >= a - 1e-6) & (P[:, k] <= b + 1e-6)]
        if len(sel):
            lo, hi = sel.min(0), sel.max(0)
            lo[k], hi[k] = max(a, mn0[k]), min(b, mx0[k])
            out.append((lo, hi))
    return out or [(mn0, mx0)]


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
        self.include: set = set()          # Geometry labels that are obstacles too (a keyhole block, a socket)
        # per-object margins instead of obj_margin (identical slices side by side in a rack: 13 mm between two
        # bread slices, the fingertips need 3.4 mm + clearance there)
        self.margin_of: dict = {}
        # label -> height map of its mesh (``precision.height_map``): a point inside that object's box only collides
        # below its top surface there (a coin bank's rounded back; the box alone blocked the hand over its edge)
        self.heightmaps: dict = {}
        # articulated objects whose links are modelled as slabs (one box per ~2 cm of the link's long axis) instead of
        # one box per link: an egg carton's open lid is a 4 cm shell behind the body with a lip on top, its link box
        # covers the back of the body's holes too
        self.slab_links: set = set()
        # objects modelled as many small boxes from their mesh (``mesh_parts``: label -> (centres, half sizes), object
        # frame) instead of one box: a mug rack is a pole with thin arms, its box covers every hook; the held object is
        # checked against them with its mesh points (``HeldBox.dense``)
        self.parts: dict = {}

    def obstacles(self, ignore=()) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        if self.world is None:
            return []
        out = []
        extra = [self.world.objects[l] for l in self.include if l in self.world.objects and l not in ignore]
        for o in self.world.obstacles(exclude=set(ignore)) + [o for o in extra if o.obj_type == "Geometry"]:
            m = self.margin_of.get(o.label, self.obj_margin)
            if o.label in self.parts:
                C, H = self.parts[o.label]
                R = o.R
                for c, h in zip(C, H):
                    out.append((R @ c + o.pos, R, h + m, o.label))
                continue
            links = getattr(o.asset, "links", None)
            if links and len(links) > 1:
                # an articulated object as one box per link (an open laptop's bbox is mostly air around its lid);
                # "label#link" in ``ignore`` drops one link (an egg carton's body while the lid stays an obstacle)
                R = o.R
                for name, lk in links.items():
                    if f"{o.label}#{name}" in ignore:
                        continue
                    for mn, mx in (link_slabs(lk) if o.label in self.slab_links else
                                   [(np.asarray(lk["bbox_min"]), np.asarray(lk["bbox_max"]))]):
                        c = (mn + mx) / 2
                        h = (mx - mn) / 2 + m
                        out.append((R @ c + o.pos, R, h, o.label))
                continue
            c, R, h = o.obb(m)
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
              obstacles=None, table: bool = True, grip_only=()) -> tuple[bool, str]:
        """``grip_only``: objects the held object may enter but the arm and gripper may not (a lock's
        block while its key goes in)."""
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
            pts_parts = pts
            if held is not None:          # a held object may sit a few mm off its model: inflate it
                T_tcp = self.kin.tcp_from_flange(Tf)
                pts = np.vstack([pts, held.points_w(T_tcp) if held.points is not None
                                 else held.corners_w(T_tcp, shrink=-0.005)])
                pts_parts = (np.vstack([pts_parts, held.dense_w(T_tcp)]) if held.dense is not None and self.parts
                             else pts)
            for c, R, h, lab in obs:
                loc = ((pts_parts if lab in self.parts else pts) - c) @ R
                inside = np.all(np.abs(loc) <= h, axis=1)
                if inside.any() and lab in self.heightmaps:
                    inside &= below_height_map(self.heightmaps[lab], loc, self.margin_of.get(lab, self.obj_margin))
                if inside.any():
                    return False, f"hits {lab}"
        if grip_only and self.world is not None:
            # dense hand samples: the corner samples straddle a narrow receptacle (a power strip 37 mm wide
            # passed between them while the palm sat 6 mm into it; twin: the hand stopped 8 mm above the
            # outlet, 7/10)
            pts = np.vstack([links, palm, fingers, dense_hand(self.kin, Tf, gripper)])
            for lab in grip_only:
                if lab not in self.world.objects:
                    continue
                c, R, h = self.world.objects[lab].obb(self.obj_margin * 0.5)
                if np.all(np.abs((pts - c) @ R) <= h, axis=1).any():
                    return False, f"gripper hits {lab}"
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
             max_step_m: float = LINE_STEP_M, max_step_deg: float = LINE_STEP_DEG, grip_only=()) -> MoveResult:
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
                                          obstacles=obs_far if near_start else obs, table=table, grip_only=grip_only)
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
            n = max(2, math.ceil(float(np.abs(q1 - q0).max()) / (0.04 if held is not None else 0.08)))
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


def finger_samples(kin: X5Kin, T_flange, gripper: float) -> np.ndarray:
    """Both fingers sampled on grids (world points; ``dense_hand``'s finger part)."""
    from dojo.skills.kinematics import FINGER_MAX, GRIP_SCALE
    qf = min(max(GRIP_SCALE[0] + gripper * (GRIP_SCALE[1] - GRIP_SCALE[0]), 0.0), FINGER_MAX)
    loc = []
    for x0, x1, yo, zh in ((0.075, 0.105, 0.0378, 0.028), (0.107, 0.126, 0.0134, 0.015), (0.128, 0.1576, 0.0034, 0.009)):
        for x in np.linspace(x0, x1, 4):
            for y in (0.0004, yo):
                for z in np.linspace(-zh, zh, 5):
                    loc += [(x, qf + y, z), (x, -(qf + y), z)]
    loc = np.asarray(loc)
    return (np.asarray(T_flange)[:3, :3] @ loc.T).T + np.asarray(T_flange)[:3, 3]


def below_height_map(hm: dict, loc: np.ndarray, margin: float) -> np.ndarray:
    """Points (OBB-local: object frame minus the bbox centre) at or below the object's top surface + ``margin``
    (``hm``: {"x0", "y0", "cell", "Z" (ny, nx) object-frame top z, -inf where empty, "center"})."""
    p = loc + hm["center"]
    if "R" in hm:                      # a world-vertical map: offsets from the object's root in world axes
        p = p @ np.asarray(hm["R"], float).T
    ix = np.clip(((p[:, 0] - hm["x0"]) / hm["cell"]).astype(int), 0, hm["Z"].shape[1] - 1)
    iy = np.clip(((p[:, 1] - hm["y0"]) / hm["cell"]).astype(int), 0, hm["Z"].shape[0] - 1)
    return p[:, 2] <= hm["Z"][iy, ix] + margin


def held_box(o, T_tcp_obj) -> HeldBox:
    """``HeldBox`` for object ``o`` held at ``T_tcp_obj``; irregular meshes (horizontal slices filling
    under 60 % of the bbox footprint on average: figurines, tools, keys) carry sparse mesh points."""
    hb = HeldBox(np.asarray(T_tcp_obj, float), o.asset.center, o.asset.extents / 2)
    mesh = getattr(o.asset, "mesh", None)
    if mesh is None or len(mesh[0]) < 60:
        return hb
    V = np.asarray(mesh[0], float)
    hb.dense = V[:: max(1, len(V) // 600)].copy()
    irr = getattr(o.asset, "irregular", None)      # decided on the full mesh when it was loaded (AssetDB)
    if irr is None:
        irr = mesh_irregular(V)
    if irr:
        hb.points = V[:: max(1, len(V) // 400)].copy()
    return hb


def mesh_parts(V: np.ndarray, F: np.ndarray, cell: float = 0.02, step: float = 0.004,
               min_half: float = 0.002) -> tuple[np.ndarray, np.ndarray]:
    """A mesh as small boxes (object frame): its surface sampled every ``step``, clustered on a ``cell`` grid, one box
    per cluster around its points (``Collision.parts``; a mug rack's pole and thin hook arms)."""
    from dojo.skills.insertion import _surface_samples
    P = _surface_samples(np.asarray(V, float), np.asarray(F, int), step=step)
    key = np.floor((P - P.min(0)) / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = int(inv.max()) + 1
    lo = np.full((n, 3), np.inf)
    hi = np.full((n, 3), -np.inf)
    np.minimum.at(lo, inv, P)
    np.maximum.at(hi, inv, P)
    return (lo + hi) / 2, np.maximum((hi - lo) / 2, min_half)


def dense_hand(kin: X5Kin, T_flange, gripper: float) -> np.ndarray:
    """Palm / finger-rail box and fingers sampled on grids (world points)."""
    from dojo.skills.kinematics import FINGER_MAX, GRIP_SCALE
    qf = min(max(GRIP_SCALE[0] + gripper * (GRIP_SCALE[1] - GRIP_SCALE[0]), 0.0), FINGER_MAX)
    loc = [(x, y, z) for x in np.linspace(0.0, 0.084, 5) for y in np.linspace(-0.08, 0.08, 9)
           for z in np.linspace(-0.03, 0.03, 5)]
    for x0, x1, yo, zh in ((0.075, 0.105, 0.0378, 0.028), (0.107, 0.126, 0.0134, 0.015), (0.128, 0.1576, 0.0034, 0.009)):
        for x in np.linspace(x0, x1, 4):
            for y in np.linspace(0.0004, yo, 3):
                for z in np.linspace(-zh, zh, 5):
                    loc += [(x, qf + y, z), (x, -(qf + y), z)]
    loc = np.asarray(loc)
    return (np.asarray(T_flange)[:3, :3] @ loc.T).T + np.asarray(T_flange)[:3, 3]


def mesh_irregular(V: np.ndarray) -> bool:
    """Does the shape fill its box poorly? Horizontal slices across the body z axis cover under 60 % of the
    bbox footprint on average (figurines, tools, keys)."""
    V = np.asarray(V, float)
    ext = np.maximum(V.max(0) - V.min(0), 1e-6)
    k = 2                                        # slices across the body z axis
    fills = []
    for z in np.linspace(V[:, k].min(), V[:, k].max(), 7)[1:-1]:
        m = np.abs(V[:, k] - z) < 0.06 * ext[k]
        if m.sum() >= 4:
            o2 = [i for i in range(3) if i != k]
            a = np.prod(V[m][:, o2].max(0) - V[m][:, o2].min(0))
            fills.append(a / (ext[o2[0]] * ext[o2[1]]))
    return bool(fills and float(np.mean(fills)) < 0.6)


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
