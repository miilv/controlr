"""Manipulation skills on top of the motion layer and the executor.

Each skill plans first (``plan_*`` -> an ``ArmPlan`` with an exact env-step cost), executes, then
re-reads the world (``provider.refresh``) and compares it with the skill's expected post-state.
The result says what happened in terms a program (or an LLM) can act on:

    SkillResult(ok, skill, status, steps, arm, expected, measured, note)

status: ``ok`` | ``ik_fail`` | ``collision`` | ``grasp_slip`` | ``place_miss`` | ``step_budget`` |
``logic`` | ``ended``. These are also the G3 failure categories.

Plans for the two arms can run in the same env steps (``parallel``), with an arm-arm distance
check on the merged schedule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dojo.skills import grasps as G
from dojo.skills.executor import ArmPlan, BudgetExceeded, EpisodeEnded, Executor
from dojo.skills.geometry import T_from, T_to_pose7, inv_T, rot_z, tilt_deg
from dojo.skills.kinematics import ARMS, TABLE_Z, X5Kin, gap_to_grip
from dojo.skills.motion import CARRY, FAST, MEDIUM, SLOW, Collision, HeldBox, Motion, MoveResult, min_arm_distance
from dojo.skills.world import AssetDB, WorldState, arms_from_proprio, objects_from_gt

OPEN = 1.0
CLOSED = 0.0
CLOSE_STEPS = 4
OPEN_STEPS = 2
FINGER_SPEED = 0.008          # m per env step per finger, measured in the twin (g3 probe: 0.15 of the 0..1 range)


def close_steps(open_gap: float, width: float) -> int:
    """Env steps for the pads to travel from ``open_gap`` to the object (+1 to build the squeeze)."""
    return max(2, math.ceil(max(0.0, open_gap - width) / 2 / FINGER_SPEED) + 1)


@dataclass
class SkillResult:
    ok: bool
    skill: str
    status: str = "ok"
    steps: int = 0
    arm: str | None = None
    expected: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    note: str = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class Held:
    label: str
    T_tcp_obj: np.ndarray
    box: HeldBox
    grasp: G.Grasp | None = None


@dataclass
class PickPlan:
    arm: str
    plan: ArmPlan
    grasp: G.Grasp
    T_tcp_obj: np.ndarray
    steps: int
    label: str


@dataclass
class PlacePlan:
    arm: str
    plan: ArmPlan
    T_obj_target: np.ndarray
    steps: int
    label: str


@dataclass
class Job:
    """One pick-and-place: ``target(sk) -> 4x4 object pose`` evaluated right before placing."""
    label: str
    arm: str
    target: object
    yaw_sym: int = 1
    ignore: object = ()
    method: str = "auto"
    pick_kw: dict = field(default_factory=dict)
    place_kw: dict = field(default_factory=dict)
    tol_xy: float = 0.01
    tol_tilt: float = 6.0
    bank: bool = True
    required: bool = True
    prefetch: bool = True
    park_other: bool = True
    any_arm: bool = False
    exclusive: bool = False       # no concurrent pick by the other arm while this one places


class GTProvider:
    """WorldState from the dev-only ground truth of the env (twin or KinematicEnv)."""

    def __init__(self, env, assets: AssetDB | None, task: str, instruction: str = "", mesh_labels: tuple = (),
                 include_clutter: bool = True):
        self.env = env
        self.assets = assets
        self.task = task
        self.instruction = instruction
        self.mesh_labels = tuple(mesh_labels)
        self.include_clutter = include_clutter
        self.home_q = None
        self.notes: dict = {}

    def refresh(self, holding: dict | None = None) -> WorldState:
        st = self.env.proprio()
        arms = arms_from_proprio(st, holding)
        if self.home_q is None:
            self.home_q = {a: arms[a].q.copy() for a in ARMS}
        rows = self.env.gt_objects()
        if self.assets is None:
            objs = _objects_from_rows_nodb(rows)
        else:
            objs = objects_from_gt(rows, self.assets, step=self.env.steps_used, include_clutter=self.include_clutter,
                                   mesh_labels=self.mesh_labels)
        return WorldState(task=self.task, instruction=self.instruction, objects=objs, arms=arms, home_q=self.home_q,
                          step=int(self.env.steps_used), step_lim=int(self.env.step_lim), notes=self.notes)


def _objects_from_rows_nodb(rows):
    """KinematicEnv rows carry 'half' extents: build AssetInfo from them (tests)."""
    from dojo.skills.world import AssetInfo, ObjectState
    out = {}
    for r in rows:
        half = np.asarray(r.get("half", [0.02, 0.02, 0.02]), float)
        info = AssetInfo(key=f"{r.get('type','Rigid')}/{r.get('category','x')}/{int(r.get('category_idx',0)):05d}",
                         bbox_min=-half, bbox_max=half, mesh=r.get("mesh"))
        out[r["label"]] = ObjectState(r["label"], r.get("type", "Rigid"), r.get("category", "x"),
                                      int(r.get("category_idx", 0)), np.asarray(r["pos"], float),
                                      np.asarray(r["quat"], float), info, clutter=bool(r.get("clutter", False)))
    return out


class Skills:
    def __init__(self, env, provider, kin: X5Kin | None = None, reserve: int = 30, log=None):
        self.env = env
        self.provider = provider
        self.kin = kin or X5Kin()
        self.ex = Executor(env, reserve=reserve)
        self.world = provider.refresh()
        self.coll = Collision(self.kin, self.world)
        self.motion = Motion(self.kin, self.coll)
        self.held: dict[str, Held | None] = {a: None for a in ARMS}
        self.history: list[SkillResult] = []
        self.log = log or (lambda *a, **k: None)

    # ------------------------------------------------------------------ world
    def refresh(self) -> WorldState:
        self.world = self.provider.refresh({a: (h.label if h else None) for a, h in self.held.items()})
        self.coll.world = self.world
        return self.world

    def tcp(self, arm: str, q=None) -> np.ndarray:
        return self.kin.tcp(arm, self.ex.q[arm] if q is None else q)

    def _record(self, r: SkillResult) -> SkillResult:
        self.history.append(r)
        self.log(f"[skill] {r.skill} {r.arm or ''} {r.status} steps={r.steps} {r.note}")
        return r

    # ------------------------------------------------------------------ pick
    def plan_pick(self, label: str, arm: str | None = None, method: str = "auto", lift: float = 0.08,
                  q_start: dict | None = None, max_eval: int = 40, keep: int = 4, grasp_kw: dict | None = None,
                  lift_to_z: float | None = None, approach_rad: float = FAST, descend_rad: float = SLOW,
                  open_g: float | None = None, settle: int = 2) -> PickPlan | SkillResult:
        w = self.world
        obj = w.obj(label)
        arms = [arm] if arm else sorted(ARMS, key=lambda a: float(np.linalg.norm(self.kin.base(a)[:2, 3] - obj.pos[:2])))
        best, notes = None, []
        for a in arms:
            q0 = (q_start or {}).get(a, self.ex.q[a])
            cands = G.candidates(obj, self.kin, a, method, **(grasp_kw or {}))
            if not cands:
                notes.append(f"{a}: no grasp candidates ({method})")
                continue
            feas = []
            n_ik = n_coll = 0
            for g in cands[:max_eval]:
                go = gap_to_grip(g.open_gap()) if open_g is None else open_g
                r = self.kin.ik(a, g.T, q0, n_random=1)
                if not r.ok:
                    n_ik += 1
                    continue
                ok, why = self.coll.check(a, r.q, go, ignore=(label,))
                if not ok:
                    n_coll += 1
                    continue
                to_pre = self.motion.to_pose(a, q0, g.pre(), approach_rad, "pick_approach", gripper=go)
                if not to_pre.ok:
                    n_coll += to_pre.status == "collision"
                    n_ik += to_pre.status == "ik_fail"
                    continue
                down = self.motion.line(a, to_pre.q_end, g.T, descend_rad, "pick_descend", gripper=go, ignore=(label,))
                if not down.ok:
                    n_coll += down.status == "collision"
                    n_ik += down.status != "collision"
                    continue
                T_tcp_obj = inv_T(g.T) @ obj.T
                box = HeldBox(T_tcp_obj, obj.asset.center, obj.asset.extents / 2)
                T_lift = g.T.copy()
                T_lift[2, 3] = max(g.T[2, 3] + lift, lift_to_z or 0.0)
                up = self.motion.line(a, down.q_end, T_lift, MEDIUM, "pick_lift", gripper=CLOSED, ignore=(label,),
                                      held=box)
                if not up.ok:
                    n_coll += up.status == "collision"
                    n_ik += up.status != "collision"
                    continue
                p = ArmPlan(a, np.asarray(q0, float), self.ex.g[a])
                p.grip(go, 0, "open_for_grasp").move(to_pre, "approach").move(down, "descend").hold(settle, "settle")
                from dojo.skills.kinematics import grip_to_gap
                p.grip(CLOSED, close_steps(grip_to_gap(go), g.width), "close").move(up, "lift")
                cost = p.steps + 15 * g.score
                feas.append((cost, PickPlan(a, p, g, T_tcp_obj, p.steps, label)))
                if len(feas) >= keep:
                    break
            if feas:
                feas.sort(key=lambda x: x[0])
                if best is None or feas[0][0] < best[0]:
                    best = feas[0]
            notes.append(f"{a}: {len(feas)} feasible of {min(len(cands), max_eval)} (ik {n_ik}, collision {n_coll})")
        if best is None:
            status = "ik_fail" if all("collision 0" in n or "no grasp" in n for n in notes) else "collision"
            return SkillResult(False, "pick", status, 0, arm, note="; ".join(notes))
        return best[1]

    def pick(self, label: str, arm: str | None = None, method: str = "auto", **kw) -> SkillResult:
        self.refresh()
        pp = self.plan_pick(label, arm, method, **kw)
        if isinstance(pp, SkillResult):
            return self._record(pp)
        return self._record(self.run_pick(pp))

    def run_pick(self, pp: PickPlan) -> SkillResult:
        before = self.world.obj(pp.label).pos.copy()
        try:
            self.ex.run({pp.arm: pp.plan}, name=f"pick {pp.label}")
        except BudgetExceeded as e:
            return SkillResult(False, "pick", "step_budget", 0, pp.arm, note=str(e))
        except EpisodeEnded as e:
            return SkillResult(False, "pick", "ended", pp.steps, pp.arm, note=str(e))
        return self.verify_pick(pp, before)

    def verify_pick(self, pp: PickPlan, before: np.ndarray) -> SkillResult:
        w = self.refresh()
        o = w.obj(pp.label)
        T_tcp = self.kin.tcp(pp.arm, w.arms[pp.arm].q)
        T_rel = inv_T(T_tcp) @ o.T
        dz = float(o.pos[2] - before[2])
        drift = float(np.linalg.norm(T_rel[:3, 3] - pp.T_tcp_obj[:3, 3]))
        exp = {pp.label: T_to_pose7(T_tcp @ pp.T_tcp_obj).tolist()}
        meas = {pp.label: T_to_pose7(o.T).tolist()}
        lift_planned = float((pp.plan.q_end is not None) and (self.kin.tcp(pp.arm, pp.plan.q_end)[2, 3] - pp.grasp.T[2, 3]))
        if dz < 0.5 * max(lift_planned, 0.02) or drift > 0.03:
            self.held[pp.arm] = None
            return SkillResult(False, "pick", "grasp_slip", pp.steps, pp.arm, exp, meas,
                               f"object rose {dz*1000:.0f} mm (planned {lift_planned*1000:.0f}), in-hand drift {drift*1000:.0f} mm")
        box = HeldBox(T_rel, o.asset.center, o.asset.extents / 2)
        self.held[pp.arm] = Held(pp.label, T_rel, box, pp.grasp)
        return SkillResult(True, "pick", "ok", pp.steps, pp.arm, exp, meas,
                           f"{pp.grasp.method} {pp.grasp.note}, in-hand drift {drift*1000:.1f} mm")

    # ------------------------------------------------------------------ place
    def plan_place(self, arm: str, T_obj_target, yaw_sym: int = 1, release_gap: float = 0.003, pre_h: float = 0.06,
                   retreat: float = 0.05, ignore=(), q_start=None, approach_rad: float = CARRY,
                   descend_rad: float = SLOW, open_g: float = OPEN, settle: int = 2, open_steps: int = OPEN_STEPS,
                   extra_yaws=(), pre_open_hold: int = 1, ease: int = 3) -> PlacePlan | SkillResult:
        """Approach above the target, descend in a straight line, settle, open, back out.
        ``yaw_sym``: n-fold symmetry of the object about the vertical (the cheapest equivalent yaw wins)."""
        h = self.held[arm]
        if h is None:
            return SkillResult(False, "place", "logic", 0, arm, note="nothing held")
        q0 = self.ex.q[arm] if q_start is None else q_start
        T_obj_target = np.asarray(T_obj_target, float)
        o = self.world.obj(h.label)
        cands = []
        yaws = [2 * math.pi * k / max(1, yaw_sym) for k in range(max(1, yaw_sym))] + list(extra_yaws)
        cw = T_obj_target[:3, :3] @ o.asset.center + T_obj_target[:3, 3]       # rotate about the bbox centre
        for dyaw in yaws:
            T = T_obj_target.copy()
            T[:3, :3] = rot_z(dyaw) @ T_obj_target[:3, :3]
            T[:3, 3] = cw - T[:3, :3] @ o.asset.center
            cands.append((abs(dyaw), T))
        tail_kw = dict(release_gap=release_gap, retreat=retreat, ignore=tuple(set(ignore) | {h.label}),
                       descend_rad=descend_rad, open_g=open_g, settle=settle, open_steps=open_steps,
                       pre_open_hold=pre_open_hold)
        best, notes = None, []
        for _, T_obj in sorted(cands, key=lambda x: x[0]):
            T_tcp = T_obj @ inv_T(h.T_tcp_obj)
            T_pre = T_tcp.copy()
            T_pre[2, 3] += release_gap + pre_h
            to_pre = self.motion.to_pose(arm, q0, T_pre, approach_rad, "place_approach", gripper=CLOSED, held=h.box,
                                         ignore=(h.label,), ease=ease)
            if not to_pre.ok:
                notes.append(f"approach: {to_pre.note}")
                continue
            tail = self._place_tail(arm, to_pre.q_end, T_obj, h, **tail_kw)
            if isinstance(tail, str):
                notes.append(tail)
                continue
            p = ArmPlan(arm, np.asarray(q0, float), self.ex.g[arm]).move(to_pre, "approach")
            p.ops.extend(tail.ops)
            if best is None or p.steps < best.steps:
                best = PlacePlan(arm, p, T_obj, p.steps, h.label)
                best.approach = to_pre
                best.tail_kw = tail_kw
        if best is None:
            st = "ik_fail" if any("unreachable" in n or "ik_fail" in n for n in notes) else "collision"
            return SkillResult(False, "place", st, 0, arm, note="; ".join(notes[:4]))
        return best

    def _place_tail(self, arm, q_pre, T_obj, h: "Held", release_gap, retreat, ignore, descend_rad, open_g, settle,
                    open_steps, pre_open_hold) -> ArmPlan | str:
        T_rel = T_obj @ inv_T(h.T_tcp_obj)
        T_rel[2, 3] += release_gap
        down = self.motion.line(arm, q_pre, T_rel, descend_rad, "place_descend", gripper=CLOSED, held=h.box,
                                ignore=ignore)
        if not down.ok:
            return f"descend: {down.note}"
        T_up = T_rel.copy()
        T_up[:3, 3] -= T_rel[:3, 0] * retreat          # back out along the approach axis
        if T_rel[2, 0] > -0.5:
            T_up[2, 3] += 0.02
        up = self.motion.line(arm, down.q_end, T_up, MEDIUM, "place_retreat", gripper=open_g, ignore=ignore)
        if not up.ok:
            return f"retreat: {up.note}"
        t = ArmPlan(arm, np.asarray(q_pre, float), CLOSED)
        t.move(down, "descend").hold(settle, "settle")
        t.grip(open_g, open_steps, "release").hold(pre_open_hold, "let_go").move(up, "retreat")
        return t

    def place(self, arm: str, T_obj_target, tol_xy: float = 0.01, tol_tilt: float = 6.0, tol_yaw: float | None = None,
              **kw) -> SkillResult:
        pp = self.plan_place(arm, T_obj_target, **kw)
        if isinstance(pp, SkillResult):
            return self._record(pp)
        return self._record(self.run_place(pp, tol_xy, tol_tilt, tol_yaw))

    def run_place(self, pp: PlacePlan, tol_xy: float = 0.01, tol_tilt: float = 6.0, tol_yaw: float | None = None,
                  other: list | None = None, other_arm: str | None = None, remeasure: bool = True) -> SkillResult:
        """Execute a place: approach (+1 still step), re-measure the object in the hand, re-plan the
        descent for the measured in-hand pose, descend, release, back out. ``other`` = the other
        arm's compiled per-step (q, g) list, run in the same env steps throughout."""
        other = list(other or [])
        try:
            if remeasure and getattr(pp, "approach", None) is not None:
                a1 = ArmPlan(pp.arm, pp.plan.q0, pp.plan.g0).move(pp.approach, "approach").hold(1, "still")
                c1 = a1.compile()[0]
                lists = {pp.arm: c1}
                if other_arm:
                    lists[other_arm] = other[:len(c1)]
                self.ex.run_lists(lists, name=f"place {pp.label} approach")
                rest_other = other[len(c1):]
                w = self.refresh()
                h = self.held[pp.arm]
                o = w.obj(pp.label)
                T_rel_meas = inv_T(self.kin.tcp(pp.arm, w.arms[pp.arm].q)) @ o.T
                shift = float(np.linalg.norm(T_rel_meas[:3, 3] - h.T_tcp_obj[:3, 3]))
                h2 = Held(h.label, T_rel_meas, HeldBox(T_rel_meas, o.asset.center, o.asset.extents / 2), h.grasp)
                tail = self._place_tail(pp.arm, self.ex.q[pp.arm], pp.T_obj_target, h2, **pp.tail_kw)
                if isinstance(tail, str):
                    tail = self._place_tail(pp.arm, self.ex.q[pp.arm], pp.T_obj_target, h, **pp.tail_kw)
                    note_rm = f"in-hand shift {shift*1000:.1f} mm (re-plan failed: {tail if isinstance(tail, str) else ''})"
                else:
                    self.held[pp.arm] = h2
                    note_rm = f"in-hand shift {shift*1000:.1f} mm, re-planned"
                if isinstance(tail, str):
                    return SkillResult(False, "place", "collision", self.ex.steps_used, pp.arm, note=tail)
                c2 = tail.compile()[0]
                lists = {pp.arm: c2}
                if other_arm:
                    lists[other_arm] = rest_other
                self.ex.run_lists(lists, name=f"place {pp.label} tail")
            else:
                note_rm = ""
                lists = {pp.arm: pp.plan.compile()[0]}
                if other_arm:
                    lists[other_arm] = other
                self.ex.run_lists(lists, name=f"place {pp.label}")
        except BudgetExceeded as e:
            return SkillResult(False, "place", "step_budget", 0, pp.arm, note=str(e))
        except EpisodeEnded as e:
            self.held[pp.arm] = None
            return SkillResult(False, "place", "ended", pp.steps, pp.arm, note=str(e))
        self.held[pp.arm] = None
        r = self.verify_place(pp, tol_xy, tol_tilt, tol_yaw)
        r.note = (r.note + "; " + note_rm).strip("; ")
        return r

    def verify_place(self, pp: PlacePlan, tol_xy=0.01, tol_tilt=6.0, tol_yaw=None) -> SkillResult:
        w = self.refresh()
        o = w.obj(pp.label)
        exp_c = pp.T_obj_target[:3, :3] @ o.asset.center + pp.T_obj_target[:3, 3]
        dxy = float(np.linalg.norm(o.center_w[:2] - exp_c[:2]))
        dz = float(o.center_w[2] - exp_c[2])
        tilt = math.degrees(math.acos(min(1.0, abs(float((o.R.T @ pp.T_obj_target[:3, :3])[2, 2])))))
        exp = {pp.label: T_to_pose7(pp.T_obj_target).tolist()}
        meas = {pp.label: T_to_pose7(o.T).tolist()}
        note = (f"xy err {dxy*1000:.1f} mm, dz {dz*1000:+.1f} mm (centre {o.center_w[2]:.4f} vs {exp_c[2]:.4f}), "
                f"rel tilt {tilt:.1f} deg")
        ok = dxy <= tol_xy and abs(dz) < 0.02 and tilt <= tol_tilt
        return SkillResult(ok, "place", "ok" if ok else "place_miss", pp.steps, pp.arm, exp, meas, note)

    # ------------------------------------------------------------------ reach
    def reach_down(self, arm: str, xyz, yaws=(0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4), tilts=(0.0, 30.0)) -> bool:
        """Can ``arm`` put its TCP at ``xyz`` pointing down (or tilted up to 30 deg towards the base)?"""
        from dojo.skills.kinematics import down_R, tilted_R
        base = self.kin.base(arm)[:3, 3]
        heading = math.atan2(xyz[1] - base[1], xyz[0] - base[0])
        for t in tilts:
            for y in yaws:
                R = down_R(y) if t == 0 else tilted_R(y, 90 - t, heading)
                if self.kin.ik(arm, T_from(xyz, R), self.world.home_q[arm], n_random=0).ok:
                    return True
        return False

    def can_pick(self, label: str, arm: str, method: str = "auto") -> bool:
        """Grasp feasibility from the arm's home pose (cached per world refresh)."""
        key = (label, arm, method, self.world.step)
        cache = self.__dict__.setdefault("_can_pick", {})
        if key not in cache:
            pp = self.plan_pick(label, arm, method, q_start={arm: self.world.home_q[arm]}, keep=1)
            cache[key] = isinstance(pp, PickPlan)
        return cache[key]

    # ------------------------------------------------------------------ job pipeline
    def run_jobs(self, jobs: list["Job"], bank_after: bool = True) -> SkillResult:
        """Pick-and-place jobs in order, two arms pipelined: while one arm places job k, the other
        picks job k+1 (when it is the other arm's). A job's target is computed right before its place
        is planned (from the refreshed world), so it can depend on earlier placements."""
        n0 = self.ex.steps_used

        def fail(status, arm, note):
            return self._record(SkillResult(False, "jobs", status, self.ex.steps_used - n0, arm, note=note))

        def holds(job):
            h = self.held.get(job.arm)
            return h is not None and h.label == job.label

        def plan_pick_job(job):
            pp = self.plan_pick(job.label, job.arm, job.method, **job.pick_kw)
            if not isinstance(pp, PickPlan) and job.any_arm:
                pp2 = self.plan_pick(job.label, None, job.method, **job.pick_kw)
                if isinstance(pp2, PickPlan):
                    job.arm = pp2.arm
                    return pp2
            return pp

        for i, job in enumerate(jobs):
            self.refresh()
            if not holds(job):
                pp = plan_pick_job(job)
                if not isinstance(pp, PickPlan):
                    return fail(pp.status, job.arm, f"pick {job.label}: {pp.note}")
                before = self.world.obj(job.label).pos.copy()
                try:
                    self.parallel([pp.plan], name=f"pick {job.label}")
                except (BudgetExceeded, EpisodeEnded) as e:
                    return fail("step_budget" if isinstance(e, BudgetExceeded) else "ended", job.arm, str(e))
                r = self._record(self.verify_pick(pp, before))
                if not r.ok:
                    try:
                        self.ex.run({pp.arm: self.plan_home(pp.arm, gripper=OPEN)}, name="reset arm")
                    except (BudgetExceeded, EpisodeEnded) as e:
                        return fail("step_budget", job.arm, str(e))
                    r = self.pick(job.label, job.arm if not job.any_arm else None)
                    if not r.ok:
                        return fail(r.status, job.arm, f"pick {job.label}: {r.note}")
                    job.arm = r.arm
            # place job i; meanwhile the other arm picks job i+1. An idle other arm near the target
            # parks first (sequenced, not concurrent).
            self.refresh()
            other = "right" if job.arm == "left" else "left"
            T = job.target(self)
            if self.held[other] is None and job.park_other:
                T_o = self.kin.tcp(other, self.ex.q[other])
                if float(np.linalg.norm(T_o[:2, 3] - T[:2, 3])) < 0.30 or T_o[2, 3] < 0.86:
                    try:
                        self.ex.run({other: self.plan_home(other)}, name=f"park {other}")
                    except (BudgetExceeded, EpisodeEnded) as e:
                        return fail("step_budget" if isinstance(e, BudgetExceeded) else "ended", other, str(e))
            ign = job.ignore(self) if callable(job.ignore) else tuple(job.ignore)
            pl = self.plan_place(job.arm, T, yaw_sym=job.yaw_sym, ignore=ign, **job.place_kw)
            if not isinstance(pl, PlacePlan):
                return fail(pl.status, job.arm, f"place {job.label}: {pl.note}")
            nxt = jobs[i + 1] if i + 1 < len(jobs) else None
            side_pick, other_list = None, None
            if (nxt is not None and nxt.arm == other and self.held[other] is None and nxt.prefetch
                    and not job.exclusive):
                pn = plan_pick_job(nxt)
                if isinstance(pn, PickPlan) and pn.arm == other and self.clear_of(pl.plan, pn.plan):
                    side_pick = (pn, self.world.obj(nxt.label).pos.copy())
                    other_list = pn.plan.compile()[0]
            r = self._record(self.run_place(pl, job.tol_xy, job.tol_tilt, other=other_list,
                                            other_arm=other if other_list else None))
            if r.status in ("step_budget", "ended"):
                return fail(r.status, job.arm, r.note)
            if side_pick is not None:
                self._record(self.verify_pick(*side_pick))          # failure: job i+1 re-picks
            if not r.ok and job.required:
                return fail(r.status, job.arm, f"place {job.label}: {r.note}")
            if bank_after and job.bank:
                self.bank()
        return self._record(SkillResult(True, "jobs", "ok", self.ex.steps_used - n0))

    # ------------------------------------------------------------------ parallel
    def clear_of(self, a: ArmPlan, b: ArmPlan, min_dist: float = 0.12) -> bool:
        """Do two arm plans keep the arms >= ``min_dist`` apart (sampled chain, palm, finger points)
        at every step of their merged schedule (both planned paths, not just current poses)?"""
        ca, cb = a.compile()[0], b.compile()[0]
        n = max(len(ca), len(cb))
        for i in range(n):
            qa = ca[min(i, len(ca) - 1)][0] if ca else a.q0
            qb = cb[min(i, len(cb) - 1)][0] if cb else b.q0
            ql, qr = (qa, qb) if a.arm == "left" else (qb, qa)
            if min_arm_distance(self.kin, ql, qr) < min_dist:
                return False
        return True

    def parallel(self, plans: list[ArmPlan], min_dist: float = 0.12, name: str = "parallel") -> dict:
        """Run two arm plans in the same env steps if their merged schedules keep the arms apart;
        otherwise one after the other."""
        if len(plans) == 2:
            a, b = plans
            ca, cb = a.compile()[0], b.compile()[0]
            n = max(len(ca), len(cb))
            dmin = 9.0
            for i in range(0, n, 2):
                qa = ca[min(i, len(ca) - 1)][0] if ca else a.q0
                qb = cb[min(i, len(cb) - 1)][0] if cb else b.q0
                ql, qr = (qa, qb) if a.arm == "left" else (qb, qa)
                dmin = min(dmin, min_arm_distance(self.kin, ql, qr))
                if dmin < min_dist:
                    break
            if dmin < min_dist:
                r1 = self.ex.run({a.arm: a}, name=name + "_seq1")
                b2 = ArmPlan(b.arm, b.q0, b.g0, b.ops)
                r2 = self.ex.run({b.arm: b2}, name=name + "_seq2")
                return {"steps": r1["steps"] + r2["steps"], "parallel": False, "min_dist": dmin}
            r = self.ex.run({a.arm: a, b.arm: b}, name=name)
            return {"steps": r["steps"], "parallel": True, "min_dist": dmin}
        r = self.ex.run({p.arm: p for p in plans}, name=name)
        return {"steps": r["steps"], "parallel": True}

    # ------------------------------------------------------------------ home / bank / finish
    def plan_home(self, arm: str, q_start=None, lift_if_below: float = 0.86, step_rad: float = FAST,
                  gripper: float | None = None) -> ArmPlan:
        q0 = self.ex.q[arm] if q_start is None else np.asarray(q_start, float)
        g = self.ex.g[arm] if gripper is None else gripper
        p = ArmPlan(arm, q0, self.ex.g[arm])
        if gripper is not None:
            p.grip(gripper, 0, "grip_for_home")
        T = self.kin.tcp(arm, q0)
        q = q0
        if T[2, 3] < lift_if_below:
            Tu = T.copy()
            Tu[2, 3] = lift_if_below + 0.02
            up = self.motion.line(arm, q0, Tu, MEDIUM, "home_lift", gripper=g, check=False)
            if up.segments:
                p.move(up, "home_lift")
                q = up.q_end
        home = self.world.home_q[arm]
        mv = self.motion.joint(arm, q, home, step_rad, "home", gripper=g, check=False)
        p.move(mv, "home")
        return p

    def bank(self, steps: int = OPEN_STEPS) -> SkillResult:
        """Open both grippers fully (the scorer banks a tier only at a step with both >= 0.8 open).
        Skipped while an arm holds something (opening would drop it)."""
        busy = [a for a in ARMS if self.held[a] is not None]
        if busy:
            return self._record(SkillResult(True, "bank", "skipped", 0, note=f"{busy} holding"))
        need = {a: OPEN for a in ARMS if self.ex.g[a] < OPEN - 1e-6}
        if not need:                      # both commanded open: the fingers finish opening during the next moves
            return self._record(SkillResult(True, "bank", "ok", 0))
        try:
            n = self.ex.grip({a: OPEN for a in ARMS}, steps, "bank", use_reserve=True)
        except (BudgetExceeded, EpisodeEnded) as e:
            return self._record(SkillResult(False, "bank", "ended" if isinstance(e, EpisodeEnded) else "step_budget",
                                            0, note=str(e)))
        return self._record(SkillResult(True, "bank", "ok", n))

    def finish(self, max_wait: int = 12) -> SkillResult:
        """Both grippers open, both arms home (straight joint moves, lifted first when low), then
        hold until the arms are at home (the checker wants both flanges within 0.15 m / 20 deg)."""
        n0 = self.ex.steps_used
        try:
            plans = [self.plan_home(a, gripper=OPEN) for a in ARMS]
            self.ex.run({p.arm: p for p in plans}, name="finish", use_reserve=True)
            for a in ARMS:
                self.held[a] = None
            self.ex.settle(tol=0.02, max_steps=max_wait, use_reserve=True)
        except EpisodeEnded as e:
            return self._record(SkillResult(True, "finish", "ended", self.ex.steps_used - n0, note=str(e)))
        except BudgetExceeded as e:
            return self._record(SkillResult(False, "finish", "step_budget", self.ex.steps_used - n0, note=str(e)))
        return self._record(SkillResult(True, "finish", "ok", self.ex.steps_used - n0))

    def home(self, arms=ARMS, gripper: float | None = None) -> SkillResult:
        n0 = self.ex.steps_used
        try:
            self.ex.run({a: self.plan_home(a, gripper=gripper) for a in arms}, name="home")
        except (BudgetExceeded, EpisodeEnded) as e:
            return self._record(SkillResult(False, "home", "ended" if isinstance(e, EpisodeEnded) else "step_budget",
                                            self.ex.steps_used - n0, note=str(e)))
        return self._record(SkillResult(True, "home", "ok", self.ex.steps_used - n0))

    def park(self, arm: str, T_tcp=None) -> ArmPlan:
        """Move an arm out of the way (to home by default) as a plan."""
        if T_tcp is None:
            return self.plan_home(arm)
        mv = self.motion.to_pose(arm, self.ex.q[arm], T_tcp, FAST, "park", gripper=self.ex.g[arm])
        return ArmPlan(arm, self.ex.q[arm].copy(), self.ex.g[arm]).move(mv, "park")

    # ------------------------------------------------------------------ press
    def plan_press(self, arm: str, xyz_top, n: int = 1, depth: float = 0.012, clearance: float = 0.025,
                   yaw: float = 0.0, press_rad: float = MEDIUM, hold_down: int = 1, hold_up: int = 1,
                   q_start=None, tip_first: bool = True) -> ArmPlan | SkillResult:
        """``n`` press-and-release cycles on a button whose top is at ``xyz_top``: closed jaws,
        fingertips straight down, ``depth`` below the top, back to ``clearance`` above."""
        from dojo.skills.kinematics import TIP_BEYOND_TCP, down_R
        q0 = self.ex.q[arm] if q_start is None else q_start
        p_top = np.asarray(xyz_top, float)
        T_up = T_from(p_top + [0, 0, clearance + TIP_BEYOND_TCP], down_R(yaw))
        T_dn = T_from(p_top + [0, 0, -depth + TIP_BEYOND_TCP], down_R(yaw))
        for yaw_try in (yaw, yaw + math.pi / 2):
            T_up = T_from(p_top + [0, 0, clearance + TIP_BEYOND_TCP], down_R(yaw_try))
            T_dn = T_from(p_top + [0, 0, -depth + TIP_BEYOND_TCP], down_R(yaw_try))
            to_up = self.motion.to_pose(arm, q0, T_up, FAST, "press_approach", gripper=CLOSED, check=False)
            if not to_up.ok:
                continue
            down = self.motion.line(arm, to_up.q_end, T_dn, press_rad, "press_down", gripper=CLOSED, check=False)
            back = self.motion.line(arm, down.q_end if down.ok else to_up.q_end, T_up, press_rad, "press_up",
                                    gripper=CLOSED, check=False)
            if not (down.ok and back.ok):
                continue
            p = ArmPlan(arm, np.asarray(q0, float), self.ex.g[arm]).grip(CLOSED, 0, "close_for_press")
            p.move(to_up, "press_approach")
            for i in range(n):
                p.move(down, f"press_{i}_down").hold(hold_down, f"press_{i}_hold").move(back, f"press_{i}_up").hold(
                    hold_up, f"press_{i}_release")
            return p
        return SkillResult(False, "press", "ik_fail", 0, arm, note="button pose unreachable")

    def press(self, arm: str, xyz_top, n: int = 1, **kw) -> SkillResult:
        p = self.plan_press(arm, xyz_top, n, **kw)
        if isinstance(p, SkillResult):
            return self._record(p)
        try:
            r = self.ex.run({arm: p}, name=f"press x{n}")
        except BudgetExceeded as e:
            return self._record(SkillResult(False, "press", "step_budget", 0, arm, note=str(e)))
        except EpisodeEnded as e:
            return self._record(SkillResult(False, "press", "ended", p.steps, arm, note=str(e)))
        return self._record(SkillResult(True, "press", "ok", r["steps"], arm, note=f"{n} presses"))

    # ------------------------------------------------------------------ push (closed loop, short segments)
    def push(self, label: str, target_xy, arm: str | None = None, tol: float = 0.008, seg: float = 0.02,
             max_iter: int = 12, height: float = 0.012, standoff: float = 0.02) -> SkillResult:
        """Push ``label``'s centre towards ``target_xy`` in segments of <= ``seg`` metres, re-reading
        the world after each one (pushing is chaotic open-loop: 1 mm -> 10-19 mm)."""
        from dojo.skills.kinematics import TIP_BEYOND_TCP, down_R
        n0 = self.ex.steps_used
        target_xy = np.asarray(target_xy, float)
        for it in range(max_iter):
            w = self.refresh()
            o = w.obj(label)
            d = target_xy - o.center_w[:2]
            dist = float(np.linalg.norm(d))
            if dist <= tol:
                return self._record(SkillResult(True, "push", "ok", self.ex.steps_used - n0, arm,
                                                note=f"within {dist*1000:.1f} mm after {it} segments"))
            u = d / dist
            a = arm or min(ARMS, key=lambda x: float(np.linalg.norm(self.kin.base(x)[:2, 3] - o.pos[:2])))
            ext = o.asset.extents
            r_obj = float(np.abs(o.R[:2, :2] @ (ext[:2] / 2)).max())
            start = o.center_w[:2] - u * (r_obj + standoff)
            z = o.bottom_z + height + TIP_BEYOND_TCP
            yaw = math.atan2(u[1], u[0]) + math.pi / 2       # jaws across the push direction
            T_s = T_from([start[0], start[1], z], down_R(yaw))
            T_above = T_s.copy()
            T_above[2, 3] = max(z, o.top_z + 0.03 + TIP_BEYOND_TCP)
            step = min(seg, dist + 0.002)
            T_e = T_s.copy()
            T_e[:2, 3] = start + u * (standoff + step)
            q0 = self.ex.q[a]
            m1 = self.motion.to_pose(a, q0, T_above, FAST, "push_above", gripper=CLOSED, ignore=(label,))
            if not m1.ok:
                return self._record(SkillResult(False, "push", m1.status, self.ex.steps_used - n0, a, note=m1.note))
            m2 = self.motion.line(a, m1.q_end, T_s, MEDIUM, "push_down", gripper=CLOSED, ignore=(label,))
            m3 = self.motion.line(a, m2.q_end if m2.ok else m1.q_end, T_e, SLOW, "push", gripper=CLOSED,
                                  ignore=(label,))
            m4 = self.motion.line(a, m3.q_end if m3.ok else m1.q_end, T_above, MEDIUM, "push_up", gripper=CLOSED,
                                  check=False)
            if not (m2.ok and m3.ok):
                return self._record(SkillResult(False, "push", "collision", self.ex.steps_used - n0, a,
                                                note=(m2.note or m3.note)))
            p = ArmPlan(a, q0, self.ex.g[a]).grip(CLOSED, 0).move(m1).move(m2).move(m3).move(m4)
            try:
                self.ex.run({a: p}, name=f"push {label} {it}")
            except (BudgetExceeded, EpisodeEnded) as e:
                return self._record(SkillResult(False, "push", "step_budget", self.ex.steps_used - n0, a, note=str(e)))
        w = self.refresh()
        dist = float(np.linalg.norm(target_xy - w.obj(label).center_w[:2]))
        return self._record(SkillResult(dist <= tol, "push", "ok" if dist <= tol else "place_miss",
                                        self.ex.steps_used - n0, arm, note=f"{dist*1000:.1f} mm left"))

    # ------------------------------------------------------------------ handover (via the table)
    def handover(self, label: str, from_arm: str, to_arm: str, xy=None, yaw_sym: int = 1) -> SkillResult:
        """Hand an object over by putting it down where the other arm reaches it (the checker never
        verifies a handover; only the end state counts). The receiving arm then picks it."""
        h = self.held[from_arm]
        if h is None:
            return self._record(SkillResult(False, "handover", "logic", 0, from_arm, note="nothing held"))
        o = self.world.obj(label)
        if xy is None:
            xy = (0.0, -0.15)
        T = o.T.copy()
        T[0, 3], T[1, 3] = xy
        T[2, 3] = TABLE_Z + (o.pos[2] - o.bottom_z) + 0.002
        r = self.place(from_arm, T, yaw_sym=yaw_sym, tol_xy=0.05, tol_tilt=30)
        if not r.ok and r.status != "place_miss":
            return self._record(SkillResult(False, "handover", r.status, r.steps, from_arm, note=r.note))
        r2 = self.pick(label, to_arm)
        return self._record(SkillResult(r2.ok, "handover", r2.status, r.steps + r2.steps, to_arm, note=r2.note))


def upright_tilt(o) -> float:
    return tilt_deg(o.R)
