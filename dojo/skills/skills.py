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
from dojo.skills.motion import (CARRY, FAST, MEDIUM, SLOW, Collision, HeldBox, Motion, MoveResult, held_box,
                                min_arm_distance)
from dojo.skills.world import AssetDB, WorldState, arms_from_proprio, objects_from_gt

OPEN = 1.0
CLOSED = 0.0
CLOSE_STEPS = 4
OPEN_STEPS = 2
FINGER_SPEED = 0.008          # m per env step per finger, measured in the twin (g3 probe: 0.15 of the 0..1 range)
TURN_FRACTIONS = (1.0, 0.85, 0.7, 0.6)   # of a place's turn, tried in order (the wrist's reach ends first)
PLACE_REACH = 0.60            # m, TCP from the arm's base in the plane: farther places are left to the other arm /
                              # a relay (twin: places 0.66-0.67 m out, the arm across the table, stalled 30 mm above
                              # the target with the commanded pose 33 mm lower, plug 7/10; a figurine's stand is 0.57)
TURN_RAD = 0.05               # rad/step of a place's turn (a key's -60 deg at 0.02 cost 52 of insert_key's 300 steps)
SCREEN_IK_BUDGET = 6000       # place IKs per arm in a transfer screen (a figurine x 81 stand poses: 9k IKs, 3 min)
UNEVEN_CARRY = 0.09           # rad/step, carries of uneven objects (tools, figurines, pens) instead of CARRY 0.14
LOOSE_OK_TOL = 0.02           # place tolerance (m) from which a loose grasp is carried on rather than redone
GUARD_LAG = 0.004             # m: measured-vs-commanded TCP gap that stops a guarded descent (contact)
GUARD_SEATED = 0.004          # m: a guarded descent that stops this close to its target counts as seated
PROBE_RADII = (0.0015, 0.003, 0.0045)   # m: probe circles around a contact beside the hole (nearest first)
PROBE_LIFT = 0.002            # m: lift above the contact height before each probe
PROBE_IN = 0.005              # m below the contact height: the probe went into the hole
LOOSE_DRIFT = 0.008           # in-hand drift after the lift beyond which a grasp is redone (good grasps: < 1 mm)
# mode B, the held-pose protocol at the pre-insert hover (``Skills._held_protocol``; only with a provider ``held_pose``)
HELD_LOOK_TRIES = 2           # looks per place: the provider's best observer look, then its best show pose
HELD_LOOK_STILL = 2           # still steps at a look before the read
HELD_LOOK_CLEAR = 0.04        # m between the arms (chain, palm, fingers) on the way to a look and back
HELD_LOOK_HAND_CLEAR = 0.015  # m between the observer's hand and the held object's box
HELD_LOOK_MIN_LEFT = 25       # env steps a look must leave beyond the insertion tail's own (probes)
SPIRAL_PITCH = 0.00015        # m per turn of the pressed spiral (``_servo_tail``, tail kw ``spiral``; mode B only)
SPIRAL_STEP = 0.0003          # m of spiral path per env step
SPIRAL_PRESS = 0.0015         # m the object is commanded below its contact height while it is dragged
SPIRAL_IN = 0.001             # m under the contact height: it went in


def spiral_points(r_max: float, pitch: float = SPIRAL_PITCH, step: float = SPIRAL_STEP) -> list[tuple[float, float]]:
    """(dx, dy) along an Archimedean spiral from the centre out to ``r_max`` (``pitch`` per turn, ``step`` of path apart),
    then once around at ``r_max``."""
    out, th, b = [(0.0, 0.0)], 0.0, pitch / (2 * math.pi)
    while b * th < r_max:
        th += step / max(b * th, b * 2.0)
        r = min(b * th, r_max)
        out.append((r * math.cos(th), r * math.sin(th)))
    th0 = th
    while th < th0 + 2 * math.pi:
        th += step / r_max
        out.append((r_max * math.cos(th), r_max * math.sin(th)))
    return out


def wrist_safe(q) -> bool:
    """Away from the wrist's limits (twin: a place with j4 1.7 and j6 at its 3.14 stop stalled 10 mm above the
    outlet while the command went 30 mm lower; the same place with a 25-deg tilted grasp went in, 4/4)."""
    q = np.asarray(q, float)
    return bool(abs(float(q[3])) <= 1.6 and abs(float(q[5])) <= 3.0)


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
    method: str = "auto"
    plan_kw: dict = field(default_factory=dict)     # the plan_pick arguments it was planned with (re-plans reuse them)
    # set by plan_transfer: re-pair(q_start) -> (PickPlan, PlacePlan) | SkillResult, the pick and place re-planned
    # together for the object as it is now; and the place paired with a re-paired pick (execute_pick fills it)
    repair: object = None
    paired_place: object = None


@dataclass
class PlacePlan:
    arm: str
    plan: ArmPlan
    T_obj_target: np.ndarray
    steps: int
    label: str
    keep_tilt: bool = False


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
    tol_z: float = 0.02
    alt_targets: list = field(default_factory=list)   # tried in order when the target cannot be planned


class GTProvider:
    """WorldState from the dev-only ground truth of the env (twin or KinematicEnv)."""

    def __init__(self, env, assets: AssetDB | None, task: str = "", instruction: str = "", meshes: bool = True,
                 include_clutter: bool = True):
        self.env = env
        self.assets = assets
        self.task = task
        self.instruction = instruction
        self.meshes = meshes
        self.include_clutter = include_clutter
        self.home_q = None
        self.notes: dict = {}
        self._actors_prev = None

    def actors_moving(self, tol: float = 2e-3) -> bool | None:
        """Whether a scripted actor (an opponent's or a demonstrator's arm) moved since the last call: dev-only GT of its
        joints in G3 (``env.actors_gt``); None when the env cannot tell (the caller then waits a fixed time). At eval
        the runtime's provider answers it from the frames (a watch read)."""
        f = getattr(self.env, "actors_gt", None)
        if f is None:
            return None
        cur = {k: np.asarray(v, float) for k, v in (f() or {}).items()}
        prev, self._actors_prev = self._actors_prev, cur
        if not cur:
            return False
        if prev is None:
            return True
        return any(k not in prev or float(np.abs(v - prev[k]).max()) > tol for k, v in cur.items())

    def watch_read(self, region: dict, prior: dict | None = None) -> dict:
        """Dev-only GT answer to perception's watch read (same dict shape, so G3 -> eval is a provider swap):
        region {"kind": "cells", "board": {"asset", "pos", "quat"}, "pieces": {"own": key, "opponent": key}} ->
        "cells": [{"k", "occupant", "conf", "pos"}]; {"kind": "container", "container": {...}, "candidates": [keys]} ->
        "inside" / "outside": [{"asset", "pos", "conf"}]. Plus t_ms, visible_frac, occluded, stable_vs_prior, new."""
        from dojo.skills.watch import gt_watch_read
        rows = self.env.gt_objects()
        objs = (_objects_from_rows_nodb(rows) if self.assets is None else
                objects_from_gt(rows, self.assets, step=self.env.steps_used, include_clutter=False, meshes=False))
        return gt_watch_read(objs, region, prior, step=int(self.env.steps_used))

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
                                   meshes=self.meshes)
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
    def __init__(self, env, provider, kin: X5Kin | None = None, reserve: int = 30, log=None, verifier=None,
                 verify_k: int = 2, n_cands: int = 6, remeasure: bool = True, on_verified=None, refine=None,
                 max_rehovers: int = 1):
        """``verifier``: a ``verify.TwinBackend`` (None = single plan). ``remeasure``: re-read the held
        object's pose before a place descent (off when the world provider cannot measure held
        objects). ``on_verified(info)``: called after each verified choice with
        {"idx", "passes", "outcomes", "actions", "kind", "label"} (also kept in ``last_verified``).
        ``refine(sk, label, arm, kind) -> {label: (pos, quat)} | None``: pre-contact wrist refine, called
        at the hover (kind "pick": label = the object to grasp; kind "place": label = the support /
        reference object or None) before the descent is planned; it only observes (no env steps); the
        skill then calls ``refresh()`` and re-plans the descent. None (GT mode) = no refine."""
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
        self.verifier = verifier
        self.verify_k = verify_k
        self.n_cands = n_cands
        self.remeasure = remeasure
        self.on_verified = on_verified
        self.refine = refine
        self.max_rehovers = int(max_rehovers)       # re-hover hints followed per pick / place (refine hook)
        self.widen_on_none = True                   # twin rejects every candidate: widen (more candidates), re-look
        self.refine_ctx: dict = {}
        # labels the wrist refine must not be asked about (perception: tiles in a dense row fit their neighbours;
        # their layout slots are better than a refine there); a place carrying one of them is skipped too
        self.no_refine: set = set()
        self._hq_cache: dict = {}
        # mode B with a provider ``held_pose``: look at the held object (``provider.show_pose``: an observer look, a
        # show pose) when the read at the pre-insert hover is not good enough (``_held_protocol``); every read logged
        self.held_looks = True
        self.held_events: list = []
        self.last_verified: dict | None = None
        self.vstats = {"segments": 0, "first_ok": 0, "later_ok": 0, "none_ok": 0, "candidates": 0}

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
                  open_g: float | None = None, settle: int = 2, return_all: bool = False, lift_rad: float = MEDIUM,
                  grasps: list | None = None, squeeze: int = 0):
        """``grasps``: explicit candidates (world TCP poses) instead of the method's generator. ``squeeze``: still
        steps with the gripper closing before the lift (a 2 mm coin between the pads: lifted right after the close
        it stayed in its holder, 3/3; four more steps and it came out, 3/3)."""
        w = self.world
        obj = w.obj(label)
        # what a re-plan of this pick (after a wrist refine) must plan with again: everything but the grasps and the
        # start (``execute_pick``)
        plan_kw = dict(lift=lift, max_eval=max_eval, grasp_kw=grasp_kw, lift_to_z=lift_to_z, approach_rad=approach_rad,
                       descend_rad=descend_rad, open_g=open_g, settle=settle, lift_rad=lift_rad, squeeze=squeeze)
        arms = [arm] if arm else sorted(ARMS, key=lambda a: float(np.linalg.norm(self.kin.base(a)[:2, 3] - obj.pos[:2])))
        best, notes, allf = None, [], []
        for a in arms:
            q0 = (q_start or {}).get(a, self.ex.q[a])
            cands = list(grasps) if grasps is not None else G.candidates(obj, self.kin, a, method, **(grasp_kw or {}))
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
                # with a wrist refine, objects longer than ~12 cm need the look from 12-15 cm to fit the
                # frame (perception); the hover is the pre-grasp
                pre_d = 0.13 if (self.refine is not None and float(obj.asset.extents.max()) > 0.12) else None
                # an irregular object's bbox is no obstacle model for its own pick (a figurine's slice grasp sits
                # inside its bbox; the slice generator keeps the finger corridors free of its mesh)
                own = (label,) if g.method in ("slice", "antipodal") else ()
                to_pre = self.motion.to_pose(a, q0, g.pre(pre_d), approach_rad, "pick_approach", gripper=go, ignore=own)
                if not to_pre.ok and pre_d is None:
                    # a tall object's top grasp: the default pre-grasp is out of reach (figurine, 19 cm)
                    to_pre = self.motion.to_pose(a, q0, g.pre(0.035), approach_rad, "pick_approach", gripper=go,
                                                 ignore=own)
                    if not to_pre.ok:
                        # at the edge of reach (a coin 0.52 m out): a 2 cm pre-grasp still plans
                        to_pre = self.motion.to_pose(a, q0, g.pre(0.02), approach_rad, "pick_approach", gripper=go,
                                                     ignore=own)
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
                box = held_box(obj, T_tcp_obj)
                T_lift = g.T.copy()
                T_lift[2, 3] = max(g.T[2, 3] + lift, lift_to_z or 0.0)
                if g.narrow:
                    # a slice between neighbours comes out along its own plane (twin: slices leaning 5 deg in their
                    # rack; a vertical lift closes the 13 mm gap to the next one by 9 mm over 10 cm)
                    u = up_in_plane(obj)
                    T_lift[:3, 3] = g.T[:3, 3] + u * (float(T_lift[2, 3] - g.T[2, 3]) / max(float(u[2]), 0.5))
                up = self.motion.line(a, down.q_end, T_lift, lift_rad, "pick_lift", gripper=CLOSED, ignore=(label,),
                                      held=box)
                if not up.ok:
                    n_coll += up.status == "collision"
                    n_ik += up.status != "collision"
                    continue
                p = ArmPlan(a, np.asarray(q0, float), self.ex.g[a])
                p.grip(go, 0, "open_for_grasp").move(to_pre, "approach").move(down, "descend").hold(settle, "settle")
                from dojo.skills.kinematics import grip_to_gap
                p.grip(CLOSED, close_steps(grip_to_gap(go), g.width), "close").hold(squeeze, "squeeze").move(up, "lift")
                cost = p.steps + 30 * g.score
                feas.append((cost, PickPlan(a, p, g, T_tcp_obj, p.steps, label, method, dict(plan_kw))))
                if len(feas) >= keep:
                    break
            if feas:
                feas.sort(key=lambda x: x[0])
                allf.extend(feas)
                if best is None or feas[0][0] < best[0]:
                    best = feas[0]
            notes.append(f"{a}: {len(feas)} feasible of {min(len(cands), max_eval)} (ik {n_ik}, collision {n_coll})")
        if return_all and allf:
            return [p for _, p in sorted(allf, key=lambda x: x[0])]
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

    def held_pose7(self, arm: str) -> tuple[str, list] | None:
        """(asset key, world pose7) of what ``arm`` holds, from the in-hand estimate (commanded TCP @
        in-hand transform), e.g. for perception to mask it as an occluder; None if empty."""
        h = self.held.get(arm)
        if h is None or h.label not in self.world.objects:
            return None
        T = self.kin.tcp(arm, self.ex.q[arm]) @ h.T_tcp_obj
        return self.world.obj(h.label).asset.key, T_to_pose7(T).tolist()

    def _refine(self, label, arm: str, kind: str):
        """Call the refine hook, then refresh. Before the call ``self.refine_ctx`` holds {"kind",
        "label", "arm", "held": (asset key, pose7) | None} (at a place the carried object sits between
        the wrist camera and the support). Returns a re-hover hint if the hook gave one:
        ``{"rehover": (pos, quat)[, "rehover_dz": m]}`` as return value or ``world.notes["rehover"]``."""
        if self.refine is None:
            return None
        held = self.held.get(arm)
        if label in self.no_refine or (held is not None and held.label in self.no_refine):
            return None
        self.refine_ctx = {"kind": kind, "label": label, "arm": arm, "held": self.held_pose7(arm)}
        out = self.refine(self, label, arm, kind)
        hint, dz = None, 0.0
        if isinstance(out, dict) and out.get("rehover") is not None:
            hint, dz = out["rehover"], float(out.get("rehover_dz", 0.0))
            self.world.notes.pop("rehover", None)
        else:
            hint = self.world.notes.pop("rehover", None)
            dz = float(self.world.notes.pop("rehover_dz", 0.0))
        self._rehover_dz = dz
        self.refresh()
        return hint

    def _refine_hover(self, label, arm: str, kind: str, rehover=None) -> None:
        """Refine at the hover; if perception could not fit but saw the object elsewhere (hint), move
        the hover over the hinted pose and refine again (up to ``self.max_rehovers`` retries, default 1).
        ``rehover()`` moves the arm (plans from the world, where the hint has been written) and returns
        whether it moved."""
        hint = self._refine(label, arm, kind)
        for i in range(max(1, int(getattr(self, "max_rehovers", 1)))):
            if hint is None or rehover is None or label is None or label not in self.world.objects:
                return
            o = self.world.obj(label)
            o.pos, o.quat = np.asarray(hint[0], float).copy(), np.asarray(hint[1], float).copy()
            self.log(f"[refine] {kind} {label}: re-hover {i + 1} over the hinted pose {np.round(o.pos, 3).tolist()}")
            if not rehover():
                self.refresh()
                return
            hint = self._refine(label, arm, kind)   # after the last allowed re-hover a further hint is ignored

    def _pick_rehover(self, label: str, arm: str, method: str, plan_kw: dict, pp: "PickPlan | None" = None):
        def f() -> bool:
            kw = {k: v for k, v in plan_kw.items() if k not in ("keep", "q_start", "return_all")}
            # the planned grasp (object frame) over the hinted pose first: a pick planned with its place must keep it
            ph = self._same_grasp_pick(pp, kw) if pp is not None else None
            if ph is None:
                ph = self.plan_pick(label, arm, method, q_start={arm: self.ex.q[arm]}, keep=1, **kw)
            if not isinstance(ph, PickPlan):
                return False
            hov = ArmPlan(arm, self.ex.q[arm].copy(), self.ex.g[arm])
            hov.ops = list(ph.plan.ops[:2])
            self.ex.run({arm: hov}, name=f"pick {label} re-hover")
            return True
        return f

    def _place_rehover(self, arm: str, retarget):
        def f() -> bool:
            h = self.held[arm]
            if h is None or retarget is None:
                return False
            T = retarget() @ inv_T(h.T_tcp_obj)
            cur = self.kin.tcp(arm, self.ex.q[arm])
            # optionally higher, so the wrist camera sees past the held object onto the support
            T[2, 3] = max(cur[2, 3], T[2, 3] + 0.03) + max(0.0, min(0.08, getattr(self, "_rehover_dz", 0.0)))
            mv = self.motion.to_pose(arm, self.ex.q[arm], T, CARRY, "place_rehover", gripper=CLOSED, held=h.box,
                                     ignore=(h.label,), ease=3)
            if not mv.ok:
                return False
            self.ex.run({arm: ArmPlan(arm, self.ex.q[arm].copy(), self.ex.g[arm]).move(mv).hold(1)},
                        name="place re-hover")
            return True
        return f

    def _same_grasp_pick(self, pp: "PickPlan", kw: dict) -> "PickPlan | None":
        """``pp``'s grasp, fixed in the object frame (``inv(T_tcp_obj)``), at the object's pose in the world now (a wrist
        refine moved it), planned from the arm's current joints with ``pp``'s own pick arguments; None if infeasible."""
        import dataclasses
        if pp.label not in self.world.objects:
            return None
        g = dataclasses.replace(pp.grasp, T=self.world.obj(pp.label).T @ inv_T(pp.T_tcp_obj))
        kw = {k: v for k, v in kw.items() if k not in ("keep", "q_start", "return_all", "grasps")}
        r = self.plan_pick(pp.label, pp.arm, pp.method, q_start={pp.arm: self.ex.q[pp.arm]}, grasps=[g], keep=1, **kw)
        if not isinstance(r, PickPlan):
            return None
        r.repair = pp.repair
        return r

    def execute_pick(self, pp: PickPlan, method: str | None = None, **plan_kw) -> PickPlan:
        """Run a pick plan. With a refine hook: hover (approach) first, refine the object's pose from
        the wrist, then re-plan and run the descent / close / lift from the refined pose. Returns the
        plan that was actually executed (for verify_pick).

        The re-plan keeps the planned grasp in the object frame (``_same_grasp_pick``): a pick planned together with its
        place (``plan_transfer``: a re-orienting transfer, a relay leg) fixes the in-hand orientation the place needs
        (mode B: a tape measure's 25-deg pinch re-planned with the default grasps came out straight down, the place
        re-planned for it had no IK at the slot, and the nominal plan put it in 33 deg tilted). Unmoved by the refine,
        the original descent runs. When that grasp is infeasible at the refined pose: a pick with a place
        (``pp.repair``) re-plans both together (the executed plan's ``paired_place``), any other pick re-plans with
        the method's grasps, as before; with no plan, the original descent."""
        if self.refine is None:
            self.parallel([pp.plan], name=f"pick {pp.label}")
            return pp
        method = pp.method if method is None else method
        kw = {**pp.plan_kw, **{k: v for k, v in plan_kw.items() if k not in ("keep", "q_start", "return_all", "grasps")}}
        hover = ArmPlan(pp.arm, pp.plan.q0, pp.plan.g0)
        hover.ops = [op for op in pp.plan.ops[:2]]            # open-for-grasp + approach to the pre-grasp
        self.parallel([hover], name=f"pick {pp.label} hover")
        T_plan = pp.grasp.T @ pp.T_tcp_obj                    # the object's pose the pick was planned for
        self._refine_hover(pp.label, pp.arm, "pick", self._pick_rehover(pp.label, pp.arm, method, kw, pp=pp))

        def rest_of(p: PickPlan, note: str) -> PickPlan:
            rest = ArmPlan(p.arm, self.ex.q[p.arm].copy(), self.ex.g[p.arm])
            rest.ops = p.plan.ops[2:]
            self.ex.run({p.arm: rest}, name=f"pick {p.label} {note}")
            return p

        o = self.world.objects.get(pp.label)
        hover_q = np.asarray(hover.compile()[0][-1][0], float) if hover.compile()[0] else np.asarray(pp.plan.q0, float)
        if o is not None and float(np.abs(self.ex.q[pp.arm] - hover_q).max()) < 1e-6:
            from dojo.skills.geometry import angle_between_R
            dp = float(np.linalg.norm(o.T[:3, 3] - T_plan[:3, 3]))
            if dp < 2e-4 and math.degrees(angle_between_R(o.R, T_plan[:3, :3])) < 0.2:
                return rest_of(pp, "descent")                 # the refine saw it where it was planned
        pp2 = self._same_grasp_pick(pp, kw)
        if pp2 is None and pp.repair is not None:
            rr = pp.repair({pp.arm: self.ex.q[pp.arm].copy()})
            if isinstance(rr, tuple):
                pp2, pl2 = rr
                pp2.paired_place = pl2
                self.log(f"[pick] {pp.label}: planned grasp infeasible at the refined pose; re-paired "
                         f"({pp2.grasp.note})")
        elif pp2 is None:
            pp2 = self.plan_pick(pp.label, pp.arm, method, q_start={pp.arm: self.ex.q[pp.arm]}, **kw)
            pp2 = pp2 if isinstance(pp2, PickPlan) else None
        if pp2 is None:
            return rest_of(pp, "descent")                     # keep the original descent
        self.ex.run({pp2.arm: pp2.plan}, name=f"pick {pp.label} refined")
        return pp2

    def run_pick(self, pp: PickPlan) -> SkillResult:
        """Execute + verify a pick; ``self.last_pick`` = the plan that ran (a refine may re-plan it)."""
        before = self.world.obj(pp.label).pos.copy()
        self.last_pick = pp
        try:
            pp = self.execute_pick(pp)
            self.last_pick = pp
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
        box = held_box(o, T_rel)
        self.held[pp.arm] = Held(pp.label, T_rel, box, pp.grasp)
        r = SkillResult(True, "pick", "ok", pp.steps, pp.arm, exp, meas,
                        f"{pp.grasp.method} {pp.grasp.note}, in-hand drift {drift*1000:.1f} mm")
        if drift > LOOSE_DRIFT:
            r.status = "loose"
            r.note += " (loose)"
        return r

    def regrasp(self, arm: str) -> SkillResult:
        """A loose grasp (the object shifted or turned in the pads while lifting; it slid out mid-carry
        in the twin): lower the object straight down to its support, open, back off, so the next pick
        plans a fresh grasp from its measured pose."""
        h = self.held[arm]
        if h is None:
            return SkillResult(False, "regrasp", "logic", 0, arm, note="nothing held")
        o = self.world.obj(h.label)
        T_tcp = self.kin.tcp(arm, self.ex.q[arm])
        T_dn = T_tcp.copy()
        T_dn[2, 3] -= max(0.0, o.bottom_z - TABLE_Z - 0.006)
        dn = self.motion.line(arm, self.ex.q[arm], T_dn, SLOW, "regrasp_down", gripper=CLOSED, check=False)
        q_dn = dn.q_end if dn.segments else self.ex.q[arm]
        T_up = self.kin.tcp(arm, q_dn)
        T_up[2, 3] += 0.06
        up = self.motion.line(arm, q_dn, T_up, MEDIUM, "regrasp_up", gripper=OPEN, check=False)
        p = ArmPlan(arm, self.ex.q[arm].copy(), self.ex.g[arm]).move(dn).hold(1).grip(OPEN, OPEN_STEPS).hold(1)
        p.move(up)
        try:
            self.ex.run({arm: p}, name=f"regrasp {h.label}")
        except (BudgetExceeded, EpisodeEnded) as e:
            return SkillResult(False, "regrasp", "step_budget", 0, arm, note=str(e))
        self.held[arm] = None
        self.refresh()
        return SkillResult(True, "regrasp", "ok", p.steps, arm, note=f"{h.label} put down to re-grasp")

    # ------------------------------------------------------------------ place
    def plan_place(self, arm: str, T_obj_target, yaw_sym: int = 1, release_gap: float = 0.003, pre_h: float = 0.06,
                   retreat: float = 0.05, ignore=(), q_start=None, approach_rad: float = CARRY,
                   descend_rad: float = SLOW, open_g: float = OPEN, settle: int = 2, open_steps: int = OPEN_STEPS,
                   extra_yaws=(), pre_open_hold: int = 1, ease: int = 3, return_all: bool = False,
                   keep_tilt: bool = False, held: "Held | None" = None, turn: float = 0.0, release: bool = True,
                   pre_settle: int = 1, grip_only=(), servo: bool = False, servo_ref: str | None = None,
                   seat_tol: float = 0.004, drop_ok: bool = False, probe: bool = True, hover_clear: float = 0.003,
                   fingers_out: bool = False, pre_clear: float = 0.015, ref_top: float | None = None,
                   sym_axis: int | None = None, guard_seated: float | None = None, guard_lag: float | None = None,
                   servo_xy_tol: float | None = None, held_tol: float | None = None, held_gate: float | None = None,
                   blind_up: float | None = None, spiral: float | None = None):
        """Approach above the target, descend in a straight line, settle, open, back out.
        ``yaw_sym``: n-fold symmetry of the object about the vertical (the cheapest equivalent yaw wins).
        ``held``: plan for this grasp instead of the current one (a place planned with its pick).
        ``turn`` (rad): after the descent, turn the object about the vertical through its root (a key in
        its lock) before letting go; ``release=False`` keeps holding (no open, no retreat).
        ``pre_settle``: still steps at the pre-place pose before the in-hand re-measure (precision).
        ``servo``: run the tail closed loop on the measured object (``_servo_tail``: sag-corrected descent,
        depth check, probing around the target, then turn / release); planned like the open-loop tail;
        ``servo_ref``: the receptacle (default the first ``grip_only`` object) whose top sets the hover.
        ``pre_clear``: the pre-place holds the object's lowest corner this far over the highest ignored support.
        ``fingers_out``: the fingertips stay 3 mm above the receptacle's (``servo_ref`` / first ignored object's) top:
        the object is let go there and drops the rest of the way (a bread slice into a toaster slot narrower than the
        open fingers).
        ``ref_top``: the receptacle's top height (world z) for ``fingers_out`` and the servo hover instead of its whole
        bbox top (an egg carton whose open lid stands 12 cm above the plate the eggs go through).
        ``sym_axis``: the object is the same turned about this axis of its own (a disc about its normal): the servo tail
        takes the turn of the target nearest to the object's measured pose (a coin spun in a rim pinch).
        ``guard_seated``: the guarded (mode B) tail's seated tolerance instead of ``GUARD_SEATED`` (a ring threaded 4 mm
        onto a peg that stops on the peg's rounded top is 2-3 mm short of its target, not seated); ``guard_lag``: its
        contact threshold instead of ``GUARD_LAG`` (twin: a ring stopped by a peg's top slid 2.6 mm in the pads and the
        arm lagged only 1.7 mm). ``servo_xy_tol``: once seated, the servo tail corrects the object's measured xy until
        it is this close to the target (two more moves at most; a ring seated 0.6 mm off its neighbour's axis on a peg
        judged at 1 mm).
        Mode B only (no in-hand re-measure; a provider ``held_pose``; ``_held_protocol``): ``held_gate`` (m) = the
        leading point's lateral sigma an in-hand read needs to be used (default half of the place's ``tol_xy``);
        ``held_tol`` (m) = the sigma below which no look is taken (default the gate: a servo place looks only when the
        hover read fails it); ``blind_up`` (m): with no usable read, the guarded tail aims this much above the target
        (a piece meant to be seated deep on a peg is only threaded: pushed down blind on a predicted pose, it binds).
        ``spiral`` (m): with an adopted read, a servo tail that is not seated after its descent first drags the object,
        pressed onto the receptacle, along a spiral out to this radius (a read 0.15-0.3 mm off, blades in +-0.1 mm
        slots), then probes as usual."""
        h = self.held[arm] if held is None else held
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
        allp = []
        tail_kw = dict(release_gap=release_gap, retreat=retreat, ignore=tuple(set(ignore) | {h.label}),
                       descend_rad=descend_rad, open_g=open_g, settle=settle, open_steps=open_steps,
                       pre_open_hold=pre_open_hold, turn=turn, release=release, grip_only=tuple(grip_only),
                       servo=bool(servo), servo_ref=servo_ref, seat_tol=float(seat_tol), drop_ok=bool(drop_ok),
                       probe=bool(probe), hover_clear=float(hover_clear), fingers_out=bool(fingers_out),
                       ref_top=None if ref_top is None else float(ref_top))
        if sym_axis is not None:
            tail_kw["sym_axis"] = int(sym_axis)
        if guard_seated is not None:
            tail_kw["guard_seated"] = float(guard_seated)
        if guard_lag is not None:
            tail_kw["guard_lag"] = float(guard_lag)
        if servo_xy_tol is not None:
            tail_kw["servo_xy_tol"] = float(servo_xy_tol)
        for k, v in (("held_tol", held_tol), ("held_gate", held_gate), ("blind_up", blind_up), ("spiral", spiral)):
            if v is not None:
                tail_kw[k] = float(v)
        # the pre-place pose must hold the object clear of what it goes onto / into (a nested bowl's
        # target is inside the base bowl: 4 cm above it the two boxes still overlap)
        sup_top = max((self.world.obj(l).top_z for l in ignore if l in self.world.objects), default=None)
        best, notes = None, []
        for _, T_obj in sorted(cands, key=lambda x: x[0]):
            T_tcp = T_obj @ inv_T(h.T_tcp_obj)
            T_pre = T_tcp.copy()
            lift = pre_h
            if sup_top is not None:
                corners = (T_obj[:3, :3] @ (np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
                                             * o.asset.extents + o.asset.bbox_min).T).T + T_obj[:3, 3]
                lift = max(pre_h, sup_top + pre_clear - float(corners[:, 2].min()))
            T_pre[2, 3] += release_gap + lift
            to_pre = self.motion.to_pose(arm, q0, T_pre, approach_rad, "place_approach", gripper=CLOSED, held=h.box,
                                         ignore=(h.label,), ease=ease)
            if not to_pre.ok:
                notes.append(f"approach: {to_pre.note}")
                continue
            tail = self._place_tail(arm, to_pre.q_end, T_obj, h, **tail_kw)
            if isinstance(tail, str):
                notes.append(tail)
                continue
            # a place planned with its pick (``held``) starts with the gripper closed on the object, not open as it is
            # now (mode B: the re-plan after the pick failed, this plan ran, and its approach let go of the charger)
            p = ArmPlan(arm, np.asarray(q0, float), CLOSED if held is not None else self.ex.g[arm]).move(to_pre, "approach")
            p.ops.extend(tail.ops)
            cand = PlacePlan(arm, p, T_obj, p.steps, h.label, keep_tilt=keep_tilt)
            cand.approach = to_pre
            cand.tail_kw = tail_kw
            cand.pre_settle = pre_settle
            allp.append(cand)
            if best is None or p.steps < best.steps:
                best = cand
        if return_all and allp:
            return sorted(allp, key=lambda c: c.steps)
        if best is None:
            st = "ik_fail" if any("unreachable" in n or "ik_fail" in n for n in notes) else "collision"
            return SkillResult(False, "place", st, 0, arm, note="; ".join(notes[:4]))
        return best

    def _place_tail(self, arm, q_pre, T_obj, h: "Held", release_gap, retreat, ignore, descend_rad, open_g, settle,
                    open_steps, pre_open_hold, turn: float = 0.0, release: bool = True,
                    grip_only=(), servo: bool = False, servo_ref: str | None = None,
                    seat_tol: float = 0.004, drop_ok: bool = False, probe: bool = True,
                    hover_clear: float = 0.003, fingers_out: bool = False, ref_top: float | None = None,
                    sym_axis: int | None = None, guard_seated: float | None = None,
                    guard_lag: float | None = None, servo_xy_tol: float | None = None,
                    held_tol: float | None = None, held_gate: float | None = None,
                    blind_up: float | None = None, spiral: float | None = None) -> ArmPlan | str:
        T_rel = T_obj @ inv_T(h.T_tcp_obj)
        T_rel[2, 3] += release_gap
        if fingers_out:
            ref = next((l for l in ((servo_ref,) + tuple(grip_only) + tuple(ignore)) if l and l != h.label
                        and l in self.world.objects), None)
            if ref_top is not None:
                T_rel[2, 3] += self.tips_raise(T_rel, float(ref_top) + 0.003)
            elif ref is not None:
                T_rel[2, 3] += self.fingers_raise(T_rel, ref)       # its top, or its height map under the fingers
        down = self.motion.line(arm, q_pre, T_rel, descend_rad, "place_descend", gripper=CLOSED, held=h.box,
                                ignore=ignore, grip_only=grip_only)
        if not down.ok:
            return f"descend: {down.note}"
        t = ArmPlan(arm, np.asarray(q_pre, float), CLOSED)
        t.move(down, "descend").hold(settle, "settle")
        q_last, T_last = down.q_end, T_rel
        if turn:
            # turn the object about the vertical through its root; the TCP follows the grasp. A shorter turn
            # when the wrist cannot go all the way (a key's checker accepts -60 +- 30 deg)
            for frac in TURN_FRACTIONS:
                T_t = np.asarray(T_obj, float).copy()
                T_t[:3, :3] = rot_z(turn * frac) @ T_t[:3, :3]
                T_tcp_t = T_t @ inv_T(h.T_tcp_obj)
                T_tcp_t[2, 3] += release_gap
                tw = self.motion.line(arm, q_last, T_tcp_t, TURN_RAD, "place_turn", gripper=CLOSED,
                                      check=False, max_step_deg=4.0)
                if tw.ok:
                    break
            if not tw.ok:
                return f"turn: {tw.note}"
            t.move(tw, "turn").hold(settle, "turned")
            q_last, T_last = tw.q_end, T_tcp_t
        if not release:
            return t
        T_up = T_last.copy()
        T_up[:3, 3] -= T_last[:3, 0] * retreat          # back out along the approach axis
        if T_last[2, 0] > -0.5:
            T_up[2, 3] += 0.02
        up = self.motion.line(arm, q_last, T_up, MEDIUM, "place_retreat", gripper=open_g, ignore=ignore)
        if not up.ok:
            # straight up instead (a hand pointing sideways over a slot: backing out along its approach left the reach)
            T_up = T_last.copy()
            T_up[2, 3] += 0.04
            up2 = self.motion.line(arm, q_last, T_up, MEDIUM, "place_retreat", gripper=open_g, ignore=ignore)
            if not up2.ok:
                return f"retreat: {up.note}"
            up = up2
        t.grip(open_g, open_steps, "release").hold(pre_open_hold, "let_go").move(up, "retreat")
        return t

    def place(self, arm: str, T_obj_target, tol_xy: float = 0.01, tol_tilt: float = 6.0, tol_yaw: float | None = None,
              **kw) -> SkillResult:
        pp = self.plan_place(arm, T_obj_target, **kw)
        if isinstance(pp, SkillResult):
            return self._record(pp)
        return self._record(self.run_place(pp, tol_xy, tol_tilt, tol_yaw))

    def run_place(self, pp: PlacePlan, tol_xy: float = 0.01, tol_tilt: float = 6.0, tol_yaw: float | None = None,
                  other: list | None = None, other_arm: str | None = None, remeasure: bool = True,
                  tol_z: float = 0.02, retarget=None, ref_label: str | None = None) -> SkillResult:
        """Execute a place: approach (+1 still step), re-measure the object in the hand, re-plan the
        descent for the measured in-hand pose, descend, release, back out. ``other`` = the other
        arm's compiled per-step (q, g) list, run in the same env steps throughout."""
        other = list(other or [])
        if self.held.get(pp.arm) is not None:
            pp.plan.g0 = min(float(pp.plan.g0), float(self.ex.g[pp.arm]))   # the gripper holds: never open on the way
        try:
            if (remeasure and (self.remeasure or self.refine is not None
                               or getattr(self.provider, "held_pose", None) is not None)
                    and getattr(pp, "approach", None) is not None):
                a1 = ArmPlan(pp.arm, pp.plan.q0, pp.plan.g0).move(pp.approach, "approach").hold(
                    int(getattr(pp, "pre_settle", 1)), "still")
                c1 = a1.compile()[0]
                lists = {pp.arm: c1}
                # mode B, a place that always looks at the held object (``held_tol`` 0): the other arm's look runs in
                # the same steps as the approach (perception-held: ~0 extra steps instead of 13-17)
                pre = None if (other_arm or self.remeasure) else self._pre_look(pp, a1)
                if other_arm:
                    lists[other_arm] = other[:len(c1)]
                elif pre is not None:
                    lists[pre["obs"]] = pre["out"].compile()[0]
                self.ex.run_lists(lists, name=f"place {pp.label} approach")
                rest_other = other[len(c1):]
                rehover = self._place_rehover(pp.arm, retarget)
                if pre is not None:
                    def rehover(f=rehover, pre=pre):          # the observer leaves before the holding arm re-hovers
                        if pre.get("out") is not None:
                            self.ex.run({pre["obs"]: pre["back"]}, name="held look back")
                            pre["out"] = None
                        return f()
                self._refine_hover(ref_label, pp.arm, "place", rehover)
                w = self.refresh()
                if retarget is not None and self.refine is not None:
                    pp.T_obj_target = retarget()                   # the support was re-measured
                h = self.held[pp.arm]
                o = w.obj(pp.label)
                measured = self._held_protocol(pp, h, o, tol_xy, tol_tilt, pre=pre) if not self.remeasure else None
                if (pp.tail_kw or {}).get("servo") and not other_arm and (self.remeasure or measured is not None):
                    # mode B with a usable in-hand read: the same closed-loop tail as G3, on the measured transform
                    # (``_servo_obj``) and the arm's measured sag
                    note_m = ""
                    if measured is not None:
                        self.held[pp.arm] = Held(h.label, measured, held_box(o, measured), h.grasp)
                        note_m = (f"held read adopted (in-hand shift "
                                  f"{float(np.linalg.norm(measured[:3, 3] - h.T_tcp_obj[:3, 3])) * 1000:.1f} mm); ")
                    r = self._servo_tail(pp)
                    self.held[pp.arm] = None if (pp.tail_kw or {}).get("release", True) else self.held[pp.arm]
                    v = self.verify_place(pp, tol_xy, tol_tilt, tol_yaw, tol_z)
                    v.note = (v.note + "; " + note_m + r).strip("; ")
                    return v
                if ((pp.tail_kw or {}).get("servo") and not self.remeasure and measured is None and not other_arm
                        and not (pp.tail_kw or {}).get("turn") and not (pp.tail_kw or {}).get("fingers_out")):
                    # no in-hand measurement (mode B): the planned tail rests on a predicted in-hand pose (a belief 3.5 mm
                    # off put a charger's blades beside the outlet): the contact-probing tail instead, never deeper than
                    # ``blind_up`` above the target (a ring is threaded onto its peg, not pushed down to its seat)
                    T_g = np.asarray(pp.T_obj_target, float).copy()
                    T_g[2, 3] += float((pp.tail_kw or {}).get("blind_up") or 0.0)
                    g = self._guarded_tail(pp.arm, T_g, h, pp.tail_kw)
                    self.held[pp.arm] = None
                    v = self.verify_place(pp, tol_xy, tol_tilt, tol_yaw, tol_z)
                    v.note = (v.note + f"; probing tail: {g}").strip("; ")
                    return v
                T_rel_meas = (inv_T(self.kin.tcp(pp.arm, w.arms[pp.arm].q)) @ o.T if self.remeasure
                              else (measured if measured is not None else h.T_tcp_obj))
                shift = float(np.linalg.norm(T_rel_meas[:3, 3] - h.T_tcp_obj[:3, 3]))
                h2 = Held(h.label, T_rel_meas, held_box(o, T_rel_meas), h.grasp)
                tail = self._place_tail(pp.arm, self.ex.q[pp.arm], pp.T_obj_target, h2, **pp.tail_kw)
                if isinstance(tail, str) and pp.keep_tilt:
                    # the object hangs tilted in the hand: put its bbox centre on the target centre with the
                    # orientation it has now (for loose upright tolerances, e.g. upper bowls <= 45 deg)
                    T_now = self.kin.tcp(pp.arm, self.ex.q[pp.arm]) @ T_rel_meas
                    c_t = pp.T_obj_target[:3, :3] @ o.asset.center + pp.T_obj_target[:3, 3]
                    T_keep = T_now.copy()
                    T_keep[:3, 3] = c_t - T_now[:3, :3] @ o.asset.center
                    tail = self._place_tail(pp.arm, self.ex.q[pp.arm], T_keep, h2, **pp.tail_kw)
                    if not isinstance(tail, str):
                        pp.T_obj_target = T_keep
                if isinstance(tail, str):
                    tail = self._place_tail(pp.arm, self.ex.q[pp.arm], pp.T_obj_target, h, **pp.tail_kw)
                    note_rm = f"in-hand shift {shift*1000:.1f} mm (re-plan failed: {tail if isinstance(tail, str) else ''})"
                else:
                    self.held[pp.arm] = h2
                    note_rm = f"in-hand shift {shift*1000:.1f} mm, re-planned"
                if isinstance(tail, str) and (pp.tail_kw or {}).get("drop_ok"):
                    # a tolerant place (a tool anywhere near its slot scores): the object hangs too low in the hand
                    # for the descent; let go here, over the target (twin: a hammer held 13 mm loose)
                    up = self.kin.tcp(pp.arm, self.ex.q[pp.arm]).copy()
                    up[2, 3] += 0.04
                    mv = self.motion.line(pp.arm, self.ex.q[pp.arm], up, MEDIUM, "drop_retreat", gripper=OPEN, check=False)
                    tail = ArmPlan(pp.arm, self.ex.q[pp.arm].copy(), CLOSED).grip(OPEN, OPEN_STEPS, "drop").hold(2)
                    if mv.segments:
                        tail.move(mv, "retreat")
                    note_rm += "; dropped over the target (descent blocked)"
                if isinstance(tail, str) and (pp.tail_kw or {}).get("servo") and not other_arm:
                    # an insertion whose final waypoints the collision model refuses (an in-hand pose predicted, not
                    # measured, in mode B: "palm at table" at the plug's last waypoint): go down slowly, unchecked, and
                    # stop where the arm meets resistance, then let go (the arm's own lag is the contact sensor)
                    g = self._guarded_tail(pp.arm, pp.T_obj_target, h2, pp.tail_kw)
                    self.held[pp.arm] = None
                    v = self.verify_place(pp, tol_xy, tol_tilt, tol_yaw, tol_z)
                    v.note = (v.note + f"; descent refused ({tail[:80]}), guarded: {g}").strip("; ")
                    return v
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
        if (getattr(pp, "tail_kw", None) or {}).get("release", True):
            self.held[pp.arm] = None
        r = self.verify_place(pp, tol_xy, tol_tilt, tol_yaw, tol_z)
        r.note = (r.note + "; " + note_rm).strip("; ")
        return r

    def fingers_raise(self, T_tcp, ref: str, clear: float = 0.003, gripper: float = CLOSED) -> float:
        """How far (m, >= 0) the TCP must rise for the fingers to be ``clear`` above receptacle ``ref``: above its top
        (``tips_raise``), or, when ``coll.heightmaps`` has ``ref``, above its surface under each finger point (a
        hand pointing sideways over a coin bank's slot: its finger roots hang 28 mm low, beside the bank's back)."""
        o = self.world.obj(ref)
        hm = self.coll.heightmaps.get(ref)
        if hm is None:
            return self.tips_raise(T_tcp, o.top_z + clear, gripper)
        from dojo.skills.motion import finger_samples
        P = finger_samples(self.kin, self.kin.flange_from_tcp(np.asarray(T_tcp, float)), gripper)
        loc = (P - o.pos) @ o.R
        ix = ((loc[:, 0] - hm["x0"]) / hm["cell"]).astype(int)
        iy = ((loc[:, 1] - hm["y0"]) / hm["cell"]).astype(int)
        inb = (ix >= 0) & (iy >= 0) & (ix < hm["Z"].shape[1]) & (iy < hm["Z"].shape[0])
        if not inb.any():
            return 0.0
        Z = hm["Z"][iy[inb], ix[inb]]
        fin = np.isfinite(Z)
        if not fin.any():
            return 0.0
        zw = o.pos[2] + o.R[2, 2] * Z[fin]                       # the surface's world z (an upright receptacle)
        return float(max(0.0, np.max(zw + clear - P[inb][fin][:, 2])))

    def _lag(self, arm: str) -> float:
        """Measured-vs-commanded TCP distance (m) of ``arm`` (proprio only; the arm pressing on something lags)."""
        w = self.refresh()
        return float(np.linalg.norm(self.kin.tcp(arm, w.arms[arm].q)[:3, 3] - self.kin.tcp(arm, self.ex.q[arm])[:3, 3]))

    def _guarded_line(self, arm: str, T_goal, rate: float, step_m: float, lag_stop: float) -> tuple[int, float, bool]:
        """Run a straight line to ``T_goal`` in short unchecked steps, stopping when the arm lags more than ``lag_stop``.
        Returns (steps run, last lag, reached)."""
        from dojo.skills.motion import Segment
        # commanded through the arm's sag (proprio only: measured vs commanded TCP), as the servo tail does
        T_cmd = self._sag(arm, self.refresh()) @ np.asarray(T_goal, float)
        mv = self.motion.line(arm, self.ex.q[arm], T_cmd, rate, "guarded", gripper=CLOSED, check=False,
                              max_step_m=step_m, max_step_deg=2.0)
        n, lag = 0, 0.0
        if not mv.segments:
            return 0, 0.0, False
        qs = np.asarray(mv.segments[-1].q)
        for q in qs[1:]:
            self.ex.run({arm: ArmPlan(arm, self.ex.q[arm].copy(), CLOSED).move(
                [Segment(np.stack([self.ex.q[arm], q]), rate, "guarded")]).hold(1)}, name="guarded")
            n += 1
            lag = self._lag(arm)
            if lag > lag_stop:
                return n, lag, False
        return n, lag, bool(mv.ok)

    def _guarded_tail(self, arm: str, T_obj, h: "Held", kw: dict, lag_stop: float = GUARD_LAG, rate: float = 0.02,
                      step_m: float = 0.003, radii=PROBE_RADII, n_dir: int = 8, min_left: int = 25) -> str:
        """Insertion tail without object measurement (proprio only): descend to ``T_obj`` (the held object's target) in
        short unchecked steps and stop where the arm meets resistance (measured TCP lags its command by > ``lag_stop``).
        Stopped more than ``GUARD_SEATED`` short of the target = on the receptacle's surface beside the hole: lift
        ``PROBE_LIFT`` and probe around the contact point (``radii`` x ``n_dir``, nearest first), each probe a guarded
        descent that counts as in when it gets ``PROBE_IN`` below the contact height; then down to the target, let go,
        back off upwards. Probes only while the step budget allows (``min_left`` above the reserve). Returns a note."""
        T_goal = np.asarray(T_obj, float) @ inv_T(h.T_tcp_obj)
        T_goal[2, 3] += float(kw.get("release_gap", 0.0))
        lag_stop = float(kw.get("guard_lag", lag_stop))
        note = ""
        try:
            n, lag, reached = self._guarded_line(arm, T_goal, rate, step_m, lag_stop)
            Tc = self.kin.tcp(arm, self.ex.q[arm])
            short = float(Tc[2, 3] - T_goal[2, 3])
            note = f"{n} steps down, {'reached' if reached else f'contact {short*1000:.1f} mm short (lag {lag*1000:.1f} mm)'}"
            k = 0
            if not reached and short > float(kw.get("guard_seated", GUARD_SEATED)):
                z_c = float(Tc[2, 3])
                probes = [(r * math.cos(2 * math.pi * i / n_dir), r * math.sin(2 * math.pi * i / n_dir))
                          for r in radii for i in range(n_dir)]
                found = False
                for dx, dy in probes:
                    if self.ex.steps_left - self.ex.reserve < min_left:
                        note += ", step budget"
                        break
                    k += 1
                    T_dn = T_goal.copy()                 # the target shifted by the probe offset
                    T_dn[0, 3] += dx
                    T_dn[1, 3] += dy
                    T_up = T_dn.copy()
                    T_up[2, 3] = z_c + PROBE_LIFT
                    up = self.motion.line(arm, self.ex.q[arm], T_up, 0.03, "probe_up", gripper=CLOSED, check=False)
                    if up.segments:
                        self.ex.run({arm: ArmPlan(arm, self.ex.q[arm].copy(), CLOSED).move(up).hold(1)}, name="probe up")
                    T_in = T_dn.copy()
                    T_in[2, 3] = max(float(T_goal[2, 3]), z_c - PROBE_IN)
                    _, lag, ok = self._guarded_line(arm, T_in, rate, 0.002, lag_stop)
                    if ok:
                        found = True
                        n2, lag, reached = self._guarded_line(arm, T_dn, rate, step_m, lag_stop)
                        note += f", in at probe {k} ({dx*1000:+.1f}, {dy*1000:+.1f}) mm"
                        break
                if not found and k:
                    note += f", {k} probes, not in"
        except (BudgetExceeded, EpisodeEnded) as e:
            note += f" ({type(e).__name__})"
        Tq = self.kin.tcp(arm, self.ex.q[arm])
        T_up = Tq.copy()
        T_up[2, 3] += 0.04
        up = self.motion.line(arm, self.ex.q[arm], T_up, MEDIUM, "place_retreat", gripper=OPEN, check=False)
        p = ArmPlan(arm, self.ex.q[arm].copy(), CLOSED).grip(float(kw.get("open_g", OPEN)), OPEN_STEPS, "release").hold(2)
        if up.segments:
            p.move(up, "retreat")
        try:
            self.ex.run({arm: p}, name="guarded release")
        except (BudgetExceeded, EpisodeEnded):
            pass
        return note

    def tips_raise(self, T_tcp, z_min: float, gripper: float = CLOSED) -> float:
        """How far (m, >= 0) the TCP pose must rise for the lowest finger point to be at or above ``z_min``."""
        low = float(self.kin.finger_points(self.kin.flange_from_tcp(np.asarray(T_tcp, float)), gripper)[:, 2].min())
        return max(0.0, z_min - low)

    # ---- closed-loop insertion tail (tail_kw "servo")
    def _sag(self, arm: str, w: WorldState) -> np.ndarray:
        """Commanded-from-measured TCP transform: under load the arm settles a few mm off its command (twin,
        plug_in_charger: a charger held still at the pre-place sat 4-10 mm off where the commanded joints put
        it, the blades landed beside the outlet). A target T is reached by commanding ``C @ T``."""
        Tm = self.kin.tcp(arm, w.arms[arm].q)
        Tc = self.kin.tcp(arm, self.ex.q[arm])
        C = Tc @ inv_T(Tm)
        return C if float(np.linalg.norm(C[:3, 3])) < 0.02 else np.eye(4)

    def _held_read(self, pp: "PlacePlan", h, o, view: str = "hover") -> dict | None:
        """One read of the provider's ``held_pose(arm, prior_T_tcp_obj, asset_key, cfg_)`` (perception's in-hand pose
        of the object in the jaws, ``dojo.perception.held``; G3 re-measures from GT instead). ``cfg_``: the place target
        (perception's insertion axis = world down there, leading point = the mesh's lowest points there) and ``view``
        (where the cameras are: "hover", "look" = the other arm's wrist looks, "show" = the holding arm at a show pose;
        perception uses every camera that sees the object, a test provider may key its noise on it). None: no provider
        or the read raised."""
        return Skills._held_call(self.provider, pp, h, o, view)

    @staticmethod
    def _held_call(provider, pp, h, o, view: str = "hover") -> dict | None:
        f = getattr(provider, "held_pose", None)
        if f is None or h is None:
            return None
        try:
            r = f(pp.arm, prior_T_tcp_obj=h.T_tcp_obj, asset_key=o.asset.key,
                  cfg_={"T_obj_target": np.asarray(pp.T_obj_target, float).tolist(), "view": view}) or {}
        except Exception:  # noqa: BLE001 (an optional read)
            return None
        return r

    @staticmethod
    def held_sigma(r: dict) -> tuple[float, float]:
        """(lateral sigma of the leading point across the insertion axis, mm; sigma of the insertion axis, deg) of an
        ok ``held_pose`` read (perception's ``lead_lat_mm`` / ``axis_deg``; else the leading point's / the object's TCP
        y, z sigma and the rotations about TCP y, z); (inf, inf) for a failed read."""
        if not r or not r.get("ok"):
            return math.inf, math.inf
        sig = r.get("sigma") or {}
        lat = sig.get("lead_lat_mm")
        if lat is None:
            lat = max(float(v) for v in (sig.get("lead_mm") or sig.get("pos_mm") or [99.0, 99.0, 99.0])[1:3])
        tilt = sig.get("axis_deg")
        if tilt is None:
            tilt = max(float(v) for v in (sig.get("rot_deg") or [99.0, 99.0, 99.0])[1:3])
        return float(lat), float(tilt)

    def _measure_held(self, pp: "PlacePlan", h, o, tol_xy: float, tol_tilt: float, view: str = "hover"):
        """Mode B: one in-hand read at the pre-insert hover (``_held_read``). Its sigma is per TCP axis (x = approach,
        y = closing, z = across): the place sees the error across the insertion axis, so the read is used only when ok,
        the leading point's lateral sigma <= ``held_gate`` (default half of ``tol_xy``) and the axis' sigma <= half of
        ``tol_tilt`` (``held_sigma``). Returns T_tcp_obj, else None (the predicted transform stands)."""
        r = Skills._held_call(self.provider, pp, h, o, view)
        lat, tilt = Skills.held_sigma(r)
        kw = getattr(pp, "tail_kw", None) or {}
        gate = 1000.0 * float(kw["held_gate"] if kw.get("held_gate") is not None else 0.5 * tol_xy)
        if lat > gate or tilt > 0.5 * tol_tilt:
            return None
        return np.asarray(r["T_tcp_obj"], float)

    def _held_protocol(self, pp: "PlacePlan", h, o, tol_xy: float, tol_tilt: float, pre: dict | None = None):
        """Mode B, at the pre-insert hover (``run_place``): the in-hand transform the tail may use, or None.

        1. Read at the hover (own wrist + head + the other wrist where it sees the object). Good enough: ok, lateral
           sigma <= ``held_tol`` (default: the gate; 0 = never: always look while the budget allows, for insertions
           whose clearance is far below what a hover read gives) and axis sigma <= half of ``tol_tilt``.
        2. Not good enough (failed, ambiguous: the own wrist looks along the approach and cannot tell a turn about the
           closing axis from a slide along it), a servo place, a provider ``show_pose`` and the step budget allowing:
           the provider's candidates (observer looks of the other arm's wrist at the hover; show poses of the holding arm
           in front of the head camera, with an observer look), planned with our own IK + collision model
           (``_plan_held_look``), at most one look and one show pose (``HELD_LOOK_TRIES``); at each, a still step or
           two, a read, back to the hover. Stops at the first ok read within the gate.
        3. The gate: ok, lateral sigma <= ``held_gate`` (default half of ``tol_xy``), axis sigma <= half of
           ``tol_tilt``. The LAST read within it is used (a look sees more than the hover; the runtime's belief keeps
           the last ok read too); none: None (the caller's safe fallback: the guarded / probing tail, ``blind_up``).
        ``pre`` (``_pre_look``): the observer already looks (it moved during the approach): the first read is that look,
        then the observer goes back; step 2 then tries a show pose only.
        Every read goes to ``self.held_events`` ({label, arm, view, ok, why, lat_mm, axis_deg, steps, adopted})."""
        if getattr(self.provider, "held_pose", None) is None or h is None:
            return None
        kw = pp.tail_kw or {}
        gate = 1000.0 * float(kw["held_gate"] if kw.get("held_gate") is not None else 0.5 * tol_xy)
        good = 1000.0 * float(kw["held_tol"]) if kw.get("held_tol") is not None else gate
        gate_tilt = 0.5 * tol_tilt
        reads = []

        def take(view: str, steps: int):
            r = self._held_read(pp, h, o, view)
            if r is None:
                return None
            lat, tilt = self.held_sigma(r)
            ev = {"label": h.label, "arm": pp.arm, "view": view, "ok": bool(r.get("ok")), "why": r.get("why", ""),
                  "lat_mm": None if math.isinf(lat) else round(lat, 3),
                  "axis_deg": None if math.isinf(tilt) else round(tilt, 3), "steps": int(steps),
                  "step": int(self.ex.steps_used), "adopted": False}
            self.held_events.append(ev)
            if r.get("ok") and r.get("T_tcp_obj") is not None:
                reads.append((lat, tilt, np.asarray(r["T_tcp_obj"], float), ev))
            self.log(f"[held] {h.label} {view}: ok={ev['ok']} {ev['why']} lat {ev['lat_mm']} mm axis {ev['axis_deg']} deg"
                     + (f" ({steps} steps)" if steps else ""))
            return lat <= (good if view == "hover" else gate) and tilt <= gate_tilt
        if pre is not None and pre.get("out") is not None:
            self.ex.run({pre["obs"]: ArmPlan(pre["obs"], self.ex.q[pre["obs"]].copy(), self.ex.g[pre["obs"]]).hold(
                HELD_LOOK_STILL)}, name="held look still")
            first = take("look", pre["steps"])
            self.ex.run({pre["obs"]: pre["back"]}, name="held look back")
            pre["out"] = None
            done = {"look"}
        else:
            first = take("hover", 0)
            done = set()
        if first is None:
            return None
        if (not first and self.held_looks and kw.get("servo")
                and callable(getattr(self.provider, "show_pose", None))):
            self._held_look(pp, h, o, take, done=done)
        ok = [x for x in reads if x[0] <= gate and x[1] <= gate_tilt]
        if not ok:
            return None
        ok[-1][3]["adopted"] = True
        return ok[-1][2]

    def _held_look(self, pp: "PlacePlan", h, o, take, done: set | None = None) -> None:
        """Step 2 of ``_held_protocol``: the provider's look / show candidates (``show_pose(arm, T_tcp_obj, asset_key,
        cfg_)`` -> {"poses": [{"name", "arm", "tcp" (4x4, the holding arm; "hover_look": where it is), "observer":
        {"arm", "tcp"} | None, ...}]}, best first), each planned by ``_plan_held_look``; run while the budget leaves the
        insertion tail its steps + ``HELD_LOOK_MIN_LEFT``."""
        try:
            props = self.provider.show_pose(pp.arm, T_tcp_obj=h.T_tcp_obj, asset_key=o.asset.key,
                                            cfg_={"T_obj_target": np.asarray(pp.T_obj_target, float).tolist()}) or {}
        except Exception as e:  # noqa: BLE001 (an optional proposal)
            self.log(f"[held] show_pose failed: {type(e).__name__}: {str(e)[:80]}")
            return
        appr = getattr(pp, "approach", None)
        tail_need = max(30, int(pp.steps) - int(appr.steps if appr is not None else 0)) + HELD_LOOK_MIN_LEFT
        tried: set = set(done or ())
        why_not = []
        for c in list(props.get("poses") or []):
            kind = "look" if c.get("name") == "hover_look" else "show"
            if kind in tried or len(tried) >= HELD_LOOK_TRIES:
                continue
            plan = self._plan_held_look(pp.arm, h, c)
            if isinstance(plan, str):
                why_not.append(f"{c.get('name')}/{(c.get('observer') or {}).get('name', '-')}: {plan}")
                continue
            if self.ex.steps_left - self.ex.reserve < plan["steps"] + tail_need:
                self.log(f"[held] {h.label}: no {kind} ({plan['steps']} steps; {self.ex.steps_left} left, the tail "
                         f"needs {tail_need})")
                continue
            tried.add(kind)
            self.ex.run(plan["out"], name=f"held {kind} {c.get('name', '')}")
            self.ex.run({a: ArmPlan(a, self.ex.q[a].copy(), self.ex.g[a]).hold(HELD_LOOK_STILL) for a in plan["out"]},
                        name=f"held {kind} still")
            good = take(kind, plan["steps"])
            self.ex.run(plan["back"], name=f"held {kind} back")
            if good:
                break
        if why_not:
            self.log(f"[held] {h.label}: infeasible candidates: " + "; ".join(why_not)[:400])

    def _pre_look(self, pp: "PlacePlan", approach: ArmPlan) -> dict | None:
        """Mode B, a servo place with ``held_tol`` 0 (it always looks), a provider ``held_pose`` + ``show_pose``, the
        other arm free: the provider's best feasible observer look at the hover the arm is approaching
        (``cfg_["q_hold"]`` = the hover's joints), planned like ``_plan_held_look`` (collision-checked, the observer's
        hand off the held object at the hover, ``HELD_LOOK_CLEAR`` between the arms on the merged approach / look
        schedule and on the way back), within the budget. -> {"obs", "out", "back": ArmPlan, "steps": the extra steps
        it costs (the look beyond the approach + still + back), "name"} or None."""
        kw = pp.tail_kw or {}
        h = self.held.get(pp.arm)
        appr = getattr(pp, "approach", None)
        if (not kw.get("servo") or kw.get("held_tol") is None or float(kw["held_tol"]) > 0 or not self.held_looks
                or h is None or appr is None or appr.q_end is None
                or getattr(self.provider, "held_pose", None) is None
                or not callable(getattr(self.provider, "show_pose", None))):
            return None
        obs = "right" if pp.arm == "left" else "left"
        if self.held.get(obs) is not None:
            return None
        o = self.world.obj(h.label)
        try:
            props = self.provider.show_pose(pp.arm, T_tcp_obj=h.T_tcp_obj, asset_key=o.asset.key,
                                            cfg_={"T_obj_target": np.asarray(pp.T_obj_target, float).tolist(),
                                                  "q_hold": np.asarray(appr.q_end, float).tolist()}) or {}
        except Exception as e:  # noqa: BLE001 (an optional proposal)
            self.log(f"[held] show_pose (pre-look) failed: {type(e).__name__}: {str(e)[:80]}")
            return None
        q_h, q_o, g_o = np.asarray(appr.q_end, float), self.ex.q[obs].copy(), float(self.ex.g[obs])
        tail_need = max(30, int(pp.steps) - int(appr.steps)) + HELD_LOOK_MIN_LEFT
        why_not = []
        for c in props.get("poses") or []:
            ob = c.get("observer") or {}
            if c.get("name") != "hover_look" or ob.get("tcp") is None:
                continue
            mv = self.motion.to_pose(obs, q_o, np.asarray(ob["tcp"], float), FAST, "held_look", gripper=g_o,
                                     ignore=(h.label,))
            if not mv.ok:
                why_not.append(f"{ob.get('name')}: {mv.note[:50]}")
                continue
            if not self._hand_clear_of_held(obs, mv.q_end, pp.arm, q_h, h):
                why_not.append(f"{ob.get('name')}: hand at the held object")
                continue
            bk = self.motion.joint(obs, mv.q_end, q_o, FAST, "held_look_back", gripper=g_o, ignore=(h.label,))
            out, back = ArmPlan(obs, q_o, g_o).move(mv), ArmPlan(obs, mv.q_end, g_o).move(bk)
            if not bk.ok or not self.clear_of(approach, out, HELD_LOOK_CLEAR) or not self.clear_of(
                    ArmPlan(pp.arm, q_h, CLOSED).hold(1), back, HELD_LOOK_CLEAR):
                why_not.append(f"{ob.get('name')}: arms too close / no way back")
                continue
            extra = max(0, out.steps - approach.steps) + HELD_LOOK_STILL + back.steps
            if self.ex.steps_left - self.ex.reserve < approach.steps + extra + tail_need:
                self.log(f"[held] {h.label}: no pre-look ({extra} steps; {self.ex.steps_left} left)")
                return None
            return {"obs": obs, "out": out, "back": back, "steps": int(extra), "name": ob.get("name")}
        if why_not:
            self.log(f"[held] {h.label}: no pre-look: " + "; ".join(why_not)[:300])
        return None

    def _plan_held_look(self, arm: str, h: "Held", c: dict) -> dict | str:
        """One ``show_pose`` candidate as executable plans: the holding arm to ``c["tcp"]`` carrying the object (CARRY,
        held box, collision-checked; nothing when it is already there), the other arm's wrist to ``c["observer"]["tcp"]``
        (FAST; that arm must be free; its hand kept ``HELD_LOOK_HAND_CLEAR`` off the held object), both back where
        they were; the arms ``HELD_LOOK_CLEAR`` apart on both schedules. -> {"out", "back": {arm: ArmPlan}, "steps"} or
        why it is infeasible (str)."""
        from dojo.skills.geometry import angle_between_R
        obs = "right" if arm == "left" else "left"
        q_h, q_o = self.ex.q[arm].copy(), self.ex.q[obs].copy()
        out, back = {}, {}
        T_now = self.kin.tcp(arm, q_h)
        q_h_end = q_h
        T_h = np.asarray(c["tcp"], float) if c.get("tcp") is not None else T_now
        if (float(np.linalg.norm(T_h[:3, 3] - T_now[:3, 3])) > 1e-3
                or math.degrees(angle_between_R(T_h[:3, :3], T_now[:3, :3])) > 1.0):
            mv = self.motion.to_pose(arm, q_h, T_h, CARRY, "held_show", gripper=CLOSED, held=h.box, ignore=(h.label,),
                                     ease=3)
            if not mv.ok:
                return f"show: {mv.note[:60]}"
            bk = self.motion.joint(arm, mv.q_end, q_h, CARRY, "held_show_back", gripper=CLOSED, held=h.box,
                                   ignore=(h.label,), ease=3)
            if not bk.ok:
                return f"show back: {bk.note[:60]}"
            q_h_end = mv.q_end
            out[arm] = ArmPlan(arm, q_h, CLOSED).move(mv)
            back[arm] = ArmPlan(arm, q_h_end, CLOSED).move(bk)
        ob = c.get("observer")
        why = "no observer" if not ob else ("observer busy" if self.held.get(obs) is not None else "")
        if ob and ob.get("tcp") is not None and self.held.get(obs) is None:
            g_o = float(self.ex.g[obs])
            # the held object is checked separately at its pose there (a show pose moves it away from where the world has it)
            mv = self.motion.to_pose(obs, q_o, np.asarray(ob["tcp"], float), FAST, "held_look", gripper=g_o,
                                     ignore=(h.label,))
            if not mv.ok:
                why = f"look: {mv.note[:60]}"
            elif not self._hand_clear_of_held(obs, mv.q_end, arm, q_h_end, h):
                why = "look: the observer's hand at the held object"
            else:
                bk = self.motion.joint(obs, mv.q_end, q_o, FAST, "held_look_back", gripper=g_o, ignore=(h.label,))
                if bk.ok:
                    out[obs] = ArmPlan(obs, q_o, g_o).move(mv)
                    back[obs] = ArmPlan(obs, mv.q_end, g_o).move(bk)
                else:
                    why = f"look back: {bk.note[:60]}"
        if obs not in out:
            # no observer, no look: a show pose seen by the own wrist + head only is no better than the hover
            # (perception-held: plug 7 / 24 within target vs 24 / 24 with the observer)
            return why or "no look"
        if obs in out:
            for sched in (out, back):
                pa = sched.get(arm) or ArmPlan(arm, q_h_end if sched is back else q_h, CLOSED).hold(1)
                if not self.clear_of(pa, sched[obs], HELD_LOOK_CLEAR):
                    return f"arms closer than {HELD_LOOK_CLEAR * 100:.0f} cm ({'out' if sched is out else 'back'})"
        steps = max(p.steps for p in out.values()) + HELD_LOOK_STILL + max(p.steps for p in back.values())
        return {"out": out, "back": back, "steps": int(steps)}

    def _hand_clear_of_held(self, obs: str, q_o, arm: str, q_h, h: "Held") -> bool:
        """The observer's links, palm and fingers stay ``HELD_LOOK_HAND_CLEAR`` outside the held object's box (held by
        ``arm`` at ``q_h``)."""
        Tf = self.kin.flange(obs, q_o)
        P = np.vstack([self.kin.link_points(obs, q_o), self.kin.palm_points(Tf), self.kin.finger_points(Tf)])
        T_obj = self.kin.tcp(arm, q_h) @ h.box.T_tcp_obj
        loc = (P - T_obj[:3, 3]) @ T_obj[:3, :3] - h.box.center
        return not bool(np.all(np.abs(loc) <= h.box.half + HELD_LOOK_HAND_CLEAR, axis=1).any())

    def _servo_obj(self, arm: str, label: str, w: WorldState):
        """The held object as the servo tail sees it: from the world with ``remeasure`` (G3: measured); mode B: the
        adopted in-hand transform (``self.held``) on the measured TCP (a belief that did not follow the read)."""
        o = w.obj(label)
        h = self.held.get(arm)
        if self.remeasure or h is None or h.label != label:
            return o
        import copy
        p7 = T_to_pose7(self.kin.tcp(arm, w.arms[arm].q) @ h.T_tcp_obj)
        o2 = copy.copy(o)
        o2.pos, o2.quat = np.asarray(p7[:3], float), np.asarray(p7[3:], float)
        return o2

    def _servo_tail(self, pp: PlacePlan, seat_tol: float = 0.004, radii=(0.0007, 0.0015, 0.003), n_dir: int = 6,
                    probe_lift: float = 0.006, min_left: int = 25) -> str:
        """Insertion tail, closed loop on the measured object (``tail_kw["servo"]``): from the pre-place, descend
        to just above the receptacle's top, re-measure the object (in-hand pose and the arm's sag) and correct,
        descend to the target slowly, then check the depth. Not seated (the leading end rests on the surface
        beside the hole): lift ``probe_lift`` and probe around the target (``radii`` x ``n_dir``) until it drops
        in. Then the optional turn, release, back out. Returns a note."""
        arm, label, kw = pp.arm, pp.label, pp.tail_kw
        seat_tol = float(kw.get("seat_tol", seat_tol))
        T_obj = np.asarray(pp.T_obj_target, float)
        rate = float(kw.get("descend_rad", SLOW))
        rec = next((l for l in ((kw.get("servo_ref"),) + tuple(kw.get("grip_only") or ())) if l in self.world.objects),
                   None)
        o = self.world.obj(label)
        if kw.get("sym_axis") is not None:
            # the target turned about the object's symmetry axis to the measured pose (a coin may spin in the pads)
            T_obj = self._nearest_spin(T_obj, self._servo_obj(arm, label, self.refresh()), int(kw["sym_axis"]))
        hover = 0.004
        rec_top = kw.get("ref_top")
        if rec_top is None and rec is not None:
            rec_top = self.world.obj(rec).top_z
        if rec_top is not None and o.asset.mesh is not None:
            V = o.asset.mesh[0] @ T_obj[:3, :3].T + T_obj[:3, 3]
            hover = max(0.004, float(rec_top) - float(V[:, 2].min()) + float(kw.get("hover_clear", 0.003)))

        def go(T_goal, step_rad, name, settle=2):
            for _ in range(3):              # a long line that stops short (IK) is resumed from where it got to
                w = self.refresh()
                ob = self._servo_obj(arm, label, w)
                T_rel = inv_T(self.kin.tcp(arm, w.arms[arm].q)) @ ob.T
                T_tcp = self._sag(arm, w) @ T_goal @ inv_T(T_rel)
                mv = self.motion.line(arm, self.ex.q[arm], T_tcp, step_rad, name, gripper=CLOSED, check=False,
                                      max_step_deg=4.0, max_step_m=0.01)
                if mv.segments:
                    self.ex.run({arm: ArmPlan(arm, self.ex.q[arm].copy(), CLOSED).move(mv, name).hold(settle, "settle")},
                                name=f"place {label} {name}")
                if mv.ok or not mv.segments or len(mv.segments[-1].q) < 2:
                    break
            return self._servo_obj(arm, label, self.refresh())

        def above(T, dz):
            T = np.asarray(T, float).copy()
            T[2, 3] += dz
            return T
        ob = go(above(T_obj, hover), rate, "servo_hover")
        # a second, corrected hover at the same height when the first left the object off in xy (sag changes
        # with the pose; twin: a pen's tip came down on the holder's rim and stayed there)
        if kw.get("hover_clear", 0.003) > 0.003 and float(np.linalg.norm(ob.pos[:2] - T_obj[:2, 3])) > 0.002:
            go(above(T_obj, hover), rate, "servo_hover2")
        if kw.get("fingers_out") and rec_top is not None:
            # let go with the fingertips above the receptacle: raise the target by what the measured grasp needs
            w = self.refresh()
            T_rel = inv_T(self.kin.tcp(arm, w.arms[arm].q)) @ self._servo_obj(arm, label, w).T
            if kw.get("ref_top") is None and rec is not None:
                dz = self.fingers_raise(T_obj @ inv_T(T_rel), rec)
            else:
                dz = self.tips_raise(T_obj @ inv_T(T_rel), float(rec_top) + 0.003)
            T_obj = above(T_obj, dz)
            if dz > 0:
                kw = {**kw, "probe": False}
        ob = go(T_obj, min(rate, 0.02), "servo_insert", settle=3)
        err = float(ob.pos[2] - T_obj[2, 3])
        note = f"servo: hover {hover*1000:.0f} mm, after the descent {err*1000:+.1f} mm"
        w_ = self.refresh()
        lag = float(np.linalg.norm(self.kin.tcp(arm, w_.arms[arm].q)[:3, 3] - self.kin.tcp(arm, self.ex.q[arm])[:3, 3]))
        if lag > 0.02 and err > seat_tol:
            # the arm is held off its command (it pushes on something): probing would drag the object around
            kw = {**kw, "probe": False}
            note += f", the arm held {lag*1000:.0f} mm off its command: no probing"
        if (err > seat_tol and kw.get("spiral") and not self.remeasure and kw.get("probe", True)
                and self.ex.steps_left - self.ex.reserve > min_left):
            # mode B with an adopted read (~0.2 mm off) and an opening of +-0.1 mm (a plug's blades): free 2 mm over
            # the contact, then pressed SPIRAL_PRESS past it and dragged along a spiral out to ``spiral``; it went in
            # when the object drops SPIRAL_IN under the contact (the arm's lag is the sensor)
            from dojo.skills.motion import Segment
            z_c = float(T_obj[2, 3]) + err
            go(above(T_obj, err + 0.002), 0.03, "spiral_up", settle=1)
            w = self.refresh()
            C0 = self._sag(arm, w)                       # measured free, before the press
            T_rel = inv_T(self.kin.tcp(arm, w.arms[arm].q)) @ self._servo_obj(arm, label, w).T
            k_sp, found = 0, None
            for dx, dy in spiral_points(float(kw["spiral"])):
                if self.ex.steps_left - self.ex.reserve <= min_left:
                    break
                T_p = T_obj.copy()
                T_p[0, 3] += dx
                T_p[1, 3] += dy
                T_p[2, 3] = z_c - SPIRAL_PRESS
                r_ik = self.kin.ik(arm, C0 @ T_p @ inv_T(T_rel), self.ex.q[arm])
                if not r_ik.ok:
                    continue
                self.ex.run({arm: ArmPlan(arm, self.ex.q[arm].copy(), CLOSED).move(
                    [Segment(np.stack([self.ex.q[arm], r_ik.q]), 0.05, "spiral")])}, name=f"place {label} spiral")
                k_sp += 1
                if float(self._servo_obj(arm, label, self.refresh()).pos[2]) < z_c - SPIRAL_IN:
                    found = T_obj.copy()
                    found[0, 3] += dx
                    found[1, 3] += dy
                    break
            note += (f", pressed spiral {k_sp} steps, " + (f"in at ({found[0, 3] - T_obj[0, 3]:+.4f}, "
                                                            f"{found[1, 3] - T_obj[1, 3]:+.4f}) m" if found is not None
                                                            else "not in"))
            if found is not None:
                T_obj = found
                ob = go(T_obj, min(rate, 0.02), "spiral_insert", settle=3)
                err = float(ob.pos[2] - T_obj[2, 3])
        probes = [(r * math.cos(2 * math.pi * k / n_dir), r * math.sin(2 * math.pi * k / n_dir))
                  for r in radii for k in range(n_dir)] if kw.get("probe", True) else []
        n = 0
        while err > seat_tol and probes and self.ex.steps_left - self.ex.reserve > min_left:
            dx, dy = probes.pop(0)
            n += 1
            T_p = T_obj.copy()
            T_p[0, 3] += dx
            T_p[1, 3] += dy
            go(above(T_p, err + probe_lift), 0.03, "probe_up", settle=1)
            ob = go(T_p, min(rate, 0.02), "probe_in", settle=2)
            err = float(ob.pos[2] - T_obj[2, 3])
            if err <= seat_tol:
                T_obj = T_p
        note += f", {n} probes, final {err*1000:+.1f} mm" if n else ""
        if kw.get("servo_xy_tol") is not None and err <= seat_tol:
            for k in range(3):
                ob = self._servo_obj(arm, label, self.refresh())
                dxy = float(np.linalg.norm(ob.pos[:2] - T_obj[:2, 3]))
                if dxy <= float(kw["servo_xy_tol"]) or k == 2:
                    break
                ob = go(T_obj, min(rate, 0.02), "servo_xy", settle=2)
            note += f", xy {dxy*1000:.1f} mm after {k} corrections"
        turn = float(kw.get("turn", 0.0) or 0.0)
        if turn:
            w = self.refresh()
            T_rel = inv_T(self.kin.tcp(arm, w.arms[arm].q)) @ self._servo_obj(arm, label, w).T
            for frac in TURN_FRACTIONS:
                T_t = T_obj.copy()
                T_t[:3, :3] = rot_z(turn * frac) @ T_t[:3, :3]
                T_tcp = self._sag(arm, w) @ T_t @ inv_T(T_rel)
                mv = self.motion.line(arm, self.ex.q[arm], T_tcp, TURN_RAD, "turn", gripper=CLOSED, check=False,
                                      max_step_deg=4.0)
                if mv.ok:
                    break
            if mv.segments:
                self.ex.run({arm: ArmPlan(arm, self.ex.q[arm].copy(), CLOSED).move(mv, "turn").hold(2, "turned")},
                            name=f"place {label} turn")
            note += f", turned {math.degrees(turn * frac):.0f} deg" + ("" if mv.ok else " (partial)")
        if not kw.get("release", True):
            return note
        Tq = self.kin.tcp(arm, self.ex.q[arm])
        T_up = Tq.copy()
        T_up[:3, 3] -= Tq[:3, 0] * float(kw.get("retreat", 0.05))
        if Tq[2, 0] > -0.5:
            T_up[2, 3] += 0.02
        up = self.motion.line(arm, self.ex.q[arm], T_up, MEDIUM, "place_retreat", gripper=OPEN, check=False)
        p = ArmPlan(arm, self.ex.q[arm].copy(), CLOSED).grip(float(kw.get("open_g", OPEN)), int(kw.get("open_steps", OPEN_STEPS)),
                                                             "release").hold(int(kw.get("pre_open_hold", 1)) + 1, "let_go")
        if up.segments:
            p.move(up, "retreat")
        self.ex.run({arm: p}, name=f"place {label} release")
        return note

    @staticmethod
    def _nearest_spin(T_obj, o, k: int, n: int = 72) -> np.ndarray:
        """``T_obj`` turned about the object's axis ``k`` through its bbox centre to the turn closest to ``o``'s pose."""
        from dojo.skills.geometry import angle_between_R, axis_angle_R
        c = o.asset.center
        ax = np.zeros(3)
        ax[k] = 1.0
        best, bd = np.asarray(T_obj, float), 1e9
        for i in range(n):
            Rk = axis_angle_R(ax, 2 * math.pi * i / n)
            Tl = np.eye(4)
            Tl[:3, :3] = Rk
            Tl[:3, 3] = c - Rk @ c
            T = np.asarray(T_obj, float) @ Tl
            d = angle_between_R(T[:3, :3], o.R)
            if d < bd:
                best, bd = T, d
        return best

    def verify_place(self, pp: PlacePlan, tol_xy=0.01, tol_tilt=6.0, tol_yaw=None, tol_z: float = 0.02) -> SkillResult:
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
        ok = dxy <= tol_xy and abs(dz) < tol_z and tilt <= tol_tilt
        return SkillResult(ok, "place", "ok" if ok else "place_miss", pp.steps, pp.arm, exp, meas, note)

    # ------------------------------------------------------------------ re-orienting transfer
    def plan_transfer(self, label: str, targets: list, arms=None, ignore=(), yaw_sym: int = 1, lift: float = 0.08,
                      place_kw: dict | None = None, pick_kw: dict | None = None, max_full: int = 24, grasps=None,
                      q_start: dict | None = None, max_pick_tilt: float = 50.0, box_tilts=(0.0, 25.0, -25.0),
                      lift_to_pre: bool = False, screen_fingers_out: bool = False, pair_ok=None, yaw_about=None):
        """Pick ``label`` with a grasp that ALSO fits a place pose in ``targets`` (object 4x4 poses, earlier =
        preferred): the grasp fixes the object's orientation in the hand, so a re-orienting place (lying ->
        standing, tip down, upright) is planned together with its pick. Grasps: ``box_grasps`` (every face
        pair, every approach) plus the category's usual candidates. A screen (IK + collision at the grasp
        and at the place) ranks the pairs; the best ``max_full`` are planned in full (pick, then the place
        from the lift's end). ``yaw_sym``: each target is also tried turned by 2 pi k / yaw_sym about the
        vertical through its bbox centre. ``max_pick_tilt``: grasps whose pick approach is further from
        straight down are not used (twin: an end-on grasp of a lying tile, 65 deg from vertical, shifted it
        4 mm while closing and held it with 18 mm drift). ``box_tilts``: approach tilts (deg, about the closing axis)
        of the box grasps (40: a top grasp that still reaches 1 m high at 0.54 m from the base, 0.32 m straight
        down). ``lift_to_pre``: lift the object at least to the pre-place height (a slice carried from its rack into
        a toaster 12 cm taller: the 8 cm lift's joint move to the pre-place cut through the toaster).
        ``screen_fingers_out``: screen each place at the pose ``place_kw["fingers_out"]`` lets go from (the TCP raised
        until the fingertips clear the receptacle's top; a coin dropped into a slot from a grasp at its middle).
        ``pair_ok(grasp, T_tcp_obj, T_obj, T_tcp) -> bool``: an extra test of each screened grasp-place pair.
        ``yaw_about``: the point (object frame) the ``yaw_sym`` turns are about, instead of the bbox centre (a star-shaped
        ring's hole sits 2.9 mm off its bbox centre: turned about the centre, 3 of 4 targets missed the peg by 4-6 mm).
        Returns (PickPlan, PlacePlan) or a SkillResult. The PickPlan's ``repair(q_start)`` re-plans the pair from the
        world as it is then, same arm, same arguments (``execute_pick``, when a refine leaves the planned grasp
        infeasible)."""
        w = self.world
        obj = w.obj(label)
        call = dict(targets=list(targets), ignore=ignore, yaw_sym=yaw_sym, lift=lift, place_kw=place_kw,
                    pick_kw=pick_kw, max_full=max_full, grasps=grasps, max_pick_tilt=max_pick_tilt,
                    box_tilts=box_tilts, lift_to_pre=lift_to_pre, screen_fingers_out=screen_fingers_out,
                    pair_ok=pair_ok, yaw_about=yaw_about)
        place_kw = dict(place_kw or {})
        pick_kw = dict(pick_kw or {})
        arms = list(arms) if arms else sorted(ARMS, key=lambda a: float(np.linalg.norm(self.kin.base(a)[:2, 3] - obj.pos[:2])))
        tg = []
        piv = obj.asset.center if yaw_about is None else np.asarray(yaw_about, float)
        for T in targets:
            T = np.asarray(T, float)
            c = T[:3, :3] @ piv + T[:3, 3]
            for k in range(max(1, yaw_sym)):
                Tk = T.copy()
                Tk[:3, :3] = rot_z(2 * math.pi * k / max(1, yaw_sym)) @ T[:3, :3]
                Tk[:3, 3] = c - Tk[:3, :3] @ piv
                tg.append(Tk)
        ign = tuple(set(ignore) | {label})
        fo_ref = None
        if screen_fingers_out and place_kw.get("fingers_out"):
            ref = next((l for l in ((place_kw.get("servo_ref"),) + tuple(place_kw.get("grip_only") or ()) + tuple(ignore))
                        if l and l in w.objects), None)
            fo_ref = ref
        screened, notes = [], []
        # irregular objects (tools, figurines): how well a grasp holds while the hand turns counts in the ranking
        # (twin: pliers squeezed across their diverging handles, a hammer across its head, slid out; the
        # handle at the centre of mass holds)
        irregular = held_box(obj, np.eye(4)).points is not None or G.com_off(obj) > 0.008
        if irregular and "approach_rad" not in place_kw:
            # uneven objects swing in the pads on fast carries (twin: a wrench held 6 mm off its centre of mass,
            # 4 mm drift after the lift, fell out on the way to its slot)
            place_kw["approach_rad"] = UNEVEN_CARRY
        for a in arms:
            q0 = (q_start or {}).get(a, self.ex.q[a])
            if callable(grasps):
                cands = grasps(a)
            elif grasps is not None:
                cands = list(grasps)
            else:
                cands = G.box_grasps(obj, self.kin, a, tilts=tuple(box_tilts)) + G.candidates(obj, self.kin, a)
                if held_box(obj, np.eye(4)).points is not None:          # irregular: real slices too
                    cands += G.slice_grasps(obj, self.kin, a)
            n_g = n_p = 0
            min_down = math.cos(math.radians(max_pick_tilt))
            n_ik = 0
            cands = sorted(cands, key=lambda g: g.score)            # the IK budget goes to the best grasps first
            for g in cands:
                if n_ik > SCREEN_IK_BUDGET:
                    notes.append(f"{a}: screen stopped after {n_ik} place IKs")
                    break
                if -float(g.T[2, 0]) < min_down:
                    continue
                r = self.kin.ik(a, g.T, q0, n_random=0)
                if not r.ok:
                    continue
                ok, why = self.coll.check(a, r.q, gap_to_grip(g.open_gap()), ignore=(label,))
                if not ok and why.startswith("hits ") and float(g.width) < 0.03:
                    # a thin object between neighbours (bread slices in a shelf): open 4 mm per side only
                    g = G.narrowed(g)
                    ok, why = self.coll.check(a, r.q, gap_to_grip(g.open_gap()), ignore=(label,))
                if not ok:
                    continue
                n_g += 1
                T_tcp_obj = inv_T(g.T) @ obj.T
                box = held_box(obj, T_tcp_obj)
                for ti, T_obj in enumerate(tg):
                    T_tcp = T_obj @ inv_T(T_tcp_obj)
                    if screen_fingers_out and fo_ref is not None:
                        T_tcp[2, 3] += self.fingers_raise(T_tcp, fo_ref)
                    if float(np.linalg.norm(T_tcp[:2, 3] - self.kin.base(a)[:2, 3])) > PLACE_REACH:
                        continue
                    if pair_ok is not None and not pair_ok(g, T_tcp_obj, T_obj, T_tcp):
                        continue
                    n_ik += 1
                    r2 = self.kin.ik(a, T_tcp, w.home_q[a], n_random=0)
                    if not r2.ok or not wrist_safe(r2.q):
                        continue
                    ok, _ = self.coll.check(a, r2.q, CLOSED, ignore=ign, held=box,
                                            grip_only=tuple(place_kw.get("grip_only", ())))
                    if not ok:
                        continue
                    n_p += 1
                    if irregular:
                        # hold quality depends on the grasp in the object frame only: cached across calls (relay
                        # legs re-plan the same object at other poses; twin: 21 min of planning for one hammer)
                        k = (obj.asset.key,) + tuple(np.round((inv_T(obj.T) @ g.T)[:3, :].ravel(), 3))
                        if k not in self._hq_cache:
                            self._hq_cache[k] = G.hold_quality(obj, g.T)
                        hq = 0.5 * self._hq_cache[k]
                    else:
                        hq = 0.0
                    if lift_to_pre:
                        # the lift to the pre-place height must be in reach (a 25-deg top grasp of a slice 0.45 m
                        # out does not reach 1.05 m; the 40-deg one does)
                        T_l = g.T.copy()
                        T_l[2, 3] = max(g.T[2, 3] + lift, self._pre_place_z(obj, g, T_obj, ignore, place_kw))
                        if not self.kin.ik(a, T_l, r.q, n_random=0).ok:
                            continue
                    screened.append((g.score + hq + 0.05 * (ti // max(1, yaw_sym)), a, g, T_obj))
            notes.append(f"{a}: {n_g}/{len(cands)} grasps reachable, {n_p} grasp-place pairs")
        screened.sort(key=lambda x: x[0])
        for _, a, g, T_obj in screened[:max_full]:
            kw_p = dict(pick_kw)
            if lift_to_pre and "lift_to_z" not in kw_p:
                kw_p["lift_to_z"] = self._pre_place_z(obj, g, T_obj, ignore, place_kw)
            pp = self.plan_pick(label, a, grasps=[g], lift=lift, keep=1, q_start=q_start, **kw_p)
            if not isinstance(pp, PickPlan):
                notes.append(f"pick {g.note}: {pp.note[:70]}")
                continue
            held = Held(label, pp.T_tcp_obj, held_box(obj, pp.T_tcp_obj), g)
            pl = self.plan_place(a, T_obj, yaw_sym=1, ignore=ignore, q_start=pp.plan.q_end, held=held, **place_kw)
            if isinstance(pl, PlacePlan):
                pl.transfer_kw = {"ignore": tuple(ignore), **place_kw}
                pp.repair = (lambda q_s, a=a: self.plan_transfer(label, arms=[a], q_start=q_s, **call))
                return pp, pl
            notes.append(f"place {g.note}: {pl.note[:70]}")
        return SkillResult(False, "transfer", "ik_fail" if not screened else "collision", 0, None,
                           note="; ".join(notes[:6]))

    def _pre_place_z(self, obj, g, T_obj, ignore, place_kw) -> float:
        """The pre-place TCP height of a grasp for a target (as ``plan_place`` puts it: the object's lowest corner 15 mm
        over the highest ignored support, at least ``pre_h`` above the target), + 1 cm."""
        T_tcp_t = T_obj @ inv_T(inv_T(g.T) @ obj.T)
        corners = (T_obj[:3, :3] @ (np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
                                    * obj.asset.extents + obj.asset.bbox_min).T).T + T_obj[:3, 3]
        sup = max((self.world.obj(l).top_z for l in ignore if l in self.world.objects), default=None)
        up = float((place_kw or {}).get("pre_h", 0.06))
        if sup is not None:
            up = max(up, sup + float((place_kw or {}).get("pre_clear", 0.015)) - float(corners[:, 2].min()))
        return float(T_tcp_t[2, 3]) + up + 0.01

    def run_transfer(self, pp: PickPlan, pl: PlacePlan, tol_xy: float = 0.01, tol_tilt: float = 6.0,
                     tol_z: float = 0.02) -> SkillResult:
        """Execute a planned transfer: pick, check the grasp, re-plan the place for the measured in-hand
        pose (same target; the nominal plan when the re-plan fails, ``run_place`` re-measures at the
        approach anyway), place."""
        r = self._record(self.run_pick(pp))
        px = getattr(self, "last_pick", None)
        if isinstance(px, PickPlan) and px is not pp and px.label == pp.label:
            # the refine re-planned the pick: a re-paired one brings its own place (same grasp: same place)
            if isinstance(px.paired_place, PlacePlan):
                pl = px.paired_place
            pp = px
        if not r.ok:
            return r
        if r.status == "ok" and self.refine is not None:
            r = self._held_review(pp, r)
        if r.status == "loose" and (tol_xy < LOOSE_OK_TOL or "in-hand view" in r.note):
            # turned / slid in the pads: put it down, the caller re-plans
            self._record(self.regrasp(pp.arm))
            return SkillResult(False, "transfer", "loose", r.steps, pp.arm, note=r.note)
        # a loose grasp on a tolerant place (a tool within 2 cm of its slot) goes on: the place is re-planned for
        # the measured in-hand pose (twin: a hammer held at its centre of mass with 12 mm drift was put down
        # twice and never placed)
        self.refresh()
        kw = dict(getattr(pl, "transfer_kw", {}) or {})
        pl2 = self.plan_place(pp.arm, pl.T_obj_target, yaw_sym=1, **kw)
        use = pl2 if isinstance(pl2, PlacePlan) else pl
        # the idle arm must be clear of the carry including the held object's reach (twin: a wrench carried
        # past the other hand, left open over a failed pick, was knocked out and flung off the table)
        other = "right" if pp.arm == "left" else "left"
        if self.held.get(other) is None:
            ext = float(np.max(self.world.obj(pp.label).asset.extents))
            try:
                self._park_if_blocking(use.plan, min_dist=0.12 + 0.5 * ext)
            except (BudgetExceeded, EpisodeEnded):
                pass
        ign = kw.get("ignore") or ()
        ref = next((l for l in ign if l in self.world.objects), None)
        return self._record(self.run_place(use, tol_xy, tol_tilt, tol_z=tol_z, ref_label=ref))

    def _held_review(self, pp: "PickPlan", r: SkillResult, max_turn_deg: float = 60.0) -> SkillResult:
        """After the lift of a re-orienting transfer, with a refine hook (mode B): let the wrist look at the held
        object (kind "held") and compare its in-hand orientation with the planned one. The place's orientation
        was planned from the pre-grasp estimate; an estimate flipped end for end (perception: pens' tip
        direction, p90 rotation error 26 deg) would put it in tip up. More than ``max_turn_deg`` off = "loose":
        the caller puts it down and re-plans from the new estimate."""
        self._refine(pp.label, pp.arm, "held")
        w = self.world
        if pp.label not in w.objects:
            return r
        T_rel = inv_T(self.kin.tcp(pp.arm, w.arms[pp.arm].q)) @ w.obj(pp.label).T
        c = (float(np.trace(pp.T_tcp_obj[:3, :3].T @ T_rel[:3, :3])) - 1.0) / 2.0
        turn = math.degrees(math.acos(max(-1.0, min(1.0, c))))
        if turn > max_turn_deg:
            r.status = "loose"
            r.note += f" (in-hand view: turned {turn:.0f} deg from the plan)"
        return r

    # ------------------------------------------------------------------ topple
    def plan_topple(self, label: str, up_axis: int, sign: float = 1.0, arm: str | None = None, q_start=None,
                    below_top: float = 0.012, standoff: float = 0.016, over: float = 0.012,
                    push_rad: float = 0.025, open_after: bool = True) -> ArmPlan | SkillResult:
        """Tip a standing object over so that its body axis ``sign * e[up_axis]`` (now horizontal) ends up
        pointing up: closed fingertips straight down ``below_top`` under the top on that side, then a slow
        horizontal push the other way until the fingertips are ``over`` past the far face (beyond the
        tipping point; gravity does the rest), then up and clear. The closing axis runs along the push, so
        the fingers' 18 mm face meets the object."""
        from dojo.skills.kinematics import TIP_BEYOND_TCP, down_R
        o = self.world.obj(label)
        n = o.R[:, up_axis] * sign
        n = np.array([n[0], n[1], 0.0])
        if np.linalg.norm(n) < 0.5:
            return SkillResult(False, "topple", "logic", 0, arm, note=f"{label}: axis {up_axis} is not horizontal")
        n /= np.linalg.norm(n)
        d = -n
        half_n = float(np.abs(o.R.T @ n) @ (o.asset.extents / 2))
        c = o.center_w
        z_tcp = o.top_z - below_top + TIP_BEYOND_TCP
        yaw = math.atan2(d[1], d[0])
        p0 = c[:2] + n[:2] * (half_n + standoff)
        p1 = c[:2] - n[:2] * (half_n + over)
        T_s = T_from([p0[0], p0[1], z_tcp], down_R(yaw))
        T_ab = T_s.copy()
        T_ab[2, 3] = o.top_z + 0.03 + TIP_BEYOND_TCP
        T_e = T_from([p1[0], p1[1], z_tcp], down_R(yaw))
        T_up = T_e.copy()
        T_up[2, 3] = o.top_z + 0.04 + TIP_BEYOND_TCP
        arms = [arm] if arm else sorted(ARMS, key=lambda a: float(np.linalg.norm(self.kin.base(a)[:2, 3] - c[:2])))
        notes = []
        for a in arms:
            q0 = self.ex.q[a] if q_start is None else q_start
            for yw in (yaw, yaw + math.pi):
                Rd = down_R(yw)
                T_ab[:3, :3] = T_s[:3, :3] = T_e[:3, :3] = T_up[:3, :3] = Rd
                m1 = self.motion.to_pose(a, q0, T_ab, FAST, "topple_above", gripper=CLOSED, ignore=(label,))
                if not m1.ok:
                    notes.append(f"{a}: above {m1.note[:60]}")
                    continue
                m2 = self.motion.line(a, m1.q_end, T_s, MEDIUM, "topple_down", gripper=CLOSED, ignore=(label,))
                if not m2.ok:
                    notes.append(f"{a}: down {m2.note[:60]}")
                    continue
                m3 = self.motion.line(a, m2.q_end, T_e, push_rad, "topple_push", gripper=CLOSED, ignore=(label,),
                                      max_step_m=0.01)
                if not m3.ok:
                    notes.append(f"{a}: push {m3.note[:60]}")
                    continue
                m4 = self.motion.line(a, m3.q_end, T_up, MEDIUM, "topple_up", gripper=CLOSED, check=False)
                p = ArmPlan(a, np.asarray(q0, float), self.ex.g[a]).grip(CLOSED, 0, "close_for_push")
                p.move(m1, "topple_above").move(m2, "topple_down").move(m3, "topple_push").hold(1, "topple_wait")
                p.move(m4, "topple_up")
                if open_after:
                    p.grip(OPEN, 1, "open_after_push")
                return p
        return SkillResult(False, "topple", "collision" if any("hits" in n_ for n_ in notes) else "ik_fail", 0, arm,
                           note="; ".join(notes[:4]))

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

    def already_there(self, job: "Job") -> bool:
        """Does the job's object already rest at the job's target (within the job's tolerances)?"""
        try:
            T = job.target(self)
        except Exception:  # noqa: BLE001 - a target that cannot be computed now is not reached
            return False
        o = self.world.obj(job.label)
        c_t = T[:3, :3] @ o.asset.center + T[:3, 3]
        if float(np.linalg.norm(o.center_w[:2] - c_t[:2])) > job.tol_xy or abs(float(o.center_w[2] - c_t[2])) > min(job.tol_z, 0.01):
            return False
        tilt = math.degrees(math.acos(min(1.0, abs(float((o.R.T @ T[:3, :3])[2, 2])))))
        return tilt <= job.tol_tilt

    def can_pick_at(self, label: str, arm: str, xy, method: str = "auto") -> bool:
        """Grasp feasibility from home if ``label`` rested level on the table at ``xy``."""
        from dojo.skills.geometry import R_to_quat
        from dojo.skills.targets import current_R, half_height, level_R
        o = self.world.obj(label)
        saved = (o.pos.copy(), o.quat.copy())
        R = level_R(current_R(self, label))
        c = np.array([xy[0], xy[1], TABLE_Z + half_height(o, R)])
        o.pos, o.quat = c - R @ o.asset.center, R_to_quat(R)
        try:
            pp = self.plan_pick(label, arm, method, q_start={arm: self.world.home_q[arm]}, keep=1)
        finally:
            o.pos, o.quat = saved
        return isinstance(pp, PickPlan)

    # ------------------------------------------------------------------ plan + verify
    def _at_rest(self, exclude=()) -> dict:
        return {o.label: o.pos.copy() for o in self.world.task_objects()
                if o.obj_type != "Geometry" and o.label not in exclude}

    def judge_pick(self, pp: "PickPlan", before: dict, moving=()):
        z0 = self.world.obj(pp.label).pos[2]
        lift = float(self.kin.tcp(pp.arm, pp.plan.q_end)[2, 3] - pp.grasp.T[2, 3])

        def j(o) -> bool:
            if o.failed or pp.label not in o.objects:
                return False
            pos = o.objects[pp.label][0]
            if pos[2] - z0 < 0.6 * max(lift, 0.02):
                return False
            exp = (self.kin.tcp(pp.arm, o.q[pp.arm]) @ pp.T_tcp_obj)[:3, 3]
            if np.linalg.norm(pos - exp) > 0.015:
                return False
            return _others_still(o, before, {pp.label, *moving})

        def score(o) -> float:
            """How close the outcome comes to a held lift (higher is better): rise minus twice the in-hand drift."""
            if o.failed or pp.label not in o.objects:
                return -1.0
            pos = o.objects[pp.label][0]
            exp = (self.kin.tcp(pp.arm, o.q[pp.arm]) @ pp.T_tcp_obj)[:3, 3]
            return float(min(pos[2] - z0, max(lift, 0.02)) - 2.0 * np.linalg.norm(pos - exp))
        j.score = score
        return j

    def judge_place(self, pl: "PlacePlan", before: dict, tol_xy: float, tol_tilt: float, tol_z: float, moving=()):
        o0 = self.world.obj(pl.label)
        exp_c = pl.T_obj_target[:3, :3] @ o0.asset.center + pl.T_obj_target[:3, 3]

        def j(o) -> bool:
            if o.failed or pl.label not in o.objects:
                return False
            p, q = o.objects[pl.label]
            from dojo.skills.geometry import quat_to_R
            R = quat_to_R(q)
            c = R @ o0.asset.center + p
            if np.linalg.norm(c[:2] - exp_c[:2]) > 0.7 * tol_xy or abs(c[2] - exp_c[2]) > tol_z:
                return False
            tilt = math.degrees(math.acos(min(1.0, abs(float((R.T @ pl.T_obj_target[:3, :3])[2, 2])))))
            if tilt > tol_tilt:
                return False
            return _others_still(o, before, {pl.label, *moving})

        def score(o) -> float:
            """Closeness of the outcome to the target (higher is better): -(xy + z error + 1 mm per degree of tilt)."""
            if o.failed or pl.label not in o.objects:
                return -1.0
            p, q = o.objects[pl.label]
            from dojo.skills.geometry import quat_to_R
            R = quat_to_R(q)
            c = R @ o0.asset.center + p
            tilt = math.degrees(math.acos(min(1.0, abs(float((R.T @ pl.T_obj_target[:3, :3])[2, 2])))))
            return -float(np.linalg.norm(c[:2] - exp_c[:2]) + abs(c[2] - exp_c[2]) + 0.001 * tilt)
        j.score = score
        return j

    def verify_choose(self, cands: list[dict], judges: list, held=(), kind: str = "", label: str = "") -> tuple[int | None, list]:
        """Roll candidate segments (per-arm (q, g) lists) in the twin; the first one whose every
        sample passes its judge wins (else the one with the most passing samples)."""
        acts = [self.ex.actions_for(c) for c in cands]
        outs = self.verifier.rollout(acts, k=self.verify_k, held_labels=tuple(held))
        passes = [0] * len(cands)
        for o in outs:
            if judges[o.candidate](o):
                passes[o.candidate] += 1
        self.vstats["segments"] += 1
        self.vstats["candidates"] += len(cands)
        chosen = None
        for i, n in enumerate(passes):
            if n == self.verify_k:
                self.vstats["first_ok" if i == 0 else "later_ok"] += 1
                chosen = i
                break
        none_ok = chosen is None
        if none_ok:
            self.vstats["none_ok"] += 1
            chosen = int(np.argmax(passes)) if max(passes) > 0 else None
            if chosen is None and outs and all(hasattr(j, "score") for j in judges):
                # nothing passed: the candidate whose twin outcome comes closest to the goal, not candidate 0
                sc = [float(np.mean([judges[i].score(o) for o in outs if o.candidate == i] or [-1e9]))
                      for i in range(len(cands))]
                chosen = int(np.argmax(sc))
                self.vstats["best_outcome"] = self.vstats.get("best_outcome", 0) + 1
        self.last_verified = {"idx": chosen, "passes": passes, "outcomes": outs, "none_ok": none_ok and max(passes) == 0,
                              "actions": acts[0 if chosen is None else chosen], "kind": kind, "label": label}
        if self.on_verified is not None:
            self.on_verified(self.last_verified)
        return chosen, passes

    def _verified_pick(self, job) -> SkillResult:
        held = tuple(h.label for h in self.held.values() if h)
        cands = self.plan_pick(job.label, None if job.any_arm else job.arm, job.method, return_all=True, keep=4,
                               **job.pick_kw)
        if not isinstance(cands, list):
            return cands
        cands = cands[:self.n_cands]
        self._park_if_blocking(cands[0].plan)
        if self.refine is not None:
            # hover unverified, refine from the wrist, then plan and verify only the tail
            c0 = cands[0]
            hover = ArmPlan(c0.arm, c0.plan.q0, c0.plan.g0)
            hover.ops = list(c0.plan.ops[:2])
            try:
                self.ex.run({c0.arm: hover}, name=f"pick {job.label} hover")
            except (BudgetExceeded, EpisodeEnded) as e:
                return SkillResult(False, "pick", "step_budget" if isinstance(e, BudgetExceeded) else "ended", 0,
                                   c0.arm, note=str(e))
            self._refine_hover(job.label, c0.arm, "pick", self._pick_rehover(job.label, c0.arm, job.method, job.pick_kw))
            tails = self.plan_pick(job.label, c0.arm, job.method, q_start={c0.arm: self.ex.q[c0.arm]},
                                   return_all=True, keep=4, **job.pick_kw)
            if isinstance(tails, list) and tails:
                cands = tails[:self.n_cands]
            else:                                   # fall back to the original plan minus its hover
                rest = ArmPlan(c0.arm, self.ex.q[c0.arm].copy(), self.ex.g[c0.arm])
                rest.ops = list(c0.plan.ops[2:])
                c0.plan = rest
                cands = [c0]
        before = self._at_rest(exclude=held)
        idx, passes = self.verify_choose([{c.arm: c.plan.compile()[0]} for c in cands],
                                         [self.judge_pick(c, before) for c in cands], held, kind="pick", label=job.label)
        if (self.last_verified or {}).get("none_ok") and self.widen_on_none:
            # the twin rejects every grasp: widen to the planner's next candidates before running the best outcome
            arm0 = cands[0].arm
            more = self.plan_pick(job.label, arm0, job.method, q_start={arm0: self.ex.q[arm0]}, return_all=True,
                                  keep=2 * self.n_cands, **job.pick_kw)
            more = more[len(cands):2 * self.n_cands] if isinstance(more, list) else []
            if more:
                self.vstats["widened"] = self.vstats.get("widened", 0) + 1
                first = (idx, passes, cands)
                idx2, passes2 = self.verify_choose([{c.arm: c.plan.compile()[0]} for c in more],
                                                   [self.judge_pick(c, before) for c in more], held, kind="pick",
                                                   label=job.label)
                if not (self.last_verified or {}).get("none_ok"):
                    idx, passes, cands = idx2, passes2, more
                else:
                    idx, passes, cands = first
        pp = cands[0 if idx is None else idx]
        self.log(f"[verify] pick {job.label}: passes {passes} -> candidate {idx}")
        b = self.world.obj(job.label).pos.copy()
        try:
            self.ex.run_actions(self.ex.actions_for({pp.arm: pp.plan.compile()[0]}), name=f"pick {job.label} (verified)")
        except BudgetExceeded as e:
            return SkillResult(False, "pick", "step_budget", 0, pp.arm, note=str(e))
        except EpisodeEnded as e:
            return SkillResult(False, "pick", "ended", pp.steps, pp.arm, note=str(e))
        job.arm = pp.arm
        r = self.verify_pick(pp, b)
        r.note += f"; verify passes {passes}"
        return r

    def _verified_place(self, job, T, ign, nxt_pick=None) -> tuple[SkillResult, "SkillResult | None"]:
        held_other = tuple(h.label for a, h in self.held.items() if h and a != job.arm)
        if self.refine is not None:
            nxt_pick = None                          # each pick gets its own hover + refine
            first = self.plan_place(job.arm, T, yaw_sym=job.yaw_sym, ignore=ign, **job.place_kw)
            if isinstance(first, PlacePlan) and getattr(first, "approach", None) is not None:
                self._park_if_blocking(first.plan)
                hover = ArmPlan(job.arm, first.plan.q0, first.plan.g0).move(first.approach, "approach").hold(1, "still")
                try:
                    self.ex.run({job.arm: hover}, name=f"place {job.label} hover")
                except (BudgetExceeded, EpisodeEnded) as e:
                    return SkillResult(False, "place", "step_budget" if isinstance(e, BudgetExceeded) else "ended",
                                       0, job.arm, note=str(e)), None
                ref = next((l for l in ign if l in self.world.objects), None)
                self._refine_hover(ref, job.arm, "place", self._place_rehover(job.arm, lambda: job.target(self)))
                T = job.target(self)                 # from the refreshed (refined) world
        other = "right" if job.arm == "left" else "left"

        def attempt(T_, kws):
            """Plan the place variants for target ``T_`` and roll them in the twin -> (idx, passes, variants, cands)."""
            variants = []
            for kw in kws:
                pk = dict(job.place_kw)
                pk.update(kw)
                r = self.plan_place(job.arm, T_, yaw_sym=job.yaw_sym, ignore=ign, return_all=True, **pk)
                if isinstance(r, list):
                    variants.extend(r[:2])
            if not variants:
                return None
            variants = variants[:self.n_cands]
            if nxt_pick is None:
                self._park_if_blocking(variants[0].plan)
            moving = {nxt_pick.label} if nxt_pick is not None else set()
            before = self._at_rest(exclude=set(held_other) | {job.label})
            cands, judges = [], []
            for pl in variants:
                c = {pl.arm: pl.plan.compile()[0]}
                jp = self.judge_place(pl, before, job.tol_xy, job.tol_tilt, job.tol_z, moving)
                if nxt_pick is not None and self.clear_of(pl.plan, nxt_pick.plan):
                    c[other] = nxt_pick.plan.compile()[0]
                    jk = self.judge_pick(nxt_pick, before, moving={job.label})
                    both = (lambda o, jp=jp, jk=jk: jp(o) and jk(o))
                    both.score = jp.score
                    judges.append(both)
                else:
                    judges.append(jp)
                cands.append(c)
            idx, passes = self.verify_choose(cands, judges, held=tuple({job.label, *held_other}), kind="place",
                                             label=job.label)
            return idx, passes, variants, cands

        res = attempt(T, ({}, {"pre_h": 0.04, "descend_rad": 0.03}, {"release_gap": 0.006, "settle": 3}))
        if res is None:
            return self.plan_place(job.arm, T, yaw_sym=job.yaw_sym, ignore=ign, **job.place_kw), None
        if (self.last_verified or {}).get("none_ok") and self.widen_on_none:
            # the twin rejects every candidate: (1) widen the search, (2) look at the support again and re-plan,
            # (3) only then run the best outcome of the first round (verify_choose already chose it)
            self.vstats["widened"] = self.vstats.get("widened", 0) + 1
            r2 = attempt(T, ({"pre_h": 0.06}, {"release_gap": 0.010, "settle": 4}, {"pre_h": 0.05, "descend_rad": 0.02}))
            if r2 is not None and not (self.last_verified or {}).get("none_ok"):
                res = r2
            elif self.refine is not None:
                ref = next((l for l in ign if l in self.world.objects), None)
                if ref is not None:
                    self._refine(ref, job.arm, "place")
                    T2 = job.target(self)
                    r3 = attempt(T2, ({}, {"pre_h": 0.04, "descend_rad": 0.03}, {"release_gap": 0.006, "settle": 3}))
                    if r3 is not None and not (self.last_verified or {}).get("none_ok"):
                        res, T = r3, T2
                        self.vstats["relook_ok"] = self.vstats.get("relook_ok", 0) + 1
        idx, passes, variants, cands = res
        i = 0 if idx is None else idx
        pl, c = variants[i], cands[i]
        self.log(f"[verify] place {job.label}: passes {passes} -> candidate {idx}")
        b_side = self.world.obj(nxt_pick.label).pos.copy() if (nxt_pick is not None and other in c) else None
        try:
            self.ex.run_actions(self.ex.actions_for(c), name=f"place {job.label} (verified)")
        except BudgetExceeded as e:
            return SkillResult(False, "place", "step_budget", 0, job.arm, note=str(e)), None
        except EpisodeEnded as e:
            self.held[job.arm] = None
            return SkillResult(False, "place", "ended", pl.steps, job.arm, note=str(e)), None
        self.held[job.arm] = None
        r = self.verify_place(pl, job.tol_xy, job.tol_tilt, None, job.tol_z)
        r.note += f"; verify passes {passes}"
        side = self.verify_pick(nxt_pick, b_side) if b_side is not None else None
        return r, side

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
            if not holds(job) and self.already_there(job):
                # re-planned episodes (resync / retry) continue instead of undoing finished work
                self._record(SkillResult(True, "jobs", "skipped", 0, job.arm, note=f"{job.label} already at its target"))
                continue
            if not holds(job) and self.verifier is not None:
                r = self._record(self._verified_pick(job))
                if not r.ok:
                    if r.status in ("step_budget", "ended"):
                        return fail(r.status, job.arm, r.note)
                    try:
                        self.ex.run({job.arm: self.plan_home(job.arm, gripper=OPEN)}, name="reset arm")
                    except (BudgetExceeded, EpisodeEnded) as e:
                        return fail("step_budget", job.arm, str(e))
                    self.refresh()
                    r = self._record(self._verified_pick(job))
                    if not r.ok:
                        return fail(r.status, job.arm, f"pick {job.label}: {r.note}")
            if not holds(job):
                pp = plan_pick_job(job)
                if not isinstance(pp, PickPlan):
                    return fail(pp.status, job.arm, f"pick {job.label}: {pp.note}")
                before = self.world.obj(job.label).pos.copy()
                try:
                    pp = self.execute_pick(pp, job.method, **job.pick_kw)
                except (BudgetExceeded, EpisodeEnded) as e:
                    return fail("step_budget" if isinstance(e, BudgetExceeded) else "ended", job.arm, str(e))
                r = self._record(self.verify_pick(pp, before))
                for _ in range(2):
                    if not (r.ok and r.status == "loose"):
                        break
                    self._record(self.regrasp(pp.arm))
                    r = self._record(self.pick(job.label, job.arm if not job.any_arm else None))
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
            for alt in job.alt_targets:
                if isinstance(pl, PlacePlan):
                    break
                pl = self.plan_place(job.arm, alt(self), yaw_sym=job.yaw_sym, ignore=ign, **job.place_kw)
            if not isinstance(pl, PlacePlan):
                return fail(pl.status, job.arm, f"place {job.label}: {pl.note}")
            nxt = jobs[i + 1] if i + 1 < len(jobs) else None
            if self.verifier is not None:
                pn = None
                if (nxt is not None and nxt.arm == other and self.held[other] is None and nxt.prefetch
                        and not job.exclusive):
                    pn = plan_pick_job(nxt)
                    pn = pn if isinstance(pn, PickPlan) and pn.arm == other else None
                r, side = self._verified_place(job, T, ign, pn)
                self._record(r)
                if side is not None:
                    self._record(side)
                if r.status in ("step_budget", "ended"):
                    return fail(r.status, job.arm, r.note)
                if not r.ok and job.required:
                    return fail(r.status, job.arm, f"place {job.label}: {r.note}")
                if bank_after and job.bank:
                    self.bank()
                continue
            side_pick, other_list = None, None
            if (nxt is not None and nxt.arm == other and self.held[other] is None and nxt.prefetch
                    and not job.exclusive and self.refine is None):
                pn = plan_pick_job(nxt)
                if isinstance(pn, PickPlan) and pn.arm == other and self.clear_of(pl.plan, pn.plan):
                    side_pick = (pn, self.world.obj(nxt.label).pos.copy())
                    other_list = pn.plan.compile()[0]
            ref = next((l for l in ign if l in self.world.objects), None)
            r = self._record(self.run_place(pl, job.tol_xy, job.tol_tilt, other=other_list,
                                            other_arm=other if other_list else None, tol_z=job.tol_z,
                                            retarget=lambda job=job: job.target(self), ref_label=ref))
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
    def _park_if_blocking(self, plan: ArmPlan, min_dist: float = 0.12) -> None:
        other = "right" if plan.arm == "left" else "left"
        if self.held.get(other) is None and not self.clear_of(
                plan, ArmPlan(other, self.ex.q[other].copy(), self.ex.g[other]), min_dist):
            self.ex.run({other: self.plan_home(other)}, name=f"park {other}")

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
        if len(plans) == 1:
            # the idle arm must be clear of this arm's whole path (twin: an arm resting over the shared
            # stack site blocked the other arm's pick; the pick missed by 7-10 cm, twice)
            p = plans[0]
            other = "right" if p.arm == "left" else "left"
            if not self.clear_of(p, ArmPlan(other, self.ex.q[other].copy(), self.ex.g[other]), min_dist):
                self.ex.run({other: self.plan_home(other)}, name=f"park {other}")
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


def _others_still(o, before: dict, skip: set, tol: float = 0.006) -> bool:
    for lab, p0 in before.items():
        if lab in skip or lab not in o.objects:
            continue
        if float(np.linalg.norm(o.objects[lab][0] - p0)) > tol:
            return False
    return True


def up_in_plane(o) -> np.ndarray:
    """The world direction of the object's body axis closest to world up (signed up): a slab's in-plane up."""
    R = o.R
    k = int(np.argmax(np.abs(R[2, :])))
    u = R[:, k] * (1.0 if R[2, k] >= 0 else -1.0)
    return u / np.linalg.norm(u)


def upright_tilt(o) -> float:
    return tilt_deg(o.R)
