"""The ``robo`` API the agent's code cells call: typed results over the dojo stack.

Every action goes through the skills layer (``dojo.skills.skills.Skills``: our IK, collision checks against the
estimate, twin verification when the bundle is on, the belief updated after each contact) or a library skill. The
API speaks scene ids (``dojo.agent.scene``); WorldState labels never reach the model.

Bounded cells: before each action ``Robo`` checks the cell's budget (env steps since the cell started,
contacts = grasps / releases / presses / pushes) and raises ``CellCheckpoint`` when it is spent, so control returns
to the model at a primitive boundary. After a contact the head camera frame is queued for the cell's result
(``auto_frames``). ``finish()`` raises ``EpisodeFinished`` after the arms are home.

``ctx`` (built by ``dojo.agent.policy``) provides: ``sk()`` (current Skills), ``new_skills()`` (after a
divergence), ``observe()`` (an official observation, no env step), ``steps_used()``, ``refine(sk, label, arm,
kind)`` and ``search()`` (perception; may be None in mode gt), ``events`` (a list).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dojo.agent.scene import DIGIT_CATEGORIES, KIND, SceneIds, caption
from dojo.agent.session import CellCheckpoint, EpisodeFinished
from dojo.skills.kinematics import ARMS

CAMERAS = {"head": "cam_head", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist",
           "cam_head": "cam_head", "cam_left_wrist": "cam_left_wrist", "cam_right_wrist": "cam_right_wrist"}


@dataclass
class Result:
    ok: bool
    status: str = "ok"
    note: str = ""
    steps: int = 0
    arm: str | None = None
    extra: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok

    def __getitem__(self, k):            # the manual calls a skill's result a dict: r["ok"] works too
        if k in ("ok", "status", "note", "steps", "arm"):
            return getattr(self, k)
        return self.extra[k]

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default

    def __repr__(self) -> str:
        x = "".join(f", {k}={v!r}" for k, v in self.extra.items())
        return (f"Result(ok={self.ok}, status={self.status!r}, steps={self.steps}"
                + (f", arm={self.arm!r}" if self.arm else "") + (f", note={self.note!r}" if self.note else "") + x + ")")


@dataclass
class Obj:
    id: str
    category: str
    kind: str
    center: tuple
    size: tuple
    yaw: float
    tilt: float
    top_z: float
    digit: int | None = None
    name: str | None = None
    colour: str | None = None
    held_by: str | None = None
    confidence: float = 1.0
    asset: int = 0

    def __repr__(self) -> str:
        c = ", ".join(f"{v:.3f}" for v in self.center)
        s = "x".join(f"{v * 100:.1f}" for v in self.size)
        extra = "".join([f", digit={self.digit}" if self.digit is not None else "",
                         f", colour={self.colour!r}" if self.colour else "",
                         f", held_by={self.held_by!r}" if self.held_by else "",
                         f", confidence={self.confidence:.2f}" if self.confidence < 0.99 else ""])
        return (f"Obj({self.id}: {self.category}, asset {self.asset}, center=({c}), size={s} cm, yaw={self.yaw:.0f}, "
                f"tilt={self.tilt:.0f}, top_z={self.top_z:.3f}{extra})")


@dataclass
class ArmView:
    tcp: tuple
    gripper: float
    holding: str | None


@dataclass
class State:
    step: int
    step_lim: int
    steps_left: int
    left: ArmView
    right: ArmView


@dataclass
class CellLimits:
    max_steps: int = 250              # env steps per cell before a checkpoint
    max_contacts: int = 4             # contact actions per cell before a checkpoint


class Robo:
    def __init__(self, ctx, ids: SceneIds, objects_root=None, limits: CellLimits | None = None, auto_frames: bool = True,
                 frame_camera: str = "cam_head"):
        self.ctx = ctx
        self.ids = ids
        self.objects_root = objects_root
        self.limits = limits or CellLimits()
        self.auto_frames = auto_frames
        self.frame_camera = frame_camera
        self.images: list[tuple[str, np.ndarray]] = []     # (label, rgb) queued for the current cell's result
        self.log: list[str] = []                            # one line per action (the cell's action receipt)
        self.finished: dict | None = None
        self.n_actions = 0
        self._cell_step0 = 0
        self._cell_contacts = 0

    # ------------------------------------------------------------------ cell bookkeeping (the loop calls these)
    def begin_cell(self) -> None:
        self._cell_step0 = self.ctx.steps_used()
        self._cell_contacts = 0
        self.images, self.log = [], []

    def drain(self) -> tuple[list[str], list[tuple[str, np.ndarray]]]:
        out = (self.log, self.images)
        self.log, self.images = [], []
        return out

    def _gate(self, name: str) -> None:
        if self.finished is not None:
            raise EpisodeFinished("finish() was already called")
        used = self.ctx.steps_used() - self._cell_step0
        if used >= self.limits.max_steps:
            raise CellCheckpoint(f"cell step budget spent ({used} env steps) before {name}()")
        if self._cell_contacts >= self.limits.max_contacts:
            raise CellCheckpoint(f"{self._cell_contacts} contacts in this cell; check the scene before {name}()")

    def _done(self, name: str, r: Result, contact: bool) -> Result:
        self.n_actions += 1
        self.log.append(f"{name}: {r!r}")
        if contact:
            self._cell_contacts += 1
            if self.auto_frames:
                self._queue_frame(f"after {name}", self.frame_camera)
        return r

    def _queue_frame(self, why: str, cam: str) -> None:
        try:
            obs = self.ctx.observe()
            img = (obs.get("images") or {}).get(cam)
            if img is not None:
                self.images.append((f"{cam} {why} (step {self.ctx.steps_used()})", np.asarray(img)))
        except Exception as e:  # noqa: BLE001 (a missing frame must not break the cell)
            self.log.append(f"(frame unavailable: {type(e).__name__})")

    # ------------------------------------------------------------------ ids
    def _label(self, oid) -> str:
        lab = self.ids.label(str(oid))
        if lab is None:
            raise KeyError(f"unknown object id {oid!r}; ids: {sorted(self.ids.to_label)[:40]}")
        return lab

    def _sync_ids(self, world) -> None:
        """Objects the estimate gained (search, re-estimate) get the next free id of their category."""
        for lab, o in world.objects.items():
            if lab in self.ids.to_id or o.category == "camera_stand":
                continue
            k = sum(1 for i in self.ids.to_label if i.rsplit("_", 1)[0] == o.category)
            oid = f"{o.category}_{k}"
            self.ids.to_label[oid], self.ids.to_id[lab] = lab, oid

    def _arm(self, arm, label=None) -> str:
        if arm in ARMS:
            return arm
        if arm is not None:
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        if label is None:
            return "left"
        x = float(self.ctx.sk().world.obj(label).center_w[0])
        return "left" if x < 0 else "right"

    def _wrap(self, name: str, fn, contact: bool, arm: str | None = None) -> Result:
        from dojo.runtime.belief import Divergence
        from dojo.skills.executor import BudgetExceeded, EpisodeEnded
        self._gate(name)
        n0 = self.ctx.steps_used()
        try:
            out = fn()
        except Divergence as e:
            self.ctx.new_skills()
            self.ctx.events.append({"kind": "divergence_replan", "labels": list(getattr(e, "labels", []) or [])})
            r = Result(False, "divergence", "the scene no longer matches the estimate; it was re-measured: read "
                       "robo.scene() again", self.ctx.steps_used() - n0, arm)
            return self._done(name, r, contact=False)
        except (EpisodeEnded, BudgetExceeded) as e:
            self.log.append(f"{name}: episode over ({type(e).__name__}: {e})")
            raise EpisodeFinished(f"{type(e).__name__}: {e}")
        except KeyError as e:
            r = Result(False, "unknown_object", str(e), 0, arm)
            return self._done(name, r, contact=False)
        r = self._to_result(out, self.ctx.steps_used() - n0, arm)
        return self._done(name, r, contact)

    def _to_result(self, out, steps: int, arm) -> Result:
        if isinstance(out, Result):
            out.steps = steps
            return out
        if isinstance(out, dict):           # library skill result
            sr = out.get("result")
            note = out.get("note") or (getattr(sr, "note", "") if sr is not None else "")
            status = out.get("stage", "ok") if not out.get("ok") else "ok"
            if sr is not None and not out.get("ok"):
                status = f"{out.get('stage')}: {getattr(sr, 'status', '')}"
            return Result(bool(out.get("ok")), status, self._ids_in(str(note))[:400], steps, arm)
        ok = bool(getattr(out, "ok", False))
        return Result(ok, getattr(out, "status", "ok"), self._ids_in(str(getattr(out, "note", "")))[:400], steps,
                      getattr(out, "arm", arm) or arm)

    def _ids_in(self, text: str) -> str:
        for lab, oid in sorted(self.ids.to_id.items(), key=lambda kv: -len(kv[0])):
            if lab and lab != oid and len(lab) > 2:
                text = text.replace(lab, oid)
        return text

    # ------------------------------------------------------------------ perception and state
    def _refresh(self):
        """sk.refresh() (the provider may run a resync check and raise Divergence: re-plan with a fresh Skills)."""
        from dojo.runtime.belief import Divergence
        try:
            return self.ctx.sk().refresh()
        except Divergence as e:
            self.ctx.events.append({"kind": "divergence_replan", "labels": list(getattr(e, "labels", []) or [])})
            self.ctx.new_skills()
            self.log.append("(the scene no longer matched the estimate: it was re-measured)")
            return self.ctx.sk().refresh()

    def scene(self) -> dict:
        w = self._refresh()
        self._sync_ids(w)
        held = {s.holding: a for a, s in w.arms.items() if s.holding}
        out = {}
        for lab, o in w.objects.items():
            oid = self.ids.oid(lab)
            if oid is None:
                continue
            cap = caption(o, self.objects_root)
            out[oid] = Obj(oid, o.category, KIND.get(o.obj_type, o.obj_type.lower()),
                           tuple(round(float(v), 4) for v in o.center_w),
                           tuple(round(float(v), 4) for v in o.asset.extents), round(math.degrees(o.yaw()), 1),
                           round(o.tilt(), 1), round(o.top_z, 4),
                           DIGIT_CATEGORIES[o.category](int(o.idx)) if o.category in DIGIT_CATEGORIES else None,
                           cap.get("name"), cap.get("color"), held.get(lab), float(o.confidence), int(o.idx))
        return dict(sorted(out.items()))

    def obj(self, oid) -> Obj:
        s = self.scene()
        if oid not in s:
            raise KeyError(f"unknown object id {oid!r}")
        return s[oid]

    def state(self) -> State:
        w = self._refresh()
        sk = self.ctx.sk()

        def view(a):
            T = sk.tcp(a)
            h = sk.held.get(a)
            return ArmView(tuple(round(float(v), 4) for v in T[:3, 3]), round(float(sk.ex.g[a]), 3),
                           self.ids.oid(h.label) if h is not None else None)
        return State(int(w.step), int(w.step_lim), int(w.step_lim - w.step), view("left"), view("right"))

    def show(self, camera: str = "head") -> str:
        cam = CAMERAS.get(camera)
        if cam is None:
            raise ValueError(f"camera must be one of head, left_wrist, right_wrist; got {camera!r}")
        self._queue_frame("(requested)", cam)
        return f"{camera} image attached"

    def refine(self, oid, arm=None) -> Obj:
        """Hover a wrist over the object and re-fit its pose (perception). Mode gt: no-op."""
        label = self._label(oid)
        if getattr(self.ctx, "refine", None) is None:
            return self.obj(oid)
        sk = self.ctx.sk()
        arm = self._arm(arm, label)

        def go():
            from dojo.skills.kinematics import down_R
            from dojo.skills.geometry import T_from
            from dojo.skills.executor import ArmPlan
            from dojo.skills.motion import FAST
            o = sk.world.obj(label)
            T = T_from((float(o.center_w[0]), float(o.center_w[1]), float(o.top_z) + 0.16), down_R(0.0))
            mv = sk.motion.to_pose(arm, sk.ex.q[arm], T, FAST, "look", gripper=sk.ex.g[arm])
            if not mv.ok:
                return Result(False, "unreachable", f"no hover pose above {oid}: {mv.note[:160]}", arm=arm)
            sk.ex.run({arm: ArmPlan(arm, sk.ex.q[arm].copy(), sk.ex.g[arm]).move(mv, "look").hold(2, "still")},
                      name="look")
            r = self.ctx.refine(sk, label, arm, "pick")
            sk.refresh()
            return Result(True, "ok" if r else "unchanged", "" if r else "fit not accepted, estimate kept", arm=arm)
        self._wrap("refine", go, contact=False, arm=arm)
        return self.obj(oid)

    def search(self, category=None) -> list:
        if getattr(self.ctx, "search", None) is None:
            return []
        before = set(self.ids.to_label)
        self._wrap("search", lambda: Result(bool(self.ctx.search()), "ok"), contact=False)
        self.scene()
        new = [i for i in self.ids.to_label if i not in before]
        return [i for i in new if category is None or i.rsplit("_", 1)[0] == category]

    # ------------------------------------------------------------------ manipulation
    def pick(self, oid, arm=None, method="auto") -> Result:
        label = self._label(oid)
        sk = self.ctx.sk()
        a = arm if arm in ARMS else None
        return self._wrap("pick", lambda: sk.pick(label, a, method), contact=True, arm=a)

    def _held(self, arm):
        h = self.ctx.sk().held.get(arm)
        if h is None:
            raise ValueError(f"the {arm} arm holds nothing")
        return h

    def place(self, arm, xyz, yaw=None, release=True) -> Result:
        from dojo.skills.targets import rest_pose
        sk = self.ctx.sk()
        h = self._held(self._arm(arm))
        x, y, z = (float(v) for v in xyz)
        yaw_r = None if yaw is None else math.radians(float(yaw))
        T = rest_pose(sk, h.label, (x, y), z, yaw_r)
        return self._wrap("place", lambda: sk.place(arm, T, tol_tilt=10.0), contact=True, arm=arm)

    def place_on(self, arm, support_id, dx=0.0, dy=0.0, yaw=None) -> Result:
        from dojo.skills.targets import on_top
        sk = self.ctx.sk()
        h = self._held(self._arm(arm))
        sup = self._label(support_id)
        b = sk.world.obj(sup)
        xy = (float(b.center_w[0]) + float(dx), float(b.center_w[1]) + float(dy))
        T = on_top(sk, h.label, sup, xy, None if yaw is None else math.radians(float(yaw)))
        return self._wrap("place_on", lambda: sk.place(arm, T, tol_tilt=10.0, ignore=(sup,)), contact=True, arm=arm)

    def drop_into(self, arm, container_id) -> Result:
        from dojo.skills import library as L
        from dojo.skills.targets import rest_pose
        sk = self.ctx.sk()
        h = self._held(self._arm(arm))
        cont = self._label(container_id)

        def go():
            pts = L.drop_points(sk, cont)
            for xy in pts[:3]:
                p = L.plan_drop(sk, arm, xy)
                if not isinstance(p, L.SkillResult):
                    sk.ex.run({arm: p}, name="drop")
                    sk.held[arm] = None
                    sk.refresh()
                    return Result(True, "ok", "released over the container", arm=arm)
            c = sk.world.obj(cont)
            T = rest_pose(sk, h.label, c.center_w[:2], float(c.top_z) + 0.01)
            return sk.place(arm, T, tol_xy=0.06, tol_tilt=90.0, ignore=(cont,))
        return self._wrap("drop_into", go, contact=True, arm=arm)

    def push(self, oid, target_xy, arm=None) -> Result:
        label = self._label(oid)
        sk = self.ctx.sk()
        a = arm if arm in ARMS else None
        xy = tuple(float(v) for v in target_xy[:2])
        return self._wrap("push", lambda: sk.push(label, xy, a), contact=True, arm=a)

    def press(self, button_id, n=1, arm=None) -> Result:
        from dojo.skills import library as L
        label = self._label(button_id)
        sk = self.ctx.sk()
        a = self._arm(arm, label)
        return self._wrap("press", lambda: sk.press(a, L.button_top(sk, label), int(n)), contact=True, arm=a)

    def handover(self, oid, from_arm, to_arm) -> Result:
        label = self._label(oid)
        sk = self.ctx.sk()
        return self._wrap("handover", lambda: sk.handover(label, from_arm, to_arm), contact=True, arm=to_arm)

    def move_to(self, arm, xyz, pitch=90.0, yaw=0.0, path="line") -> Result:
        from dojo.skills.executor import ArmPlan
        from dojo.skills.geometry import T_from
        from dojo.skills.kinematics import tilted_R
        from dojo.skills.motion import FAST, MEDIUM
        sk = self.ctx.sk()
        arm = self._arm(arm)

        def go():
            yaw_r = math.radians(float(yaw))
            base = sk.kin.base(arm)[:3, 3]
            heading = math.atan2(float(xyz[1]) - base[1], float(xyz[0]) - base[0])
            T = T_from(tuple(float(v) for v in xyz), tilted_R(yaw_r, float(pitch), heading))
            q0, g = sk.ex.q[arm].copy(), sk.ex.g[arm]
            h = sk.held.get(arm)
            kw = {"held": h.box, "ignore": (h.label,)} if h is not None else {}
            mv = (sk.motion.line(arm, q0, T, MEDIUM, "move_to", gripper=g, check=False)
                  if path == "line" else sk.motion.to_pose(arm, q0, T, FAST, "move_to", gripper=g, **kw))
            if not mv.ok:
                return Result(False, mv.status, mv.note[:200], arm=arm,
                              extra={"alternatives": mv.alternatives[:3]} if mv.alternatives else {})
            sk.ex.run({arm: ArmPlan(arm, q0, g).move(mv, "move_to")}, name="move_to")
            return Result(True, "ok", arm=arm, extra={"dist_err_mm": round(mv.pos_err * 1000, 1)})
        return self._wrap("move_to", go, contact=False, arm=arm)

    def grip(self, arm, opening) -> Result:
        from dojo.skills.executor import ArmPlan
        sk = self.ctx.sk()
        arm = self._arm(arm)
        g = float(np.clip(opening, 0.0, 1.0))

        def go():
            sk.ex.run({arm: ArmPlan(arm, sk.ex.q[arm].copy(), sk.ex.g[arm]).grip(g, 4, "grip")}, name="grip")
            if g > 0.5 and sk.held.get(arm) is not None:
                sk.held[arm] = None
            sk.refresh()
            return Result(True, "ok", arm=arm, extra={"gripper": round(float(sk.ex.g[arm]), 3)})
        return self._wrap("grip", go, contact=True, arm=arm)

    def wait(self, steps) -> Result:
        sk = self.ctx.sk()
        n = int(max(0, steps))
        return self._wrap("wait", lambda: (sk.ex.hold(n), Result(True, "ok"))[1], contact=False)

    def bank(self) -> Result:
        sk = self.ctx.sk()
        return self._wrap("bank", sk.bank, contact=False)

    def home(self, arms="both") -> Result:
        sk = self.ctx.sk()
        sel = ARMS if arms in ("both", None) else (arms,)
        return self._wrap("home", lambda: sk.home(sel), contact=False)

    def skill(self, name: str, **args) -> Result:
        from dojo.skills import library as L
        if name not in L.SKILLS:
            raise ValueError(f"unknown skill {name!r}; skills: {sorted(L.SKILLS)}")
        sk = self.ctx.sk()
        mapped = self.ids.map_args(args, to="label")
        if name == "press_sequence" and "presses" in mapped:
            mapped["presses"] = [(p[0], int(p[1])) for p in mapped["presses"]]
        if name == "arrange" and "steps" in mapped:
            mapped["steps"] = [tuple(s) for s in mapped["steps"]]
        self.ctx.events.append({"kind": "agent_skill", "skill": name, "args": args})
        return self._wrap(f"skill {name}", lambda: L.SKILLS[name](sk, **mapped), contact=True)

    # ------------------------------------------------------------------ end
    def finish(self, status: str = "done", summary: str = "") -> None:
        if self.finished is None:
            sk = self.ctx.sk()
            r = None
            try:
                r = sk.finish()
            except Exception as e:  # noqa: BLE001 (the episode ends either way)
                self.log.append(f"finish: {type(e).__name__}: {e}")
            self.finished = {"status": str(status), "summary": str(summary)[:500],
                             "home": getattr(r, "status", None) if r is not None else None}
            self.log.append(f"finish({status!r}): arms home, grippers open")
        raise EpisodeFinished(str(status))


__all__ = ["Robo", "Result", "Obj", "State", "CellLimits"]
