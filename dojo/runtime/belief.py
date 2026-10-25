"""The WorldState the expert programs see, and how it is kept between observations.

``RuntimeProvider`` is the skills ``provider`` (``refresh(holding) -> WorldState``). Arms always come
from the official proprio. Objects come from one of two sources:

* mode ``gt`` (dev-only upper bound): the GT rows the driver captured with the latest observation;
* mode ``perception``: a BELIEF. It starts from perception's layout record (inverse layout + looks) and
  is carried forward between checkpoints by predictions, because perception only estimates objects
  resting on the table:
    - the twin's nominal outcome of the verified candidate that actually ran (``on_verified``),
    - otherwise ``KinematicBelief``: an object follows the jaws that closed on it and, when they open,
      drops onto the highest support under its footprint;
  At a resync checkpoint (right after an object was released) the predicted state is rendered against
  the new frames (perception service ``check``: per object, the fraction of the pixels where it is predicted
  front-most that are observed foreground); an object whose prediction the frames do not show is a divergence: the table-level objects are re-estimated, and
  ``Divergence`` asks the policy to re-plan from the corrected state.

The provider also keeps the last at-rest checkpoint (estimate + robot + executed-action index) so the
verifier can load the twin from an estimate while an object is in the jaws (prefix replay), and it
measures the twin's prediction error against GT in mode gt (a dev metric, never fed back).
"""

from __future__ import annotations

import copy
import math

import numpy as np

from dojo.skills.geometry import T_from, T_to_pose7, inv_T
from dojo.skills.kinematics import ARMS, TABLE_Z, X5Kin
from dojo.skills.world import ObjectState, WorldState, arms_from_proprio, table_contains_xy

from .estimate import est_objects, objects_from_gt_rows, objects_from_record, robot_from_state

MAX_JOINT_STEP = 0.2


class Divergence(Exception):
    """The frames disagree with the predicted state beyond the threshold: re-plan."""

    def __init__(self, labels: list[str], detail: dict):
        super().__init__(f"divergence on {labels}")
        self.labels = labels
        self.detail = detail


class KinematicBelief:
    def __init__(self, objects: dict[str, ObjectState], kin: X5Kin, state: dict, attach_margin: float = 0.015):
        self.kin = kin
        self.objects = {k: copy.deepcopy(v) for k, v in objects.items()}
        self.q = {a: np.asarray(state[f"{a}_arm_joint_state"], float).reshape(-1)[:6].copy() for a in ARMS}
        self.g = {a: float(np.asarray(state[f"{a}_ee_joint_state"]).reshape(-1)[0]) for a in ARMS}
        self.attached: dict[str, tuple[str, np.ndarray]] = {}
        self.margin = attach_margin
        self.released: list[str] = []          # labels released since the last ``pop_released``

    def movable(self):
        return [o for o in self.objects.values() if o.obj_type != "Geometry" and not o.clutter]

    def _object_at(self, p) -> str | None:
        best, bd = None, 1e9
        for o in self.movable():
            loc = o.R.T @ (p - o.center_w)
            if np.all(np.abs(loc) <= o.asset.extents / 2 + self.margin):
                d = float(np.linalg.norm(loc))
                if d < bd:
                    best, bd = o.label, d
        return best

    def _set_T(self, label: str, T: np.ndarray) -> None:
        o = self.objects[label]
        p7 = T_to_pose7(T)
        o.pos, o.quat = np.asarray(p7[:3], float), np.asarray(p7[3:], float)
        o.source = "predicted"

    def _drop(self, label: str) -> None:
        o = self.objects[label]
        lo, hi = o.aabb()
        support = TABLE_Z if table_contains_xy(o.center_w) else 0.05
        for other in self.movable():
            if other.label == label:
                continue
            olo, ohi = other.aabb()
            if (olo[0] < hi[0] - 1e-3 and ohi[0] > lo[0] + 1e-3 and olo[1] < hi[1] - 1e-3 and ohi[1] > lo[1] + 1e-3
                    and ohi[2] <= lo[2] + 0.01):
                support = max(support, float(ohi[2]))
        o.pos = o.pos + np.array([0.0, 0.0, support - lo[2]])
        o.source = "predicted"

    def step(self, action: dict) -> None:
        for a in ARMS:
            tgt = np.asarray(action[f"{a}_arm_joint_state"], float).reshape(-1)[:6]
            self.q[a] = self.q[a] + np.clip(tgt - self.q[a], -MAX_JOINT_STEP, MAX_JOINT_STEP)
            g_new = float(np.clip(np.asarray(action[f"{a}_ee_joint_state"]).reshape(-1)[0], 0, 1))
            T_tcp = self.kin.tcp(a, self.q[a])
            if g_new < 0.5 <= self.g[a] and a not in self.attached:
                lab = self._object_at(T_tcp[:3, 3])
                if lab is not None and lab not in [v[0] for v in self.attached.values()]:
                    self.attached[a] = (lab, inv_T(T_tcp) @ self.objects[lab].T)
            if g_new > 0.6 and a in self.attached:
                lab, _ = self.attached.pop(a)
                self._drop(lab)
                self.released.append(lab)
            self.g[a] = g_new
            if a in self.attached:
                lab, T_rel = self.attached[a]
                self._set_T(lab, T_tcp @ T_rel)

    def sync_robot(self, state: dict) -> None:
        for a in ARMS:
            self.q[a] = np.asarray(state[f"{a}_arm_joint_state"], float).reshape(-1)[:6].copy()
            if a in self.attached:
                lab, T_rel = self.attached[a]
                self._set_T(lab, self.kin.tcp(a, self.q[a]) @ T_rel)

    def adopt(self, objects: dict) -> list[str]:
        """Twin prediction ``{label: (pos, quat)}`` for every object it carries; attached objects keep
        following their arm with the twin's in-hand pose."""
        done = []
        for lab, (p, q) in objects.items():
            if lab in self.objects:
                o = self.objects[lab]
                o.pos, o.quat = np.asarray(p, float).copy(), np.asarray(q, float).copy()
                o.source = "predicted"
                done.append(lab)
        for a, (lab, _) in list(self.attached.items()):
            self.attached[a] = (lab, inv_T(self.kin.tcp(a, self.q[a])) @ self.objects[lab].T)
        return done

    def held_labels(self) -> tuple:
        return tuple(v[0] for v in self.attached.values())

    def pop_released(self) -> list[str]:
        r, self.released = self.released, []
        return r


class RuntimeProvider:
    def __init__(self, bridge, task: str, instruction: str, mode: str, assets=None, kin: X5Kin | None = None,
                 gt_source=None, checker=None, div_iou: float = 0.2, div_min_vis: float = 0.5,
                 div_prec: float = 0.4, div_min_px: int = 40, div_min_frac: float = 0.3, confirm_mm: float = 15.0,
                 replan_on_divergence: bool = True, log=None):
        self.bridge = bridge
        self.task, self.instruction, self.mode = task, instruction, mode
        self.assets, self.kin = assets, kin or X5Kin()
        self.gt_source = gt_source              # mode gt only: () -> GT rows of the latest observation
        self.checker = checker                  # mode perception: (obs, objects) -> {label: check}, (obs) -> record
        self.div_iou, self.div_min_vis = div_iou, div_min_vis
        self.div_prec, self.div_min_px, self.div_min_frac = div_prec, div_min_px, div_min_frac
        self.confirm_mm = confirm_mm
        self.replan = replan_on_divergence
        self.log = log or (lambda *a, **k: None)
        self.home_q = None
        self.notes: dict = {}
        self.belief: KinematicBelief | None = None
        self.pred = None
        self.n_seen = 0
        self.checkpoint = None
        self.events: list[dict] = []
        self.stats = {"refresh": 0, "adopted": 0, "resync": 0, "divergence": 0, "pred_err_mm": []}

    # ------------------------------------------------------------------ setup
    def init_objects(self, objects: dict[str, ObjectState]) -> None:
        st = self.bridge.observe()["state"]
        self.belief = KinematicBelief(objects, self.kin, st)
        self.n_seen = self.bridge.n_executed
        self._checkpoint(st)

    def init_from_record(self, record: dict) -> None:
        self.init_objects(objects_from_record(record, self.assets, self.bridge.steps_used))

    def init_from_gt(self) -> None:
        self.init_objects(self._gt_objects())

    def _gt_objects(self) -> dict[str, ObjectState]:
        return objects_from_gt_rows(self.gt_source(), self.assets, self.bridge.steps_used)

    # ------------------------------------------------------------------ predictions
    def on_verified(self, info: dict) -> None:
        """skills hook: the candidate that is about to run, with its twin outcomes."""
        idx = info.get("idx")
        idx = 0 if idx is None else int(idx)
        nom = next((o for o in info["outcomes"] if o.candidate == idx and o.sample == 0), None)
        if nom is None:
            return
        from .verifier import chosen_len
        self.pred = {"objects": nom.objects, "n_start": self.bridge.n_executed, "n_actions": chosen_len(info, idx),
                     "kind": info.get("kind"), "label": info.get("label"), "extra": getattr(nom, "extra", {})}

    def _advance(self, st: dict) -> None:
        allx = self.bridge.executed
        for a in allx[self.n_seen:]:
            self.belief.step(a)
        self.n_seen = len(allx)
        self.belief.sync_robot(st)
        if self.pred is not None and self.n_seen - self.pred["n_start"] >= self.pred["n_actions"]:
            if self.n_seen - self.pred["n_start"] == self.pred["n_actions"] or self.pred["n_actions"] > 0:
                self.belief.adopt(self.pred["objects"])
                self.stats["adopted"] += 1
            self.pred = None

    def _checkpoint(self, st: dict) -> None:
        self.checkpoint = {"objects": est_objects(self.belief.objects), "robot": robot_from_state(st),
                           "n": self.bridge.n_executed, "step": self.bridge.steps_used}

    # ------------------------------------------------------------------ resync
    def resync(self, obs: dict) -> dict:
        """Render-compare the predicted state against the frames; on divergence re-estimate and raise."""
        self.stats["resync"] += 1
        objs = [o for o in self.belief.movable()]
        chk = self.checker.check(obs, objs)
        bad = {}
        for lab, c in chk.items():
            head = c.get("head") or {}
            if head.get("prec") is not None:            # z-buffered: predicted front-most pixels vs foreground
                need = max(self.div_min_px, self.div_min_frac * head.get("n_px", 0))   # judge only what is visible
                if head.get("n_vis", 0) >= need and head["prec"] < self.div_prec:
                    bad[lab] = c
            elif head and head.get("vis", 0) >= self.div_min_vis and head.get("iou", 1.0) < self.div_iou:
                bad[lab] = c
        ev = {"kind": "resync", "step": self.bridge.steps_used, "check": {k: c.get("head") for k, c in chk.items()},
              "diverged": sorted(bad)}
        self.events.append(ev)
        if not bad:
            return ev
        self.stats["checks_failed"] = self.stats.get("checks_failed", 0) + 1
        rec = self.checker.estimate(obs)
        est = objects_from_record(rec, self.assets, self.bridge.steps_used)
        upd = {}
        for lab in bad:
            e = est.get(lab)
            if e is not None and e.confidence >= 0.5 and lab not in self.belief.held_labels():
                o = self.belief.objects[lab]
                d_mm = float(np.linalg.norm(e.center_w[:2] - o.center_w[:2])) * 1000
                if d_mm < self.confirm_mm:
                    continue
                upd[lab] = {"from": np.round(o.pos, 4).tolist(), "to": np.round(e.pos, 4).tolist(), "mm": round(d_mm, 1)}
                o.pos, o.quat, o.source, o.confidence = e.pos.copy(), e.quat.copy(), "perception", e.confidence
        ev["updated"] = upd
        if not upd:
            # the frames disagree but the re-estimate does not place the object elsewhere: keep the belief and
            # go on (re-planning a half-built structure from scratch is worse than a doubtful check)
            self.events.append({"kind": "divergence_unconfirmed", "step": self.bridge.steps_used, "labels": sorted(bad)})
            self.log(f"[runtime] unconfirmed divergence {sorted(bad)}: continue")
            return ev
        self.stats["divergence"] += 1
        self.events.append({"kind": "divergence", "step": self.bridge.steps_used, "labels": sorted(bad), "updated": upd})
        self.log(f"[runtime] divergence {sorted(bad)} updated {sorted(upd)}")
        if self.replan:
            raise Divergence(sorted(bad), ev)
        return ev

    # ------------------------------------------------------------------ skills provider
    def refresh(self, holding: dict | None = None) -> WorldState:
        obs = self.bridge.observe()
        st = obs["state"]
        self.stats["refresh"] += 1
        arms = arms_from_proprio(st, holding)
        if self.home_q is None:
            self.home_q = {a: arms[a].q.copy() for a in ARMS}
        had_pred = self.pred is not None
        self._advance(st)
        released = self.belief.pop_released()
        if self.mode == "gt":
            objects = self._gt_objects()
            if had_pred and self.pred is None:          # dev metric: twin prediction vs GT after the segment
                errs = [float(np.linalg.norm(objects[l].pos - o.pos)) * 1000 for l, o in self.belief.objects.items()
                        if l in objects and not o.clutter]
                if errs:
                    self.stats["pred_err_mm"].append(round(max(errs), 2))
            for l, o in objects.items():                  # keep the belief on the truth (prefix checkpoints)
                if l in self.belief.objects and l not in self.belief.held_labels():
                    self.belief.objects[l].pos, self.belief.objects[l].quat = o.pos.copy(), o.quat.copy()
        else:
            if released and self.checker is not None and not self.belief.held_labels():
                self.resync(obs)
            objects = {k: copy.deepcopy(v) for k, v in self.belief.objects.items()}
        if not self.belief.held_labels() and not any((holding or {}).values()):
            self._checkpoint(st)
        return WorldState(task=self.task, instruction=self.instruction, objects=objects, arms=arms,
                          home_q=self.home_q, step=int(obs["steps_used"]), step_lim=int(obs["step_lim"]),
                          notes=self.notes)


def angle_deg(qa, qb) -> float:
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(qa, qb))))))


__all__ = ["RuntimeProvider", "KinematicBelief", "Divergence"]
