# Vendored unchanged from robot-harness (Mikhail's side) a21dd6f6f2db323bd0fe84f8afd20312cb8dffc6, skills/planar_drag_rgb/belief.py.
# Ported into controlr by stream P (2026-10-28) through dojo/skills/ported/harness_robot.py; logic not modified.
"""belief.py: a tier-B stand-in for the `objects` field that tier-A RoboDojo skills read.

BeliefRobot wraps the skill's robot handle. Every call passes through unchanged except:
  obs()          the real obs (images / proprio / calib; the server is NOT privileged, so it has no `objects`) plus an
                 `objects` dict built from the planner's LOCATED poses (from pixels), each carried with the gripper
                 that holds it (flange pose x the object-in-flange transform fixed at the close);
  gripper()      on `close`, if the fingers stopped on something (grasp_detected or both finger joints > 3 mm), the
                 located object nearest the TCP (xy within attach_xy) is attached to that arm, together with its
                 `contents` (e.g. balls in a cup); on `open`, everything on that arm is detached where it is;
  return_home()  detaches the arm's objects first when it opens the grippers (the default).
Nothing here reads simulator state: object poses come from the planner; their motion comes from proprioception
(eef flange poses). The belief is open loop between looks: the planner re-looks and re-locates after each stage.
"""
import copy
import math

import numpy as np


def rotm(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_of(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        q = [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, float)
    return (q / np.linalg.norm(q)).tolist()


def yaw_quat(yaw_deg):
    h = math.radians(float(yaw_deg)) / 2
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


class BeliefRobot:
    def __init__(self, robot, objects, contents=None, attach_xy=0.03):
        self._r = robot
        self.objs = {k: copy.deepcopy(v) for k, v in objects.items()}
        self.contents = dict(contents or {})          # parent name -> [child names]
        self.att = {"left": {}, "right": {}}          # arm -> {name: (r in flange, R in flange)}
        self.attach_xy = float(attach_xy)
        self.events = []

    def __getattr__(self, name):
        return getattr(self._r, name)

    # ---------------------------------------------------------------- belief update
    @staticmethod
    def _flange(eef, arm):
        fl = eef[arm]["flange"]
        return np.asarray(fl["pos"], float), rotm(fl["quat"])

    def _propagate(self, eef):
        for arm, held in self.att.items():
            if not held or arm not in eef:
                continue
            p, R = self._flange(eef, arm)
            for k, (r, Ro) in held.items():
                o = self.objs[k]
                pos = p + R @ r
                d = pos - np.asarray(o["pos"], float)
                o["pos"] = pos.tolist()
                o["quat"] = quat_of(R @ Ro)
                if o.get("aabb"):
                    o["aabb"] = [(np.asarray(o["aabb"][0], float) + d).tolist(), (np.asarray(o["aabb"][1], float) + d).tolist()]

    def _real_obs(self):
        o = self._r.obs(cameras=[])
        if o.get("objects") is not None:
            raise RuntimeError("BeliefRobot is for tier B; the server returned privileged objects")
        if o.get("ok") and o.get("eef"):
            self._propagate(o["eef"])
        return o

    def obs(self, cameras=None, grasps=None, **kw):
        kw.pop("objects", None)
        o = self._r.obs(cameras=[] if cameras is None else cameras, **kw)
        if o.get("objects") is not None:
            raise RuntimeError("BeliefRobot is for tier B; the server returned privileged objects")
        if o.get("ok") and o.get("eef"):
            self._propagate(o["eef"])
        o = dict(o)
        o["objects"] = copy.deepcopy(self.objs)
        o["objects_source"] = "belief: located from pixels, carried by proprioception"
        return o

    def _attach(self, arm, eef):
        tcp = np.asarray(eef[arm]["pos"], float)
        held = {k for a in self.att.values() for k in a}
        cands = [(math.hypot(o["pos"][0] - tcp[0], o["pos"][1] - tcp[1]), k) for k, o in self.objs.items()
                 if k not in held and not o.get("fixed")]
        cands = [c for c in cands if c[0] <= self.attach_xy]
        if not cands:
            self.events.append({"attach": None, "arm": arm, "tcp": [round(float(v), 4) for v in tcp]})
            return
        k = min(cands)[1]
        p, R = self._flange(eef, arm)
        for name in [k] + list(self.contents.get(k, [])):
            o = self.objs[name]
            self.att[arm][name] = (R.T @ (np.asarray(o["pos"], float) - p), R.T @ rotm(o["quat"]))
        self.events.append({"attach": k, "arm": arm, "xy_off_mm": round(1000 * min(cands)[0], 1)})

    def _detach(self, arm):
        if self.att[arm]:
            self.events.append({"detach": sorted(self.att[arm]), "arm": arm,
                                "at": {k: [round(float(v), 4) for v in self.objs[k]["pos"]] for k in self.att[arm]}})
        self.att[arm] = {}

    # ---------------------------------------------------------------- wrapped commands
    def gripper(self, arm=None, action=None, **kw):
        if "cmd" in kw and action is None:
            action = kw.pop("cmd")
        if action in ("open", None) or (isinstance(action, (int, float)) and float(action) >= 0.5):
            self._real_obs()
            self._detach(arm)
            return self._r.gripper(arm=arm, action=action, **kw)
        r = self._r.gripper(arm=arm, action=action, **kw)
        if action == "close":
            held = bool(r.get("grasp_detected")) or min(r.get("finger_q") or [0.0]) > 0.003
            o = self._real_obs()
            if held and o.get("eef"):
                self._attach(arm, o["eef"])
        return r

    def return_home(self, arms=None, open=True, **kw):
        if open:
            self._real_obs()
            for a in (arms or ["left", "right"]):
                self._detach(a)
        return self._r.return_home(arms=arms, open=open, **kw)
