# Vendored unchanged from robot-harness (Mikhail's side) a21dd6f6f2db323bd0fe84f8afd20312cb8dffc6, skills/planar_drag_rgb_strict/belief_strict.py.
# Ported into controlr by stream P (2026-10-28) through dojo/skills/ported/harness_robot.py; logic not modified.
"""belief_strict.py: STRICT tier-B variant of planar_drag_rgb/belief.py (loaded read-only, unchanged).

BeliefRobotStrict = BeliefRobot with ONE change, in gripper(close): the attach decision no longer reads
grasp_detected / finger_q (the strict server has no finger measurement). Instead, right after the close it calls the
server's held_look (the server saves this step's wrist + head images; the PERCEIVER answers "object between the
fingers? yes/no"; 0 env steps) and attaches the nearest located object only on "yes". The answer per arm is kept in
`last_held` (None after an open) and every look in `looks`, for the strict skills that branch on it.
Open / return_home behave exactly as BeliefRobot (detach; no measurement involved).
"""
import importlib.util
import os

_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "planar_drag_rgb", "belief.py")
_spec = importlib.util.spec_from_file_location("rgb_belief_ro_for_strict", _p)
_B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_B)

rotm, quat_of, yaw_quat = _B.rotm, _B.quat_of, _B.yaw_quat


class BeliefRobotStrict(_B.BeliefRobot):
    def __init__(self, robot, objects, contents=None, attach_xy=0.03, tag="belief_strict"):
        super().__init__(robot, objects, contents=contents, attach_xy=attach_xy)
        self.last_held = {"left": None, "right": None}
        self.looks = []
        self.tag = tag

    def gripper(self, arm=None, action=None, **kw):
        if "cmd" in kw and action is None:
            action = kw.pop("cmd")
        if action in ("open", None) or (isinstance(action, (int, float)) and float(action) >= 0.5):
            self.last_held[arm] = None
            return super().gripper(arm=arm, action=action, **kw)
        r = self._r.gripper(arm=arm, action=action, **kw)
        if action == "close":
            lk = self._r.held_look(arm=arm, stage=f"{self.tag}:after_close")
            held = bool(lk.get("held"))
            self.last_held[arm] = held
            self.looks.append({k: lk.get(k) for k in ("ask", "arm", "answer", "note", "wait_s", "step")})
            o = self._real_obs()
            if held and o.get("eef"):
                self._attach(arm, o["eef"])
        return r

    def return_home(self, arms=None, open=True, **kw):
        if open:
            for a in (arms or ["left", "right"]):
                self.last_held[a] = None
        return super().return_home(arms=arms, open=open, **kw)


SCRUB = ("gap_m", "finger_q", "grasp_detected", "fingers_stopped", "blocked", "detected")


def scrub(x):
    """Drop the frozen tier-A code's own log fields for finger signals when they are null (they always are under the
    strict server, which never provides them). A non-null value is kept, so an audit would still see it."""
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items() if not (k in SCRUB and v is None)}
    if isinstance(x, list):
        return [scrub(v) for v in x]
    return x
