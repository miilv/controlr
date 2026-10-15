"""Prompt assembly for the FK agent: system text, few-shot sessions, this episode's history, and
the reply parser.

One request per step = ``[system, user]``. The user message is: (optional) few-shot example
sessions (text + their key-point photos), then the task and this episode's history — every
step's joints, motion, FK pose and result photo — and the reply instruction. Few-shot first and
byte-stable across steps, so the router's prompt cache covers it.

The reply is one JSON object (the last ``{...}`` block in the text):

    {"object": {"xyz_mm": [x, y, z], "uncertainty_mm": r, "visible": true, "seen": "..."},
     "action": "move" | "grip" | "wait" | "give_up",
     "q_deg": [pan, lift, elbow, wrist_flex, wrist_roll],
     "gripper_open_pct": 35, "expected_tcp_mm": [x, y, z], "reason": "..."}
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from controlr.fkagent import geometry as G
from controlr.llm.transcript import ImagePart

SYSTEM = """\
You control a real SO-101 robot arm (5 joints + a parallel gripper) by choosing JOINT ANGLES.
Your only sensor is a camera mounted on the gripper (wrist camera) plus the joint encoders.

GEOMETRY
* Base frame, millimetres: origin on the table under the arm's base, +x forward (away from the
  robot), +y to the robot's left, +z up. The table surface is z = 0.
* Joints, degrees (kinematic model frame): q = [shoulder_pan, shoulder_lift, elbow_flex,
  wrist_flex, wrist_roll].
  - shoulder_pan turns the whole arm about the vertical axis: + turns it to the robot's RIGHT
    (-y), - to the LEFT (+y). Pan 0 = straight ahead.
  - shoulder_lift, elbow_flex and wrist_flex move the gripper in the vertical plane of the pan
    angle (reach and height) and set the tool pitch (how steeply the gripper points down).
    Their effect depends on the pose: every observation lists the table "d TCP per +1 deg" for
    that pose (mm the gripper tip moves for +1 deg of each joint) - use it to correct a pose.
  - wrist_roll turns the gripper about its own axis; about -90 keeps the camera image upright
    (image up = away from the robot). Keep it near -90 unless you have a reason.
* TCP = the point between the two jaw TIPS. Every pose is shown with its TCP position computed by
  forward kinematics (FK) from the measured joints (accurate to a few mm) and the tool pitch
  (degrees below horizontal; 90 = pointing straight down).
* The camera rides on the gripper, ~10-14 cm behind and above the jaw tips, looking along the
  jaws. The green jaws are visible at the bottom of every photo; an object that is GRASPABLE is
  between the jaw tips. Objects higher in the image are farther along the gripper's pointing
  direction. The camera mount is not calibrated (its exact angle can differ between sessions):
  estimate distances from the object's apparent size and position relative to the jaws.

{episode}
SAFETY LIMITS (enforced in code; a target that breaks them is pulled back toward your previous
target, or refused, and you are told)
* TCP workspace: x {box_lo[0]:.0f}..{box_hi[0]:.0f}, y {box_lo[1]:.0f}..{box_hi[1]:.0f}, z {box_lo[2]:.0f}..{box_hi[2]:.0f} mm - at the target and all along the path.
* Joint limits (deg): {joint_limits}.
* From your previous target to the next one: at most {max_step_mm:.0f} mm of TCP motion and
  {max_joint_step_deg:.0f} deg per joint (the first target is free within the workspace).

GRASPING HINTS
* Objects lie on the table. A good grasp has the jaw tips straddling the object's middle at
  about its mid-height or a bit lower (TCP z typically 2-30 mm for objects up to ~8 cm tall), the
  tool pointing down steeply (pitch 65-90 deg), the jaws open wider than the object
  (gripper_open_pct {gripper_min:.0f}..{gripper_max:.0f}, never wider: a wider jaw hits the camera).
* The table limits how low the TCP can go (z >= {box_lo[2]:.0f} mm).

REPLY: the JSON object FIRST (no text before it, nothing after it); keep "reason" under 30 words:
{{"object": {{"xyz_mm": [x, y, z], "uncertainty_mm": r, "visible": true|false, "seen": "where it is in which photo"}},
 "action": {actions},
 "q_deg": [pan, lift, elbow, wrist_flex, wrist_roll],
 "gripper_open_pct": 30,
 "expected_tcp_mm": [x, y, z],
 "grasp_q_deg": [pan, lift, elbow, wrist_flex, wrist_roll],
 "reason": "one or two sentences"}}
"object.xyz_mm" = your best estimate of the object's centre in the base frame (mm);
"uncertainty_mm" = the radius you believe it lies within; "expected_tcp_mm" = where you expect
the TCP to end up with your q_deg; "grasp_q_deg" = the joints you would close the jaws at if
you had to grip NOW with your current estimate (equal to q_deg when action is "grip").
"""

FEWSHOT_INTRO = """\
EXAMPLES: sessions recorded while a human teleoperated this arm and picked objects up. Same
joint frame, same FK, same camera type (the mount may be at a slightly different angle). Time
t starts when the arm left its start pose; key points: approach / pre_grasp (on the way), miss
(the jaws closed empty), grasp (the jaws stalled on the object), lift. "motion" = joint change
since the previous key point. The TCP at "grasp" is where the object was (its centre is within
a few cm of it). Note how the object looks in the photos at each distance, and the joint
angles that put the jaw tips around it.
"""


EP_RETURN = """HOW AN EPISODE RUNS
* Every step you give ONE joint target. The arm moves from its START POSE to a point
  {approach_dz:.0f} mm above your target and then straight down onto it, the photo is taken there, and
  the arm goes back to the start pose (with the jaws open as you set them). The next step again
  starts from the start pose. You see every previous step's target, the joints actually reached,
  the FK pose and the photo taken at the target.
* action "grip": the same motion, then the jaws close at the target. If something is between
  them they stall on it (grasp): the arm lifts it {lift_dz:.0f} mm, and the episode ends with success.
  If they close empty, the attempt is reported and you continue. Grip only when the object is
  between the jaw tips in the photo of the previous step at (nearly) the same target, or when
  you are confident the target puts the jaw tips around the object.
* action "move": look again from a new pose (to locate the object or to check the alignment).
* action "wait": do not move at all; a new photo is taken at the start pose (q_deg is ignored).
* action "give_up": the object cannot be reached.
* PEOPLE: if a person, a hand or an arm is visible in ANY current photo, do NOT move and do NOT
  close the gripper - answer "wait" until no person is visible.
* You have at most {max_steps} steps: get close fast, then refine."""

EP_CONT = """HOW AN EPISODE RUNS (continuous)
* The arm starts at its START POSE. Every step you give ONE joint target and the arm goes there
  FROM WHERE IT IS NOW: first straight up {approach_dz:.0f} mm (when possible), then to a point
  {approach_dz:.0f} mm above your target, then straight down onto it. It stays there; the photo is
  taken there; you plan the next step from that pose. It returns to the start pose only when
  the whole task is done.
* Aim to FINISH the task in each move: when you can see the object, give the grasp pose itself
  (action "grip"), not a small step toward it. Use "move" only when you need a better view.
* action "grip": go to q_deg, then the jaws close there. If something is between them they stall
  on it (grasp) and the arm lifts it {lift_dz:.0f} mm straight up and keeps holding it. If they
  close empty, they reopen and the arm stays there. A "grip" whose q_deg equals the current pose
  closes the jaws in place.
* action "place" (only while holding): carry the object to q_deg (the jaw tips at the place
  spot, about the height at which it was grasped), open the jaws there; the arm lifts and goes
  back to the start pose: the task is done.
* action "move": go to q_deg and look (holding or not).
* action "wait": do not move at all; a new photo is taken where the arm is (q_deg is ignored).
* action "give_up": the task cannot be done.
* Do NOT revisit a point you already tried: a target within {revisit_mm:.0f} mm of an earlier target is
  refused. After an empty close, change the target according to what the photos show.
* PEOPLE: if a person, a hand or an arm is visible in ANY current photo, do NOT move and do NOT
  close the gripper - answer "wait" until no person is visible.
* You have at most {max_steps} steps."""


EP_PLAN = """HOW AN EPISODE RUNS (plan once)
* You plan the WHOLE task ONCE, up front, from the photos you have: the joint targets of every
  stage. The code then runs the stages without asking you again:
  1. grasp: from the start pose go to grasp_q_deg (straight down onto it from {approach_dz:.0f} mm above,
     jaws open to gripper_open_pct), close the jaws. CHECK: the jaws stall on the object.
  2. lift: straight up {lift_dz:.0f} mm. CHECK: the jaws still hold it.
  3. carry: go to place_q_deg (up {approach_dz:.0f} mm first, then above the place pose, then down).
     CHECK: the arm reached place_q_deg.
  4. release: open the jaws to release_open_pct. CHECK: the jaws opened. Then lift {lift_dz:.0f} mm.
  5. return to the start pose: done.
* A move longer than the per-step limits is split into several moves toward the same target.
* If a CHECK fails, you get everything that happened (joints, FK, photos of every stage) and
  re-plan the REMAINING stages from the current state (at most {max_replans} re-plans). While
  holding the object you only give the place pose.
* PEOPLE: if a person, a hand or an arm is visible in ANY photo, set "action": "wait"
  (nothing moves; you are asked again with a new photo).
* You have at most {max_steps} motions in total."""

REPLY_PLAN = """REPLY: the JSON object FIRST (no text before it, nothing after it); keep "reason" under 30 words:
{"object": {"xyz_mm": [x, y, z], "uncertainty_mm": r, "visible": true|false, "seen": "where it is in which photo"},
 "action": "plan" | "wait" | "give_up",
 "grasp_q_deg": [pan, lift, elbow, wrist_flex, wrist_roll],
 "gripper_open_pct": 30,
 "expected_grasp_tcp_mm": [x, y, z],
 "place_q_deg": [pan, lift, elbow, wrist_flex, wrist_roll],
 "expected_place_tcp_mm": [x, y, z],
 "release_open_pct": 35,
 "reason": "one or two sentences"}
"grasp_q_deg" may be null when the object is already held. All poses must respect the safety
limits; the carry and place poses also the carry height limit if one is stated in the task.
"""


def system_text(lim: G.Limits, *, max_steps: int, lift_dz: float = 50.0, gripper_min: float = 10.0,
                gripper_max: float = 35.0, loop: str = "return", revisit_mm: float = 10.0,
                max_replans: int = 3) -> str:
    jl = ", ".join(f"{n} {lo:.0f}..{hi:.0f}" for n, lo, hi in zip(G.ARM, lim.joint_lo_deg, lim.joint_hi_deg))
    ep = {"continuous": EP_CONT, "plan_once": EP_PLAN}.get(loop, EP_RETURN).format(
        approach_dz=lim.approach_dz_mm, lift_dz=lift_dz, max_steps=max_steps, revisit_mm=revisit_mm,
        max_replans=max_replans)
    actions = ('"move" | "grip" | "place" | "wait" | "give_up"' if loop == "continuous"
               else '"move" | "grip" | "wait" | "give_up"')
    text = SYSTEM.format(episode=ep, actions=actions, gripper_min=gripper_min, gripper_max=gripper_max, approach_dz=lim.approach_dz_mm, lift_dz=lift_dz, max_steps=max_steps,
                         box_lo=lim.box_lo_mm, box_hi=lim.box_hi_mm, joint_limits=jl,
                         max_step_mm=lim.max_step_mm, max_joint_step_deg=lim.max_joint_step_deg)
    if loop == "plan_once":
        text = text[:text.index("REPLY:")] + REPLY_PLAN
    return text


def _v(a, d: int = 1) -> str:
    return "[" + ", ".join(f"{float(x):.{d}f}" for x in a) + "]"


def jac_text(q_deg) -> str:
    J = G.jacobian_mm_per_deg(q_deg)
    return "d TCP per +1 deg (dx, dy, dz mm): " + "; ".join(
        f"{n} ({J[0, i]:+.1f}, {J[1, i]:+.1f}, {J[2, i]:+.1f})" for i, n in enumerate(G.ARM[:4]))


def pose_text(q_deg, gripper=None, *, jacobian: bool = True) -> str:
    s = G.pose_summary(q_deg)
    out = f"q_deg {_v(q_deg)}; TCP (FK) {_v(s['tcp_mm'], 0)} mm, tool pitch {s['tool_pitch_deg']:.0f} deg"
    if gripper is not None:
        out += f"; gripper {float(gripper):.0f} %"
    if jacobian:
        out += "\n  " + jac_text(q_deg)
    return out


def image_part(path_or_bytes, *, long_edge: int = 640, label: str = "") -> ImagePart:
    """JPEG bytes (or a file) -> ImagePart, downscaled to ``long_edge``; deterministic."""
    from PIL import Image

    data = Path(path_or_bytes).read_bytes() if not isinstance(path_or_bytes, (bytes, bytearray)) else bytes(path_or_bytes)
    with Image.open(io.BytesIO(data)) as im:
        if max(im.size) <= long_edge and im.format == "JPEG":
            return ImagePart.from_jpeg(data, label)
        im = im.convert("RGB")
        sc = long_edge / max(im.size)
        im = im.resize((max(1, round(im.size[0] * sc)), max(1, round(im.size[1] * sc))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    return ImagePart.from_jpeg(buf.getvalue(), label)


# ------------------------------------------------------------------------------- few-shot

def fewshot_parts(sessions: list[dict], root, *, long_edge: int = 640, max_photos: int = 5) -> list:
    """Text + photos of the example sessions (key points in time order)."""
    root = Path(root)
    parts: list = [FEWSHOT_INTRO]
    for n, s in enumerate(sessions, 1):
        g = s["grasp"]
        head = (f"\nEXAMPLE {n}: pick up the {s['object']} (grasp after {g['t']:.1f} s; jaws opened to "
                f"{g['gripper_open']:.0f} %, stalled at {g['gripper_stall']:.0f} %"
                + (f"; {len(s['misses'])} empty close(s) before" if s.get("misses") else "") + ")")
        parts.append(head)
        kps = s["keypoints"]
        with_photo = [k for k in kps if k.get("photo")]
        keep = {id(k) for k in with_photo[-max_photos:]}
        for k in kps:
            line = f"- {k['name']} t={k['t']:.2f} s: " + pose_text(k["q_deg"], k.get("gripper"), jacobian=False)
            if k.get("motion_deg") is not None:
                line += f"; motion {_v(k['motion_deg'])}"
            parts.append(line)
            if id(k) in keep:
                parts.append(image_part(root / k["photo"], long_edge=long_edge, label=f"ex{n}:{k['name']}"))
        parts.append(f"=> grasped the {s['object']} with q_deg {_v(g['q_deg'])}, TCP {_v(g['tcp_mm'], 0)} mm, "
                     f"pitch {g['tool_pitch_deg']:.0f} deg.")
    return parts


# ------------------------------------------------------------------------------- history

@dataclass
class Step:
    """One step of an episode (step 0 = the observation at the start pose)."""
    index: int
    action: str                          # observe | move | grip | rejected
    q_deg: list                          # joints reached at the target (measured, model frame)
    gripper: float | None = None
    photo: bytes | None = None           # JPEG at the target
    requested_q_deg: list | None = None
    executed_q_deg: list | None = None
    motion_deg: list | None = None       # target - start pose
    notes: list = field(default_factory=list)
    result: str = ""                     # e.g. "the jaws closed empty", "grasped: ..."
    estimate: dict | None = None         # the agent's object estimate at this step
    where: str = "the start pose"        # observe steps: where the photo was taken
    holding: bool | None = None          # continuous loop: holding after this step


def history_parts(task: str, steps: list[Step], *, long_edge: int = 640, jacobian: bool = True,
                  max_steps: int = 8, final: str | None = None, keep: int = 25, full_photos: int = 5,
                  old_edge: int = 320) -> list:
    """The last ``keep`` steps (joints, motion, results, photos); photos of steps older than the
    last ``full_photos`` are downscaled to ``old_edge`` px; steps before the last ``keep`` are
    summarised as text only (step 0 is always kept)."""
    parts: list = [f"\nTASK: {task}\n\nTHIS EPISODE SO FAR:"]
    shown = steps if len(steps) <= keep else [steps[0]] + steps[-(keep - 1):]
    if len(shown) < len(steps):
        parts.append(f"({len(steps) - len(shown)} older steps omitted)")
    n_shown = len(shown)
    for idx, st in enumerate(shown):
        edge = long_edge if idx >= n_shown - full_photos or idx == 0 else old_edge
        if st.action == "observe":
            parts.append(f"Step {st.index} - observation at {st.where}: "
                         + pose_text(st.q_deg, st.gripper, jacobian=jacobian))
        else:
            head = f"Step {st.index} - {st.action}"
            if st.requested_q_deg is not None:
                head += f": you asked q_deg {_v(st.requested_q_deg)}"
            if st.executed_q_deg is not None and st.requested_q_deg is not None and \
                    np.max(np.abs(np.subtract(st.executed_q_deg, st.requested_q_deg))) > 0.05:
                head += f", executed {_v(st.executed_q_deg)}"
            if st.motion_deg is not None:
                head += f" (motion from the start pose {_v(st.motion_deg)})"
            parts.append(head)
            for n in st.notes:
                parts.append(f"  NOTE: {n}")
            if st.q_deg is not None:
                parts.append("  reached: " + pose_text(st.q_deg, st.gripper, jacobian=jacobian))
            if st.result:
                parts.append(f"  result: {st.result}")
        if st.photo is not None:
            parts.append(f"  photo of step {st.index}:")
            parts.append(image_part(st.photo, long_edge=edge, label=f"step{st.index}"))
    left = max_steps - sum(1 for s in steps if s.action not in ("observe",))
    if steps and steps[-1].holding is not None:
        parts.append("\nNOW: the arm is at the pose of step %d, %s." % (
            steps[-1].index, "HOLDING the object" if steps[-1].holding else "the jaws are empty (open)"))
    parts.append(final if final is not None else
                 f"\nSteps left: {left}. Decide the next target. End with the JSON object.")
    return parts


def build_messages(system: str, user_parts: list) -> list[dict]:
    """OpenAI chat format; adjacent text parts are joined (fewer parts, same bytes every step)."""
    content: list[dict] = []
    buf: list[str] = []
    for p in user_parts:
        if isinstance(p, str):
            buf.append(p)
            continue
        if buf:
            content.append({"type": "text", "text": "\n".join(buf)})
            buf = []
        content.append({"type": "image_url", "image_url": {"url": p.data_url}})
    if buf:
        content.append({"type": "text", "text": "\n".join(buf)})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def count_images(messages: list[dict]) -> int:
    return sum(1 for m in messages if isinstance(m["content"], list)
               for p in m["content"] if p.get("type") == "image_url")


# ------------------------------------------------------------------------------- reply

class ReplyError(ValueError):
    pass


@dataclass
class Decision:
    action: str
    q_deg: list | None
    gripper_open_pct: float | None
    object_xyz_mm: list | None
    uncertainty_mm: float | None
    visible: bool | None
    expected_tcp_mm: list | None
    reason: str
    raw: dict
    grasp_q_deg: list | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "raw"}


def _json_objects(text: str) -> list[str]:
    """Balanced top-level {...} blocks (strings respected), in order."""
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, c in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"' and depth > 0:
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                out.append(text[start:i + 1])
    return out


def _nums(v, n: int, what: str) -> list | None:
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or len(v) < n:
        raise ReplyError(f"{what} must be a list of {n} numbers, got {v!r}")
    try:
        out = [float(x) for x in list(v)[:n]]
    except (TypeError, ValueError) as e:
        raise ReplyError(f"{what}: {e}") from e
    if not all(np.isfinite(out)):
        raise ReplyError(f"{what} has a non-finite value")
    return out


def parse_reply(text: str) -> Decision:
    blocks = _json_objects(re.sub(r"```(?:json)?", "", text or ""))
    obj = None
    for b in reversed(blocks):
        try:
            cand = json.loads(b)
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict) and ("action" in cand or "q_deg" in cand):
            obj = cand
            break
    if obj is None:
        raise ReplyError("no JSON object with an 'action' in the reply")
    action = str(obj.get("action", "move")).strip().lower()
    if action not in ("move", "grip", "place", "wait", "give_up"):
        raise ReplyError(f"unknown action {action!r} (move | grip | place | wait | give_up)")
    q = _nums(obj.get("q_deg"), 5, "q_deg")
    if q is None and action not in ("give_up", "wait"):
        raise ReplyError("q_deg is missing")
    try:
        grasp = _nums(obj.get("grasp_q_deg"), 5, "grasp_q_deg")
    except ReplyError:
        grasp = None                     # optional: a bad grasp_q_deg does not void the step
    if grasp is None and action == "grip":
        grasp = q
    o = obj.get("object") or {}
    if not isinstance(o, dict):
        o = {}
    unc = o.get("uncertainty_mm")
    try:
        unc = None if unc is None else float(unc)
    except (TypeError, ValueError):
        unc = None
    g = obj.get("gripper_open_pct")
    try:
        g = None if g is None else float(g)
    except (TypeError, ValueError):
        g = None
    vis = o.get("visible")
    return Decision(action, q, g, _nums(o.get("xyz_mm"), 3, "object.xyz_mm"), unc,
                    None if vis is None else bool(vis), _nums(obj.get("expected_tcp_mm"), 3, "expected_tcp_mm"),
                    str(obj.get("reason", ""))[:500], obj, grasp)


@dataclass
class StagePlan:
    action: str                         # plan | wait | give_up
    grasp_q_deg: list | None
    gripper_open_pct: float | None
    place_q_deg: list | None
    release_open_pct: float | None
    object_xyz_mm: list | None
    expected_grasp_tcp_mm: list | None
    expected_place_tcp_mm: list | None
    reason: str
    raw: dict

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "raw"}


def _num(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def parse_plan(text: str, *, need_grasp: bool = True) -> StagePlan:
    blocks = _json_objects(re.sub(r"```(?:json)?", "", text or ""))
    obj = None
    for b in reversed(blocks):
        try:
            cand = json.loads(b)
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict) and ("action" in cand or "place_q_deg" in cand):
            obj = cand
            break
    if obj is None:
        raise ReplyError("no JSON object with an 'action' in the reply")
    action = str(obj.get("action", "plan")).strip().lower()
    if action not in ("plan", "wait", "give_up"):
        raise ReplyError(f"unknown action {action!r} (plan | wait | give_up)")
    grasp = _nums(obj.get("grasp_q_deg"), 5, "grasp_q_deg")
    place = _nums(obj.get("place_q_deg"), 5, "place_q_deg")
    if action == "plan":
        if place is None:
            raise ReplyError("place_q_deg is missing")
        if need_grasp and grasp is None:
            raise ReplyError("grasp_q_deg is missing (the object is not held)")
    o = obj.get("object") if isinstance(obj.get("object"), dict) else {}
    return StagePlan(action, grasp, _num(obj.get("gripper_open_pct")), place, _num(obj.get("release_open_pct")),
                     _nums(o.get("xyz_mm"), 3, "object.xyz_mm"),
                     _nums(obj.get("expected_grasp_tcp_mm"), 3, "expected_grasp_tcp_mm"),
                     _nums(obj.get("expected_place_tcp_mm"), 3, "expected_place_tcp_mm"),
                     str(obj.get("reason", ""))[:500], obj)


def json_complete(text: str) -> bool:
    """stop_when for streaming: True once the reply holds a complete JSON object with an action
    (the stream is closed there: nothing after it matters, and the arm can start moving)."""
    if "}" not in text:
        return False
    for b in _json_objects(re.sub(r"```(?:json)?", "", text)):
        try:
            o = json.loads(b)
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and "action" in o:
            return True
    return False


# ------------------------------------------------------------------------------- pixel agent

PIXEL_SYSTEM = """\
You are the eyes of a robot arm with a camera on its gripper (wrist camera). You never give
joint angles or coordinates: you point at things in the CURRENT photo and judge the task from
the photos. Code computes every motion from your pixels (camera pose from the joint encoders).

Each photo is {w}x{h} px, (0, 0) = top-left. The two green jaws are at the bottom of the photo;
an object that is held is between them.

Each step you get the task, the history in order (code's facts: phase, gripper, what the code
did and its 3D estimate from your previous pixels - for orientation only), and the newest photo.
Reply with ONLY this JSON (no other text):
{{"object_px": [u, v] or null, "target_px": [u, v] or null, "held": true|false, "done": true|false,
 "person": true|false, "note": "at most 12 words"}}
* object_px: the centre of the object to pick up, in the NEWEST photo (null if not visible or
  if it is held).
* target_px: the centre of the place target's top (where the object must end up), in the newest
  photo (null if not visible).
* held: the object is visibly held between the jaws in the newest photo.
* done: the photos show the task is complete (e.g. the object rests on the target and the jaws
  are open and away from it).
* person: a person, hand or arm is visible (then nothing moves).
"""


@dataclass
class PixelReply:
    object_px: list | None
    target_px: list | None
    held: bool
    done: bool
    person: bool
    note: str
    raw: dict

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "raw"}


def _px(v, w: int, h: int, what: str) -> list | None:
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        raise ReplyError(f"{what} must be [u, v] or null")
    try:
        u, vv = float(v[0]), float(v[1])
    except (TypeError, ValueError) as e:
        raise ReplyError(f"{what}: {e}") from e
    if not (0 <= u <= w and 0 <= vv <= h):
        raise ReplyError(f"{what} {v} is outside the {w}x{h} photo")
    return [u, vv]


def parse_pixel_reply(text: str, w: int, h: int) -> PixelReply:
    blocks = _json_objects(re.sub(r"```(?:json)?", "", text or ""))
    obj = None
    for b in reversed(blocks):
        try:
            cand = json.loads(b)
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict) and ("object_px" in cand or "done" in cand):
            obj = cand
            break
    if obj is None:
        raise ReplyError("no JSON object with object_px / done in the reply")
    return PixelReply(_px(obj.get("object_px"), w, h, "object_px"), _px(obj.get("target_px"), w, h, "target_px"),
                      bool(obj.get("held", False)), bool(obj.get("done", False)), bool(obj.get("person", False)),
                      str(obj.get("note", ""))[:200], obj)


def pixel_json_complete(text: str) -> bool:
    if "}" not in text:
        return False
    for b in _json_objects(re.sub(r"```(?:json)?", "", text)):
        try:
            o = json.loads(b)
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and ("object_px" in o or "done" in o):
            return True
    return False


def pixel_history_parts(task: str, steps: list[Step], *, w: int, h: int, keep: int = 25, full_photos: int = 3,
                        old_edge: int = 320) -> list:
    parts: list = [f"TASK: {task}\n\nHISTORY (oldest first):"]
    shown = steps if len(steps) <= keep else [steps[0]] + steps[-(keep - 1):]
    if len(shown) < len(steps):
        parts.append(f"({len(steps) - len(shown)} older steps omitted)")
    n_shown = len(shown)
    for idx, st in enumerate(shown):
        s = G.pose_summary(st.q_deg) if st.q_deg is not None else None
        line = f"Step {st.index} [{st.action}]"
        if s:
            line += f": joints {_v(st.q_deg)}, TCP {_v(s['tcp_mm'], 0)} mm"
        if st.gripper is not None:
            line += f", gripper {st.gripper:.0f} %"
        parts.append(line)
        for n in st.notes:
            parts.append(f"  code: {n}")
        if st.result:
            parts.append(f"  result: {st.result}")
        if st.estimate and st.estimate.get("judgement"):
            parts.append(f"  your judgement: {st.estimate['judgement']}")
        if st.photo is not None:
            newest = idx == n_shown - 1
            edge = max(w, h) if (idx >= n_shown - full_photos) else old_edge
            parts.append(f"  photo of step {st.index}" + (" (NEWEST)" if newest else "") + ":")
            parts.append(image_part(st.photo, long_edge=edge, label=f"step{st.index}"))
    parts.append(f"\nPoint at the object and the target in the NEWEST photo ({w}x{h} px). JSON only.")
    return parts
