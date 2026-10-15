"""Prompt assembly for the FK agent: system text, few-shot sessions, this episode's history, and
the reply parser.

One request per step = ``[system, user]``. The user message is: (optional) few-shot example
sessions (text + their key-point photos), then the task and this episode's history — every
step's joints, motion, FK pose and result photo — and the reply instruction. Few-shot first and
byte-stable across steps, so the router's prompt cache covers it.

The reply is one JSON object (the last ``{...}`` block in the text):

    {"object": {"xyz_mm": [x, y, z], "uncertainty_mm": r, "visible": true, "seen": "..."},
     "action": "move" | "grip" | "give_up",
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

HOW AN EPISODE RUNS
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
* action "give_up": the object cannot be reached.
* You have at most {max_steps} steps: get close fast, then refine.

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
  (gripper_open_pct 0..60; ~35 = about 25 mm per jaw side, soft objects: wider).
* The table limits how low the TCP can go (z >= {box_lo[2]:.0f} mm).

REPLY: think it through briefly, then end with exactly one JSON object:
{{"object": {{"xyz_mm": [x, y, z], "uncertainty_mm": r, "visible": true|false, "seen": "where it is in which photo"}},
 "action": "move" | "grip" | "give_up",
 "q_deg": [pan, lift, elbow, wrist_flex, wrist_roll],
 "gripper_open_pct": 35,
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


def system_text(lim: G.Limits, *, max_steps: int, lift_dz: float = 50.0) -> str:
    jl = ", ".join(f"{n} {lo:.0f}..{hi:.0f}" for n, lo, hi in zip(G.ARM, lim.joint_lo_deg, lim.joint_hi_deg))
    return SYSTEM.format(approach_dz=lim.approach_dz_mm, lift_dz=lift_dz, max_steps=max_steps,
                         box_lo=lim.box_lo_mm, box_hi=lim.box_hi_mm, joint_limits=jl,
                         max_step_mm=lim.max_step_mm, max_joint_step_deg=lim.max_joint_step_deg)


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


def history_parts(task: str, steps: list[Step], *, long_edge: int = 640, jacobian: bool = True,
                  max_steps: int = 8) -> list:
    parts: list = [f"\nTASK: {task}\n\nTHIS EPISODE SO FAR:"]
    for st in steps:
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
            parts.append(image_part(st.photo, long_edge=long_edge, label=f"step{st.index}"))
    left = max_steps - sum(1 for s in steps if s.action not in ("observe",))
    parts.append(f"\nSteps left: {left}. Decide the next target. End with the JSON object.")
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
    if action not in ("move", "grip", "give_up"):
        raise ReplyError(f"unknown action {action!r} (move | grip | give_up)")
    q = _nums(obj.get("q_deg"), 5, "q_deg")
    if q is None and action != "give_up":
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
