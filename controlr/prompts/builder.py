"""System prompt (robot operating manual) and planner prompt assembly.

The system prompt is the robot equivalent of a coding agent's system prompt:
what the robot is, how the loop works, the exact action grammar, what feedback
means, safety and technique. It is rendered once per episode from the config,
the RobotSpec and (optionally) the camera calibration, then cached by the
provider for the whole episode — so it must be byte-stable (no timestamps, no
dict-order or float noise) and long enough to be cacheable.

Templates are plain markdown with ``{placeholder}`` fields filled by a single
regex pass (no jinja): substituted text is never re-scanned, unknown
placeholders are an error, and literal braces elsewhere are left alone.

Cache minimums (Anthropic): 512 tokens (Opus 5.x / Fable), 1024 (Sonnet 5),
4096 (Haiku 4.5). The manual is ~2.5-3.5k tokens with the generated appendix
(worked example exchange + conventions), which caches on Opus/Sonnet. For
Haiku, add *useful* reference text — task-specific rules via
``prompt.extra_rules`` or a recorded exchange via ``prompt.fewshot`` (a .md file
or a run directory, appended as "Appendix C") — never filler; ``cache_warning``
tells the loop when the prefix is too short.

Ideas borrowed (paraphrased, not copied) from research/sources/:
  * inspect-robots (Robocurve): "small, deliberate motions; re-check the
    observation after every motion"; safety approvers below the model, which
    the model is told about.
  * GPT-Policy (cheng-haha): stop to observe at contact / gripper changes; a
    fully-open command does not prove release; one failure is not impossibility.
  * innate-os PR #817: centring and descending are separate moves; change the
    approach after two identical failures.
  * RoboProbe / RoboDojo: arrival feedback, "travel first, rotate later",
    few-mm corrections near contact.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
import numpy as np

from controlr.config import Config
from controlr.protocol.feedback import format_feedback, format_state
from controlr.protocol.grammar import (
    EE_DELTA_STEPS,
    EE_MODES,
    JOINT_MODES,
    ang_decimals,
    ang_factor,
    example_xyz,
    fmt_num,
    grammar_spec,
    parse_reply,
    pos_decimals,
    pos_factor,
)
from controlr.types import (
    Action,
    ActionMode,
    CameraInfo,
    ExecReport,
    Observation,
    RobotSpec,
    RobotState,
    SafetyEvent,
    EventLevel,
)

PROMPT_DIR = Path(__file__).resolve().parent
_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")

# minimum cacheable prefix (tokens) per model family; see module docstring
_CACHE_MIN = (("haiku", 4096), ("sonnet", 1024), ("opus", 512), ("fable", 512))


# ---------------------------------------------------------------------------
# template plumbing
# ---------------------------------------------------------------------------

def load_template(name: str) -> str:
    """``system_v0`` -> controlr/prompts/system_v0.md; a path to an existing
    .md file is also accepted (for prompt experiments outside the package)."""
    p = Path(name)
    if p.suffix == ".md" and p.exists():
        return p.read_text()
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt template {name!r} not found in {PROMPT_DIR}")
    return path.read_text()


def fill(template: str, values: dict[str, str]) -> str:
    """Single-pass ``{key}`` substitution; a placeholder without a value raises."""
    missing = sorted({m.group(1) for m in _PLACEHOLDER.finditer(template)} - set(values))
    if missing:
        raise KeyError(f"template placeholders without values: {missing}")
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~3.6 chars/token for this kind of English+numbers)."""
    return int(math.ceil(len(text) / 3.6))


def cache_warning(model: str, system_text: str) -> str | None:
    """A warning string if the system prompt is likely below the model's minimum
    cacheable prefix (then every turn pays full price for it), else None."""
    m = model.lower()
    for key, n in _CACHE_MIN:
        if key in m:
            est = estimate_tokens(system_text)
            if est < n:
                return (f"system prompt ~{est} tokens < {n}-token cache minimum for {model}; "
                        f"add reference material via prompt.extra_rules / prompt.fewshot")
            return None
    return None


# ---------------------------------------------------------------------------
# value rendering
# ---------------------------------------------------------------------------

def _len(cfg: Config, v_m: float, extra_dec: int = 0) -> str:
    a = cfg.action
    return f"{fmt_num(v_m / pos_factor(a), pos_decimals(a) + extra_dec)} {a.pos_unit}"


def _num(cfg: Config, v_m: float) -> str:
    return fmt_num(v_m / pos_factor(cfg.action), pos_decimals(cfg.action))


def _angle(cfg: Config, v_rad: float) -> str:
    a = cfg.action
    return f"{fmt_num(v_rad / ang_factor(a), ang_decimals(a))} {a.ang_unit}"


def _joint_table(cfg: Config, spec: RobotSpec) -> str:
    u = cfg.action.ang_unit
    rows = [f"| # | joint | lower ({u}) | upper ({u}) | home ({u}) |", "|---|---|---|---|---|"]
    f = ang_factor(cfg.action)
    d = ang_decimals(cfg.action)
    for i, j in enumerate(spec.joints):
        home = spec.home_q[i] if i < len(spec.home_q) else float("nan")
        rows.append(f"| {i + 1} | {j.name} | {fmt_num(j.lower / f, d)} | {fmt_num(j.upper / f, d)} "
                    f"| {fmt_num(home / f, d)} |")
    return "\n".join(rows)


def _gripper_doc(cfg: Config, spec: RobotSpec) -> str:
    s = (f"parallel-jaw gripper; finger opening from 0 (closed) to "
         f"{_len(cfg, spec.gripper_max_mm * 1e-3)} (fully open). ")
    if cfg.action.gripper == "binary":
        s += "You command it open or closed; closing stops when the fingers meet an object."
    else:
        s += "You command a target opening width; it stops early if the fingers meet an object."
    return s


def _camera_axes(cfg: Config, spec: RobotSpec, cameras: dict[str, CameraInfo] | None) -> str:
    """Per-camera axis directions computed from the calibration (if given)."""
    if not cameras:
        return ""
    from controlr.observation.renderers import describe_axes   # numpy/PIL only
    lo, hi = np.asarray(spec.workspace_lo, float), np.asarray(spec.workspace_hi, float)
    from controlr.observation.renderers import resolve_grid_z
    centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, resolve_grid_z(cfg.observation, spec)])
    from controlr.observation.renderers import axis_directions
    names = list(cfg.observation.cameras) or sorted(cameras)
    lines = ["How the base axes appear in the camera images (at the middle of the workspace; pixel "
             "lengths of a move of that size at that spot):"]
    for n in names:
        if n in cameras:
            lines.append(f"- `{n}`: {describe_axes(cameras[n], centre, cfg.observation.size, _len(cfg, 0.1))}.")
            dirs = axis_directions(cameras[n], centre)
            for a, b in (("x", "y"), ("x", "z"), ("y", "z")):
                if dirs[a] is not None and dirs[a] == dirs[b]:
                    cue = "the STATE line" if cfg.observation.state_text else "the EXEC line"
                    if "ee_marker" in cfg.observation.renderers:
                        cue += " or the TCP drop line"
                    lines.append(f"  In `{n}` a pure +{a} move and a pure +{b} move both shift the gripper "
                                 f"{dirs[a]}; only the pixel lengths differ. Tell them apart with {cue}, "
                                 f"never from the image direction alone.")
    return "\n".join(lines) if len(lines) > 1 else ""


def _observation_doc(cfg: Config, spec: RobotSpec | None = None) -> str:
    from controlr.observation.renderers import resolve_grid_z
    o = cfg.observation
    cams = ", ".join(f"`{c}`" for c in o.cameras) or "all configured cameras"
    lines = [f"- Camera image(s): {cams}, resized to {o.size} px on the long edge"
             + (f" ({o.first_turn_size} px on the first turn)" if o.first_turn_size else "")
             + ", taken after the robot came to rest."]
    ov = [r for r in o.renderers if r in ("grid", "axes", "ee_marker")]
    if "grid" in ov:
        lines.append(f"- Grid overlay: base-frame lines every {_len(cfg, o.grid_step)} drawn on the "
                     f"plane z={_len(cfg, resolve_grid_z(o, spec))} (the table), labelled with their x= / y= value.")
    if "axes" in ov:
        lines.append(f"- Axes overlay: arrows from the TCP along base +x (red), +y (green) and "
                     f"+z (blue), each {_len(cfg, 0.05)} long.")
    if "ee_marker" in ov:
        lines.append("- TCP marker: magenta circle + cross at the projected TCP with its height, "
                     "yellow squares at the two fingertips (current opening), and a magenta drop "
                     "line from the TCP straight down to the table (its foot marks the TCP's x, y).")
    if not ov:
        lines.append("- No overlays: the images are the raw camera view.")
    if "diff" in o.renderers:
        lines.append("- Diff image (from the second turn on): grey image per camera, bright where "
                     "pixels changed since the previous image.")
    if "heatmap" in o.renderers:
        lines.append("- Heatmap image (from the second turn on): the current frame with hot colours "
                     "where it changed since the previous image.")
    if o.tile or "tile" in o.renderers:
        lines.append("- All images of a turn are tiled into one picture with a label above each panel.")
    lines.append("- A line `IMAGES: ...` names the images in order.")
    if o.state_text:
        lines.append("- The feedback block ends with a STATE line (measured, see below).")
    else:
        lines.append("- The feedback has no line with the measured robot state: judge it from the "
                     "images. The EXEC line still reports the measured motion of each command"
                     + (" and the TCP marker shows its height" if "ee_marker" in ov else "") + ".")
    return "\n".join(lines)


def _state_values(cfg: Config, spec: RobotSpec) -> dict[str, str]:
    """Manual text that depends on whether a STATE line is given (observation.state_text):
    with it off, the manual must not mention, show or rely on a STATE line."""
    if cfg.observation.state_text:
        st = _example_state(cfg, spec)
        doc = ("The STATE line reports the measured robot state after the last motion, e.g.:\n\n"
               f"    {format_state(st, cfg)}\n\n"
               "`grip` is the measured finger opening and whether the last gripper command was open or\n"
               "closed; `holding` is the robot's own grasp detection (`unknown` if it cannot tell).\n"
               "A closed gripper whose opening is near 0 is holding nothing; a closed gripper stopped\n"
               "at roughly an object's width is probably holding it — confirm in the image.")
        return {"state_item": ", the robot STATE", "state_check": "the STATE line",
                "overlay_state": "the overlays and the STATE line", "state_doc": doc,
                "state_line": "    STATE: <measured state>\n",
                "grasp_check": "STATE grip width, `holding`, the image",
                "depth_cues": "the overlays (if any), the STATE numbers"}
    return {"state_item": "", "state_check": "the EXEC line",
            "overlay_state": "the overlays (if any) and the EXEC lines",
            "state_doc": ("The feedback does not report the measured robot state: the position of the TCP "
                          "is not given as numbers. Track it from the images and from the achieved motion in "
                          "each EXEC line."),
            "state_line": "",
            "grasp_check": "the image, and whether the fingers stopped at the object's width",
            "depth_cues": "the overlays (if any), the achieved motion in EXEC"}


def _load_fewshot(cfg: Config) -> str:
    """``prompt.fewshot`` -> "Appendix C" text (empty when unset). A markdown/text file is
    used verbatim; a run directory's messages.jsonl is rendered as a plain exchange with
    images shown as ``<image>`` (the demo then lands in the cached system prefix)."""
    ref = cfg.prompt.fewshot
    if not ref:
        return ""
    p = Path(ref)
    if p.is_dir():
        mp = p / "messages.jsonl"
        if not mp.exists():
            raise FileNotFoundError(f"prompt.fewshot: {p} has no messages.jsonl")
        out = []
        for line in mp.read_text().splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            if m.get("role") == "system":
                continue
            c = m.get("content")
            parts = c if isinstance(c, list) else [c]
            txt = []
            for part in parts:
                if isinstance(part, str):
                    txt.append(part)
                elif isinstance(part, dict) and ("image_sha" in part or part.get("type") == "image_url"):
                    txt.append("<image>")
                elif isinstance(part, dict):
                    txt.append(str(part.get("text", "")))
            out.append(f"{m.get('role')}:")
            out += ["    " + ln for ln in "\n".join(t for t in txt if t).splitlines()]
        body = "\n".join(out)
    elif p.is_file():
        body = p.read_text().strip()
    else:
        raise FileNotFoundError(f"prompt.fewshot: {ref} not found")
    return ("## Appendix C: demonstration (a recorded episode; images omitted)\n\n" + body).strip()


def _table_top(spec: RobotSpec) -> float:
    return float(spec.table_z) if spec.table_z is not None else float(spec.workspace_lo[2])


def _example_state(cfg: Config, spec: RobotSpec, pos=None, grip_mm=None,
                   closed=False, holding=False, q=None) -> RobotState:
    n = len(spec.joints)
    return RobotState(
        t=0.0,
        q=np.asarray(q if q is not None else (list(spec.home_q) + [0.0] * n)[:n], dtype=float),
        tcp_pos=np.asarray(example_xyz(spec) if pos is None else pos, dtype=float),
        tcp_rotvec=np.array([math.pi, 0.0, 0.0]),
        gripper_mm=spec.gripper_max_mm if grip_mm is None else grip_mm,
        gripper_closed=closed,
        holding=holding,
    )


def _ee_values(cfg: Config, xyz: tuple[float, float, float], yaw: float = 0.0) -> tuple[float, ...]:
    """SI Action.values for an ee move in the configured rotation layout."""
    r = cfg.action.rotation
    if r == "none":
        return tuple(xyz)
    roll = 0.0 if cfg.action.mode == "ee_delta" else math.pi
    return tuple(xyz) + (roll, 0.0, yaw)


def _example_steps(cfg: Config, spec: RobotSpec) -> list[tuple[Action, RobotState, RobotState, list[SafetyEvent], str]]:
    """A short synthetic episode (approach above, descend in a small step that stays
    above the table, grasp, lift and check) in the configured action space:
    (action, before, after, events, status line). It demonstrates the technique of
    section 10 — no blind descents into the safety clamp."""
    a = cfg.action
    n = len(spec.joints)
    home = np.asarray((list(spec.home_q) + [0.0] * n)[:n], dtype=float)
    d1, d2, d3 = (np.asarray(v, float) for v in EE_DELTA_STEPS)
    p0 = np.asarray(example_xyz(spec, 0.12))
    p1, p2 = p0 + d1, p0 + d1 + d2
    p3 = p2 + d3
    gmax = spec.gripper_max_mm
    s0 = _example_state(cfg, spec, p0, q=home)
    if a.mode in JOINT_MODES:
        dq1 = np.zeros(n)
        dq1[0] = math.radians(8)
        dq2 = np.zeros(n)
        if n > 2:
            dq2[1], dq2[2] = math.radians(4), math.radians(6)
        q1, q2 = home + dq1, home + dq1 + dq2
        mode = ActionMode(a.mode)
        v1 = tuple(dq1) if a.mode == "joint_delta" else tuple(q1)
        v2 = tuple(dq2) if a.mode == "joint_delta" else tuple(q2)
        v3 = tuple(-0.5 * dq2) if a.mode == "joint_delta" else tuple(home + dq1 + 0.5 * dq2)
        q3 = home + dq1 + 0.5 * dq2
    else:
        mode = ActionMode(a.mode)
        delta = a.mode == "ee_delta"
        v1 = _ee_values(cfg, tuple(d1) if delta else tuple(p1))
        v2 = _ee_values(cfg, tuple(d2) if delta else tuple(p2))
        v3 = _ee_values(cfg, tuple(d3) if delta else tuple(p3))
        q1 = q2 = q3 = home
    s1 = _example_state(cfg, spec, p1, q=q1)
    s2 = _example_state(cfg, spec, p2, q=q2)
    grip_w = 32.0
    s3 = _example_state(cfg, spec, p2, grip_mm=grip_w, closed=True, holding=True, q=q2)
    s4 = _example_state(cfg, spec, p3, grip_mm=grip_w, closed=True, holding=True, q=q3)
    return [
        (Action(mode, v1, gmax * 1e-3), s0, s1, [], "STATUS OK moving above the block, gripper open"),
        (Action(mode, v2, None), s1, s2, [], "STATUS OK fingers straddle the block, descending"),
        (Action(None, None, 0.0), s2, s3, [], "STATUS OK closing on the block"),
        (Action(mode, v3, None), s3, s4, [], "STATUS OK lifting to check the grasp"),
    ]


def _example_exchange(cfg: Config, spec: RobotSpec) -> str:
    """Generated with the real formatter/parser so the example can never drift
    from the grammar actually in force."""
    from controlr.protocol.feedback import format_action
    out = ["## Appendix B: example exchange (illustrative numbers; images omitted)", ""]
    steps = _example_steps(cfg, spec)
    obs0 = Observation(t=0.0, images={}, cameras={}, state=steps[0][1])
    out.append("user:")
    out += ["    " + ln for ln in format_feedback(0, None, None, None, obs0, cfg).splitlines()]
    out.append("    <images>")
    for k, (act, before, after, events, status) in enumerate(steps):
        reply = f"{format_action(act, cfg.action)}\n{status}"
        parsed = parse_reply(reply, cfg.action, spec)
        out.append("assistant:")
        out += ["    " + ln for ln in reply.splitlines()]
        report = ExecReport(requested=parsed.actions, executed=parsed.actions, events=events,
                            state_before=before, state_after=after, duration_s=0.4 + 0.1 * k)
        obs = Observation(t=0.0, images={}, cameras={}, state=after)
        out.append("user:")
        out += ["    " + ln for ln in format_feedback(k + 1, parsed, report, None, obs, cfg).splitlines()]
        out.append("    <images>")
    out.append("")
    out.append("(The episode continues: carry the block above the target, lower it, GRIP open, "
               "lift away, check the image, then STATUS DONE.)")
    return "\n".join(out)


def _conventions_appendix(cfg: Config, spec: RobotSpec) -> str:
    a = cfg.action
    lines = ["## Appendix A: conventions reference", ""]
    lines.append(f"- Length unit: {a.pos_unit} (1 cm = 10 mm, 1 m = 1000 mm). Angle unit: {a.ang_unit} "
                 f"(1 rad = 57.3 deg, 90 deg = 1.571 rad).")
    lines.append(f"- Joint-limit margin used by the harness: {_angle(cfg, cfg.safety.joint_margin_rad)} "
                 f"inside each hard limit; a WARN appears within {_angle(cfg, cfg.safety.near_limit_rad)}.")
    if a.mode in EE_MODES:
        lines.append(f"- Orientations ({'STATE and ' if cfg.observation.state_text else ''}rotation commands) "
                     "are extrinsic roll/pitch/yaw about the "
                     f"fixed base x, y, z axes: R = Rz(yaw)·Ry(pitch)·Rx(roll). A tool pointing straight "
                     f"down has roll {_angle(cfg, math.pi)}, pitch 0; yaw then turns the jaw line about the vertical.")
        lines.append("- Positive rotation about an axis is counter-clockwise when looking from the positive "
                     "end of that axis toward the origin (right-hand rule).")
    else:
        lines.append("- Joint angles follow the right-hand rule about each joint axis; the TCP position "
                     "in STATE is computed from them (forward kinematics).")
    lo, hi = spec.workspace_lo, spec.workspace_hi
    lines.append(f"- Workspace centre: x={_num(cfg, (lo[0] + hi[0]) / 2)} y={_num(cfg, (lo[1] + hi[1]) / 2)} "
                 f"z={_num(cfg, (lo[2] + hi[2]) / 2)} {a.pos_unit}; size "
                 f"{_num(cfg, hi[0] - lo[0])} x {_num(cfg, hi[1] - lo[1])} x {_num(cfg, hi[2] - lo[2])} {a.pos_unit}.")
    lines.append("- Perspective: objects farther from the camera look smaller and higher in the image; "
                 "a height difference can look like a horizontal offset. Prefer the numeric overlays "
                 + ("and STATE " if cfg.observation.state_text else "and EXEC ") + "over pixel distances.")
    return "\n".join(lines)


def _worked_example(cfg: Config, spec: RobotSpec) -> str:  # noqa: C901
    a = cfg.action
    if a.mode in JOINT_MODES:
        n = len(spec.joints)
        home = (list(spec.home_q) + [0.0] * n)[:n]
        d = [0.0] * n
        d[0] = math.radians(10)
        f, dd = ang_factor(a), ang_decimals(a)
        if a.mode == "joint_delta":
            vals = " ".join(fmt_num(v / f, dd) for v in d)
            return (f"to turn the first joint ({spec.joints[0].name}) by +{fmt_num(math.radians(10) / f, dd)} "
                    f"{a.ang_unit} and keep all other joints, write `MOVE joint_delta {vals}`. The TCP "
                    f"then swings about the vertical base axis; read its new position in STATE.")
        vals = " ".join(fmt_num((h + dv) / f, dd) for h, dv in zip(home, d))
        return (f"from the home pose, turning the first joint ({spec.joints[0].name}) by "
                f"+{fmt_num(math.radians(10) / f, dd)} {a.ang_unit} is `MOVE joint_abs {vals}` "
                f"(all joints listed, absolute).")
    cur = example_xyz(spec)
    x0, y0 = cur[0], cur[1]
    tgt = (x0 + 0.04, y0 + 0.03)
    rot = {"none": "", "yaw": " 0", "full": ""}[a.rotation]
    if a.mode == "ee_delta":
        r = rot
        if a.rotation == "full":
            r = " 0 0 0"
        m1 = f"MOVE ee_delta {_num(cfg, tgt[0] - cur[0])} {_num(cfg, tgt[1] - cur[1])} 0{r}"
        m2 = f"MOVE ee_delta 0 0 {_num(cfg, -0.05)}{r}"
    else:
        r = ""
        if a.rotation == "yaw":
            r = " " + fmt_num(0.0, ang_decimals(a))
        elif a.rotation == "full":
            r = f" {fmt_num(math.pi / ang_factor(a), ang_decimals(a))} 0 0"
        m1 = f"MOVE ee_abs {_num(cfg, tgt[0])} {_num(cfg, tgt[1])} {_num(cfg, cur[2])}{r}"
        m2 = f"MOVE ee_abs {_num(cfg, tgt[0])} {_num(cfg, tgt[1])} {_num(cfg, cur[2] - 0.05)}{r}"
    where = (f"STATE says `tcp x={_num(cfg, cur[0])} y={_num(cfg, cur[1])} z={_num(cfg, cur[2])}`"
             if cfg.observation.state_text else
             f"the TCP is at about x={_num(cfg, cur[0])} y={_num(cfg, cur[1])} z={_num(cfg, cur[2])}")
    return (f"{where} and you "
            f"estimate the object's centre at x={_num(cfg, tgt[0])} y={_num(cfg, tgt[1])} {a.pos_unit}. "
            f"First move above it at the same height: `{m1}`. Check the new image; if the fingers are "
            f"centred over the object, descend: `{m2}`.")


def _vec(v) -> str:
    return "(" + ", ".join(f"{float(x):+.2f}" for x in v) + ")"


def _tool_doc(cfg: Config, spec: RobotSpec, state0: RobotState | None) -> str:
    """Where the tool points when its orientation is not commanded (rotation=none).
    (Directions printed with 2 decimals from the reference state — the loop passes
    ``Robot.reference_state()`` when the backend has one, so settle jitter cannot
    change the cached manual between episodes.)

    The model cannot see the orientation in STATE and cannot change it, yet it decides
    what the gripper body and wrist sweep through and along which line the jaws close.
    A tilted tool (the Isaac rig: ~35 deg below horizontal) is the difference between a
    clean grasp and driving the gripper housing into the box wall. Computed from the
    first observation, so it is exact for every backend."""
    if state0 is None or cfg.action.mode not in EE_MODES or cfg.action.rotation != "none":
        return ""
    from controlr.robot.kinematics import rotvec_to_matrix
    R = rotvec_to_matrix(np.asarray(state0.tcp_rotvec, float))
    a, jaw = R[:, 2], R[:, 0]
    below = math.degrees(math.asin(max(-1.0, min(1.0, -float(a[2])))))
    off = float(np.linalg.norm(spec.tcp_offset[:3])) if spec.tcp_offset is not None else 0.18
    horiz = [f"{'+' if a[i] > 0 else '-'}{n}" for i, n in ((0, "x"), (1, "y")) if abs(a[i]) > 0.25]
    if below > 80:
        where = "straight down"
    else:
        where = f"{below:.0f} deg below horizontal, toward {' and '.join(horiz) or 'the vertical'}"
    from controlr.robot.safety import finger_drop
    s = (f"Tool orientation (fixed for the whole episode; you command the TCP position only): the "
         f"fingers point along base {_vec(a)}, i.e. {where}. The fingertips straddle the TCP along "
         f"the jaw line; the finger bases, the gripper housing and the wrist lie BEHIND the TCP along "
         f"{_vec(-a)} (up to {_len(cfg, off)} back to the flange, then the wrist) — that side of the TCP "
         f"hits walls and objects first. The jaws open and close along {_vec(jaw)}: an object to be "
         f"grasped must have its thin side along that line.")
    if spec.finger_pad is not None:
        lo_open = finger_drop(spec, R, spec.gripper_max_mm * 1e-3)
        lo_shut = finger_drop(spec, R, 0.0)
        s += (f" The lowest fingertip is {_len(cfg, lo_open)} below the TCP with the gripper fully open "
              f"({_len(cfg, lo_shut)} closed).")
    return s


def _values(cfg: Config, spec: RobotSpec, task_text: str,
            cameras: dict[str, CameraInfo] | None, state0: RobotState | None = None) -> dict[str, str]:
    a, s, e = cfg.action, cfg.safety, cfg.episode
    lo, hi = spec.workspace_lo, spec.workspace_hi
    goal_line = {
        "never": "",
        "on_done": "    GOAL: <only after you say DONE: whether the harness's task check passed>\n",
        "always": "    GOAL: <whether the task is complete, as measured by the harness>\n",
    }.get(e.goal_feedback, "")
    done_rule = ("DONE ends the episode immediately, so be sure." if e.trust_done else
                 "The harness checks the task after DONE; if the check fails you will be told "
                 "and must continue.")
    stop_rule = ("- STOP means the arm stopped early (e.g. it pushed against something too hard) and now "
                 "holds still. Look at the image, back off, and try a different way. "
                 + ("The first STOP ends the episode." if e.max_stops <= 1 else
                    f"The episode ends after {e.max_stops} STOPs."))
    step_limits = f"at most {_len(cfg, s.max_step_m)} of TCP travel per MOVE line"
    if a.mode in JOINT_MODES or a.rotation != "none":
        step_limits += f" and at most {_angle(cfg, s.max_step_rad)} of rotation / joint change per line"
    extra = "\n".join(f"- {r}" for r in cfg.prompt.extra_rules) if cfg.prompt.extra_rules else "(none)"
    appendix = _conventions_appendix(cfg, spec) + "\n\n" + _example_exchange(cfg, spec)
    # z lower bound actually enforced for the TCP (the table clearance applies to the lowest
    # fingertip, which depends on the tool orientation and the opening)
    z_lo, z_note = lo[2], ""
    if spec.table_z is not None:
        from controlr.robot.kinematics import rotvec_to_matrix
        from controlr.robot.safety import finger_drop
        floor = spec.table_z + s.table_clearance_m
        if state0 is not None and a.rotation == "none" and spec.finger_pad is not None:
            R0 = rotvec_to_matrix(np.asarray(state0.tcp_rotvec, float))
            z_lo = max(lo[2], floor + finger_drop(spec, R0, spec.gripper_max_mm * 1e-3))
            z_note = (f" (TCP z lower bound with the gripper fully open; closed it is "
                      f"{_num(cfg, max(lo[2], floor + finger_drop(spec, R0, 0.0)))})")
        else:
            z_lo = max(lo[2], floor)
            if spec.finger_pad is not None:
                z_note = " (higher when the fingertips reach below the TCP; see section 8)"
    clearance_point = "the lowest fingertip" if spec.finger_pad is not None else "the TCP"
    path_rule = ("Each MOVE runs along a straight line. If the far end of the line is out of reach (or "
                 "would need a large joint swing), the move stops at the reachable part and a CLAMP "
                 "line says how far it got." if a.mode in EE_MODES else
                 "Joint moves are checked along the whole joint path; a move that would leave the "
                 "envelope on the way is shortened and a CLAMP line says so.")
    return {
        "robot_name": spec.name,
        "n_joints": str(len(spec.joints)),
        "joint_names": ", ".join(j.name for j in spec.joints),
        "joint_table": _joint_table(cfg, spec),
        "base_frame_doc": spec.base_frame_doc.strip(),
        "tcp_doc": spec.tcp_doc.strip(),
        "tool_doc": _tool_doc(cfg, spec, state0),
        "gripper_doc": _gripper_doc(cfg, spec),
        "ws_x": f"{_num(cfg, lo[0])}..{_num(cfg, hi[0])}",
        "ws_y": f"{_num(cfg, lo[1])}..{_num(cfg, hi[1])}",
        "ws_z": f"{_num(cfg, z_lo)}..{_num(cfg, hi[2])}",
        "ws_z_note": z_note,
        "clearance_point": clearance_point,
        "path_rule": path_rule,
        "pos_unit": a.pos_unit,
        "ang_unit": a.ang_unit,
        "camera_axes": _camera_axes(cfg, spec, cameras),
        "worked_example": _worked_example(cfg, spec),
        "observation_doc": _observation_doc(cfg, spec),
        **_state_values(cfg, spec),
        "grammar": grammar_spec(a, spec),
        "goal_line": goal_line,
        "done_rule": done_rule,
        "stop_rule": stop_rule,
        "table_clearance": _len(cfg, s.table_clearance_m),
        "step_limits": step_limits,
        "clamp_mode": "clamped to the nearest allowed value" if s.clamp else "rejected (not executed)",
        "coarse_step": _len(cfg, s.max_step_m * 0.5),
        "fine_step": _len(cfg, 0.01),
        "extra_rules": extra,
        "task": task_text.strip() or "(given in the first user turn)",
        "appendix": appendix,
        "fewshot": _load_fewshot(cfg),
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def build_system_prompt(cfg: Config, spec: RobotSpec, task_text: str = "",
                        cameras: dict[str, CameraInfo] | None = None,
                        state0: RobotState | None = None) -> str:
    """The operating manual for one episode.

    ``task_text`` empty (the loop's default) -> the task is given in the first
    user turn and the manual is identical across tasks, which lets the provider
    cache it across episodes too. ``cameras`` (optional): calibrated cameras of
    the first observation; when given, the manual states how the base axes
    appear in each image. ``state0`` (optional): the reset state; with rotation=none
    the manual then states the fixed tool orientation."""
    text = fill(load_template(cfg.prompt.system), _values(cfg, spec, task_text, cameras, state0))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def build_planner_prompt(cfg: Config, spec: RobotSpec, task_text: str = "",
                         cameras: dict[str, CameraInfo] | None = None, max_words: int = 250,
                         state0: RobotState | None = None) -> str:
    """System text for the one-off planning call: the same manual the controller
    gets (same frame, units and limits, so the plan's numbers transfer) followed
    by the planning instructions. The TASK and the first observation go in the
    user message (``controlr.loop.run_planner``)."""
    manual = build_system_prompt(cfg, spec, task_text, cameras, state0)
    planner = fill(load_template(cfg.planner.prompt),
                   {"pos_unit": cfg.action.pos_unit, "max_words": str(max_words)})
    return manual + "\n---\n\n" + planner.strip() + "\n"
