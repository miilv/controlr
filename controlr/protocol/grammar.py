"""Reply grammar: the text contract between the model and the harness.

The model controls the robot with a handful of line-oriented commands::

    MOVE <mode> v1 v2 ... [GRIP open|close|<width>]
    GRIP open|close|<width>
    HOLD
    STATUS OK|DONE|FAIL|STUCK|LIMIT [short note]

Why plain text lines and not JSON / tool calls: lines stream, so the client can
close the stream the moment a STATUS line lands (``is_complete``) — this is a
large part of the ~1 s/turn budget — and a malformed line costs one line, not
the whole reply. Prior LLM-as-policy harnesses (Robocurve inspect-robots,
RoboProbe, innate-os PR #817) all learned that validation errors must go back
to the model as correctable text instead of killing the run; ``parse_reply``
therefore never raises — it collects errors for ``controlr.protocol.feedback``.

Units: the model speaks the configured LLM units (``ActionConfig.pos_unit`` /
``ang_unit``, default mm / deg); everything returned here is SI (m, rad), as
``controlr.types`` requires.

Rotation conventions (contract additions documented here, see types.Action):
  * ``ee_delta`` + rotation=yaw: values = (dx, dy, dz, 0, 0, dyaw);
    rotation=full: (dx, dy, dz, droll, dpitch, dyaw) — an extrinsic rotation
    about the base x, y, z axes (applied in that order) about the TCP point,
    pre-multiplied onto the current orientation: R_new = Rz Ry Rx · R_cur.
  * ``ee_abs`` + rotation=full: (x, y, z, roll, pitch, yaw), absolute extrinsic
    RPY about base x/y/z: R = Rz(yaw) Ry(pitch) Rx(roll).
  * ``ee_abs`` + rotation=yaw: values = (x, y, z, yaw) — 4 values: ``yaw`` is the
    absolute heading about base z; the tool keeps its tilt (roll/pitch of the
    episode's reference orientation). A forced top-down tool is unreachable on the
    Isaac rig, so the heading is the only thing this mode commands. The STATE line
    reports yaw extracted with the same convention, so the numbers round-trip.

Command vs prose (models write prose despite the manual): a line is a command
when its keyword is UPPERCASE (``MOVE``/``GRIP``/``HOLD``/``STATUS``). A
mixed/lower-case keyword line ("Move the gripper left", "Hold on", "Status: ok so
far") is a command only if it parses cleanly (and, for STATUS, only as the last
line of the reply); otherwise it is prose and ignored without an error. The
streaming early stop (``is_complete``) only fires on an uppercase ``STATUS``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from controlr.config import ActionConfig
from controlr.types import Action, ActionMode, ParsedReply, RobotSpec, Status

STATUS_WORDS: tuple[str, ...] = tuple(s.value for s in Status)
EE_MODES = (ActionMode.EE_DELTA.value, ActionMode.EE_ABS.value)
JOINT_MODES = (ActionMode.JOINT_DELTA.value, ActionMode.JOINT_ABS.value)
TOP_DOWN_ROLL = math.pi    # roll used for ee_abs + rotation=yaw (tool pointing down)

_POS_TO_M = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}
_ANG_TO_RAD = {"deg": math.pi / 180.0, "rad": 1.0}


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

def pos_factor(cfg: ActionConfig) -> float:
    """LLM position unit -> metres."""
    try:
        return _POS_TO_M[cfg.pos_unit]
    except KeyError:
        raise ValueError(f"unknown pos_unit {cfg.pos_unit!r} (mm|cm|m)") from None


def ang_factor(cfg: ActionConfig) -> float:
    """LLM angle unit -> radians."""
    try:
        return _ANG_TO_RAD[cfg.ang_unit]
    except KeyError:
        raise ValueError(f"unknown ang_unit {cfg.ang_unit!r} (deg|rad)") from None


def pos_decimals(cfg: ActionConfig, fine: bool = False) -> int:
    """Fixed decimals per unit so text is deterministic (cache stability) and
    roughly 1 mm (state) / 0.1 mm (achieved deltas) resolution."""
    base = {"mm": 0, "cm": 1, "m": 3}[cfg.pos_unit]
    return base + (1 if fine else 0)


def ang_decimals(cfg: ActionConfig, fine: bool = False) -> int:
    base = {"deg": 0, "rad": 2}[cfg.ang_unit]
    return base + (1 if fine else 0)


def fmt_num(x: float, decimals: int) -> str:
    """Fixed-decimal formatting without '-0' artefacts."""
    s = f"{x:.{decimals}f}"
    if float(s) == 0.0:
        s = f"{0.0:.{decimals}f}"
    return s


# ---------------------------------------------------------------------------
# grammar description (system prompt / reminders)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Layout:
    names: tuple[str, ...]        # value names shown to the model
    kinds: tuple[str, ...]        # "pos" | "ang" per value


def _layout(cfg: ActionConfig, spec: RobotSpec) -> _Layout:
    mode = cfg.mode
    if mode in JOINT_MODES:
        pre = "d" if mode == "joint_delta" else ""
        return _Layout(tuple(pre + j.name for j in spec.joints), ("ang",) * len(spec.joints))
    if mode not in EE_MODES:
        raise ValueError(f"unknown action mode {mode!r}")
    pre = "d" if mode == "ee_delta" else ""
    names = [pre + "x", pre + "y", pre + "z"]
    kinds = ["pos"] * 3
    if cfg.rotation == "yaw":
        names += [pre + "yaw"]
        kinds += ["ang"]
    elif cfg.rotation == "full":
        names += [pre + "roll", pre + "pitch", pre + "yaw"]
        kinds += ["ang"] * 3
    elif cfg.rotation != "none":
        raise ValueError(f"unknown rotation {cfg.rotation!r} (none|yaw|full)")
    return _Layout(tuple(names), tuple(kinds))


def _grip_syntax(cfg: ActionConfig) -> str:
    return "open|close" if cfg.gripper == "binary" else f"open|close|<width {cfg.pos_unit}>"


def example_xyz(spec: RobotSpec, dz_above_table: float = 0.15) -> tuple[float, float, float]:
    """A plausible TCP position for examples: the workspace centre (rounded to 10 mm) at
    ``dz_above_table`` above the table — same signs and magnitudes as the real rig (a fixed
    x=+300 example contradicted a workspace at negative x and invited sign errors)."""
    lo, hi = spec.workspace_lo, spec.workspace_hi
    top = float(spec.table_z) if spec.table_z is not None else float(lo[2])
    return (round((lo[0] + hi[0]) / 2, 2), round((lo[1] + hi[1]) / 2, 2), round(top + dz_above_table, 2))


# example sequence (approach, descend, grasp, lift, release, lift away + DONE)
EE_DELTA_STEPS = ([0.03, 0.02, 0.0], [0.0, 0.0, -0.03], [0.0, 0.0, 0.05])


def _example_values(cfg: ActionConfig, spec: RobotSpec, which: int) -> list[float]:
    """Plausible example values in LLM units (for the spec text only)."""
    p, a = pos_factor(cfg), ang_factor(cfg)
    lay = _layout(cfg, spec)
    if cfg.mode == "ee_delta":
        si = EE_DELTA_STEPS[which]
        rot = {0: [0.0, 0.0, math.radians(10)], 1: [0.0, 0.0, 0.0], 2: [0.0, 0.0, 0.0]}[which]
    elif cfg.mode == "ee_abs":
        x, y, z = example_xyz(spec)
        si = [[x + 0.03, y + 0.02, z], [x + 0.03, y + 0.02, z - 0.03], [x + 0.03, y + 0.02, z + 0.02]][which]
        rot = {0: [180.0, 0.0, 15.0], 1: [180.0, 0.0, 15.0], 2: [180.0, 0.0, 15.0]}[which]
        rot = [math.radians(v) for v in rot]
    else:
        n = len(spec.joints)
        out = []
        for i in range(n):
            if cfg.mode == "joint_delta":
                v = [math.radians(5) if i == 0 else 0.0, math.radians(-3) if i == 1 else 0.0,
                     math.radians(4) if i == 2 else 0.0][which]
            else:
                v = spec.home_q[i] if i < len(spec.home_q) else 0.0
                if i == which:   # a different joint off home in each example
                    v += math.radians([10, 5, -5][which])
            out.append(v / a)
        return out
    vals = [v / p for v in si]
    if cfg.rotation == "yaw":
        vals.append(rot[2] / a)
    elif cfg.rotation == "full":
        vals += [r / a for r in rot]
    assert len(vals) == len(lay.names)
    return vals


def _fmt_values(vals: list[float], cfg: ActionConfig, spec: RobotSpec) -> str:
    lay = _layout(cfg, spec)
    out = []
    for v, k in zip(vals, lay.kinds):
        d = pos_decimals(cfg) if k == "pos" else ang_decimals(cfg)
        if cfg.pos_unit == "mm" and k == "pos":
            d = 0
        out.append(fmt_num(v, d))
    return " ".join(out)


def move_syntax(cfg: ActionConfig, spec: RobotSpec | None) -> str:
    """``MOVE ee_delta dx dy dz [GRIP open|close]`` for the configured mode
    (``spec`` may be None when joint names are unknown)."""
    if spec is None and cfg.mode in JOINT_MODES:
        names = "<one value per joint>"
    else:
        names = " ".join(_layout(cfg, spec).names)   # type: ignore[arg-type]
    return f"MOVE {cfg.mode} {names} [GRIP {_grip_syntax(cfg)}]"


def grammar_reminder(cfg: ActionConfig, spec: RobotSpec | None = None) -> str:
    """One line, appended to feedback after parse errors."""
    unit = f"{cfg.pos_unit}/{cfg.ang_unit}"
    return (f"GRAMMAR: up to {cfg.max_chunk} line(s) of `{move_syntax(cfg, spec)}` | "
            f"`GRIP {_grip_syntax(cfg)}` | `HOLD`, then exactly one "
            f"`STATUS {'|'.join(STATUS_WORDS)} [note]` as the last line ({unit}).")


def grammar_spec(cfg: ActionConfig, spec: RobotSpec) -> str:
    """The grammar block inserted into the system prompt.

    Rendered from the configuration so the prompt can never disagree with the
    parser (mode, arity, units, rotation, gripper, chunk size)."""
    lay = _layout(cfg, spec)
    P, A = cfg.pos_unit, cfg.ang_unit
    delta = cfg.mode.endswith("_delta")
    half_turn = fmt_num(math.pi / ang_factor(cfg), ang_decimals(cfg))
    lines: list[str] = []
    lines.append("Reply with plain text lines, nothing else (no markdown, no code fences):")
    lines.append("")
    lines.append(f"    {move_syntax(cfg, spec)}")
    lines.append(f"    GRIP {_grip_syntax(cfg)}")
    lines.append("    HOLD")
    lines.append(f"    STATUS {'|'.join(STATUS_WORDS)} [short note]")
    lines.append("")
    # values
    if cfg.mode in EE_MODES:
        what = ("a relative move of the TCP from where it is now" if delta
                else "an absolute TCP target")
        lines.append(f"- MOVE {cfg.mode}: {what}, in the robot base frame. "
                     f"Exactly {len(lay.names)} numbers: {' '.join(lay.names)}.")
        lines.append(f"  Positions in {P}." + (f" Angles in {A}." if len(lay.names) > 3 else ""))
        if cfg.rotation == "none":
            lines.append("  Orientation is not commanded: the tool keeps its current orientation.")
        elif cfg.rotation == "yaw":
            if delta:
                lines.append("  dyaw turns the tool about the vertical line through the TCP (positive = "
                             "counter-clockwise seen from above); the tilt is kept. A line with both a "
                             "move and a turn does both along the way; dyaw 0 keeps the current heading.")
            else:
                lines.append("  yaw is the absolute tool heading about the vertical base z axis; the "
                             "tool keeps its tilt (only the heading is commanded).")
        else:
            if delta:
                lines.append("  droll dpitch dyaw rotate the tool about the base x, y, z axes "
                             "(extrinsic, applied in that order), pivoting about the TCP.")
            else:
                lines.append(f"  roll pitch yaw are absolute extrinsic angles about base x, y, z "
                             f"(R = Rz(yaw)·Ry(pitch)·Rx(roll)); tool straight down = roll {half_turn} pitch 0.")
    else:
        what = "added to the current joint angles" if delta else "absolute joint angles"
        lines.append(f"- MOVE {cfg.mode}: {what}, exactly {len(lay.names)} numbers in joint "
                     f"order ({', '.join(j.name for j in spec.joints)}), in {A}.")
    # gripper
    if cfg.gripper == "binary":
        lines.append(f"- GRIP open = fully open ({fmt_num(spec.gripper_max_mm, 0)} mm), "
                     f"GRIP close = close until the fingers stop (on an object or fully shut).")
    else:
        lines.append(f"- GRIP <width>: target finger opening in {P} "
                     f"(0 = closed, {fmt_num(spec.gripper_max_mm * 1e-3 / pos_factor(cfg), pos_decimals(cfg))} "
                     f"= fully open); open/close also work.")
    lines.append("  GRIP may stand alone or be appended to a MOVE line (gripper acts after the motion).")
    lines.append("  The gripper keeps its last commanded state until you change it.")
    lines.append("- HOLD: do nothing this turn (e.g. to take another look).")
    if cfg.max_chunk == 1:
        lines.append("- At most 1 MOVE or GRIP line per reply. You see a new image after it.")
    else:
        lines.append(f"- At most {cfg.max_chunk} MOVE/GRIP lines per reply, executed in order "
                     f"without looking; you see a new image after the last one. Extra lines are dropped.")
    lines.append("- STATUS is mandatory and must be the LAST line; anything after it is ignored. "
                 "The note after the status word is optional and short (<= 12 words).")
    # examples
    lines.append("")
    lines.append("Examples (each block is one whole reply, from successive turns):")
    ex0 = _fmt_values(_example_values(cfg, spec, 0), cfg, spec)
    ex1 = _fmt_values(_example_values(cfg, spec, 1), cfg, spec)
    ex2 = _fmt_values(_example_values(cfg, spec, 2), cfg, spec)
    lines.append(f"    MOVE {cfg.mode} {ex0} GRIP open")
    lines.append("    STATUS OK moving above the object, gripper open")
    lines.append("")
    if cfg.max_chunk >= 2:
        lines.append(f"    MOVE {cfg.mode} {ex1}")
        lines.append("    GRIP close")
        lines.append("    STATUS OK descended and grasping")
    else:
        lines.append(f"    MOVE {cfg.mode} {ex1}")
        lines.append("    STATUS OK aligned, descending to grasp height")
        lines.append("")
        lines.append("    GRIP close")
        lines.append("    STATUS OK grasping")
    lines.append("")
    lines.append("    HOLD")
    lines.append("    STATUS OK checking the grasp in a new image")
    lines.append("")
    lines.append("    GRIP open")
    lines.append("    STATUS OK releasing above the destination")
    lines.append("")
    lines.append(f"    MOVE {cfg.mode} {ex2}")
    lines.append("    STATUS DONE object released at the destination, gripper lifted clear")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

_KEYWORDS = ("MOVE", "GRIP", "HOLD", "STATUS")
# leading list/quote/markdown decoration: "- ", "* ", "> ", "1. ", "1) ", "**", "`"
_DECOR_RE = re.compile(r"^(?:[-*>•]\s+|\d+[.)]\s+|[*_`>]+)+")
# keyword case-sensitive (uppercase STATUS only), status word any case
_STATUS_LINE_RE = re.compile(
    r"^\s*(?:[-*>•]\s+|\d+[.)]\s+|[*_`>]+)*\s*STATUS\b[\s:=*_`]*(?i:(" + "|".join(STATUS_WORDS)
    + r"))(?=[\s.,;:!*_`)]|$)",
    re.MULTILINE,
)
_NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$", re.IGNORECASE)


def _clean_line(line: str) -> str:
    s = line.replace("\u2212", "-").strip()           # Unicode minus -> ASCII
    s = _DECOR_RE.sub("", s).strip()
    # strip markdown emphasis/backticks anywhere; tolerate "STATUS: OK"
    s = s.replace("`", "").replace("**", "").strip()
    return s


def _split_tokens(s: str) -> list[str]:
    s = s.replace(",", " ").replace("[", " ").replace("]", " ").replace("(", " ").replace(")", " ")
    s = s.replace(":", " ") if s.upper().startswith("STATUS") else s
    return s.split()


def _parse_number(tok: str, kind: str, cfg: ActionConfig) -> float:
    """Number in LLM units -> SI. Tolerates ``dx=20`` and a unit suffix equal
    to the configured unit (``20mm``); a different unit suffix is an error
    (silently converting would hide a misunderstanding of the grammar)."""
    t = tok.split("=", 1)[1] if "=" in tok else tok
    unit = cfg.pos_unit if kind == "pos" else cfg.ang_unit
    m = re.match(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)([a-z°]*)$", t, re.IGNORECASE)
    if not m:
        raise ValueError(f"not a number: {tok!r}")
    num, suffix = m.group(1), m.group(2).lower()
    if suffix == "°":
        suffix = "deg"
    if suffix and suffix != unit:
        raise ValueError(f"{tok!r}: use {unit} without a different unit suffix")
    v = float(num)
    if not math.isfinite(v):
        raise ValueError(f"not finite: {tok!r}")
    return v * (pos_factor(cfg) if kind == "pos" else ang_factor(cfg))


def _parse_grip(tokens: list[str], cfg: ActionConfig, spec: RobotSpec) -> float:
    """``open|close|<width>`` -> opening in metres."""
    if not tokens:
        raise ValueError("GRIP needs open|close" + ("" if cfg.gripper == "binary" else "|<width>"))
    if len(tokens) > 1:
        raise ValueError(f"GRIP takes one argument, got {' '.join(tokens)!r}")
    t = tokens[0].lower()
    if t in ("open", "opened"):
        return spec.gripper_max_mm * 1e-3
    if t in ("close", "closed"):
        return 0.0
    if cfg.gripper == "binary":
        raise ValueError(f"GRIP {tokens[0]!r}: gripper is binary, use GRIP open|close")
    w = _parse_number(tokens[0], "pos", cfg)
    if w < 0:
        raise ValueError(f"GRIP width must be >= 0, got {tokens[0]!r}")
    return w


def _values_to_action(vals: list[float], cfg: ActionConfig) -> tuple[float, ...]:
    """Map parsed SI values onto Action.values per the documented conventions."""
    if cfg.mode in JOINT_MODES or cfg.rotation == "none":
        return tuple(vals)
    if cfg.rotation == "yaw":
        x, y, z, yaw = vals
        if cfg.mode == "ee_delta":
            return (x, y, z, 0.0, 0.0, yaw)
        return (x, y, z, yaw)           # absolute heading; the envelope keeps the reference tilt
    return tuple(vals)


def _parse_move(tokens: list[str], cfg: ActionConfig, spec: RobotSpec) -> Action:
    """tokens after 'MOVE'."""
    lay = _layout(cfg, spec)
    rest = list(tokens)
    if rest and not _looks_numeric(rest[0]):
        mode = rest.pop(0).lower().rstrip(":")
        if mode != cfg.mode:
            raise ValueError(f"mode {mode!r} is not available; use MOVE {cfg.mode}")
    grip: float | None = None
    upper = [t.upper() for t in rest]
    if "GRIP" in upper:
        i = upper.index("GRIP")
        grip = _parse_grip(rest[i + 1:], cfg, spec)
        rest = rest[:i]
    # a trailing unit word equal to the configured unit ("MOVE ee_delta 0 0 -10 mm")
    if len(rest) == len(lay.names) + 1 and rest[-1].lower() in (cfg.pos_unit, cfg.ang_unit):
        rest = rest[:-1]
    if len(rest) != len(lay.names):
        raise ValueError(f"MOVE {cfg.mode} needs {len(lay.names)} numbers "
                         f"({' '.join(lay.names)}), got {len(rest)}")
    vals = [_parse_number(t, k, cfg) for t, k in zip(rest, lay.kinds)]
    return Action(mode=ActionMode(cfg.mode), values=_values_to_action(vals, cfg), gripper=grip)


def _looks_numeric(tok: str) -> bool:
    t = tok.split("=", 1)[1] if "=" in tok else tok
    return bool(re.match(r"^[+-]?(?:\d|\.\d)", t))


def parse_reply(text: str, cfg: ActionConfig, spec: RobotSpec) -> ParsedReply:
    """Tolerant parser. Never raises on model output; problems go to ``errors``.

    Accepted noise: markdown fences, list bullets, bold/backticks, prose lines
    (ignored), commas between numbers, ``dx=20`` tokens, a unit suffix or trailing
    unit word equal to the configured unit, Unicode minus. An UPPERCASE keyword line
    that is malformed is dropped and reported; a mixed-case keyword line that does
    not parse is prose (see the module docstring)."""
    out = ParsedReply(actions=[], status=None)
    n_action_lines = 0
    dropped = 0
    after_status: list[str] = []
    lines = [ln for ln in (_clean_line(r) for r in text.splitlines()) if ln and not ln.startswith("```")]
    for idx, line in enumerate(lines):
        toks = _split_tokens(line)
        if not toks:
            continue
        head = toks[0].rstrip(":")
        kw = head.upper()
        if kw not in _KEYWORDS:
            continue   # prose
        strict = head == kw            # uppercase keyword: a command, errors are reported
        if out.complete:
            if kw != "STATUS" and strict:
                after_status.append(line)
            continue
        if kw == "STATUS":
            word = toks[1].upper().rstrip(".,;:!") if len(toks) > 1 else ""
            if not strict and idx != len(lines) - 1:
                continue               # "Status: ok so far. Next I'll ..." mid-reply is prose
            if word not in STATUS_WORDS:
                if strict:
                    out.errors.append(f"STATUS needs one of {'|'.join(STATUS_WORDS)}, got {line!r}")
                continue
            out.status = Status(word)
            note_m = re.match(r"^\s*STATUS\b[\s:=]*\S+\s*(.*)$", line, re.IGNORECASE)
            out.note = note_m.group(1).strip() if note_m else ""
            out.complete = True
            continue
        try:
            if kw == "HOLD":
                if len(toks) > 1:
                    raise ValueError(f"HOLD takes no arguments: {line!r}")
                continue   # explicit no-op: no Action (an empty action list = hold)
            if kw == "GRIP":
                action = Action(mode=None, values=None, gripper=_parse_grip(toks[1:], cfg, spec), raw=line)
            else:
                a = _parse_move(toks[1:], cfg, spec)
                action = Action(mode=a.mode, values=a.values, gripper=a.gripper, raw=line)
        except ValueError as e:
            if strict:
                out.errors.append(f"{line!r}: {e}")
            continue
        n_action_lines += 1
        if n_action_lines > cfg.max_chunk:
            dropped += 1
            continue
        out.actions.append(action)
    if dropped:
        out.errors.append(f"{n_action_lines} action lines, max is {cfg.max_chunk}: "
                          f"executed the first {cfg.max_chunk}, dropped {dropped}")
    if after_status:
        out.errors.append(f"ignored {len(after_status)} line(s) after STATUS (STATUS must be last)")
    if not out.complete:
        out.errors.append("missing STATUS line (end every reply with STATUS "
                          + "|".join(STATUS_WORDS) + ")")
    return out


def is_complete(text: str) -> bool:
    """True once a full STATUS line is present: a recognised status word
    followed by whitespace/punctuation or the end of the text.

    Used as the streaming early-stop predicate, so partial words must not
    count: ``"STATUS O"`` and ``"STATUS DON"`` are incomplete, ``"STATUS OK"``
    (end of text) and ``"STATUS OK\\n"`` are complete. Stopping right after the
    word may truncate the optional note — that is accepted (notes are optional,
    latency is not)."""
    return _STATUS_LINE_RE.search(text) is not None


def strip_partial_note(text: str) -> str:
    """Reply text to STORE after an early stop whose STATUS line was cut mid-note.

    The cut point of a stream depends on the router's chunking ("STATUS OK moving
    above est"), and the stored reply is re-sent (cached) every later turn. If the
    STATUS line is newline-terminated it was complete and the text is kept up to that
    newline; otherwise the partial note is dropped (``STATUS OK``) — deterministic
    whatever the chunking."""
    m = _STATUS_LINE_RE.search(text)
    if m is None:
        return text
    nl = text.find("\n", m.end())
    if nl >= 0:
        return text[:nl]
    return text[:m.end()]
