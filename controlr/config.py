"""Experiment configuration.

One YAML file (plus optional ``--set a.b=c`` overrides) fully describes an
episode/experiment. Every experiment axis Ilia listed is a field here, so an
ablation is a sweep over config values — never a code change.

Loading order: dataclass defaults <- base YAML(s) (``extends:``) <- file <- overrides.
``${VAR}`` in string values is expanded from the environment (and ``.env``).
"""

from __future__ import annotations

import copy
import dataclasses
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    base_url: str = "${OMNIROUTE_BASE_URL}"
    api_key_env: str = "OMNIROUTE_API_KEY"
    model: str = "claude/claude-sonnet-5-5"
    max_tokens: int = 2000         # counts thinking tokens too; 400 starved claude/claude-sonnet-5 (empty replies)
    temperature: float | None = None
    extra_body: dict = field(default_factory=dict)   # passed through verbatim (e.g. reasoning_effort)
    cache: str = "auto"            # auto | anthropic | none   (auto: anthropic markers for claude routes)
    cache_ttl: str = "5m"          # 5m | 1h  (anthropic marker ttl)
    early_stop: bool = True        # close the stream once the reply grammar is complete
    # Execute as soon as the STATUS word has streamed (the action lines are complete then);
    # the rest of the STATUS note and the usage chunk are read in the background during
    # execution. The stored reply and the executed actions are the same as without it —
    # only the turn gets shorter (~0.3 s mean on cx/; journal/2026-10-04-turn-latency.md).
    overlap_tail: bool = True
    timeout_s: float = 180.0
    max_retries: int = 3
    # after an early stop keep reading up to this long for the trailing usage chunk (and the
    # rest of the STATUS note line); 0 = close immediately (usage then None -> no cache/token
    # stats). Same value as configs/base.yaml (single source for the default).
    usage_grace_s: float = 0.3
    # Prefix the first user turn (and the planner request) with the run id. omniroute
    # replays cached *responses* for byte-identical requests (~0.4 s, no usage counters),
    # so two episodes with the same seed would otherwise get a replayed turn 0 / plan.
    # The system prompt stays identical, so the provider's prompt cache is still shared.
    request_nonce: bool = True
    # Extra HTTP request headers (e.g. a router lease/session header to pin one upstream
    # account per run; omniroute advertises X-OmniRoute-Lease-Owner — unverified, see
    # docs/reviews/2026-10-02-fixlog.md). Values may use ${VAR}.
    extra_headers: dict = field(default_factory=dict)


@dataclass
class PlannerConfig:
    enabled: bool = True
    # Effort is encoded in the model id: through omniroute, extra_body.reasoning_effort on
    # claude/claude-opus-5-5 changed reasoning_tokens only marginally (low 162 vs xhigh 253 on
    # the same prompt) — not a reliable switch. Integration re-check (2026-10-02): reasoning_effort
    # "low" on a harder prompt still produced 2237 reasoning tokens / 25 s, while the -xhigh id
    # thinks 5.7-7.4k tokens / 77-90 s on the planning prompt. The suffixed id is unambiguous.
    model: str = "claude/claude-opus-5-5-xhigh"
    extra_body: dict = field(default_factory=dict)
    max_tokens: int = 16000
    include_in_context: bool = True   # plan text goes into the first user turn of the control transcript
    prompt: str = "planner_v0"
    # Opus xhigh thinks 54-90 s before the first content byte; the control timeout (180 s per
    # read) is too tight for long thinks, and a timeout retry re-pays the whole plan.
    timeout_s: float = 600.0
    max_retries: int = 1
    # Use this plan text instead of calling the planner (control-side ablations then compare
    # on an identical plan; a sweep can pin one plan for all points).
    plan_file: str | None = None


@dataclass
class RobotConfig:
    backend: str = "isaac"          # isaac | mock | replay | ur (later)
    params: dict = field(default_factory=dict)   # backend-specific (scene, physics, cameras, replay dir...)


@dataclass
class TaskConfig:
    name: str = "pick_place"        # reach | push | pick_place | (backend-defined)
    instruction: str = ""           # empty -> task default instruction
    params: dict = field(default_factory=dict)   # randomisation ranges, object choices, success tolerances


@dataclass
class ObservationConfig:
    cameras: list[str] = field(default_factory=lambda: ["scene"])
    size: int = 448                  # long edge in px after resize (images are never upscaled)
    renderers: list[str] = field(default_factory=lambda: ["raw"])
    # raw | grid | ee_marker | axes | diff | heatmap  — applied per camera in order;
    # "diff"/"heatmap" add an extra image derived from the previous observation.
    tile: bool = False               # tile all cameras (and derived images) into one image
    state_text: bool = True          # append the measured state as a STATE line
    # Fingertip (tactile pad) information in the feedback (feedback.level=full only): pad
    # forces, "touched the packet" events and the pad-based `holding` field of STATE. False
    # (default since 2026-10-02): none of it reaches the model. Tactile data and the gripper's
    # own object detection are still logged (turns.jsonl "backend").
    tactile: bool = False
    jpeg_quality: int = 90
    first_turn_size: int | None = None   # optional larger image on the first turn
    grid_z: float | None = None      # m, base-frame height of the "grid" overlay plane; None = RobotSpec.table_z (else 0)
    grid_step: float = 0.05          # m, grid line spacing (labels every 2 lines)


@dataclass
class ActionConfig:
    mode: str = "ee_delta"           # ee_delta | ee_abs | joint_delta | joint_abs
    rotation: str = "none"           # none | yaw | full   (ee modes only)
    pos_unit: str = "mm"             # mm | cm | m
    ang_unit: str = "deg"            # deg | rad
    max_chunk: int = 1               # max MOVE lines per reply executed before the next observation
    gripper: str = "binary"          # binary (open/close) | width (mm)
    format: str = "text"             # text (MOVE/STATUS lines); "tool" is not implemented yet


@dataclass
class SafetyConfig:
    max_step_m: float = 0.10         # max TCP translation per MOVE line
    max_step_rad: float = 0.5236     # max rotation / joint change per MOVE line (30 deg)
    workspace_margin_m: float = 0.0  # shrink RobotSpec workspace box by this much
    table_clearance_m: float = 0.005 # min TCP height above the table surface
    joint_margin_rad: float = 0.0873 # soft margin inside hard joint limits (5 deg)
    near_limit_rad: float = 0.2618   # warn the model when within this of a joint limit (15 deg)
    clamp: bool = True               # clamp to the envelope (True) or reject the action (False)
    max_tcp_speed_m_s: float = 0.15  # execution speed (PEAK of the velocity profile)
    contact_force_stop_n: float = 80.0
    # arm/gripper vs the box (a light touch stops the arm and is reported: contact is
    # information). None -> contact_force_stop_n (the behaviour before 2026-10-02).
    box_force_stop_n: float | None = 30.0
    # the box is light (it slides at a few newtons, under any force limit): touching it while it
    # has moved this far since the motion began also stops the arm ("pushed the blue box").
    # Isaac only; 0 = off.
    box_push_stop_m: float = 0.005
    # robot-vs-manipulated-object force that stops arm motion while the object is NOT held
    # (pressing the packet into the mat made the solver diverge). 0 = off.
    object_force_stop_n: float = 40.0
    # while the object IS held: force between the object and the environment (table, mat,
    # box) that stops the arm ("the held packet pushed against the box wall"); the grip's own
    # pad forces never count. None -> object_force_stop_n.
    held_object_force_stop_n: float | None = None
    # predictive link-vs-box check before execution: the wrist / gripper housing centre line
    # against the box walls (from the CURRENT box pose) inflated by box_clearance_m.
    # block = shorten / reject the move (reported like a clamp); warn = execute, add a WARN;
    # off = no check (before 2026-10-02).
    box_collision: str = "warn"
    box_clearance_m: float = 0.05
    # ee modes: interpolate the TCP along a straight line in steps of at most this (IK per
    # step): the robot follows the commanded line instead of a joint-space arc, and an
    # unreachable or near-singular stretch shortens the move instead of dropping it.
    path_step_m: float = 0.005


@dataclass
class PromptConfig:
    system: str = "system_v0"        # template name in controlr/prompts/
    # demo appended to the system prompt (cached prefix): a .md/.txt file, or a run dir whose
    # messages.jsonl is rendered as text (images shown as <image>).
    fewshot: str | None = None
    extra_rules: list[str] = field(default_factory=list)


@dataclass
class FeedbackConfig:
    # What text the model gets with each turn's images:
    #   short - only TASK (turn 0), STATE (no `holding`), WARN (one short line per issue:
    #           clamps, skipped moves, unusable reply lines, near limits, box proximity, a DONE
    #           whose check failed) and STOP (motion stopped: contact, instability). Default
    #           since 2026-10-02 (Ilia: "keep feedback super short").
    #   full  - TURN / EXEC / CLAMP / WARN / EVENT / STOP / PARSE ERROR / GOAL / STATE (the
    #           behaviour before 2026-10-02).
    level: str = "short"
    repeat_task: bool = False        # short: repeat the TASK line in every turn, not only turn 0


@dataclass
class EpisodeConfig:
    max_turns: int = 40
    max_parse_errors: int = 3        # consecutive unparsable replies -> end episode
    goal_feedback: str = "on_done"   # never | on_done | always   (what goal info the model sees)
    trust_done: bool = False         # end on DONE even if the goal check fails
    end_on_fail: bool = True
    # STOP events (e.g. a contact-force stop) end the episode only once this many have
    # happened: the arm holds still after a stop, so the model can back off and retry, as an
    # operator would clear a protective stop. 1 = end on the first STOP. A STOP of kind
    # "unstable" (the simulator diverged) always ends the episode.
    max_stops: int = 3
    # end the episode (outcome "goal_reached") as soon as the goal check passes, even without
    # DONE — saves the remaining turns in sweeps that only measure reaching the goal
    end_on_goal: bool = False


@dataclass
class LogConfig:
    root: str = "runs"
    save_images: bool = True
    # also store every observation at native resolution without overlays
    # (raw/<turn>_<camera>.png) — the clean frames an action head trains on
    save_raw_frames: bool = False


@dataclass
class Config:
    name: str = "default"
    seed: int = 0
    llm: LLMConfig = field(default_factory=LLMConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines, existing env wins."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml_with_extends(path: Path) -> dict:
    import yaml   # lazy: the Isaac server imports this module (via safety) without PyYAML
    data = yaml.safe_load(path.read_text()) or {}
    parents = data.pop("extends", None)
    if not parents:
        return data
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for parent in parents:
        merged = _deep_merge(merged, _read_yaml_with_extends((path.parent / parent).resolve()))
    return _deep_merge(merged, data)


def parse_override(expr: str) -> tuple[list[str], Any]:
    """``a.b.c=value`` -> (["a","b","c"], yaml-parsed value)."""
    import yaml
    if "=" not in expr:
        raise ValueError(f"override must be key=value, got {expr!r}")
    key, raw = expr.split("=", 1)
    return key.strip().split("."), yaml.safe_load(raw)


def _set_path(d: dict, keys: list[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _build(cls, data: dict):
    """Recursively build a dataclass from a dict; unknown keys are an error
    (typos in experiment configs must not be silently ignored)."""
    if not dataclasses.is_dataclass(cls):
        return data
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(names)
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    hints = {f.name: f.type for f in dataclasses.fields(cls)}
    for k, v in data.items():
        sub = _SECTION_TYPES.get((cls, k))
        kwargs[k] = _build(sub, v) if sub is not None and isinstance(v, dict) else v
    return cls(**kwargs)


_SECTION_TYPES = {
    (Config, "llm"): LLMConfig,
    (Config, "planner"): PlannerConfig,
    (Config, "robot"): RobotConfig,
    (Config, "task"): TaskConfig,
    (Config, "observation"): ObservationConfig,
    (Config, "action"): ActionConfig,
    (Config, "safety"): SafetyConfig,
    (Config, "prompt"): PromptConfig,
    (Config, "feedback"): FeedbackConfig,
    (Config, "episode"): EpisodeConfig,
    (Config, "log"): LogConfig,
}


# Allowed values of enum-like fields. Unknown KEYS already raise in ``_build``; unknown
# VALUES used to fall back silently (``goal_feedback: alwyas`` meant "never").
_CHOICES: dict[tuple[str, str], tuple] = {
    ("action", "mode"): ("ee_delta", "ee_abs", "joint_delta", "joint_abs"),
    ("action", "rotation"): ("none", "yaw", "full"),
    ("action", "pos_unit"): ("mm", "cm", "m"),
    ("action", "ang_unit"): ("deg", "rad"),
    ("action", "gripper"): ("binary", "width"),
    ("action", "format"): ("text",),            # "tool" is not implemented
    ("episode", "goal_feedback"): ("never", "on_done", "always"),
    ("llm", "cache"): ("auto", "anthropic", "none"),
    ("llm", "cache_ttl"): ("5m", "1h"),
    ("robot", "backend"): ("isaac", "mock", "replay"),
    ("feedback", "level"): ("short", "full"),
    ("safety", "box_collision"): ("block", "warn", "off"),
}
_RENDERERS = ("raw", "grid", "axes", "ee_marker", "diff", "heatmap", "tile")


def validate(cfg: Config) -> Config:
    """Reject values that would silently do something else than written.
    Called by ``load_config`` and ``run_episode``; returns ``cfg`` for chaining."""
    errs: list[str] = []
    # YAML 1.1 reads a bare `off` as False (`box_collision: off`, `--set safety.box_collision=off`)
    if cfg.safety.box_collision is False:
        cfg.safety.box_collision = "off"
    for (sec, key), allowed in _CHOICES.items():
        v = getattr(getattr(cfg, sec), key)
        if v not in allowed:
            errs.append(f"{sec}.{key}={v!r} (allowed: {', '.join(allowed)})")
    bad = [r for r in cfg.observation.renderers if r not in _RENDERERS]
    if bad:
        errs.append(f"observation.renderers has unknown {bad} (known: {', '.join(_RENDERERS)})")
    for name, ok in (("action.max_chunk >= 1", cfg.action.max_chunk >= 1),
                     ("observation.size > 0", cfg.observation.size > 0),
                     ("observation.jpeg_quality in 1..100", 1 <= cfg.observation.jpeg_quality <= 100),
                     ("episode.max_turns >= 1", cfg.episode.max_turns >= 1),
                     ("episode.max_stops >= 1", cfg.episode.max_stops >= 1),
                     ("safety.max_step_m > 0", cfg.safety.max_step_m > 0),
                     ("safety.path_step_m > 0", cfg.safety.path_step_m > 0),
                     ("safety.box_clearance_m >= 0", cfg.safety.box_clearance_m >= 0),
                     ("observation.cameras non-empty", bool(cfg.observation.cameras))):
        if not ok:
            errs.append(name)
    if errs:
        raise ValueError("invalid config: " + "; ".join(errs))
    return cfg


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                dotenv: str | Path | None = ".env") -> Config:
    """``dotenv``: read relative to the CWD first, then the repo root — running from
    another directory must not silently lose the endpoint/key."""
    if dotenv:
        load_dotenv(dotenv)
        if not Path(dotenv).is_absolute():
            load_dotenv(REPO_ROOT / dotenv)
    data = Config().to_dict()
    if path:
        data = _deep_merge(data, _read_yaml_with_extends(Path(path).resolve()))
    for expr in overrides or []:
        keys, value = parse_override(expr)
        _set_path(data, keys, value)
    return validate(_build(Config, _expand_env(data)))
