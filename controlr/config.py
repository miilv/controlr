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

import yaml


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    base_url: str = "${OMNIROUTE_BASE_URL}"
    api_key_env: str = "OMNIROUTE_API_KEY"
    model: str = "claude/claude-sonnet-5"
    max_tokens: int = 400
    temperature: float | None = None
    extra_body: dict = field(default_factory=dict)   # passed through verbatim (e.g. reasoning_effort)
    cache: str = "auto"            # auto | anthropic | none   (auto: anthropic markers for claude routes)
    cache_ttl: str = "5m"          # 5m | 1h  (anthropic marker ttl)
    early_stop: bool = True        # close the stream once the reply grammar is complete
    timeout_s: float = 180.0
    max_retries: int = 3


@dataclass
class PlannerConfig:
    enabled: bool = True
    model: str = "claude/claude-opus-5-5"
    extra_body: dict = field(default_factory=lambda: {"reasoning_effort": "xhigh"})
    max_tokens: int = 16000
    include_in_context: bool = True   # plan text goes into the first user turn of the control transcript
    prompt: str = "planner_v0"


@dataclass
class RobotConfig:
    backend: str = "mujoco"         # mujoco | mock | replay | ur (later) | isaac (later)
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
    state_text: bool = True          # append proprioceptive state as text
    jpeg_quality: int = 90
    first_turn_size: int | None = None   # optional larger image on the first turn


@dataclass
class ActionConfig:
    mode: str = "ee_delta"           # ee_delta | ee_abs | joint_delta | joint_abs
    rotation: str = "none"           # none | yaw | full   (ee modes only)
    pos_unit: str = "mm"             # mm | cm | m
    ang_unit: str = "deg"            # deg | rad
    max_chunk: int = 1               # max MOVE lines per reply executed before the next observation
    gripper: str = "binary"          # binary (open/close) | width (mm)
    format: str = "text"             # text (MOVE/STATUS lines) | tool (later)


@dataclass
class SafetyConfig:
    max_step_m: float = 0.10         # max TCP translation per MOVE line
    max_step_rad: float = 0.5236     # max rotation / joint change per MOVE line (30 deg)
    workspace_margin_m: float = 0.0  # shrink RobotSpec workspace box by this much
    table_clearance_m: float = 0.005 # min TCP height above the table surface
    joint_margin_rad: float = 0.0873 # soft margin inside hard joint limits (5 deg)
    near_limit_rad: float = 0.2618   # warn the model when within this of a joint limit (15 deg)
    clamp: bool = True               # clamp to the envelope (True) or reject the action (False)
    max_tcp_speed_m_s: float = 0.15  # execution speed
    contact_force_stop_n: float = 80.0


@dataclass
class PromptConfig:
    system: str = "system_v0"        # template name in controlr/prompts/
    fewshot: str | None = None       # path to a recorded run / demo file injected into the cached prefix
    extra_rules: list[str] = field(default_factory=list)


@dataclass
class EpisodeConfig:
    max_turns: int = 40
    max_parse_errors: int = 3        # consecutive unparsable replies -> end episode
    goal_feedback: str = "on_done"   # never | on_done | always   (what goal info the model sees)
    trust_done: bool = False         # end on DONE even if the goal check fails
    end_on_fail: bool = True


@dataclass
class LogConfig:
    root: str = "runs"
    save_images: bool = True


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
    (Config, "episode"): EpisodeConfig,
    (Config, "log"): LogConfig,
}


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                dotenv: str | Path | None = ".env") -> Config:
    if dotenv:
        load_dotenv(dotenv)
    data = Config().to_dict()
    if path:
        data = _deep_merge(data, _read_yaml_with_extends(Path(path).resolve()))
    for expr in overrides or []:
        keys, value = parse_override(expr)
        _set_path(data, keys, value)
    return _build(Config, _expand_env(data))
