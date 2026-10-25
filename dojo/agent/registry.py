"""The shared skill registry (J2; robot-harness issue #5): load, validate, select, render, resolve.

``skills/registry.yaml`` holds one entry per skill variant from either team (format in the file's header). The
agent's chooser builds the manual's skill section from it:

* ``Registry.select(strict=..., backends=...)``: strict runs keep only ``obs_needs.contract_only: true`` entries
  (BEFORE the prompt is built), the runtime keeps only backends it can call, and losers of a J5 comparison
  (``fallback_for``) are never prompted;
* ``render(entries)``: the manual's skill list (signature built from ``args`` + ``doc``);
* ``resolve(entry)``: the callable. ``dojo_skills``: ``module:function`` called as ``fn(sk, **args)``;
  ``robot_server``: ``skills/<name>/skill_robodojo.py:run`` under ``$HARNESS_ROOT`` (robot-harness), called as
  ``run(robot, **args)`` with a robot client (``harness/robot_client.py``) the caller provides.

Docs must not name tasks, layouts or seeds (P4); ``validate`` checks every doc and pattern against the public
RoboDojo config names.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REGISTRY = Path(__file__).resolve().parents[2] / "skills" / "registry.yaml"
BACKENDS = ("dojo_skills", "robot_server")
HELD_LOOK = ("wrist", "operator", False)
REQUIRED = ("name", "side", "doc", "args", "obs_needs", "backend", "entry")
# public RoboDojo config names (task/RoboDojo/config/*.yml): a doc naming one is routing by task (P4)
TASK_NAMES = (
    "align_blocks", "arrange_largest_number", "build_tower", "classify_objects", "classify_objects_by_language",
    "cover_blocks", "deposit_coin", "fasten_screws", "fill_egg_holder", "fill_pen_holder", "fold_clothes",
    "general_pickup", "hang_mugs", "imitate_sorting_sequence", "insert_key", "insert_tubes", "make_kong", "make_toast",
    "match_and_pick_from_conveyor", "organize_table", "pack_objects_into_box", "pick_from_conveyor_by_image",
    "play_Xylophone", "play_stacking_toy", "play_tic_tac_toe", "plug_in_charger", "pour_balls_into_vase",
    "pour_by_language", "pour_liquid_into_cup", "press_by_number", "push_T", "put_bottles_into_dustbin",
    "solve_equation", "sort_nesting_dolls_by_size", "stack_blocks", "stack_blocks_by_language", "stack_bowls",
    "store_laptop_and_headphones", "store_tools_in_toolbox", "swap_T", "swap_blocks", "sweep_blocks")


class RegistryError(ValueError):
    pass


@dataclass
class SkillEntry:
    name: str
    side: str
    doc: str
    args: dict
    obs_needs: dict
    backend: str
    entry: str
    selects_on: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    measured: list = field(default_factory=list)
    default: bool = True
    fallback_for: str | None = None

    @property
    def contract_only(self) -> bool:
        return bool(self.obs_needs.get("contract_only"))

    def signature(self) -> str:
        parts = []
        for a, spec in self.args.items():
            t = spec.get("type", "any") if isinstance(spec, dict) else str(spec)
            if isinstance(spec, dict) and "default" in spec:
                d = spec["default"]
                d = "null" if d is None else (str(d).lower() if isinstance(d, bool) else d)
                parts.append(f"{a}: {t} = {d}")
            else:
                parts.append(f"{a}: {t}")
        return f"{self.name}({', '.join(parts)})"


@dataclass
class Registry:
    version: int
    entries: list[SkillEntry]
    path: str = ""

    def by_name(self) -> dict[str, SkillEntry]:
        return {e.name: e for e in self.entries}

    def select(self, strict: bool = False, backends=BACKENDS, sides=None) -> list[SkillEntry]:
        """The entries a run may prompt: strict -> contract_only only; callable backends; J5 losers dropped."""
        out = []
        for e in self.entries:
            if strict and not e.contract_only:
                continue
            if e.backend not in backends or (sides is not None and e.side not in sides):
                continue
            if e.fallback_for:
                continue
            out.append(e)
        return out


def load(path: str | Path | None = None) -> Registry:
    p = Path(path or os.environ.get("DOJO_SKILL_REGISTRY") or REGISTRY)
    raw = yaml.safe_load(p.read_text()) or {}
    entries = []
    for i, r in enumerate(raw.get("skills") or []):
        miss = [k for k in REQUIRED if k not in r]
        if miss:
            raise RegistryError(f"{p}: skill #{i} ({r.get('name')}) lacks {miss}")
        entries.append(SkillEntry(
            name=str(r["name"]), side=str(r["side"]), doc=" ".join(str(r["doc"]).split()), args=dict(r["args"] or {}),
            obs_needs=dict(r["obs_needs"] or {}), backend=str(r["backend"]), entry=str(r["entry"]),
            selects_on=list(r.get("selects_on") or []), inputs=dict(r.get("inputs") or {}),
            measured=list(r.get("measured") or []), default=bool(r.get("default", True)),
            fallback_for=r.get("fallback_for")))
    reg = Registry(int(raw.get("version", 1)), entries, str(p))
    validate(reg)
    return reg


def validate(reg: Registry) -> None:
    names = [e.name for e in reg.entries]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise RegistryError(f"duplicate skill names {sorted(dup)} (one entry per variant, each with its own name)")
    task_re = re.compile(r"\b(" + "|".join(re.escape(t) for t in TASK_NAMES) + r")(_random)?\b")
    for e in reg.entries:
        if e.backend not in BACKENDS:
            raise RegistryError(f"{e.name}: backend {e.backend!r} not in {BACKENDS}")
        if e.obs_needs.get("held_look", False) not in HELD_LOOK:
            raise RegistryError(f"{e.name}: held_look must be one of wrist / operator / false")
        if e.side not in ("dojo", "harness"):
            raise RegistryError(f"{e.name}: side must be dojo or harness")
        for text in [e.doc] + [str(s) for s in e.selects_on]:
            m = task_re.search(text)
            if m:
                raise RegistryError(f"{e.name}: names the task {m.group(0)!r} (P4: effects, args, patterns only)")
        if e.fallback_for and e.fallback_for not in names:
            raise RegistryError(f"{e.name}: fallback_for {e.fallback_for!r} is not a registered skill")


def render(entries: list[SkillEntry]) -> str:
    """The manual's skill list."""
    return "\n".join(f"- `{e.signature()}`\n  {e.doc}" for e in entries)


def resolve(e: SkillEntry):
    """The callable behind an entry: ``fn(sk, **args)`` (dojo_skills) or ``run(robot, **args)`` (robot_server)."""
    mod, _, fn = e.entry.partition(":")
    if e.backend == "dojo_skills":
        return getattr(importlib.import_module(mod), fn)
    root = Path(os.environ.get("HARNESS_ROOT", "~/robot-harness")).expanduser()
    f = Path(mod) if Path(mod).is_absolute() else root / mod
    if not f.is_file():
        raise RegistryError(f"{e.name}: {f} not found (set HARNESS_ROOT to the robot-harness checkout)")
    spec = importlib.util.spec_from_file_location(f"harness_skill_{e.name}", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return getattr(m, fn or "run")


__all__ = ["Registry", "SkillEntry", "RegistryError", "load", "validate", "render", "resolve", "REGISTRY"]
