"""The shared J2 skill registry (schema ``j2-registry/1``): load, validate, select, render, resolve.

The shared truth is robot-harness's ``skills/registry.yaml`` (spec ``docs/REGISTRY.md``, validator
``harness/registry.py``, both owned by Mikhail's side). The ilia side lives in ``skills/registry.yaml`` here, in the
same shape, until J4 moves it under robot-harness's ``# ---- ilia side ----`` marker. This loader reads either file
unchanged (a pinned copy of robot-harness's is in ``tests/fixtures/``) and validates them the way
``harness/registry.py`` does, minus the checks that need robot-harness's tree: entry files, ``results/ledger.jsonl``,
``coord/LANES.md``.

* ``select(entries, strict, benchmark, backends, candidates)``: ``harness/registry.filter_strict`` semantics
  (``obs_needs.contract_only is True``, one benchmark, no ``status: candidate`` entries unless ``candidates=True``),
  plus the backends this runtime can call. J5 losers (``fallback``) are not prompted. A candidate is a new version
  not promoted yet (robot-harness ``d918f85`` / ``a4f4d5e``); no status = promoted, and no chooser offers a candidate
  unless the caller opts in, whatever its backend.
* ``contract_only`` is derived from ``obs_needs.reads``: a ``true`` that reads a non-contract token is an error,
  as in ``harness/registry.validate``.
* ``chooser_docs(entries)`` is the chooser block: name + args + doc only, byte for byte
  ``harness/registry.chooser_docs``. measured / stats / notes are for people, and never reach the prompt.
* ``resolve(entry)``: ``dojo_skills`` ``package.module:func`` is called as ``fn(sk, **args)``; ``robot_server``
  ``skills/<dir>/<file>.py:run`` under ``$HARNESS_ROOT`` (a robot-harness checkout) is called as
  ``run(robot, **args)``, with ``harness/robot_client_v2.Robot(server_id)`` as ``robot``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

REGISTRY = Path(__file__).resolve().parents[2] / "skills" / "registry.yaml"
SCHEMA = "j2-registry/1"
SIDES = ("misha", "ilia")
BENCHMARKS = ("robodojo", "libero_pro")
BACKENDS = ("robot_server", "dojo_skills")
HELD = ("wrist", "operator", False)
TIERS = ("A_gt", "B_camera", "official")
STATUSES = ("candidate", "promoted")         # absent = promoted (harness/registry.STATUSES)
REQUIRED = ("name", "side", "benchmark", "doc", "args", "obs_needs", "backend", "entry", "measured")
# catalog_mesh: grasps / geometry from the PUBLIC asset catalogue mesh of the PERCEIVED asset at the PERCEIVED pose
# (contract; mesh_grasps = a mesh at a privileged pose or identity, non-contract), as in harness/registry.py (f7f9017)
CONTRACT_READS = {"joints", "flange", "grip_cmd", "instruction", "steps", "free_ik", "held_look", "object_args",
                  "catalog_mesh"}
NON_CONTRACT_READS = {"privileged_poses", "mesh_grasps", "contacts", "finger_meas", "depth", "server_internals"}
# the RoboDojo task table (harness/registry.LANES_TASKS); doc / args naming one, underscored or spaced, is P4
TASKS = (
    "arrange_largest_number", "fold_clothes", "hang_mugs", "make_toast", "pack_objects_into_box", "pour_liquid_into_cup",
    "push_T", "sort_nesting_dolls_by_size", "stack_blocks", "stack_bowls", "store_laptop_and_headphones", "sweep_blocks",
    "build_tower", "deposit_coin", "fasten_screws", "insert_key", "insert_tubes", "play_Xylophone", "plug_in_charger",
    "pour_balls_into_vase", "classify_objects", "fill_egg_holder", "fill_pen_holder", "make_kong", "organize_table",
    "play_stacking_toy", "play_tic_tac_toe", "put_bottles_into_dustbin", "cover_blocks", "imitate_sorting_sequence",
    "match_and_pick_from_conveyor", "press_by_number", "swap_T", "swap_blocks", "align_blocks",
    "classify_objects_by_language", "general_pickup", "pick_from_conveyor_by_image", "pour_by_language", "solve_equation",
    "stack_blocks_by_language", "store_tools_in_toolbox")
P4_FIXED = (
    (r"\bseeds?\b", "seed"), (r"\blayouts?\b", "layout"), (r"\bL[0-9]\b", "layout id"),
    (r"\blibero_[a-z0-9_]+", "LIBERO suite id"),
    (r"\bKITCHEN_SCENE\d*|\bLIVING_ROOM_SCENE\d*|\bSTUDY_SCENE\d*", "LIBERO scene id"),
    (r"\bt[0-9]{1,2}\b|\btask\s*#?\s*[0-9]+\b", "task id"),
    (r"Eval[_]Layout|Assets[/]Traj|_result\.json|\.bddl\b", "eval file"),     # [_] / [/]: no literal eval path here
)


class RegistryError(ValueError):
    pass


def is_candidate(e: dict) -> bool:
    """``harness/registry.is_candidate``: ``status: candidate`` (absent or ``promoted`` = selectable)."""
    return e.get("status") == "candidate"


def _p4_patterns():
    pats = []
    for t in TASKS:
        words = t.split("_")
        pats.append((re.compile(r"\b" + re.escape(t) + r"\b", re.I), f"task name {t}"))
        if len(words) > 1:
            pats.append((re.compile(r"\b" + r"[\s\-]+".join(map(re.escape, words)) + r"\b", re.I),
                         f"task name {t} (spaced)"))
    return pats + [(re.compile(rx), why) for rx, why in P4_FIXED]


P4 = _p4_patterns()


@dataclass
class SkillEntry:
    """One registry entry; ``raw`` is the mapping exactly as in the file."""
    raw: dict

    def __getattr__(self, k):
        raw = self.__dict__.get("raw") or {}
        if k in raw:
            return raw[k]
        raise AttributeError(k)

    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def contract_only(self) -> bool:
        return (self.raw.get("obs_needs") or {}).get("contract_only") is True

    @property
    def is_candidate(self) -> bool:
        """A new version not promoted yet (``status: candidate``): no chooser offers it unless the caller opts in."""
        return is_candidate(self.raw)

    @property
    def derived_contract_only(self) -> bool | None:
        reads = (self.raw.get("obs_needs") or {}).get("reads")
        return None if reads is None else set(reads) <= CONTRACT_READS


@dataclass
class Registry:
    schema: str
    entries: list[SkillEntry]
    path: str = ""

    def by_name(self) -> dict[str, SkillEntry]:
        return {e.name: e for e in self.entries}

    def select(self, strict: bool = False, benchmark: str | None = "robodojo", backends=BACKENDS,
               candidates: bool = False) -> list[SkillEntry]:
        """What a run may prompt: strict -> contract_only true (``filter_strict``); one benchmark; callable backends;
        no J5 losers; no ``status: candidate`` entry unless ``candidates=True`` (dev runs only, never a scored one)."""
        out = []
        for e in self.entries:
            if e.is_candidate and not candidates:
                continue
            if strict and not e.contract_only:
                continue
            if benchmark is not None and e.raw.get("benchmark") != benchmark:
                continue
            if e.raw.get("backend") not in backends or e.raw.get("fallback"):
                continue
            out.append(e)
        return out


def load(path: str | Path | None = None, check: bool = True) -> Registry:
    """``{schema, entries: [...]}`` (or a bare list, as ``harness/registry.load`` accepts)."""
    p = Path(path or os.environ.get("DOJO_SKILL_REGISTRY") or REGISTRY)
    data = yaml.safe_load(p.read_text())
    schema = SCHEMA
    if isinstance(data, dict):
        schema = str(data.get("schema", SCHEMA))
        data = data.get("entries")
    if not isinstance(data, list):
        raise RegistryError(f"{p}: expected a list of entries or a mapping with 'entries'")
    if schema != SCHEMA:
        raise RegistryError(f"{p}: schema {schema!r}, this loader reads {SCHEMA!r}")
    reg = Registry(schema, [SkillEntry(dict(e)) if isinstance(e, dict) else SkillEntry({"_bad": e}) for e in data],
                   str(p))
    if check:
        errors, _ = validate(reg.entries)
        if errors:
            raise RegistryError(f"{p}: " + "; ".join(errors[:8]) + (f" (+{len(errors) - 8} more)" if len(errors) > 8 else ""))
    return reg


def validate(entries: list[SkillEntry]) -> tuple[list[str], list[str]]:
    """(errors, warnings): ``harness/registry.validate``'s checks that need no robot-harness tree."""
    errors, warnings, names = [], [], {}
    for i, se in enumerate(entries):
        e = se.raw
        tag = f"entry {i} ({e.get('name', '?')})"
        if "_bad" in e:
            errors.append(f"{tag}: not a mapping")
            continue
        errors += [f"{tag}: missing field {k}" for k in REQUIRED if k not in e]
        name = e.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name or ""):
            errors.append(f"{tag}: name must be lower_snake_case")
        elif name in names:
            errors.append(f"{tag}: duplicate name (also entry {names[name]})")
        else:
            names[name] = i
        if name in TASKS:
            warnings.append(f"{tag}: the name equals a benchmark task name")
        for k, allowed in (("side", SIDES), ("benchmark", BENCHMARKS), ("backend", BACKENDS)):
            if e.get(k) not in allowed:
                errors.append(f"{tag}: {k} must be one of {allowed}")
        doc = e.get("doc")
        if not isinstance(doc, str) or not doc.strip():
            errors.append(f"{tag}: doc must be non-empty text")
            doc = ""
        else:
            n = len([ln for ln in doc.strip().splitlines() if ln.strip()])
            if not 1 <= n <= 6:
                errors.append(f"{tag}: doc has {n} lines (1-6 allowed)")
        args = e.get("args")
        if not isinstance(args, dict):
            errors.append(f"{tag}: args must be a mapping (name -> description)")
            args = {}
        visible = doc + "\n" + "\n".join(f"{k}: {v}" for k, v in args.items())
        for rx, why in P4:
            for m in rx.finditer(visible):
                errors.append(f"{tag}: P4 word in doc/args: {why}: {m.group(0)!r}")
        on = e.get("obs_needs")
        if not isinstance(on, dict):
            errors.append(f"{tag}: obs_needs must be a mapping")
            on = {}
        if not isinstance(on.get("contract_only"), bool):
            errors.append(f"{tag}: obs_needs.contract_only must be true or false")
        if on.get("held_look", None) not in HELD:
            errors.append(f"{tag}: obs_needs.held_look must be wrist, operator or false")
        reads = on.get("reads")
        if reads is not None:
            unknown = set(reads) - CONTRACT_READS - NON_CONTRACT_READS
            if unknown:
                errors.append(f"{tag}: obs_needs.reads has unknown tokens {sorted(unknown)}")
            if on.get("contract_only") is True and not set(reads) <= CONTRACT_READS:
                errors.append(f"{tag}: contract_only true but reads {sorted(set(reads) - CONTRACT_READS)}")
            if on.get("contract_only") is False and set(reads) <= CONTRACT_READS:
                warnings.append(f"{tag}: contract_only false although every read is a contract token")
            if on.get("held_look") and "held_look" not in reads:
                errors.append(f"{tag}: held_look set but 'held_look' not in reads")
        elif e.get("side") == "ilia":
            errors.append(f"{tag}: ilia-side entries must list obs_needs.reads (contract_only is derived from it)")
        entry = e.get("entry")
        if not (isinstance(entry, str) and ":" in entry):
            errors.append(f"{tag}: entry must be 'path/to/file.py:func' or 'package.module:func'")
        elif e.get("backend") == "dojo_skills" and entry.rsplit(":", 1)[0].endswith(".py"):
            errors.append(f"{tag}: dojo_skills entries are 'package.module:func'")
        meas = e.get("measured")
        if meas != "none":
            if not isinstance(meas, list):
                errors.append(f"{tag}: measured must be 'none' or a list")
                meas = []
            for m in meas:
                if not isinstance(m, dict) or not {"line", "tier", "success", "n"} <= set(m):
                    errors.append(f"{tag}: measured item needs line, tier, success, n: {m!r}")
                elif m["tier"] not in TIERS:
                    errors.append(f"{tag}: measured tier {m['tier']!r} not in {TIERS}")
        for k in ("stats", "inputs", "selects_on"):
            if k in e and not isinstance(e[k], (list, dict)):
                errors.append(f"{tag}: {k} must be a list or mapping")
        if "status" in e and e["status"] not in STATUSES:
            errors.append(f"{tag}: status must be one of {STATUSES}")
        if is_candidate(e) and e.get("default"):
            errors.append(f"{tag}: a candidate cannot be the default (promote it first)")
    return errors, warnings


def chooser_docs(entries: list[SkillEntry]) -> str:
    """The chooser block, identical to ``harness/registry.chooser_docs``: name, args and doc, nothing else."""
    blocks = []
    for se in entries:
        e = se.raw
        args = e.get("args") or {}
        lines = [f"### {e['name']}"] + (["args:"] + [f"  {k}: {v}" for k, v in args.items()] if args else ["args: none"])
        lines += ["doc:"] + ["  " + ln for ln in str(e.get("doc", "")).strip().splitlines()]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


render = chooser_docs


def resolve(se: SkillEntry):
    """``fn(sk, **args)`` (dojo_skills) or ``run(robot, **args)`` (robot_server, from ``$HARNESS_ROOT``)."""
    mod, _, fn = se.raw["entry"].rpartition(":")
    if se.raw["backend"] == "dojo_skills":
        return getattr(importlib.import_module(mod), fn)
    root = Path(os.environ.get("HARNESS_ROOT", "~/robot-harness")).expanduser()
    f = Path(mod) if Path(mod).is_absolute() else root / mod
    if not f.is_file():
        raise RegistryError(f"{se.name}: {f} not found (set HARNESS_ROOT to a robot-harness checkout)")
    spec = importlib.util.spec_from_file_location(f"harness_skill_{se.name}", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return getattr(m, fn or "run")


__all__ = ["Registry", "SkillEntry", "RegistryError", "load", "validate", "chooser_docs", "render", "resolve",
           "is_candidate", "REGISTRY", "SCHEMA", "STATUSES"]
