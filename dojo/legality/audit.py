"""Legality audit: a transitive call-graph scan and an instruction sentinel, one tool for every tree (7 Nov 2026).

Static scan. From each entry point ("module:qualname"), the repository-local call graph is followed transitively:
direct calls, module-alias calls, imports inside functions, function references passed as values, ``self.m``, and any
other ``x.m(...)`` over-approximated to EVERY indexed method named ``m`` (minus the profile's ``detached`` modules,
whose objects a scanned function never holds). Every reachable function is scanned, docstrings and all strings
included, with no length exemption, for:

* a task name (``task_re``: snake, space, dash and CamelCase forms of every public task id);
* a read of the instruction or the task id (``.instruction``, ``.task``, ``getattr(..., "instruction")``), any
  non-literal ``getattr`` / ``setattr`` / ``hasattr``, ``__dict__`` (except a literal-key cache), ``vars()``;
* task data: ``tasks.json``, ``task/RoboDojo/config``, ``task/RoboDojo/tasks``, ``Eval_Layout``, ``Assets/Traj``,
  ``.settled.json``, ``layout_file``; checker state (``reward_manager``, ``func_parser``, ``take_action_cnt``,
  ``success``-flag arrays read off an env); task-vocabulary symbols (``VOCABULARIES``, ``perception_vocabulary``,
  ``task_map``, ...); imports of task / checker modules.

A profile pins the few legitimate exceptions, each by its exact source line: carriers (code that hands the episode's
text on without reading it) and closed reflective dispatchers. A pinned line that changes is itself a violation.

Each violation carries the call path from its entry point, so a finding in a deep helper says how it is reached.

Runtime sentinel. ``InstructionSentinel`` is a ``str`` whose every Python-level content access raises
``InstructionRead`` (a ``BaseException``: no ``except Exception`` in a skill can swallow it). Put it where the episode's
instruction lives and run the code; any read fails the run.

CLI::

    python -m dojo.legality.audit [--profile controlr-glib] [--root module:qualname ...] [--index PATH ...] [--json]

prints one line per violation (entry, function, why, call path) and exits 1 when there is any. With no ``--root`` the
profile's own entry points are used (for ``controlr-glib``: every implementation of the general skill library).
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------------------------------- task names
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "robodojo_instructions.json"


def task_names(path: str | Path | None = None) -> list[str]:
    """The public task ids (``_random`` twins folded): from a fixture holding ``{"instructions": {task: [...]}}``."""
    d = json.loads(Path(path or DEFAULT_FIXTURE).read_text())
    return sorted({t.replace("_random", "") for t in (d.get("instructions") or d)})


def _variants(name: str) -> list[str]:
    parts = [p for p in name.split("_") if p]
    camel = "".join(p[:1].upper() + p[1:] for p in parts)
    return sorted({name, " ".join(parts), "-".join(parts), camel})


def task_re(names=None) -> re.Pattern:
    """One regex matching every task id in its snake, space, dash and CamelCase forms (case-insensitive)."""
    names = task_names() if names is None else list(names)
    alts = sorted({re.escape(v) for n in names for v in _variants(n)}, key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9])(" + "|".join(alts) + r")(?![A-Za-z0-9])", re.I)


# ---------------------------------------------------------------------------------------------------- sentinel
class InstructionRead(BaseException):
    """Raised by any content access of an ``InstructionSentinel`` (a BaseException: not swallowed by skills)."""


def _boom(*a, **k):
    raise InstructionRead("the instruction was read")


class InstructionSentinel(str):
    """A ``str`` whose every Python-level content access raises ``InstructionRead``. (C-level fast paths that read the
    buffer directly, e.g. ``json.dumps``, are not intercepted: check what crosses a boundary for ``SENTINEL_TEXT``.)"""


SENTINEL_TEXT = "SENTINEL-c0ffee-the-instruction-must-not-be-read"
for _m in ("__str__", "__repr__", "__format__", "__len__", "__bool__", "__iter__", "__contains__", "__getitem__",
           "__eq__", "__ne__", "__lt__", "__gt__", "__hash__", "__add__", "__radd__", "__mod__", "__mul__", "__rmul__",
           "lower", "upper", "casefold", "split", "rsplit", "splitlines", "strip", "lstrip", "rstrip", "find", "rfind",
           "index", "rindex", "count", "replace", "startswith", "endswith", "encode", "join", "partition", "rpartition",
           "format", "format_map", "title", "capitalize", "swapcase", "isdigit", "isalpha", "isalnum", "islower",
           "isupper", "isspace", "zfill", "center", "ljust", "rjust", "expandtabs", "translate", "removeprefix",
           "removesuffix"):
    setattr(InstructionSentinel, _m, _boom)


def sentinel() -> InstructionSentinel:
    return InstructionSentinel(SENTINEL_TEXT)


# ---------------------------------------------------------------------------------------------------- profile
INSTRUCTION_ATTRS = ("instruction", "task")
TASK_SYMBOLS = {"TASK_CONFIGS", "perception_vocabulary", "VOCABULARIES", "vocabulary_candidates", "task_map",
                "catalog_vocabulary", "fixed_slot_labels", "public_constants", "LAYOUTS", "Eval_Layout", "load_task_config",
                "task_config", "tasks_json"}
TASK_ACCESSORS = {'task_map', 'label_groups', 'label_candidates', 'instance_range', 'fixed_slot_labels',
                  'scene_keys', 'public_constants', 'task_candidates', 'load_sets', '_placement_prior', 'quota_lists'}
# Concrete predicate identified by the partner hybrid re-check; provenance debt,
# not a general ban on physical containment predicates.
AUDITED_CHECKER_MODULES = {'dojo.skills.ported.skills.pour_p1.skill_robodojo'}
CHECKER_ATTRS = ("reward_manager", "func_parser", "take_action_cnt", "end_flag", "obs_manager")
DATA_PATHS = re.compile(r"tasks\.json|task/RoboDojo/(config|tasks)|Eval_Layout|Assets/Traj|\.settled\.json|layout_file|"
                        r"settled_index|(?:^|/)(?:Traj|logs?|runs)(?:/|$)")
TASK_MODULES = re.compile(r"(^|\.)(tasks?|checkers?|reward(_manager)?|func_parser|eval_layout)($|\.)", re.I)


@dataclass
class Profile:
    """What a tree indexes, which entry points it starts from, and the pinned exceptions."""
    name: str
    index: list = field(default_factory=list)             # .py files (absolute) the call graph may enter
    roots: list = field(default_factory=list)             # "module:qualname"
    detached: set = field(default_factory=set)            # modules whose methods an untyped x.m() never reaches
    carriers: dict = field(default_factory=dict)          # (module, qualname) -> the exact line that may touch the text
    reflective_ok: dict = field(default_factory=dict)     # (module, qualname) -> the exact reflective line
    root_dir: Path = ROOT
    scan_globals: bool = False
    report_unresolved: bool = False
    resources: list = field(default_factory=list)


def glib_roots() -> list[str]:
    """Every implementation of the general skill library, plus its own run / check / run_steps."""
    from dojo.skills import glib
    roots = sorted({s.impl for s in glib.REGISTRY.values()})
    return roots + ["dojo.skills.glib:run", "dojo.skills.glib:check", "dojo.skills.glib:run_steps"]


def controlr_glib() -> Profile:
    d = ROOT / "dojo"
    index = sorted(set((d / "skills").glob("*.py")) | {d / "skills" / "ported" / "__init__.py",
                                                       d / "skills" / "ported" / "harness_robot.py",
                                                       d / "runtime" / "belief.py", d / "runtime" / "estimate.py",
                                                       d / "runtime" / "hybrid.py"})
    return Profile(
        name="controlr-glib", index=index, roots=glib_roots(),
        # planner-side and verifier modules: a direct call or import into them is found, but no scanned function holds
        # one of their objects (``glib.run`` switches the twin verifier off for every verb: tests/dojo/test_glib_audit.py)
        detached={"dojo.skills.planner", "dojo.skills.routes", "dojo.skills.model_registry", "dojo.skills.model_planner",
                  "dojo.skills.g3", "dojo.skills.verify", "dojo.runtime.hybrid"},
        carriers={
            ("dojo.skills.skills", "GTProvider.refresh"): "WorldState(task=self.task, instruction=self.instruction,",
            ("dojo.runtime.belief", "RuntimeProvider.refresh"):
                "return WorldState(task=self.task, instruction=self.instruction, objects=objects, arms=arms,",
            ("dojo.skills.ported.harness_robot", "HarnessRobot.cmd_obs"):
                '"instruction": "" if self.hide_instruction else str(world.instruction or "")',
            ("dojo.skills.env", "RunnerEnv.instruction"): "return str(self.r.env.obs_manager.instruction[self.i])",
            # the dev runner's step counter and episode-end flag: what every official observation reports anyway
            # (steps_used, ended), not checker state
            ("dojo.skills.env", "RunnerEnv.steps_used"): "return int(self.r.env.take_action_cnt[self.i])",
            ("dojo.skills.env", "RunnerEnv.step"): "return bool(self.r.env.end_flag[self.i])",
        },
        reflective_ok={
            ("dojo.skills.glib", "impl"): "return getattr(importlib.import_module(mod), fn)",
            ("dojo.skills.ported.harness_robot", "HarnessRobot.call"):
                'rep = {"ok": True, **getattr(self, "cmd_" + cmd)(kw)}',
        })


def controlr_inference():
    from .profiles import controlr_inference as make
    return make()


PROFILES = {"controlr-glib": controlr_glib, "controlr-inference": controlr_inference}


# ---------------------------------------------------------------------------------------------------- index
def _modname(p: Path, root: Path) -> str:
    rel = Path(p).resolve().relative_to(root).with_suffix("")
    return ".".join(rel.parts[:-1] if rel.name == "__init__" else rel.parts)


def _imports(body, m, package=False) -> dict:
    out = {}
    for n in body:
        if isinstance(n, ast.Import):
            for a in n.names:
                out[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ''
            if n.level:
                parts = m.split('.') if package else m.split('.')[:-1]
                base = '.'.join(parts[:len(parts) - n.level + 1] + ([base] if base else []))
            for a in n.names:
                out[a.asname or a.name] = (base, a.name)
    return out


class Index:
    def __init__(self, profile: Profile):
        self.p = profile
        self.funcs, self.by_name, self.methods, self.imports, self.mods = {}, {}, {}, {}, {}
        self.packages = {_modname(p, profile.root_dir) for p in profile.index if Path(p).name == '__init__.py'}
        for path in profile.index:
            self.add_module(_modname(path, profile.root_dir), Path(path).read_text())

    def add_module(self, m: str, src: str) -> None:
        tree = ast.parse(src)
        self.mods[m] = (tree, src.splitlines())
        self.by_name[m] = {}
        self.imports[m] = _imports(tree.body, m, m in self.packages)
        if self.p.scan_globals:
            def global_node(n):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return None
                if isinstance(n, ast.ClassDef):
                    n = copy.copy(n)
                    n.body = [child for item in n.body if (child := global_node(item)) is not None]
                return n
            body = [child for item in tree.body if (child := global_node(item)) is not None]
            synthetic = ast.FunctionDef(name='<module>', args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                                        body=body, decorator_list=[], lineno=1, end_lineno=max(1, len(src.splitlines())))
            self.funcs[(m, '<module>')] = synthetic
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.funcs[(m, n.name)] = n
                self.by_name[m][n.name] = n.name
            elif isinstance(n, ast.ClassDef):
                for f in n.body:
                    if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        q = f"{n.name}.{f.name}"
                        self.funcs[(m, q)] = f
                        if m not in self.p.detached:
                            self.methods.setdefault(f.name, []).append((m, q))
            elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Lambda):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        self.funcs[(m, t.id)] = n.value
                        self.by_name[m][t.id] = t.id

    def resolve(self, m, name, local):
        imp = {**self.imports.get(m, {}), **local}
        if name in self.by_name.get(m, {}):
            return (m, name)
        tgt = imp.get(name)
        if isinstance(tgt, tuple) and tgt in self.funcs:
            return tgt
        return None

    def callees(self, key):
        m, q = key
        node = self.funcs[key]
        local = {}
        for n in ast.walk(node):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                local.update(_imports([n], m, m in self.packages))
        imp = {**self.imports.get(m, {}), **local}
        out = set()
        cls = q.split(".")[0] if "." in q else None
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                r = self.resolve(m, n.id, local)
                if r:
                    out.add(r)
            elif isinstance(n, ast.Attribute):
                base = n.value
                if isinstance(base, ast.Name):
                    tgt = imp.get(base.id)
                    mod = tgt if isinstance(tgt, str) else (f"{tgt[0]}.{tgt[1]}" if isinstance(tgt, tuple) else None)
                    if mod in self.mods:
                        resolved = self.resolve(mod, n.attr, {})
                        if resolved:
                            out.add(resolved)
                            continue
                    if base.id == "self" and cls and (m, f"{cls}.{n.attr}") in self.funcs:
                        out.add((m, f"{cls}.{n.attr}"))
                        continue
                for mk in self.methods.get(n.attr, []):
                    out.add(mk)
        if self.p.scan_globals:
            out.add((m, '<module>'))
            for target in imp.values():
                modules = [target] if isinstance(target, str) else [target[0], '.'.join(target)]
                out.update((mod, '<module>') for mod in modules if (mod, '<module>') in self.funcs)
        out.discard(key)
        return out

    def source(self, key) -> str:
        m, _ = key
        node = self.funcs[key]
        lines = self.mods[m][1]
        return "\n".join(lines[node.lineno - 1: getattr(node, "end_lineno", node.lineno)])


# ---------------------------------------------------------------------------------------------------- scan
@dataclass
class Violation:
    root: str
    func: str
    why: str
    path: list

    def line(self) -> str:
        return f"{self.root}  {self.func}: {self.why}   [via {' -> '.join(self.path)}]"


def scan_function(idx: Index, key, rx: re.Pattern) -> list[str]:
    """The legality problems in one function's own body (no recursion)."""
    p = idx.p
    node = idx.funcs[key]
    src = idx.source(key)
    # a carrier is exempt on its pinned line only (the line that hands the text or counter on), nowhere else
    pinned = set()
    if key in p.carriers:
        for i, ln in enumerate(src.splitlines()):
            if p.carriers[key].strip() == ln.strip():
                pinned.add(node.lineno + i)
    reflective_lines = {node.lineno + i for i, line in enumerate(src.splitlines())
                        if key in p.reflective_ok and line.strip() == p.reflective_ok[key].strip()}
    out = []
    if len(pinned) > 1:
        out.append('pinned carrier line is ambiguous: re-audit it')
        pinned.clear()
    if len(reflective_lines) > 1:
        out.append('pinned reflective line is ambiguous: re-audit it')
        reflective_lines.clear()
    if key in p.carriers and not pinned:
        out.append("pinned carrier line changed: re-audit it")
    if key in p.reflective_ok and not reflective_lines:
        out.append("pinned reflective line changed: re-audit it")
    cache_dicts = set()
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("setdefault", "pop", "get")
                and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "__dict__" and n.args
                and isinstance(n.args[0], ast.Constant) and n.args[0].value not in INSTRUCTION_ATTRS):
            cache_dicts.add(id(n.func.value))
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            # Pin a multiline literal to the actual offending line, never its
            # opening triple quote. Inventory every match, not only the first.
            def literal_line(offset):
                return min(getattr(n, 'end_lineno', n.lineno),
                           n.lineno + n.value[:offset].count('\n'))
            for hit in rx.finditer(n.value):
                out.append(f"task name in a string: {hit.group(0)!r} (line {literal_line(hit.start())})")
            for hit in DATA_PATHS.finditer(n.value):
                out.append(f"task data path: {hit.group(0)!r} (line {literal_line(hit.start())})")
        if isinstance(n, ast.Attribute):
            if n.attr in INSTRUCTION_ATTRS and n.lineno not in pinned:
                out.append(f".{n.attr} read (line {n.lineno})")
            if n.attr == "__dict__" and id(n) not in cache_dicts:
                out.append(f"__dict__ (line {n.lineno})")
            if n.attr in CHECKER_ATTRS and n.lineno not in pinned:
                out.append(f"checker / env state .{n.attr} (line {n.lineno})")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id == "vars":
                out.append(f"vars() (line {n.lineno})")
            if n.func.id in ("getattr", "setattr", "hasattr", "delattr") and len(n.args) >= 2:
                a = n.args[1]
                literal = isinstance(a, ast.Constant) and isinstance(a.value, str)
                if not literal and n.lineno not in reflective_lines:
                    out.append(f"non-literal {n.func.id} (line {n.lineno})")
                elif literal and a.value in INSTRUCTION_ATTRS + CHECKER_ATTRS:
                    out.append(f"{n.func.id}(..., {a.value!r}) (line {n.lineno})")
        if isinstance(n, ast.Name) and n.id in TASK_SYMBOLS | TASK_ACCESSORS:
            out.append(f"task symbol {n.id} (line {n.lineno})")
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
            for mod in mods:
                if TASK_MODULES.search(mod):
                    out.append(f"imports task / checker module {mod} (line {n.lineno})")
    if p.report_unresolved:
        local = _imports([x for x in ast.walk(node) if isinstance(x, (ast.Import, ast.ImportFrom))], key[0], key[0] in idx.packages)
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                loader = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else ''
                if loader in {'spec_from_file_location', 'exec_module', 'import_module', 'SourceFileLoader'} or (
                        loader == 'run_path' and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'runpy'):
                    out.append(f'dynamic code loader {ast.unparse(n.func)} (line {n.lineno})')
                known = isinstance(n.func, ast.Name) and idx.resolve(key[0], n.func.id, local)
                if isinstance(n.func, ast.Attribute):
                    known = bool(idx.methods.get(n.func.attr))
                    if isinstance(n.func.value, ast.Name):
                        target = {**idx.imports.get(key[0], {}), **local}.get(n.func.value.id)
                        mod = target if isinstance(target, str) else '.'.join(target) if isinstance(target, tuple) else None
                        known = known or (mod, n.func.attr) in idx.funcs
                if not known:
                    out.append(f"unresolved call {ast.unparse(n.func)} (line {n.lineno})")
    if p.scan_globals:
        if key[0] in AUDITED_CHECKER_MODULES:
            if getattr(node, 'name', '') == '_in_target':
                out.append(f'audited checker-shaped acceptance predicate (line {node.lineno})')
            for item in ast.walk(node):
                if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'verified' for t in item.targets):
                    out.append(f'audited checker predicate drives verified outcome (line {item.lineno})')
        if getattr(node, 'name', '') in {'tower_from_boards', 'declare_kong', 'copy_demonstration',
                                         'play_board_game', 'row_by_size', 'align_in_row', 'sweep_into'}:
            out.append(f'benchmark composite (line {node.lineno})')
        # Provenance comments matter; mask definitions excluded from a synthetic
        # module so unreachable function comments do not become global evidence.
        lines = set(range(node.lineno, node.end_lineno + 1))
        if key[1] == '<module>':
            for child in idx.mods[key[0]][0].body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    lines.difference_update(range(child.lineno, child.end_lineno + 1))
        for line in sorted(lines):
            if line > len(idx.mods[key[0]][1]):
                continue
            text = idx.mods[key[0]][1][line - 1].strip()
            if re.search(r'\b(?:watch_relocate_task|key_scope_task|sam_task_routes|sam_task_banks|ported_plan|ported_routes|ported_\w+|new_routes)\b', text):
                out.append(f"task switch / policy route (line {line})")
            if re.search(r'checker|scorer', text, re.I):
                out.append(f"checker provenance (line {line})")
                if re.search(r'\d', text):
                    out.append(f"checker-shaped tolerance (line {line})")
            if re.search(r'\b(?:episode_over|env_end|success_latch|termination_feedback)\b', text):
                out.append(f'termination feedback surface (line {line})')
            if 'CATEGORY_ATTRIBUTES' in text or ('number_card' in text and 'idx' in text):
                out.append(f"category-index identity rule (line {line})")
    return out


def _key(s: str):
    m, q = s.split(":", 1)
    return (m, q)


def audit_callgraph(roots=None, profile: Profile | None = None, names=None) -> list[Violation]:
    """Every violation reachable from ``roots`` ("module:qualname"), each with its call path. A root that is not in
    the index is itself a violation (the audit cannot vouch for it)."""
    p = profile or controlr_glib()
    idx = Index(p)
    rx = task_re(names)
    return _walk(idx, [r for r in (roots or p.roots)], rx)


def _walk(idx: Index, roots, rx) -> list[Violation]:
    out = {}
    def record(v):
        out.setdefault((v.func, v.why), v)
    graph_cache, scan_cache = {}, {}
    for r in roots:
        k = _key(r)
        if k not in idx.funcs:
            k = idx.resolve(k[0], k[1], {}) or k
        if k not in idx.funcs:
            record(Violation(r, r, "entry point not in the index (cannot be audited)", [r]))
            continue
        parent = {k: None}
        q = deque([k])
        while q:
            cur = q.popleft()
            if cur not in graph_cache:
                graph_cache[cur] = sorted(idx.callees(cur))
            for c in graph_cache[cur]:
                if c not in parent:
                    parent[c] = cur
                    q.append(c)
        for m in sorted({k_[0] for k_ in parent}):            # module-level imports of every module the graph enters
            first = next(k_ for k_ in parent if k_[0] == m)
            for n in idx.mods[m][0].body:
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    for mod in ([a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]):
                        if TASK_MODULES.search(mod):
                            chain, x = [], first
                            while x is not None:
                                chain.append(f"{x[0]}:{x[1]}")
                                x = parent[x]
                            record(Violation(r, f"{m}:<module>", f"imports task / checker module {mod} "
                                                 f"(line {n.lineno})", list(reversed(chain))))
        for key in parent:
            if key not in scan_cache:
                scan_cache[key] = scan_function(idx, key, rx)
            probs = [why for why in scan_cache[key] if (f"{key[0]}:{key[1]}", why) not in out]
            if not probs:
                continue
            chain, x = [], key
            while x is not None:
                chain.append(f"{x[0]}:{x[1]}")
                x = parent[x]
            for why in probs:
                record(Violation(r, f"{key[0]}:{key[1]}", why, list(reversed(chain))))
    for resource in idx.p.resources:
        rel = str(Path(resource).relative_to(idx.p.root_dir))
        for line, text in enumerate(Path(resource).read_text().splitlines(), 1):
            whys = []
            hit = rx.search(text)
            if hit:
                whys.append(f'task name in a string: {hit.group(0)!r} (line {line})')
            if DATA_PATHS.search(text):
                whys.append(f'task data path: {text[:60]!r} (line {line})')
            if re.search(r'checker|scorer', text, re.I):
                whys.append(f'checker provenance (line {line})')
            for why in whys:
                record(Violation('resource:' + rel, rel + ':<resource>', why, ['resource:' + rel]))
    return list(out.values())


def reachable(roots=None, profile: Profile | None = None) -> set:
    """The (module, qualname) set the audit scans from ``roots``."""
    p = profile or controlr_glib()
    idx = Index(p)
    seen, todo = set(), [idx.resolve(*_key(r), {}) or _key(r) for r in (roots or p.roots)]
    while todo:
        k = todo.pop()
        if k in seen or k not in idx.funcs:
            continue
        seen.add(k)
        todo.extend(idx.callees(k) - seen)
    return seen


def audit_source(source: str, name: str, profile: Profile | None = None, names=None) -> list[Violation]:
    """Audit a function given as source text (a learned verb): its body and everything it reaches in the index."""
    p = profile or controlr_glib()
    idx = Index(p)
    idx.add_module("__learned__", source)
    rx = task_re(names)
    roots = [f"__learned__:{name}"]
    if ("__learned__", name) not in idx.funcs:
        return [Violation(roots[0], roots[0], f"no function {name!r} in the source", roots)]
    return _walk(idx, roots, rx)


# ---------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="transitive legality audit (call graph from entry points)")
    ap.add_argument("--profile", default="controlr-inference", choices=sorted(PROFILES))
    ap.add_argument("--root", action="append", default=[], help="entry point module:qualname (default: the profile's)")
    ap.add_argument("--index", action="append", default=[], help="extra .py files or directories to index")
    ap.add_argument("--tasks", default=None, help="fixture with the public task ids (default: tests/fixtures/...)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    p = PROFILES[a.profile]()
    for x in a.index:
        x = Path(x).resolve()
        p.index = sorted(set(p.index) | (set(x.rglob("*.py")) if x.is_dir() else {x}))
    vs = audit_callgraph(a.root or None, p, task_names(a.tasks) if a.tasks else None)
    n = len(reachable(a.root or None, p))
    if a.json:
        print(json.dumps({"profile": p.name, "reachable": n, "violations": [v.__dict__ for v in vs]}, indent=1))
    else:
        for v in vs:
            print(v.line())
        print(f"{p.name}: {n} functions reachable, {len(vs)} violations")
    return 1 if vs else 0


if __name__ == "__main__":
    sys.exit(main())
