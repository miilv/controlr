"""Legality audit: a transitive call-graph scan and an instruction sentinel, one tool for every tree (7 Nov 2026).

Controlr additions over partner 1413f374: conservative inference/global inventory,
path-loaded helper roots (in profiles.py), unresolved callbacks with shadow-aware
numeric/record leaves, resources and provenance signals, unique exact-line pins,
and precise multiline evidence. See the 8 November convergence journal for the
complete semantic comparison. Both teams use this engine.

Static scan. From each entry point ("module:qualname"), the repository-local call graph is followed transitively:
direct calls, module-alias and relative imports (also inside functions), function references passed as values,
``getattr(x, "literal")``, and method calls under rapid type analysis: ``self.m`` resolves within self's class tree;
any other ``x.m`` reaches every indexed method named ``m`` of a class that may have instances, i.e. one the reached code
names (constructs or passes as a factory) or one the profile declares ``injected`` (built by the runtime outside the
graph and handed in), to a fixpoint. Naming a class also reaches its dunders (constructor, ``__call__``,
``__getattr__``, ...). Module globals (``=NAME``) are nodes reached where code names them, class bodies (``C.<body>``)
with the class, a module's docstring (``<doc>``) with any of its functions. A profile also links string-built
dispatchers to every target they can reach (``dispatch``: the dispatcher's line is pinned, the targets are edges),
declares the runtime's callbacks as further roots (``callbacks``), and may gate branches its validated configuration
turns off (``gates``: an ``if`` with that exact test is not followed; ``dojo.runtime.scored.validate`` enforces the
configuration). Every reachable node is scanned, strings, f-strings and concatenations (folded from their constant
parts) and docstrings included, with no length exemption, for:

* a task name (``task_re``: snake, space, dash and CamelCase forms of every public task id);
* a read of the instruction or the task id (``.instruction``, ``.task``, ``getattr(..., "instruction")``), any
  non-literal ``getattr`` / ``setattr`` / ``hasattr``, ``__dict__`` (except a literal-key cache), ``vars()``,
  introspection (``__doc__``, ``__globals__``, ``__closure__``, ``__code__``, ``__self__``, ``__func__``), a dynamic
  import (``spec_from_file_location``, ``exec_module``, ``import_module``, ``__import__``) that is not a pinned line;
* task data: ``tasks.json``, ``task/RoboDojo/config``, ``task/RoboDojo/tasks``, ``Eval_Layout``, ``Assets/Traj``,
  ``.settled.json``, ``layout_file``, and any data file name not in the profile's allowlist (``data_files``: name ->
  sha256 for the library's own data); checker state (``reward_manager``, ``func_parser``, ``take_action_cnt``,
  ``success``-flag arrays read off an env); task-vocabulary symbols (``VOCABULARIES``, ``perception_vocabulary``,
  ``task_map``, ...), also under an alias; imports of task / checker modules.

A profile pins the few legitimate exceptions, each by its exact source line: carriers (code that hands the episode's
text on without reading it: exempt on that line only) and closed reflective dispatchers. A pinned line or gate that
changes is itself a violation.

The type analysis is checked live (tests/dojo/test_hybrid.py): in a scored hybrid episode every library function that
runs inside ``glib.run`` must be in the reachable set, every library object in the live graph must be of a class the
analysis allows, and every callable stored on those objects must be reachable or a declared callback.

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

import copy
import argparse
import ast
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


def camel_split(t: str) -> str:
    """``PushTVerb`` -> ``Push T Verb`` (so an acronym-ended CamelCase task name inside a longer word still matches)."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", t)


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
TASK_ACCESSORS = {'task_map', 'label_groups', 'label_candidates', 'instance_range', 'fixed_slot_labels',
                  'scene_keys', 'public_constants', 'task_candidates', 'load_sets', '_placement_prior', 'quota_lists'}
# Concrete predicate identified by the partner hybrid re-check; provenance debt,
# not a general ban on physical containment predicates.
AUDITED_CHECKER_MODULES = {'dojo.skills.ported.skills.pour_p1.skill_robodojo'}
TASK_SYMBOLS = {"TASK_CONFIGS", "perception_vocabulary", "VOCABULARIES", "vocabulary_candidates", "task_map",
                "catalog_vocabulary", "fixed_slot_labels", "public_constants", "LAYOUTS", "Eval_Layout", "load_task_config",
                "task_config", "tasks_json"}
REFLECT_CALLS = ("eval", "exec", "compile", "globals", "locals", "__import__", "breakpoint")
REFLECT_ATTRS = ("__class__", "__subclasses__", "__bases__", "__mro__", "__builtins__", "__module__")
IDENTITY_KEYS = ("instruction", "task", "task_name", "layout_id", "layout_file", "episode_id")
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
    # dynamic dispatch, LINKED (not just allowed): (module, qualname) -> (the exact line, [targets "module:qualname"]).
    # The dispatcher's reflective line is allowed only because every target it can reach is an edge of the graph
    dispatch: dict = field(default_factory=dict)
    data_files: dict = field(default_factory=dict)        # data file name -> sha256: the only data a scanned path may name
    # branches switched off by the validated scored configuration: (module, qualname) -> {exact ``if`` test: why}. The
    # body of such an ``if`` is neither followed nor scanned (its ``else`` is); ``dojo.runtime.scored.validate`` refuses
    # any configuration that could turn one on, and tests/dojo/test_glib_audit.py checks the live provider's switches
    gates: dict = field(default_factory=dict)
    # rapid type analysis: an untyped ``x.m()`` reaches ``C.m`` only for a class C some reachable code names (constructs or
    # passes as a factory) or that the runtime constructs outside the graph and hands in: ``injected`` ("module:Class").
    # tests/dojo/test_glib_audit.py walks the live object graph of a hybrid episode and checks every library class in it
    # is injected or named by reachable code
    injected: list = field(default_factory=list)
    # functions the runtime hands the library as callbacks (a provider's ``log``): audited as further roots; the live
    # test lists every callable stored on the library's objects and checks each is reachable or one of these
    callbacks: list = field(default_factory=list)
    # a module docstring that names a function of its own module which happens to share a task's name: (module, name)
    # -> why. Allowed in that docstring only, and only while the audit finds the function unreachable
    doc_ok: dict = field(default_factory=dict)
    # the agent's side (profile controlr-hybrid-agent): the agent is the planner and may read the instruction text,
    # never the task id; flat imports (``import gen_tools``, a package dir on sys.path) map to their modules
    instruction_attrs: tuple = ("instruction", "task")
    identity_keys: tuple = ("instruction", "task", "task_name", "layout_id", "layout_file", "episode_id")
    aliases: dict = field(default_factory=dict)
    # lines that must stay in a function for the profile's reasoning to hold (a guard that narrows a dispatch)
    required_lines: dict = field(default_factory=dict)
    scan_globals: bool = False
    report_unresolved: bool = False
    resources: list = field(default_factory=list)
    root_dir: Path = ROOT


def glib_roots() -> list[str]:
    """Every implementation of the general skill library, plus its own run / check / run_steps."""
    from dojo.skills import glib
    roots = sorted({s.impl for s in glib.REGISTRY.values()})
    return roots + ["dojo.skills.glib:run", "dojo.skills.glib:check", "dojo.skills.glib:run_steps"]


def _sha(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def controlr_glib() -> Profile:
    d = ROOT / "dojo"
    index = sorted(set((d / "skills").rglob("*.py")) | set((d / "perception").glob("*.py"))
                   | {d / "runtime" / f for f in ("belief.py", "estimate.py", "hybrid.py", "bridge.py", "firewall.py",
                                                  "policy.py")})
    hr = "dojo.skills.ported.harness_robot"
    hr_src = (d / "skills" / "ported" / "harness_robot.py").read_text()
    cmds = sorted(set(re.findall(r"    def (cmd_[a-z_]+)\(", hr_src)))
    vendored = sorted(_modname(f, ROOT) + ":run" for f in (d / "skills" / "ported" / "skills").glob("*/skill*.py"))
    data = d / "skills" / "data"
    return Profile(
        name="controlr-glib", index=index, roots=glib_roots(),
        # planner-side and verifier modules: a direct call or import into them is found, but no scanned function holds
        # one of their objects (``glib.run`` switches the twin verifier off for every verb: tests/dojo/test_glib_audit.py)
        # (and a prefix covers a package): their methods are reached only through their class named in code, or a linked
        # dispatch, never through an untyped ``x.m()`` of the same name (e.g. ``Spec.call`` is not ``HarnessRobot.call``)
        detached={"dojo.skills.planner", "dojo.skills.routes", "dojo.skills.model_registry", "dojo.skills.model_planner",
                  "dojo.skills.g3", "dojo.skills.verify", "dojo.runtime.hybrid", "dojo.skills.ported.harness_robot",
                  "dojo.skills.ported.skills"},
        # the one carrier the library reaches: the production provider builds the WorldState that carries the text on.
        # (The test provider GTProvider, the runner env and the vendored-skill adapter are not reachable from the
        # library: no pin, so reaching them would be a violation.)
        carriers={
            ("dojo.runtime.belief", "RuntimeProvider.refresh"):
                "return WorldState(task=self.task, instruction=self.instruction, objects=objects, arms=arms,",
        },
        reflective_ok={("dojo.skills.glib", "impl"): "return getattr(importlib.import_module(mod), fn)"},
        dispatch={
            (hr, "HarnessRobot.call"): ('rep = {"ok": True, **getattr(self, "cmd_" + cmd)(kw)}',
                                        [f"{hr}:HarnessRobot.{c}" for c in cmds]),
            # the vendored skill loader: whatever reaches it reaches EVERY vendored skill file
            (hr, "HarnessRobot.cmd_skill"): ("spec = importlib.util.spec_from_file_location(", vendored),
        },
        # the library's own per-asset data (hash-pinned) and the intrinsic per-asset files of the public object catalogue
        gates={("dojo.runtime.belief", "RuntimeProvider.refresh"): {
            "self.mode == 'gt'": "simulator truth: scored MUST mode='perception'",
            "self.grip_check": "scored MUST grip_check='off'",
            "self.held_checker is not None": "scored MUST held_look='off'",
            "self.track_reader is not None": "set only when the planner is not hybrid (policy._setup)",
            "released and self.checker is not None and (not self.belief.held_labels())":
                "checker is None for hybrid (policy._setup); scored MUST resync=False",
            "self.watcher is not None and (not self.belief.held_labels())": "set only when not hybrid (policy._setup)",
            "self.a1 != 'off' and self.a1_client is not None": "scored MUST a1='off'"}},
        injected=["dojo.skills.skills:Skills", "dojo.runtime.belief:RuntimeProvider", "dojo.runtime.belief:KinematicBelief",
                  "dojo.runtime.bridge:Bridge"],
        # the runtime's callbacks into the library's world: the policy's log; planner "hybrid"'s measure hook
        # (glib.MEASURE), the provider's locator confirm, and the hook's locate they both call (review F10)
        callbacks=["dojo.runtime.policy:RuntimePolicy.log", "dojo.runtime.hybrid:_measure",
                   "dojo.runtime.hybrid:_locator_confirm", "dojo.runtime.hybrid:HybridApi.locate"],
        doc_ok={("dojo.skills.library", "build_tower"): "the module docstring lists the rules path's build_tower"},
        data_files={**{f.name: _sha(f) for f in sorted(data.glob("*.json"))},
                    **{n: "intrinsic asset file" for n in ("metadata.json", "description.json", "links.json", "geom.npz")},
                    # the public robot and camera description: named where constants were copied from it (arm roots,
                    # TCP offset, gripper scale, head camera pose; the RoboDojo commit they came from); not read at run
                    # time
                    **{n: "robot / camera description, not read" for n in ("dual_x5.yml", "robot_config.yml",
                                                                           "camera_config.yml", "robodojo_commit.txt")}})


def controlr_hybrid_agent() -> Profile:
    """The agent's side of planner "hybrid" (review F5): the hook's agent facade (``dojo.runtime.hybrid``) and the
    vendored agent (``dojo/agent/rdgen``: its agent, locator and learned-verb runner), on top of everything the library
    profile indexes. The agent is the planner: it may read the instruction text (the hook hands it over), never the
    task id, a checker, task data, or the runtime behind the facade."""
    base = controlr_glib()
    d = ROOT / "dojo"
    rd = d / "agent" / "rdgen"
    flat = {f.stem: f"dojo.agent.rdgen.{f.stem}" for f in rd.glob("*.py")}
    # api.prim reaches the strict-server adapter only through its PRIMITIVES guard: in this profile HarnessRobot.call
    # dispatches to those commands alone (never cmd_skill, the vendored per-task skills); the guard line and the
    # tuple are checked here and by the scan
    hy = (d / "runtime" / "hybrid.py").read_text()
    prims = next(ast.literal_eval(n.value) for n in ast.parse(hy).body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "PRIMITIVES" for t in n.targets))
    if "skill" in prims or "held_look" in prims:
        raise ValueError(f"hybrid.PRIMITIVES admits a skill command: {prims}")
    hr = "dojo.skills.ported.harness_robot"
    dispatch = dict(base.dispatch)
    dispatch[(hr, "HarnessRobot.call")] = (base.dispatch[(hr, "HarnessRobot.call")][0],
                                           [f"{hr}:HarnessRobot.cmd_{c}" for c in prims])
    dispatch.pop((hr, "HarnessRobot.cmd_skill"), None)
    reflective = dict(base.reflective_ok)
    # the hook's loader of the configured entry points (fixed by the scored config, fingerprinted)
    reflective[("dojo.runtime.hybrid", "load")] = "return getattr(importlib.import_module(mod), fn)"
    # the strict-server adapter's guard proxy: forwards every attribute but the simulator's truth (TruthDenied)
    reflective[(hr, "TruthGuard.__getattr__")] = 'return getattr(object.__getattribute__(self, "_env"), name)'
    # the cell worker's robot API: a fixed name list, linked (the robot methods a model's cell can call are audited)
    cp = "dojo.agent.rdgen.gen_cellproc"
    if (rd / "gen_cellproc.py").is_file():
        api_names = next((ast.literal_eval(n.value) for n in ast.parse((rd / "gen_cellproc.py").read_text()).body
                          if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ROBOT_API"
                                                               for t in n.targets)), ())
        tgt = []
        for f, cls in ((rd / "rdgen_hybrid.py", "HybridRobot"), (rd / "gen_tools.py", "Robot")):
            if f.is_file():
                meths = {b.name for c in ast.parse(f.read_text()).body if isinstance(c, ast.ClassDef) and c.name == cls
                         for b in c.body if isinstance(b, ast.FunctionDef)}
                tgt += [f"dojo.agent.rdgen.{f.stem}:{cls}.{n}" for n in api_names if n in meths]
        # rd-gen's later version takes the names from a constructor argument that only an explicit robot_api changes;
        # the hook's runner never passes one (checked here), so the served names stay ROBOT_API
        for f in rd.glob("*.py"):                         # an AST check over every ProcSession(...) call (A2)
            for n in ast.walk(ast.parse(f.read_text())):
                if isinstance(n, ast.Call) and (getattr(n.func, "id", None) == "ProcSession"
                                                or getattr(n.func, "attr", None) == "ProcSession"):
                    if any(k.arg == "robot_api" or k.arg is None for k in n.keywords) or len(n.args) > 1:
                        raise ValueError(f"{f.name}:{n.lineno}: ProcSession gets robot_api (or **kwargs / positional "
                                         f"extras): re-audit the served robot API")
        dispatch[(cp, "ProcSession.__init__")] = (
            ("self.api = [n for n in ROBOT_API if self.robot is not None and callable(getattr(self.robot, n, None))]",
             "self.api = [n for n in names if self.robot is not None and callable(getattr(self.robot, n, None))]",
             "self.sigs[n] = self._sig(getattr(self.robot, n))"),          # the signatures of that same list
            tgt)
        # serving one call from a model's cell: only a name in that same list reaches the robot
        dispatch[(cp, "ProcSession._serve")] = ("f = getattr(self.robot, fn) if fn in self.api else self.funcs[fn]", tgt)
    roots = [f"dojo.runtime.hybrid:HybridApi.{n}" for n in ("observe", "steps_left", "prior", "known", "scene",
                                                           "locate", "watch", "skills", "check", "run", "prim")]
    roots += ["dojo.runtime.hybrid:AgentApi.__init__", "dojo.runtime.hybrid:program"]
    roots += [f"dojo.agent.rdgen.rdgen_hybrid:{f}" for f in ("agent", "locate", "learned_runner")]
    return Profile(
        name="controlr-hybrid-agent", index=sorted(set(base.index) | set(rd.glob("*.py")) | {d / "runtime" / "policy.py"}),
        # harness_robot is not detached here: naming the class must not bring every command (methods by name, under
        # the type analysis; HarnessRobot.call by the narrowed dispatch above)
        roots=roots, detached=base.detached - {"dojo.runtime.hybrid", hr}, carriers=dict(base.carriers),
        reflective_ok=reflective, dispatch=dispatch,
        data_files={**base.data_files, "SOURCE.txt": "provenance note in the adapter's docstring, not read",
                    "block_config.json": "the relay block's settings record (keyguard), not task data"},
        required_lines={("dojo.runtime.hybrid", "HybridApi.prim"): "if cmd not in PRIMITIVES:"},
        gates=dict(base.gates), injected=list(base.injected) + ["dojo.runtime.hybrid:HybridApi",
                                                                "dojo.runtime.hybrid:AgentApi"],
        callbacks=list(base.callbacks), doc_ok=dict(base.doc_ok), instruction_attrs=("task",),
        identity_keys=("task", "task_name", "layout_id", "layout_file", "episode_id"), aliases=flat)


def controlr_inference():
    from .profiles import controlr_inference as make
    return make()


PROFILES = {"controlr-glib": controlr_glib, "controlr-hybrid-agent": controlr_hybrid_agent}
if (Path(__file__).parent / "profiles.py").exists():         # Ilia's inventory profile ships in their profiles.py
    PROFILES["controlr-inference"] = controlr_inference


# ---------------------------------------------------------------------------------------------------- index
def _modname(p: Path, root: Path) -> str:
    rel = Path(p).resolve().relative_to(root).with_suffix("")
    return ".".join(rel.parts[:-1] if rel.name == "__init__" else rel.parts)


def _imports(body, m, aliases=None, package=False) -> dict:
    aliases = aliases or {}
    out = {}
    for n in body:
        if isinstance(n, ast.Import):
            for a in n.names:
                full = aliases.get(a.name, a.name)
                out[a.asname or (a.name if a.name in aliases else a.name.split(".")[0])] = full
        elif isinstance(n, ast.ImportFrom) and (n.module is not None or n.level):
            base = n.module or ""
            if n.level:
                parts = m.split(".") if package else m.split(".")[:-1]
                base = ".".join(parts[:len(parts) - n.level + 1] + ([n.module] if n.module else []))
            base = aliases.get(base, base)
            for a in n.names:
                out[a.asname or a.name] = (base, a.name)
    return out


def walk_gated(node, tests: dict | None):
    """``ast.walk`` that skips the body of every ``if`` whose test is one of ``tests`` (its test and ``else`` are kept)."""
    if not tests:
        yield from ast.walk(node)
        return
    todo = [node]
    while todo:
        n = todo.pop()
        yield n
        if isinstance(n, ast.If) and ast.unparse(n.test) in tests:
            todo += [n.test, *n.orelse]
        else:
            todo += list(ast.iter_child_nodes(n))


class Index:
    def __init__(self, profile: Profile):
        self.p = profile
        self.funcs, self.by_name, self.methods, self.imports, self.mods, self.classes = {}, {}, {}, {}, {}, {}
        self.records = set()
        self.packages = {_modname(p, profile.root_dir) for p in profile.index if Path(p).name == "__init__.py"}
        self.bases, self.spans, self.comments = {}, {}, {}
        for path in profile.index:
            self.add_module(_modname(path, profile.root_dir), Path(path).read_text())

    def detached(self, m: str) -> bool:
        return any(m == d or m.startswith(d + ".") for d in self.p.detached)

    def add_module(self, m: str, src: str) -> None:
        import io
        import tokenize
        tree = ast.parse(src)
        self.mods[m] = (tree, src.splitlines())
        cm = {}
        try:
            for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                if tok.type == tokenize.COMMENT:
                    cm[tok.start[0]] = tok.string
        except (tokenize.TokenError, SyntaxError):
            pass
        self.comments[m] = cm
        self.by_name[m] = {}
        self.imports[m] = _imports(tree.body, m, self.p.aliases, m in self.packages)
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
                # A field-only, base-free stdlib dataclass has a generated record
                # constructor; its defaults/factories are scanned as module code.
                decorators = [d.func if isinstance(d, ast.Call) else d for d in n.decorator_list]
                if (not n.bases and len(decorators) == 1 and isinstance(decorators[0], ast.Name)
                        and self.imports[m].get(decorators[0].id) == ('dataclasses', 'dataclass')
                        and not any(isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for f in n.body)):
                    self.records.add((m, n.name))
                meths = []
                for f in n.body:
                    if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        q = f"{n.name}.{f.name}"
                        self.funcs[(m, q)] = f
                        meths.append((m, q))
                        if not self.detached(m):
                            self.methods.setdefault(f.name, []).append((m, q))
                # a class named in code: every method of it, and of its (indexed) bases named in its header, may run
                bases = [b.id for b in n.bases if isinstance(b, ast.Name)]
                self.classes[(m, n.name)] = meths
                self.bases[(m, n.name)] = bases
            elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Lambda):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        self.funcs[(m, t.id)] = n.value
                        self.by_name[m][t.id] = t.id
        # module globals (``=NAME``: every module-level statement binding NAME), class bodies (``C.<body>``: a class's
        # docstring, fields, defaults) and the module docstring (``<doc>``) are graph nodes too: a global is reached
        # where reached code names it, a class body with the class, the docstring with any function of the module
        glob, cbody = {}, {}
        for n in tree.body:
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name) and (m, x.id) not in self.funcs:
                            glob.setdefault(x.id, []).append(n)
            elif isinstance(n, ast.ClassDef):
                body = [b for b in n.body if not isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if body:
                    cbody[n.name] = body
        for name, stmts in glob.items():
            self._pseudo(m, "=" + name, stmts)
            self.by_name[m].setdefault(name, "=" + name)
        for c, stmts in cbody.items():
            self._pseudo(m, c + ".<body>", stmts)
        if ast.get_docstring(tree) is not None:
            self._pseudo(m, "<doc>", [tree.body[0]])

    def _pseudo(self, m, q, stmts) -> None:
        node = ast.Module(body=list(stmts), type_ignores=[])
        node.lineno, node.end_lineno = stmts[0].lineno, max(getattr(x, "end_lineno", x.lineno) for x in stmts)
        self.funcs[(m, q)] = node
        self.spans[(m, q)] = [(x.lineno, getattr(x, "end_lineno", x.lineno)) for x in stmts]

    def node_comments(self, key) -> list[tuple[int, str]]:
        """(line, text) of every comment inside ``key``'s code and in the comment block right above it."""
        m = key[0]
        node, lines, cm = self.funcs[key], self.mods[m][1], self.comments.get(m, {})
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        spans = self.spans.get(key) or [(first, getattr(node, "end_lineno", node.lineno))]
        out = []
        for a, b in spans:
            out += [(i, cm[i]) for i in range(a, b + 1) if i in cm]
            i = a - 1
            while i >= 1 and lines[i - 1].strip().startswith("#"):
                out.append((i, cm.get(i, lines[i - 1].strip())))
                i -= 1
        return sorted(set(out))

    def key_at(self, m: str, line: int):
        """The graph node of module m whose code holds ``line`` (the outermost: a nested function is its parent's)."""
        best = None
        for k, node in self.funcs.items():
            if k[0] != m:
                continue
            for a, b in self.spans.get(k) or [(node.lineno, getattr(node, "end_lineno", node.lineno))]:
                if a <= line <= b and (best is None or b - a > best[1]):
                    best = (k, b - a)
        return best[0] if best else None

    def resolve(self, m, name, local):
        imp = {**self.imports.get(m, {}), **local}
        if name in self.by_name.get(m, {}):
            return (m, self.by_name[m][name])
        tgt = imp.get(name)
        if isinstance(tgt, tuple) and tgt in self.funcs:
            return tgt
        if isinstance(tgt, tuple) and (tgt[0], "=" + tgt[1]) in self.funcs:   # an imported module global
            return (tgt[0], "=" + tgt[1])
        return None

    def resolve_class(self, m, name, local):
        """A bare name in module m that is an indexed class (by definition here or by import) -> (module, name)."""
        if (m, name) in self.classes:
            return (m, name)
        tgt = {**self.imports.get(m, {}), **local}.get(name)
        return tgt if isinstance(tgt, tuple) and tgt in self.classes else None

    def walk(self, key):
        return walk_gated(self.funcs[key], self.p.gates.get(key))

    def fork(self) -> "Index":
        """A copy that can take one more module (a learned verb) without touching this index."""
        import copy
        new = copy.copy(self)
        for k in ("funcs", "by_name", "imports", "mods", "classes", "bases", "spans", "comments"):
            setattr(new, k, dict(getattr(self, k)))
        new.records = set(self.records)
        new.packages = set(self.packages)
        new.__dict__.pop("_inventory_edges", None)
        new.by_name = {m: dict(v) for m, v in self.by_name.items()}
        new.methods = {k: list(v) for k, v in self.methods.items()}
        return new

    def hierarchy(self, c) -> set:
        """Class ``c``, its indexed bases and every indexed class deriving from it: whatever ``self`` may be in c."""
        seen, todo = set(), [c]
        while todo:
            x = todo.pop()
            if x in seen:
                continue
            seen.add(x)
            todo += [bc for b in self.bases.get(x, []) for bc in [self.resolve_class(x[0], b, {})] if bc]
            todo += [d for d, bs in self.bases.items() if any(self.resolve_class(d[0], b, {}) == x for b in bs)]
        return seen

    def module_of(self, value, m, local=None):
        """The indexed module an expression names: ``alias`` (an imported module) or a dotted chain ``pkg.mod`` through
        ``import pkg`` / ``import pkg.mod`` (review-hs2 A1); None otherwise."""
        imp = {**self.imports.get(m, {}), **(local or {})}
        parts = []
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if not isinstance(value, ast.Name):
            return None
        tgt = imp.get(value.id)
        if isinstance(tgt, tuple):
            base = f"{tgt[0]}.{tgt[1]}"
        elif isinstance(tgt, str):
            base = tgt if not parts else tgt.split(".")[0] if tgt.split(".")[0] == value.id else tgt
        else:
            return None
        mod = ".".join([base] + list(reversed(parts)))
        return mod if mod in self.mods else None

    def named_classes(self, nodes, m) -> set:
        """The indexed classes ``nodes`` (of module m) name: each may have instances from here on."""
        out = set()
        for n in nodes:
            if isinstance(n, ast.Name):
                c = self.resolve_class(m, n.id, {})
                if c:
                    out.add(c)
            elif isinstance(n, ast.Attribute):                 # module.Class, pkg.mod.Class
                mod = self.module_of(n.value, m)
                if mod and (mod, n.attr) in self.classes:
                    out.add((mod, n.attr))
            elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr" and len(n.args) >= 2
                  and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str)):
                mod = self.module_of(n.args[0], m)           # getattr(module, "Class")
                if mod and (mod, n.args[1].value) in self.classes:
                    out.add((mod, n.args[1].value))
        return out

    def ups(self, classes) -> set:
        """``classes`` and all their indexed bases: whose methods an instance of one of them may run."""
        seen, todo = set(), list(classes)
        while todo:
            x = todo.pop()
            if x in seen:
                continue
            seen.add(x)
            todo += [bc for b in self.bases.get(x, []) for bc in [self.resolve_class(x[0], b, {})] if bc]
        return seen

    def callees(self, key, live=None):
        """``live``: the classes that may have instances (``ups`` of the instantiated ones); None = every class."""
        m, q = key
        node = self.funcs[key]
        local = _imports([n for n in self.walk(key) if isinstance(n, (ast.Import, ast.ImportFrom))], m, self.p.aliases, m in self.packages)
        imp = {**self.imports.get(m, {}), **local}
        out = {(m, "<doc>")} & set(self.funcs)
        own = (m, q.split(".")[0]) if "." in q and (m, q.split(".")[0]) in self.classes else None
        if own and (m, own[1] + ".<body>") in self.funcs:
            out.add((m, own[1] + ".<body>"))
        for n in self.walk(key):
            if isinstance(n, ast.Name):
                r = self.resolve(m, n.id, local)
                if r:
                    out.add(r)
                named = self.resolve_class(m, n.id, local)
                for c in (self.ups([named]) if named else ()):
                    if (c[0], c[1] + ".<body>") in self.funcs:
                        out.add((c[0], c[1] + ".<body>"))
                    # implicit calls on an instance: its dunders (constructor, __call__, __getattr__, ...); a class in
                    # a detached module brings every method (they are never reached by name)
                    out.update(k_ for k_ in self.classes[c] if self.detached(c[0]) or k_[1].split(".")[-1].startswith("__"))
            elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr" and len(n.args) >= 2
                  and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str)):
                gm = self.module_of(n.args[0], m, local)         # getattr(module, "name"): that function / class
                if gm and (gm, n.args[1].value) in self.funcs:
                    out.add((gm, n.args[1].value))
                if gm and (gm, n.args[1].value) in self.classes:
                    for c in self.ups([(gm, n.args[1].value)]):
                        if (c[0], c[1] + ".<body>") in self.funcs:
                            out.add((c[0], c[1] + ".<body>"))
                        out.update(k_ for k_ in self.classes[c]
                                   if self.detached(c[0]) or k_[1].split(".")[-1].startswith("__"))
                for mk in self.methods.get(n.args[1].value, []):  # getattr(x, "name"): the attribute x.name
                    if live is None or (mk[0], mk[1].split(".")[0]) in live:
                        out.add(mk)
            elif isinstance(n, ast.Attribute):
                base = n.value
                dm = self.module_of(base, m, local) if isinstance(base, ast.Attribute) else None
                if dm and (dm, n.attr) in self.funcs:             # pkg.mod.func (A1)
                    out.add((dm, n.attr))
                    continue
                if dm and (dm, n.attr) in self.classes:           # pkg.mod.Cls(...) (A1)
                    for c in self.ups([(dm, n.attr)]):
                        if (c[0], c[1] + ".<body>") in self.funcs:
                            out.add((c[0], c[1] + ".<body>"))
                        out.update(k_ for k_ in self.classes[c]
                                   if self.detached(c[0]) or k_[1].split(".")[-1].startswith("__"))
                    continue
                if isinstance(base, ast.Name):
                    tgt = imp.get(base.id)
                    mod = tgt if isinstance(tgt, str) else (f"{tgt[0]}.{tgt[1]}" if isinstance(tgt, tuple) else None)
                    if mod in self.mods and (mod, n.attr) in self.funcs:
                        out.add((mod, n.attr))
                        continue
                    if mod in self.mods and (mod, n.attr) in self.classes:   # alias.Cls(...): its constructor too
                        for c in self.ups([(mod, n.attr)]):                 # (lib-hybrid found it missing)
                            if (c[0], c[1] + ".<body>") in self.funcs:
                                out.add((c[0], c[1] + ".<body>"))
                            out.update(k_ for k_ in self.classes[c]
                                       if self.detached(c[0]) or k_[1].split(".")[-1].startswith("__"))
                        continue
                    if base.id in ("self", "cls") and own:     # typed: only the methods self's class tree defines
                        hit = {(c[0], f"{c[1]}.{n.attr}") for c in ([own] if self.p.scan_globals else self.hierarchy(own))
                               if (c[0], f"{c[1]}.{n.attr}") in self.funcs}
                        if hit:
                            out |= hit
                            continue
                if n.attr in INSTRUCTION_ATTRS:                  # a data read (flagged where it happens), not a call
                    continue
                if n.attr.startswith("__") and not self.p.scan_globals:                      # dunders: constructors come with their class, super()
                    continue                                     # calls with the class's bases (``classes``)
                for mk in self.methods.get(n.attr, []):
                    if live is None or (mk[0], mk[1].split(".")[0]) in live:
                        out.add(mk)
        if key in self.p.dispatch:                               # linked dynamic dispatch
            for t in self.p.dispatch[key][1]:
                tk = _key(t)
                if tk in self.funcs:
                    out.add(tk)
        if self.p.scan_globals:
            out.add((m, '<module>'))
            for target in imp.values():
                modules = [target] if isinstance(target, str) else [target[0], '.'.join(target)]
                out.update((mod, '<module>') for mod in modules if (mod, '<module>') in self.funcs)
        out.discard(key)
        return out

    def resolved_leaf(self, key, expr, local, bound):
        """Known language/library operations, not exemptions for their callers.

        Shadowed names remain unresolved. Never bless numpy.load/importlib/open.
        """
        m, _ = key
        imp = {**self.imports.get(m, {}), **local}
        if isinstance(expr, ast.Name) and expr.id not in bound:
            if expr.id in {'float', 'list'} and expr.id not in imp and expr.id not in self.by_name[m]:
                return True
            target = imp.get(expr.id, (m, expr.id))
        elif isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id not in bound:
            module = imp.get(expr.value.id)
            target = (module, expr.attr) if isinstance(module, str) else None
        else:
            return False
        return target in self.records or target in {('dataclasses', 'field'), ('numpy', 'ascontiguousarray')}

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


DATA_NAME = re.compile(r"[\w.-]+\.(json|ya?ml|npz|npy|csv|pkl|pickle|txt|toml)\b")


def _consts(xs) -> list[str] | None:
    vals = []
    for x in xs:
        if isinstance(x, ast.Constant) and isinstance(x.value, (str, int, float)):
            vals.append(str(x.value))
        else:
            return None
    return vals


def _folded(n) -> str | None:
    """The literal text of an f-string, a ``+`` chain of string constants (its constant parts joined), ``"sep".join``
    of constants, ``"fmt" % constants`` or ``"fmt".format(constants)``."""
    if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Constant)
            and isinstance(n.func.value.value, str)):
        sep = n.func.value.value
        if n.func.attr == "join" and len(n.args) == 1 and isinstance(n.args[0], (ast.List, ast.Tuple)):
            vals = _consts(n.args[0].elts)
            return sep.join(vals) if vals is not None else None
        if n.func.attr == "format":
            vals = _consts(n.args)
            try:
                return sep.format(*vals) if vals is not None and not n.keywords else None
            except (IndexError, KeyError, ValueError):
                return None
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod) and isinstance(n.left, ast.Constant) \
            and isinstance(n.left.value, str):
        vals = _consts(n.right.elts if isinstance(n.right, ast.Tuple) else [n.right])
        try:
            return n.left.value % tuple(vals) if vals is not None else None
        except (TypeError, ValueError):
            return None
    if isinstance(n, ast.JoinedStr):
        return "".join(v.value for v in n.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
        parts, stack = [], [n]
        while stack:
            x = stack.pop()
            if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Add):
                stack += [x.right, x.left]
            elif isinstance(x, ast.Constant) and isinstance(x.value, str):
                parts.append(x.value)
            else:
                return None
        return "".join(parts)
    return None


def scan_strings(nodes, rx: re.Pattern, data_files: dict, precise=False) -> list[str]:
    """Task names, task data paths and unlisted data files in every string of ``nodes`` (docstrings included), and in
    f-strings / string concatenations folded from their constant parts. The literal attribute name of a ``getattr`` /
    ``setattr`` / ``hasattr`` / ``delattr`` is an identifier, not a path: it is checked for task names, not for
    task data paths (``getattr(policy, "log", None)`` is no ``log/`` directory)."""
    nodes = list(nodes)
    attr_names = {id(n.args[1]) for n in nodes
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and len(n.args) >= 2
                  and n.func.id in ("getattr", "setattr", "hasattr", "delattr")}
    out = []
    for n in nodes:
        texts = []
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            texts = [n.value, camel_split(n.value)]
        else:
            f = _folded(n)
            if f is not None:                                    # the gaps (formatted values) may be any text
                g = re.sub(r"\s+", " ", f).strip()
                texts = [f, g, g.replace(" ", "_")]
        for t in texts:
            def suffix(offset=0):
                if not precise:
                    return ''
                line = min(getattr(n, 'end_lineno', n.lineno), n.lineno + t[:offset].count('\n'))
                return f" (line {line})"
            for match in rx.finditer(t):
                out.append(f"task name in a string: {match.group(0)!r}" + suffix(match.start()))
            for match in (() if id(n) in attr_names else DATA_PATHS.finditer(t)):
                label = match.group(0) if precise else t[:60]
                out.append(f"task data path: {label!r}" + suffix(match.start()))
            for d in DATA_NAME.finditer(t):
                if d.group(0) not in data_files and not d.group(0).startswith(("requirements", "pyproject")):
                    out.append(f"data file not in the allowlist: {d.group(0)!r}" + suffix(d.start()))
    return sorted(set(out))


def _pinned(line, src: str) -> bool:
    """A pinned line (or one of a tuple of accepted versions of it) is in the source."""
    return any(x in src for x in (line if isinstance(line, tuple) else (line,)))


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
    dispatch_lines = set()
    if key in p.dispatch:
        texts = p.dispatch[key][0]
        for text in texts if isinstance(texts, tuple) else (texts,):
            matches = {node.lineno + i for i, line in enumerate(src.splitlines())
                       if line.strip() == text.strip()}
            if len(matches) > 1:
                out.append('pinned dispatch line is ambiguous: re-audit it')
            else:
                dispatch_lines |= matches
        if not dispatch_lines:
            out.append("pinned dispatch line changed: re-audit it")
    reflective_lines |= dispatch_lines
    if key in p.required_lines and p.required_lines[key] not in src:
        out.append(f"required guard line changed: {p.required_lines[key]!r}: re-audit it")
    gates = p.gates.get(key) or {}
    present = {ast.unparse(n.test) for n in ast.walk(node) if isinstance(n, ast.If)}
    out += [f"pinned gate changed: {g!r}: re-audit it" for g in gates if g not in present]
    hits = scan_strings(idx.walk(key), rx, p.data_files, precise=p.scan_globals)
    for n in idx.walk(key):                                    # identifiers: def / class / arg / name / attribute / kw
        ident = (n.name if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else
                 n.arg if isinstance(n, (ast.arg, ast.keyword)) and n.arg else n.id if isinstance(n, ast.Name) else
                 n.attr if isinstance(n, ast.Attribute) else None)
        if ident:
            m_ = rx.search(ident) or rx.search(camel_split(ident))
            if m_:
                hits.append(f"task name in an identifier: {ident!r} (line {getattr(n, 'lineno', node.lineno)})")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in REFLECT_CALLS:
            hits.append(f"{n.func.id}() (line {n.lineno})")
        if isinstance(n, ast.Attribute) and n.attr in REFLECT_ATTRS:
            hits.append(f"introspection .{n.attr} (line {n.lineno})")
        if (isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and n.slice.value in p.identity_keys
                and not isinstance(n.ctx, ast.Store)):
            hits.append(f"[{n.slice.value!r}] read (line {n.lineno})")
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("get", "pop")
                and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value in p.identity_keys):
            hits.append(f".{n.func.attr}({n.args[0].value!r}) read (line {n.lineno})")
    for ln, text in idx.node_comments(key):                    # comments too: run history belongs in docs/notes
        m_ = rx.search(text)
        if m_:
            hits.append(f"task name in a comment: {m_.group(0)!r} (line {ln})")
    if key[1] == "<doc>":
        hits = [h for h in hits if not any(h == f"task name in a string: {n!r}" for (m_, n) in p.doc_ok if m_ == key[0])]
    out += hits
    cache_dicts = set()
    for n in idx.walk(key):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("setdefault", "pop", "get")
                and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "__dict__" and n.args
                and isinstance(n.args[0], ast.Constant) and n.args[0].value not in INSTRUCTION_ATTRS):
            cache_dicts.add(id(n.func.value))
    for n in idx.walk(key):
        if isinstance(n, ast.Attribute) and n.attr in TASK_SYMBOLS:
            out.append(f"task symbol .{n.attr} (line {n.lineno})")
        if isinstance(n, ast.Name) and n.id not in TASK_SYMBOLS:  # an alias of one (``from x import VOCABULARIES as V``)
            tgt = idx.imports.get(key[0], {}).get(n.id)
            if isinstance(tgt, tuple) and tgt[1] in TASK_SYMBOLS:
                out.append(f"task symbol {tgt[1]} as {n.id} (line {n.lineno})")
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name in TASK_SYMBOLS:
                    out.append(f"imports task symbol {a.name}" + (f" as {a.asname}" if a.asname else "")
                               + f" (line {n.lineno})")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in (
                "spec_from_file_location", "exec_module", "import_module", "__import__") and n.lineno not in reflective_lines:
            out.append(f"dynamic import .{n.func.attr} (line {n.lineno})")
        if isinstance(n, ast.Attribute):
            if n.attr in p.instruction_attrs and n.lineno not in pinned and not isinstance(n.ctx, ast.Store):
                out.append(f".{n.attr} read (line {n.lineno})")
            if n.attr == "__dict__" and id(n) not in cache_dicts:
                out.append(f"__dict__ (line {n.lineno})")
            if n.attr in ("__doc__", "__globals__", "__closure__", "__code__", "__self__", "__func__"):
                out.append(f"introspection .{n.attr} (line {n.lineno})")   # docstrings / callbacks' insides as data
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
                elif literal and a.value in tuple(p.instruction_attrs) + CHECKER_ATTRS:
                    out.append(f"{n.func.id}(..., {a.value!r}) (line {n.lineno})")
        if isinstance(n, ast.Name) and n.id in TASK_SYMBOLS | TASK_ACCESSORS:
            out.append(f"task symbol {n.id} (line {n.lineno})")
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
            for mod in mods:
                if TASK_MODULES.search(mod):
                    out.append(f"imports task / checker module {mod} (line {n.lineno})")
    if p.report_unresolved:
        bound = {n.id for n in idx.walk(key) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        bound |= {n.arg for n in idx.walk(key) if isinstance(n, ast.arg)}
        # Global rebinding also prevents trusting a module alias or builtin.
        for statement in idx.mods[key[0]][0].body:
            if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                bound |= {n.id for n in ast.walk(statement) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        local = _imports([x for x in idx.walk(key) if isinstance(x, (ast.Import, ast.ImportFrom))], key[0], idx.p.aliases)
        for n in idx.walk(key):
            if isinstance(n, ast.Call):
                loader = n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else ''
                if loader in {'spec_from_file_location', 'exec_module', 'import_module', 'SourceFileLoader'} or (
                        loader == 'run_path' and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'runpy'):
                    out.append(f'dynamic code loader {ast.unparse(n.func)} (line {n.lineno})')
                target_key = idx.resolve(key[0], n.func.id, local) if isinstance(n.func, ast.Name) else None
                # A binding node is scanned for provenance; it is not proof that
                # a callback's eventual target is resolved.
                known = target_key is not None and isinstance(idx.funcs[target_key],
                                                              (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
                if isinstance(n.func, ast.Attribute):
                    known = bool(idx.methods.get(n.func.attr))
                    if isinstance(n.func.value, ast.Name):
                        target = {**idx.imports.get(key[0], {}), **local}.get(n.func.value.id)
                        mod = target if isinstance(target, str) else '.'.join(target) if isinstance(target, tuple) else None
                        known = known or (mod, n.func.attr) in idx.funcs
                known = known or idx.resolved_leaf(key, n.func, local, bound)
                if not known:
                    out.append(f"unresolved call {ast.unparse(n.func)} (line {n.lineno})")
    if p.scan_globals:
        if key[0] in AUDITED_CHECKER_MODULES:
            if getattr(node, 'name', '') == '_in_target':
                out.append(f'audited checker-shaped acceptance predicate (line {node.lineno})')
            for item in idx.walk(key):
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
    return _walk(idx, list(roots or (list(p.roots) + list(p.callbacks))), rx)


def _reach(idx: Index, k) -> dict:
    """{reached function: the function it was reached from} from root k, under rapid type analysis to a fixpoint:
    the instantiable classes grow with what the reached code names, and with them the untyped calls' targets."""
    if idx.p.scan_globals:
        # The inventory profile deliberately keeps the original conservative
        # all-method graph. Partner profiles use rapid type analysis below.
        cache = getattr(idx, '_inventory_edges', {})
        idx._inventory_edges = cache
        parent, q = {k: None}, deque([k])
        while q:
            cur = q.popleft()
            if cur not in cache:
                cache[cur] = sorted(idx.callees(cur))
            for c in cache[cur]:
                if c not in parent:
                    parent[c] = cur
                    q.append(c)
        idx.last_inst = set(idx.classes)
        return parent
    inst = {_key(c) for c in idx.p.injected if _key(c) in idx.classes}
    while True:
        live = idx.ups(inst)
        parent = {k: None}
        q = deque([k])
        while q:
            cur = q.popleft()
            for c in idx.callees(cur, live):
                if c not in parent:
                    parent[c] = cur
                    q.append(c)
        new = set()
        for f in parent:
            new |= idx.named_classes(idx.walk(f), f[0])
        if new <= inst:
            idx.last_inst = inst
            return parent
        inst |= new


def _walk(idx: Index, roots, rx) -> list[Violation]:
    out = []
    scan_cache = {}
    recorded = set()
    checked_modules = set()
    for r in roots:
        k = _key(r)
        if k not in idx.funcs:
            k = idx.resolve(*k, {}) or k
        if k not in idx.funcs:
            out.append(Violation(r, r, "entry point not in the index (cannot be audited)", [r]))
            continue
        parent = _reach(idx, k)
        first_by_module = {}
        for reached in parent:
            first_by_module.setdefault(reached[0], reached)
        for m, first in sorted(first_by_module.items()):
            if m in checked_modules:
                continue
            checked_modules.add(m)
            chain0, x = [], first
            while x is not None:
                chain0.append(f"{x[0]}:{x[1]}")
                x = parent[x]
            for n in idx.mods[m][0].body:
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    for mod in ([a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]):
                        if TASK_MODULES.search(mod):
                            chain, x = [], first
                            while x is not None:
                                chain.append(f"{x[0]}:{x[1]}")
                                x = parent[x]
                            out.append(Violation(r, f"{m}:<module>", f"imports task / checker module {mod} "
                                                 f"(line {n.lineno})", list(reversed(chain))))
        for (m_, n_), why in idx.p.doc_ok.items():
            if (m_, n_) in parent:
                out.append(Violation(r, f"{m_}:{n_}", f"reached, so its docstring mention is no longer allowed ({why})",
                                     [f"{m_}:{n_}"]))
        for key in parent:
            if key not in scan_cache:
                scan_cache[key] = scan_function(idx, key, rx)
            probs = [why for why in scan_cache[key] if (key, why) not in recorded]
            if not probs:
                continue
            chain, x = [], key
            while x is not None:
                chain.append(f"{x[0]}:{x[1]}")
                x = parent[x]
            for why in probs:
                identity = (key, why)
                if identity not in recorded:
                    recorded.add(identity)
                    out.append(Violation(r, f"{key[0]}:{key[1]}", why, list(reversed(chain))))
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
                out.append(Violation('resource:' + rel, rel + ':<resource>', why, ['resource:' + rel]))
    seen, uniq = set(), []
    for v in out:                                  # one line per (function, why), kept with the first path found
        if (v.func, v.why) not in seen:
            seen.add((v.func, v.why))
            uniq.append(v)
    return uniq


def reachable(roots=None, profile: Profile | None = None) -> set:
    """The (module, qualname) set the audit scans from ``roots``."""
    p = profile or controlr_glib()
    idx = Index(p)
    seen = set()
    for r in (roots or (list(p.roots) + list(p.callbacks))):
        k = idx.resolve(*_key(r), {}) or _key(r)
        if k in idx.funcs:
            seen |= set(_reach(idx, k))
    return seen


def instantiable(roots=None, profile: Profile | None = None) -> set:
    """"module:Class" of every indexed class the audit lets have instances (injected, or named by reachable code)."""
    p = profile or controlr_glib()
    idx = Index(p)
    out = set()
    for r in (roots or (list(p.roots) + list(p.callbacks))):
        if _key(r) in idx.funcs:
            _reach(idx, _key(r))
            out |= {f"{c[0]}:{c[1]}" for c in idx.last_inst}
    return out


_SHARED: dict = {}


def shared_index(profile: Profile | None = None) -> "Index":
    """One index of the profile's tree per process (building it reads every indexed file, ~0.4 s): pass it to
    ``audit_source(..., index=)``, which audits on a fork of it."""
    p = profile or controlr_glib()
    if p.name not in _SHARED:
        _SHARED[p.name] = Index(p)
    return _SHARED[p.name]


def audit_source(source: str, name: str, profile: Profile | None = None, names=None,
                 index: "Index | None" = None) -> list[Violation]:
    """Audit a function given as source text (a learned verb): its body and everything it reaches in the index.
    ``index``: a prebuilt index (``shared_index()``) to audit on a fork of, instead of building one."""
    p = profile or controlr_glib()
    idx = index.fork() if index is not None else Index(p)
    idx.add_module("__learned__", source)
    rx = task_re(names)
    roots = [f"__learned__:{name}"]
    if ("__learned__", name) not in idx.funcs:
        return [Violation(roots[0], roots[0], f"no function {name!r} in the source", roots)]
    return _walk(idx, roots, rx)


# ---------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="transitive legality audit (call graph from entry points)")
    ap.add_argument("--profile", default="controlr-inference" if "controlr-inference" in PROFILES else "controlr-glib",
                    choices=sorted(PROFILES))
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
    seen = reachable(a.root or None, p)
    n = len(seen)
    nf = sum(1 for k in seen if not (k[1].startswith(("=", "<")) or k[1].endswith(".<body>")))
    if a.json:
        print(json.dumps({"profile": p.name, "reachable": n, "functions": nf, "violations": [v.__dict__ for v in vs]},
                         indent=1))
    else:
        for v in vs:
            print(v.line())
        print(f"{p.name}: {n} nodes reachable ({nf} functions, {n - nf} module globals / class bodies / docstrings), "
              f"{len(vs)} violations")
    return 1 if vs else 0


if __name__ == "__main__":
    sys.exit(main())
