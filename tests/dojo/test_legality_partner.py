"""Legality audit of the general skill library (Codex review 7 Nov, findings 1-5 and 20), through the shared tool
``dojo.legality.audit`` (CLI: ``python -m dojo.legality.audit``).

Static: ``audit_callgraph()`` follows the call graph from every implementation of ``glib.REGISTRY`` and scans every
reachable function (rules in the module's docstring). The auditor itself is tested on a planted module.
Dynamic: every verb's implementation is entered on the kinematic env (frame fixtures for the frame verbs) with
``InstructionSentinel`` as the episode's instruction (every content access raises a ``BaseException``); a whole hybrid
episode runs the same sentinel through the production ``RuntimeProvider``, checks its gated switches are off, and checks
every library object in the live graph is a class the audit's type analysis allows. No verb reaches a vendored skill.
The model planner and the runtime import no other registry; no verb runs with the twin verifier.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from dojo.legality import audit as AU
from dojo.legality.audit import SENTINEL_TEXT, InstructionRead
from dojo.legality.audit import InstructionSentinel as Sentinel
try:
    from dojo.skills import glib as G
except ImportError:
    G = None

ROOT = Path(__file__).resolve().parents[2]
TASK_RE = AU.task_re()


@pytest.mark.skipif(G is None, reason="partner glib and its runtime modules are absent from controlr")
def test_the_library_passes_the_transitive_audit():
    vs = AU.audit_callgraph()
    assert not vs, "\n".join(v.line() for v in vs[:40]) + f"\n... {len(vs)} in all"


@pytest.mark.skipif(G is None, reason="partner glib and its runtime modules are absent from controlr")
def test_the_graph_reaches_deep_and_never_the_rule_planner_or_a_registry_of_task_policies():
    seen = AU.reachable()
    mods = {m for m, _ in seen}
    assert len(seen) > 300, len(seen)                                  # the audit follows real depth
    assert {"dojo.skills.skills", "dojo.skills.motion", "dojo.skills.library"} <= mods
    for bad in ("dojo.skills.planner", "dojo.skills.routes", "dojo.skills.model_registry", "dojo.skills.model_planner",
                "dojo.skills.g3", "dojo.skills.verify"):
        assert bad not in mods, sorted(k for k in seen if k[0] == bad)[:10]


def test_the_auditor_finds_each_kind_of_violation_with_its_call_path(tmp_path):
    mod = tmp_path / "pkg" / "bad.py"
    mod.parent.mkdir()
    mod.write_text('''
import json
from robodojo.tasks.checkers import stack_check


def entry(sk):
    return helper(sk)


def helper(sk):
    """Tuned on stack_blocks."""
    w = sk.refresh()
    x = w.instruction.lower()
    y = getattr(w, "task")
    z = getattr(w, sk.name)
    cfg = json.load(open("task/RoboDojo/config/x.yml"))
    n = sk.env.reward_manager
    v = VOCABULARIES
    return "PushT" + "pour by language"


def clean(sk):
    return sk.refresh()
''')
    p = AU.Profile(name="t", index=[mod], roots=["pkg.bad:entry", "pkg.bad:clean"], root_dir=tmp_path)
    vs = AU.audit_callgraph(profile=p)
    whys = {v.why.split(" (line")[0].split(":")[0] for v in vs}
    assert {"task name in a string", ".instruction read", "getattr(..., 'task')", "non-literal getattr",
            "task data path", "checker / env state .reward_manager", "task symbol VOCABULARIES",
            "imports task / checker module robodojo.tasks.checkers"} <= whys, whys
    assert all(v.func == "pkg.bad:helper" and v.path == ["pkg.bad:entry", "pkg.bad:helper"] for v in vs
               if v.why.startswith(".instruction"))
    assert not [v for v in vs if v.root == "pkg.bad:clean"]
    assert AU.main(["--root", "pkg.bad:clean"]) in (0, 1)                # the CLI runs (profile index differs here)


def test_the_rechecks_bypasses_are_caught(tmp_path):
    """Codex's re-check, "Scanner bypasses" and "Static call-graph blind spots": a relative import, a module global and
    a module docstring, a folded f-string / concatenation, an aliased task symbol, a literal getattr to a method, a
    string-built dispatch (linked, so its target is audited), a dynamically loaded file (flagged unless linked), a data
    file outside the allowlist, a class's methods reached only once something names the class (type analysis), and a
    gated branch (not followed while its pinned test is present)."""
    pkg = tmp_path / "pk"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "util.py").write_text('''"""Helpers first written for stack_blocks."""
LIMITS = {"dist": "stack_blocks"}


def helper(x):
    return x
''')
    (pkg / "mod.py").write_text('''
from .util import LIMITS, helper
from robodojo.tasks import VOCABULARIES as V


class Dispatcher:
    def call(self, cmd):
        return getattr(self, "cmd_" + cmd)()

    def cmd_peek(self):
        return self.world.instruction


class Unnamed:
    def shared(self):
        return "push_T"


class Named:
    def shared(self):
        return 1

    def guarded(self):
        if self.flag_off:
            return "pour_by_language"
        return 0


def entry(sk, d):
    # this offset was tuned on pour_by_language
    a = LIMITS["dist"]
    b = f"stack {sk.n} blocks".replace(" ", "_")
    c = "push" + "_T"
    m = getattr(sk, "shared")
    d.call("peek")
    open("secret_table.json")
    importlib.util.spec_from_file_location("x", "y.py")
    Named().guarded()
    return helper(a), b, c, V, m
''')
    base = dict(name="t", index=[pkg / "util.py", pkg / "mod.py"], roots=["pk.mod:entry"], root_dir=tmp_path,
                injected=["pk.mod:Dispatcher"])
    names = ["stack_blocks", "push_T", "pour_by_language"]
    linked = AU.Profile(**base, dispatch={("pk.mod", "Dispatcher.call"): ('return getattr(self, "cmd_" + cmd)()',
                                                                         ["pk.mod:Dispatcher.cmd_peek"])},
                        gates={("pk.mod", "Named.guarded"): {"self.flag_off": "off in this configuration"}})
    got = {(v.func, v.why.split(" (line")[0]) for v in AU.audit_callgraph(profile=linked, names=names)}
    assert ("pk.util:=LIMITS", "task name in a string: 'stack_blocks'") in got           # relative import + global
    assert ("pk.util:<doc>", "task name in a string: 'stack_blocks'") in got             # module docstring
    assert ("pk.mod:entry", "task name in a string: 'stack blocks'") in got              # folded f-string
    assert ("pk.mod:entry", "task name in a string: 'push_T'") in got                    # folded concatenation
    assert ("pk.mod:entry", "task symbol VOCABULARIES as V") in got                      # aliased task symbol
    assert ("pk.mod:Dispatcher.cmd_peek", ".instruction read") in got                    # linked dispatch target
    assert ("pk.mod:entry", "data file not in the allowlist: 'secret_table.json'") in got
    assert ("pk.mod:entry", "dynamic import .spec_from_file_location") in got
    assert ("pk.mod:entry", "task name in a comment: 'pour_by_language'") in got         # comments (lead, 7 Nov)
    assert not any(f == "pk.mod:Named.guarded" for f, _ in got)                          # gated branch not followed
    seen = AU.reachable(profile=linked)
    assert ("pk.mod", "Named.shared") in seen and ("pk.mod", "Unnamed.shared") not in seen   # literal getattr, RTA
    ungated = AU.Profile(**base)
    vs2 = AU.audit_callgraph(profile=ungated, names=names)
    assert any(v.func == "pk.mod:Named.guarded" and "pour_by_language" in v.why for v in vs2)
    assert any(v.func == "pk.mod:Dispatcher.call" and "non-literal getattr" in v.why for v in vs2)   # unlinked
    changed = AU.Profile(**base, gates={("pk.mod", "Named.guarded"): {"self.flag_on": "renamed"}})
    assert any("pinned gate changed" in v.why for v in AU.audit_callgraph(profile=changed, names=names))


def test_learn_gates_rules_are_shared_rules(tmp_path):
    """What learn-gate's lint caught and the shared audit missed (7 Nov): task names in identifiers and acronym
    CamelCase, strings built by join / % / format, reflection calls and attributes, identity reads by subscript."""
    planted_profile = AU.Profile(name="planted", root_dir=tmp_path)
    names = ["stack_blocks", "push_T", "pour_by_language"]
    src = '''def push_t_helper(robot, stack_blocks=1):
    a = "_".join(["stack", "blocks"])
    b = "%s_%s" % ("pour_by", "language")
    c = "{}_{}".format("stack", "blocks")
    class PushTVerb:
        pass
    d = eval("1")
    e = globals()
    f = robot.__class__
    g = robot.obs["instruction"]
    h = robot.info.get("layout_id")
    return a, b, c, d, e, f, g, h
'''
    whys = [v.why.split(" (line")[0] for v in AU.audit_source(src, "push_t_helper", names=names, profile=planted_profile)]
    for w in ("task name in an identifier: 'stack_blocks'", "task name in an identifier: 'PushTVerb'",
              "task name in a string: 'stack_blocks'", "task name in a string: 'pour_by_language'", "eval()",
              "globals()", "introspection .__class__", "['instruction'] read", ".get('layout_id') read"):
        assert w in whys, (w, whys)
    shared = AU.shared_index(planted_profile)
    assert AU.shared_index(planted_profile) is shared                                           # one per process
    a = sorted((v.func, v.why) for v in AU.audit_source(src, "push_t_helper", names=names, index=shared, profile=planted_profile))
    b = sorted((v.func, v.why) for v in AU.audit_source(src, "push_t_helper", names=names, profile=planted_profile))
    assert a == b and ("__learned__", "push_t_helper") not in shared.funcs        # the fork leaves it untouched


def test_a_constructor_called_through_a_module_alias_is_scanned(tmp_path):
    """lib-hybrid's find: ``import mod as M; M.Cls(...)`` reached no ``Cls.__init__``; now the class's dunders come with
    it, as with a bare ``Cls(...)``."""
    pkg = tmp_path / "qq"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "worker.py").write_text('''
class Session:
    def __init__(self, robot, name):
        self.f = getattr(robot, name)
''')
    (pkg / "entry.py").write_text('''
import qq.worker as W


def go(robot):
    return W.Session(robot, "x")
''')
    p = AU.Profile(name="t", index=[pkg / "worker.py", pkg / "entry.py"], roots=["qq.entry:go"], root_dir=tmp_path)
    assert ("qq.worker", "Session.__init__") in AU.reachable(profile=p)
    assert any(v.func == "qq.worker:Session.__init__" and "non-literal getattr" in v.why
               for v in AU.audit_callgraph(profile=p, names=["stack_blocks"]))


def test_task_names_match_in_every_spelling():
    rx = AU.task_re(["stack_blocks", "push_T", "pour_by_language"])
    for s in ("stack_blocks", "stack blocks", "Stack-Blocks", "StackBlocks", "push_T", "PushT", "pour by language"):
        assert rx.search(s), s
    for s in ("stacked blocks", "pushed", "unstack_blocks_x"):
        assert not rx.search(s), s


@pytest.mark.skipif(G is None, reason="requires partner glib behavior and source modules")
def test_a_learned_verb_given_as_source_is_audited_through_what_it_calls():
    good = "def tidy_two(sk, a, b):\n    from dojo.skills.glib import run\n    return run(sk, 'place_on', {'obj': a, 'support': b})\n"
    assert AU.audit_source(good, "tidy_two") == []
    bad = "def cheat(sk):\n    return sk.world.instruction.split()[0]\n"
    vs = AU.audit_source(bad, "cheat")
    assert vs and vs[0].why.startswith(".instruction read")
    assert AU.audit_source(good, "missing")[0].why.startswith("no function")


@pytest.mark.skipif(G is None, reason="partner glib and its runtime modules are absent from controlr")
def test_the_model_planner_and_the_runtime_name_only_the_general_library():
    from dojo.skills import model_planner as MP
    for f in ("dojo/skills/model_planner.py", "dojo/runtime/policy.py", "dojo/runtime/hybrid.py"):
        src = (ROOT / f).read_text()
        assert "model_registry" not in src.replace("``model_registry", ""), f
    text = MP.system_prompt()
    assert G.describe() in text
    for name in ("build_tower", "tower_from_boards", "declare_kong", "fold_garment", "play_board_game",
                 "copy_demonstration", "cover_and_uncover", "sort_into_baskets", "pour_bottle_into_cup", "do_steps"):
        assert f"- {name}:" not in text, name
    assert not TASK_RE.search(text)
    import inspect
    assert set(inspect.signature(MP.ModelPlanner).parameters) == {"chat", "model", "objects_root", "max_calls", "log"}


@pytest.mark.skipif(G is None, reason="partner glib and its runtime modules are absent from controlr")
def test_the_superseded_registry_cannot_expose_a_task_or_text_reading_entry_whatever_the_caller_asks():
    from dojo.skills import model_registry as REG
    from dojo.agent.scene import SceneIds
    for name, e in REG.ENTRIES.items():
        if e.get("scope") == "task" or e.get("reads_text") or e.get("replaced"):
            assert f"- {name}:" not in REG.describe() and f"- {name}:" not in REG.describe(sealed_only=False, hide=())
            assert REG.check(name, {}, SceneIds()) == f"unknown skill {name!r}"
            with pytest.raises(ValueError, match="not an eligible"):
                REG.to_runtime(name, {})


# ---------------------------------------------------------------------------------------------------- sentinel
class FrameAssets:
    """An asset db for the kinematic env: boxes by row size, and the named frames (pose7 in the object frame) the
    frame verbs need: a rack's hook, a mug's handle, a lever box's lever, a holder's two slots, a base's two pegs, a
    drum's two strike points."""
    FRAMES = {"rack": ({"hook": [[0.0, 0.0, 0.10, 1, 0, 0, 0]]}, {}),
              "mug": ({"handle": [[0.03, 0.0, 0.0, 1, 0, 0, 0]]}, {}),
              "lever_box": ({}, {"lever": [[0.0, 0.0, 0.03, 1, 0, 0, 0]]}),
              "holder": ({"slot_0": [[-0.03, 0.0, 0.03, 1, 0, 0, 0]], "slot_1": [[0.03, 0.0, 0.03, 1, 0, 0, 0]]}, {}),
              "base": ({"peg_0": [[-0.04, 0.0, 0.05, 1, 0, 0, 0]], "peg_1": [[0.04, 0.0, 0.05, 1, 0, 0, 0]]}, {}),
              "drum": ({"point_0": [[-0.02, 0.0, 0.04, 1, 0, 0, 0]], "point_1": [[0.02, 0.0, 0.04, 1, 0, 0, 0]]}, {})}

    def __init__(self, rows):
        self.half = {r["category"]: np.asarray(r["half"], float) for r in rows}

    def get(self, typ, cat, idx, clutter=False, mesh=False):
        from dojo.skills.world import AssetInfo
        h = self.half.get(cat, np.full(3, 0.015))
        pas, act = self.FRAMES.get(cat, ({}, {}))
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=-h, bbox_max=h,
                         functional={k: {"frame": v} for k, v in pas.items()},
                         active_functional={k: {"frame": v} for k, v in act.items()})


def _kin(rows, step_lim=900):
    from dojo.skills.env import KinematicEnv
    from dojo.skills.skills import GTProvider, Skills
    env = KinematicEnv(rows, step_lim=step_lim)
    prov = GTProvider(env, FrameAssets(rows), instruction=AU.sentinel())
    return env, Skills(env, prov)


def _row(label, xyz, half=0.015, typ="Rigid", cat="block"):
    return {"label": label, "category": cat, "category_idx": 0, "pos": list(xyz), "quat": [1, 0, 0, 0],
            "half": [half] * 3 if np.isscalar(half) else list(half), "type": typ}


def test_the_sentinel_itself_raises_on_a_read():
    s = Sentinel(SENTINEL_TEXT)
    for f in (lambda: str(s), lambda: s.lower(), lambda: s or "", lambda: "x" in s, lambda: f"{s}", lambda: s + "x"):
        with pytest.raises(InstructionRead):
            f()


SCENE = [_row("b0", (-0.25, -0.15, 0.78)), _row("b1", (-0.1, -0.1, 0.78)), _row("b2", (0.05, -0.2, 0.78)),
         _row("m", (0.2, -0.2, 0.7675), (0.04, 0.04, 0.0025), "Geometry", "mat"),
         _row("box", (0.25, 0.05, 0.80), (0.06, 0.05, 0.035), "Geometry", "basket")]
FRAME_SCENE = [_row("mug0", (-0.25, -0.15, 0.80), (0.03, 0.03, 0.035), cat="mug"),
               _row("ring0", (-0.15, -0.2, 0.7725), (0.025, 0.025, 0.0075), cat="ring"),
               _row("ring1", (-0.05, -0.2, 0.7725), (0.025, 0.025, 0.0075), cat="ring"),
               _row("tile0", (0.0, -0.1, 0.79), (0.03, 0.005, 0.025), cat="tile"),
               _row("hammer", (-0.2, 0.0, 0.78), (0.10, 0.015, 0.015), cat="hammer"),
               _row("rack", (0.2, 0.05, 0.85), (0.05, 0.05, 0.085), "Geometry", "rack"),
               _row("lever_box", (0.15, -0.2, 0.795), (0.04, 0.04, 0.03), "Articulated", "lever_box"),
               _row("holder", (0.25, -0.05, 0.795), (0.06, 0.03, 0.03), "Geometry", "holder"),
               _row("base", (0.05, 0.05, 0.775), (0.07, 0.03, 0.01), "Geometry", "base"),
               _row("drum", (-0.05, 0.1, 0.80), (0.04, 0.04, 0.035), "Geometry", "drum")]
CALLS = {
    "lift": {"obj": "b0", "height_m": 0.08}, "place_on": {"obj": "b1", "support": "m"},
    "place_flat": {"obj": "b2", "pad": "m", "yaw_deg": 0.0}, "place_at": {"obj": "b1", "x": 0.0, "y": -0.3},
    "put_into": {"objects": ["b0"], "container": "box"}, "stack": {"objects": ["b0", "b1", "b2"]},
    "move_aside": {"obj": "b1"}, "cover_with": {"cover": "b0", "target": "b2"}, "put_back": {"obj": "b0"},
    "push_onto": {"obj": "b1", "support": "m"}, "slide_to": {"obj": "b1", "x": -0.05, "y": -0.2, "yaw_deg": 0.0},
    "press_button": {"button": "b2"}, "insert": {"obj": "b0", "receptacles": ["box"]},
    "insert_each": {"objects": ["b0", "b1"], "receptacle": "box"},
    "place_in_compartments": {"objects": ["b0"], "container": "box"},
    "drop_into_holder": {"objects": ["b1"], "holder": "box"}, "screw_on": {"pairs": [["b0", "m"]]},
    "look_at": {"objects": ["b0", "b2"]}, "pour": {"container": "b1", "target": "box"},
    "handover": {"obj": "b1"}, "place_into_held": {"obj": "b2", "container": "b0"},
    "pose_to": {"obj": "b1", "x": -0.05, "y": -0.2, "yaw_deg": 30.0}, "tip_over": {"obj": "b0", "direction_deg": 0.0},
}
FRAME_CALLS = {
    "hang": {"objects": ["mug0"], "rack": "rack", "hook_frame": "hook"},
    "press_part": {"obj": "lever_box", "frame": "lever"},
    "put_in_slots": {"objects": ["tile0"], "container": "holder", "slots": ["slot_0"]},
    "thread_on_pegs": {"groups": [["ring0"], ["ring1"]], "base": "base", "pegs": ["peg_0", "peg_1"]},
    "strike": {"tool": "hammer", "instrument": "drum", "points": ["point_0", "point_1"]},
}


@pytest.mark.skipif(G is None, reason="partner glib and its runtime modules are absent from controlr")
def test_every_verb_has_a_sentinel_call_that_passes_its_checks():
    from dojo.skills.env import KinematicEnv  # noqa: F401
    assert set(CALLS) | set(FRAME_CALLS) == set(G.REGISTRY) and not set(CALLS) & set(FRAME_CALLS)
    for calls, scene in ((CALLS, SCENE), (FRAME_CALLS, FRAME_SCENE)):
        env, sk = _kin(scene)
        for name, args in calls.items():
            assert not G.check(name, args, sk.refresh()), (name, G.check(name, args, sk.refresh()))


@pytest.mark.parametrize("name", sorted(G.REGISTRY) if G else [])
@pytest.mark.skipif(G is None, reason="partner glib and its runtime modules are absent from controlr")
def test_every_verb_runs_with_a_sentinel_instruction_and_never_reads_it(name, monkeypatch):
    import importlib
    from dojo.skills import ported
    monkeypatch.setattr(ported, "ported", lambda *a, **k: pytest.fail("a library verb called a vendored skill"))
    mod, fn = G.REGISTRY[name].impl.split(":")
    real, entered = getattr(importlib.import_module(mod), fn), []

    def spy(*a, **k):                             # the implementation root really runs
        entered.append(name)
        return real(*a, **k)
    monkeypatch.setattr(importlib.import_module(mod), fn, spy)
    calls, scene = (CALLS, SCENE) if name in CALLS else (FRAME_CALLS, FRAME_SCENE)
    env, sk = _kin(scene)
    try:
        G.run(sk, name, calls[name])              # ok or not: only an instruction read fails this test
    except InstructionRead:
        raise
    except Exception:  # noqa: BLE001 (motion on stand-in geometry may fail in any way; reads may not)
        pass
    assert entered == [name], name


@pytest.mark.skipif(G is None, reason="requires partner glib behavior and source modules")
def test_a_vendored_call_without_handles_would_read_it_so_the_sentinel_catches_a_leak():
    from dojo.skills.ported import ported
    env, sk = _kin(SCENE)
    with pytest.raises(InstructionRead):
        ported(sk, "carry_keep_pose", {"obj": "b0", "xy": [0.0, -0.1]})       # no "@": the old, text-grounding path


@pytest.mark.skipif(G is None, reason="partner glib and its runtime modules are absent from controlr")
def test_no_verb_runs_with_the_twin_verifier_on():
    from dojo.skills import general
    seen = []
    env, sk = _kin(SCENE)
    sk.verifier = object()                        # a twin as the runtime would give one
    orig = general.place_on
    try:
        general.place_on = lambda sk_, **k: seen.append(sk_.verifier) or {"ok": True}
        G.run(sk, "place_on", {"obj": "b1", "support": "m"})
    finally:
        general.place_on = orig
    assert seen == [None] and sk.verifier is not None

@pytest.mark.xfail(strict=True, reason="A1: nested module imports and literal getattr constructors are not resolved upstream")
@pytest.mark.parametrize('expression', ['pkg.mod.Cls()', 'getattr(M, "Cls")()'])
def test_upstream_a1(tmp_path, expression):
    pkg = tmp_path / 'pkg'
    pkg.mkdir()
    mod = pkg / 'mod.py'
    mod.write_text('class Cls:\n    def __init__(self):\n        getattr(self, self.name)\n')
    entry = pkg / 'entry.py'
    entry.write_text('import pkg.mod\nimport pkg.mod as M\ndef run():\n    return ' + expression + '\n')
    p = AU.Profile(name='a1', root_dir=tmp_path, index=[mod, entry], roots=['pkg.entry:run'])
    assert any('non-literal getattr' in v.why for v in AU.audit_callgraph(profile=p, names=['unused']))

@pytest.mark.xfail(strict=True, reason="A2: upstream pinned dispatch exempts reflection throughout its function")
def test_upstream_a2(tmp_path):
    mod = tmp_path / 'entry.py'
    mod.write_text('def run(obj, name):\n    getattr(obj, name)()\n    getattr(obj, obj.secret)()\n')
    p = AU.Profile(name='a2', root_dir=tmp_path, index=[mod], roots=['entry:run'],
                   dispatch={('entry', 'run'): ('getattr(obj, name)()', [])})
    assert any('non-literal getattr (line 3)' == v.why for v in AU.audit_callgraph(profile=p, names=['unused']))

@pytest.mark.parametrize('imports,expression', [
    ('import qq.worker as W', 'W.Session(robot, "x")'),
    ('from qq.worker import Session', 'Session(robot, "x")'),
    ('', 'Local(robot, "x")'),
])
def test_constructor_spellings_reach_the_dunder(tmp_path, imports, expression):
    pkg = tmp_path / 'qq'
    pkg.mkdir()
    body = '    def __init__(self, robot, name):\n        self.f = getattr(robot, name)\n'
    worker = pkg / 'worker.py'
    worker.write_text('class Session:\n' + body)
    entry = pkg / 'entry.py'
    entry.write_text(imports + '\nclass Local:\n' + body + '\ndef go(robot):\n    return ' + expression + '\n')
    p = AU.Profile(name='constructors', root_dir=tmp_path, index=[worker, entry], roots=['qq.entry:go'])
    target = ('qq.entry', 'Local.__init__') if not imports else ('qq.worker', 'Session.__init__')
    assert target in AU.reachable(profile=p)
    assert any(v.func == ':'.join(target) and 'non-literal getattr' in v.why
               for v in AU.audit_callgraph(profile=p, names=['unused']))
