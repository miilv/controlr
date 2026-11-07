"""Legality audit of the general skill library (Codex review 7 Nov, findings 1-5 and 20), through the shared tool
``dojo.legality.audit`` (CLI: ``python -m dojo.legality.audit``).

Static: ``audit_callgraph()`` follows the call graph from every implementation of ``glib.REGISTRY`` and scans every
reachable function (rules in the module's docstring). The auditor itself is tested on a planted module.
Dynamic: every verb runs (or is refused) on the kinematic env with ``InstructionSentinel`` as the episode's instruction
(every content access raises a ``BaseException``), and every ``cmd_obs`` reply a vendored skill receives is checked for the
sentinel's text. The model planner and the runtime import no other registry; no verb runs with the twin verifier.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from dojo.legality import audit as AU
from dojo.legality.audit import SENTINEL_TEXT, InstructionRead
from dojo.legality.audit import InstructionSentinel as Sentinel
from importlib.util import find_spec
if find_spec("dojo.skills.glib") is not None:
    from dojo.skills import glib as G
else:
    G = None

ROOT = Path(__file__).resolve().parents[2]
TASK_RE = AU.task_re()


@pytest.mark.skipif(G is None, reason="partner glib is not in this tree")
def test_the_library_passes_the_transitive_audit():
    vs = AU.audit_callgraph()
    assert not vs, "\n".join(v.line() for v in vs[:40]) + f"\n... {len(vs)} in all"


@pytest.mark.skipif(G is None, reason="partner glib is not in this tree")
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


def test_task_names_match_in_every_spelling():
    rx = AU.task_re(["stack_blocks", "push_T", "pour_by_language"])
    for s in ("stack_blocks", "stack blocks", "Stack-Blocks", "StackBlocks", "push_T", "PushT", "pour by language"):
        assert rx.search(s), s
    for s in ("stacked blocks", "pushed", "unstack_blocks_x"):
        assert not rx.search(s), s


@pytest.mark.skipif(G is None, reason="partner glib is not in this tree")
def test_a_learned_verb_given_as_source_is_audited_through_what_it_calls():
    good = "def tidy_two(sk, a, b):\n    from dojo.skills.glib import run\n    return run(sk, 'place_on', {'obj': a, 'support': b})\n"
    assert AU.audit_source(good, "tidy_two") == []
    bad = "def cheat(sk):\n    return sk.world.instruction.split()[0]\n"
    vs = AU.audit_source(bad, "cheat")
    assert vs and vs[0].why.startswith(".instruction read")
    assert AU.audit_source(good, "missing")[0].why.startswith("no function")


@pytest.mark.skipif(G is None, reason="partner glib is not in this tree")
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


@pytest.mark.skipif(G is None, reason="partner glib is not in this tree")
def test_the_superseded_registry_cannot_expose_a_task_or_text_reading_entry_whatever_the_caller_asks():
    from dojo.skills import model_registry as REG
    from dojo.agent.scene import SceneIds
    for name, e in REG.ENTRIES.items():
        if e.get("scope") == "task" or e.get("reads_text") or e.get("replaced"):
            assert f"- {name}:" not in REG.describe() and f"- {name}:" not in REG.describe(sealed_only=False, hide=())
            assert REG.check(name, {}, SceneIds()) == f"unknown skill {name!r}"


# ---------------------------------------------------------------------------------------------------- sentinel
def _kin(rows, step_lim=900):
    from dojo.skills.env import KinematicEnv
    from dojo.skills.skills import GTProvider, Skills
    env = KinematicEnv(rows, step_lim=step_lim)
    prov = GTProvider(env, None, instruction=AU.sentinel())
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
CALLS = {
    "lift": {"obj": "b0", "height_m": 0.08}, "place_on": {"obj": "b1", "support": "m"},
    "place_flat": {"obj": "b2", "pad": "m", "yaw_deg": 0.0}, "place_at": {"obj": "b1", "x": 0.0, "y": -0.3},
    "put_into": {"objects": ["b0"], "container": "box"}, "stack": {"objects": ["b0", "b1", "b2"]},
    "move_aside": {"obj": "b1"}, "cover_with": {"cover": "b0", "target": "b2"}, "put_back": {"obj": "b0"},
    "swap_places": {"a": "b0", "b": "b2"}, "push_onto": {"obj": "b1", "support": "m"},
    "slide_to": {"obj": "b1", "x": -0.05, "y": -0.2, "yaw_deg": 0.0}, "slide_to_pose": {"obj": "b1", "target": "m"},
    "press_button": {"button": "b2"}, "insert": {"obj": "b0", "receptacles": ["box"]},
    "insert_each": {"objects": ["b0", "b1"], "receptacle": "box"},
    "place_in_compartments": {"objects": ["b0"], "container": "box"},
    "drop_into_holder": {"objects": ["b1"], "holder": "box"}, "screw_on": {"pairs": [["b0", "m"]]},
    "pour_liquid": {"bottle": "b0", "target": "box"}, "pour_out": {"container": "b1", "target": "box"},
    "look_at": {"objects": ["b0", "b2"]},
}


VENDORED = {"swap_places", "slide_to_pose", "pour_liquid", "pour_out", "cover_with", "put_back"}


@pytest.mark.parametrize("name", sorted(G.REGISTRY) if G else ["partner-glib-unavailable"])
def test_every_verb_runs_with_a_sentinel_instruction_and_never_reads_it(name, monkeypatch):
    if G is None:
        pytest.skip("partner glib is not in this tree")
    from dojo.skills.ported import harness_robot as HR
    replies = []
    real = HR.HarnessRobot.cmd_obs

    def spy(self, a):
        r = real(self, a)
        replies.append(json.dumps({k: v for k, v in r.items() if k in ("instruction",)}, default=str))
        return r
    monkeypatch.setattr(HR.HarnessRobot, "cmd_obs", spy)
    env, sk = _kin(SCENE)
    args = CALLS.get(name)
    if args is None:                              # a frame verb on frameless boxes: refused by check, no code runs
        o = G.run(sk, name, {"obj": "b0"})
        assert not o.ok
        return
    try:
        G.run(sk, name, args)                     # ok or not: only an instruction read fails this test
    except InstructionRead:
        raise
    except Exception:  # noqa: BLE001 (motion on stand-in geometry may fail in any way; reads may not)
        pass
    assert all(SENTINEL_TEXT not in r for r in replies), replies
    assert all('"instruction": ""' in r for r in replies), replies
    if name in VENDORED:                          # the vendored path really ran and asked for observations
        assert replies, name


@pytest.mark.skipif(G is None, reason="partner glib is not in this tree")
def test_a_vendored_call_without_handles_would_read_it_so_the_sentinel_catches_a_leak():
    from dojo.skills.ported import ported
    env, sk = _kin(SCENE)
    with pytest.raises(InstructionRead):
        ported(sk, "carry_keep_pose", {"obj": "b0", "xy": [0.0, -0.1]})       # no "@": the old, text-grounding path


@pytest.mark.skipif(G is None, reason="partner glib is not in this tree")
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
