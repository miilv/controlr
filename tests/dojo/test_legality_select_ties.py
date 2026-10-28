"""D-C17 (robot-harness coord/DECISIONS.md, 5e0eaf8): B1's select-list identity ties (``same`` / ``unique`` /
``hierarchical``) are a disclosed per-task prior only where the task's checker scores neither asset sharing nor an
identity correspondence (dojo.perception.select_ties). A new policy-side consumer of the select modes fails here;
the reviewed tasks' table is checked against the catalogue where one exists (a box)."""

import ast
import os
from pathlib import Path

import numpy as np
import pytest

from dojo.perception.select_ties import INERT, KEPT, MODES, TIE_OFF, tie_mode

REPO = Path(__file__).resolve().parents[2]

# (file, function) that name a select mode, each reviewed under D-C17
REVIEWED = {
    ("dojo/perception/inverse_layout.py", "_options"):
        "D-C17 (1) kept: acts (>= 2 options) only on stack_blocks(_random) / stack_bowls(_random), checkers unordered",
    ("dojo/perception/inverse_layout.py", "quota_lists"):
        "behind category_quota (off by default, #233): imitate's quota is a config identity relation, D-C17 (2)",
    ("dojo/perception/select_ties.py", "<module>"): "the D-C17 table itself (MODES)",
    ("dojo/perception/select_ties.py", "tie_mode"): "the D-C17 switch: TIE_OFF tasks use no tie",
    ("dojo/perception/select_ties.py", "assign_keys"): "the identity step's assignment, given tie_mode's mode",
    ("dojo/perception/synth.py", "sample_layout"): "dev: a synthetic scene sampler (a generator), never the policy",
    ("dojo/runtime/estimate.py", "<module>"): "estimate.TIED_MODES (P1-3): tied identities are EXCLUDED from the twin",
}


class _ModeFinder(ast.NodeVisitor):
    def __init__(self):
        self.stack, self.hits = [], set()

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in MODES:
            self.hits.add(self.stack[0] if self.stack else "<module>")


def test_every_policy_side_consumer_of_a_select_mode_is_reviewed():
    found = set()
    for d in ("dojo/perception", "dojo/runtime", "dojo/skills", "dojo/agent", "dojo/bridge"):
        for f in sorted((REPO / d).rglob("*.py")):
            v = _ModeFinder()
            v.visit(ast.parse(f.read_text()))
            found |= {(str(f.relative_to(REPO)), fn) for fn in v.hits}
    assert found - set(REVIEWED) == set(), sorted(found - set(REVIEWED))
    assert set(REVIEWED) - found == set(), sorted(set(REVIEWED) - found)        # a stale entry is reviewed again


def test_the_reviewed_tasks_are_disjoint_and_the_switch_keys_on_them_only():
    assert not (set(TIE_OFF) & set(KEPT)) and not (set(TIE_OFF) & set(INERT))
    assert set(TIE_OFF) == {"make_kong", "stack_blocks_by_language", "fasten_screws"}
    for t in TIE_OFF:
        assert tie_mode(t, "unique") == tie_mode(t, "same") == tie_mode(t, None) == "allow_duplicate"
    for t in KEPT:
        assert tie_mode(t, "same") == "same" and tie_mode(t, "hierarchical") == "hierarchical"
    assert tie_mode("anything", None) == "unique"                               # the step's old default, unchanged


def test_a_tie_off_task_takes_each_identity_from_its_own_costs():
    """D-C17's rule as a regression test: identical costs (observations), the hidden select relation changed -> the same
    keys in a TIE_OFF task; in a kept task the disclosed prior acts."""
    pytest.importorskip("scipy")
    from dojo.perception.select_ties import assign_keys
    C = np.array([[0.10, 0.12, 0.90],          # member 0: key 0 best, key 1 close
                  [0.11, 0.50, 0.90],          # member 1: key 0 best (same asset as member 0 on its own evidence)
                  [0.90, 0.95, 0.20]])
    for t in TIE_OFF:
        keys = {m: assign_keys(C, tie_mode(t, m)) for m in ("same", "unique", "hierarchical", None)}
        assert all(k == [0, 0, 2] for k in keys.values()), keys
    assert assign_keys(C, tie_mode("stack_blocks", "hierarchical")) == [1, 0, 2]   # distinct keys (the kept prior)
    assert assign_keys(C, tie_mode("stack_bowls", "same")) == [0, 0, 0]


@pytest.mark.skipif(not (os.environ.get("DOJO_CATALOG") and (Path(os.environ.get("DOJO_CATALOG", "/nonexistent"))
                                                              / "tasks.json").exists()),
                    reason="needs the catalogue (DOJO_CATALOG with tasks.json): run on a box")
def test_the_catalogue_firing_table_equals_the_review():
    """Where the tie can act over the public catalogue: every task is in TIE_OFF / KEPT / INERT (a new task or a changed
    config fails here, on a box)."""
    pytest.importorskip("cv2")
    from dojo.perception import inverse_layout as IL
    acting, inert = set(), set()
    for task in IL.task_map():
        for g in IL.label_groups(task):
            mode = (g.get("select_mode") or {}).get("mode")
            rng = IL.instance_range(task, g)
            n = rng[1] if rng else len(g["labels"])
            geom = IL._geometry_groups(g, IL._placement_prior(g))
            multi = [len(v) for v in geom.values() if len(v) > 1]
            opts = IL._options(g, geom)
            ident = bool(multi) and n >= 2 and (mode or "unique") in ("same", "unique", "hierarchical")
            if ident or (opts is not None and len(opts) >= 2 and n >= 2):
                acting.add(task)
            elif opts is not None:
                inert.add(task)
    assert acting == set(TIE_OFF) | set(KEPT), sorted(acting ^ (set(TIE_OFF) | set(KEPT)))
    assert inert - acting <= set(INERT) | set(KEPT) | set(TIE_OFF), sorted(inert - acting - set(INERT))


def test_the_uniform_policy_turns_every_tie_off_and_the_default_is_unchanged():
    from dojo.perception.select_ties import POLICIES
    assert POLICIES[0] == "dc17"                                                 # the default policy
    for t in [*KEPT, *TIE_OFF, "anything"]:
        assert tie_mode(t, "same", "off") == tie_mode(t, "hierarchical", "off") == "allow_duplicate"
    assert tie_mode("stack_bowls", "same", "dc17") == "same"
    with pytest.raises(ValueError):
        tie_mode("stack_bowls", "same", "nope")
