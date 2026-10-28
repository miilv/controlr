"""Legality (audit 2026-10-28 P1-2): make_kong's matching tiles come from glyphs read from the images, never from the
task config's identity ties (select_mode ``same`` / ``same_index_as_label``). CPU only, fakes only: synthetic B1
records, the kinematic env, no catalogue, no images (the cv2 / scipy parts skip without the perception extras)."""

from __future__ import annotations

import ast
import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from dojo.perception.identity_marks import identity_known, identity_marks
from dojo.runtime.estimate import objects_from_record
from dojo.skills.env import KinematicEnv
from dojo.skills.geometry import quat_to_R
from dojo.skills.library import declare_kong, kong_match
from dojo.skills.skills import GTProvider, Skills
from dojo.skills.world import WorldState

REPO = Path(__file__).resolve().parents[2]
KONG = "Wait for the opponent to discard a tile, then declare a kong with the matching tiles."
HALF = [0.02245, 0.03235, 0.0164]                     # a mahjong tile's half extents
FRONT = [0.7071, 0.7071, 0, 0]                        # standing, face towards the robot
BACK = [0, 0, 0.7071, 0.7071]                         # standing, face away
DOWN = [0, 0.7071, 0.7071, 0]                         # lying face down
CANDS = [f"Rigid/mahjong/{i:05d}" for i in range(42)]  # the config's candidate pool of every tile label
GLYPH = [37, 25, 40, 31]                              # the scene's true glyph of hand triple g (GT: perception never has it)
PAIR = 11
PLACEHOLDER = 0                                       # B1's index for a slot it did not read: the first candidate
HAND = [f"mahjong{g}_{i}" for g in range(4) for i in range(3)] + ["mahjong4_0", "mahjong4_1"]


def scene_rows(discard: int = 0) -> list[dict]:
    """make_kong at the public config's slots on the kinematic env; push tile mahjong(5+discard)_0 lies face up (the
    opponent's discard)."""
    rows = []
    xs = [[-0.325, -0.279, -0.233], [-0.187, -0.141, -0.095], [-0.049, -0.003, 0.043], [0.089, 0.135, 0.181]]
    for g in range(4):
        for i, x in enumerate(xs[g]):
            rows.append(dict(label=f"mahjong{g}_{i}", category="mahjong", category_idx=GLYPH[g], pos=[x, -0.15, 0.7974],
                             quat=FRONT, half=HALF))
    for i, x in enumerate([0.227, 0.273]):
        rows.append(dict(label=f"mahjong4_{i}", category="mahjong", category_idx=PAIR, pos=[x, -0.15, 0.7974],
                         quat=FRONT, half=HALF))
    for g, x in enumerate([0.0, 0.046, -0.046, -0.092]):
        rows.append(dict(label=f"mahjong{5 + g}_0", category="mahjong", category_idx=GLYPH[g], pos=[x, 0.05, 0.7974],
                         quat=BACK, half=HALF))
    for lab, idx, y, z in (("other0", 2, -0.05, 0.7815), ("other1", 36, -0.002, 0.7815), ("other2", 14, -0.002, 0.8145),
                           ("mahjong9_0", PAIR, -0.05, 0.8145)):
        rows.append(dict(label=lab, category="mahjong", category_idx=idx, pos=[-0.4, y, z], quat=DOWN, half=HALF))
    d = next(r for r in rows if r["label"] == f"mahjong{5 + discard}_0")
    d["pos"], d["quat"] = [d["pos"][0], 0.0, 0.7814], [1, 0, 0, 0]
    return rows


def _label(sel):
    return {"type": "Rigid", "candidates": list(CANDS), "placement": {}, "select_mode": sel}


def tied_labels() -> dict:
    """The public make_kong config's select modes: each hand triple / the pair ``same``, push tile 5+k
    ``same_index_as_label`` triple k, mahjong9 with the pair, other0..2 ``unique``."""
    labels = {}
    for g, n in ((0, 3), (1, 3), (2, 3), (3, 3), (4, 2)):
        lst = [f"mahjong{g}_{i}" for i in range(n)]
        for lab in lst:
            labels[lab] = _label({"nums": n, "mode": "same", "allow_global_same": False, "label": lst})
    for g in range(4):
        labels[f"mahjong{5 + g}_0"] = _label({"nums": 1, "mode": "same_index_as_label", "same_label": f"mahjong{g}_0",
                                              "label": [f"mahjong{5 + g}_0"]})
    labels["mahjong9_0"] = _label({"nums": 1, "mode": "same_index_as_label", "same_label": "mahjong4_0",
                                   "label": ["mahjong9_0"]})
    for lab in ("other0", "other1", "other2"):
        labels[lab] = _label({"nums": 1, "mode": "unique", "label": [lab]})
    return labels


def untied_labels() -> dict:
    """The same labels and candidate pools with every identity tie removed (each label its own ``unique`` draw)."""
    labels = copy.deepcopy(tied_labels())
    for lab, d in labels.items():
        d["select_mode"] = {"nums": 1, "mode": "unique", "label": [lab]}
    return labels


def rotated_labels() -> dict:
    """Ties that point elsewhere: push tile 5+k tied to triple (k+1) % 4."""
    labels = tied_labels()
    for g in range(4):
        labels[f"mahjong{5 + g}_0"]["select_mode"]["same_label"] = f"mahjong{(g + 1) % 4}_0"
    return labels


def b1_record(rows: list[dict], reads: dict | None = None) -> dict:
    """What B1 writes at the config slots: every tile at its pose with the placeholder index (42 candidates > the
    slot identity read's limit: no glyph read, ``_slot_identity`` never ran), or, for labels in ``reads``, the index
    the pixels chose and ``identity: "read"`` (B1's slot read with a margin / a learned identity read)."""
    reads = reads or {}
    out = []
    for r in rows:
        p = {"slot": True, "geom_group": list(CANDS), "confidence": 0.9, "place_tag": "straight_front"}
        idx = PLACEHOLDER
        if r["label"] in reads:
            idx, p["identity"] = reads[r["label"]], "read"
        out.append({"category": "mahjong", "category_idx": idx, "label": r["label"],
                    "default_pos": [float(v) for v in r["pos"]], "default_ori": [float(v) for v in r["quat"]],
                    "scale": [1.0, 1.0, 1.0], "_perception": p})
    return {"task": "make_kong", "instruction": KONG, "Rigid": {"mahjong": out}}


class PerceivedIdentities(GTProvider):
    """Poses from the kinematic env (what the runtime's belief tracks), asset identities exactly as perception's record
    claims them (``objects_from_record``): the skill sees B1's identities and nothing of the scene's GT glyphs."""

    def __init__(self, env, perceived: dict):
        super().__init__(env, None, instruction=KONG)
        self.perceived = perceived

    def refresh(self, holding=None):
        w = super().refresh(holding)
        for lab, o in w.objects.items():
            p = self.perceived[lab]
            o.idx, o.identity_known = p.idx, p.identity_known
            o.asset = dataclasses.replace(o.asset, key=p.asset.key)
        return w


def run_kong(labels: dict, reads: dict | None = None, discard: int = 0):
    """The real path from B1's record to the skill: ``identity_marks`` (what ``inverse_layout.estimate`` runs where
    the config ties used to be applied) -> ``objects_from_record`` -> ``declare_kong`` on the kinematic env."""
    rec = identity_marks(b1_record(scene_rows(discard), reads), "make_kong", labels=labels)
    objs = objects_from_record(rec, None)
    env = KinematicEnv(scene_rows(discard), step_lim=600, instruction=KONG)
    sk = Skills(env, PerceivedIdentities(env, objs))
    res = declare_kong(sk, sorted(objs))
    return rec, objs, res, env, sk


def _standing(row) -> bool:
    return abs(quat_to_R(row["quat"])[2, 1]) > 0.99


def _face_up(row) -> bool:
    return quat_to_R(row["quat"])[2, 2] > 0.85


# ---------------------------------------------------------------------------------------- (a) the hidden config tie
def test_hidden_config_tie_changes_no_identity_and_no_kong_selection():
    """Identical records and observations, only the hidden config tie differs (the real ties, none, rotated ties):
    identical inferred identities (all unknown) and identical declare_kong outcomes (no match, nothing moved)."""
    runs = [run_kong(lbl) for lbl in (tied_labels(), untied_labels(), rotated_labels())]
    recs = [r[0] for r in runs]
    assert recs[0] == recs[1] == recs[2]
    assert {r["_perception"]["identity"] for r in recs[0]["Rigid"]["mahjong"]} == {"unknown"}
    assert {r["category_idx"] for r in recs[0]["Rigid"]["mahjong"]} == {PLACEHOLDER}   # no index written from a tie
    ids = [{lab: (o.idx, o.asset.key, o.identity_known) for lab, o in r[1].items()} for r in runs]
    assert ids[0] == ids[1] == ids[2]
    assert not any(known for _, _, known in ids[0].values())
    for _rec, _objs, res, env, sk in runs:
        assert res["ok"] is False and res["stage"] == "roles", res
        assert "glyph was not read from the images" in res["note"]
        kong = sk.world.notes["kong"]
        assert kong["discard"] == "mahjong5_0" and kong["match"] == [] and sorted(kong["hand"]) == sorted(HAND)
        rows = {o["label"]: o for o in env.objects}
        assert all(_standing(rows[lab]) for lab in HAND)                  # stopped before any push
    assert runs[0][2] == runs[1][2] == runs[2][2]
    # the audit's probe: key equality alone "matches" every placeholder tile; the skill's selection matches none
    w, hand = runs[0][4].world, runs[0][4].world.notes["kong"]["hand"]
    assert [lab for lab in hand if w.obj(lab).asset.key == w.obj("mahjong5_0").asset.key] == hand
    assert kong_match(w, hand, "mahjong5_0")[0] == []


def test_an_unread_placeholder_never_matches_a_read_glyph():
    """The discard's glyph is read and happens to equal the placeholder index of every unread hand tile: no match."""
    rec = identity_marks(b1_record(scene_rows(0), {"mahjong5_0": PLACEHOLDER}), "make_kong", labels=tied_labels())
    objs = objects_from_record(rec, None)
    w = WorldState("make_kong", KONG, objs, {}, {})
    assert objs["mahjong5_0"].identity_known and objs["mahjong5_0"].asset.key == objs["mahjong0_0"].asset.key
    assert kong_match(w, HAND, "mahjong5_0") == ([], "no hand tile's glyph was read from the images")


# ---------------------------------------------------------------------------------------- (b) image evidence
def _reads(discard_glyph: int) -> dict:
    """Every hand glyph and the discard's glyph read from the images (B1's slot read / a learned identity read)."""
    reads = {f"mahjong{g}_{i}": GLYPH[g] for g in range(4) for i in range(3)}
    reads.update({"mahjong4_0": PAIR, "mahjong4_1": PAIR, "mahjong5_0": discard_glyph})
    return reads


def test_glyphs_read_from_the_images_drive_the_selection():
    """Read glyphs select the matching triple, the same under the real config ties, none or rotated ones."""
    for labels in (tied_labels(), untied_labels(), rotated_labels()):
        for g in range(4):
            rec = identity_marks(b1_record(scene_rows(0), _reads(GLYPH[g])), "make_kong", labels=labels)
            objs = objects_from_record(rec, None)
            assert all(objs[lab].identity_known for lab in _reads(0)) and not objs["other0"].identity_known
            w = WorldState("make_kong", KONG, objs, {}, {})
            assert kong_match(w, HAND, "mahjong5_0") == ([f"mahjong{g}_{i}" for i in range(3)], None)


def test_read_glyphs_declare_the_kong_against_the_config_tie():
    """End to end on the kinematic env: the config ties push tile mahjong5_0 to triple 0, the pixels read its glyph as
    triple 2's; the skill tips triple 2 face up, leaves triple 0 standing and completes the kong."""
    rec, objs, res, env, sk = run_kong(tied_labels(), _reads(GLYPH[2]))
    assert res["ok"], res
    assert sk.world.notes["kong"]["match"] == ["mahjong2_0", "mahjong2_1", "mahjong2_2"]
    rows = {o["label"]: o for o in env.objects}
    assert all(_face_up(rows[f"mahjong2_{i}"]) for i in range(3))
    assert all(_standing(rows[lab]) for lab in HAND if not lab.startswith("mahjong2_"))


# ---------------------------------------------------------------------------------------- the marks themselves
def test_identity_marks_contract():
    labels = {"one": _label({"mode": "unique", "label": ["one"]})}
    labels["one"]["candidates"] = ["Rigid/card/00003"]
    rec = {"Rigid": {"card": [
        {"label": "one", "category_idx": 3, "_perception": {"slot": True}},                        # one candidate
        {"label": "read", "category_idx": 7, "_perception": {"slot": True, "geom_group": CANDS[:9], "identity": "read"}},
        {"label": "old", "category_idx": 4, "_perception": {"slot": True, "geom_group": CANDS, "identity": "structural"}},
        {"label": "fit", "category_idx": 5, "_perception": {"geom_group": CANDS[:3]}},               # chosen by a fit
        {"label": "fixed", "category_idx": 0, "_perception": {"place_tag": "fixed", "geom_group": CANDS[:2]}}]}}
    identity_marks(rec, "t", labels=labels)
    marks = {r["label"]: r["_perception"].get("identity") for r in rec["Rigid"]["card"]}
    assert marks == {"one": None, "read": "read", "old": "unknown", "fit": None, "fixed": "unknown"}
    assert [r["category_idx"] for r in rec["Rigid"]["card"]] == [3, 7, 4, 5, 0]                # indices untouched
    assert identity_known({}) and identity_known({"identity": "read"}) and not identity_known({"identity": "structural"})


class _RelationFinder(ast.NodeVisitor):
    RELATIONS = {"same_index_as_label", "same_as_label", "same_label"}

    def __init__(self):
        self.stack, self.hits = [], []

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Constant(self, node):
        if node.value in self.RELATIONS:
            self.hits.append((self.stack[-1] if self.stack else "<module>", node.lineno))


def test_policy_code_reads_no_config_identity_relation():
    """No policy-side module (perception, runtime, skills, agent, bridge) names the config's per-label identity ties;
    only the layout generator (dojo/simlab, dev) does. The one reviewed set-level use is listed."""
    reviewed = {("dojo/perception/inverse_layout.py", "quota_lists"),  # behind category_quota (default off,
                # legality triage 2026-10-28, test_legality_quota)
                # estimate.TIED_MODES (P1-3): tied identities are EXCLUDED from the twin's public constants; no
                # relation's value is read
                ("dojo/runtime/estimate.py", "<module>")}  # a same_as_label copy list = the source list's
    # assets as a set (imitate_sorting_sequence: "the corresponding objects"); journal 2026-10-28-legality-kong-identity
    found = set()
    for d in ("dojo/perception", "dojo/runtime", "dojo/skills", "dojo/agent", "dojo/bridge"):
        for f in sorted((REPO / d).rglob("*.py")):
            v = _RelationFinder()
            v.visit(ast.parse(f.read_text()))
            found |= {(str(f.relative_to(REPO)), fn) for fn, _ in v.hits}
    assert not found - reviewed, sorted(found - reviewed)


# ---------------------------------------------------------------------------------------- perception extras only
def test_index_structure_is_gone_and_estimate_marks_without_the_config(monkeypatch):
    """The audit's probe on the real module: ``inverse_layout.identity_marks`` (what ``estimate`` runs) with the hidden
    config tie changed through ``task_map``: identical records."""
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import inverse_layout as IL

    assert not hasattr(IL, "index_structure")
    out = []
    for labels in (tied_labels(), untied_labels()):
        monkeypatch.setattr(IL, "task_map", lambda labels=labels: {"make_kong": {"labels": labels}})
        rec = b1_record(scene_rows(0))
        for r in rec["Rigid"]["mahjong"]:
            del r["_perception"]["geom_group"]                       # candidate pools from the (patched) config
        out.append(IL.identity_marks(rec, "make_kong"))
    assert out[0] == out[1]
    assert {r["_perception"]["identity"] for r in out[0]["Rigid"]["mahjong"]} == {"unknown"}


def test_slot_identity_reads_only_with_a_margin(monkeypatch):
    """``_slot_identity`` marks "read" when one candidate beats the runner-up, not when every candidate scores the
    same (a tile's back)."""
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import inverse_layout as IL

    class SM:
        tag = "straight_front"

        def __init__(self, key):
            self.key = key
            self.asset = type("A", (), {"extents": np.array([0.045, 0.065, 0.033])})()

        def pose(self, x, y, yaw):
            return np.eye(4)

    monkeypatch.setattr(IL, "_slot_shape", lambda k, placement, tag: SM(k))
    monkeypatch.setattr(IL, "shape_by_tag", lambda k, tag: SM(k))
    monkeypatch.setattr(IL, "observed_color", lambda sm, T, views: (None, 0))
    monkeypatch.setattr(IL, "_app", lambda col, keys: 0.0)
    cfg = IL.EstimatorConfig()
    g = {"candidates": CANDS[:5]}
    h = IL.Hypothesis(0, CANDS[0], "straight_front", 0.0, -0.15, 0.0, 0.1, {"sil_loss": 0.1, "slot": True})
    monkeypatch.setattr(IL, "texture_cost", lambda sm, T, views, k, **kw: 0.1 if k == CANDS[3] else 0.6)
    out = IL._slot_identity(h, g, {}, "straight_front", {}, cfg)
    assert out.key == CANDS[3] and out.per_view.get("identity") == "read"
    monkeypatch.setattr(IL, "texture_cost", lambda sm, T, views, k, **kw: 0.4)        # every candidate the same
    out = IL._slot_identity(h, g, {}, "straight_front", {}, cfg)
    assert "identity" not in out.per_view


def test_kong_discard_reports_the_knocked_tile_only(monkeypatch):
    """The watcher's read: which push tile changed in the head view; no hand triple from the config's ties."""
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import inverse_layout as IL
    from dojo.perception import task_state
    from dojo.perception.cameras import TABLE_Z, head_camera

    slots = {"mahjong5_0": (0.0, 0.05, "straight_back"), "mahjong6_0": (0.046, 0.05, "straight_back"),
             "mahjong7_0": (-0.046, 0.05, "straight_back"), "mahjong8_0": (-0.092, 0.05, "straight_back")}
    monkeypatch.setattr(IL, "fixed_slot_labels", lambda task: slots)
    cam = head_camera()
    before = np.full((cam.h, cam.w, 3), 120, np.uint8)
    after = before.copy()
    uv, _ = cam.project(np.array([[0.046, 0.05, TABLE_Z + 0.03]]))
    u, v = np.round(uv[0]).astype(int)
    after[v - 6:v + 6, u - 4:u + 4] = (240, 30, 30)
    r = task_state.kong_discard(before, after)
    assert r["knocked"] == "mahjong6_0" and r["confident"]
    assert "triple" not in r and not hasattr(task_state, "PUSH_TO_TRIPLE")
