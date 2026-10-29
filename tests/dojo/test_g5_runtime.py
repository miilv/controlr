"""Misha's G5 verification, items 3 / 4 / 9 / 13 (docs/notes/G5-VERIFICATION-2026-10-29.md, "For Ilia")."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy

from test_runtime import FakeAssets, FakeEpisode, _env, _stacked, run_episode


# ------------------------------------------------------------------ #9: a look-fuse error keeps the frame-0 record
def test_a_look_fuse_error_keeps_frame0_and_the_episode_runs():
    from dojo.skills.geometry import T_from, T_to_pose7
    from dojo.skills.kinematics import X5Kin, down_R
    env = _env()
    reachable = T_to_pose7(X5Kin().flange_from_tcp(T_from((-0.25, -0.15, 0.90), down_R(0.0)))).tolist()

    class P:
        def estimate(self, frames, vocabulary, fold, prior_record=None):
            if len(frames) > 1:
                raise RuntimeError("fuse broke")
            return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": o["label"],
                                         "default_pos": list(o["pos"]), "default_ori": [1, 0, 0, 0],
                                         "_perception": {"confidence": 0.5}} for o in env.objects]}}

        def look_plan(self, frame, record, max_looks=2):
            return [{"arm": "left", "purpose": "low_confidence", "label": "block_1", "target": [-0.26, -0.19, 0.78],
                     "flange_pose": reachable}]

        def look_poses(self, target, arm, variants=None):
            return []

        def check(self, frame, objects, fold):
            return {o["label"]: {"head": {"prec": 1.0, "n_vis": 500, "n_px": 600}} for o in objects}

    pol = RuntimePolicy(RuntimeConfig(mode="perception"), assets=FakeAssets(), perception_client=P())
    rep = run_episode(pol, FakeEpisode(env), "perception")
    assert rep["error"] is None, rep["error"]
    assert any("fuse_error" in lk for lk in rep["looks"]) and not rep["records"].get("fused")
    assert _stacked(env)                                         # the buffered home ran with the first skill


# ------------------------------------------------------------------ #13: held looks and the agentic budget
@pytest.mark.parametrize("count_held,runs", [(True, False), (False, True)])
def test_held_looks_and_the_agentic_budget(count_held, runs):
    """agentic_max 3; one estimate plus two held looks: today the budget is spent; with agentic_count_held False two
    more estimates may still run."""
    calls = []

    class P:
        def needs_agent(self, rec):
            return True, "risk"

    pol = RuntimePolicy(RuntimeConfig(mode="perception", agentic="budget", agentic_max=3, agentic_count_held=count_held))
    pol._perception = P()
    pol.perception = lambda: P()
    pol.bridge = SimpleNamespace(steps_used=5)
    pol.report = {"agentic": [{"tag": "frame0"}, {"tag": "held"}, {"tag": "held"}], "events": []}
    pol._agentic = lambda obs, scene, rec, trigger, extra=None: calls.append(trigger) or {"agentic": True}
    out = pol._agentic_gate({}, "scene", {"b1": True}, "risk")
    assert (calls == ["risk"]) is runs and (out == {"agentic": True}) is runs
    assert RuntimeConfig().agentic_count_held is True


# ------------------------------------------------------------------ #3: a late search merges only the new finds
def _prov_with_belief():
    from dojo.runtime.belief import KinematicBelief, RuntimeProvider
    from dojo.skills.kinematics import X5Kin

    from test_runtime import STACK, objects_from_gt_rows
    env = _env()
    pol = RuntimePolicy(RuntimeConfig(), assets=FakeAssets())
    pol.reset(FakeEpisode(env).info())
    b = pol.bridge
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    return prov, env


def _rec_of(prov, extra=None):
    from dojo.runtime.estimate import est_objects
    rec = {"Rigid": {}}
    for o in est_objects(prov.belief.objects):
        rec["Rigid"].setdefault(o["category"], []).append(
            {"category": o["category"], "category_idx": o["category_idx"], "label": o["label"],
             "default_pos": [float(v) + 0.2 for v in o["pos"][:2]] + [float(o["pos"][2])], "default_ori": o["quat"]})
    for r in extra or []:
        rec["Rigid"].setdefault(r["category"], []).append(r)
    return rec


@pytest.mark.parametrize("found", [False, True])
def test_a_late_search_keeps_the_held_object_and_the_poses(found):
    """Misha's probe: pick, then a search for an unresolved label, with and without a find. The held label stays
    attached, the others keep their poses, only the find is added; nothing found leaves the belief as it was."""
    from dojo.agent.policy import _AgentCtx
    prov, env = _prov_with_belief()
    lab = sorted(prov.belief.objects)[0]
    prov.belief.attached["left"] = (lab, np.eye(4))
    before = {k: o.pos.copy() for k, o in prov.belief.objects.items()}
    new = {"category": "block", "category_idx": 5, "label": "found_block", "default_pos": [0.1, -0.2, 0.78],
           "default_ori": [1, 0, 0, 0]}
    rec = _rec_of(prov, [new] if found else None)
    refreshed = []
    c = SimpleNamespace(events=[], obs0={}, instruction="", perception=None, prov=prov,
                        sk=SimpleNamespace(refresh=lambda: refreshed.append(1)), search=lambda: rec if found else None)
    ctx = _AgentCtx(c, None)
    out = ctx._search()
    assert prov.belief.attached["left"][0] == lab
    for k, p in before.items():
        assert np.allclose(prov.belief.objects[k].pos, p)
    if found:
        assert out is rec and "found_block" in prov.belief.objects and refreshed == [1]
    else:
        assert out is None and "found_block" not in prov.belief.objects and not refreshed


def test_late_search_returns_none_when_nothing_was_found():
    pol = RuntimePolicy(RuntimeConfig(mode="perception"))
    pol.report = {"search": []}
    pol._search = lambda sk, obs0, rec, scene, kin, **kw: (pol.report["search"].append({"found": []}) or rec)
    assert pol._late_search(None, {}, {"_not_found": [{"label": "x"}]}, "s", None) is None
    pol._search = lambda sk, obs0, rec, scene, kin, **kw: (pol.report["search"].append({"found": ["x"]}) or rec)
    assert pol._late_search(None, {}, {"_not_found": [{"label": "x"}]}, "s", None) is not None


# ------------------------------------------------------------------ #4: an unmeasured release is not a verified place
@pytest.mark.parametrize("flag,measured,status", [(False, False, "ok"), (True, False, "unverified"), (True, True, "ok")])
def test_verify_place_needs_a_measurement_since_the_release(flag, measured, status):
    from dojo.skills.geometry import T_from
    from dojo.skills.skills import PlacePlan, Skills
    prov, env = _prov_with_belief()
    prov.place_measured = flag
    lab = sorted(prov.belief.objects)[0]
    prov._unmeasured = set() if measured else {lab}
    o = prov.belief.objects[lab]
    sk = Skills(prov.bridge, prov, reserve=30)
    pp = SimpleNamespace(label=lab, T_obj_target=o.T.copy(), steps=3, arm="left")
    r = sk.verify_place(pp)
    assert r.status == status and r.ok is (status == "ok")
    assert RuntimeConfig().place_verdict == "belief"


def test_release_marks_unmeasured_and_a_measurement_clears_it():
    """A release (``refresh`` pops it) marks the label unmeasured; a confirmed pose clears it; a new belief starts clean."""
    prov, env = _prov_with_belief()
    lab = sorted(prov.belief.objects)[0]
    prov.belief.released = [lab]
    prov.refresh()
    assert not prov.measured_since_release(lab)
    o = prov.belief.objects[lab]
    prov._post_place_confirmed = lambda *a, **k: True
    assert prov.confirm_pose(lab, o.pos, o.quat)["adopted"]
    assert prov.measured_since_release(lab)
    prov._unmeasured.add(lab)
    prov.init_objects(dict(prov.belief.objects))
    assert prov._unmeasured == set()
