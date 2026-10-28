"""Stream M's grasp-time perception metric (dojo.perception.grasp_metric): events from the gripper commands, the
corridor, footprint distances, the per-event classes and the belief rebuilt from a report. numpy only (no catalogue:
footprints fall back to discs, groups to one label each, the tolerance is passed in)."""

import numpy as np

from dojo.perception import grasp_metric as gm
from dojo.skills.kinematics import X5Kin

Q0 = [0.0, 0.6, 0.6, 0.0, 0.0, 0.0]
TOL = (15.0, None)


def _act(gl=1.0, gr=1.0, ql=Q0, qr=Q0):
    return {"left_arm_joint_state": list(ql), "left_ee_joint_state": [gl],
            "right_arm_joint_state": list(qr), "right_ee_joint_state": [gr]}


def _row(label, pos, cat="cube", idx=0, typ="Rigid", clutter=False):
    return {"label": label, "type": typ, "category": cat, "category_idx": idx, "pos": list(pos),
            "quat": [1, 0, 0, 0], "clutter": clutter}


def _bel(pos, cat="cube", idx=0):
    return {"key": f"Rigid/{cat}/{idx:05d}", "pos": list(pos), "quat": [1, 0, 0, 0]}


def test_grip_events_close_and_release():
    acts = [_act(1.0), _act(0.8), _act(0.4), _act(0.0), _act(0.0), _act(0.3), _act(1.0), _act(0.45), _act(1.0, 0.2)]
    ev = gm.grip_events(acts)
    # the closing run 0.4 -> 0.0 ends at step 4 (the last closing command); right's close at 9 never opens
    assert [(e["kind"], e["step"], e["arm"]) for e in ev] == [
        ("grasp", 4, "left"), ("release", 6, "left"), ("grasp", 8, "left"), ("release", 9, "left"), ("grasp", 9, "right")]


def test_grip_events_pre_shape_is_not_the_grasp():
    # hang_mugs' wall grasp: the jaws pre-shape to 0.43 on the approach, close at 76, open at 150
    acts = [_act(1.0)] * 41 + [_act(0.43)] * 34 + [_act(0.0)] * 74 + [_act(1.0)] * 3
    ev = gm.grip_events(acts)
    assert [(e["kind"], e["step"]) for e in ev] == [("grasp", 76), ("release", 150)]


def test_corridor_stops_after_its_length():
    track = []
    for z in np.linspace(1.0, 0.8, 21):            # 1 cm per step down
        T = np.eye(4)
        T[:3, 3] = [0.1, 0.0, z]
        track.append(T)
    c = gm.corridor(track, 20, length=0.05)
    assert np.allclose(c[0], [0.1, 0.0, 0.8])
    assert 0.85 <= c[-1][2] <= 0.861 and len(c) == 6


def test_footprint_distance():
    sq = np.array([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]], float)
    assert gm.footprint_dist(sq, np.array([[0.05, 0.05, 1.0], [0.05, 0.05, 0.9]])) == 0.0
    assert abs(gm.footprint_dist(sq, np.array([[0.2, 0.05, 1.0]])) - 0.1) < 1e-9
    assert abs(gm.footprint_dist(sq, np.array([[0.2, -0.05, 1.0], [0.2, 0.2, 1.0]])) - 0.1) < 1e-9
    h = gm.hull2d(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]))
    assert len(h) == 4


def _event(kind="grasp", target="a", point=(0.0, 0.0, 0.8), ref=None):
    return gm.Event(kind, target, list(point), [list(point), [point[0], point[1], point[2] + 0.1]], reference=ref)


def test_score_event_clean_and_classes():
    gt = [_row("a", (0.0, 0.0, 0.78)), _row("b", (0.07, 0.0, 0.78), idx=1), _row("far", (0.5, 0.5, 0.78), idx=2)]
    clean = {"a": _bel((0.003, 0.0, 0.78)), "b": _bel((0.07, 0.004, 0.78), idx=1), "far": _bel((0.5, 0.5, 0.78), idx=2)}
    s = gm.score_event(_event(), clean, gt, "t", tol=TOL)
    assert s["clean"] and s["clean_r"] == {"5": True, "10": True, "15": True}
    assert s["required"][0]["cls"] == "ok" and [n["label"] for n in s["neighbours"]] == ["b"]
    # the target 25 mm off: overshoot (15 < 25 <= 30); 40 mm: gross
    s = gm.score_event(_event(), {**clean, "a": _bel((0.025, 0.0, 0.78))}, gt, "t", tol=TOL)
    assert s["required"][0]["cls"] == "overshoot" and not s["clean"]
    s = gm.score_event(_event(), {**clean, "a": _bel((0.0, 0.04, 0.78))}, gt, "t", tol=TOL)
    assert s["required"][0]["cls"] == "gross"
    # wrong asset at the right place: identity
    s = gm.score_event(_event(), {**clean, "a": _bel((0.0, 0.0, 0.78), idx=7)}, gt, "t", tol=TOL)
    assert s["required"][0]["cls"] == "identity"
    # the target believed on its neighbour: wrong_object
    s = gm.score_event(_event(), {**clean, "a": _bel((0.07, 0.0, 0.78))}, gt, "t", tol=TOL)
    assert s["required"][0]["cls"] == "wrong_object"
    # not in the belief
    s = gm.score_event(_event(), {k: v for k, v in clean.items() if k != "a"}, gt, "t", tol=TOL)
    assert s["required"][0]["cls"] == "not_believed"


def test_neighbours_and_phantoms_per_radius():
    gt = [_row("a", (0.0, 0.0, 0.78)), _row("b", (0.10, 0.0, 0.78), idx=1)]
    # b's disc footprint (3 cm) is 7 cm from the corridor: inside R 10 / 15, outside R 5
    bel = {"a": _bel((0.0, 0.0, 0.78))}
    s = gm.score_event(_event(), bel, gt, "t", tol=TOL)
    assert s["clean_r"] == {"5": True, "10": False, "15": False}
    assert s["clean_required"] and any("neighbour b not found" in r for r in s["reasons"]["10"])
    # b found 10 mm off: clean; 25 mm off: not
    s = gm.score_event(_event(), {**bel, "b": _bel((0.11, 0.0, 0.78), idx=1)}, gt, "t", tol=TOL)
    assert s["clean"]
    s = gm.score_event(_event(), {**bel, "b": _bel((0.125, 0.0, 0.78), idx=1)}, gt, "t", tol=TOL)
    assert not s["clean_r"]["10"]
    # a believed object where nothing is, 9 cm away: a phantom at R 10, not at R 5
    gt2 = [_row("a", (0.0, 0.0, 0.78))]
    s = gm.score_event(_event(), {**bel, "ghost": _bel((0.0, 0.12, 0.78), idx=3)}, gt2, "t", tol=TOL)
    assert s["clean_r"]["5"] and not s["clean_r"]["10"] and s["phantoms"][0]["label"] == "ghost"
    # clutter counts as a neighbour unless excluded; camera_stand never does
    gt3 = [_row("a", (0.0, 0.0, 0.78)), _row("junk", (0.06, 0.0, 0.78), idx=4, clutter=True),
           _row("camera_stand", (0.0, 0.05, 0.78), cat="camera_stand", typ="Geometry")]
    assert not gm.score_event(_event(), bel, gt3, "t", tol=TOL)["clean"]
    assert gm.score_event(_event(), bel, gt3, "t", tol=TOL, include_clutter=False)["clean"]


def test_place_reference_and_not_held():
    gt = [_row("a", (0.0, 0.0, 0.83)), _row("base", (0.0, 0.0, 0.78), cat="plate", idx=0)]
    bel = {"a": _bel((0.0, 0.0, 0.83)), "base": _bel((0.002, 0.0, 0.78), cat="plate")}
    ev = _event("place", "a", (0.0, 0.0, 0.9), ref="base")
    s = gm.score_event(ev, bel, gt, "t", tol=TOL)
    assert s["clean"] and [r["role"] for r in s["required"]] == ["held", "reference"]
    s = gm.score_event(ev, {**bel, "base": _bel((0.05, 0.0, 0.78), cat="plate")}, gt, "t", tol=TOL)
    assert s["required"][1]["cls"] == "gross" and not s["clean_required"] and s["clean_target"]
    ev.meta["held_drift_mm"] = 80.0
    s = gm.score_event(ev, bel, gt, "t", tol=TOL)
    assert s["required"][0]["cls"] == "not_held"
    ev.meta["held_gap_mm"] = 0.0                    # turned in the hand, the jaws still on it: the gap decides
    assert gm.score_event(ev, bel, gt, "t", tol=TOL)["required"][0]["cls"] == "ok"


def test_episode_events_rebuilds_the_belief():
    kin = X5Kin()
    p = kin.tcp("left", Q0)[:3, 3]
    q1 = list(Q0)
    q1[0] = 0.3                                    # the carry: joint 1 turns the arm
    p1 = kin.tcp("left", q1)[:3, 3]
    rec = {"Rigid": {"cube": [{"label": "a", "category_idx": 0, "default_pos": list(p), "default_ori": [1, 0, 0, 0]},
                              {"label": "b", "category_idx": 1, "default_pos": [0.3, 0.3, 0.78],
                               "default_ori": [1, 0, 0, 0]}]}}
    acts = [_act(1.0), _act(1.0), _act(0.0), _act(0.0, ql=q1), _act(0.0, ql=q1), _act(1.0, ql=q1), _act(1.0, ql=q1)]
    rep = {"records": {"initial": rec, "fused": rec},
           "events": [{"kind": "refine", "label": "a", "arm": "left", "for": "pick", "step": 1, "ok": True,
                       "pose7": list(p + [0.004, 0, 0]) + [1, 0, 0, 0]},
                      {"kind": "belief_drop", "label": "a", "step": 6, "z": 0.7801}]}
    evs = gm.episode_events(rep, acts, kin)
    kinds = [(e.kind, e.target, e.step) for e, _ in evs]
    assert kinds == [("grasp", "a", 3), ("place", "a", 6)]
    (g, bg), (pl, bp) = evs
    assert g.meta["attributed"] == "refine"
    assert np.allclose(bg["a"]["pos"], p + [0.004, 0, 0])       # the refine was applied before the close
    assert np.allclose(bp["a"]["pos"], p1 + [0.004, 0, 0], atol=0.02)   # carried with the TCP to the release
    assert np.linalg.norm(np.asarray(bp["a"]["pos"]) - p) > 0.01
    # a GT where a really travelled: the place is clean
    gt = [_row("a", list(np.asarray(bp["a"]["pos"]))), _row("b", (0.3, 0.3, 0.78), idx=1)]
    assert gm.score_event(pl, bp, gt, "t", tol=TOL)["clean_required"]
    s = gm.summarize([{**gm.score_event(g, bg, gt, "t", tol=TOL)}, gm.score_event(pl, bp, gt, "t", tol=TOL)])
    assert s["events"] == 2 and s["grasp"] == 1 and s["place"] == 1


def test_close_without_a_target_is_not_a_grasp():
    kin = X5Kin()
    rec = {"Rigid": {"cube": [{"label": "a", "category_idx": 0, "default_pos": [0.4, 0.4, 0.78],
                               "default_ori": [1, 0, 0, 0]}]}}
    acts = [_act(1.0), _act(0.0), _act(1.0)]
    evs = gm.episode_events({"records": {"initial": rec}, "events": []}, acts, kin)
    assert [e.kind for e, _ in evs] == ["close_other", "release_other"]


def test_quat_roundtrip():
    for q in ([1, 0, 0, 0], [0.7071, 0, 0, 0.7071], [0.5, 0.5, -0.5, 0.5], [0, 1, 0, 0]):
        q = np.asarray(q, float) / np.linalg.norm(q)
        r = gm._R_quat(gm._quat_R(q))
        assert min(np.linalg.norm(r - q), np.linalg.norm(r + q)) < 1e-6


def test_flip_of_an_off_centre_origin_is_a_rotation_error():
    # a 10 cm bar whose origin sits at one end: flipped 180 deg about z, the body lies where it was
    V = np.array([[x, y, z] for x in (0.0, 0.1) for y in (-0.01, 0.01) for z in (0.0, 0.02)])
    fp = gm.Footprints(loader=lambda key: V)
    gt = [{**_row("bar", (0.0, 0.0, 0.78), cat="bar"), "quat": [1, 0, 0, 0]}]
    flipped = {"bar": {**_bel((0.1, 0.0, 0.78), cat="bar"), "quat": [0, 0, 0, 1]}}
    s = gm.score_event(_event(target="bar", point=(0.05, 0, 0.8)), flipped, gt, "t", tol=(15.0, 10.0), footprints=fp)
    r = s["required"][0]
    assert r["xy_mm"] < 1.0 and r["cls"] in ("rot", "ok")   # "rot" where eval_g1 (scipy) measures the turn
    s = gm.score_event(_event(target="bar", point=(0.05, 0, 0.8)), {"bar": _bel((0.06, 0.0, 0.78), cat="bar")}, gt,
                       "t", tol=(15.0, None), footprints=fp)
    assert s["required"][0]["cls"] == "gross" and abs(s["required"][0]["xy_mm"] - 60.0) < 0.5


def test_belief_probe_snapshots_replace_the_rebuild_and_carry_held_objects():
    from types import SimpleNamespace

    from dojo.runtime.eval import BeliefProbe
    kin = X5Kin()
    q1 = list(Q0)
    q1[0] = 0.3
    p0 = kin.tcp("left", Q0)[:3, 3]
    asset = SimpleNamespace(key="Rigid/cube/00000")
    cube = SimpleNamespace(asset=asset, pos=np.asarray(p0 + [0.002, 0, 0]), quat=np.array([1.0, 0, 0, 0]), source="perception")
    bel = SimpleNamespace(objects={"a": cube}, attached={}, q={"left": np.asarray(Q0), "right": np.asarray(Q0)})
    pol = SimpleNamespace(_live={"prov": SimpleNamespace(belief=bel)})
    acts = [_act(1.0), _act(1.0), _act(0.0), _act(0.0, ql=q1), _act(0.0, ql=q1), _act(1.0, ql=q1)]
    pr = BeliefProbe(pol)
    for k, a in enumerate(acts, 1):
        if k == 6:                                  # by the release the belief holds a, last stepped at Q0
            bel.attached = {"left": ("a", None)}
        pr.on_action(a, k)
    rep = pr.report()
    assert [s["step"] for s in rep["snaps"]] == [2, 5] and rep["errors"] == 0
    evs = gm.episode_events({"records": {}, "events": [], "dev_belief_probe": rep}, acts, kin)
    (g, bg), (pl, bp) = evs
    assert g.kind == "grasp" and g.target == "a" and g.meta["belief"] == "probe"
    assert np.allclose(bg["a"]["pos"], p0 + [0.002, 0, 0])
    T = kin.tcp("left", q1) @ np.linalg.inv(kin.tcp("left", Q0))
    want = T[:3, :3] @ (p0 + [0.002, 0, 0]) + T[:3, 3]           # rides the TCP from Q0 to q1
    assert pl.kind == "place" and np.allclose(bp["a"]["pos"], want, atol=1e-6) and np.linalg.norm(want - p0) > 0.05
    assert bp["a"]["source"] == "probe_held"


def test_release_needs_the_runtime_opening_and_attach_needs_the_object_at_the_pads():
    # a 0.0 -> 0.1 loosening is not a release (KinematicBelief: > tightest + 0.15 or > 0.6)
    acts = [_act(1.0), _act(0.0), _act(0.1), _act(0.0), _act(0.7)]
    assert [(e["kind"], e["step"]) for e in gm.grip_events(acts)] == [("grasp", 4), ("release", 5)]
    kin = X5Kin()
    p = kin.tcp("left", Q0)[:3, 3]
    V = np.array([[x, y, z] for x in (-0.02, 0.02) for y in (-0.02, 0.02) for z in (-0.02, 0.02)])
    fp = gm.Footprints(loader=lambda key: V)
    acts = [_act(1.0), _act(1.0), _act(0.0), _act(0.0), _act(1.0)]
    for off, attached in ((0.01, True), (0.05, False)):     # inside bbox + 15 mm / 3 cm beyond it
        rec = {"Rigid": {"cube": [{"label": "a", "category_idx": 0, "default_pos": list(p + [off, 0, 0]),
                                   "default_ori": [1, 0, 0, 0]}]}}
        rep = {"records": {"initial": rec},
               "events": [{"kind": "refine", "label": "a", "arm": "left", "for": "pick", "step": 1, "ok": False}]}
        evs = gm.episode_events(rep, acts, kin, footprints=fp)
        assert [e.kind for e, _ in evs] == ["grasp", "place" if attached else "release_other"]
        assert evs[0][0].target == "a" and evs[0][0].meta["attached"] == ("a" if attached else None)
