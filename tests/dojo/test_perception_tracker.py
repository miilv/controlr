"""dojo.perception.tracker: the episode object-state filter (pure python + numpy)."""
import pytest

pytest.importorskip("scipy")

from dojo.perception.tracker import Tracker, tcp_from_flange  # noqa: E402

FAR = {"left": [-0.3, -0.45, 1.2, 1, 0, 0, 0], "right": [0.3, -0.45, 1.2, 1, 0, 0, 0]}  # TCPs far from the objects


def blk(label, x, y, conf=0.9):
    return {"label": label, "category": "block", "asset": "Rigid/block/00007", "pos": [x, y, 0.7825], "quat": [1, 0, 0, 0],
            "size": [0.035, 0.035, 0.035], "confidence": conf}


def test_untouched_object_ignores_a_bad_reread_and_jitter():
    tr = Tracker()
    tr.update([blk("base", 0.0, -0.1), blk("b2", 0.2, -0.1)], FAR, {})
    objs, _, ev = tr.update([blk("base", 0.019, -0.1, conf=0.3), blk("b2", 0.2015, -0.1)], FAR, {})
    base = next(o for o in objs if o["label"] == "base")
    assert base["pos"][0] == 0.0 and base["source"] == "prior"
    assert any(e["event"] == "implausible_move" for e in ev)
    b2 = next(o for o in objs if o["label"] == "b2")
    assert b2["pos"][0] == 0.2  # within jitter: the tracked pose stays


def test_touched_object_moves_and_far_moves_relocate():
    tr = Tracker()
    tr.update([blk("a", 0.0, -0.1), blk("b", 0.2, -0.1)], FAR, {})
    near_a = {"left": [-0.15, -0.1, 0.80, 1, 0, 0, 0]}  # TCP = flange + 0.15 m along +x -> (0, -0.1, 0.8)
    assert abs(tcp_from_flange(near_a["left"])[0]) < 1e-3  # TCP offset 0.1501 m
    objs, _, ev = tr.update([blk("a", 0.03, -0.1, conf=0.6), blk("b", 0.2, -0.1)], near_a, {})
    assert next(o for o in objs if o["label"] == "a")["pos"][0] == 0.03
    # a pick-and-place far away (> pair_m) with no TCP sample near the source: conservation relocates it
    objs, last, ev = tr.update([blk("a", 0.03, -0.1), blk("b", -0.3, 0.05)], FAR, {})
    assert next(o for o in objs if o["label"] == "b")["pos"][0] == -0.3 and any(e["event"] == "relocated" for e in ev)
    assert len(objs) == 2 and not last


def test_unseen_untouched_is_carried_unseen_touched_goes_to_last_seen():
    tr = Tracker()
    tr.update([blk("a", 0.0, -0.1), blk("b", 0.2, -0.1)], FAR, {})
    objs, last, ev = tr.update([blk("b", 0.2, -0.1)], FAR, {})
    a = next(o for o in objs if o["label"] == "a")
    assert a["source"] == "prior" and a["age"] == 1 and not last
    held = {"left": [-0.15, -0.1, 0.80, 1, 0, 0, 0]}
    objs, last, ev = tr.update([blk("b", 0.2, -0.1)], held, {"left": 0.0})
    assert [o["label"] for o in objs] == ["b"] and last[0]["label"] == "a"


def test_floating_reading_loses_confidence():
    tr = Tracker()
    tr.update([blk("a", 0.0, -0.1)], FAR, {})
    up = blk("c", 0.3, 0.0)
    up["pos"][2] = 0.85
    objs, _, ev = tr.update([blk("a", 0.0, -0.1), up], FAR, {})
    assert next(o for o in objs if o["label"] == "c")["confidence"] == pytest.approx(0.45)


def test_scan_pose_constants_mirror_and_steps():
    from dojo.perception import scan

    el, tl = scan.eye_target("left")
    er, tr_ = scan.eye_target("right")
    assert el[0] == -er[0] and el[1:].tolist() == er[1:].tolist() and tl[0] == -tr_[0]
    assert scan.steps_between([0] * 6, [0.0, -0.24, 1.29, -1.78, 0.63, 0.15]) == 12  # 1.78 rad at 0.16 rad/step
    # the eye sits behind and above its arm's corner of the table, outside the object region
    (x0, x1), (y0, y1) = scan.OBJECT_REGION
    assert el[1] < y0 and el[2] > 1.0


def test_scan_fill_from_frame0_takes_only_unexplained_readings(monkeypatch):
    from dojo.perception import scan

    monkeypatch.setattr(scan, "_groups_by_label", lambda task: {"t": 0, "target_t": 1, "b0": 2, "b1": 2})
    rec = {"Rigid": {"t": [{"label": "t", "default_pos": [0.1, -0.1, 0.77]}],
                     "block": [{"label": "b0", "default_pos": [0.2, 0.0, 0.78]}]},
           "_not_found": [{"label": "target_t"}, {"label": "b1"}]}
    rec0 = {"Geometry": {"t_cushion": [{"label": "target_t", "default_pos": [-0.4, -0.2, 0.766], "_perception": {"confidence": 0.7}}]},
            "Rigid": {"block": [{"label": "b0", "default_pos": [0.201, 0.0, 0.78]},          # the same block: not a fill
                                {"label": "b1", "default_pos": [-0.3, -0.05, 0.78], "_perception": {"confidence": 0.6}}]}}
    out, filled = scan.fill_from_frame0(rec, rec0, "push_T")
    assert sorted(filled) == ["b1", "target_t"] and out["_not_found"] == []
    assert out["Geometry"]["t_cushion"][0]["_perception"]["source"] == "frame0_fill"
    assert [r["default_pos"][0] for r in out["Rigid"]["block"]] == [0.2, -0.3]


def test_index_structure_ties_config_groups(monkeypatch):
    from dojo.perception import inverse_layout as IL

    cands = [f"Rigid/mahjong/{i:05d}" for i in range(42)]
    labels = {"m0_0": {"select_mode": {"mode": "same", "label": ["m0_0", "m0_1"]}, "candidates": cands},
              "m0_1": {"select_mode": {"mode": "same", "label": ["m0_0", "m0_1"]}, "candidates": cands},
              "m1_0": {"select_mode": {"mode": "same", "label": ["m1_0"]}, "candidates": cands},
              "m5_0": {"select_mode": {"mode": "same_index_as_label", "same_label": "m0_0"}, "candidates": cands},
              "m6_0": {"select_mode": {"mode": "same_index_as_label", "same_label": "m1_0"}, "candidates": cands},
              "other0": {"select_mode": {"mode": "unique", "label": ["other0"]}, "candidates": cands},
              "bowl": {"select_mode": {"mode": "same", "label": ["bowl"]}, "candidates": cands[:3]}}
    monkeypatch.setattr(IL, "task_map", lambda: {"t": {"labels": labels}})
    rec = {"Rigid": {"mahjong": [{"label": k, "category_idx": 0, "_perception": {"slot": True}}
                                 for k in ("m0_0", "m0_1", "m1_0", "m5_0", "m6_0", "other0")],
                     "bowl": [{"label": "bowl", "category_idx": 2}]}}
    IL.index_structure(rec, "t")
    idx = {r["label"]: r["category_idx"] for r in rec["Rigid"]["mahjong"]}
    assert idx["m0_0"] == idx["m0_1"] == idx["m5_0"] and idx["m1_0"] == idx["m6_0"] and idx["m0_0"] != idx["m1_0"]
    assert idx["other0"] not in (idx["m0_0"], idx["m1_0"])
    assert rec["Rigid"]["bowl"][0]["category_idx"] == 2  # readable identity untouched


def test_truncated_frac_counts_vertices_outside_the_view():
    import numpy as np

    from dojo.perception import cameras, refine

    cam = cameras.head_camera()
    inside = np.array([[0.0, -0.1, 0.78], [0.05, -0.1, 0.78]])
    outside = np.array([[3.0, -0.1, 0.78], [0.0, -0.1, 2.0]])  # far right; above (behind) the camera
    assert refine.truncated_frac(cam, inside) == 0.0
    assert refine.truncated_frac(cam, np.r_[inside, outside]) == 0.5


def test_watch_arrival_order_and_polygon():
    import numpy as np

    from dojo.perception.watch import ArrivalOrder, inside_polygon

    poly = np.array([[0, 0], [0.3, 0], [0.3, 0.2], [0, 0.2]])
    assert inside_polygon((0.1, 0.1), poly) and not inside_polygon((0.4, 0.1), poly)
    ao = ArrivalOrder(n_consecutive=2, expected=2)
    seq = [[], ["a"], ["a"], ["a"], ["a", "b"], ["a", "b"], ["a", "b"]]
    occl = [False, False, True, False, False, False, False]  # an occluded read does not count
    out = None
    for inside, oc in zip(seq, occl):
        out = ao.update({"inside": [{"asset": k} for k in inside], "occluded": oc})
    assert out["order"] == ["a", "b"] and out["done"] is True
