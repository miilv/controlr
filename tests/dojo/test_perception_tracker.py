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
