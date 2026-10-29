"""copy_demonstration's one-to-one twins (Skills.imitate_pairing "one_to_one", default "greedy")."""
from types import SimpleNamespace

from dojo.skills.interactive import twin_assignment


def _w(objs):
    return SimpleNamespace(objects=objs, obj=lambda l: objs[l])


def _o(key):
    return SimpleNamespace(asset=SimpleNamespace(key=key), category=key.split("/")[1])


def test_one_misread_per_side_still_pairs_every_twin():
    """5.1 frame 0 of 100002: our toy car (t2) read as garage/00015, the demonstrator's watch (aim3) as an action
    camera. Greedy first-free matching takes a garage twice and stops; the assignment pairs all five."""
    objs = {"aim0": _o("Rigid/garage/00014"), "aim1": _o("Rigid/action_camera/00000"), "aim2": _o("Rigid/toy_car/00001"),
            "aim3": _o("Rigid/action_camera/00002"), "aim4": _o("Rigid/phone/00000"),
            "t0": _o("Rigid/garage/00014"), "t1": _o("Rigid/action_camera/00000"), "t2": _o("Rigid/garage/00015"),
            "t3": _o("Rigid/watch/00000"), "t4": _o("Rigid/phone/00000")}
    tw = twin_assignment(_w(objs), [f"aim{i}" for i in range(5)], [f"t{i}" for i in range(5)])
    assert tw == {f"aim{i}": f"t{i}" for i in range(5)}


def test_missing_objects_are_skipped():
    objs = {"aim0": _o("Rigid/watch/00000"), "t0": _o("Rigid/watch/00000"), "t1": _o("Rigid/pen/00000")}
    assert twin_assignment(_w(objs), ["aim0", "aim9"], ["t0", "t1"]) == {"aim0": "t0"}
    assert twin_assignment(_w(objs), [], ["t0"]) == {}


def test_a_wait_timeout_keeps_holding_only_with_imitate_wait_origin(monkeypatch):
    """camera 100007 / 100009: the copy started on a wait_actors_still timeout while the demonstrator's arm was still
    moving, and RoboDojo's query ended the episode. "origin" holds round after round until it reads at rest, keeping
    the copy's steps; "still" (default) proceeds as before."""
    from dojo.skills import interactive as ia

    calls = []

    def fake_wait(sk, fallback=70):
        calls.append(1)
        return "timeout" if len(calls) < 3 else "still after 5"

    monkeypatch.setattr(ia, "wait_actors_still", fake_wait)
    ex = SimpleNamespace(steps_left=lambda: 900)
    sk = SimpleNamespace(imitate_wait="origin", ex=ex)
    assert ia.wait_demo_done(sk, 110, 5) == "still after 5 (after 2 timeouts)" and len(calls) == 3
    calls.clear()
    sk.imitate_wait = "still"
    assert ia.wait_demo_done(sk, 110, 5) == "timeout" and len(calls) == 1
    calls.clear()
    sk = SimpleNamespace(imitate_wait="origin", ex=SimpleNamespace(steps_left=lambda: 250))   # 250 - 160 < one copy
    assert ia.wait_demo_done(sk, 110, 5).startswith("timeout x1, budget kept") and len(calls) == 1


def test_looks_go_to_our_objects_in_pairs_whose_assets_differ(monkeypatch):
    """imitate_look "ambiguous": a wrist look at OUR object of every pair whose believed assets differ (at most
    MAX_TWIN_LOOKS, budget kept for the copy); pairs with the same asset are not looked at."""
    import numpy as np

    from dojo.skills import interactive as ia
    from dojo.skills import regrasp

    objs = {"aim0": _o("Rigid/garage/00014"), "aim1": _o("Rigid/watch/00000"),
            "t0": _o("Rigid/action_camera/00000"), "t1": _o("Rigid/watch/00000")}
    for o in objs.values():
        o.center_w = np.zeros(3)
    looked = []
    monkeypatch.setattr(regrasp, "_look_over", lambda sk, arm, lab, T: looked.append(lab) or (T, None))
    w = _w(objs)
    sk = SimpleNamespace(refresh=lambda: w, ex=SimpleNamespace(steps_left=lambda: 900), log=lambda *a: None,
                         kin=SimpleNamespace(base=lambda a: np.eye(4)), home=lambda **k: None)
    assert ia.look_twins(sk, {"aim0": "t0", "aim1": "t1"}) == ["t0"] and looked == ["t0"]
    sk.ex = SimpleNamespace(steps_left=lambda: 200)                                  # no budget for a look
    looked.clear()
    assert ia.look_twins(sk, {"aim0": "t0"}) == [] and looked == []
