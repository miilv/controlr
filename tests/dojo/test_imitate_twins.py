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
    ex = SimpleNamespace(steps_left=900)                                   # a property on Executor
    sk = SimpleNamespace(imitate_wait="origin", ex=ex)
    assert ia.wait_demo_done(sk, 110, 5) == "still after 5 (after 2 timeouts)" and len(calls) == 3
    calls.clear()
    sk.imitate_wait = "still"
    assert ia.wait_demo_done(sk, 110, 5) == "timeout" and len(calls) == 1
    calls.clear()
    sk = SimpleNamespace(imitate_wait="origin", ex=SimpleNamespace(steps_left=250))   # 250 - 160 < one copy
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
    sk = SimpleNamespace(refresh=lambda: w, ex=SimpleNamespace(steps_left=900), log=lambda *a: None,
                         kin=SimpleNamespace(base=lambda a: np.eye(4)), home=lambda **k: None)
    assert ia.look_twins(sk, {"aim0": "t0", "aim1": "t1"}) == ["t0"] and looked == ["t0"]
    sk.ex = SimpleNamespace(steps_left=200)                                  # no budget for a look
    looked.clear()
    assert ia.look_twins(sk, {"aim0": "t0"}) == [] and looked == []


def test_the_wait_and_the_looks_run_on_the_real_executor_interface(monkeypatch):
    """the runtime's / G3's ``Executor.steps_left`` is a property (an int): the D / E rendered rows crashed calling it.
    wait_demo_done and look_twins against a real Executor on a fake env."""
    import numpy as np

    from dojo.skills import interactive as ia
    from dojo.skills import regrasp
    from dojo.skills.executor import Executor, steps_left_of

    st = {f"{a}_arm_joint_state": np.zeros(6) for a in ("left", "right")}
    st.update({f"{a}_ee_joint_state": np.ones(1) for a in ("left", "right")})
    env = SimpleNamespace(proprio=lambda: st, steps_used=830, step_lim=1600)
    ex = Executor(env)
    assert isinstance(Executor.__dict__["steps_left"], property) and steps_left_of(ex) == 770
    assert steps_left_of(SimpleNamespace(steps_left=lambda: 5)) == 5 and steps_left_of(SimpleNamespace()) is None
    calls = []
    monkeypatch.setattr(ia, "wait_actors_still", lambda sk, fallback=70: calls.append(1) or ("timeout" if len(calls) < 2 else "still after 3"))
    sk = SimpleNamespace(imitate_wait="origin", ex=ex)
    assert ia.wait_demo_done(sk, 110, 5) == "still after 3 (after 1 timeouts)"
    objs = {"aim0": _o("Rigid/garage/00014"), "t0": _o("Rigid/action_camera/00000")}
    for o in objs.values():
        o.center_w = np.zeros(3)
    monkeypatch.setattr(regrasp, "_look_over", lambda sk, arm, lab, T: (T, None))
    w = _w(objs)
    sk = SimpleNamespace(refresh=lambda: w, ex=ex, log=lambda *a: None, kin=SimpleNamespace(base=lambda a: np.eye(4)),
                         home=lambda **k: None)
    assert ia.look_twins(sk, {"aim0": "t0"}) == ["t0"]


def test_equal_cost_swap_finds_duplicate_same_asset_pairs():
    from dojo.skills.interactive import tied_twins
    objs = {"demo_g": _o("Rigid/garage/00014"), "demo_w": _o("Rigid/watch/00000"),
            "misread_watch": _o("Rigid/garage/00014"), "garage": _o("Rigid/garage/00014")}
    w = _w(objs)
    twins = {"demo_g": "misread_watch", "demo_w": "garage"}
    assert tied_twins(w, twins) == ["misread_watch", "garage"]
    # A row-local tie is insufficient: the watch's unique match must not be stolen.
    objs["garage"] = _o("Rigid/watch/00000")
    assert tied_twins(w, twins) == []
    # Missing labels are harmless; labels themselves do not carry identity.
    assert tied_twins(w, dict(twins, vanished="absent")) == []


def test_tie_look_corrects_same_asset_misread_then_reassigns(monkeypatch):
    import numpy as np
    from dojo.skills import interactive as ia, regrasp

    objs = {"demo_g": _o("Rigid/garage/00014"), "demo_w": _o("Rigid/watch/00000"),
            "misread_watch": _o("Rigid/garage/00014"), "garage": _o("Rigid/garage/00014")}
    for o in objs.values():
        o.center_w = np.zeros(3)
    w = _w(objs)
    looked = []

    def look(sk, arm, label, target, **kw):
        assert kw.get("read_kind") == ("imitate" if sk.imitate_tie_look else None)
        looked.append(label)
        if label == "misread_watch":
            objs[label].asset.key = "Rigid/watch/00000"
            objs[label].category = "watch"
        return target, None

    monkeypatch.setattr(regrasp, "_look_over", look)
    sk = SimpleNamespace(refresh=lambda: w, ex=SimpleNamespace(steps_left=900), log=lambda *a: None,
                         kin=SimpleNamespace(base=lambda a: np.eye(4)), home=lambda **k: None,
                         imitate_tie_look=True)
    twins = {"demo_g": "misread_watch", "demo_w": "garage"}
    assert ia.look_twins(sk, twins) == ["misread_watch", "garage"]
    assert ia.twin_assignment(w, list(twins), list(twins.values())) == {
        "demo_g": "garage", "demo_w": "misread_watch"}
    # Default off preserves the existing look policy, including its same-asset blind spot.
    looked.clear()
    objs["misread_watch"].asset.key = "Rigid/garage/00014"
    objs["misread_watch"].category = "garage"
    sk.imitate_tie_look = False
    assert ia.look_twins(sk, twins) == ["garage"]
    sk.imitate_tie_look = True
    sk.ex.steps_left = 200
    assert ia.look_twins(sk, twins) == []


def test_tie_look_is_capped_and_reads_no_hidden_relations(monkeypatch):
    import numpy as np
    from dojo.skills import interactive as ia, regrasp

    objs = {l: _o("Rigid/watch/00000") for l in ["a", "b", "c", "d", "w", "x", "y", "z"]}
    for o in objs.values():
        o.center_w = np.zeros(3)
    w = _w(objs)
    twins = dict(zip("abcd", "wxyz"))
    looked = []
    monkeypatch.setattr(regrasp, "_look_over", lambda sk, arm, l, T, **kw: looked.append(l))
    sk = SimpleNamespace(refresh=lambda: w, ex=SimpleNamespace(steps_left=1000), log=lambda *a: None,
                         kin=SimpleNamespace(base=lambda a: np.eye(4)), home=lambda **k: None,
                         imitate_tie_look="on")
    for hidden in ({"a": "w"}, {"a": "z"}):
        w.hidden_relation = hidden
        looked.clear()
        assert ia.look_twins(sk, twins) == list("wxy")
        assert looked == list("wxy")


def test_copy_routes_tie_flag_after_wait_and_reassigns_before_pick(monkeypatch):
    import numpy as np
    from dojo.skills import interactive as ia, regrasp

    objs = {"demo_g": _o("Rigid/garage/00014"), "demo_w": _o("Rigid/watch/00000"),
            "misread_watch": _o("Rigid/garage/00014"), "garage": _o("Rigid/garage/00014"),
            "basket0": _o("Geometry/basket/00005"), "basket1": _o("Geometry/basket/00005")}
    for label, o in objs.items():
        o.pos = np.array([0., 0.3 if label.startswith("demo") else -0.2, 0.8])
        o.quat = np.array([1., 0., 0., 0.])
        o.center_w = o.pos
    w = _w(objs)
    w.step, w.notes = 0, {}
    phases, puts, placed = [], [], set()
    sk = SimpleNamespace(refresh=lambda: w, ex=SimpleNamespace(steps_left=1500, steps_used=5, hold=lambda n: None),
                         kin=SimpleNamespace(base=lambda a: np.eye(4)), log=lambda *a: None, bank=lambda: None,
                         home=lambda **kw: None, imitate_pairing="one_to_one", imitate_tie_look=True,
                         held={"left": None, "right": None})
    # Far half relative to our arms: the two demonstrator objects.
    monkeypatch.setattr(ia, "watch", lambda *a: {"gone": [{"label": l, "asset": objs[l].asset.key}
                                                          for l in ("demo_g", "demo_w")]})
    monkeypatch.setattr(ia, "wait_demo_done", lambda *a: phases.append("wait") or "home")

    def look(sk, arm, label, T, **kw):
        assert kw.get("read_kind") == "imitate"
        assert phases[0] == "wait"
        phases.append("look")
        if label == "misread_watch":
            objs[label].asset.key, objs[label].category = "Rigid/watch/00000", "watch"

    def put(sk, label, container):
        puts.append(label)
        placed.add(label)
        return SimpleNamespace(note="")

    monkeypatch.setattr(regrasp, "_look_over", look)
    monkeypatch.setattr(ia, "inside_xy", lambda o, c: any(o is objs[l] for l in placed))
    monkeypatch.setattr(ia, "put_in_open", put)
    r = ia.copy_demonstration(sk, "basket0", "basket1", list(objs))
    assert r["ok"] and puts == ["garage", "misread_watch"]
    assert phases == ["wait", "look", "look"]


def test_look_read_passes_explicit_identity_kind_without_changing_default():
    from dojo.skills.regrasp import _look_read
    seen = []
    sk = SimpleNamespace(last_refine={"yaw_source": "test"},
                         _refine=lambda label, arm, kind: seen.append(kind))
    assert _look_read(sk, "left", "obj", 0.18) == "test"
    assert _look_read(sk, "left", "obj", 0.18, read_kind="imitate") == "test"
    assert seen == ["look", "imitate"]
