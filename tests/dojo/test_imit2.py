"""IMIT2 switches for imitate_sorting_sequence (all default off / today's behaviour):

- ``imitate_pairing`` "cues": twins by appearance (``dojo.perception.twin_match``), one_to_one without a reader;
- ``imitate_order`` "persist": a departure counts after PERSIST_READS reads in a row, dated by the first;
- ``imitate_carry`` "reach": far-side objects by the side arm / a table handover, centred grasps at long reach;
- ``imitate_retry`` "refind": more tries while steps remain, hands up before home, re-find at the start spot.
FakeLLM / mocks only: no simulator, no network."""
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.perception import twin_match as TM
from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import interactive as ia
from dojo.skills.world import AssetInfo


# ---------------------------------------------------------------------------------------------------- defaults
def test_defaults_are_todays_behaviour():
    c = RuntimeConfig()
    assert (c.imitate_pairing, c.imitate_order, c.imitate_carry, c.imitate_retry) == ("greedy", "v1", "v1", "v1")
    from dojo.skills.g3 import SKILL_ATTRS
    assert SKILL_ATTRS["imitate_pairing"][0] == "greedy" and "cues" in SKILL_ATTRS["imitate_pairing"]
    assert SKILL_ATTRS["imitate_order"] == ("v1", "persist", "elim", "persist_elim")
    for k, alt in (("imitate_carry", "reach"), ("imitate_retry", "refind")):
        assert SKILL_ATTRS[k] == ("v1", alt)


class _Assets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.015), bbox_max=np.full(3, 0.015))


@pytest.mark.parametrize("field,bad", [("imitate_order", "x"), ("imitate_carry", "x"), ("imitate_retry", "x"),
                                       ("imitate_pairing", "x")])
def test_invalid_values_are_refused(field, bad):
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", **{field: bad}), assets=_Assets())


def test_runtime_copies_the_switches_into_skills():
    from dojo.skills.env import KinematicEnv
    env = KinematicEnv([{"label": "b", "category": "block", "category_idx": 5, "pos": (0.3, -0.1, 0.78),
                         "quat": [1, 0, 0, 0], "half": [0.015] * 3}], step_lim=50, instruction="Stack the blocks.")
    for cfg, want in ((RuntimeConfig(mode="gt", verify="off"), ("v1", "v1", "v1")),
                      (RuntimeConfig(mode="gt", verify="off", imitate_order="persist", imitate_carry="reach",
                                     imitate_retry="refind", imitate_pairing="cues"), ("persist", "reach", "refind"))):
        pol = RuntimePolicy(cfg, assets=_Assets())
        pol.reset({"task": "x", "instruction": env.instruction, "step_lim": 50, "layout_id": 1})
        obs = {"images": {c: np.zeros((8, 8, 3), np.uint8) for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
               "state": env.proprio(), "instruction": env.instruction, "steps_used": 0, "step_lim": 50,
               "_dev_gt": env.gt_objects()}
        pol.next(obs)
        sk = pol._live["sk"]
        assert (sk.imitate_order, sk.imitate_carry, sk.imitate_retry) == want


# ---------------------------------------------------------------------------------------------------- twin_match
def test_histogram_and_distance():
    a = np.array([[50.0, 20.0, -10.0]] * 30)
    b = np.array([[50.0, 20.0, -10.0]] * 10)
    c = np.array([[90.0, -30.0, 30.0]] * 10)
    ha, hb, hc = TM.histogram(a), TM.histogram(b), TM.histogram(c)
    assert ha.sum() == pytest.approx(2.0)
    assert TM.bhattacharyya(ha, hb) == pytest.approx(0.0, abs=1e-9)
    assert TM.bhattacharyya(ha, hc) == pytest.approx(1.0, abs=1e-9)
    assert TM.histogram(np.zeros((0, 3))).sum() == 0


def _views(colours: dict, n=500, area=0.002):
    return {l: {"cam_head": {"n_px": n, "area_m2": area, "hist": TM.histogram(np.array([c] * 20, float))}}
            for l, c in colours.items()}


def test_joint_assignment_by_colour_ignores_the_believed_identity():
    """Our watch believed a garage, the demonstrator's watch believed an action camera: colour pairs them anyway."""
    cols = {"d_car": (70, 10, 60), "d_watch": (20, 0, 0), "d_phone": (40, 5, -40),
            "o_car": (71, 10, 58), "o_watch": (21, 1, 1), "o_phone": (41, 4, -38)}
    objs = [{"label": l, "key": k} for l, k in (("d_car", "Rigid/toy_car/00000"), ("d_watch", "Rigid/action_camera/00000"),
                                                 ("d_phone", "Rigid/phone/00000"), ("o_car", "Rigid/toy_car/00000"),
                                                 ("o_watch", "Rigid/garage/00014"), ("o_phone", "Rigid/phone/00000"))]
    cfg = dict(TM.DEFAULTS)
    C, cues = TM.cost_matrix(_views(cols), objs, ["d_car", "d_watch", "d_phone"], ["o_car", "o_watch", "o_phone"], cfg)
    pairs = TM.assign(C)
    assert {i: j for i, j in pairs} == {0: 0, 1: 1, 2: 2}
    m = TM.margins(C, pairs)
    assert min(m.values()) > cfg["tie_eps"]
    # the believed-identity prior alone would pair the misread watch with nothing sensible
    cfg_b = dict(cfg, w_color=0.0, w_area=0.0, w_cat=1.0)
    Cb, _ = TM.cost_matrix(_views(cols), objs, ["d_car", "d_watch", "d_phone"], ["o_car", "o_watch", "o_phone"], cfg_b)
    assert Cb[1, 1] == 1.0


def test_equal_appearance_is_a_tie():
    cols = {"d1": (20, 0, 0), "d2": (20, 0, 0), "o1": (20, 0, 0), "o2": (20, 0, 0)}
    objs = [{"label": l, "key": "Rigid/watch/00000"} for l in cols]
    C, _ = TM.cost_matrix(_views(cols), objs, ["d1", "d2"], ["o1", "o2"], dict(TM.DEFAULTS))
    pairs = TM.assign(C)
    assert all(v < TM.DEFAULTS["tie_eps"] for v in TM.margins(C, pairs).values())


def test_twin_costs_reads_no_hidden_relation(monkeypatch):
    """D-C17: the same pixels give the same pairing whatever a hidden config relation says (none is an input)."""
    cols = {"d1": (70, 10, 60), "d2": (20, 0, 0), "o1": (21, 0, 0), "o2": (69, 11, 59)}
    monkeypatch.setattr(TM, "object_views", lambda images, state, objects, cfg: _views(cols))
    objs = [{"label": l, "key": "Rigid/toy_car/00000", "pos": [0, 0, 0.8], "quat": [1, 0, 0, 0]} for l in cols]
    frame = {"images": {}, "state": {}}
    r1 = TM.twin_costs(frame, objs, ["d1", "d2"], ["o1", "o2"])
    objs2 = [dict(o, same_as_label="o1" if o["label"] == "d1" else None) for o in objs]
    r2 = TM.twin_costs(frame, objs2, ["d1", "d2"], ["o1", "o2"])
    assert r1["assign"] == r2["assign"] == {"d1": "o2", "d2": "o1"}
    assert TM.twin_costs(frame, objs, [], ["o1"])["assign"] == {}


# ---------------------------------------------------------------------------------------------------- skill: cues
def _o(key, pos):
    return SimpleNamespace(asset=SimpleNamespace(key=key, extents=np.array([0.04, 0.04, 0.03])),
                           category=key.split("/")[1], pos=np.array(pos, float), quat=np.array([1.0, 0, 0, 0]),
                           center_w=np.array(pos, float))


def _world(objs):
    w = SimpleNamespace(objects=objs, obj=lambda l: objs[l], step=0, notes={})
    return w


def _scene():
    return {"demo_a": _o("Rigid/garage/00014", (0.0, 0.3, 0.8)), "demo_b": _o("Rigid/watch/00000", (0.1, 0.3, 0.8)),
            "ours_x": _o("Rigid/garage/00014", (0.0, -0.2, 0.8)), "ours_y": _o("Rigid/garage/00014", (0.1, -0.2, 0.8)),
            "basket0": _o("Geometry/basket/00005", (-0.4, -0.15, 0.8)),
            "basket1": _o("Geometry/basket/00005", (0.35, 0.25, 0.8))}


def _sk(w, pairing, provider=None, **kw):
    return SimpleNamespace(refresh=lambda: w, ex=SimpleNamespace(steps_left=1500, steps_used=5, hold=lambda n: None),
                           kin=SimpleNamespace(base=lambda a: np.eye(4)), log=lambda *a: None, bank=lambda: None,
                           home=lambda **k: None, imitate_pairing=pairing, held={"left": None, "right": None},
                           provider=provider or SimpleNamespace(), **kw)


def _run_copy(monkeypatch, sk, objs):
    puts, placed = [], set()
    monkeypatch.setattr(ia, "watch", lambda *a: {"gone": [{"label": l, "asset": objs[l].asset.key}
                                                          for l in ("demo_a", "demo_b")]})
    monkeypatch.setattr(ia, "wait_demo_done", lambda *a: "home")

    def put(sk_, label, container, **kw):
        puts.append(label)
        placed.add(label)
        return SimpleNamespace(note="")

    monkeypatch.setattr(ia, "inside_xy", lambda o, c: any(o is objs[l] for l in placed))
    monkeypatch.setattr(ia, "put_in_open", put)
    return ia.copy_demonstration(sk, "basket0", "basket1", list(objs)), puts


def test_cues_pair_by_the_providers_appearance_read(monkeypatch):
    """Our watch believed a garage (both of ours read garage): the believed identities pair demo_b (watch) with
    whichever garage is left; the appearance read says ours_x looks like the watch."""
    objs = _scene()
    seen = []

    def twin_costs(objects, demo, ours):
        seen.append((sorted(o["label"] for o in objects), list(demo), list(ours)))
        return {"assign": {"demo_a": "ours_y", "demo_b": "ours_x"}, "ties": [], "margin": {},
                "cost": {"demo_a": {"ours_x": 0.6, "ours_y": 0.1}, "demo_b": {"ours_x": 0.1, "ours_y": 0.6}}}

    sk = _sk(_world(objs), "cues", SimpleNamespace(twin_costs=twin_costs))
    r, puts = _run_copy(monkeypatch, sk, objs)
    assert r["ok"] and puts == ["ours_y", "ours_x"]
    assert len(seen) == 1 and set(seen[0][1]) == {"demo_a", "demo_b"} and set(seen[0][2]) == {"ours_x", "ours_y"}


def test_cues_without_a_reader_fall_back_to_one_to_one(monkeypatch):
    """G3 parity: no provider.twin_costs -> the same twins as one_to_one."""
    objs = _scene()
    r1, p1 = _run_copy(monkeypatch, _sk(_world(objs), "cues"), objs)
    objs = _scene()
    r2, p2 = _run_copy(monkeypatch, _sk(_world(objs), "one_to_one"), objs)
    assert p1 == p2 and r1["ok"] == r2["ok"]


def test_cue_reassign_over_what_is_left():
    cue = {"cost": {"d1": {"o1": 0.1, "o2": 0.9, "o3": 0.5}, "d2": {"o1": 0.2, "o2": 0.3, "o3": 0.8}}}
    assert ia.cue_reassign(cue, ["d1", "d2"], ["o2", "o3"]) == {"d1": "o3", "d2": "o2"}
    assert ia.cue_reassign(cue, ["d9"], ["o1"]) == {}


def test_cue_twins_rejects_a_read_that_does_not_cover_the_labels():
    objs = _scene()
    w = _world(objs)
    bad = SimpleNamespace(twin_costs=lambda o, d, u: {"assign": {"demo_a": "ours_x", "demo_b": "ours_x"}})
    assert ia.cue_twins(_sk(w, "cues", bad), w, ["demo_a", "demo_b"], ["ours_x", "ours_y"]) is None
    none = SimpleNamespace(twin_costs=lambda o, d, u: None)
    assert ia.cue_twins(_sk(w, "cues", none), w, ["demo_a", "demo_b"], ["ours_x", "ours_y"]) is None


# ---------------------------------------------------------------------------------------------------- order: persist
def _seq_sk(objs, mode):
    return _sk(_world(objs), "one_to_one", imitate_order=mode), {"n": 0}


def test_persist_drops_a_one_read_departure_and_dates_real_ones_by_their_first_read(monkeypatch):
    """demo_b reads gone once (the demonstrator's shadow) at the first read, then present; demo_a leaves for good at
    the second read, demo_b at the sixth. v1 records demo_b first (a wrong order); persist records demo_a, demo_b."""
    reads = [{"demo_b"}, {"demo_a"}, {"demo_a"}, {"demo_a"}, {"demo_a"}, {"demo_a", "demo_b"}, {"demo_a", "demo_b"},
             {"demo_a", "demo_b"}, {"demo_a", "demo_b"}] + [{"demo_a", "demo_b"}] * 20
    orders = {}
    for mode in ("v1", "persist"):
        objs = _scene()
        sk, step = _seq_sk(objs, mode)

        class Ex:
            steps_left = 1500

            @property
            def steps_used(self):
                return step["n"]

            def hold(self, n):
                step["n"] += n

        sk.ex = Ex()
        it = iter(reads)
        priors = []

        def watch(sk_, region, prior):
            priors.append(prior)
            g = next(it)
            # perception (v1): a departure read once stays (the prior's gone list is carried)
            if prior and prior.get("gone"):
                g = g | {e["label"] for e in prior["gone"]}
            return {"gone": [{"label": l, "asset": objs[l].asset.key} for l in sorted(g)]}

        monkeypatch.setattr(ia, "watch", watch)
        monkeypatch.setattr(ia, "wait_demo_done", lambda *a: "home")
        monkeypatch.setattr(ia, "inside_xy", lambda o, c: True)
        monkeypatch.setattr(ia, "put_in_open", lambda *a, **k: SimpleNamespace(note=""))
        ia.copy_demonstration(sk, "basket0", "basket1", list(objs))
        orders[mode] = list(sk.refresh().notes["imitate"]["order"])
        if mode == "persist":
            assert all(not (p or {}).get("gone") for p in priors)      # perception's departure is per read
    assert orders["v1"] == ["demo_b", "demo_a"]
    assert orders["persist"] == ["demo_a", "demo_b"]


# ---------------------------------------------------------------------------------------------------- carry: reach
def _kin(left=(-0.3, -0.6), right=(0.3, -0.6)):
    def base(a):
        T = np.eye(4)
        T[:2, 3] = left if a == "left" else right
        return T
    return SimpleNamespace(base=base)


def test_far_side_is_the_other_arm_only_with_a_clear_margin():
    objs = {"o_far": _o("Rigid/watch/00000", (0.30, -0.10, 0.8)), "o_mid": _o("Rigid/watch/00000", (0.0, -0.10, 0.8)),
            "o_near": _o("Rigid/watch/00000", (-0.25, -0.10, 0.8)),
            "basket0": _o("Geometry/basket/00005", (-0.4, -0.15, 0.8))}
    sk = SimpleNamespace(world=_world(objs), kin=_kin())
    assert ia.far_side(sk, "o_far", "basket0") == "right"
    assert ia.far_side(sk, "o_mid", "basket0") is None
    assert ia.far_side(sk, "o_near", "basket0") is None


def test_centred_filter_drops_off_centre_grasps():
    gs = [SimpleNamespace(note="box close 0 approach -1 tilt -25 off +45mm"), SimpleNamespace(note="top axis 0 tilt 0"),
          SimpleNamespace(note="box close 1 approach +2 tilt +0 off +0mm")]
    assert [g.note for g in ia._centred(gs)] == ["top axis 0 tilt 0", "box close 1 approach +2 tilt +0 off +0mm"]


# ---------------------------------------------------------------------------------------------------- retry: refind
def test_refind_proposes_the_start_spot_only_when_the_belief_moved_away():
    objs = {"t": _o("Rigid/watch/00000", (-0.5, -0.25, 0.85)), "basket0": _o("Geometry/basket/00005", (-0.4, -0.15, 0.8))}
    calls = []
    prov = SimpleNamespace(confirm_pose=lambda l, p, q: calls.append((l, p.tolist())) or {"adopted": True})
    import dojo.skills.interactive as mod
    sk = SimpleNamespace(refresh=lambda: _world(objs), provider=prov, log=lambda *a: None)
    orig = mod.inside_xy
    try:
        mod.inside_xy = lambda o, c: False
        ia.refind_twin(sk, "t", "basket0", (np.array([0.29, -0.06, 0.78]), np.array([1.0, 0, 0, 0])))
        assert calls == [("t", [0.29, -0.06, 0.78])]
        calls.clear()
        ia.refind_twin(sk, "t", "basket0", (np.array([-0.49, -0.25, 0.85]), np.array([1.0, 0, 0, 0])))
        assert calls == []                                       # within 40 mm: nothing to re-find
        mod.inside_xy = lambda o, c: True
        ia.refind_twin(sk, "t", "basket0", (np.array([0.29, -0.06, 0.78]), np.array([1.0, 0, 0, 0])))
        assert calls == []                                       # in the container: done
    finally:
        mod.inside_xy = orig
    ia.refind_twin(SimpleNamespace(refresh=lambda: _world(objs), provider=SimpleNamespace(), log=lambda *a: None),
                   "t", "basket0", (np.zeros(3), np.array([1.0, 0, 0, 0])))   # G3 provider: no read, no write


def test_refind_retries_until_the_step_floor(monkeypatch):
    objs = _scene()
    left = {"n": 1000}

    class Ex:
        steps_used = 5

        @property
        def steps_left(self):
            return left["n"]

        def hold(self, n):
            pass

    sk = _sk(_world(objs), "one_to_one", imitate_retry="refind")
    sk.ex = Ex()
    tries = []
    monkeypatch.setattr(ia, "watch", lambda *a: {"gone": [{"label": l, "asset": objs[l].asset.key}
                                                          for l in ("demo_a", "demo_b")]})
    monkeypatch.setattr(ia, "wait_demo_done", lambda *a: "home")
    monkeypatch.setattr(ia, "inside_xy", lambda o, c: False)
    monkeypatch.setattr(ia, "_lift_open", lambda sk_: None)
    monkeypatch.setattr(ia, "refind_twin", lambda *a: None)
    monkeypatch.setattr(ia, "_steps_left", lambda sk_: left["n"])

    def put(sk_, label, container, **kw):
        tries.append(label)
        left["n"] -= 400
        return SimpleNamespace(note="slip")

    monkeypatch.setattr(ia, "put_in_open", put)
    r = ia.copy_demonstration(sk, "basket0", "basket1", list(objs))
    assert not r["ok"] and len(tries) == 3                      # 1000 -> 600 -> 200 -> -200: the 4th is refused


def test_elim_appends_the_one_unseen_departure_last(monkeypatch):
    """camera 100006: the 5th departure was never read; with "elim" the remaining demonstrator object goes last."""
    for mode, want in (("v1", ["ours_y"]), ("elim", ["ours_y", "ours_x"])):
        objs = _scene()
        sk = _sk(_world(objs), "one_to_one", imitate_order=mode)
        puts, placed = [], set()
        monkeypatch.setattr(ia, "watch", lambda *a: {"gone": [{"label": "demo_a", "asset": objs["demo_a"].asset.key}]})
        monkeypatch.setattr(ia, "wait_demo_done", lambda *a: "home")
        monkeypatch.setattr(ia, "inside_xy", lambda o, c: any(o is objs[l] for l in placed))

        def put(sk_, label, container, **kw):
            puts.append(label)
            placed.add(label)
            return SimpleNamespace(note="")

        monkeypatch.setattr(ia, "put_in_open", put)
        sk.ex = SimpleNamespace(steps_left=1500, steps_used=10_000, hold=lambda n: None)   # watching ends at once
        ia.copy_demonstration(sk, "basket0", "basket1", list(objs), max_wait=0)
        order = sk.refresh().notes["imitate"]["order"]
        assert order[:1] == ["demo_a"] and (order == ["demo_a", "demo_b"]) == (mode == "elim")
        assert len(puts) == len(want)
