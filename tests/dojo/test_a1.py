"""A1 (stream W): region-free re-detection (``dojo.perception.redetect``) and the tick's write rules
(``dojo.runtime.tick``). A synthetic table scene (a box, a top-down camera; CPU silhouettes, no catalogue, no GPU)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from dojo.runtime.tick import Discrepancy, TickConfig, TickState, Trigger, build_request, decide, seeds_for  # noqa: E402

HX, HY, HZ = 0.03, 0.02, 0.015
Q0 = [1.0, 0.0, 0.0, 0.0]


def _box_asset(key="Rigid/box/00000", hx=HX, hy=HY, hz=HZ):
    V = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)], float)
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return SimpleNamespace(key=key, V=V, F=F, Vs=V, Fs=F, Vg=V, Fg=F, extents=np.array([2 * hx, 2 * hy, 2 * hz]),
                           info={"bbox_min": (-hx, -hy, -hz), "bbox_max": (hx, hy, hz)})


@pytest.fixture
def world(monkeypatch):
    """A top-down 'head' camera over the table, the box asset, the table-region priors made to explode if touched."""
    from dojo.perception import inverse_layout as IL
    from dojo.perception import redetect as RD
    from dojo.perception import render
    from dojo.perception import show_pose as SP
    from dojo.perception.cameras import TABLE_Z, Camera, intrinsics

    monkeypatch.setattr(render, "_GL", False)                 # CPU silhouettes
    asset = _box_asset()
    monkeypatch.setattr(render, "load_asset", lambda key: asset)
    monkeypatch.setattr(IL, "load_asset", lambda key: asset, raising=False)
    sm = IL.ShapeModel(asset.key, "default", asset, np.eye(3), -HZ, 2 * HZ, 2, False)
    monkeypatch.setattr(IL, "shape_models", lambda key: (sm,))
    # legality: the mid-episode path may not consult generator regions, yaw ranges, placement priors or label groups
    for name in ("origin_bounds", "yaw_range", "_placement_prior", "label_groups"):
        monkeypatch.setattr(IL, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("config prior used")))
    target = np.array([0.0, -0.1, TABLE_Z])
    cam = Camera("cam_head", intrinsics(13.0, 20.955), SP.look_at_camera(target + [0.0, -0.05, 0.45], target))

    class Scene(RD.Scene):
        def __init__(self, frame):
            super().__init__(frame, cams={"cam_head": cam})

        def robot(self, cam_name):
            return np.zeros((cam.h, cam.w), np.uint8)

    return SimpleNamespace(RD=RD, IL=IL, sm=sm, cam=cam, Scene=Scene, asset=asset)


def _pose(x, y, yaw=0.0, z=None):
    from dojo.perception.cameras import TABLE_Z
    z = TABLE_Z + HZ if z is None else z
    return [x, y, z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _mask(w, p7):
    from dojo.perception.cameras import pose_to_T
    from dojo.perception.render import silhouette, transform
    return silhouette(w.cam, transform(w.asset.Vs, pose_to_T(p7[:3], p7[3:])), w.asset.Fs)


class _Src:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, img):
        return self.dets


def _frame(w):
    return {"images": {"cam_head": np.zeros((w.cam.h, w.cam.w, 3), np.uint8)}, "state": {}}


def test_fit_in_mask_recovers_the_pose_without_any_region_prior(world):
    true = _pose(0.03, -0.12, 0.4)
    fr = _frame(world)
    fits = world.RD.fit_in_mask("Rigid/box/00000", "cam_head", world.Scene(fr), _mask(world, true), world.RD.RDConfig())
    assert fits and fits[0][1] > 0.85
    assert np.hypot(fits[0][0][0] - true[0], fits[0][0][1] - true[1]) < 0.004


def test_redetect_relocates_a_missing_object_into_its_unexplained_mask(world):
    from dojo.perception.seg_twin import Det
    RD = world.RD
    believed, true = _pose(-0.05, -0.08), _pose(0.04, -0.12, 0.3)
    fr = _frame(world)
    req = build_request([{"label": "b", "key": "Rigid/box/00000", "pose7": believed}], {},
                        [Trigger("release", 10, "left", "b")])
    # a mask of the same category but another asset is another object: never a seed
    out = RD.redetect(fr, req, _Src([Det(_mask(world, true), "Rigid/box/00001", 0.9)]), RD.RDConfig(),
                      scene=world.Scene(fr))
    assert [h["src"] for h in out["hyps"]["b"]] == ["incumbent"] and out["assignment"] == {}
    dets = [Det(_mask(world, true), "Rigid/box/00000", 0.9)]
    out = RD.redetect(fr, req, _Src(dets), RD.RDConfig(), scene=world.Scene(fr))
    hs = out["hyps"]["b"]
    assert hs[0]["src"] == "incumbent" and hs[0]["support"]["cam_head"][0] == 0.0
    assert out["assignment"]["b"][:2] == ["cam_head", 0]
    won = hs[out["assignment"]["b"][2]]
    assert np.hypot(won["pose7"][0] - true[0], won["pose7"][1] - true[1]) < 0.005
    assert won["support"]["cam_head"][0] > 0.85
    # the same request twice: the same output (deterministic)
    out2 = RD.redetect(fr, req, _Src(dets), RD.RDConfig(), scene=world.Scene(fr))
    assert out2["hyps"] == out["hyps"] and out2["dets_hash"] == out["dets_hash"]


def test_untouched_objects_are_promoted_only_when_clearly_visible_and_unexplained(world):
    from dojo.perception.seg_twin import Det
    RD = world.RD
    a, b = _pose(-0.05, -0.08), _pose(0.06, -0.12)
    fr = _frame(world)
    objs = [{"label": "a", "key": "Rigid/box/00000", "pose7": a}, {"label": "b", "key": "Rigid/box/00000", "pose7": b}]
    req = build_request(objs, {}, [])
    out = RD.redetect(fr, req, _Src([Det(_mask(world, a), "Rigid/box/00000", 0.9)]), RD.RDConfig(),
                      scene=world.Scene(fr))
    assert out["assoc"]["a"]["cam_head"][0] == 0 and out["active"] == {"b": "missing"}
    # below 32 x 32 px in its only camera: no promotion (the segmenter's recall there is too low to call it missing)
    out = RD.redetect(fr, req, _Src([Det(_mask(world, a), "Rigid/box/00000", 0.9)]),
                      RD.RDConfig(promote_min_px=10 ** 6), scene=world.Scene(fr))
    assert out["active"] == {}


def test_joint_assignment_gives_a_mask_to_one_object_only(world):
    from dojo.perception.seg_twin import Det
    RD = world.RD
    true = _pose(0.04, -0.12)
    fr = _frame(world)
    objs = [{"label": "x", "key": "Rigid/box/00000", "pose7": _pose(-0.08, -0.06)},
            {"label": "y", "key": "Rigid/box/00000", "pose7": _pose(-0.08, -0.16)}]
    req = build_request(objs, {}, [Trigger("release", 5, "left", "x"), Trigger("contact", 5, None, "y")])
    out = RD.redetect(fr, req, _Src([Det(_mask(world, true), "Rigid/box/00000", 0.9)]), RD.RDConfig(),
                      scene=world.Scene(fr))
    assert len(out["assignment"]) == 1


def test_an_elongated_neighbour_is_not_taken_and_an_explained_object_does_not_jump(world, monkeypatch):
    """Two lying bottles side by side (elongated, same category). B is believed where it is and its mask explains it;
    A is active (released) and not seen. A must not take B's mask, nor a mask whose fit would overlap B's footprint;
    and an object a camera associates refits in its own mask, never jumping to a far unexplained one."""
    from dojo.perception.seg_twin import Det
    from dojo.perception import inverse_layout as IL
    from dojo.perception import render
    RD = world.RD
    bottle = _box_asset("Rigid/bottle/00000", hx=0.10, hy=0.03, hz=0.03)
    monkeypatch.setattr(render, "load_asset", lambda key: bottle)
    sm = IL.ShapeModel(bottle.key, "lying", bottle, np.eye(3), -0.03, 0.06, 2, False)
    monkeypatch.setattr(IL, "shape_models", lambda key: (sm,))
    world.asset = bottle
    fr = _frame(world)
    a_bel, b_true = _pose(-0.02, -0.17, z=0.795), _pose(0.0, -0.10, z=0.795)
    overl = _pose(0.03, -0.07, z=0.795)                     # half on B
    objs = [{"label": "A", "key": bottle.key, "pose7": a_bel}, {"label": "B", "key": bottle.key, "pose7": b_true}]
    dets = [Det(_mask(world, b_true), bottle.key, 0.95), Det(_mask(world, overl), bottle.key, 0.9)]
    req = build_request(objs, {}, [Trigger("release", 10, "left", "A")])
    out = RD.redetect(fr, req, _Src(dets), RD.RDConfig(), scene=world.Scene(fr))
    assert out["assoc"]["B"]["cam_head"][0] == 0                       # B explains its mask
    srcs = [h["src"] for h in out["hyps"]["A"]]
    assert "det:cam_head:0" not in srcs                                # B's own mask is never A's seed
    for h in out["hyps"]["A"][1:]:
        assert h["overlap"][0] == "B" and h["overlap"][1] > 0.2        # the remaining fit sits half on B ...
    st = TickState()
    w, _ = decide(out, [{**o, "touched": True} for o in objs], [Trigger("release", 10, "left", "A")], st, APPLY, 10)
    assert "A" not in w                                                # ... so it is not taken
    # B active too (a contact), associated with its own mask: refits there, never jumps to the far mask
    far = _pose(0.12, -0.20, z=0.795)
    dets = [Det(_mask(world, b_true), bottle.key, 0.95), Det(_mask(world, far), bottle.key, 0.9)]
    req = build_request(objs, {}, [Trigger("contact", 10, None, "B")])
    out = RD.redetect(fr, req, _Src(dets), RD.RDConfig(), scene=world.Scene(fr))
    srcs = [h["src"] for h in out["hyps"]["B"]]
    assert "own:cam_head:0" in srcs and not any(s_.startswith("det:") for s_ in srcs)


def test_the_tick_refuses_ground_truth_in_its_input(world):
    RD = world.RD
    fr = _frame(world)
    req = build_request([{"label": "b", "key": "Rigid/box/00000", "pose7": _pose(0, -0.1)}], {}, [])
    for bad in ({**fr, "gt": []}, {**fr, "state": {"_dev_gt": []}}):
        with pytest.raises(ValueError):
            RD.redetect(bad, req, _Src([]), RD.RDConfig(), scene=world.Scene(fr))
    with pytest.raises(ValueError):
        RD.redetect(fr, {**req, "layout_id": 100003}, _Src([]), RD.RDConfig(), scene=world.Scene(fr))


# ------------------------------------------------------------------------------------------- decide (write rules)
def _resp(label, inc_sup, cand_sup, cand7, inc7=None, src="det:cam_head:0", active="release"):
    inc7 = inc7 or _pose(0.0, -0.1)
    hs = [{"src": "incumbent", "pose7": inc7, "support": inc_sup},
          {"src": src, "pose7": cand7, "support": cand_sup, "fit_iou": 0.9}]
    asg = {label: ["cam_head", 0, 1]} if src.startswith("det:") else {}
    return {"hyps": {label: hs}, "assignment": asg, "active": {label: active}, "dets_hash": "h"}


OBJ = [{"label": "b", "key": "Rigid/box/00000", "pose7": _pose(0.0, -0.1), "touched": True}]
APPLY = TickConfig(mode="apply")


def test_a_supported_relocation_writes_and_logs():
    st = TickState()
    w, ds = decide(_resp("b", {"cam_head": (0.05, 900)}, {"cam_head": (0.9, 900), "cam_left_wrist": (0.8, 5000)},
                         _pose(0.06, -0.1)), OBJ, [Trigger("release", 9, "left", "b")], st, APPLY, 10)
    assert w["b"][:2] == pytest.approx([0.06, -0.1]) and ds[0].kind == "relocated" and ds[0].applied
    assert ds[0].before7[:2] == pytest.approx([0.0, -0.1]) and st.collapsed["b"] == w["b"]


def test_the_incumbent_competes_and_a_near_tie_never_writes():
    st = TickState()
    w, ds = decide(_resp("b", {"cam_head": (0.69, 900)}, {"cam_head": (0.70, 900)}, _pose(0.03, -0.1)), OBJ,
                   [Trigger("release", 9, "left", "b")], st, APPLY, 10)
    assert w == {} and not any(d.kind == "relocated" for d in ds)


def test_no_write_below_the_resolution_floor_and_small_moves_need_a_tight_fit():
    st = TickState()
    w, _ = decide(_resp("b", {"cam_head": (0.1, 900)}, {"cam_head": (0.95, 900)}, _pose(0.002, -0.1)), OBJ,
                  [Trigger("release", 9, "left", "b")], st, APPLY, 10)
    assert w == {}                                        # 2 mm < max(3 mm, 1.5 head px = 3.6 mm)
    w, ds = decide(_resp("b", {"cam_head": (0.1, 900)}, {"cam_head": (0.7, 900), "cam_left_wrist": (0.6, 3000)},
                         _pose(0.007, -0.1)), OBJ, [Trigger("release", 9, "left", "b")], st, APPLY, 11)
    assert w == {}                                        # 7 mm at IoU 0.7 < 0.8


def test_small_and_fixture_held_objects_need_two_views_or_two_ticks():
    small = [{**OBJ[0], "small": True}]
    st = TickState()
    r = _resp("b", {"cam_head": (0.0, 900)}, {"cam_head": (0.9, 900)}, _pose(0.05, -0.1))
    w, _ = decide(r, small, [Trigger("release", 9, "left", "b")], st, APPLY, 10)
    assert w == {} and "b" in st.pending                   # one camera: wait for a second tick
    w, ds = decide(r, small, [Trigger("hover", 19, "left", "b")], st, APPLY, 20)
    assert w and ds[0].kind == "relocated"                 # the same hypothesis again: written
    # a nut on its thread (fixture-held): a 20 mm "correction" never writes, a gross one with two views does
    fx = [{**OBJ[0], "fixture": True}]
    st = TickState()
    w, _ = decide(_resp("b", {"cam_head": (0.2, 900)}, {"cam_head": (0.9, 900), "cam_right_wrist": (0.9, 4000)},
                        _pose(0.02, -0.1)), fx, [Trigger("release", 9, "left", "b")], st, APPLY, 10)
    assert w == {}
    w, _ = decide(_resp("b", {"cam_head": (0.0, 900)}, {"cam_head": (0.9, 900), "cam_right_wrist": (0.9, 4000)},
                        _pose(0.08, -0.1)), fx, [Trigger("release", 9, "left", "b")], st, APPLY, 10)
    assert w


def test_an_empty_close_never_reasserts_the_falsified_pose():
    st = TickState()
    st.falsified["b"] = [([0.0, -0.1, 0.78], [0.02, 0.02, 0.03])]
    r = _resp("b", {"cam_head": (0.0, 900)}, {"cam_head": (0.9, 900), "cam_left_wrist": (0.9, 5000)},
              _pose(0.005, -0.1), src="near_miss")
    w, ds = decide(r, OBJ, [Trigger("close_empty", 9, "left", "b")], st, APPLY, 10)
    assert w == {}                                         # the only candidate sits in the pad volume
    assert [d.kind for d in ds] == ["uncertain"] and "b" in st.uncertain


def test_log_mode_reports_but_never_writes():
    st = TickState()
    w, ds = decide(_resp("b", {"cam_head": (0.05, 900)}, {"cam_head": (0.9, 900), "cam_left_wrist": (0.8, 5000)},
                         _pose(0.06, -0.1)), OBJ, [Trigger("release", 9, "left", "b")], st, TickConfig(mode="log"), 10)
    assert w == {} and ds[0].kind == "relocated" and not ds[0].applied


def test_never_left_needs_a_real_lift_with_no_rise_evidence():
    g7, d7 = _pose(0.0, -0.1), _pose(0.0, -0.12)
    t = Trigger("release", 50, "left", "nut")
    assert [s["src"] for s in seeds_for(t, {}, {"grasp7": g7, "drop7": d7, "lift_m": 0.0})] == ["under_jaws"]  # a turn
    assert [s["src"] for s in seeds_for(t, {}, {"grasp7": g7, "lift_m": 0.08, "rose": True})] == []           # it rose
    assert [s["src"] for s in seeds_for(t, {}, {"grasp7": g7, "lift_m": 0.08})] == ["never_left"]
    assert seeds_for(Trigger("close_empty", 50, "left", "nut"), {}, {"grasp7": g7, "lift_m": 0.08}) == []


def test_the_screws_ratchet_gives_no_write():
    """Rendered arm 9 fasten_screws: the nut turned in place (no lift), the read fits poorly; the incumbent (the
    kinematic belief, one turn lower) and the "pre-grasp" pose one turn higher score alike -> nothing writes."""
    nut = [{"label": "nut0", "key": "Rigid/factory_nut/00000", "pose7": _pose(0.2, -0.15, z=0.8126), "touched": True,
            "small": True, "fixture": True}]
    st = TickState()
    r = _resp("nut0", {"cam_head": (0.69, 300), "cam_right_wrist": (0.70, 2500)},
              {"cam_head": (0.70, 300), "cam_right_wrist": (0.70, 2500)}, _pose(0.2, -0.15, z=0.8156),
              inc7=_pose(0.2, -0.15, z=0.8126), src="never_left")
    for step in (10, 20, 30):
        w, ds = decide(r, nut, [Trigger("release", step - 1, "right", "nut0")], st, APPLY, step)
        assert w == {} and not any(d.kind == "relocated" for d in ds)


def test_added_obstacles_need_two_views_or_two_ticks():
    st = TickState()
    a = {"key": "Rigid/bottle/00002", "pose7": _pose(0.1, -0.1), "support": {"cam_head": (0.7, 800)}, "corridor_m": 0.04}
    _, ds = decide({"hyps": {}, "added": [a]}, [], [], st, APPLY, 10)
    assert ds == [] and st.added_pending
    _, ds = decide({"hyps": {}, "added": [a]}, [], [], st, APPLY, 20)
    assert [d.kind for d in ds] == ["added"] and ds[0].label.startswith("det_bottle_")


def test_discrepancy_round_trips_to_a_dict():
    d = Discrepancy("relocated", "b", 3, "release", [0] * 7, [1] * 7, {"x": 1}, True)
    assert d.to_dict()["after7"] == [1] * 7


# ------------------------------------------------------------------------------------------- the provider (runtime)
def _prov():
    from dojo.runtime.belief import RuntimeProvider
    from dojo.runtime.bridge import Bridge
    from dojo.runtime.estimate import objects_from_gt_rows
    from tests_a1_env import STACK, FakeAssets, env_and_obs
    env, obs = env_and_obs()
    b = Bridge()
    b.obs, b.fresh = obs, True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    return prov, b


class _Client:
    """The perception op "redetect": block_1 found 60 mm away by two cameras after its empty close."""

    def __init__(self, found7=None):
        self.reqs, self.found7 = [], found7

    def __call__(self, obs, req):
        self.reqs.append(req)
        hyps = {}
        for o in req["objects"]:
            if not o["active"]:
                continue
            hs = [{"src": "incumbent", "pose7": o["pose7"], "support": {"cam_head": (0.0, 900)}}]
            if o["label"] == "block_1" and self.found7 is not None:
                hs.append({"src": "det:cam_head:0", "pose7": self.found7, "fit_iou": 0.9,
                           "support": {"cam_head": (0.85, 900), "cam_left_wrist": (0.9, 6000)}})
            hyps[o["label"]] = hs
        asg = {"block_1": ["cam_head", 0, 1]} if self.found7 is not None and "block_1" in hyps else {}
        return {"hyps": hyps, "assignment": asg, "active": {o["label"]: o["active"] for o in req["objects"]
                                                              if o["active"]}, "dets_hash": "x", "time_s": {"total": 0.1}}


def test_provider_ticks_after_an_empty_close_and_writes_once_found():
    prov, b = _prov()
    o = prov.belief.objects["block_1"]
    found = [float(o.pos[0]) + 0.06, float(o.pos[1]), float(o.pos[2])] + [float(v) for v in o.quat]
    prov.a1, prov.a1_client = "apply", _Client(found)
    prov.events.append({"kind": "grip_check", "dropped": "air", "label": "block_1", "arm": "left", "step": 40})
    v0 = prov.versions(["block_1", "block_0"])
    prov.a1_tick(b.obs, prov._a1_triggers(b.obs, []))
    req = prov.a1_client.reqs[0]
    act = {r["label"]: r["active"] for r in req["objects"]}
    assert act["block_1"] == "close_empty" and "gt" not in req and "layout_id" not in req
    assert np.allclose(o.pos, found[:3]) and o.source == "a1"
    assert prov._a1_state.falsified["block_1"]                     # the pad volume of the empty close
    assert prov.fresh(v0) == ["block_1"]                          # a semantic move bumps the version ...
    v1 = prov.versions(["block_1"])
    assert prov.fresh(v1) == []                                   # ... once
    kinds = [e["kind"] for e in prov.events]
    assert "a1_tick" in kinds and "a1_discrepancy" in kinds and "belief_write" in kinds
    w = next(e for e in prov.events if e["kind"] == "belief_write")
    assert w["label"] == "block_1" and w["writer"] == "a1" and len(w["before7"]) == 7
    # the same trigger is consumed once: a second tick without new events does not run the op
    n = len(prov.a1_client.reqs)
    prov.a1_tick(b.obs, prov._a1_triggers(b.obs, []))
    assert len(prov.a1_client.reqs) == n


def test_provider_log_mode_and_unsupported_close_marks_uncertain():
    prov, b = _prov()
    o = prov.belief.objects["block_1"]
    p0 = o.pos.copy()
    prov.a1, prov.a1_client = "log", _Client(found7=None)
    prov.events.append({"kind": "grip_check", "dropped": "air", "label": "block_1", "arm": "left", "step": 40})
    prov.a1_tick(b.obs, prov._a1_triggers(b.obs, []))
    assert np.allclose(o.pos, p0) and not o.uncertain           # log: nothing written, no flag
    prov, b = _prov()
    o = prov.belief.objects["block_1"]
    prov.a1, prov.a1_client = "apply", _Client(found7=None)
    prov.events.append({"kind": "grip_check", "dropped": "air", "label": "block_1", "arm": "left", "step": 40})
    prov.a1_tick(b.obs, prov._a1_triggers(b.obs, []))
    assert np.allclose(o.pos, p0) and o.uncertain               # never re-asserted as certain; nothing found


def test_kinematic_carry_never_bumps_a_version_and_corridor_newcomers_are_stale():
    prov, b = _prov()
    bel = prov.belief
    prov._bump_versions()
    v0 = prov.versions(sorted(bel.objects))
    bel.attached["left"] = ("block_1", np.eye(4))
    bel.objects["block_1"].pos = bel.objects["block_1"].pos + [0.0, 0.0, 0.10]   # carried up 10 cm
    assert prov.fresh(v0) == []
    bel.objects["block_1"].pos = bel.objects["block_1"].pos - [0.0, 0.0, 0.10]   # put back where it was
    bel.attached.pop("left")
    assert prov.fresh(v0) == []
    c = bel.objects["block_2"].center_w
    assert prov.fresh({k: v for k, v in v0.items() if k != "block_2"}, corridor=[c, c + [0, 0, 0.12]]) == ["block_2"]


def test_a_whole_runtime_episode_with_a1_on_ticks_and_stacks():
    """RuntimePolicy with ``a1="apply"`` over the kinematic env and a fake perception service whose "redetect" sees
    every object where the GT is (so nothing must be written): the episode still stacks, the tick runs at hovers,
    lifts and releases, and every A1 event is well-formed."""
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from tests_a1_env import FakeAssets, FakeEpisode, env_and_obs, run_episode
    env, _ = env_and_obs()

    class Service:
        ticks = []

        def estimate(self, frames, vocabulary, fold, prior_record=None, fuse_mode=None):
            return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": o["label"],
                                         "default_pos": list(o["pos"]), "default_ori": [1, 0, 0, 0],
                                         "_perception": {"confidence": 0.9}} for o in env.objects]}}

        def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None, **kw):
            return {objects[0]["label"]: {"ok": False, "why": "perception not ok"}}

        def check(self, frame, objects, fold):
            return {o["label"]: {"head": {"prec": 1.0, "n_vis": 500, "n_px": 600}} for o in objects}

        def look_plan(self, *a, **k):
            return []

        def redetect(self, frame, req, cfg=None):
            self.ticks.append(sorted({t for t in (r["active"] for r in req["objects"]) if t}))
            hyps = {r["label"]: [{"src": "incumbent", "pose7": r["pose7"], "support": {"cam_head": [0.9, 800]}}]
                    for r in req["objects"] if r["active"]}
            return {"hyps": hyps, "assignment": {}, "active": {r["label"]: r["active"] for r in req["objects"]
                                                                if r["active"]}, "dets_hash": "h", "time_s": {"total": 0.01}}

    pc = Service()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", refine=True, a1="apply"), assets=FakeAssets(),
                        perception_client=pc)
    rep = run_episode(pol, FakeEpisode(env))
    assert rep["error"] is None, rep["error"]
    ev = rep["events"]
    ticks = [e for e in ev if e["kind"] == "a1_tick"]
    assert ticks and any(("hover", ) == tuple(t[0] for t in e["triggers"])[:1] for e in ticks)
    assert not any(e["kind"] == "a1_discrepancy" and e.get("a1_kind") == "relocated" for e in ev)   # nothing to fix
    assert all(e["writes"] == [] for e in ticks)
    p = sorted((np.asarray(o["pos"]) for o in env.objects), key=lambda v: v[2])
    assert all(np.linalg.norm(a[:2] - b[:2]) < 0.006 for a, b in zip(p[:-1], p[1:]))   # stacked


def test_a_low_visibility_view_never_moves_a_belief():
    """M's arm-8 audit: a fit from a view showing 0.53-0.63 of the object slid 30-47 mm along it. A candidate whose only
    supporting views show < 0.9 of it does not write; the same support from a full view does."""
    st = TickState()
    part = {"cam_left_wrist": [0.9, 6000, 0.6, 0.8], "cam_head": [0.0, 900, 1.0, None]}
    w, ds = decide(_resp("b", {"cam_head": [0.0, 900, 1.0, None]}, part, _pose(0.04, -0.1)), OBJ,
                   [Trigger("release", 9, "left", "b")], st, APPLY, 10)
    assert w == {} and not any(d.kind == "relocated" for d in ds)
    full = {"cam_left_wrist": [0.9, 6000, 0.95, 0.8], "cam_head": [0.0, 900, 1.0, None]}
    w, ds = decide(_resp("b", {"cam_head": [0.0, 900, 1.0, None]}, full, _pose(0.04, -0.1)), OBJ,
                   [Trigger("release", 9, "left", "b")], st, APPLY, 11)
    assert w and ds[0].evidence["cand"]["scores"]["cam_left_wrist"] == 0.8     # the detection score is logged


def test_borderline_detections_explain_but_never_seed(world):
    """C's JPEG-vs-raw check: detections at 0.25-0.46 flip between encodings. They may explain an object (block a
    "missing" promotion) but never seed or support a write (seed_min_score 0.5)."""
    from dojo.perception.seg_twin import Det
    RD = world.RD
    believed, true = _pose(-0.05, -0.08), _pose(0.04, -0.12, 0.3)
    fr = _frame(world)
    req = build_request([{"label": "b", "key": "Rigid/box/00000", "pose7": believed}], {},
                        [Trigger("release", 10, "left", "b")])
    out = RD.redetect(fr, req, _Src([Det(_mask(world, true), "Rigid/box/00000", 0.4)]), RD.RDConfig(),
                      scene=world.Scene(fr))
    assert [h["src"] for h in out["hyps"]["b"]] == ["incumbent"]
    # explaining: an untouched object over a 0.4 detection is associated (not promoted "missing")
    req = build_request([{"label": "c", "key": "Rigid/box/00000", "pose7": true}], {}, [])
    out = RD.redetect(fr, req, _Src([Det(_mask(world, true), "Rigid/box/00000", 0.4)]), RD.RDConfig(),
                      scene=world.Scene(fr))
    assert out["assoc"]["c"] and out["active"] == {}
