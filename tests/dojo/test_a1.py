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
            super().__init__(frame, cams={"cam_head": cam}, require_gl=False)

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


# ------------------------------------------------------------------------------- the reviewer's must-fix round
def _sq(x, y, h=0.02):
    return [[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h]]


def test_a_write_onto_another_objects_final_footprint_is_refused():
    """S1: the overlap guard covers every object's FINAL pose: the believed footprints of the others and this tick's
    earlier writes. Two active identical pieces must not end up on one another."""
    objs = [{"label": "a", "key": "Rigid/box/00000", "pose7": _pose(0.0, -0.1), "touched": True},
            {"label": "c", "key": "Rigid/box/00000", "pose7": _pose(0.2, -0.1), "touched": True}]
    good = {"cam_head": [0.9, 900, 1.0, 0.9], "cam_left_wrist": [0.9, 6000, 1.0, 0.9]}
    bad = {"cam_head": [0.0, 900, 1.0, None]}

    def resp(c_to):
        hyps = {"a": [{"src": "incumbent", "pose7": _pose(0.0, -0.1), "support": bad, "footprint": _sq(0.0, -0.1)},
                      {"src": "det:cam_head:0", "pose7": _pose(0.1, -0.1), "support": good, "fit_iou": 0.9,
                       "footprint": _sq(0.1, -0.1)}],
                "c": [{"src": "incumbent", "pose7": _pose(0.2, -0.1), "support": bad, "footprint": _sq(0.2, -0.1)},
                      {"src": "det:cam_head:1", "pose7": _pose(*c_to), "support": good, "fit_iou": 0.9,
                       "footprint": _sq(*c_to)}]}
        return {"hyps": hyps, "assignment": {"a": ["cam_head", 0, 1], "c": ["cam_head", 1, 1]},
                "active": {"a": "release", "c": "release"}, "dets_hash": "h",
                "footprints": {"a": _sq(0.0, -0.1), "c": _sq(0.2, -0.1), "z": _sq(0.3, -0.1)}}

    trig = [Trigger("release", 9, "left", "a"), Trigger("release", 9, "right", "c")]
    w, ds = decide(resp((0.1, -0.1)), objs, trig, TickState(), APPLY, 10)       # c onto a's new pose
    assert sorted(w) == ["a"] and any(d.evidence.get("why") == "overlap:a" for d in ds if d.label == "c")
    w, ds = decide(resp((0.3, -0.1)), objs, trig, TickState(), APPLY, 10)       # c onto an untouched z
    assert "c" not in w and any(str(d.evidence.get("why")).startswith("overlap:z") for d in ds if d.label == "c")
    w, ds = decide(resp((0.2, -0.05)), objs, trig, TickState(), APPLY, 10)      # clear of everything
    assert sorted(w) == ["a", "c"]


def test_run_falsifies_the_close_at_the_tcp_and_passes_the_held_path():
    """``tick.run`` (the online and offline entry point): the pad volume of an empty close is centred on the TCP at
    the close, the held objects' path goes to the perception op, and nothing GT-like is in the request."""
    seen = []

    def client(obs, req):
        seen.append(req)
        return {"hyps": {"b": [{"src": "incumbent", "pose7": req["objects"][0]["pose7"],
                                "support": {"cam_head": [0.0, 900, 1.0, None]}}]},
                "assignment": {}, "active": {"b": "close_empty"}, "dets_hash": "h"}

    from dojo.runtime.tick import run
    st = TickState()
    hp = {"right": {"key": "Rigid/box/00000", "T_tcp_obj": np.eye(4).tolist()}}
    w, ds, resp = run({}, OBJ, {}, [Trigger("close_empty", 5, "left", "b")], {}, st, APPLY, client, 5,
                      q_path={"left": [[0.0] * 6]}, held_path=hp, close_tcp={"b": [0.01, -0.09, 0.8]})
    assert st.falsified["b"][0][0] == [0.01, -0.09, 0.8] and seen[0]["held_path"] == hp
    assert seen[0]["q_path"] == {"left": [[0.0] * 6]} and not {"gt", "layout_id", "task"} & set(seen[0])
    assert w == {} and "b" in st.uncertain                             # the falsified pose is not re-asserted
    st2 = TickState()
    run({}, OBJ, {}, [Trigger("close_empty", 5, "left", "b")], {}, st2, APPLY, client, 5)
    assert st2.falsified["b"][0][0] == pytest.approx(OBJ[0]["pose7"][:3])   # no TCP: the believed pose


def test_triggers_from_events_are_the_online_rules():
    from dojo.runtime.tick import triggers_from_events
    ev = [{"kind": "grip_check", "dropped": "air", "label": "a", "arm": "left"},
          {"kind": "grip_check", "dropped": "slip", "label": "b", "arm": "right"},
          {"kind": "grip_check", "verdict": "held", "label": "c", "arm": "left"},
          {"kind": "held_look", "held": False, "kept": True, "label": "d", "arm": "left"},
          {"kind": "held_look", "held": False, "label": "e", "arm": "right"}]
    lifted = set()
    t = triggers_from_events(ev, [("f", "left")], [("left", "g", 3, 0.80, 0.84), ("right", "h", 4, 0.80, 0.81)],
                             lifted, 50)
    assert [(x.kind, x.label, x.arm) for x in t] == [("close_empty", "a", "left"), ("slip", "b", "right"),
                                                      ("close_empty", "e", "right"), ("release", "f", "left"),
                                                      ("lift", "g", "left")]
    assert lifted == {("g", 3)}
    assert triggers_from_events([], [], [("left", "g", 3, 0.80, 0.90)], lifted, 60) == []   # once per grasp


def test_fixture_labels_follow_the_latest_fixture_drop_and_grasp():
    from dojo.runtime.tick import fixture_labels
    ev = [{"kind": "belief_drop", "label": "mug", "how": "held by a fixture", "step": 100},
          {"kind": "belief_drop", "label": "nut", "how": "on a fixture", "step": 120},
          {"kind": "belief_drop", "label": "cup", "how": "something else", "step": 130}]
    assert fixture_labels(ev) == {"mug", "nut"}
    assert fixture_labels(ev, held_now={"nut"}) == {"mug"}
    assert fixture_labels(ev, grasps=[("mug", 150)]) == {"nut"}          # re-grasped after the fixture took it
    assert fixture_labels(ev, grasps=[("mug", 90)]) == {"mug", "nut"}


def test_contact_sweeps_the_held_object_riding_the_tcp(world, monkeypatch):
    """M2: the contact sweep covers the held object's points at every executed q, not just the links."""
    from dojo.perception import held as H
    from dojo.perception import render
    RD = world.RD
    monkeypatch.setattr(render, "robot_link_meshes", lambda: {})
    target = _pose(0.0, -0.1)
    T = np.eye(4)
    T[:3, 3] = [0.0, -0.1, target[2] + 2 * HZ + 0.005]                 # the held box rides just over the target
    monkeypatch.setattr(H, "tcp_T", lambda arm, q, model=None: T.copy())
    model = SimpleNamespace(fk_world=lambda arm, q: {})
    objs = [{"label": "t", "key": "Rigid/box/00000", "pose7": target}]
    hp = {"left": {"key": "Rigid/box/00000", "T_tcp_obj": np.eye(4).tolist()}}
    qp = {"left": [[0.0] * 6]}
    assert RD.contact_suspects(qp, objs, {"left": "h"}, RD.RDConfig(), model=model, held_path=hp) == ["t"]
    assert RD.contact_suspects(qp, objs, {"left": "h"}, RD.RDConfig(), model=model) == []


def test_the_provider_sweeps_every_executed_action_since_the_last_tick():
    """M2 online: q_path is every 3rd executed action since the last TICK (not since the last refresh), and what each
    arm carried along it rides its TCP, from the attach log: a release tick still sweeps the carry."""
    prov, b = _prov()
    prov.a1, prov.a1_client = "apply", _Client()
    q = lambda v: {"left_arm_joint_state": [v] * 6 + [0.0], "right_arm_joint_state": [0.0] * 7}   # noqa: E731
    b.executed = [q(0.01 * i) for i in range(9)]
    T = np.eye(4)
    T[2, 3] = 0.05
    prov.belief.attached["right"] = ("block_2", T)                    # carried from action 0 ...
    prov._a1_log_attach(0)
    prov.belief.attached.pop("right")                                 # ... released at action 5
    prov._a1_log_attach(5)
    prov.a1_tick(b.obs, [Trigger("release", 9, "right", "block_2")])
    r = prov.a1_client.reqs[-1]
    assert [x[0] for x in r["q_path"]["left"]] == pytest.approx([0.0, 0.03, 0.06])
    seg = r["held_path"]["right"]
    assert [(x["i0"], x["i1"], x["label"]) for x in seg] == [(0, 2, "block_2")]   # q 0 and 3 carried it, q 6 not
    assert seg[0]["key"] == prov.belief.objects["block_2"].asset.key and seg[0]["T_tcp_obj"][2][3] == 0.05
    b.executed += [q(0.5), q(0.6)]
    prov.a1_tick(b.obs, [])                                               # no trigger: no tick, nothing consumed
    b.executed += [q(0.7)]
    prov.a1_tick(b.obs, [Trigger("hover", 12, "left", "block_1")])
    r = prov.a1_client.reqs[-1]
    assert [x[0] for x in r["q_path"]["left"]] == pytest.approx([0.5]) and not r.get("held_path")


def test_held_path_from_log_segments_the_carries():
    from dojo.runtime.tick import held_path_from_log
    T = np.eye(4)
    log = [(2, "left", "a", T), (8, "left", None, None), (9, "right", "b", T), (11, "left", "c", T)]
    hp = held_path_from_log(log, {"left": [0, 3, 6, 9, 12], "right": [0, 3, 6, 9, 12]}, {"a": "ka", "b": "kb", "c": "kc"})
    assert [(x["i0"], x["i1"], x["label"], x["key"]) for x in hp["left"]] == [(1, 3, "a", "ka"), (4, 5, "c", "kc")]
    assert [(x["i0"], x["i1"], x["label"]) for x in hp["right"]] == [(3, 5, "b")]
    assert held_path_from_log(log, {"left": [0, 1]}, {"a": "ka"}) == {}


def test_a_release_tick_flags_the_neighbour_the_carry_swept(world, monkeypatch):
    """The reviewer's M2 case: the arm carried an object over a neighbour, then released it; at the release tick
    nothing is held, the carry's segment of the path still sweeps the neighbour, and the carried object is not a
    suspect by its own points."""
    from dojo.perception import held as H
    from dojo.perception import render
    RD = world.RD
    monkeypatch.setattr(render, "robot_link_meshes", lambda: {})
    nb = _pose(0.0, -0.1)
    Ts = []
    for x in (-0.2, 0.0, 0.2):                                          # the TCP passes over the neighbour
        T = np.eye(4)
        T[:3, 3] = [x, -0.1, nb[2] + 2 * HZ + 0.005]
        Ts.append(T)
    monkeypatch.setattr(H, "tcp_T", lambda arm, q, model=None: Ts[int(round(q[0]))].copy())
    model = SimpleNamespace(fk_world=lambda arm, q: {})
    carried = _pose(0.2, -0.1)                                          # released at the end of the path
    objs = [{"label": "nb", "key": "Rigid/box/00000", "pose7": nb},
            {"label": "car", "key": "Rigid/box/00000", "pose7": carried}]
    qp = {"left": [[0.0] * 6, [1.0] * 6, [2.0] * 6]}
    seg = [{"i0": 0, "i1": 3, "label": "car", "key": "Rigid/box/00000", "T_tcp_obj": np.eye(4).tolist()}]
    assert RD.contact_suspects(qp, objs, {}, RD.RDConfig(), model=model, held_path={"left": seg}) == ["nb"]
    seg_late = [{**seg[0], "i0": 2}]                                    # carried only over the last q: misses nb
    assert RD.contact_suspects(qp, objs, {}, RD.RDConfig(), model=model, held_path={"left": seg_late}) == []


def test_the_overlap_guard_allows_supports_containers_and_stacks():
    """Reviewer round 3: an overlap blocks only within one height layer and never against a static support /
    container. A bottle into the bin and a block on a believed block are allowed; two same-key pieces on one spot are
    refused."""
    from dojo.runtime.tick import blocking_overlap
    sq = _sq(0.0, 0.0)
    bin_ = {"xy": _sq(0.0, 0.0, 0.08), "z": [0.75, 0.95], "static": True}
    bottle = {"xy": sq, "z": [0.76, 0.96], "static": False}
    assert blocking_overlap(bottle, bin_) == 0.0
    assert blocking_overlap(bottle, {**bin_, "static": False}) > 0.2                 # a movable box in its layer
    block, lower = {"xy": sq, "z": [0.80, 0.83]}, {"xy": sq, "z": [0.77, 0.80]}
    assert blocking_overlap(block, lower) == 0.0 and blocking_overlap(lower, block) == 0.0
    assert blocking_overlap(block, {"xy": sq, "z": [0.80, 0.83]}) == pytest.approx(1.0)   # one spot, one layer
    assert blocking_overlap(block, {"xy": sq, "z": [0.82, 0.90]}) == 0.0            # < half the smaller height
    assert blocking_overlap(sq, sq) == pytest.approx(1.0)                           # bare hulls: xy only


def test_the_per_hypothesis_guard_uses_the_same_rule(world):
    RD = world.RD
    at = _pose(0.0, -0.1)
    on_top = _pose(0.0, -0.1, z=at[2] + 2 * HZ)
    others = [{"label": "o", "key": "Rigid/box/00000", "pose7": at}]
    assert RD.footprint_overlap("Rigid/box/00000", at, others)[1] == pytest.approx(1.0)
    assert RD.footprint_overlap("Rigid/box/00000", on_top, others)[1] == 0.0
    assert RD.footprint_overlap("Rigid/box/00000", at, [{**others[0], "key": "Geometry/bin/00000"}])[1] == 0.0
    fp = RD.footprint("Geometry/bin/00000", at)
    assert fp["static"] and fp["z"][1] - fp["z"][0] == pytest.approx(2 * HZ, abs=1e-4)


def test_a_far_hover_relocation_rehovers_over_the_new_pose():
    """S3: a hover tick that moves the target > rehover_min_shift_mm returns a re-hover hint (skills hovers over the
    new pose and refines there) instead of refining from the old hover."""
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from tests_a1_env import FakeAssets, FakeEpisode, env_and_obs
    env, _ = env_and_obs()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", a1="apply"), assets=FakeAssets(), perception_client=object())
    pol.reset(FakeEpisode(env).info())
    prov, b = _prov()
    pol.bridge = b
    o = prov.belief.objects["block_1"]
    found = [float(o.pos[0]) + 0.04, float(o.pos[1]), float(o.pos[2])] + [float(v) for v in o.quat]
    prov.a1, prov.a1_client = "apply", _Client(found)
    out = pol._refine_hook(prov, None, "block_1", "left", "pick")
    assert out and np.allclose(out["rehover"][0], found[:3]) and np.allclose(o.pos, found[:3])
    assert any(e["kind"] == "a1_rehover" for e in pol.report["events"])


def test_decide_writes_into_a_bin_and_onto_a_block_but_not_onto_a_same_layer_piece():
    good = {"cam_head": [0.9, 900, 1.0, 0.9], "cam_left_wrist": [0.9, 6000, 1.0, 0.9]}
    bad = {"cam_head": [0.0, 900, 1.0, None]}
    objs = [{"label": "a", "key": "Rigid/bottle/00000", "pose7": _pose(0.0, -0.1), "touched": True}]

    def resp(other, z):
        cand = {"xy": _sq(0.2, -0.1), "z": z}
        return {"hyps": {"a": [{"src": "incumbent", "pose7": _pose(0.0, -0.1), "support": bad,
                                "footprint": {"xy": _sq(0.0, -0.1), "z": [0.75, 0.78]}},
                               {"src": "det:cam_head:0", "pose7": _pose(0.2, -0.1), "support": good, "fit_iou": 0.9,
                                "footprint": cand}]},
                "assignment": {"a": ["cam_head", 0, 1]}, "active": {"a": "release"}, "dets_hash": "h",
                "footprints": {"o": other}}

    trig = [Trigger("release", 9, "left", "a")]
    into_bin = {"xy": _sq(0.2, -0.1, 0.08), "z": [0.75, 0.95], "static": True}
    w, _ = decide(resp(into_bin, [0.76, 0.90]), objs, trig, TickState(), APPLY, 10)
    assert "a" in w
    block = {"xy": _sq(0.2, -0.1), "z": [0.75, 0.78], "static": False}
    w, _ = decide(resp(block, [0.78, 0.81]), objs, trig, TickState(), APPLY, 10)     # on the believed block
    assert "a" in w
    w, ds = decide(resp(block, [0.75, 0.78]), objs, trig, TickState(), APPLY, 10)    # the same spot, same layer
    assert w == {} and any(d.evidence.get("why") == "overlap:o" for d in ds)


def test_a_periodic_tick_activates_and_seeds_nothing():
    """The always-on arm (S): Trigger("periodic", s) between events. No object is active because of it and nothing is
    seeded; the touched objects stay active under the normal rules (here a second tick confirms a small object)."""
    objs = [{"label": "b", "key": "Rigid/box/00000", "pose7": _pose(0.0, -0.1), "small": True, "touched": True},
            {"label": "c", "key": "Rigid/box/00000", "pose7": _pose(0.2, -0.1)}]
    per = [Trigger("periodic", 20)]
    req = build_request(objs, {}, per, touched={"b"})
    assert {r["label"]: (r["active"], r["seeds"]) for r in req["objects"]} == {"b": ("touched", []), "c": (None, [])}
    assert req["lift"] == [] and req["corridor"] is None
    st = TickState()
    one = {"cam_head": [0.9, 900, 1.0, 0.9]}
    r = _resp("b", {"cam_head": [0.0, 900, 1.0, None]}, one, _pose(0.03, -0.1), src="own:cam_head:0", active="touched")
    w, _ = decide(r, objs, per, st, APPLY, 20)
    assert w == {} and "b" in st.pending                              # small: one view, first tick
    w, _ = decide(r, objs, [Trigger("periodic", 22)], st, APPLY, 22)
    assert "b" in w                                                   # the next tick confirms it


def test_a_far_twin_relocation_needs_two_cameras():
    """The lead's a-priori twin rule: an object with a same-key twin in the belief moves > 50 mm only when two cameras
    agree on the candidate; one head view (the dev read's ttt breaks: a correctly released piece moved 160-380 mm onto
    an identical piece's mask) is not enough. Near moves and twin-less objects keep the normal bar."""
    twins = OBJ + [{"label": "b2", "key": "Rigid/box/00000", "pose7": _pose(0.3, -0.1)}]
    inc = {"cam_head": [0.0, 150, 0.35, None]}
    head = {"cam_head": [0.68, 444, 1.0, 0.92]}
    two = {"cam_head": [0.68, 444, 1.0, 0.92], "cam_left_wrist": [0.7, 3000, 1.0, 0.9]}
    trig = [Trigger("release", 9, "left", "b")]
    far = _pose(0.2, -0.1)
    w, ds = decide(_resp("b", inc, head, far), twins, trig, TickState(), APPLY, 10)
    assert w == {}
    w, _ = decide(_resp("b", inc, two, far), twins, trig, TickState(), APPLY, 10)
    assert "b" in w
    w, _ = decide(_resp("b", inc, head, far), OBJ, trig, TickState(), APPLY, 10)       # no twin: one camera suffices
    assert "b" in w
    w, _ = decide(_resp("b", inc, head, _pose(0.04, -0.1)), twins, trig, TickState(), APPLY, 10)   # a near move
    assert "b" in w


def test_twin_agreement_counts_only_full_views_and_seed_score_detections():
    from dojo.runtime.tick import _agreeing_cams
    h = {"support": {"cam_head": [0.68, 444, 1.0, 0.92], "cam_left_wrist": [0.7, 3000, 0.6, 0.9],
                     "cam_right_wrist": [0.8, 3000, 1.0, 0.4]}}
    assert _agreeing_cams(h, APPLY) == 1                     # the low-visibility wrist and the 0.4 detection do not count
    twins = OBJ + [{"label": "b2", "key": "Rigid/box/00000", "pose7": _pose(0.3, -0.1)}]
    w, ds = decide(_resp("b", {"cam_head": [0.0, 150, 0.35, None]}, h["support"], _pose(0.2, -0.1)), twins,
                   [Trigger("release", 9, "left", "b")], TickState(), APPLY, 10)
    assert w == {}
    other_key = OBJ + [{"label": "b2", "key": "Rigid/box/00001", "pose7": _pose(0.3, -0.1)}]   # same category only
    w, _ = decide(_resp("b", {"cam_head": [0.0, 150, 0.35, None]}, h["support"], _pose(0.2, -0.1)), other_key,
                  [Trigger("release", 9, "left", "b")], TickState(), APPLY, 10)
    assert "b" in w


def test_held_identity_guard_lets_go_of_the_intended_object_on_a_confident_mismatch():
    """RuntimeConfig.held_identity_guard: a lift identity of another category (score >= 0.5, IoU >= 0.3) means the
    jaws took something else: the intended object goes back out of the jaws to its pose at the close (the pick's check
    then fails and the program re-plans). Off, or a weak / same-category detection: nothing changes."""
    def lift(cat_ok, score=0.9, iou=0.6):
        return {"hyps": {}, "assignment": {}, "active": {}, "dets_hash": "x",
                "lift": [{"arm": "left", "label": "block_1", "agrees": cat_ok, "exact": False, "other_labels": ["block_2"],
                          "det": {"key": "Geometry/stand/00000", "score": score, "px": 900, "iou_with_belief": iou}}]}

    for guard, resp, let_go in ((True, lift(False), True), (False, lift(False), False), (True, lift(True), False),
                                (True, lift(False, score=0.4), False), (True, lift(False, iou=0.1), False)):
        prov, b = _prov()
        o = prov.belief.objects["block_1"]
        p_close = o.pos.copy()
        T = np.eye(4)
        prov.belief.attached["left"] = ("block_1", T)
        prov.belief.grasp_z["left"] = (0.8, 1, np.r_[p_close, o.quat].astype(float))
        o.pos = o.pos + [0.0, 0.0, 0.05]                                  # carried up 5 cm
        prov.a1, prov.held_identity_guard = "apply", guard
        prov.a1_client = lambda obs, req, r=resp: r
        prov.a1_tick(b.obs, [Trigger("lift", 30, "left", "block_1")])
        ev = [e for e in prov.events if e["kind"] == "held_identity_mismatch"]
        assert ("left" not in prov.belief.attached) == let_go and bool(ev) == let_go
        if let_go:
            assert np.allclose(o.pos[:2], p_close[:2]) and abs(o.pos[2] - p_close[2]) < 0.01
            assert {"block_1", "block_2"} <= prov.touched and ev[0]["det"]["key"] == "Geometry/stand/00000"
