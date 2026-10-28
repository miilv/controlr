"""Background perception (dojo.perception.bgperc): fusion rules on a fake refine, and the sequence capture policy."""
import gzip
import json

import numpy as np

from dojo.perception.bgperc import BackgroundPerceiver, BgConfig
from dojo.perception.seq_capture import SeqReplayPolicy

Q = [1.0, 0.0, 0.0, 0.0]


class FakeRefine:
    """Returns the queued fits in order: dict(ok, xy=(x, y) | None, iou, vis); records the calls."""

    def __init__(self, fits):
        self.fits = list(fits)
        self.calls = []

    def __call__(self, key, img, state, prior, arm, others, radius):
        self.calls.append({"key": key, "prior": list(prior), "arm": arm, "radius": radius})
        f = self.fits.pop(0)
        xy = f.get("xy")
        pose = list(prior) if xy is None else [xy[0], xy[1], prior[2]] + list(prior[3:])
        out = {"ok": f["ok"], "iou": f["iou"], "visible_frac": f["vis"], "pose": pose if f["ok"] else list(prior)}
        if not f["ok"] and xy is not None:
            out["fit_pose"] = pose
        return out


def make(fits, cfg=None, objs=None):
    bp = BackgroundPerceiver(cfg or BgConfig(every=1, step_gap=1), refine_fn=FakeRefine(fits))
    bp.init(objs or {"a": {"key": "Rigid/block/00000", "pos": [0.0, 0.0, 0.8], "quat": Q, "conf": 0.4}})
    # every believed object is in clear view of the left wrist (no catalogue / cameras in unit tests)
    bp.visibility = lambda frame, labels: [{"label": lab, "arm": "left", "px": 1000, "trunc": 0.0, "robot": 0.0,
                                            "dist": 0.3} for lab in labels]
    return bp


FRAME = {"images": {"cam_left_wrist": np.zeros((4, 4, 3), np.uint8)}, "state": {}}


def test_close_fit_confirms_without_moving():
    bp = make([{"ok": True, "xy": (0.002, 0.0), "iou": 0.9, "vis": 0.9}])
    assert bp.step(FRAME, 0, required=["a"]) == []
    t = bp.tracks["a"]
    assert t.confirmed == 0 and t.moved is None
    assert np.allclose(t.pose[:2], [0.0, 0.0])


def test_far_fit_needs_a_second_view_unless_strong():
    cfg = BgConfig(every=1, step_gap=3)
    bp = make([{"ok": True, "xy": (0.02, 0.0), "iou": 0.8, "vis": 0.9},
               {"ok": True, "xy": (0.021, 0.001), "iou": 0.8, "vis": 0.9}], cfg)
    assert bp.step(FRAME, 0, required=["a"]) == []              # a candidate: one view only
    assert bp.tracks["a"].confirmed is None
    assert bp.step(FRAME, 1, required=["a"]) == []              # too soon for a second view (step_gap)
    ch = bp.step(FRAME, 3, required=["a"])
    assert len(ch) == 1 and ch[0]["n_views"] == 2
    assert np.allclose(bp.tracks["a"].pose[:2], [0.0205, 0.0005], atol=1e-6)
    strong = make([{"ok": True, "xy": (0.012, 0.0), "iou": 0.95, "vis": 0.9}])
    assert len(strong.step(FRAME, 0, required=["a"])) == 1     # one very good fit within strong_mm moves it


def test_fit_on_a_neighbour_is_ignored():
    bp = make([{"ok": True, "xy": (0.09, 0.0), "iou": 0.95, "vis": 0.9}])
    assert bp.step(FRAME, 0, required=["a"]) == []
    assert bp.tracks["a"].obs == [] and bp.tracks["a"].confirmed is None


def test_absence_leads_to_a_search_and_doubt():
    bp = make([{"ok": False, "xy": None, "iou": 0.05, "vis": 0.9},
               {"ok": False, "xy": None, "iou": 0.05, "vis": 0.9}])
    bp.step(FRAME, 0, required=["a"])
    assert bp.tracks["a"].absent == 1 and bp._refine.calls[0]["radius"] is None
    bp.step(FRAME, 1, required=["a"])
    assert bp._refine.calls[1]["radius"] == bp.cfg.search_radius    # required + empty place: radius search
    assert bp.tracks["a"].doubt
    assert bp.needs_look(["a"]) == [{"label": "a", "why": "doubt", "absent": 2}]


def test_touch_starts_a_new_epoch_and_held_objects_are_skipped():
    bp = make([{"ok": True, "xy": (0.0, 0.0), "iou": 0.9, "vis": 0.9}])
    bp.step(FRAME, 0, required=["a"])
    assert bp.tracks["a"].confirmed == 0
    bp.touch("a", held=True)
    assert bp.tracks["a"].confirmed is None and bp.tracks["a"].epoch == 1
    assert bp.step(FRAME, 5, required=["a"]) == [] and len(bp._refine.calls) == 1


def test_needs_look_lists_unconfirmed_low_confidence_required_objects():
    bp = make([], objs={"a": {"key": "k", "pos": [0, 0, 0.8], "quat": Q, "conf": 0.3},
                        "b": {"key": "k", "pos": [0.2, 0, 0.8], "quat": Q, "conf": 0.9}})
    assert bp.needs_look(["a", "b"]) == [{"label": "a", "why": "unconfirmed_low_conf", "conf": 0.3}]


def test_budget_one_refine_per_step_required_first():
    bp = make([{"ok": True, "xy": (0.2, 0.0), "iou": 0.9, "vis": 0.9}],
              objs={"a": {"key": "k", "pos": [0, 0, 0.8], "quat": Q, "conf": 0.9},
                    "b": {"key": "k", "pos": [0.2, 0, 0.8], "quat": Q, "conf": 0.9}})
    bp.step(FRAME, 0, required=["b"])
    assert len(bp._refine.calls) == 1 and bp._refine.calls[0]["prior"][0] == 0.2


def test_seq_replay_policy_keeps_every_observation(tmp_path, monkeypatch):
    acts = [{"left_arm_joint_state": [0.0] * 6, "left_ee_joint_state": [1.0 - 0.5 * i],
             "right_arm_joint_state": [0.0] * 6, "right_ee_joint_state": [1.0]} for i in range(3)]
    src = tmp_path / "run"
    src.mkdir()
    with gzip.open(src / "stack_blocks_perception_verify-bundle_100001_actions.json.gz", "wt") as f:
        json.dump({"actions": acts}, f)
    monkeypatch.setenv("DOJO_SEQ_SRC", str(src))
    monkeypatch.setenv("DOJO_SEQ_OUT", str(tmp_path / "seq"))
    pol = SeqReplayPolicy()
    pol.reset({"task": "stack_blocks", "layout_id": 100001, "instruction": "stack", "step_lim": 10})
    img = np.zeros((8, 8, 3), np.uint8)
    n, out = 0, None
    while True:
        obs = {"images": {"cam_head": img, "cam_left_wrist": img}, "state": {"left_ee_pose": np.zeros(7)},
               "steps_used": n, "_dev_gt": [{"label": "block0", "pos": [0, 0, 0.8]}]}
        out = pol.next(obs)
        if out.get("done"):
            break
        n += len(out["actions"])
    d = tmp_path / "seq" / "stack_blocks_100001"
    rows = [json.loads(ln) for ln in open(d / "steps.jsonl")]
    assert [r["step"] for r in rows] == [0, 1, 2, 3]
    assert rows[2]["grip_cmds"] == [{"left_ee_joint_state": 0.5, "right_ee_joint_state": 1.0}]
    assert rows[0]["gt"][0]["label"] == "block0"
    assert (d / "frames" / "0000_cam_head.png").exists() and (d / "frames" / "0003_cam_left_wrist.jpg").exists()
    assert pol.episode_report()["seq"]["replayed"] == 3


def test_reread_moves_after_two_agreeing_reads_across_labels():
    objs = {"block0": {"key": "Rigid/block/00000", "pos": [0.0, 0.0, 0.8], "quat": Q, "conf": 0.9},
            "block1": {"key": "Rigid/block/00000", "pos": [0.3, 0.0, 0.8], "quat": Q, "conf": 0.9}}
    bp = make([], BgConfig(every=1, reread_every=10), objs)
    # the re-read names them the other way round (interchangeable labels) and finds block1 60 mm off
    r = {"block1": {"key": "Rigid/block/00000", "pos": [0.001, 0.0, 0.8], "quat": Q, "conf": 0.9},
         "block0": {"key": "Rigid/block/00000", "pos": [0.3, 0.06, 0.8], "quat": Q, "conf": 0.9}}
    assert bp.reread(0, r) == []
    assert bp.tracks["block0"].confirmed == 0 and bp.tracks["block1"].rereads
    assert not bp.reread_due(5) and bp.reread_due(10)
    r2 = {"block0": {"key": "Rigid/block/00000", "pos": [0.302, 0.061, 0.8], "quat": Q, "conf": 0.9}}
    ch = bp.reread(10, r2)
    assert len(ch) == 1 and ch[0]["label"] == "block1" and ch[0]["reread"]
    assert np.allclose(bp.tracks["block1"].pose[:2], [0.301, 0.0605], atol=1e-6)


def test_reread_ignores_held_and_unconfident_readings():
    bp = make([], BgConfig(every=1, reread_every=10))
    bp.reread(0, {"a": {"key": "Rigid/block/00000", "pos": [0.05, 0, 0.8], "quat": Q, "conf": 0.2}})
    assert bp.tracks["a"].rereads == []
    bp.touch("a", held=True)
    bp.reread(10, {"a": {"key": "Rigid/block/00000", "pos": [0.05, 0, 0.8], "quat": Q, "conf": 0.9}})
    assert bp.tracks["a"].rereads == []


def test_reread_drops_readings_at_a_held_object():
    objs = {"m0": {"key": "Rigid/mug/00000", "pos": [0.0, 0.0, 0.8], "quat": Q, "conf": 0.9},
            "m1": {"key": "Rigid/mug/00000", "pos": [0.2, 0.0, 0.8], "quat": Q, "conf": 0.9}}
    bp = make([], BgConfig(every=1, reread_every=1), objs)
    bp.touch("m1", held=True)
    bp.tracks["m1"].pose[:3] = [0.25, 0.1, 1.0]                 # carried
    r = {"m0": {"key": "Rigid/mug/00000", "pos": [0.25, 0.1, 0.99], "quat": Q, "conf": 0.9}}   # the held mug
    for s in (0, 1):
        assert bp.reread(s, r) == []
    assert bp.tracks["m0"].rereads == [] and np.allclose(bp.tracks["m0"].pose[:2], [0.0, 0.0])


def test_sync_starts_epochs_on_external_changes_only():
    bp = make([{"ok": True, "xy": (0.02, 0.0), "iou": 0.95, "vis": 0.9}])
    bp.sync({"a": {"key": "Rigid/block/00000", "pos": [0.0, 0.0, 0.8], "quat": Q}})
    assert bp.tracks["a"].epoch == 0                             # unchanged: no new epoch
    bp.sync({"a": {"key": "Rigid/block/00000", "pos": [0.01, 0.0, 0.8], "quat": Q}})
    assert bp.tracks["a"].epoch == 1 and np.isclose(bp.tracks["a"].pose[0], 0.01)
    bp.sync({"a": {"key": "Rigid/block/00000", "pos": [0.01, 0.0, 0.9], "quat": Q}}, held={"a"})
    assert bp.tracks["a"].held
    bp.sync({"a": {"key": "Rigid/block/00000", "pos": [0.2, 0.0, 0.8], "quat": Q}})
    assert not bp.tracks["a"].held and bp.tracks["a"].epoch == 3
    # a caller that writes our moves back: the moved pose is not an external change
    wb = make([{"ok": True, "xy": (0.012, 0.0), "iou": 0.95, "vis": 0.9}])
    wb.writes_back = True
    ch = wb.step(FRAME, 0, required=["a"])
    assert len(ch) == 1
    wb.sync({"a": {"key": "Rigid/block/00000", "pos": ch[0]["to"][:3], "quat": Q}})
    assert wb.tracks["a"].epoch == 0 and wb.tracks["a"].moved == 0


def test_bridge_extra_observations_and_observer():
    import threading

    from dojo.runtime.bridge import Bridge
    b = Bridge()
    seen = []
    b.on_obs, b.obs_every = (lambda o: seen.append(o["steps_used"])), 3
    act = {"left_arm_joint_state": np.zeros(6), "left_ee_joint_state": np.ones(1)}

    def program():
        for _ in range(7):
            b.step(act)
        b.observe()
        return "ok"
    out = b.start(program, {"steps_used": 0, "step_lim": 100, "images": {}, "state": {}})
    n, chunks = 0, []
    while not out.get("done"):
        chunks.append(len(out["actions"]))
        n += len(out["actions"])
        out = b.resume({"steps_used": n, "step_lim": 100, "images": {}, "state": {}})
    assert chunks == [3, 3, 1] and seen == [3, 6, 7] and b.n_extra_obs == 2
    assert len(b.executed) == 7 and not threading.active_count() > 50


class _Asset:
    def __init__(self, key):
        self.key = key


class _Obj:
    def __init__(self, label, pos):
        self.label, self.pos, self.quat = label, np.asarray(pos, float), np.array(Q)
        self.asset, self.clutter, self.confidence = _Asset("Rigid/block/00000"), False, 0.5


class _Belief:
    def __init__(self, objs, held=()):
        self.objects = {o.label: o for o in objs}
        self._held = list(held)

    def held_labels(self):
        return self._held


def test_bg_hook_writes_moves_of_free_objects_only():
    from dojo.runtime.bg import BgHook

    class PC:
        def __init__(self):
            self.calls = []

        def bg_step(self, obs, step, belief, held, required, scene, fold, cfg, reset=False):
            self.calls.append({"held": held, "required": required, "reset": reset, "labels": [o["label"] for o in belief]})
            return {"moves": [{"label": "a", "to": [0.01, 0.0, 0.8] + Q, "n_views": 2},
                              {"label": "h", "to": [0.5, 0.5, 0.8] + Q, "n_views": 2}], "needs_look": [{"label": "a"}]}

    class Prov:
        mode = "perception"
        notes = {"params": {"label": "a", "arm": "left"}}
        belief = _Belief([_Obj("a", [0, 0, 0.8]), _Obj("h", [0.3, 0, 0.9])], held=["h"])

    pc, prov, ev = PC(), Prov(), []
    hook = BgHook(pc, prov, "stack_blocks", 0, {}, ev)
    st = {"left_ee_joint_state": [1.0], "right_ee_joint_state": [1.0], "left_ee_pose": [0, 0.3, 1.2, 1, 0, 0, 0],
          "right_ee_pose": [0, -0.3, 1.2, 1, 0, 0, 0]}
    hook({"steps_used": 10, "state": st, "images": {}})
    assert pc.calls[0]["held"] == ["h"] and pc.calls[0]["required"] == ["a"] and pc.calls[0]["reset"]
    assert np.allclose(prov.belief.objects["a"].pos, [0.01, 0.0, 0.8])
    assert np.allclose(prov.belief.objects["h"].pos, [0.3, 0.0, 0.9])        # held: never moved by the background
    assert [e["kind"] for e in ev] == ["bg_move", "bg_needs_look"] and hook.report()["moves"] == 1
    hook({"steps_used": 12, "state": st, "images": {}})
    assert not pc.calls[1]["reset"]


def test_bg_hook_treats_objects_at_a_closed_jaw_as_held():
    from dojo.runtime.bg import BgHook

    class PC:
        held = None

        def bg_step(self, obs, step, belief, held, required, scene, fold, cfg, reset=False):
            PC.held = held
            return {"moves": [{"label": "a", "to": [0.3, 0.0, 0.8] + Q}]}

    class Prov:
        mode, notes = "perception", {}
        belief = _Belief([_Obj("a", [0.15, 0.3, 1.0]), _Obj("b", [0.0, -0.3, 0.8])])

    st = {"left_ee_joint_state": [0.0], "right_ee_joint_state": [1.0], "left_ee_pose": [0.0, 0.3, 1.0, 1, 0, 0, 0],
          "right_ee_pose": [0, -0.3, 1.2, 1, 0, 0, 0]}
    BgHook(PC(), Prov(), "s", 0)({"steps_used": 5, "state": st, "images": {}})
    assert PC.held == ["a"]                                     # the left jaw is closed 0 mm from it (flange + 0.15 m)
    assert np.allclose(Prov.belief.objects["a"].pos, [0.15, 0.3, 1.0])


def test_service_bg_step_syncs_steps_and_rereads(monkeypatch):
    from dojo.perception import bgperc
    from dojo.runtime import perception_service as ps

    calls = []
    monkeypatch.setattr(bgperc.BackgroundPerceiver, "step", lambda self, f, s, required=(): calls.append(("step", s)) or [])
    monkeypatch.setattr(bgperc.BackgroundPerceiver, "reread_due", lambda self, s, f=None: s >= 10)
    monkeypatch.setattr(bgperc.BackgroundPerceiver, "reread", lambda self, s, r: calls.append(("reread", s, sorted(r))) or [])
    P = ps._Perception()
    monkeypatch.setattr(bgperc, "reread_estimate", lambda images, state, task, instr, bgs, cfg, views: {"Rigid": {"block": [
        {"label": "block0", "category_idx": 0, "default_pos": [0.1, 0, 0.8], "default_ori": Q,
         "_perception": {"confidence": 0.8}}]}})
    monkeypatch.setattr(P, "bgs", lambda fold: {})
    monkeypatch.setattr(P, "cfg", lambda fold: None)
    fr = ps.encode_frame({"images": {"cam_head": np.zeros((2, 2, 3), np.uint8)}, "state": {"left_ee_pose": [0] * 7}})
    belief = [{"label": "block0", "key": "Rigid/block/00000", "pos": [0.1, 0, 0.8], "quat": Q}]
    r = P.bg_step(fr, 4, belief, [], ["block0"], "stack_blocks", 0, {"every": 1}, reset=True)
    assert r["moves"] == [] and calls == [("step", 4)]
    P.bg_step(fr, 10, belief, [], [], "stack_blocks", 0)
    assert calls[-1] == ("reread", 10, ["block0"]) and P._bg.cfg.every == 1     # state kept between calls


def test_tcp_from_flange_matches_the_tracker():
    import pytest
    pytest.importorskip("scipy")
    from dojo.perception.tracker import tcp_from_flange as ref
    from dojo.runtime.bg import tcp_from_flange
    for f in ([0.1, 0.2, 1.0, 1, 0, 0, 0], [0.0, -0.3, 0.9, 0.7071, 0.0, 0.7071, 0.0], [0.2, 0.1, 1.1, 0.5, 0.5, -0.5, 0.5]):
        assert np.allclose(tcp_from_flange(f), ref(f), atol=1e-6)


def test_held_checks_drop_after_two_confident_empty_reads(monkeypatch):
    from dojo.perception.bgperc import BackgroundPerceiver
    monkeypatch.setattr(BackgroundPerceiver, "_jaws_closed", staticmethod(lambda r, key: r.get("closed", True)))
    bp = make([], BgConfig(every=1, held_every=5, held_after=8))
    bp.touch("a", held=True)
    assert bp.held_due(10, {"left": "a"}) == []                  # first sight of the hold: the close is step 10
    assert bp.held_due(17, {"left": "a"}) == []                  # still lifting
    assert bp.held_due(18, {"left": "a"}) == [("left", "a")]
    assert bp.held_result(18, "left", "a", {"held": False, "confidence": 1.0}) is None
    assert bp.held_due(20, {"left": "a"}) == []                  # held_every
    assert bp.held_result(23, "left", "a", {"held": True, "confidence": 0.9}) is None    # held again: count reset
    assert bp.held_result(28, "left", "a", {"held": False, "confidence": 0.95}) is None
    d = bp.held_result(33, "left", "a", {"held": False, "confidence": 0.85})
    assert d == {"label": "a", "arm": "left", "step": 33, "reads": 2}
    assert bp.held_result(38, "left", "a", {"held": False, "confidence": 0.5}) is None  # unsure reads never count
    assert bp.held_result(43, "left", "a", {"held": False, "confidence": 1.0, "closed": False}) is None
    assert bp.held_result(48, "left", "a", {"held": False, "confidence": 1.0, "closed": False}) is None  # jaws not
    assert bp.tracks["a"].not_held == 0                                                     # measured closed: kept


def test_contact_makes_an_untouched_object_a_suspect(monkeypatch):
    import sys
    import types

    class A:
        extents = np.array([0.04, 0.04, 0.04])
    fake = types.ModuleType("dojo.perception.render")       # the catalogue (and cv2) are not in the unit-test env
    fake.load_asset = lambda key: A()
    monkeypatch.setitem(sys.modules, "dojo.perception.render", fake)
    bp = make([{"ok": True, "xy": (0.03, 0.0), "iou": 0.8, "vis": 0.9}], BgConfig(every=1, contact=True))
    assert bp.contacts(5, [np.array([0.2, 0.0, 0.8])]) == []      # far
    assert bp.contacts(6, [np.array([0.01, 0.0, 0.95])]) == []    # above its top
    assert bp.contacts(7, [np.array([0.01, 0.0, 0.81])]) == ["a"]
    assert bp.tracks["a"].suspect == 7
    bp.step(FRAME, 8, required=[])
    assert bp._refine.calls[0]["radius"] == bp.cfg.search_radius  # searched for at its next view
    assert bp.tracks["a"].suspect is None                          # an ok fit clears it


def test_runtime_config_bg_perception_is_off_by_default_and_validated():
    import pytest

    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    cfg = RuntimeConfig()
    assert cfg.bg_perception == "off" and cfg.bg_obs_every == 0 and cfg.bg_cfg == {}
    RuntimePolicy(RuntimeConfig(bg_perception="reread", bg_obs_every=4))
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(bg_perception="always"))
