"""MATCH2 switches for match_and_pick_from_conveyor (all default off): ``belt_ref`` (the first object from the riders the
belt tracker holds, matched against its own read history), ``belt_veto`` (an empty wrist look vetoes only with
evidence), ``belt_speed`` (the belt speed locked from the first long clean track), ``belt_high_lift`` and
``belt_retry``. CPU only: the tracker on synthetic reads, the skill's decision on a replay stub."""

from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.belief import RuntimeProvider
from dojo.runtime.bridge import Bridge
from dojo.runtime.estimate import objects_from_record
from dojo.skills.env import KinematicEnv
from dojo.skills.world import AssetInfo

TASK = "match_and_pick_from_conveyor"
CONVEYOR = {"category": "conveyor", "category_idx": 0, "label": "conveyor", "default_pos": [-2.08, 0.175, 0.0],
            "default_ori": [0.0, 0.0, 0.0, 1.0]}                       # 180 deg about z: the belt runs along world x
SWITCHES = {"belt_ref": ("off", "tracked", "similar"), "belt_veto": ("off", "evidence"), "belt_speed": ("off", "lock"),
            "belt_high_lift": ("off", "on"), "belt_retry": ("off", "window"), "belt_grasp": ("off", "solid"),
            "belt_wrist_id": ("off", "on")}


class _Assets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        if typ == "Dynamic":
            return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.array([-3.96, -0.6, 0.0]),
                             bbox_max=np.array([0.0, 0.58, 0.92]))
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.04), bbox_max=np.full(3, 0.04))


def _rider(cat, x, y, conf=0.9, label="rider0", idx=0):
    return {"category": cat, "category_idx": idx, "label": label, "default_pos": [x, y, 0.8],
            "default_ori": [1.0, 0.0, 0.0, 0.0], "_perception": {"confidence": conf}}


def _provider(start: dict):
    env = KinematicEnv([], step_lim=700, instruction="")
    b = Bridge()
    obs = {"images": {}, "state": env.proprio(), "instruction": "", "steps_used": 0, "step_lim": 700}
    b.obs, b.fresh = obs, True
    prov = RuntimeProvider(b, TASK, "", "perception", _Assets())
    rec = {"Rigid": {}, "Dynamic": {"conveyor": [CONVEYOR]}}
    for r in start:
        rec["Rigid"].setdefault(r["category"], []).append(r)
    prov.init_objects(objects_from_record(rec, _Assets(), 0))
    return prov, b, obs


def _feed(prov, b, obs, reads: dict):
    prov.track_reader = lambda o: reads[o["steps_used"]]
    for s in sorted(reads):
        b.obs, b.fresh = dict(obs, steps_used=s), True
        prov._track(b.obs)


def _rec(*riders):
    out: dict = {"Rigid": {}}
    for r in riders:
        out["Rigid"].setdefault(r["category"], []).append(r)
    return out


# ---------------------------------------------------------------- defaults


def test_switches_default_off_and_validated():
    from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
    from dojo.skills.g3 import SKILL_ATTRS
    cfg = RuntimeConfig()
    for k, vals in SWITCHES.items():
        assert getattr(cfg, k) == "off"
        with pytest.raises(ValueError):
            RuntimePolicy(RuntimeConfig(**{k: "typo"}))
        assert SKILL_ATTRS[k] == vals
    assert RuntimeProvider.__init__.__code__ is not None
    prov, _, _ = _provider([])
    assert prov.belt_speed == "off"


def test_presets_leave_the_switches_off():
    from dojo.runtime import presets
    for name in getattr(presets, "PRESETS", {}):
        p = presets.PRESETS[name]
        for k in SWITCHES:
            assert (p.get(k, "off") if isinstance(p, dict) else getattr(p, k, "off")) == "off", (name, k)


# ---------------------------------------------------------------- identity evidence (RuntimeProvider)


def test_identity_dist_accumulates_every_matched_read():
    prov, b, obs = _provider([_rider("ice_cream", 0.45, 0.04, label="target0")])
    reads = {s: _rec(_rider("ice_cream" if s % 12 else "phone", 0.45 + 0.004 * s, 0.04, 0.95 if s % 12 else 0.5))
             for s in range(4, 40, 4)}
    _feed(prov, b, obs, reads)
    d, n = prov.identity_dist("target0")
    assert n == 9 and max(d, key=d.get) == "Rigid/ice_cream/00000" and 0.75 < d["Rigid/ice_cream/00000"] < 0.85
    assert prov.identity_dist("nobody") == ({}, 0)
    assert prov.track_age("target0", 40) == 4 and prov.track_age("nobody", 40) is None


def test_off_the_existing_tracker_is_unchanged():
    """The new bookkeeping only records: the off path's beliefs and velocities equal a provider without it."""
    def run(extra: bool):
        prov, b, obs = _provider([_rider("bottle", 0.45, 0.04, label="target0", idx=47)])
        reads = {s: _rec(_rider("bottle", 0.45 + 0.0039 * s + (0.002 if s % 8 else -0.002), 0.04, 0.9, idx=47))
                 for s in range(4, 80, 4)}
        if not extra:
            prov._hist = lambda *a, **k: None
        _feed(prov, b, obs, reads)
        return prov.belt_velocity(), {k: o.pos.copy() for k, o in prov.belief.objects.items()}
    v1, p1 = run(True)
    v0, p0 = run(False)
    assert np.allclose(v1, v0) and p1.keys() == p0.keys() and all(np.allclose(p1[k], p0[k]) for k in p1)


# ---------------------------------------------------------------- belt speed lock


def _speed_run(mode: str):
    """A first rider read cleanly at 4.0 mm / step for 60 steps (beyond the arms), then only a second rider whose reads
    drift at 3.3 mm / step (the view's entry, the arms half hiding it)."""
    prov, b, obs = _provider([_rider("phone", 0.40, 0.04, label="target0")])
    prov.belt_speed = mode
    reads = {}
    for s in range(0, 64, 4):
        reads[s] = _rec(_rider("phone", 0.40 + 0.004 * s, 0.04, 0.95))
    for s in range(64, 200, 4):
        reads[s] = _rec(_rider("camera", -0.60 + 0.0033 * (s - 64), -0.05, 0.9, label="rider1", idx=2))
    _feed(prov, b, obs, reads)
    return prov


def test_belt_speed_lock_keeps_the_first_clean_measurement():
    off = _speed_run("off")
    on = _speed_run("lock")
    assert abs(float(off.belt_velocity()[0]) - 0.0033) < 3e-4          # today: the newer, biased track
    assert abs(float(on.belt_velocity()[0]) - 0.004) < 1e-4 and abs(float(on.belt_velocity()[1])) < 1e-9
    lock = [e for e in on.events if e["kind"] == "belt_lock"]
    assert len(lock) == 1 and lock[0]["label"] == "target0" and lock[0]["span"] >= on.belt_lock_span
    assert not [e for e in off.events if e["kind"] == "belt_lock"]


def test_belt_speed_lock_refuses_a_noisy_track():
    prov, b, obs = _provider([_rider("phone", 0.40, 0.04, label="target0")])
    prov.belt_speed = "lock"
    rng = np.random.default_rng(0)
    reads = {s: _rec(_rider("phone", 0.40 + 0.004 * s + rng.normal(0, 0.012), 0.04 + rng.normal(0, 0.012), 0.9))
             for s in range(0, 64, 4)}
    _feed(prov, b, obs, reads)
    assert prov._belt_lock is None


# ---------------------------------------------------------------- reference-calibrated identity (skills)


def test_ref_match_is_calibrated_by_the_first_objects_own_reads():
    from dojo.skills.interactive import ref_match
    ice = {"Rigid/ice_cream/00000": 0.9, "Rigid/phone/00000": 0.1}
    assert ref_match(ice, {"Rigid/ice_cream/00000": 0.8, "Rigid/bottle/00047": 0.2})[0]
    assert not ref_match(ice, {"Rigid/camera/00002": 1.0})[0]
    # a camera the head view reads half as a phone: a rider read the same way matches, a phone read as one does not
    cam = {"Rigid/camera/00002": 0.5, "Rigid/phone/00000": 0.35, "Rigid/puppet/00000": 0.15}
    assert ref_match(cam, {"Rigid/camera/00002": 0.6, "Rigid/phone/00000": 0.4})[0]
    ok, sim, competing = ref_match(cam, {"Rigid/phone/00000": 1.0})
    assert not ok and competing
    assert ref_match({}, {"x": 1.0}) == (False, 0.0, False)


# ---------------------------------------------------------------- the wrist veto's evidence


def test_absence_needs_a_view_of_the_prior_or_a_stale_head_track():
    from dojo.skills.interactive import _absence_seen, wrist_look_absent
    empty_seen = {"ok": False, "iou": 0.0, "visible_frac": 1.0}
    empty_hidden = {"ok": False, "iou": 0.0, "visible_frac": 0.0}
    assert wrist_look_absent(empty_seen) and wrist_look_absent(empty_hidden)
    fresh = SimpleNamespace(provider=SimpleNamespace(track_age=lambda label: 6))
    stale = SimpleNamespace(provider=SimpleNamespace(track_age=lambda label: 44))
    never = SimpleNamespace(provider=SimpleNamespace(track_age=lambda label: None))
    gt = SimpleNamespace(provider=object())
    assert _absence_seen(fresh, "r", empty_seen)
    assert not _absence_seen(fresh, "r", empty_hidden)
    assert _absence_seen(stale, "r", empty_hidden)
    assert not _absence_seen(never, "r", empty_hidden)
    assert not _absence_seen(gt, "r", {})


# ---------------------------------------------------------------- match_and_pick's decision on a replay


class _Ex:
    def __init__(self, sk):
        self.sk = sk
        self.steps_left = None

    def hold(self, n):
        from dojo.skills.executor import BudgetExceeded
        self.sk.step += int(n)
        if self.sk.step > self.sk.last_read:
            raise BudgetExceeded(f"replay ends at step {self.sk.last_read}")


class _ReplaySkills:
    def __init__(self, prov, b, obs, reads: dict, **attrs):
        self.provider, self.b, self.obs, self.reads = prov, b, obs, reads
        prov.track_reader = lambda o: reads[o["steps_used"]]
        self.step, self.read_upto, self.last_read = 0, -1, max(reads)
        self.ex = _Ex(self)
        self.held: dict = {}
        self.logs: list = []
        for k, v in attrs.items():
            setattr(self, k, v)

    def refresh(self):
        import copy

        from dojo.skills.world import WorldState
        for s in sorted(r for r in self.reads if self.read_upto < r <= self.step):
            self.b.obs, self.b.fresh = dict(self.obs, steps_used=s), True
            self.provider._track(self.b.obs)
        self.read_upto = self.step
        self.provider._predict(self.step)
        return WorldState(TASK, "", copy.deepcopy(self.provider.belief.objects), {}, {}, step=self.step, step_lim=700,
                          notes=self.provider.notes)

    def log(self, *a, **k):
        self.logs.append(a[0] if a else "")

    def home(self, **k):
        pass


def _phantom_scene():
    """The compute4 ice-cream class: frame 0 holds the ice cream (target0) and a phantom "camera" (target1) just ahead of
    it, which no belt read confirms (dropped at step 4). Upstream: a real camera rider, then the second ice cream."""
    start = [_rider("ice_cream", 0.446, 0.038, 0.8, "target0"), _rider("camera", 0.505, 0.066, 0.4, "target1", idx=2)]
    reads = {}
    for s in range(0, 400, 4):
        rs = []
        x0 = 0.446 + 0.004 * s
        if x0 < 0.85:
            rs.append(_rider("ice_cream", x0, 0.038, 0.95))
        xc = -0.90 + 0.004 * (s - 100)
        if s >= 100 and -0.8 < xc < 0.8:
            rs.append(_rider("camera", xc, -0.02, 0.9, label="rider1", idx=2))
        xi = -0.90 + 0.004 * (s - 160)
        if s >= 160 and -0.8 < xi < 0.8:
            rs.append(_rider("ice_cream", xi, -0.10, 0.9 if s % 8 else 0.6, label="rider2"))
        reads[s] = _rec(*rs)
    return start, reads


@pytest.mark.parametrize("mode, want", [("off", "Rigid/camera/00002"), ("tracked", "Rigid/ice_cream/00000")])
def test_the_first_object_is_the_rider_the_tracker_holds(mode, want, monkeypatch):
    import dojo.skills.interactive as I
    start, reads = _phantom_scene()
    prov, b, obs = _provider(start)
    sk = _ReplaySkills(prov, b, obs, reads, belt_ref=mode)
    picked = []

    def pick(sk_, label, v, **k):
        w = sk_.refresh()
        picked.append((label, w.obj(label).asset.key, w.obj(label).pos[:2].copy()))
        return SimpleNamespace(ok=True, status="ok", note="", arm="left")
    monkeypatch.setattr(I, "pick_from_belt", pick)
    r = I.match_and_pick(sk, ["target0", "target1"], conveyor="conveyor")
    st = sk.refresh().notes["belt"]
    assert st["first_key"] == want
    assert r["ok"] and len(picked) == 1
    assert picked[0][1] == want                       # off: the camera rider (the wrong object); tracked: the ice cream
    if mode == "tracked":
        assert abs(float(picked[0][2][1]) + 0.10) < 0.02
        assert any("start riders the tracker dropped" in str(x) for x in sk.logs)


def test_high_lift_goes_on_up_only_with_the_switch(monkeypatch):
    import dojo.skills.interactive as I
    start, reads = _phantom_scene()
    lifts = []
    monkeypatch.setattr(I, "_lift_more", lambda sk_, arm, label, dz: lifts.append((arm, label, dz)))
    monkeypatch.setattr(I, "pick_from_belt", lambda sk_, label, v, **k: SimpleNamespace(ok=True, status="ok", note="",
                                                                                         arm="right"))
    for mode, n in (("off", 0), ("on", 1)):
        prov, b, obs = _provider(start)
        sk = _ReplaySkills(prov, b, obs, reads, belt_ref="tracked", belt_high_lift=mode)
        lifts.clear()
        assert I.match_and_pick(sk, ["target0", "target1"], conveyor="conveyor")["ok"]
        assert len(lifts) == n
        if n:
            assert lifts[0][0] == "right" and lifts[0][2] == I.BELT_HIGH_LIFT_M


def test_retry_window_takes_more_rounds_while_steps_remain(monkeypatch):
    import dojo.skills.interactive as I
    start, reads = _phantom_scene()
    calls = []

    def pick(sk_, label, v, **k):
        calls.append(label)
        return SimpleNamespace(ok=False, status="absent", note="nothing there", arm="left")
    monkeypatch.setattr(I, "pick_from_belt", pick)
    counts = {}
    for mode in ("off", "window"):
        prov, b, obs = _provider(start)
        sk = _ReplaySkills(prov, b, obs, reads, belt_ref="tracked", belt_wrist_veto="on", belt_retry=mode)
        calls.clear()
        r = I.match_and_pick(sk, ["target0", "target1"], conveyor="conveyor")
        counts[mode] = len(calls)
        assert not r["ok"]
    # one ice cream comes: it is vetoed once and stays absent, so both run out of matches; the window never adds a
    # second pick of an absent track
    assert counts["off"] == counts["window"] == 1


# ---------------------------------------------------------------- the competing-category veto at the hover


def test_competing_look_needs_a_confident_fit_of_another_category():
    from dojo.skills.interactive import competing_look
    ref = "Rigid/camera/00002"
    assert competing_look({"ok": True, "identity_ok": True, "obj_id": "Rigid/phone/00000"}, ref)
    assert not competing_look({"ok": True, "identity_ok": True, "obj_id": "Rigid/camera/00002"}, ref)
    assert not competing_look({"ok": True, "identity_ok": False, "obj_id": "Rigid/phone/00000"}, ref)
    assert not competing_look({"ok": False, "identity_ok": True, "obj_id": "Rigid/phone/00000"}, ref)
    assert not competing_look({}, ref)


def test_rider_candidates_unite_the_groups_that_admit_the_key(tmp_path, monkeypatch):
    import json

    import dojo.runtime.vocab as V
    from dojo.runtime.policy import rider_candidates
    cat = {TASK: {"labels": {"target0": {"type": "Rigid", "candidates": ["Rigid/camera/00002", "Rigid/phone/00000"]},
                             "other0": {"type": "Rigid", "candidates": ["Rigid/phone/00000", "Rigid/puppet/00000"]},
                             "conveyor": {"type": "Dynamic", "candidates": ["Dynamic/conveyor/00000"]}}}}
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(cat))
    monkeypatch.setattr(V, "catalog_path", lambda: str(p))
    assert rider_candidates(TASK, "Rigid/phone/00000") == ["Rigid/phone/00000", "Rigid/camera/00002",
                                                          "Rigid/puppet/00000"]
    assert rider_candidates(TASK, "Rigid/camera/00002") == ["Rigid/camera/00002", "Rigid/phone/00000"]
    assert rider_candidates(TASK, "Rigid/unknown/00000") == []
    assert rider_candidates("no_such_task", "Rigid/phone/00000") == []


def test_flat_rider_tells_a_dish_from_a_body():
    from dojo.skills.interactive import flat_rider
    from dojo.skills.world import ObjectState
    def obj(ext):
        a = AssetInfo(key="Rigid/x/00000", bbox_min=-np.asarray(ext) / 2, bbox_max=np.asarray(ext) / 2)
        return ObjectState("x", "Rigid", "x", 0, np.zeros(3), np.array([1.0, 0, 0, 0]), a)
    assert flat_rider(obj([0.20, 0.20, 0.025]))        # a plate
    assert not flat_rider(obj([0.12, 0.07, 0.08]))     # a camera
