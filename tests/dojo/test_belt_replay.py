"""The belt tracker on recorded belt-read sequences (``tests/fixtures/belt_reads_51.json``).

Genuine-5.1 match_and_pick_from_conveyor on compute4 (main ad5b50e, 2026-10-28) ended every episode with "nothing
moves". That box's bg_folds_51 had no ``scene/`` background, so the belt read segmented the dark belt against the TABLE
background and saw no rider. The tracker counted the first two blind reads (steps 0 and 4) against the five frame-0
labels and dropped all of them (``track_drop``). The same reads with the belt's scene background (offline, genuine-5.1
belt_bg record frames of our dev layouts, plain and with DINOv2 identity) track the first rider from step 0, give the
belt's velocity by the 4th read and keep it between riders; match_and_pick's decision on them takes the next rider of
the first one's identity, never a ghost of the first. CPU only: the tracker on the recorded reads, no perception."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dojo.runtime.belief import RuntimeProvider
from dojo.runtime.bridge import Bridge
from dojo.runtime.estimate import objects_from_record
from dojo.skills.env import KinematicEnv
from dojo.skills.world import AssetInfo

FIX = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "belt_reads_51.json").read_text())
TASK = "match_and_pick_from_conveyor"
IDX = {"puppet": 0, "game_machine": 0, "phone": 0, "camera": 2, "ice_cream": 0, "garage": 16, "bottle": 47, "bag": 0}
CONVEYOR = {"category": "conveyor", "category_idx": 0, "label": "conveyor", "default_pos": [-2.08, 0.175, 0.0],
            "default_ori": [0.0, 0.0, 0.0, 1.0]}                       # 180 deg about z: the belt runs along world x


class _Assets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        if typ == "Dynamic":
            return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.array([-3.96, -0.6, 0.0]),
                             bbox_max=np.array([0.0, 0.58, 0.92]))
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.04), bbox_max=np.full(3, 0.04))


def _provider(start: dict):
    env = KinematicEnv([], step_lim=700, instruction="")
    b = Bridge()
    obs = {"images": {}, "state": env.proprio(), "instruction": "", "steps_used": 0, "step_lim": 700}
    b.obs, b.fresh = obs, True
    prov = RuntimeProvider(b, TASK, "", "perception", _Assets())
    prov.init_objects(objects_from_record({**start, "Dynamic": {"conveyor": [CONVEYOR]}}, _Assets(), 0))
    return prov, b, obs


def _replay(prov, b, obs, reads: dict, at=None) -> dict:
    """Every recorded read through the tracker at its step; ``at(step)`` after each. Returns {step: belt velocity}."""
    prov.track_reader = lambda o: reads[o["steps_used"]]
    v = {}
    for s in sorted(reads):
        b.obs, b.fresh = dict(obs, steps_used=s), True
        prov._track(b.obs)
        v[s] = prov.belt_velocity().copy()
        if at is not None:
            at(s)
    return v


def _compute4_reads(name: str, blind: bool) -> dict:
    """The logged reads ([label, confidence, category, [x, y]]) as belt-read records; ``blind``: as the fixed service
    marks a read whose belt has no scene background."""
    out = {}
    for r in FIX[name]["reads"]:
        rec: dict = {"Rigid": {}}
        for lab, conf, cat, (x, y) in r["found"]:
            rec["Rigid"].setdefault(cat, []).append({"category": cat, "category_idx": IDX[cat], "label": lab,
                                                     "default_pos": [x, y, 0.8], "default_ori": [1.0, 0.0, 0.0, 0.0],
                                                     "_perception": {"confidence": conf}})
        if blind:
            rec["_perception"] = {"scene_bg": None, "blind": "no scene background for Dynamic/conveyor/00000"}
        out[r["step"]] = rec
    return out


def _scene_reads(name: str) -> dict:
    out = {}
    for r in FIX[name]["reads"]:
        rec: dict = {"Rigid": {}, "_perception": {"scene_bg": "Dynamic/conveyor/00000"}}
        for k, (cat, idx, conf, x, y, z, *q) in enumerate(r["found"]):
            rec["Rigid"].setdefault(cat, []).append({"category": cat, "category_idx": idx, "label": f"rider{k}",
                                                     "default_pos": [x, y, z], "default_ori": q,
                                                     "_perception": {"confidence": conf}})
        out[r["step"]] = rec
    return out


@pytest.mark.parametrize("name", ["compute4_emb_100000", "compute4_off_100000"])
def test_blind_reads_dropped_every_frame0_label_at_the_second_read(name):
    """What main did with compute4's reads (the off arm read nothing, the DINOv2 arm static fixture blobs): the
    five frame-0 labels leave the belief at step 4, and no read ever gives the belt a velocity."""
    d = FIX[name]
    assert d["events"]["track_drop"] == 5 and d["result"]["note"] == "nothing moves"     # the recorded episode
    prov, b, obs = _provider(d["start"])
    v = _replay(prov, b, obs, _compute4_reads(name, blind=False))
    drops = [e for e in prov.events if e["kind"] == "track_drop"]
    assert sorted(e["label"] for e in drops) == ["other0", "other1", "other2", "target0", "target1"]
    assert {e["step"] for e in drops} == {4}
    assert all(float(np.linalg.norm(x)) == 0.0 for x in v.values())


@pytest.mark.parametrize("name", ["compute4_emb_100000", "compute4_off_100000"])
def test_a_blind_read_neither_drops_nor_adds_and_the_skill_says_why(name):
    """A read the service marks blind (its belt has no scene background) is no read: the frame-0 labels stay, no
    static blob joins, one ``belt_blind`` event, and the belt skills' "nothing moves" names the cause."""
    from dojo.skills.interactive import _nothing_moves
    d = FIX[name]
    prov, b, obs = _provider(d["start"])
    start = {l: o.pos.copy() for l, o in prov.belief.objects.items()}
    v = _replay(prov, b, obs, _compute4_reads(name, blind=True))
    assert not [e for e in prov.events if e["kind"] in ("track_drop", "track_new", "track_lost")]
    assert sorted(prov.belief.objects) == sorted(start)
    assert all(np.allclose(prov.belief.objects[l].pos, p) for l, p in start.items())
    blind = [e for e in prov.events if e["kind"] == "belt_blind"]
    assert len(blind) == 1 and blind[0]["step"] == 0 and "Dynamic/conveyor/00000" in blind[0]["why"]
    assert prov.stats["track_blind"] == len(d["reads"]) and not prov._tracks
    assert all(float(np.linalg.norm(x)) == 0.0 for x in v.values())
    r = _nothing_moves(SimpleNamespace(provider=prov))
    assert r["stage"] == "belt" and r["note"].startswith("nothing moves (belt reads blind: no scene background")
    assert _nothing_moves(SimpleNamespace(provider=object()))["note"] == "nothing moves"   # G3: as before


@pytest.mark.parametrize("name", ["scene_100000", "scene_100001", "scene_100002"])
def test_with_the_belt_background_the_first_rider_is_tracked_from_frame0_and_gives_the_velocity(name):
    """The same scene read with the belt's scene background: frame-0 B1 finds the rider in view (the four upstream
    labels not found: no phantom to drop), every read in view matches it, its identity stays the true one, and the
    belt velocity (4.0 mm / step along +x from the dev GT) comes from its track by the 4th read (step 30 at a read
    every 10 steps), while the rider is still in view. (Before its first fit, 3 reads over >= 8 steps, the belief keeps
    the frame-0 pose: 8 steps at the runtime's read every 4. Riding out of view past x ~0.72 the border cuts it and its
    reads jump 4-11 cm across the belt: not checked here.)"""
    d = FIX[name]
    prov, b, obs = _provider(d["start"])
    gt = {g["step"]: {r[0]: r for r in g["riders"]} for g in d["gt"]}
    assert [l for l, o in prov.belief.objects.items() if o.obj_type == "Rigid"] == ["target0"]
    t0 = [(s, g["target0"]) for s, g in gt.items() if "target0" in g]
    (s_a, a), (s_b, bb) = t0[0], t0[-1]
    v_gt = np.array([bb[2] - a[2], bb[3] - a[3]]) / (s_b - s_a)
    assert abs(float(v_gt[0]) - 0.004) < 1e-4 and abs(float(v_gt[1])) < 1e-4

    err = {}

    def at(s):
        g = gt[s].get("target0")
        if g is not None and g[2] <= 0.70 and "target0" in prov.belief.objects:
            o = prov.belief.objects["target0"]
            assert o.asset.key == g[1], (s, o.asset.key, g[1])
            assert "target0" not in prov._missed, s                             # this read matched it
            err[s] = float(np.hypot(o.pos[0] - g[2], o.pos[1] - g[3]))
    v = _replay(prov, b, obs, _scene_reads(name), at)
    assert not [e for e in prov.events if e["kind"] == "track_drop"]
    assert not [e for e in prov.events if e["kind"] == "belt_blind"]
    in_view = [s for s, g in t0 if g[2] <= 0.70]
    assert set(in_view) <= set(err)                                             # in the belief at every read in view
    assert max(err[s] for s in in_view[2:]) < 0.02, err                         # on it once fitted
    for s in in_view:
        if s >= 30:
            assert float(np.linalg.norm(v[s] - v_gt)) < 2e-4, (s, v[s], v_gt)
    # the belt runs on between riders: after the first rider left the view (no trusted track until the next one has
    # its reads) the belt keeps its last measured velocity instead of falling back to zeros
    assert all(float(np.linalg.norm(v[s])) > 0 for s in v if s >= 30)
    for s in range(100, 151, 10):
        assert float(np.linalg.norm(v[s] - v_gt)) < 3e-4, (s, v[s], v_gt)


def test_belt_bases_and_why_a_read_is_blind(monkeypatch):
    pytest.importorskip("cv2")                                              # the perception extra (CI: dev only)
    pytest.importorskip("scipy")
    from dojo.perception import belt as B
    from dojo.perception import inverse_layout as IL
    on_belt = {"placement": {"relative_plane": "conveyor/conveyor/0"}}
    tasks = {
        TASK: {"labels": {"conveyor": {"type": "Dynamic", "candidates": ["Dynamic/conveyor/00000"], "placement": {}},
                          "target0": {"type": "Rigid", "candidates": ["Rigid/puppet/00000"], **on_belt}}},
        "by_image": {"labels": {"conveyor": {"type": "Dynamic", "candidates": ["Dynamic/conveyor/00001"], "placement": {}},
                                "basket": {"type": "Rigid", "candidates": ["Rigid/basket_grasp/00002"],
                                           "placement": {"relative_plane": "conveyor"}},
                                "photo": {"type": "Geometry", "candidates": ["Geometry/photo/00000"], **on_belt}}},
        "stack_blocks": {"labels": {"block_0": {"type": "Rigid", "candidates": ["Rigid/block/00005"],
                                                "placement": {"relative_plane": "Table"}}}},
    }
    monkeypatch.setattr(IL, "task_map", lambda: tasks)
    assert B.moving_bases(TASK) == ["Dynamic/conveyor/00000"]
    assert B.moving_bases("by_image") == ["Dynamic/conveyor/00001"]
    assert B.moving_bases("stack_blocks") == [] and B.moving_bases("nope") == []
    assert "Dynamic/conveyor/00000" in B.blind_reason(TASK, None)
    assert B.blind_reason(TASK, "Dynamic/conveyor/00000") is None
    assert "Dynamic/conveyor/00001" in B.blind_reason("by_image", "Dynamic/conveyor/00000")   # another belt's background
    assert B.blind_reason("stack_blocks", None) is None and B.blind_reason("nope", None) is None


@pytest.fixture
def folds(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from dojo.perception import inverse_layout as IL
    root = tmp_path / "bg_folds_51"
    for f in ("fold0", "fold1"):
        for c in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            (root / f).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / f / f"{c}.png"), np.full((4, 6, 3), 10, np.uint8))
    sc = root / "fold1" / "scene" / "Dynamic_conveyor_00000"          # fold 1 has the belt's background, fold 0 not
    sc.mkdir(parents=True)
    cv2.imwrite(str(sc / "cam_head.png"), np.full((4, 6, 3), 200, np.uint8))
    monkeypatch.setenv("DOJO_BG_FOLDS", str(root))
    monkeypatch.setattr(IL, "task_map", lambda: {TASK: {"labels": {
        "conveyor": {"type": "Dynamic", "candidates": ["Dynamic/conveyor/00000"], "placement": {}},
        "target0": {"type": "Rigid", "candidates": ["Rigid/puppet/00000"],
                    "placement": {"relative_plane": "conveyor/conveyor/0"}}}}})
    return root


def test_the_service_says_which_background_a_belt_read_had_and_warns_once(folds, capsys):
    from dojo.perception.segment import scene_bg_key
    from dojo.runtime.perception_service import _Perception
    assert scene_bg_key(folds / "fold1", ["Dynamic/conveyor/00000"]) == "Dynamic/conveyor/00000"
    assert scene_bg_key(folds / "fold0", ["Dynamic/conveyor/00000"]) is None
    p = _Perception()
    p.set_scene(TASK)
    assert p.bg_scene(1) == "Dynamic/conveyor/00000" and int(p.bgs(1)["cam_head"][0, 0, 0]) == 200
    assert "WARNING" not in capsys.readouterr().out
    assert p.bg_scene(0) is None and int(p.bgs(0)["cam_head"][0, 0, 0]) == 10
    p.bg_scene(0)
    out = capsys.readouterr().out
    assert out.count("WARNING: no scene background for ['Dynamic/conveyor/00000']") == 1    # once per fold and scene


def test_the_belt_read_op_marks_a_blind_read(folds, monkeypatch):
    """Through the service process: a belt read on a fold without the belt's background comes back ``blind``."""
    import secrets
    import threading
    from multiprocessing.connection import Listener

    import dojo.perception.belt as B
    import dojo.runtime.perception_service as ps
    monkeypatch.setattr(B, "belt_read", lambda *a, **k: {"Rigid": {}, "_perception": {"n_head_blobs": 19}})
    key = secrets.token_bytes(16)
    lst = Listener(("127.0.0.1", 0), authkey=key)
    port = lst.address[1]
    lst.close()
    th = threading.Thread(target=ps.serve, args=(port, key), daemon=True)
    th.start()
    pc = ps.PerceptionClient(port, key, timeout_s=10)
    obs = {"images": {"cam_head": np.zeros((4, 6, 3), np.uint8)}, "state": {"x": np.zeros(2)}, "instruction": "i"}
    r0 = pc.belt_read(obs, TASK, 0)["_perception"]
    r1 = pc.belt_read(obs, TASK, 1)["_perception"]
    pc.close(stop_service=True)
    th.join(5)
    assert r0["scene_bg"] is None and "Dynamic/conveyor/00000" in r0["blind"] and r0["n_head_blobs"] == 19
    assert r1["scene_bg"] == "Dynamic/conveyor/00000" and "blind" not in r1


class _Ex:
    def __init__(self, sk):
        self.sk = sk

    def hold(self, n):
        from dojo.skills.executor import BudgetExceeded
        self.sk.step += int(n)
        if self.sk.step > self.sk.last_read:
            raise BudgetExceeded(f"replay ends at step {self.sk.last_read}")


class _ReplaySkills:
    """Skills as match_and_pick sees them, over the tracker replaying recorded reads: ``ex.hold`` moves the clock, a
    refresh runs every recorded read since the last one and returns the belief at that step."""

    def __init__(self, prov, b, obs, reads: dict):
        self.provider, self.b, self.obs, self.reads = prov, b, obs, reads
        prov.track_reader = lambda o: reads[o["steps_used"]]
        self.step, self.read_upto, self.last_read = 0, -1, max(reads)
        self.ex = _Ex(self)
        self.held: dict = {}

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
        pass


@pytest.mark.parametrize("name, want", [("scene_100001", "target1"), ("scene_100002", "target1"),
                                        ("scene_emb_100001", "target1"), ("scene_emb_100002", "target1"),
                                        ("scene_100000", None), ("scene_emb_100000", None)])
def test_match_and_pick_on_the_recorded_reads_takes_the_next_one_not_a_ghost_of_the_first(name, want, monkeypatch):
    """match_and_pick's decision (first object, velocity, the match) on the tracker replaying the recorded reads (robot
    at home, nothing picked). 100002 without DINOv2: the first phone's reads at the view's border jump 11 cm across the
    belt and start a second phone track there; once the first's own track is lost that ghost, riding out of view, was
    "the next phone" (step 120). The next one must come from behind where the belt has carried the first. 100000: the
    puppet behind reads as a camera in both arms (BACKLOG: identity on the belt), so nothing is picked rather than a
    wrong object."""
    import dojo.skills.interactive as I
    d = FIX[name]
    prov, b, obs = _provider(d["start"])
    sk = _ReplaySkills(prov, b, obs, _scene_reads(name))
    picked = []

    def pick(sk_, label, v, **k):
        w = sk_.refresh()
        picked.append((sk_.step, label, w.obj(label).pos[:2].copy(), np.asarray(v, float).copy()))
        return SimpleNamespace(ok=True, status="ok", note="", arm="left")
    monkeypatch.setattr(I, "pick_from_belt", pick)
    r = I.match_and_pick(sk, ["target0", "target1", "other0", "other1", "other2"], conveyor="conveyor")
    gt = {g["step"]: {x[0]: x for x in g["riders"]} for g in d["gt"]}
    if want is None:
        assert not picked and r["ok"] is False and r["stage"] == "match", r
        return
    assert r["ok"] and len(picked) == 1, r
    s, label, xy, v = picked[0]
    assert abs(float(v[0]) - 0.004) < 3e-4 and abs(float(v[1])) < 1e-6
    g_s = max(k for k in gt if k <= s)                                  # the dev GT at the last read before the pick
    near = min(gt[g_s].values(), key=lambda x: np.hypot(x[2] - xy[0], x[3] - xy[1]))
    assert near[0] == want, (s, label, xy, near)
