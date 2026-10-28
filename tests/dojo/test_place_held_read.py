"""Stream H: the held object's in-hand read at the hover of places that did not ask for it (``RuntimeConfig.
place_held_read``; ``Skills._wants_held_read`` -> ``RuntimeProvider.held_read_wanted``). CPU only: the kinematic env and a
fake perception client."""

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills.env import KinematicEnv
from dojo.skills.grasps import swing_lever
from dojo.skills.world import AssetInfo

STACK = "Stack the three blocks with different textures."
POS = [(0.412, -0.09, 0.78), (-0.26, -0.189, 0.78), (-0.344, -0.1, 0.78)]
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _env():
    rows = [{"label": f"block_{i}", "category": "block", "category_idx": 5, "pos": p, "quat": [1, 0, 0, 0],
             "half": [0.015] * 3} for i, p in enumerate(POS)]
    return KinematicEnv(rows, step_lim=550, instruction=STACK)


class _Assets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.015), bbox_max=np.full(3, 0.015))


class _Episode:
    def __init__(self, env):
        self.env = env

    def info(self):
        return {"stable": True, "task": "stack_blocks", "layout_id": 1, "instruction": self.env.instruction,
                "step_lim": self.env.step_lim, "reset_s": 0.0, "render_stage_s": 0.0}

    def observe(self):
        img = np.zeros((48, 64, 3), np.uint8)
        return {"images": {c: img for c in CAMS}, "state": self.env.proprio(), "instruction": self.env.instruction,
                "steps_used": self.env.steps_used, "step_lim": self.env.step_lim, "renderer": "fake"}


def _run(policy, ep, max_calls=2000):
    policy.reset(ep.info())
    for _ in range(max_calls):
        d = policy.next(ep.observe())
        if d.get("done"):
            return policy.episode_report()
        for a in d["actions"]:
            if ep.env.step(a):
                return policy.episode_report()
    raise AssertionError("episode did not finish")


def test_swing_lever_is_the_centre_of_mass_off_the_pads_axis_across_gravity():
    T_tcp = np.eye(4)
    T_tcp[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)   # approach (TCP x) straight down
    T_obj = np.eye(4)
    assert swing_lever(T_obj, [0.04, 0.0, 0.0], T_tcp) == pytest.approx(0.0)        # hanging below the pads: no lever
    assert swing_lever(T_obj, [0.0, 0.03, 0.0], T_tcp) == pytest.approx(0.0)        # along the closing axis: no turn
    assert swing_lever(T_obj, [0.0, 0.0, 0.045], T_tcp) == pytest.approx(0.045)     # a mug held by its wall
    T_obj[:3, 3] = [0.0, 0.0, -0.02]
    assert swing_lever(T_obj, [0.0, 0.0, 0.0], T_tcp) == pytest.approx(0.02)       # the object's origin is the offset


class _Perception:
    """B1 frame 0 from the env's truth; held_pose answers ``answer(prior)``; refine says ``refine_ok``."""

    def __init__(self, env, answer):
        self.env, self.answer, self.calls = env, answer, []

    def estimate(self, frames, vocabulary, fold, prior_record=None):
        return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": o["label"],
                                     "default_pos": list(o["pos"]), "default_ori": [1, 0, 0, 0],
                                     "_perception": {"confidence": 0.9}} for o in self.env.objects]}}

    def check(self, frame, objects, fold):
        return {o["label"]: {"head": {"iou": 0.9, "vis": 1.0, "n_px": 100}} for o in objects}

    def look_plan(self, frame, record, max_looks=2):
        return []

    def held_pose(self, frame, arm, asset_key, prior_T_tcp_obj, cfg=None):
        self.calls.append(dict(cfg or {}))
        return self.answer(np.asarray(prior_T_tcp_obj, float))


def _read(ok=True, why="", sigma_mm=1.0, iou=0.9, dx=0.002):
    def answer(prior):
        T = prior.copy()
        T[:3, 3] += [dx, 0.0, 0.0]
        r = {"ok": ok, "why": why, "shift_mm": dx * 1000, "turn_deg": 0.0, "iou": {"cam_left_wrist": iou},
             "sigma": {"pos_mm": [sigma_mm] * 3, "rot_deg": [0.5] * 3, "lead_lat_mm": sigma_mm, "axis_deg": 0.5}}
        r["T_tcp_obj" if ok else "fit_T_tcp_obj"] = T.tolist()
        if not ok:
            r["T_tcp_obj"] = prior.tolist()
        return r
    return answer


@pytest.mark.parametrize("mode, reads", [("off", 0), ("swing", 0), ("all", 3)])
def test_places_read_at_the_hover_only_when_the_rule_wants_it(mode, reads):
    env = _env()
    pc = _Perception(env, _read())
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off", held_pose="request", place_held_read=mode),
                        assets=_Assets(), perception_client=pc)
    rep = _run(pol, _Episode(env))
    assert rep["error"] is None, rep["error"]
    places = [h for h in rep["history"] if h["skill"] == "place"]
    assert len(places) == 3 and len(pc.calls) == reads
    asked = [e for e in rep["events"] if e["kind"] == "place_held_read"]
    if mode == "off":
        assert asked == []                                  # held_pose request alone: only places that ask read
        return
    assert len(asked) == 3 and all(e["lever_mm"] is not None and e["lever_mm"] < 20 for e in asked)   # blocks: no lever
    assert sum(e["wanted"] for e in asked) == reads
    if reads:
        assert all(c.get("trigger") == "place" for c in pc.calls)
        hp = [e for e in rep["events"] if e["kind"] == "held_pose"]
        assert len(hp) == 3 and all(e["ok"] and e["trigger"] == "place" for e in hp)
        assert all("in-hand shift 2.0 mm, re-planned" in h["note"] for h in places)   # the tail used the read
        assert rep["provider"]["held_pose"] == 3 and rep["provider"]["place_held_reads"] == 3


@pytest.mark.parametrize("coarse_mm, adopted", [(0.0, False), (3.0, True)])
def test_an_ambiguous_read_is_taken_only_within_the_coarse_bounds(coarse_mm, adopted):
    env = _env()
    pc = _Perception(env, _read(ok=False, why="ambiguous", sigma_mm=1.5, iou=0.9))
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off", held_pose="request", place_held_read="all",
                                      place_read_coarse_mm=coarse_mm), assets=_Assets(), perception_client=pc)
    rep = _run(pol, _Episode(env))
    assert rep["error"] is None, rep["error"]
    hp = [e for e in rep["events"] if e["kind"] == "held_pose"]
    assert len(hp) == 3 and all(e["ok"] == adopted for e in hp)
    assert all(e["why"] == ("coarse" if adopted else "ambiguous") for e in hp)
    assert rep["provider"].get("held_pose", 0) == (3 if adopted else 0)


def test_the_doubt_rule_reads_a_grasp_whose_pick_refine_fit_poorly():
    from dojo.runtime.belief import RuntimeProvider
    from dojo.runtime.bridge import Bridge
    from dojo.runtime.estimate import objects_from_gt_rows
    env = _env()
    b = Bridge()
    b.obs, b.fresh = _Episode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", _Assets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    o = prov.belief.objects["block_1"]
    assert not prov.held_read_wanted("left", "block_1", np.eye(4), o)          # off: never
    prov.place_held_read = "doubt"
    assert not prov.held_read_wanted("left", "block_1", np.eye(4), o)
    prov.held_doubt.add("block_1")
    assert prov.held_read_wanted("left", "block_1", np.eye(4), o)
    ev = prov.events[-1]
    assert ev["kind"] == "place_held_read" and ev["doubt"] and ev["wanted"]


def test_config_is_validated_and_off_by_default():
    assert RuntimeConfig().place_held_read == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", place_held_read="sometimes"), assets=_Assets())
