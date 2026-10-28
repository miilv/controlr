"""Stream H: one in-hand read per grasp from the frame of the lift (``RuntimeConfig.carry_read``; perception's held_pose).
CPU only: the kinematic env and a fake perception client."""

import numpy as np
import pytest

from dojo.runtime.belief import RuntimeProvider
from dojo.runtime.bridge import Bridge
from dojo.runtime.estimate import objects_from_gt_rows
from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills.env import KinematicEnv
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


def _obs(env):
    img = np.zeros((48, 64, 3), np.uint8)
    return {"images": {c: img for c in CAMS}, "state": env.proprio(), "instruction": env.instruction,
            "steps_used": env.steps_used, "step_lim": env.step_lim, "renderer": "fake"}


def _fit(T_prior, dx=0.0, ok=True, why="", sigma_mm=0.5):
    T = np.asarray(T_prior, float).copy()
    T[0, 3] += dx
    return {"ok": ok, "why": why, "T_tcp_obj" if ok else "fit_T_tcp_obj": T.tolist(), "shift_mm": abs(dx) * 1000,
            "turn_deg": 0.0, "sigma": {"pos_mm": [sigma_mm] * 3, "rot_deg": [0.5] * 3}, "iou": {"cam_left_wrist": 0.9},
            "views": ["cam_left_wrist"]}


def _prov(lift=0.05, reply=None):
    env = _env()
    b = Bridge()
    b.obs, b.fresh = _obs(env), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", _Assets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    bel = prov.belief
    o = bel.objects["block_1"]
    z0 = float(bel.kin.tcp("left", bel.q["left"])[2, 3])
    T_tcp = bel.kin.tcp("left", bel.q["left"])
    T_rel = np.linalg.inv(T_tcp) @ o.T
    bel.attached["left"] = ("block_1", T_rel)
    bel.grasp_z["left"] = (z0 - lift, 1, np.r_[o.pos, o.quat].copy())
    bel.n_since_g["left"] = 10
    calls = []

    def reader(obs, arm, lab, key, T):
        calls.append((arm, lab, key))
        return reply(T) if reply is not None else _fit(T, 0.010)
    prov.carry_reader = reader
    return prov, b.obs, T_rel, calls


def test_an_ok_read_that_moved_reanchors_the_in_hand_transform_once_per_grasp():
    prov, obs, T_rel, calls = _prov()
    p0 = prov.belief.objects["block_1"].pos.copy()
    prov._carry_read(obs)
    prov._carry_read(obs)
    assert calls == [("left", "block_1", "Rigid/block/00005")]                 # once per grasp
    lab, T_new = prov.belief.attached["left"]
    assert lab == "block_1" and abs(float(T_new[0, 3] - T_rel[0, 3]) - 0.010) < 1e-9
    ev = [e for e in prov.events if e["kind"] == "carry_read"]
    assert len(ev) == 1 and ev[0]["adopted"] and ev[0]["shift_mm"] == 10.0
    moved = float(np.linalg.norm(prov.belief.objects["block_1"].pos - p0))
    assert abs(moved - 0.010) < 1e-6                                           # the object follows the new transform
    assert prov.stats["carry_adopted"] == 1 and "block_1" in prov.touched


@pytest.mark.parametrize("reply, adopted", [
    (lambda T: _fit(T, 0.002), False),                                         # below the noise floor
    (lambda T: _fit(T, 0.010, ok=False, why="fit_poor"), False),
    (lambda T: _fit(T, 0.010, ok=False, why="ambiguous", sigma_mm=2.0), False),  # off unless carry_max_sigma_mm > 0
    (lambda T: _fit(T, 0.024), False),                                         # centre past half the block (15 mm)
])
def test_reads_that_are_not_adopted(reply, adopted):
    prov, obs, T_rel, _ = _prov(reply=reply)
    prov._carry_read(obs)
    assert np.allclose(prov.belief.attached["left"][1], T_rel) and not adopted
    assert prov.events[-1]["kind"] == "carry_read" and prov.events[-1]["adopted"] is False


def test_an_ambiguous_read_within_the_coarse_sigma_bound_is_adopted_when_asked():
    prov, obs, T_rel, _ = _prov(reply=lambda T: _fit(T, 0.010, ok=False, why="ambiguous", sigma_mm=2.0))
    prov.carry_max_sigma_mm = 3.0
    prov._carry_read(obs)
    assert prov.events[-1]["adopted"] and not np.allclose(prov.belief.attached["left"][1], T_rel)


def test_no_read_before_the_lift_and_an_error_keeps_the_belief():
    prov, obs, _, calls = _prov(lift=0.01)
    prov._carry_read(obs)
    assert calls == [] and not prov.events

    def boom(T):
        raise RuntimeError("service down")
    prov, obs, T_rel, _ = _prov(reply=boom)
    prov._carry_read(obs)
    assert prov.events[-1]["error"].startswith("RuntimeError") and np.allclose(prov.belief.attached["left"][1], T_rel)


class _Perception:
    def __init__(self):
        self.reads = []

    def estimate(self, frames, vocabulary, fold, prior_record=None):
        return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": f"block_{i}", "default_pos": list(p),
                                     "default_ori": [1, 0, 0, 0], "_perception": {"confidence": 0.9}}
                                    for i, p in enumerate(POS)]}}

    def check(self, frame, objects, fold):
        return {o["label"]: {"head": {"iou": 0.9, "vis": 1.0, "n_px": 100}} for o in objects}

    def look_plan(self, frame, record, max_looks=2):
        return []

    def held_pose(self, frame, arm, key, prior_T_tcp_obj, cfg=None):
        self.reads.append((arm, key, dict(cfg or {})))
        return _fit(prior_T_tcp_obj, 0.001)                                     # the prior was right: nothing changes


@pytest.mark.parametrize("mode", ["off", "on"])
def test_episode_reads_every_grasp_once_without_extra_steps(mode):
    env = _env()
    pc = _Perception()
    pol = RuntimePolicy(RuntimeConfig(mode="perception", verify="off", carry_read=mode), assets=_Assets(),
                        perception_client=pc)
    pol.reset({"task": "stack_blocks", "instruction": STACK, "step_lim": env.step_lim, "layout_id": 1})
    for _ in range(2000):
        d = pol.next(_obs(env))
        if d.get("done"):
            break
        for a in d["actions"]:
            env.step(a)
    rep = pol.episode_report()
    assert rep["error"] is None
    picks = sum(1 for h in rep["history"] if h["skill"] == "pick" and h["ok"])
    ev = [e for e in rep["events"] if e["kind"] == "carry_read"]
    if mode == "off":
        assert ev == [] and pc.reads == []
    else:
        assert picks >= 2 and len(ev) == len(pc.reads) == picks and not any(e["adopted"] for e in ev)
        assert all(c["max_shift_mm"] == 25.0 and "others" in c for _, _, c in pc.reads)
