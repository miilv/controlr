"""Consumer-tolerance events (``Skills._consume``, ``RuntimeConfig.consume_log``): every contact-sensitive skill action
logs the belief pose it consumed and the precision it needs. Logging only: the executed actions must be identical with
it on or off. CPU only: the kinematic env and a fake perception client."""

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills.env import KinematicEnv
from dojo.skills.world import AssetInfo

STACK = "Stack the three blocks with different textures."
POS = [(0.412, -0.09, 0.78), (-0.26, -0.189, 0.78), (-0.344, -0.1, 0.78)]
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
KEYS = {"kind", "step", "skill", "label", "key", "pose7_used", "need", "purpose", "arm"}


def _env():
    rows = [{"label": f"block_{i}", "category": "block", "category_idx": 5, "pos": p, "quat": [1, 0, 0, 0],
             "half": [0.015] * 3} for i, p in enumerate(POS)]
    return KinematicEnv(rows, step_lim=550, instruction=STACK)


class _Assets:
    def get(self, typ, cat, idx, clutter=False, mesh=False):
        return AssetInfo(key=f"{typ}/{cat}/{int(idx):05d}", bbox_min=np.full(3, -0.015), bbox_max=np.full(3, 0.015))


class _Perception:
    def __init__(self, env):
        self.env = env

    def estimate(self, frames, vocabulary, fold, prior_record=None):
        return {"Rigid": {"block": [{"category": "block", "category_idx": 5, "label": o["label"],
                                     "default_pos": list(o["pos"]), "default_ori": [1, 0, 0, 0],
                                     "_perception": {"confidence": 0.9}} for o in self.env.objects]}}

    def check(self, frame, objects, fold):
        return {o["label"]: {"head": {"iou": 0.9, "vis": 1.0, "n_px": 100}} for o in objects}

    def look_plan(self, frame, record, max_looks=2):
        return []


def _episode(consume_log: bool, mode: str):
    env = _env()
    acts = []
    step0 = env.step

    def step(a):
        acts.append({k: np.asarray(v, float).copy() for k, v in a.items()})
        return step0(a)
    env.step = step
    pol = RuntimePolicy(RuntimeConfig(mode=mode, verify="off", consume_log=consume_log), assets=_Assets(),
                        perception_client=_Perception(env))
    pol.reset({"task": "stack_blocks", "instruction": STACK, "step_lim": env.step_lim, "layout_id": 1})
    for _ in range(2000):
        st = env.proprio()
        obs = {"images": {c: np.zeros((48, 64, 3), np.uint8) for c in CAMS}, "state": st, "instruction": STACK,
               "steps_used": env.steps_used, "step_lim": env.step_lim}
        if mode == "gt":
            obs["_dev_gt"] = env.gt_objects()
        d = pol.next(obs)
        if d.get("done"):
            break
        for a in d["actions"]:
            if env.step(a):
                break
    return pol.episode_report(), acts


@pytest.mark.parametrize("mode", ["perception", "gt"])
def test_logging_never_changes_an_action(mode):
    rep_on, acts_on = _episode(True, mode)
    rep_off, acts_off = _episode(False, mode)
    assert rep_on["error"] is None and rep_off["error"] is None
    assert len(acts_on) == len(acts_off) > 50
    for a, b in zip(acts_on, acts_off):
        assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
    ev_on = [e for e in rep_on["events"] if e["kind"] == "consume"]
    assert not [e for e in rep_off["events"] if e["kind"] == "consume"]
    picks = sum(1 for h in rep_on["history"] if h["skill"] == "pick")
    places = sum(1 for h in rep_on["history"] if h["skill"] == "place")
    assert sum(e["purpose"] == "grasp" for e in ev_on) == picks >= 2
    assert sum(e["purpose"] == "place" and e.get("role") == "held" for e in ev_on) == places >= 2


def test_event_schema_and_needs_from_the_skills_own_geometry():
    rep, _ = _episode(True, "perception")
    ev = [e for e in rep["events"] if e["kind"] == "consume"]
    assert ev and all(KEYS <= set(e) and len(e["pose7_used"]) == 7 for e in ev)
    assert all(set(e["need"]) == {"xy_mm", "z_mm", "yaw_deg"} for e in ev)
    grasp = [e for e in ev if e["purpose"] == "grasp"]
    # a 30 mm block in jaws opened to its width + 2 x 8 mm clearance + 6 mm: 11 mm per side
    assert all(e["skill"] == "pick" and 5.0 < e["need"]["xy_mm"] < 20.0 and e["need"]["z_mm"] is None for e in grasp)
    place = [e for e in ev if e["purpose"] == "place"]
    assert all(e["need"]["xy_mm"] > 0 and e["need"]["z_mm"] > 0 for e in place)
    held = [e for e in place if e["role"] == "held"]
    for e in held:                                   # the held object's pose7 is the target the place puts it at
        assert e["label"].startswith("block_") and e["step"] >= 0 and e["key"] == "Rigid/block/00005"
        assert len(e["T_tcp_obj7"]) == 7                 # and the in-hand transform the belief used (M's split)


def test_skills_emits_to_its_own_list_and_the_providers_hook_and_never_raises():
    from types import SimpleNamespace

    from dojo.skills.skills import Skills
    got = []
    sk = Skills.__new__(Skills)
    sk.consume_log, sk.consume_events = True, []
    sk.provider = SimpleNamespace(on_consume=got.append)
    sk.ex = SimpleNamespace(steps_used=7)
    sk._consume("pick", "a", np.eye(4), {"xy_mm": 3.0}, "grasp", "left")
    assert got == sk.consume_events and got[0]["need"] == {"xy_mm": 3.0, "z_mm": None, "yaw_deg": None}
    assert got[0]["step"] == 7 and got[0]["pose7_used"][3:] == [1.0, 0.0, 0.0, 0.0]
    sk._consume("pick", "a", "not a pose", {}, "grasp")               # logging only: a bad input is skipped
    sk.consume_log = False
    sk._consume("pick", "a", np.eye(4), {}, "grasp")
    assert len(sk.consume_events) == 1
