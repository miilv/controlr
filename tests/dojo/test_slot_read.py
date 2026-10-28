"""RuntimeConfig.slot_read: the pick hover of a slot-pinned object (deposit_coin's coin in its stand)."""
from pathlib import Path

import numpy as np
import pytest

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy


def test_fill_holes_fills_enclosed_background_only():
    pytest.importorskip("cv2")
    from dojo.perception.slot_read import _fill_holes
    fg = np.zeros((40, 40), np.uint8)
    fg[10:30, 10:30] = 1
    fg[15:25, 15:25] = 0                   # the stand's table-coloured insert inside its rim
    fg[0:5, 0:40] = 1                      # a band along the border: the background under it touches the border
    out = _fill_holes(fg, max_area=200)
    assert out[20, 20] == 1 and out[7, 20] == 0 and out[35, 35] == 0
    assert _fill_holes(fg, max_area=50)[20, 20] == 0          # a hole larger than allowed stays background


def test_slot_read_default_off():
    assert RuntimeConfig().slot_read == "off"
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(slot_read="maybe"))


def _setup(cfg, pc):
    from dojo.runtime.belief import RuntimeProvider
    from test_runtime import STACK, FakeAssets, FakeEpisode, _env, objects_from_gt_rows

    env = _env()
    pol = RuntimePolicy(cfg, assets=FakeAssets(), perception_client=pc)
    pol.reset(FakeEpisode(env).info())
    b = pol.bridge
    b.obs, b.fresh = FakeEpisode(env).observe(), True
    prov = RuntimeProvider(b, "stack_blocks", STACK, "perception", FakeAssets())
    prov.init_objects(objects_from_gt_rows(env.gt_objects(), None))
    return pol, prov


class SlotPerception:
    def __init__(self, applicable=True, ok=True):
        self.applicable, self.ok, self.slot_calls, self.refines = applicable, ok, [], 0

    def slot_read(self, frame, task, label, objects, arm, cfg=None):
        self.slot_calls.append((task, label, arm, sorted(o["label"] for o in objects)))
        if not self.applicable:
            return {"applicable": False, "why": "not slot-pinned"}
        ob = {o["label"]: o for o in objects}
        child = list(ob[label]["pos"]) + [0.0, 0.0, 0.0, 1.0]
        child[1] += 0.011
        base = list(ob["block_0"]["pos"]) + [0.92387953, 0.0, 0.0, 0.38268343]
        return {"applicable": True, "base_label": "block_0", "ok": self.ok, "why": None if self.ok else "low iou",
                "iou": 0.6 if self.ok else 0.3, "yaw_margin": 0.2, "child_pose": child, "base_pose": base}

    def refine(self, frame, objects, fold, max_shift_m=0.03, arm=None, identity_vocabulary=None, held=None, **kw):
        self.refines += 1
        return {objects[0]["label"]: {"ok": False, "pos": objects[0]["pos"], "quat": objects[0]["quat"]}}


def test_slot_read_ok_writes_child_and_base_and_skips_the_refine():
    pc = SlotPerception()
    pol, prov = _setup(RuntimeConfig(mode="perception", refine=True, slot_read="on"), pc)
    y0 = float(prov.belief.objects["block_1"].pos[1])
    out = pol._refine_hook(prov, None, "block_1", "left", "pick")
    assert pc.slot_calls and pc.slot_calls[0][:3] == ("stack_blocks", "block_1", "left")
    assert pc.refines == 0 and set(out) == {"block_1"}
    assert abs(float(prov.belief.objects["block_1"].pos[1]) - y0 - 0.011) < 1e-9
    assert np.allclose(prov.belief.objects["block_0"].quat, [0.92387953, 0.0, 0.0, 0.38268343])
    ev = [e for e in pol.report["events"] if e["kind"] == "slot_read"][-1]
    assert ev["applied"] and ev["base"] == "block_0"


@pytest.mark.parametrize("applicable,ok", [(True, False), (False, True)])
def test_slot_read_not_ok_or_not_applicable_leaves_the_refine(applicable, ok):
    pc = SlotPerception(applicable, ok)
    pol, prov = _setup(RuntimeConfig(mode="perception", refine=True, slot_read="on"), pc)
    p0 = prov.belief.objects["block_1"].pos.copy()
    pol._refine_hook(prov, None, "block_1", "left", "pick")
    assert pc.refines == 1 and np.allclose(prov.belief.objects["block_1"].pos, p0)
    evs = [e for e in pol.report["events"] if e["kind"] == "slot_read"]
    assert (len(evs) == 1 and evs[0]["applied"] is False) if applicable else not evs


def test_slot_read_off_or_place_never_calls_it():
    pc = SlotPerception()
    pol, prov = _setup(RuntimeConfig(mode="perception", refine=True), pc)
    pol._refine_hook(prov, None, "block_1", "left", "pick")
    pol2, prov2 = _setup(RuntimeConfig(mode="perception", refine=True, slot_read="on"), pc)
    pol2._refine_hook(prov2, None, "block_1", "left", "place")
    assert pc.slot_calls == [] and pc.refines == 2


CATALOG = Path("~/dojo-perception-data/catalog/tasks.json").expanduser()


# recorded pick hovers (dev replays, GT stand): deposit_coin 100006 step 24 (6.0 render, right wrist) and 100002 step 59
# (5.1 camera row c4b_coin_off: the stand read +32.6 deg vs GT -15.9, the coin turned with it, nothing lifted)
HOVERS = [("right", [0.3807, -0.1944, 0.7736], [0.9803, 0.0, 0.0, 0.1977], [-0.3359, 0.3529, 0.3554, -0.6975, 0.0233, 0.0147]),
          ("left", [-0.2386, -0.0552, 0.7736], [0.9904, 0.0, 0.0, -0.1384], [-0.1446, 2.0831, 1.8455, -1.3361, 0.0106, 0.855])]


@pytest.mark.skipif(not CATALOG.is_file(), reason="needs the perception catalogue (GPU boxes)")
@pytest.mark.parametrize("case", HOVERS)
def test_slot_read_recovers_a_48_deg_stand_turn_on_a_synthetic_hover_frame(case):
    """The coin + stand rendered at a known pose (white stand, gold coin, the stand's insert table-coloured, on a
    wood-coloured table) under a recorded pick hover; the read starts from the stand turned 48 deg and 8 mm off."""
    import cv2

    from dojo.perception.cameras import R_to_quat_wxyz, pose_to_T, quat_wxyz_to_R, wrist_camera
    from dojo.perception.inverse_layout import support_slot_pose, task_map
    from dojo.perception.render import load_asset, silhouette, transform
    from dojo.perception.slot_read import slot_read

    stand_k, coin_k = "Geometry/vertical_coin_stand/00000", "Rigid/coin/00000"
    pl = task_map()["deposit_coin"]["labels"]["coin0"]["placement"]
    arm, p_stand, q_stand, joints = case
    q_stand = list(np.asarray(q_stand) / np.linalg.norm(q_stand))
    state = {f"{arm}_arm_joint_state": np.asarray(joints, float),
             f"{'right' if arm == 'left' else 'left'}_arm_joint_state": np.zeros(6)}
    cam = wrist_camera(arm, state[f"{arm}_arm_joint_state"])
    cp, cq = support_slot_pose(stand_k, p_stand, q_stand, coin_k, pl)
    a_s, a_c = load_asset(stand_k), load_asset(coin_k)
    s_sil = silhouette(cam, transform(a_s.Vs, pose_to_T(p_stand, q_stand)), a_s.Fs)
    c_sil = silhouette(cam, transform(a_c.Vs, pose_to_T(cp, cq)), a_c.Fs)
    if s_sil.sum() < 200:
        pytest.skip("the synthetic hover does not see the stand")
    rng = np.random.default_rng(0)
    img = np.clip(np.array([110, 50, 30]) + rng.normal(0, 6, (cam.h, cam.w, 3)), 0, 255).astype(np.uint8)
    rim = s_sil.astype(bool) & ~cv2.erode(s_sil, np.ones((7, 7), np.uint8)).astype(bool)
    img[rim] = (225, 225, 225)
    img[c_sil.astype(bool)] = (220, 170, 40)
    R = np.array([[np.cos(0.84), -np.sin(0.84), 0], [np.sin(0.84), np.cos(0.84), 0], [0, 0, 1.0]]) @ quat_wxyz_to_R(q_stand)
    prior = [p_stand[0] + 0.008, p_stand[1], p_stand[2]] + list(R_to_quat_wxyz(R))
    out = slot_read(stand_k, prior, coin_k, pl, img, state, arm)
    assert out["ok"], out
    k = int(np.argmin(a_c.extents))
    n_gt, n = quat_wxyz_to_R(cq)[:, k], quat_wxyz_to_R(out["child_pose"][3:])[:, k]
    assert np.degrees(np.arccos(min(1.0, abs(float(n @ n_gt))))) < 8.0
    e = np.asarray(out["child_pose"][:3]) - np.asarray(cp)
    assert abs(float(e @ n_gt)) < 0.004
