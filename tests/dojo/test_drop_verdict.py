"""``library.drop_verdict`` (``Skills.drop_verdict`` / ``RuntimeConfig.drop_verdict``): how a drop into a container is
judged. Numbers from rendered put_bottles (arm 8, gcon 100003 @136): a lying bottle let go over the dustbin's open strip
past the table edge, believed afterwards on the floor beside it ([-0.805, -0.1, 0.176]: "missed") while it had fallen in.
CPU only."""

from types import SimpleNamespace

import numpy as np
import pytest

from dojo.skills.library import drop_verdict
from dojo.skills.world import AssetInfo, ObjectState

LYING = [0.70711, 0.0, 0.70711, 0.0]             # lying, its long axis along the table edge's normal (world x)


def _obj(label, typ, pos, quat, half, source="predicted"):
    a = AssetInfo(key=f"{typ}/{label}/00000", bbox_min=-np.asarray(half, float), bbox_max=np.asarray(half, float))
    return ObjectState(label, typ, label, 0, np.asarray(pos, float), np.asarray(quat, float), a, source=source)


def _sk(mode="release", bottle_pos=(-0.805, -0.1, 0.176), source="predicted", remeasure=False):
    bottle = _obj("bottle0", "Rigid", bottle_pos, LYING, [0.0275, 0.0275, 0.085], source)
    bin_ = _obj("dustbin", "Geometry", [-0.63, -0.1, 0.35], [1, 0, 0, 0], [0.235, 0.235, 0.325])
    objs = {"bottle0": bottle, "dustbin": bin_}
    placed = []
    world = SimpleNamespace(objects=objs, obj=lambda lab: objs[lab])
    prov = SimpleNamespace(place_inside=lambda lab, c, centre: placed.append((lab, c, np.asarray(centre, float))))
    return SimpleNamespace(world=world, provider=prov, drop_verdict=mode, remeasure=remeasure), placed


def _release(x):
    T = np.eye(4)
    T[:3, 3] = [x, -0.1, 0.88]                     # the object's centre at the release (held at the TCP)
    return T


HELD = SimpleNamespace(T_tcp_obj=np.eye(4), label="bottle0")


def test_belief_mode_is_todays_check_and_misses_the_bottle():
    sk, placed = _sk("belief")
    assert drop_verdict(sk, "bottle0", "dustbin", HELD, _release(-0.805), True) == (False, "belief")
    assert placed == []


def test_release_over_the_footprint_with_the_jaws_holding_is_in_and_the_belief_puts_it_inside():
    sk, placed = _sk("release")
    ok, why = drop_verdict(sk, "bottle0", "dustbin", HELD, _release(-0.805), True)
    assert ok and why == "released over the opening"
    assert placed and placed[0][:2] == ("bottle0", "dustbin") and abs(placed[0][2][0] + 0.805) < 1e-9


@pytest.mark.parametrize("x, holding, why", [(-0.90, True, "released outside the container's footprint"),
                                             (-0.805, False, "jaws empty before the release")])
def test_release_outside_or_with_empty_jaws_is_a_miss(x, holding, why):
    sk, placed = _sk("release")
    assert drop_verdict(sk, "bottle0", "dustbin", HELD, _release(x), holding) == (False, why)
    assert placed == []


def test_a_post_release_measurement_decides_through_the_belief():
    sk, _ = _sk("release", bottle_pos=(-0.62, -0.1, 0.11), source="perception")      # seen inside the bin
    assert drop_verdict(sk, "bottle0", "dustbin", HELD, _release(-0.805), True) == (True, "measured")
    sk, _ = _sk("release", bottle_pos=(-1.2, -0.1, 0.03), source="perception")       # seen on the floor
    assert drop_verdict(sk, "bottle0", "dustbin", HELD, _release(-0.805), True) == (False, "measured")


def test_mode_gt_keeps_the_measured_world():
    sk, placed = _sk("release", remeasure=True)
    assert drop_verdict(sk, "bottle0", "dustbin", HELD, _release(-0.805), True) == (False, "belief")


def test_provider_place_inside_rests_the_object_on_the_containers_floor():
    from dojo.runtime.belief import RuntimeProvider
    from dojo.runtime.bridge import Bridge
    b = Bridge()
    st = {f"{a}_{k}": np.zeros(n) for a in ("left", "right") for k, n in (("arm_joint_state", 6), ("ee_pose", 7))}
    st.update({"left_ee_joint_state": np.array([1.0]), "right_ee_joint_state": np.array([1.0])})
    b.obs, b.fresh = {"images": {}, "state": st, "instruction": "x", "steps_used": 0, "step_lim": 700}, True
    prov = RuntimeProvider(b, "x", "x", "perception")
    sk, _ = _sk("release")
    prov.init_objects(dict(sk.world.objects))
    prov.place_inside("bottle0", "dustbin", [-0.805, -0.1, 0.88])
    o = prov.belief.objects["bottle0"]
    lo, hi = o.aabb()
    mn, mx = prov.belief.objects["dustbin"].aabb()
    assert mn[0] <= lo[0] and hi[0] <= mx[0] and abs(float(lo[2]) - (float(mn[2]) + 0.005)) < 1e-9
    assert prov.events[-1]["kind"] == "drop_inside" and "bottle0" in prov.touched
    assert prov.holds("left", "bottle0") is False
