"""Stream H: ``put_into``'s "unblock" option (``Skills.put_into_opts``): a handover spot where the handed-over object
would leave another object no arm can pick is skipped. G3 put_bottles v15 draws: 2 of 16 episodes ended "no arm can
pick" on a lying bottle handed over in front of a standing one (either one moved away freed the other). CPU only,
kinematic env; the geometry is 100035's end state."""

import numpy as np

from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills import library
from dojo.skills.env import KinematicEnv
from dojo.skills.geometry import R_to_quat, quat_to_R, rot_z
from dojo.skills.skills import GTProvider, Skills

Z = 0.765
LYING = [0.4563, 0.7117, 0.0574, -0.5311]


def _sk(lying_xy=(0.25, -0.05)):
    rows = [dict(label="bottle3", category="bottle", category_idx=50, pos=[-0.042, -0.141, Z + 0.095],
                 quat=[0.9537, -0.0009, 0.0011, -0.3009], half=[0.0275, 0.0275, 0.095]),
            dict(label="bottle2", category="bottle", category_idx=5, pos=[lying_xy[0], lying_xy[1], Z + 0.0312],
                 quat=LYING, half=[0.0312, 0.0312, 0.087])]
    env = KinematicEnv(rows, step_lim=700, instruction="Put all the bottles into the dustbin.")
    return Skills(env, GTProvider(env, None, instruction=env.instruction), reserve=30)


def _T(xy, quat=LYING):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = quat_to_R(np.asarray(quat, float)), [xy[0], xy[1], Z + 0.0312]
    return T


def test_a_spot_in_front_of_a_standing_bottle_blocks_it():
    sk = _sk()
    sk.refresh()
    assert any(sk.can_pick("bottle3", a) for a in ("left", "right"))
    assert library._spot_blocks(sk, "bottle2", _T((-0.063, -0.257)), ["bottle3"]) == ["bottle3"]
    assert library._spot_blocks(sk, "bottle2", _T((0.2, -0.25)), ["bottle3"]) == []
    other_yaw = R_to_quat(rot_z(np.radians(90.0)) @ quat_to_R(np.asarray(LYING, float)))   # the same spot, another yaw
    assert library._spot_blocks(sk, "bottle2", _T((-0.063, -0.257), other_yaw), ["bottle3"]) == []
    o = sk.world.obj("bottle2")
    assert np.allclose(o.pos[:2], [0.25, -0.05])                  # the check leaves the world as it was


def test_unblock_is_off_by_default_and_validated():
    assert RuntimeConfig().put_into_opts == []
    RuntimePolicy(RuntimeConfig(mode="perception", put_into_opts=["spot", "unblock"]), assets=None)
    import pytest
    with pytest.raises(ValueError):
        RuntimePolicy(RuntimeConfig(mode="perception", put_into_opts=["unblok"]), assets=None)


def test_gentle_slows_the_drop_carry(monkeypatch):
    from dojo.skills.motion import CARRY, HeldBox, MoveResult
    from dojo.skills.skills import Held
    sk = _sk()
    sk.refresh()
    T = np.eye(4)                                    # the bottle's box centred on the TCP
    sk.held["left"] = Held("bottle3", T, HeldBox(T, np.zeros(3), np.array([0.0275, 0.0275, 0.095])))
    seen = []

    def to_pose(arm, q0, T_goal, step_rad=0.18, name="move", gripper=1.0, ignore=(), held=None, check=True, ease=4):
        seen.append((step_rad, ease))
        return MoveResult("ik_fail", arm, note="recorded")
    monkeypatch.setattr(sk.motion, "to_pose", to_pose)
    library.plan_drop(sk, "left", (-0.78, -0.1))
    assert seen and set(seen) == {(CARRY, 3)}                                    # today
    seen.clear()
    sk.put_into_opts = {"gentle"}
    library.plan_drop(sk, "left", (-0.78, -0.1))
    assert seen and set(seen) == {library.DROP_GENTLE}
