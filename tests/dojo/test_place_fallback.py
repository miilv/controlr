"""A place planned together with its pick must not open the gripper on the way (the fallback after a failed
post-pick re-plan ran such a plan, and its approach let go of the object)."""

import numpy as np

from dojo.skills.env import KinematicEnv
from dojo.skills.kinematics import TABLE_Z
from dojo.skills.skills import GTProvider, Skills


def test_a_place_planned_with_its_pick_keeps_the_gripper_closed():
    env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.25, -0.15, TABLE_Z + 0.015],
                         "quat": [1, 0, 0, 0], "half": [0.015] * 3}])
    sk = Skills(env, GTProvider(env, None))
    T = np.asarray(sk.world.obj("b").T, float).copy()
    T[:2, 3] = (-0.2, -0.2)
    r = sk.plan_transfer("b", [T])
    assert not hasattr(r, "note"), r
    pp, pl = r
    assert pl.plan.g0 == 0.0                        # planned before the pick, while the gripper is open
    comp = pl.plan.compile()[0]
    assert all(g == 0.0 for _, g in comp[:5])       # the approach does not open it


def test_run_place_keeps_a_holding_gripper_closed_with_a_stale_plan():
    env = KinematicEnv([{"label": "b", "category": "block", "pos": [-0.25, -0.15, TABLE_Z + 0.015],
                         "quat": [1, 0, 0, 0], "half": [0.015] * 3}])
    sk = Skills(env, GTProvider(env, None))
    T = np.asarray(sk.world.obj("b").T, float).copy()
    T[:2, 3] = (-0.2, -0.2)
    pp, pl = sk.plan_transfer("b", [T])
    pl.plan.g0 = 1.0                                # a plan made while the gripper was open
    assert sk.run_pick(pp).ok
    seen = []
    step = env.step

    def spy(action):
        seen.append(float(np.asarray(action["left_ee_joint_state"]).reshape(-1)[0]))
        return step(action)
    env.step = spy
    sk.run_place(pl, remeasure=False)
    k = next(i for i, g in enumerate(seen) if g > 0.5) if any(g > 0.5 for g in seen) else len(seen)
    assert k > 5 and abs(sk.refresh().obj("b").pos[0] + 0.2) < 0.01     # opened only at the place, which it reached
