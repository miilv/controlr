"""FakeLLM + kinematic mock episode: logging cannot change actions or the real belief."""
import json
from types import SimpleNamespace

import numpy as np

from controlr.llm import FakeLLM
from dojo.runtime.belief import KinematicBelief
from dojo.runtime.estimate import objects_from_gt_rows
from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from dojo.skills.geometry import T_from
from dojo.skills.kinematics import X5Kin
from test_runtime import FakeAssets, FakeEpisode, _env


def _episode(logging):
    env, kin = _env(), X5Kin()
    ep = FakeEpisode(env)
    belief = KinematicBelief(objects_from_gt_rows(env.gt_objects(), FakeAssets()), kin, env.proprio())
    belief.attached['left'] = ('block_0', T_from([.015, 0, 0]))
    # Cached pose at the old arm position, before transporting it to the show pose.
    belief.sync_robot(env.proprio())
    syncs, observations, actions, states = [], [], [], []
    sync = belief.sync_robot
    belief.sync_robot = lambda state: (syncs.append(1), sync(state))[-1]
    pol = RuntimePolicy(RuntimeConfig(peg_held_read='on' if logging else 'off'))
    pol.report = {'events': [], 'skill': 'stack_on_pegs'}
    pol.bridge = SimpleNamespace(steps_used=0, observe=lambda: (observations.append(1), ep.observe())[-1])
    pol.perception = lambda: SimpleNamespace(held_pose=lambda obs, arm, key, prior, cfg: {
        'ok': True, 'T_tcp_obj': (np.asarray(prior) @ T_from([0, .003, 0])).tolist()})
    prov = SimpleNamespace(belief=belief, stats={})
    commands = [{'left_arm_joint_state': [q, .2, .3, 0, 0, 0], 'left_ee_joint_state': [0.],
                 'right_arm_joint_state': [0.] * 6, 'right_ee_joint_state': [1.]} for q in (.15, -.2, .3)]
    llm = FakeLLM([json.dumps(c) for c in commands])
    for _ in commands:
        command = json.loads(llm.complete('fake', [], max_tokens=200).text)
        actions.append(command)
        env.step(command)
        pol.bridge.steps_used = env.steps_used
        pol._held_pose(prov, 'left', belief.attached['left'][1].copy(), 'Rigid/block/00005')
        states.append(([o.T.copy() for o in belief.objects.values()], belief.attached['left'][1].copy(),
                       {a: q.copy() for a, q in belief.q.items()}, dict(prov.stats)))
    return actions, states, pol.report, len(syncs), len(observations)


def test_logging_transported_prior_changes_only_report_not_actions_or_belief():
    off, on = _episode(False), _episode(True)
    assert off[0] == on[0] and off[3:] == on[3:] == (3, 6)  # same actions, single sync and two observes per read
    for before, after in zip(off[1], on[1]):
        for x, y in zip(before[0], after[0]):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(before[1], after[1])
        for arm in before[2]:
            np.testing.assert_array_equal(before[2][arm], after[2][arm])
        assert before[3] == after[3]
    writes = [e for e in on[2]['events'] if e['kind'] == 'belief_write']
    assert len(writes) == 3
    for w in writes:
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(w['after7'][:3]) - w['before7'][:3]), .003, atol=1e-12)
        assert np.linalg.norm(np.asarray(w['before7'][:3]) - w['before_sync7'][:3]) > .01
