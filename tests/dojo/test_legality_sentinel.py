"""Mock episodes with task identity poisoned and schema reads tripwired."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.legality.sentinel import ForbiddenRead, TaskIdentity, instruction_only
from controlr.llm import FakeLLM
from dojo.agent.loop import AgentConfig
from dojo.agent.policy import AgentPolicy
from dojo.runtime.policy import RuntimeConfig, RuntimePolicy
from test_runtime import FakeAssets, FakeEpisode, _env, run_episode


class Perception:
    def __init__(self, positions):
        self.positions = positions
        self.calls = []

    def estimate(self, frames, vocabulary, fold, prior_record=None):
        # Observations only. This mock replaces the external perception service,
        # not runtime routing or the planner. Its vocabulary is inventoried statically.
        return {'Rigid': {'block': [dict(label=f'block_{i}', category='block', category_idx=5,
                    default_pos=list(p), default_ori=[1, 0, 0, 0], _perception={'confidence': .9})
                    for i, p in enumerate(self.positions)]}}

    def check(self, frame, objects, fold):
        return {o['label']: {'head': {'iou': .9, 'vis': 1., 'n_px': 100}} for o in objects}

    def look_plan(self, *args, **kwargs):
        return []


@pytest.mark.parametrize('model', [False, 'stack_objects', 'arrange'])
@pytest.mark.parametrize('instruction', ['Stack the three blocks with different textures.',
                                        'Stack the three bowls together.'])
def test_instruction_only_runtime_episode(monkeypatch, tmp_path, model, instruction):
    env = _env(instruction=instruction)
    pc = Perception([o['pos'] for o in env.objects])
    cfg = RuntimeConfig(mode='perception', verify='off', vocab_explain=False)
    if model:
        def reply(messages):
            import re
            ids = re.findall(r'^- (block_\d+):', messages[1]['content'][0]['text'], re.M)
            assert instruction in str(messages)
            call = ("robo.skill('stack_objects', objects=%r)" % ids if model == 'stack_objects'
                    else "robo.skill('arrange', steps=[['place', %r, %r]])" % (ids[0], ids[1]))
            return "```python\n" + call + "\nfinish()\n```"
        policy = AgentPolicy(cfg, agent_cfg=AgentConfig(model='fake', max_calls=2),
                             client=FakeLLM(reply), assets=FakeAssets(), perception_client=pc, log_root=tmp_path)
    else:
        policy = RuntimePolicy(cfg, assets=FakeAssets(), perception_client=pc)
    with instruction_only(monkeypatch) as hits:
        report = run_episode(policy, FakeEpisode(env, task=TaskIdentity()), 'perception')
    assert not hits, hits
    assert report['error'] is None, report['error']
    assert report.get('history'), report


def test_sentinel_catches_file_reads_even_when_exception_is_swallowed(monkeypatch):
    with instruction_only(monkeypatch) as hits:
        with pytest.raises(ForbiddenRead):
            Path('/nonexistent/tasks.json').read_text()
    assert hits


def test_sentinel_catches_preimported_schema_alias(monkeypatch):
    from dojo.runtime.vocab import load_sets as alias
    with instruction_only(monkeypatch) as hits:
        with pytest.raises(ForbiddenRead):
            alias(None)
    assert hits


def test_identity_cannot_be_used_for_routing():
    with pytest.raises(ForbiddenRead):
        str(TaskIdentity())


def test_unknown_instruction_exposes_legacy_schema_read(monkeypatch):
    # This records the current blocker without xfail hiding an unrelated error.
    # A future generic path should replace this with a zero-hit episode.
    from dojo.runtime.supervisor import perception_vocabulary
    with instruction_only(monkeypatch) as hits:
        with pytest.raises(ForbiddenRead):
            perception_vocabulary('Observe the unfamiliar scene and describe it.')
    assert hits and 'load_sets' in hits[0]


def test_ported_internal_boundary_currently_leaks_instruction(monkeypatch):
    # Partner glib findings 3/20 exist here too, below registry wrappers.
    # Preserve a precise negative control until the ported boundary is retired.
    from types import SimpleNamespace
    from dojo.skills.ported.harness_robot import HarnessRobot
    robot = object.__new__(HarnessRobot)
    monkeypatch.setattr(robot, '_world', lambda: SimpleNamespace(instruction=TaskIdentity()))
    monkeypatch.setattr(robot, 'eef', lambda: {})
    with pytest.raises(ForbiddenRead):
        robot.cmd_obs({'objects': False})


@pytest.mark.parametrize('skill', ['stack_objects', 'arrange'])
def test_explicit_argument_skills_accept_the_shared_instruction_sentinel(monkeypatch, skill):
    from dojo.legality.audit import sentinel
    from dojo.skills.skills import GTProvider, Skills
    from dojo.skills import library
    env = _env()
    sk = Skills(env, GTProvider(env, None, instruction=sentinel()))
    with instruction_only(monkeypatch) as hits:
        if skill == 'stack_objects':
            result = library.stack_objects(sk, ['block_0', 'block_1', 'block_2'])
        else:
            result = library.arrange(sk, [('place', 'block_0', 'block_1')])
    assert not hits
    assert result is not None
