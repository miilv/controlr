"""Portable RSI commands and desktop/lane admission: CPU only, no real GPU queries."""
import argparse
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'baselines/rsi'))
import host as H
import runner as R


@pytest.fixture
def host(tmp_path):
    return H.Host(root=tmp_path / 'install', state=tmp_path / 'state', code=tmp_path / 'source',
                  rdhome=tmp_path / 'farm', checkpoint=tmp_path / 'models/sam3.pt',
                  lane='heavy1', lock=tmp_path / 'gpu.lock', profile='desktop')


def sample(**changes):
    return dict(used_mb=671, free_mb=32000, power_w=10, load1=1, available_gb=30) | changes


@pytest.mark.parametrize('changed,allowed', [({}, True), ({'used_mb': 2048}, True),
    ({'used_mb': 2049}, False), ({'load1': 8}, True), ({'load1': 8.01}, False),
    ({'power_w': 300}, True), ({'power_w': 300.01}, False)])
def test_desktop_thresholds(host, changed, allowed):
    assert H.decision(host, sample(**changed))[0] == allowed


def test_allowlist_and_portable_commands(host, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'do-not-forward')
    monkeypatch.setenv('OMNIROUTE_API_KEY', 'do-not-forward-either')
    policy, client, env = R.commands(host, 'deposit_coin', host.state / 'layouts', 5, 12345, host.state / 'out')
    assert env['EVAL_NUM'] == '5'
    assert env['HOME'] == str(host.root / 'home')
    assert env['SAM3_CHECKPOINT_PATH'] == str(host.checkpoint)
    assert 'OPENAI_API_KEY' not in env and 'OMNIROUTE_API_KEY' not in env
    assert policy[0] == str(host.root / 'venv/bin/python')
    assert client[0] == str(host.python)
    assert '--num_envs' in client and client[client.index('--num_envs') + 1] == '1'
    assert R.tasks()[:3] == ['imitate_sorting_sequence', 'match_and_pick_from_conveyor', 'pour_by_language']
    assert len(R.tasks()) == len(set(R.tasks())) == 54


def test_parser_requires_explicit_paths_and_one_desktop_lane(host):
    parser = argparse.ArgumentParser()
    H.arguments(parser)
    parsed = parser.parse_args(host.args())
    assert H.host_from(parsed) == host
    parsed.lane = 'heavy2'
    with pytest.raises(ValueError, match='one job'):
        H.host_from(parsed)
    parsed.profile = 'camera'
    with pytest.raises(ValueError, match='lane-state'):
        H.host_from(parsed)


def test_desktop_yields_five_minutes_and_holds_slot(host, monkeypatch):
    host.lock.touch()
    monkeypatch.setattr(H, 'validate_scope', lambda h: None)
    monkeypatch.setattr(H.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100 * 1024**3))
    samples = iter([sample(used_mb=3000), sample()])
    sleeps = []
    with H.admission(host, sample_fn=lambda: next(samples), sleep=sleeps.append):
        with host.lock.open('r+') as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert sleeps == [300]
    rows = [json.loads(line) for line in (host.state / 'admission.jsonl').read_text().splitlines()]
    assert [r['admitted'] for r in rows] == [False, True]
    assert all(r['own_load_subtracted'] == 0 for r in rows)


def test_failed_query_yields_and_stop_refuses_start(host, monkeypatch):
    host.lock.touch()
    monkeypatch.setattr(H, 'validate_scope', lambda h: None)
    def fail():
        raise subprocess.CalledProcessError(1, 'nvidia-smi')
    def stop_after_wait(seconds):
        assert seconds == 300
        (host.state / 'STOP').touch()
    with pytest.raises(RuntimeError, match='STOP'):
        with H.admission(host, sample_fn=fail, sleep=stop_after_wait):
            pytest.fail('query failure must not admit')


def test_lock_symlink_refused(host):
    target = host.lock.with_name('foreign.lock')
    target.touch()
    host.lock.symlink_to(target)
    with pytest.raises(OSError):
        H.regular_lock(host.lock)


def test_camera_caps_share_existing_lane_counters(host, monkeypatch):
    state = host.root.parent / 'camera-state'
    state.mkdir()
    (state / 'lane2.caps').write_text('2 4')
    (state / 'running.heavy1').write_text(str(os.getpid()))
    (state / 'side.heavy1').write_text('1')
    host = replace(host, profile='camera', lane='heavy2', lane_state=state, stream='integ-test')
    monkeypatch.setattr(H.subprocess, 'check_output', lambda *a, **kw: '')
    assert H.camera_caps(host)
    (state / 'lane2.caps').write_text('1 4')
    assert not H.camera_caps(host)
    assert not H.decision(host, sample(used_mb=16385))[0]


def test_cleanup_ignores_completed_or_unrecorded_child(host, monkeypatch):
    process = Mock(pid=123, poll=lambda: None)
    pidfile = host.lock.with_name('child.pid')
    pidfile.write_text('124')
    kill = Mock()
    monkeypatch.setattr(H.os, 'killpg', kill)
    H.terminate(process, pidfile)
    kill.assert_not_called()


def test_dry_run_never_queries_gpu_or_writes(host, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['runner.py', 'run', *host.args(), '--dry-run'])
    monkeypatch.setattr(H.subprocess, 'check_output', lambda *a, **k: pytest.fail('GPU query'))
    R.main()
    result = json.loads(capsys.readouterr().out)
    assert result['lock'] == str(host.lock)
    assert not host.state.exists()


def test_incremental_publication_requires_closed_cell_count(host):
    import manifest as M
    host.directories()
    layouts = host.state / 'yardstick-layouts'
    (layouts / 'stack_blocks').mkdir(parents=True)
    entries = []
    for position, layout_id in enumerate(range(100000, 100010)):
        path = layouts / f'stack_blocks/{layout_id}.json'
        path.write_text('{}')
        entries.append(dict(id=layout_id, position=position, file=f'stack_blocks/{layout_id}.json', sha256=M.digest(path)))
    manifest = layouts / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema='rsi-yardstick/1', salt=M.SALT, tasks=dict(stack_blocks=entries))))
    (layouts / 'manifest.sha256').write_text(M.digest(manifest))
    run = host.state / 'yardstick/20261107T123456Z_stack_blocks'
    (run / 'eval_result').mkdir(parents=True)
    details = {}
    for position in range(5):
        details[str(position)] = dict(layout_id=position, success=True, score=1)
        ep = run / f'episodes/episode{position}'
        ep.mkdir(parents=True)
        (ep / 'result.json').write_text(json.dumps(dict(task='stack_blocks', seed=position,
                                                       env_steps_total=100, total_wall_s=2)))
    result_path = run / 'eval_result/_result.json'
    result_path.write_text(json.dumps(dict(details=details)))
    (host.state / 'yardstick/stack_blocks.done').write_text(str(run))
    output = R.publish(host)
    cell = output['per_cell']['stack_blocks']
    assert cell['n'] == 5 and cell['successes'] == 5
    assert [e['layout_id'] for e in cell['episodes']] == list(range(100000, 100005))
    assert output['per_cell']['stack_blocks_random']['n'] == 0
    original = (host.state / 'per_cell.json').read_bytes()
    result_path.write_text(json.dumps(dict(details={})))
    with pytest.raises(ValueError, match='expected scored count'):
        R.publish(host)
    assert (host.state / 'per_cell.json').read_bytes() == original


@pytest.mark.parametrize('memory,quota,allowed', [
    (str(16 * 1024**3), '800000 100000', True),
    (str(17 * 1024**3), '800000 100000', False),
    ('max', '800000 100000', False),
    (str(16 * 1024**3), '900000 100000', False),
    (str(16 * 1024**3), 'max 100000', False),
])
def test_desktop_checks_actual_scope_limits(host, monkeypatch, memory, quota, allowed):
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path == Path('/proc/self/cgroup'):
            return '0::/user.slice/rsi-test.scope\n'
        if path.name == 'memory.max':
            return memory
        if path.name == 'cpu.max':
            return quota
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    if allowed:
        H.validate_scope(host)
    else:
        with pytest.raises(RuntimeError, match='scope must have'):
            H.validate_scope(host)


def test_suffix_keeps_native_positions_and_first_pass_results(host):
    import importlib.util
    spec = importlib.util.spec_from_file_location('rsi_suffix', Path(R.__file__).with_name('client.py'))
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    class Seeds:
        def init_eval(self, completed_layout_ids=None, abandoned_layout_ids=None):
            self.seed_info = {i: 100000 + i for i in range(10)}
            excluded = set(completed_layout_ids or []) | set(abandoned_layout_ids or [])
            self.seed_list = [i for i in range(10) if i not in excluded]
    client.exclude_prefix(Seeds, 5)
    manager = Seeds()
    manager.init_eval(abandoned_layout_ids=[7])
    assert manager.seed_list == [5, 6, 8, 9]
    assert manager.seed_info[5] == 100005
    policy, command, env = R.commands(host, 'deposit_coin', host.state / 'layouts', 5, 12345,
                                      host.state / 'out', start_position=5)
    assert command[2].endswith('/client.py') and env['RSI_START_POSITION'] == '5'
    with pytest.raises(ValueError, match='desktop permission'):
        R.run(host, 'deposit_coin', host.state / 'layouts', 5, 'yardstick-remaining', start_position=5)
