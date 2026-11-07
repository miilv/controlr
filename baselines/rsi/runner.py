"""Portable RSI smoke/yardstick runner. Every GPU job shares the configured lane gate."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import socket
import sys
import subprocess
import time
import uuid

from host import CHECKPOINT_SHA256, Host, admission, arguments, host_from, stopped, terminate, wait_job, install_signals, regular_lock
from manifest import digest, verify
from summarize import episodes, per_cell

PRIORITY = ['imitate_sorting_sequence', 'match_and_pick_from_conveyor', 'pour_by_language',
            'classify_objects_by_language', 'deposit_coin', 'classify_objects', 'organize_table',
            'put_bottles_into_dustbin', 'fill_pen_holder', 'fill_egg_holder', 'pick_from_conveyor_by_image']


def tasks() -> list[str]:
    from dojo.scoreboard.board import cells
    return PRIORITY + [task for task, _, _ in cells() if task not in PRIORITY]


def spawn(command: list[str], host: Host, out: Path, label: str, env: dict) -> subprocess.Popen:
    """Owned session leader; log/PID files are created before waiting or cleanup."""
    with (out / f'{label}.log').open('w') as stream:
        process = subprocess.Popen(['nice', '-n', '10', 'ionice', '-c3', *command], cwd=out,
                                   env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    (out / f'{label}.pid').write_text(str(process.pid))
    return process


def commands(host: Host, task: str, layouts: Path, count: int, port: int, out: Path):
    rd = host.rdhome / 'RoboDojo'
    xpl = rd / 'XPolicyLab'
    policy = [str(host.root / 'venv/bin/python'), str(xpl / 'setup_policy_server.py'),
              '--config_path', str(xpl / 'policy/PyRUA_RSI/deploy.yml'), '--overrides',
              f'port={port}', 'host=127.0.0.1', 'policy_name=PyRUA_RSI', 'bench_name=RoboDojo',
              f'task_name={task}', 'ckpt_name=offline', 'env_cfg_type=arx_x5', 'seed=0', 'action_type=joint']
    client = [str(host.python), '-u', str(host.code / 'dojo/simlab/dev_eval.py'),
              '--task_name', task, '--env_cfg_type', 'arx_x5', '--num_envs', '1', '--enable_cameras',
              '--kit_args', ' --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera',
              '--device_id', '0', '--policy_name', 'PyRUA_RSI', '--port', str(port), '--protocol', 'ws',
              '--policy_server_url', f'ws://127.0.0.1:{port}', '--additional_info', 'rsi_yardstick',
              '--seed', '0', '--host', '127.0.0.1', '--headless']
    env = host.env() | dict(SIMLAB_DEV_LAYOUTS=str(layouts), EVAL_NUM=str(count),
                           PYRUALEAN_RESULT_DIR=str(out / 'episodes'), ROBODOJO_RUN_ID=out.name.split('_')[0])
    return policy, client, env


def run(host: Host, task: str, layouts: Path, count: int, phase: str, timeout: int = 3600) -> Path:
    import fcntl
    host.directories()
    with regular_lock(host.state / 'freeze.lock', create=True) as freeze:
        fcntl.flock(freeze, fcntl.LOCK_EX)
        manifest = verify(layouts)
        manifest_sha256 = digest(layouts / 'manifest.json')
    if task not in manifest['tasks'] or not 1 <= count <= len(manifest['tasks'][task]):
        raise ValueError('task/count absent from frozen inputs')
    if digest(host.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('checkpoint hash mismatch')
    host.directories()
    with admission(host) as release_startup:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + uuid.uuid4().hex[:6] + 'Z'
        out = host.state / phase / f'{stamp}_{task}'
        (out / 'episodes').mkdir(parents=True)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        policy_cmd, client_cmd, env = commands(host, task, layouts, count, port, out)
        (out / 'call.json').write_text(json.dumps(dict(task=task, count=count, lane=host.lane,
            manifest_sha256=manifest_sha256, ids=[e['id'] for e in manifest['tasks'][task][:count]])))
        children = []
        try:
            policy = spawn(policy_cmd, host, out, 'policy', env)
            children.append(('policy', policy))
            ready = False
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline and policy.poll() is None:
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=1):
                        ready = True
                        break
                except OSError:
                    time.sleep(1)
            if not ready:
                raise RuntimeError(f'policy did not become ready: {out}')
            client = spawn(client_cmd, host, out, 'eval', env)
            children.append(('eval', client))
            wait_job(client, timeout, release_startup)
            if client.returncode:
                raise RuntimeError(f'eval client failed ({client.returncode}): {out}')
        finally:
            for label, process in reversed(children):
                terminate(process, out / f'{label}.pid')
    return out


def publish(host: Host, expected: int = 5) -> dict:
    """Atomic per-cell publication; missing cells stay missing; duplicate calls refuse."""
    import fcntl
    with regular_lock(host.state / 'publication.lock', create=True) as lock, regular_lock(host.state / 'freeze.lock', create=True) as freeze:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fcntl.flock(freeze, fcntl.LOCK_EX)
        rows = []
        completed = set()
        for marker in sorted((host.state / 'yardstick').glob('*.done')):
            completed.add(marker.stem)
            run_dir = Path(marker.read_text().strip())
            manifest = verify(host.state / 'yardstick-layouts')
            rows.extend(episodes(run_dir, manifest))
        result = dict(schema='rsi-yardstick-results/1', expected_per_cell=expected, per_cell=per_cell(rows))
        for task, cell in result['per_cell'].items():
            if task in completed and cell['n'] != expected:
                raise ValueError('cell did not produce its expected scored count')
        path = host.state / 'per_cell.json'
        temp = path.with_suffix('.json.partial')
        temp.write_text(json.dumps(result, indent=2) + '\n')
        temp.replace(path)
    return result


def batch(host: Host, count: int, shard: int, shards: int) -> None:
    if host.profile == 'desktop' and (shards != 1 or count != 5):
        raise ValueError('desktop permission: one worker, five episodes per cell')
    if not 0 <= shard < shards <= 2:
        raise ValueError('at most two disjoint shards')
    manifest_path = host.state / 'yardstick-layouts'
    for index, task in enumerate(tasks()):
        if index % shards != shard:
            continue
        if stopped(host):
            return
        done = host.state / 'yardstick' / f'{task}.done'
        if done.exists():
            continue
        # Freeze missing cells before this cell is scored; prior cells/records are immutable.
        driver = host.state / f'prepare-driver.{host.lane}.pid'
        process = subprocess.Popen(['nice', '-n', '10', 'ionice', '-c3', str(host.python),
                                    str(Path(__file__).with_name('prepare.py')), *host.args(), '--tasks', task],
                                   env=host.env(), start_new_session=True)
        driver.write_text(str(process.pid))
        try:
            process.wait()
            if process.returncode:
                raise RuntimeError(f'preparation exited {process.returncode}: {task}')
        finally:
            terminate(process, driver)
        start = done.with_suffix('.started')
        start.parent.mkdir(parents=True, exist_ok=True)
        with start.open('x') as stream:
            stream.write(datetime.datetime.now(datetime.timezone.utc).isoformat())
        out = run(host, task, manifest_path, count, 'yardstick')
        done.write_text(str(out))
        publish(host, count)
        print(task, 'scored', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'batch', 'smoke', 'collect'))
    arguments(parser)
    parser.add_argument('--task')
    parser.add_argument('--layouts', type=Path)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--phase', choices=('smoke', 'yardstick'), default='yardstick')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true', help='CPU only: print resolved commands, no writes or GPU query')
    args = parser.parse_args()
    host = host_from(args)
    if args.dry_run:
        policy, client, env = commands(host, args.task or PRIORITY[0], args.layouts or host.state / 'yardstick-layouts',
                                       args.episodes, 12345, host.state / 'dry-run')
        print(json.dumps(dict(policy=policy, client=client, environment=env, profile=host.profile,
                              lane=host.lane, lock=str(host.lock), order=tasks()), indent=2))
        return
    install_signals()
    host.directories()
    if args.mode == 'batch':
        (host.state / f'runner.{host.lane}.pid').write_text(str(__import__('os').getpid()))
        watcher = spawn([sys.executable, str(Path(__file__).with_name('power_watch.py')),
                         '--state', str(host.state)], host, host.state, 'power-watch.' + host.lane, host.env())
        try:
            batch(host, args.episodes, args.shard, args.shards)
        finally:
            terminate(watcher, host.state / f'power-watch.{host.lane}.pid')
    elif args.mode == 'collect':
        publish(host, args.episodes)
    elif args.mode == 'smoke':
        for task in ('stack_blocks', 'press_by_number', 'insert_tubes'):
            run(host, task, args.layouts or host.state / 'smoke-layouts', 2, 'smoke')
    elif args.task and args.layouts:
        print(run(host, args.task, args.layouts, args.episodes, args.phase))
    else:
        parser.error('run requires --task and --layouts')


if __name__ == '__main__':
    main()
