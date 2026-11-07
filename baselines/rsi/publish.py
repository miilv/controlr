"""Publish completed RSI bars from our state file to a dedicated docs worktree.

CPU/SSH file reads only; no GPU commands, remote writes or reboot services.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import time

from dojo.scoreboard.board import cells
from dojo.simlab.paths import is_our_measurement_id

DESTINATION = Path('docs/experiments/data/rsi-yardstick/per_cell.json')


def validate(payload: dict) -> set[str]:
    if payload.get('schema') != 'rsi-yardstick-results/1' or payload.get('expected_per_cell') != 5:
        raise ValueError('expected the first-five result schema')
    rows = payload['per_cell']
    if set(rows) != {task for task, _, _ in cells()}:
        raise ValueError('expected exactly 54 cells')
    done = set()
    for task, cell in rows.items():
        if cell['n'] == 0:
            if cell['official_score'] is not None or cell['episodes']:
                raise ValueError('missing cells must remain null')
            continue
        if cell['n'] != 5 or len(cell['episodes']) != 5 or not 0 <= cell['successes'] <= 5:
            raise ValueError('only closed five-episode cells can publish')
        score = cell['official_score']
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError('invalid official score')
        ids = [episode['layout_id'] for episode in cell['episodes']]
        if ids != sorted(set(ids)) or any(not is_our_measurement_id(i) for i in ids):
            raise ValueError('off-slice, repeated or reordered input')
        if [episode['position'] for episode in cell['episodes']] != list(range(5)):
            raise ValueError('first five call positions required')
        done.add(task)
    return done


def appended(previous: dict | None, payload: dict) -> set[str]:
    done = validate(payload)
    old = validate(previous) if previous else set()
    if not old <= done or any(previous['per_cell'][task] != payload['per_cell'][task] for task in old):
        raise ValueError('a published cell changed; refusing overwrite')
    return done - old


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(checkout), *args], text=True).strip()


def publish(checkout: Path, payload: dict) -> int:
    if git(checkout, 'status', '--porcelain'):
        raise ValueError('publisher checkout must be clean')
    path = checkout / DESTINATION
    git(checkout, 'fetch', 'origin', 'main')
    git(checkout, 'rebase', 'origin/main')
    changed = set(git(checkout, 'diff', '--name-only', 'origin/main...HEAD').splitlines())
    if changed - {str(DESTINATION)}:
        raise ValueError('publisher branch contains non-bar changes; refusing direct main push')
    previous = json.loads(path.read_text()) if path.exists() else None
    added = appended(previous, payload)
    if not added:
        return len(validate(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)
    git(checkout, 'add', str(DESTINATION))
    git(checkout, 'commit', '-m', 'docs: publish RSI first-five bars for ' + ', '.join(sorted(added)))
    # Only this dedicated worktree's docs commit goes to main. Never force main.
    for attempt in range(3):
        try:
            git(checkout, 'push', 'origin', 'HEAD:main')
            return len(validate(payload))
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            git(checkout, 'fetch', 'origin', 'main')
            git(checkout, 'rebase', 'origin/main')
    raise AssertionError('unreachable')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--remote-state', type=Path, required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--max-hours', type=float, default=36)
    args = parser.parse_args()
    if not args.remote_state.is_absolute() or args.host.startswith('-') or args.interval < 60:
        parser.error('absolute state path, SSH host and interval>=60 required')
    deadline = time.monotonic() + args.max_hours * 3600
    last = None
    while True:
        remote = 'nice -n 10 ionice -c3 cat -- ' + shlex.quote(str(args.remote_state / 'per_cell.json'))
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host, remote],
                                capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            payload = json.loads(result.stdout)
            validate(payload)
            if payload != last:
                total = publish(args.checkout, payload)
                last = payload
                print(f'published {total}/54 cells', flush=True)
                if total == 54:
                    return
        elif not args.watch:
            raise RuntimeError('state file unavailable')
        if not args.watch or time.monotonic() >= deadline:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
