"""Prepare ten stable inputs per cell, in permitted id order, before scoring.

Run only after the smoke decision. One physics job at a time. Checkpointed
per-cell preparation permits restart without redrawing an existing record.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import argparse
import fcntl

from manifest import SALT, digest, validate_ids, verify
from host import Host, admission, arguments, host_from, regular_lock, terminate, wait_job, install_signals
from dojo.scoreboard.board import cells

IDS = list(range(100000, 100010)) + list(range(100020, 100030))
RAW = {'make_toast', 'make_toast_random', 'match_and_pick_from_conveyor', 'pick_from_conveyor_by_image'}


def run_gpu(host: Host, args: list[str], work: Path, log: Path) -> None:
    env = host.env() | {'DOJO_DATA': str(work)}
    pidfile = log.with_suffix('.pid')
    with admission(host) as release_startup, log.open('a') as stream:
        process = subprocess.Popen(['nice', '-n', '10', 'ionice', '-c3', str(host.python), '-m', *args],
                                   env=env, cwd=work, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        pidfile.write_text(str(process.pid))
        try:
            wait_job(process, 14400, release_startup)
            if process.returncode:
                raise RuntimeError(f'preparation failed: {args[0]}, rc={process.returncode}')
        finally:
            terminate(process, pidfile)


def prepare(host: Host, selected_tasks: list[str]) -> None:
    from dojo.simlab.gen_layouts import GENERATOR, LayoutError, generate
    validate_ids(IDS)
    work = host.state / 'preparation'
    target = host.state / 'yardstick-layouts'
    target.mkdir(exist_ok=True)
    manifest_path = target / 'manifest.json'
    source_hashes = {name: digest(host.code / name) for name in (
        'dojo/simlab/gen_layouts.py', 'dojo/simlab/presettle.py',
        'dojo/simlab/demo_gen.py', 'dojo/simlab/dev_eval.py')}
    previous = verify(target) if manifest_path.exists() else None
    if previous and previous.get('source_sha256') != source_hashes:
        raise ValueError('generator/runtime source changed across frozen cells')
    entries = previous['tasks'] if previous else {}
    for task in selected_tasks:
        checkpoint = work / f'{task}.prepared.json'
        if task in entries:
            continue
        if checkpoint.exists():
            chosen = json.loads(checkpoint.read_text())
            for entry in chosen:
                if digest(target / entry['file']) != entry['sha256']:
                    raise ValueError('prepared checkpoint hash mismatch')
            entries[task] = chosen
            continue
        directory = work / 'simlab/layouts' / task
        directory.mkdir(parents=True, exist_ok=True)
        for layout_id in IDS:
            path = directory / f'{layout_id}.json'
            if path.exists():
                continue
            try:
                path.write_text(json.dumps(generate(task, layout_id, seed_salt=SALT), indent=1))
            except LayoutError:
                (directory / f'{layout_id}.generation-failed').touch()
        log = work / f'{task}.log'
        args = ['dojo.simlab.presettle', '--task', task, '--ids', *map(str, IDS)]
        if task in RAW:
            run_gpu(host, args + ['--raw-check'], work, log)
            status = json.loads((directory / 'raw_index.json').read_text())
            eligible = [i for i in IDS if status.get(str(i), {}).get('status') == 'raw ok']
        else:
            run_gpu(host, args, work, log)
            run_gpu(host, args + ['--keep-raw-unmoved'], work, log)
            eligible = [i for i in IDS if (directory / f'{i}.settled.json').exists()]
        if task == 'imitate_sorting_sequence':
            pending, demonstrated = eligible, []
            while pending and len(demonstrated) < 10:
                batch, pending = pending[:10 - len(demonstrated)], pending[10 - len(demonstrated):]
                run_gpu(host, ['dojo.simlab.demo_gen', '--task', task, '--speed', '1.5', '--ids',
                              *map(str, batch)], work, log)
                demonstrated += [i for i in batch if (work / f'simlab/traj/{task}/{i}.pkl').exists()]
            eligible = demonstrated
        if len(eligible) < 10:
            raise SystemExit(f'{task}: only {len(eligible)} eligible inputs; no padding or scoring')
        chosen = []
        (target / task).mkdir(exist_ok=True)
        for position, layout_id in enumerate(eligible[:10]):
            suffix = '' if task in RAW else '.settled'
            src = directory / f'{layout_id}{suffix}.json'
            dst = target / task / f'{layout_id}.json'
            shutil.copyfile(src, dst)
            entry = dict(id=layout_id, position=position, file=str(dst.relative_to(target)),
                         sha256=digest(dst), source_form='raw' if task in RAW else 'settled')
            if task == 'imitate_sorting_sequence':
                demo = host.state / f'simlab/traj/{task}/{layout_id}.pkl'
                demo.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(work / f'simlab/traj/{task}/{layout_id}.pkl', demo)
                entry['demo_sha256'] = digest(demo)
            chosen.append(entry)
        entries[task] = chosen
        checkpoint.write_text(json.dumps(chosen, indent=2))
        print(task, 'prepared', flush=True)
    manifest = dict(schema='rsi-yardstick/1', salt=SALT, generator=GENERATOR,
                    tasks=entries, source_sha256=source_hashes,
                    preparation_script_sha256=digest(Path(__file__)),
                    preparation='first ten stable ids; imitation demo speed 1.5')
    temporary = manifest_path.with_suffix('.partial')
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    temporary.replace(manifest_path)
    (target / 'manifest.sha256').write_text(digest(manifest_path) + '  manifest.json\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    parser.add_argument('--tasks', nargs='+', required=True)
    args = parser.parse_args()
    host = host_from(args)
    install_signals()
    if not set(args.tasks) <= {task for task, _, _ in cells()}:
        parser.error('unknown task')
    host.directories()
    # Freeze operations are serialized; no existing selected input is redrawn.
    with regular_lock(host.state / 'freeze.lock', create=True) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prepare(host, args.tasks)


if __name__ == '__main__':
    main()
