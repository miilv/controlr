"""Prepare ten stable inputs per cell, in permitted id order, before scoring.

Run only after the smoke decision. One physics job at a time. Checkpointed
per-cell preparation permits restart without redrawing an existing record.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from manifest import SALT, digest, validate_ids
from dojo.scoreboard.board import cells

ROOT = Path('/home/robotics/controlr-rsi-repro')
DATA = Path('/home/robotics/dojo-data/rsi-repro')
IDS = list(range(100000, 100010)) + list(range(100020, 100030))
RAW = {'make_toast', 'make_toast_random', 'match_and_pick_from_conveyor', 'pick_from_conveyor_by_image'}


def run_gpu(args: list[str], work: Path, log: Path) -> None:
    while True:
        sample = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw',
                                          '--format=csv,noheader,nounits'], text=True)
        if all(float(v) <= 450 for v in sample.splitlines()):
            break
        time.sleep(30)
    env = dict(os.environ, DOJO_DATA=str(work), DOJO_GPU_LOCK=str(DATA / 'gpu.lock'), SIMLAB_GPU='1')
    with log.open('a') as stream:
        subprocess.run(['bash', str(ROOT / 'code/scripts/simlab/rdpy.sh'), '-m', *args],
                       env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    from dojo.simlab.gen_layouts import GENERATOR, LayoutError, generate
    validate_ids(IDS)
    work = DATA / 'preparation'
    target = DATA / 'yardstick-layouts'
    target.mkdir(exist_ok=True)
    manifest_path = target / 'manifest.json'
    if manifest_path.exists():
        raise SystemExit('already frozen; verify instead of preparing again')
    entries = {}
    for task, _, _ in cells():
        checkpoint = work / f'{task}.prepared.json'
        if checkpoint.exists():
            entries[task] = json.loads(checkpoint.read_text())
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
            run_gpu(args + ['--raw-check'], work, log)
            status = json.loads((directory / 'raw_index.json').read_text())
            eligible = [i for i in IDS if status.get(str(i), {}).get('status') == 'raw ok']
        else:
            run_gpu(args, work, log)
            run_gpu(args + ['--keep-raw-unmoved'], work, log)
            eligible = [i for i in IDS if (directory / f'{i}.settled.json').exists()]
        if task == 'imitate_sorting_sequence':
            run_gpu(['dojo.simlab.demo_gen', '--task', task, '--speed', '1.5', '--ids',
                     *map(str, eligible)], work, log)
            eligible = [i for i in eligible if (work / f'simlab/traj/{task}/{i}.pkl').exists()]
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
                demo = DATA / f'simlab/traj/{task}/{layout_id}.pkl'
                demo.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(work / f'simlab/traj/{task}/{layout_id}.pkl', demo)
                entry['demo_sha256'] = digest(demo)
            chosen.append(entry)
        entries[task] = chosen
        checkpoint.write_text(json.dumps(chosen, indent=2))
        print(task, 'prepared', flush=True)
    manifest = dict(schema='rsi-yardstick/1', salt=SALT, generator=GENERATOR,
                    tasks=entries, preparation='first ten stable ids; imitation demo speed 1.5')
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    (target / 'manifest.sha256').write_text(digest(manifest_path) + '  manifest.json\n')


if __name__ == '__main__':
    main()
