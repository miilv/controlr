#!/usr/bin/env python3
"""Read only wave progress from our lane logs (no episode or sealed data reads)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Prospective occupied lane-hours, excluding preparation and resource waits.
HOURS = dict(waist=2.15, reloc=4.00, key=2.00, peg=4.14, disc=3.35, tie=14.64)


def summarize(schedule: dict, logs: str, lanes: int = 2) -> list[dict]:
    """Count each scheduled job once; retries retain their terminal launcher rc."""
    jobs = {}
    for pair in schedule['pairs']:
        for name in pair['order']:
            jobs[name] = pair['family']
    state = {}
    for line in logs.splitlines():
        match = re.search(r'\b(start|done) (c4b_\S+)(?: rc=(\d+))?', line)
        if not match:
            continue
        kind, attempt, rc = match.groups()
        name = attempt.split('__retry')[0]
        if name in jobs:
            state[name] = (kind, int(rc) if rc else None)
    rows = []
    for family, hours in HOURS.items():
        names = [n for n, f in jobs.items() if f == family]
        done = [n for n in names if state.get(n, ('',))[0] == 'done']
        running = [n for n in names if state.get(n, ('',))[0] == 'start']
        rc = {str(code): sum(state[n][1] == code for n in done) for code in sorted({state[n][1] for n in done})}
        rows.append(dict(block=family, total=len(names), done=len(done), running=len(running), rc=rc,
                         eta_hours=round(hours * (1 - len(done) / len(names)) / lanes, 2) if names else 0))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state', type=Path, default=Path.home() / 'dojo-integ-c4-state')
    ap.add_argument('--schedule', type=Path, default=Path(__file__).with_name('2026-11-06-wave.json'))
    ap.add_argument('--lanes', type=int, default=2)
    a = ap.parse_args()
    if a.lanes < 1:
        ap.error('--lanes must be positive')
    # Sort timestamped log lines so restarted lanes and retries resolve in time order.
    lines = []
    for lane in ('heavy1', 'heavy2'):
        p = a.state / f'lane2_{lane}.log'
        if p.exists():
            lines.extend(p.read_text(errors='replace').splitlines())
    print(json.dumps(dict(blocks=summarize(json.loads(a.schedule.read_text()), '\n'.join(sorted(lines)), a.lanes),
                          stopped=(a.state / 'STOP').exists(),
                          eta_note='Planning occupancy / lanes; excludes preparation, waits, retries and review barriers. '
                                   'done/rc are launcher results, not validated episode pairs.'), indent=2))


if __name__ == '__main__':
    main()
