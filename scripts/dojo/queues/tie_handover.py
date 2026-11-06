"""Run on compute4 only after a clean compute6 first start; keep partially started pairs on compute4."""
import fcntl
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

home = Path.home()
state = home / 'dojo-integ-c4-state'
wave = home / 'dojo-data/wave-1106'
source = (wave / 'tie-queue.txt').read_text().splitlines()
assert len(source) == 108
def pair(name):
    result, count = re.subn(r'_tie[AB]r', '_tiePAIRr', name)
    assert count == 1, name
    return result
groups = {}
for line in source:
    name = line.split()[0]
    groups.setdefault(pair(name), set()).add(name)
assert all(len(names) == 2 for names in groups.values())
with (state / 'queue2.txt.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    q = state / 'queue2.txt'
    before = q.read_text()
    lines = before.splitlines()
    queued = [line.split()[0] for line in lines if line.strip()]
    expected_lines = {line.split()[0]: line for line in source}
    current_lines = {line.split()[0]: line for line in lines if line.strip()}
    eligible = {key for key, names in groups.items()
                if all(queued.count(name) == 1 and current_lines[name] == expected_lines[name] for name in names)}
    assert pair(source[0].split()[0]) in eligible, 'first compute6 pair already started on compute4; STOP and reconcile'
    moved_names = set().union(*(groups[key] for key in eligible))
    removed = [line for line in lines if line.strip() and line.split()[0] in moved_names]
    kept = [line for line in lines if not line.strip() or line.split()[0] not in moved_names]
    archive = wave / 'tie-moved-to-c6.txt'
    assert not archive.exists(), 'handover archive already exists; do not repeat handover'
    archive.write_text('\n'.join(removed) + '\n')
    after = '\n'.join(kept) + ('\n' if kept else '')
    tmp = state / 'queue2.txt.integ6.tmp'
    tmp.write_text(after)
    tmp.replace(q)
    report = {'time': datetime.now(timezone.utc).isoformat(), 'moved_rows': len(removed),
              'moved_names': sorted(moved_names), 'pairs_left_on_c4': sorted(set(groups) - eligible),
              'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
              'after_sha256': hashlib.sha256(after.encode()).hexdigest(), 'archive': str(archive)}
    with (wave / 'lead-notes.log').open('a') as log:
        log.write(json.dumps({'action': 'tie handover to compute6 after clean first start', **report}) + '\n')
    print(json.dumps(report))
