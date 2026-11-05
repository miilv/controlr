#!/usr/bin/env python3
"""Aggregate old imitate candidate rows without emitting episode diagnostics.

Only the measurement slice and explicitly named candidate families are opened.
Run on a read-only host via ``ssh HOST python3 - ROOT < this_file``.
This reports attempts (including retries), not the board's candidate selection.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


ROW_DIR = re.compile(
    r"c4b_imitatesg_(gcon|o2o|o2otwin|o2owait|o2owaitlook|bundleon|bundleoff|bundlewid)_"
    r"(100\d{3})(?:__retry)?(?P<archive>\.codebug|\.layoutfreeze)?"
)
FIELDS = ("category", "success", "rc", "steps_used")


def census(root: Path) -> list[dict]:
    groups = defaultdict(list)
    for directory in sorted(root.iterdir()):
        match = ROW_DIR.fullmatch(directory.name)
        if not match:
            continue
        lid = int(match[2])
        if not (100005 <= lid <= 100009 or 100020 <= lid <= 100029):
            continue
        if directory.is_symlink() or not directory.is_dir():
            continue
        for file in sorted(directory.glob("*_episodes.jsonl")):
            if file.is_symlink() or not file.is_file():
                continue
            with file.open() as stream:
                for line in stream:
                    if line.strip():
                        raw = json.loads(line)
                        # Do not retain scores, reasons, trajectories, identities, or report bodies.
                        groups[match[1], match['archive'] or 'active'].append(
                            {key: raw.get(key) for key in FIELDS})
    result = []
    for (family, archive), rows in sorted(groups.items()):
        n = len(rows)
        categories = Counter(str(row['category']) for row in rows)
        infra = categories.get('infra_failure', 0)
        result.append(dict(family=family, archive=archive, attempts=n,
                           categories=dict(sorted(categories.items())),
                           infra=infra, infra_rate=round(infra / n, 4),
                           successes=sum(row['success'] is True for row in rows),
                           rc=dict(sorted(Counter(str(row['rc']) for row in rows).items())),
                           zero_steps=sum(row['steps_used'] == 0 for row in rows)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(census(parser.parse_args().root), indent=2))
