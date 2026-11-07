"""Join official-client verdicts to RSI records after a call closes.

Usage: PYTHONPATH=. python baselines/rsi/summarize.py LAYOUTS RUN...
Returns episode rows only: incomplete/unstable cells must not become successes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from manifest import verify
from dojo.scoreboard.board import cells


def episodes(run: Path, manifest: dict) -> list[dict]:
    results = list((run / 'eval_result').rglob('_result.json'))
    if len(results) != 1:
        raise ValueError(f'{run}: expected one official result, found {len(results)}')
    result = json.loads(results[0].read_text())
    records = {}
    for path in (run / 'episodes').glob('*/result.json'):
        record = json.loads(path.read_text())
        key = (record['task'], int(record['seed']))
        if key in records:
            raise ValueError('duplicate episode record')
        records[key] = record
    rows = []
    task = run.name.split('_', 1)[1]
    entries = manifest['tasks'][task]
    seen = set()
    for detail in result['details'].values():
        position = int(detail['layout_id'])
        if position < 0 or position >= len(entries) or position in seen:
            raise ValueError('invalid or duplicate official layout position')
        seen.add(position)
        record = records[(task, position)]
        score = float(detail['score'])
        if not 0 <= score <= 1:
            raise ValueError('official score outside [0,1]')
        rows.append(dict(task=task, layout_id=entries[position]['id'], position=position,
                         layout_sha256=entries[position]['sha256'], success=bool(detail['success']),
                         official_score=score * 100, steps=record['env_steps_total'],
                         wall_s=record['total_wall_s'], run=str(run)))
    return rows


def per_cell(rows: list[dict]) -> dict:
    seen = set()
    for row in rows:
        key = row['task'], row['layout_id']
        if key in seen:
            raise ValueError(f'duplicate measured input: {key}')
        seen.add(key)
    output = {}
    for task, dimension, weight in cells():
        selected = [r for r in rows if r['task'] == task]
        scores = [r['official_score'] for r in selected]
        output[task] = dict(dimension=dimension, weight=weight, n=len(selected),
                            successes=sum(r['success'] for r in selected),
                            official_score=statistics.mean(scores) if scores else None,
                            score_sd=statistics.stdev(scores) if len(scores) > 1 else None,
                            steps=sum(r['steps'] for r in selected),
                            wall_s=sum(r['wall_s'] for r in selected), episodes=selected)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layouts', type=Path)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    manifest = verify(args.layouts)
    rows = [row for run in args.runs for row in episodes(run, manifest)]
    print(json.dumps(dict(schema='rsi-yardstick-results/1', per_cell=per_cell(rows)), indent=2))
