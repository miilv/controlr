"""Prune the strict server's observation PNGs of a CLOSED bridge episode in OUR bench run dir (isr-lab disk budget).

    .venv/bin/python scripts/dojo/isr_prune.py ID [ID ...] [--bench ~/data2/ilia/bench] [--runs ~/data2/ilia/dojo-data/bridge/runs]
        [--every 20] [--dry-run]

The server writes 3 PNGs per ``obs`` (``<bench>/run/<id>/obs/s<step>_<cam>.png``, ~1 MB per observation). Kept: the
first and the last observation, every ``--every``-th, and every step a perception or held-look call used:
``step`` / ``steps_used`` values anywhere in our ``report.json.gz`` (looks, refines, checks, agentic estimates, events)
and in the server's ``perceive.jsonl``. Refuses unless the episode closed (``launch_done.json``) and refuses any path
outside the given bench (only our own farm, never Mikhail's ``~/data2/bench``).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from pathlib import Path

PNG = re.compile(r"s(\d+)_(cam_[a-z_]+)\.png$")
STEP_KEYS = ("step", "steps_used", "step0", "at_step")


def used_steps(obj, out: set[int]) -> set[int]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in STEP_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool):
                out.add(int(v))
            else:
                used_steps(v, out)
    elif isinstance(obj, list):
        for v in obj:
            used_steps(v, out)
    return out


def plan(obs_dir: Path, keep_steps: set[int], every: int) -> tuple[list[Path], list[Path]]:
    by_step: dict[int, list[Path]] = {}
    for p in obs_dir.iterdir():
        m = PNG.match(p.name)
        if m:
            by_step.setdefault(int(m.group(1)), []).append(p)
    steps = sorted(by_step)
    if not steps:
        return [], []
    keep = {steps[0], steps[-1]} | {s for i, s in enumerate(steps) if every > 0 and i % every == 0} | (keep_steps & set(steps))
    kept = [p for s in steps if s in keep for p in by_step[s]]
    drop = [p for s in steps if s not in keep for p in by_step[s]]
    return kept, drop


def main(argv=None) -> None:
    ilia = Path("/home/anywherevla/data2/ilia")
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--bench", default=str(ilia / "bench"))
    ap.add_argument("--runs", default=str(ilia / "dojo-data" / "bridge" / "runs"))
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    bench = Path(os.path.realpath(a.bench))
    if bench == Path(os.path.realpath("/home/anywherevla/data2/bench")):
        raise SystemExit("refusing to prune Mikhail's bench")
    for rid in a.ids:
        rd = bench / "run" / rid
        if not (rd / "launch_done.json").exists():
            print(f"[prune] {rid}: not closed (no launch_done.json), skipped")
            continue
        steps: set[int] = set()
        rep = Path(a.runs) / rid / "report.json.gz"
        if rep.exists():
            with gzip.open(rep, "rt") as f:
                used_steps(json.load(f), steps)
        pj = rd / "perceive.jsonl"
        if pj.exists():
            for line in pj.read_text().splitlines():
                try:
                    used_steps(json.loads(line), steps)
                except json.JSONDecodeError:
                    pass
        obs = rd / "obs"
        if not obs.is_dir():
            print(f"[prune] {rid}: no obs dir")
            continue
        kept, drop = plan(obs, steps, a.every)
        freed = sum(p.stat().st_size for p in drop)
        if not a.dry_run:
            for p in drop:
                p.unlink()
        print(f"[prune] {rid}: kept {len(kept)} PNGs, {'would drop' if a.dry_run else 'dropped'} {len(drop)} "
              f"({freed / 1e6:.1f} MB); perception steps referenced: {len(steps)}")


if __name__ == "__main__":
    main()
