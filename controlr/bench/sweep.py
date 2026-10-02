"""Sweeps: an ablation is a cartesian product of config overrides x seeds.

Every experiment axis is a config field (controlr.config), so a sweep file is
just "base config + grid of ``--set`` overrides + seeds"::

    name: model_x_size
    config: ../sim_reach.yaml          # relative to this file
    set: ["planner.enabled=false"]     # fixed overrides for every run
    seeds: [0, 1, 2]
    grid:                              # key -> list of values (product over keys)
      llm.model: [claude/claude-sonnet-5, no-think/claude/claude-haiku-4-5-20251001]
      observation.size: [224, 448]

Each point runs as a normal episode with ``log.root`` = the sweep directory,
so every run dir is self-contained; ``aggregate.csv`` has one row per run.

Control-side ablations should pin the plan (``planner.plan_file`` in ``set``):
the Opus planner is slow and nondeterministic, so re-planning at every point
confounds each control axis with a different plan. ``--resume <sweep dir>``
skips points that already finished; ``--fake-llm`` dry-runs the whole sweep.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

import csv

from controlr.bench.cache_probe import default_out_dir
from controlr.runlog import write_csv

# outcomes that do not count as a finished point (re-run on --resume, non-zero exit)
FAILED_OUTCOMES = ("error", "llm_error", "interrupted")


@dataclass
class SweepPoint:
    index: int
    overrides: list[str]               # fixed + grid overrides (``key=value``)
    seed: int
    label: str                         # short human-readable id (grid values + seed)


@dataclass
class Sweep:
    name: str
    config: str | None
    set: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [0])
    grid: dict[str, list] = field(default_factory=dict)
    out_root: str = "runs"

    def points(self) -> list[SweepPoint]:
        keys = list(self.grid)
        combos = list(itertools.product(*(self.grid[k] for k in keys))) if keys else [()]
        pts: list[SweepPoint] = []
        for combo in combos:
            grid_ov = [f"{k}={_yaml_scalar(v)}" for k, v in zip(keys, combo)]
            for seed in self.seeds:
                label = ",".join([f"{k}={_short(v)}" for k, v in zip(keys, combo)] + [f"seed={seed}"])
                pts.append(SweepPoint(len(pts), list(self.set) + grid_ov, int(seed), label))
        return pts


def _yaml_scalar(v: Any) -> str:
    """Render a value so ``parse_override`` (yaml.safe_load) gives it back unchanged."""
    s = yaml.safe_dump(v, default_flow_style=True, width=10_000).strip()
    return s[:-4].strip() if s.endswith("\n...") else s.removesuffix("...").strip()


def _short(v: Any) -> str:
    s = str(v)
    return s.rsplit("/", 1)[-1][:32]


def load_sweep(path: str | Path) -> Sweep:
    p = Path(path)
    data = yaml.safe_load(p.read_text()) or {}
    data.setdefault("name", p.stem)
    cfg = data.get("config")
    if cfg and not Path(cfg).is_absolute():
        data["config"] = str((p.parent / cfg).resolve())
    data["set"] = list(data.get("set") or [])
    data["seeds"] = [int(s) for s in (data.get("seeds") or [0])]
    data["grid"] = {k: (v if isinstance(v, list) else [v]) for k, v in (data.get("grid") or {}).items()}
    return Sweep(**data)


def aggregate_row(point: SweepPoint, summary: dict) -> dict:
    lat = summary.get("latency") or {}
    tot = summary.get("totals") or {}
    row: dict[str, Any] = {"index": point.index, "label": point.label, "seed": point.seed}
    for ov in point.overrides:
        k, v = ov.split("=", 1)
        row[k] = v
    row.update(outcome=summary.get("outcome"), success=summary.get("success"),
               success_verified=summary.get("success_verified"), llm_backend=summary.get("llm_backend"),
               cache_regressions=len(summary.get("cache_regressions") or []),
               turns=summary.get("turns"),
               llm_end_p50=(lat.get("llm_end_s") or {}).get("p50"),
               llm_end_p90=(lat.get("llm_end_s") or {}).get("p90"),
               cycle_p50=(lat.get("cycle_s") or {}).get("p50"),
               prompt_tokens=tot.get("prompt_tokens"), completion_tokens=tot.get("completion_tokens"),
               cache_read_tokens=tot.get("cache_read_tokens"),
               cache_read_share=summary.get("cache_read_share"),
               error=summary.get("error"), run_dir=summary.get("run_dir"))
    return row


def check_overrides(sweep: Sweep, extra_overrides: list[str] | None) -> None:
    """``--set`` is applied after the grid values, so a ``--set`` on a grid key would
    silently collapse that axis to one value: refuse it."""
    keys = {ov.split("=", 1)[0].strip() for ov in extra_overrides or []}
    clash = sorted(keys & set(sweep.grid))
    if clash:
        raise ValueError(f"--set overrides grid key(s) {clash}; change the grid in the sweep file instead")


def _done_rows(out: Path) -> dict[int, dict]:
    """Rows of an existing aggregate.csv whose point finished (for --resume)."""
    p = out / "aggregate.csv"
    if not p.exists():
        return {}
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(r["index"]): r for r in rows if r.get("outcome") and r["outcome"] not in FAILED_OUTCOMES}


def run_sweep(sweep: Sweep, *, run_fn: Callable[[Any], Any] | None = None,
              extra_overrides: list[str] | None = None, out_dir: str | Path | None = None,
              resume: bool = False,
              printer: Callable[[str], None] | None = print) -> tuple[list[dict], Path]:
    """Run every point (sequentially — one robot, one sim). ``run_fn(cfg)``
    defaults to ``controlr.loop.run_episode``; it must return an object with
    ``summary()``. A failing point is recorded and the sweep continues; Ctrl-C
    (outcome "interrupted") stops the sweep after writing the aggregate.
    ``resume`` (with ``out_dir`` = an existing sweep dir): points already finished
    in its aggregate.csv are kept and not re-run."""
    from controlr.config import load_config

    check_overrides(sweep, extra_overrides)
    if run_fn is None:
        from controlr.loop import run_episode as run_fn  # type: ignore[assignment]
    out = Path(out_dir) if out_dir else default_out_dir(sweep.out_root, f"sweep_{sweep.name}")
    out.mkdir(parents=True, exist_ok=True)
    done = _done_rows(out) if resume else {}
    (out / "sweep.yaml").write_text(yaml.safe_dump(
        {"name": sweep.name, "config": sweep.config, "set": sweep.set, "seeds": sweep.seeds,
         "grid": sweep.grid}, sort_keys=False))
    pts = sweep.points()
    rows: list[dict] = []
    for pt in pts:
        if pt.index in done:
            rows.append(done[pt.index])
            if printer:
                printer(f"[{pt.index + 1}/{len(pts)}] {pt.label}  (done: {done[pt.index]['outcome']})")
            continue
        ov = pt.overrides + list(extra_overrides or []) + [f"seed={pt.seed}", f"log.root={out}"]
        if printer:
            printer(f"[{pt.index + 1}/{len(pts)}] {pt.label}")
        try:
            cfg = load_config(sweep.config, ov)
            cfg.name = f"{sweep.name}_{pt.index:03d}"
            summary = run_fn(cfg).summary()
        except Exception as e:  # noqa: BLE001 — one bad point must not kill the sweep
            summary = {"outcome": "error", "error": f"{type(e).__name__}: {e}"}
        row = aggregate_row(pt, summary)
        rows.append(row)
        write_csv(out / "aggregate.csv", rows)      # rewritten each point: crash-safe
        if printer:
            printer(f"    -> {row['outcome']} turns={row['turns']} success={row['success']}")
        if summary.get("outcome") == "interrupted":
            break
    return rows, out
