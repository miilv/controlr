"""Command line: ``controlr run | bench-cache | bench-latency | sweep | report | models``.

Thin argparse layer: every subcommand loads a Config (or a bench/sweep file)
and calls one function in ``controlr.loop`` / ``controlr.bench``; no logic
lives here that a script could not call directly.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Sequence


def _seeds(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="controlr", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run episode(s) of a config")
    r.add_argument("-c", "--config", required=True)
    r.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--episodes", type=int, default=1, help="N episodes, seeds cfg.seed .. cfg.seed+N-1")
    g.add_argument("--seeds", type=_seeds, default=None, help="comma list, e.g. 0,1,2")
    r.add_argument("--fake-llm", nargs="?", const="-", default=None, metavar="FILE",
                   help="no network: scripted FakeLLM replies (FILE: replies separated by lines "
                        "'---'; no FILE: a short built-in script ending in DONE). Dry runs of the "
                        "whole pipeline, any backend.")

    b = sub.add_parser("bench-cache", help="growing image transcript: per-turn latency + cache-read share")
    b.add_argument("--model", required=True)
    b.add_argument("--turns", type=int, default=20)
    b.add_argument("--size", type=int, default=448)
    b.add_argument("--frames", default=None, help="frame dir or run dir (default: synthetic frames)")
    b.add_argument("-c", "--config", default=None, help="config for llm endpoint/cache settings")
    b.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")

    lt = sub.add_parser("bench-latency", help="latency matrix (models x sizes x history x variants)")
    lt.add_argument("-c", "--config", required=True, help="e.g. configs/bench/latency.yaml")
    lt.add_argument("--dry-run", action="store_true", help="print the call count and exit")
    lt.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                    help="llm endpoint overrides (llm.base_url=..., llm.timeout_s=...)")

    s = sub.add_parser("sweep", help="cartesian product of overrides x seeds")
    s.add_argument("sweep")
    s.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    s.add_argument("--dry-run", action="store_true", help="list the points and exit")

    rp = sub.add_parser("report", help="summary table of run dir(s)")
    rp.add_argument("paths", nargs="+", help="run dirs, sweep dirs or globs")
    rp.add_argument("--csv", default=None, help="also write the table as CSV")

    m = sub.add_parser("models", help="list models on the endpoint (GET /models)")
    m.add_argument("--filter", default=None)
    m.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    return p


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def _progress(rec: dict) -> None:
    llm = rec.get("llm") or {}
    u = rec.get("usage") or {}
    end = llm.get("t_end")
    print(f"  turn {rec['turn']:>3} {rec.get('status') or '-':>5}  llm {end if end is None else round(end, 2)} s"
          f"  read {u.get('cache_read_tokens', '-')}/{u.get('prompt_tokens', '-')}"
          f"  {' | '.join(a['raw'] for a in rec.get('actions') or [])[:80]}", flush=True)


# Built-in dry-run script: a few small moves in the default text grammar, then DONE.
# Runs against any backend / ee_delta config; the goal check decides the outcome.
FAKE_SCRIPT = [
    "Moving right a little.\nMOVE ee_delta 20 0 0\nSTATUS OK",
    "MOVE ee_delta 0 20 0\nSTATUS OK",
    "MOVE ee_delta 0 0 -10 GRIP close\nSTATUS OK",
    "HOLD\nSTATUS DONE dry run finished",
]


def fake_llm_from_arg(arg: str):
    """``--fake-llm`` value -> FakeLLM (``-`` = built-in script)."""
    from controlr.llm.fake import FakeLLM

    if arg == "-":
        return FakeLLM(list(FAKE_SCRIPT), ttft_s=0.05)
    text = Path(arg).read_text()
    replies = [r.strip() for r in text.split("\n---\n") if r.strip()]
    return FakeLLM(replies, ttft_s=0.05)


def cmd_run(args) -> int:
    from controlr.config import load_config
    from controlr.loop import run_episode

    cfg0 = load_config(args.config, args.sets)
    seeds = args.seeds if args.seeds is not None else [cfg0.seed + i for i in range(args.episodes)]
    results = []
    for seed in seeds:
        cfg = load_config(args.config, args.sets + [f"seed={seed}"])
        print(f"episode {cfg.name} seed={seed} model={cfg.llm.model} backend={cfg.robot.backend}", flush=True)
        llm = fake_llm_from_arg(args.fake_llm) if args.fake_llm else None
        res = run_episode(cfg, llm=llm, on_turn=_progress)
        results.append(res)
        print(f"-> {res.outcome} success={res.success} turns={res.turns} "
              f"cache_share={_fmt(res.cache_read_share)} {res.run_dir}"
              + (f"\n   error: {res.error}" if res.error else ""), flush=True)
        if res.outcome == "interrupted":
            break
    if len(results) > 1:
        n = sum(r.success for r in results)
        print(f"success {n}/{len(results)}")
    return 0 if all(r.outcome not in ("error", "llm_error", "interrupted") for r in results) else 1


def _client(cfg_path: str | None, sets: list[str]):
    from controlr.config import load_config
    from controlr.loop import make_llm_client

    cfg = load_config(cfg_path, sets)
    return cfg, make_llm_client(cfg)


def cmd_bench_cache(args) -> int:
    from controlr.bench.cache_probe import default_out_dir, run_cache_probe

    cfg, llm = _client(args.config, args.sets)
    out = default_out_dir(cfg.log.root, "bench_cache")
    rows = run_cache_probe(llm, args.model, turns=args.turns, size=args.size, frames_dir=args.frames,
                           cache=cfg.llm.cache, ttl=cfg.llm.cache_ttl,
                           jpeg_quality=cfg.observation.jpeg_quality, out_dir=out)
    print(f"wrote {out}/probe.csv ({len(rows)} calls)")
    return 0 if not any(r.error for r in rows) else 1


def cmd_bench_latency(args) -> int:
    from controlr.bench.latency import load_bench, run_latency

    bench = load_bench(args.config)
    if args.dry_run:
        print(f"{len(bench.cells())} cells, {bench.plan_calls()} calls")
        return 0
    # grace: keep reading after STATUS for the usage chunk, else no cache stats;
    # the control-relevant latency is t_complete (STATUS seen), reported separately
    _, llm = _client(None, ["llm.usage_grace_s=1.0"] + args.sets)
    run_latency(bench, llm)
    return 0


def cmd_sweep(args) -> int:
    from controlr.bench.sweep import load_sweep, run_sweep

    sw = load_sweep(args.sweep)
    if args.dry_run:
        for pt in sw.points():
            print(f"{pt.index:3d} {pt.label}  {' '.join(pt.overrides)}")
        return 0
    rows, out = run_sweep(sw, extra_overrides=args.sets)
    print(f"wrote {out}/aggregate.csv ({len(rows)} runs)")
    return 0


def _fmt(v, nd: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def report_rows(paths: Sequence[str]) -> list[dict]:
    from controlr.runlog import find_run_dirs, load_run

    dirs: list[Path] = []
    for pat in paths:
        hits = glob.glob(pat) or [pat]
        for h in sorted(hits):
            dirs.extend(d for d in find_run_dirs(h) if d not in dirs)
    rows = []
    for d in dirs:
        s = load_run(d)
        lat = s.get("latency") or {}
        tot = s.get("totals") or {}
        rows.append({
            "run": d.name, "model": s.get("model"), "outcome": s.get("outcome"),
            "success": s.get("success"), "turns": s.get("turns"),
            "llm_p50": (lat.get("llm_end_s") or {}).get("p50"),
            "llm_p90": (lat.get("llm_end_s") or {}).get("p90"),
            "cycle_p50": (lat.get("cycle_s") or {}).get("p50"),
            "prompt_tok": tot.get("prompt_tokens"), "compl_tok": tot.get("completion_tokens"),
            "cache_share": s.get("cache_read_share"),
        })
    return rows


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "(no runs)"
    keys = list(rows[0])
    cells = [[_fmt(r.get(k)) for k in keys] for r in rows]
    w = [max(len(k), *(len(c[i]) for c in cells)) for i, k in enumerate(keys)]
    line = lambda xs: "  ".join(x.ljust(w[i]) for i, x in enumerate(xs))  # noqa: E731
    return "\n".join([line(keys), line(["-" * x for x in w])] + [line(c) for c in cells])


def cmd_report(args) -> int:
    rows = report_rows(args.paths)
    print(format_table(rows))
    if args.csv:
        from controlr.bench.cache_probe import write_csv
        write_csv(args.csv, rows)
    return 0 if rows else 1


def cmd_models(args) -> int:
    import httpx

    from controlr.config import load_config

    cfg = load_config(None, args.sets)
    key = os.environ.get(cfg.llm.api_key_env, "")
    r = httpx.get(cfg.llm.base_url.rstrip("/") + "/models",
                  headers={"Authorization": f"Bearer {key}"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    ids = sorted(m.get("id", "") for m in (data.get("data") or []))
    for i in ids:
        if not args.filter or args.filter.lower() in i.lower():
            print(i)
    return 0


COMMANDS = {"run": cmd_run, "bench-cache": cmd_bench_cache, "bench-latency": cmd_bench_latency,
            "sweep": cmd_sweep, "report": cmd_report, "models": cmd_models}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.cmd](args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
