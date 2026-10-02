"""Command line: ``controlr run | prompt | bench-cache | bench-latency | sweep | report | models``.

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
    s.add_argument("--resume", default=None, metavar="SWEEP_DIR",
                   help="continue an interrupted sweep in SWEEP_DIR (finished points are kept)")
    s.add_argument("--fake-llm", nargs="?", const="-", default=None, metavar="FILE",
                   help="no network: every point runs with FakeLLM (see run --fake-llm)")

    pr = sub.add_parser("prompt", help="print the system prompt of a config (no robot reset, no LLM)")
    pr.add_argument("-c", "--config", required=True)
    pr.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE")
    pr.add_argument("--setup", default=None, metavar="RUN_DIR",
                    help="use a recorded run's reference state + cameras (setup.json / cameras.json)")
    pr.add_argument("--planner", action="store_true", help="print the planner system prompt instead")

    rp = sub.add_parser("report", help="summary table of run dir(s)")
    rp.add_argument("paths", nargs="+", help="run dirs, sweep dirs or globs")
    rp.add_argument("--csv", default=None, help="also write the table as CSV")
    rp.add_argument("--include-fake", action="store_true", help="also list --fake-llm dry runs")

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


FAKE_PLAN = "1. Locate the target.\n2. Move above it.\n3. Descend and finish."


def fake_llm_from_arg(arg: str, planner: bool = False):
    """``--fake-llm`` value -> FakeLLM (``-`` = built-in script). With the planner
    enabled the built-in script starts with a plan reply, so the control replies are
    not consumed by the planner call (a FILE must then start with its own plan)."""
    from controlr.llm.fake import FakeLLM

    if arg == "-":
        return FakeLLM(([FAKE_PLAN] if planner else []) + list(FAKE_SCRIPT), ttft_s=0.05)
    text = Path(arg).read_text()
    replies = [r.strip() for r in text.split("\n---\n") if r.strip()]
    return FakeLLM(replies, ttft_s=0.05)


def _status(msg: str) -> None:
    print(f"  {msg}", flush=True)


def _uses_planner(cfg) -> bool:
    return cfg.planner.enabled and not cfg.planner.plan_file


def cmd_run(args) -> int:
    from controlr.config import load_config
    from controlr.loop import run_episode

    cfg0 = load_config(args.config, args.sets)
    seeds = args.seeds if args.seeds is not None else [cfg0.seed + i for i in range(args.episodes)]
    results = []
    for seed in seeds:
        cfg = load_config(args.config, args.sets + [f"seed={seed}"])
        print(f"episode {cfg.name} seed={seed} model={cfg.llm.model} backend={cfg.robot.backend}"
              + (" [fake llm]" if args.fake_llm else ""), flush=True)
        llm = fake_llm_from_arg(args.fake_llm, _uses_planner(cfg)) if args.fake_llm else None
        res = run_episode(cfg, llm=llm, on_turn=_progress, on_status=_status)
        results.append(res)
        print(f"-> {res.outcome} success={res.success} turns={res.turns} "
              f"cache_share={_fmt(res.cache_read_share)} {res.run_dir}"
              + (f"\n   error: {res.error}" if res.error else ""), flush=True)
        if res.cache_regressions:
            print(f"   WARNING: cache regressions at turns {[r['turn'] for r in res.cache_regressions]} "
                  f"(a turn read less than the previous call cached; upstream rotation?)", flush=True)
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
    from controlr.bench.sweep import FAILED_OUTCOMES, check_overrides, load_sweep, run_sweep

    sw = load_sweep(args.sweep)
    check_overrides(sw, args.sets)
    if args.dry_run:
        for pt in sw.points():
            print(f"{pt.index:3d} {pt.label}  {' '.join(pt.overrides)}")
        return 0
    run_fn = None
    if args.fake_llm:
        from controlr.loop import run_episode

        def run_fn(cfg):      # a fresh scripted LLM per point
            return run_episode(cfg, llm=fake_llm_from_arg(args.fake_llm, _uses_planner(cfg)), on_status=_status)
    rows, out = run_sweep(sw, run_fn=run_fn, extra_overrides=args.sets, out_dir=args.resume,
                          resume=bool(args.resume))
    print(f"wrote {out}/aggregate.csv ({len(rows)} runs)")
    bad = [r for r in rows if r.get("outcome") in FAILED_OUTCOMES]
    if bad:
        print(f"{len(bad)} point(s) failed: {', '.join(str(r['index']) for r in bad)} "
              f"(rerun with --resume {out})", file=sys.stderr)
        return 1
    return 0


def cmd_prompt(args) -> int:
    """Render the manual without a backend reset: the mock rig's spec + D435 camera,
    or a recorded run's reference state and cameras (``--setup RUN_DIR``)."""
    import json

    import numpy as np

    from controlr.config import load_config
    from controlr.prompts.builder import build_planner_prompt, build_system_prompt
    from controlr.robot.mock import d435_camera
    from controlr.robot.spec import ur3_cb3_spec
    from controlr.types import CameraInfo, RobotState

    cfg = load_config(args.config, args.sets)
    spec = ur3_cb3_spec()
    cams = {"scene": d435_camera()}
    state0 = None
    if args.setup:
        run = Path(args.setup)
        setup = json.loads((run / "setup.json").read_text())
        st = setup.get("reference_state") or setup.get("state0")
        if st:
            state0 = RobotState(t=0.0, q=np.asarray(st["q"]), tcp_pos=np.asarray(st["tcp_pos"]),
                                tcp_rotvec=np.asarray(st["tcp_rotvec"]), gripper_mm=float(st["gripper_mm"]),
                                gripper_closed=bool(st["gripper_closed"]), holding=st.get("holding"))
        if (run / "cameras.json").exists():
            cams = {n: CameraInfo(n, c["width"], c["height"], np.asarray(c["K"]), np.asarray(c["T_cam_base"]))
                    for n, c in json.loads((run / "cameras.json").read_text()).items()}
        if (run / "spec.json").exists():
            from controlr.types import JointSpec, RobotSpec
            d = json.loads((run / "spec.json").read_text())
            d["joints"] = tuple(JointSpec(**j) for j in d["joints"])
            spec = RobotSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})
    build = build_planner_prompt if args.planner else build_system_prompt
    print(build(cfg, spec, cameras=cams, state0=state0), end="")
    return 0


def _fmt(v, nd: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def report_rows(paths: Sequence[str], include_fake: bool = False) -> list[dict]:
    """One row per run dir. Dry runs (``llm_backend: fake``, or a ``fake_`` dir name) are
    skipped unless ``include_fake``: they must not leak into model comparisons."""
    from controlr.runlog import find_run_dirs, load_run

    dirs: list[Path] = []
    for pat in paths:
        hits = glob.glob(pat) or [pat]
        for h in sorted(hits):
            dirs.extend(d for d in find_run_dirs(h) if d not in dirs)
    rows = []
    for d in dirs:
        s = load_run(d)
        backend = s.get("llm_backend") or ("fake" if "_fake_" in f"_{d.name.split('_', 1)[-1]}" else "live")
        if backend == "fake" and not include_fake:
            continue
        lat = s.get("latency") or {}
        tot = s.get("totals") or {}
        rows.append({
            "run": d.name, "name": s.get("name"), "task": s.get("task"), "seed": s.get("seed"),
            "backend": s.get("backend"), "llm": backend, "model": s.get("model"), "outcome": s.get("outcome"),
            "success": s.get("success"), "verified": s.get("success_verified"), "turns": s.get("turns"),
            "llm_p50": (lat.get("llm_end_s") or {}).get("p50"),
            "llm_p90": (lat.get("llm_end_s") or {}).get("p90"),
            "cycle_p50": (lat.get("cycle_s") or {}).get("p50"),
            "prompt_tok": tot.get("prompt_tokens"), "compl_tok": tot.get("completion_tokens"),
            "cache_share": s.get("cache_read_share"),
            "cache_regr": len(s.get("cache_regressions") or []) if "cache_regressions" in s else None,
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
    rows = report_rows(args.paths, include_fake=args.include_fake)
    print(format_table(rows))
    if args.csv:
        from controlr.runlog import write_csv
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


COMMANDS = {"run": cmd_run, "prompt": cmd_prompt, "bench-cache": cmd_bench_cache,
            "bench-latency": cmd_bench_latency, "sweep": cmd_sweep, "report": cmd_report, "models": cmd_models}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.cmd](args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
