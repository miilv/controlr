"""Do the local Assets hold every object the task configs can ask for?

For every task config: each (type, category, index) in its pools (no ``index`` = every instance
folder present), plus every Clutter entry in ``Clutter/clutter.yml``: USD (``object.usdz`` or
``object.usd``) and ``metadata.json`` present, and ``active.place`` usable by the placement engine.
For ``<task>_random`` it also lists the held-out indices (in the random pool, not in the standard
one). File checks only; ``Assets/Eval_Layout`` is not touched.

    scripts/simlab/rdpy.sh -m dojo.simlab.check_assets [--json out.json]
"""

from __future__ import annotations

import argparse
import json

from dojo.simlab import paths
from dojo.simlab.gen_layouts import available_indices, load_yaml, metadata, usd_exists
from dojo.simlab.layout import OBJECT_TYPES


def pools(task: str) -> dict[tuple[str, str], set[int]]:
    cfg = load_yaml(paths.TASK_CONFIGS / f"{task}.yml")
    out: dict[tuple[str, str], set[int]] = {}
    for t in OBJECT_TYPES:
        for g in cfg.get(t) or []:
            for e in g.get("category") or []:
                idx = e.get("index")
                idx = available_indices(t, e["name"]) if idx is None else idx
                out.setdefault((t, e["name"]), set()).update(int(i) for i in idx)
    return out


def check_one(t: str, cat: str, i: int, clutter: bool = False) -> str | None:
    if not usd_exists(t, cat, i, clutter):
        return "usd missing"
    m = metadata(t, cat, i, clutter)
    if m is None:
        return "metadata missing"
    if not ((m.get("active") or {}).get("place")) and t != "Garment":
        return "no active.place"
    if "oriented_bbox" not in (m.get("geometry") or {}):
        return "no oriented_bbox"
    return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    tasks = sorted(p.stem for p in paths.TASK_CONFIGS.glob("*.yml") if not p.stem.startswith("_"))
    report: dict = {"tasks": {}, "clutter": {}}
    n_ok = n_bad = 0
    for task in tasks:
        pl = pools(task)
        rows = {}
        for (t, cat), idx in sorted(pl.items()):
            bad = {i: r for i in sorted(idx) if (r := check_one(t, cat, i))}
            n_bad += len(bad)
            n_ok += len(idx) - len(bad)
            rows[f"{t}/{cat}"] = {"n": len(idx), "problems": bad}
            if not idx:
                rows[f"{t}/{cat}"]["problems"] = {"-": "no instances on disk"}
                n_bad += 1
        entry = {"pools": rows}
        if task.endswith("_random"):
            std = pools(task[: -len("_random")])
            entry["held_out"] = {f"{t}/{c}": sorted(idx - std.get((t, c), set())) for (t, c), idx in pl.items()
                                 if idx - std.get((t, c), set())}
        report["tasks"][task] = entry
    yml = load_yaml(paths.OBJECTS / "Clutter" / "clutter.yml")["Clutter"]
    c_ok, c_bad = 0, {}
    for cat, idx in yml.items():
        for i in idx:
            r = check_one("Rigid", cat, int(i), clutter=True)
            if r:
                c_bad[f"{cat}/{int(i):05d}"] = r
            else:
                c_ok += 1
    report["clutter"] = {"categories": len(yml), "instances": sum(len(v) for v in yml.values()), "ok": c_ok,
                         "problems": c_bad}
    for task, e in report["tasks"].items():
        probs = {k: v["problems"] for k, v in e["pools"].items() if v["problems"]}
        ho = e.get("held_out")
        print(f"{task:40s} instances={sum(v['n'] for v in e['pools'].values()):4d} problems={probs or '-'}"
              + (f" held_out={ho}" if ho else ""))
    print(f"task-object instances ok={n_ok} bad={n_bad}; clutter {c_ok}/{report['clutter']['instances']} ok, "
          f"problems={c_bad or '-'}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
