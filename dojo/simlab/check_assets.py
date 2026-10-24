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
import subprocess
from pathlib import Path

import yaml

from dojo.simlab import paths
from dojo.simlab import gen_layouts as G
from dojo.simlab.layout import OBJECT_TYPES

load_yaml = G.load_yaml
available_indices, metadata, usd_exists = G.available_indices, G.metadata, G.usd_exists


class GitTree:
    """The asset set as published (HF dataset git tree, e.g. a pointer-only clone), instead of the
    files on disk: lets a sparse install answer "is it in the public Assets?"."""

    def __init__(self, repo: Path):
        self.repo = repo
        out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD", "--",
                              "Assets/Object/RoboDojo"], capture_output=True, text=True, check=True).stdout
        self.files = set(out.split())
        if any("Eval_Layout" in f for f in self.files):
            raise RuntimeError("tree listing must not include eval data")

    def _dir(self, obj_type, category, clutter):
        return f"Assets/Object/RoboDojo/{'Clutter' if clutter else obj_type}/{category}"

    def available_indices(self, obj_type, category, clutter=False):
        pre = self._dir(obj_type, category, clutter) + "/"
        return sorted({int(f[len(pre):].split("/")[0]) for f in self.files
                       if f.startswith(pre) and f[len(pre):].split("/")[0].isdigit()})

    def usd_exists(self, obj_type, category, idx, clutter=False):
        d = f"{self._dir(obj_type, category, clutter)}/{idx:05d}"
        return f"{d}/object.usdz" in self.files or f"{d}/object.usd" in self.files

    def metadata(self, obj_type, category, idx, clutter=False):
        f = f"{self._dir(obj_type, category, clutter)}/{idx:05d}/metadata.json"
        if f not in self.files:
            return None
        txt = subprocess.run(["git", "-C", str(self.repo), "show", f"HEAD:{f}"], capture_output=True,
                             text=True).stdout
        if txt.startswith("version https://git-lfs"):
            return {"_lfs_pointer": True, "active": {"place": {"?": 1}}, "geometry": {"oriented_bbox": 1}}
        return json.loads(txt)

    def clutter_yml(self):
        txt = subprocess.run(["git", "-C", str(self.repo), "show", "HEAD:Assets/Object/RoboDojo/Clutter/clutter.yml"],
                             capture_output=True, text=True, check=True).stdout
        d = yaml.safe_load(txt)
        if not isinstance(d, dict):          # an LFS pointer: use the checked-out copy
            d = load_yaml(self.repo / "Assets/Object/RoboDojo/Clutter/clutter.yml")
        return d


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
    ap.add_argument("--git-tree", default=None, help="check against this HF git clone's tree, not the disk")
    a = ap.parse_args(argv)
    global available_indices, metadata, usd_exists
    tree = None
    if a.git_tree:
        tree = GitTree(Path(a.git_tree))
        available_indices, metadata, usd_exists = tree.available_indices, tree.metadata, tree.usd_exists
        G.available_indices = tree.available_indices          # pools() resolves index-less entries here
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
    yml = (tree.clutter_yml() if tree else load_yaml(paths.OBJECTS / "Clutter" / "clutter.yml"))["Clutter"]
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
