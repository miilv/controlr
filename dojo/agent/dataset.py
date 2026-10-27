"""Phase 1 dataset: (instruction, scene) items from every RoboDojo task config on our dev layouts.

    python -m dojo.agent.dataset --layouts ~/dojo-data/simlab/layouts ~/dojo-data/agent/layouts \\
        --per-config 3 --out ~/dojo-data/agent/grounding/dataset.jsonl

Run on a host with RoboDojo's public files (``RD_ROOT``: task sources + asset metadata / captions); CPU only,
no simulator. Per config it takes the first ``--per-config`` layouts (pre-settled records preferred) and

* the instruction from the task's OWN ``gen_instruction`` (its source is exec'd against a stub that answers the
  three label queries those methods make: labels by prefix, an instance's asset index, its category), with
  RoboDojo's ``DescManager`` substitution of ``<label>`` slots by one of the labelled asset's descriptions
  (numpy seeded with the layout id);
* the scene as the runtime presents it (``dojo.agent.scene``: neutral ids, never checker labels);
* the rule-based planner's output (``dojo.skills.planner.plan``) with its labels mapped to neutral ids.

Ground truth labels are kept per item (``labels``: neutral id -> dev label) only for scoring; the agent never
reads them.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dojo.agent.scene import assign_ids, scene_text
from dojo.skills import planner
from dojo.skills.world import AssetDB, WorldState, objects_from_gt
from dojo.simlab.layout import iter_objects

ALL_TYPES = ("Rigid", "Geometry", "Articulation", "Garment", "Fluid")


def rows_from_layout(lay: dict) -> list[dict]:
    rows, n = [], 0
    for otype, cat, inst in iter_objects(lay):
        clutter = inst.get("type") == "cluttered"
        n += 1
        rows.append({"label": inst.get("label") or f"{cat}{n}", "inst_name": inst.get("label"), "type": otype,
                     "category": cat, "category_idx": int(inst["category_idx"]), "clutter": clutter,
                     "pos": np.asarray(inst["default_pos"], float), "quat": np.asarray(inst["default_ori"], float)})
    return rows


class _FuncParserStub:
    """The label queries RoboDojo's gen_instruction methods make, answered from a layout record."""

    def __init__(self, lay: dict):
        self.recs = []
        for t in ALL_TYPES:
            for cat, insts in (lay.get(t) or {}).items():
                for inst in insts:
                    self.recs.append((t, cat, inst))

    def _rec(self, label):
        return next(((t, c, i) for t, c, i in self.recs if i.get("label") == label), None)

    def get_label_by_prefix(self, prefix):
        return [[i["label"] for _, _, i in self.recs if (i.get("label") or "").startswith(prefix)]]

    def get_label_cat_index(self, labels):
        out = []
        for lab in labels:
            r = self._rec(lab)
            out.append([int(r[2]["category_idx"]) if r else None])
        return out

    def get_category_by_label(self, label, env_idx=0):
        r = self._rec(label)
        return r[1] if r else None


def _gen_instruction_fn(task_src: str):
    tree = ast.parse(task_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "gen_instruction":
            ns: dict = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<gen_instruction>", "exec"), ns)  # noqa: S102
            return ns["gen_instruction"]
    return None


def instruction_for(task_src: str, lay: dict, objects_root: Path, seed: int) -> str:
    fn = _gen_instruction_fn(task_src)
    if fn is None:
        return ""
    stub = SimpleNamespace(reward_manager=SimpleNamespace(func_parser=_FuncParserStub(lay)))
    templates = fn(stub, env_idx=0)
    rng = np.random.RandomState(seed)
    t = str(rng.choice(templates))
    for label in re.findall(r"<([^<>]+)>", t):
        rec = stub.reward_manager.func_parser._rec(label)
        if rec is None:
            continue
        typ, cat, inst = rec
        sub = "Clutter" if inst.get("type") == "cluttered" else typ
        try:
            desc = json.loads((objects_root / sub / cat / f"{int(inst['category_idx']):05d}" / "description.json")
                              .read_text()).get("description") or []
        except (OSError, ValueError):
            desc = []
        desc = _flat_strings(desc)
        if desc:
            t = t.replace(f"<{label}>", desc[int(rng.randint(len(desc)))])
    return t


def _flat_strings(v) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        v = list(v.values())
    out = []
    for x in v or []:
        out.extend(_flat_strings(x))
    return out


def step_lim_of(task_src: str) -> int:
    m = re.search(r"step_lim\s*=\s*(\d+)", task_src)
    return int(m.group(1)) if m else 0


def pick_layouts(task: str, roots: list[Path], n: int) -> list[Path]:
    for root in roots:
        d = root / task
        if not d.is_dir():
            continue
        settled = sorted(d.glob("*.settled.json"))
        raw = sorted(p for p in d.glob("*.json") if not p.name.endswith(".settled.json"))
        ids = {p.name.split(".")[0] for p in settled}
        files = settled + [p for p in raw if p.stem not in ids]
        files = sorted(files, key=lambda p: p.name.split(".")[0])
        if files:
            return files[:n]
    return []


def build_item(task: str, lay_path: Path, rd_root: Path, assets: AssetDB, scene_version: int = 2,
               planner_meshes: bool = False) -> dict:
    """``planner_meshes``: the rule planner reads a world with the catalogue meshes loaded, as at runtime (the mesh
    colours are what it pairs nuts and bolts by); the scene text never depends on them."""
    lay = json.loads(lay_path.read_text())
    src = (rd_root / "task" / "RoboDojo" / "tasks" / f"{task}.py").read_text()
    lid = int(lay_path.name.split(".")[0])
    instr = instruction_for(src, lay, assets.root, seed=lid)
    lim = step_lim_of(src)
    objs = objects_from_gt(rows_from_layout(lay), assets, include_clutter=True, meshes=False)
    from dojo.skills.world import ArmState
    arms = {a: ArmState(q=np.zeros(6), flange=np.zeros(7), gripper=1.0) for a in ("left", "right")}
    world = WorldState(task="", instruction=instr, objects=objs, arms=arms, home_q={}, step=0, step_lim=lim)
    text, ids = scene_text(world, objects_root=assets.root, version=scene_version)
    pworld = world
    if planner_meshes:
        pobjs = objects_from_gt(rows_from_layout(lay), assets, include_clutter=True, meshes=True)
        pworld = WorldState(task="", instruction=instr, objects=pobjs, arms=arms, home_q={}, step=0, step_lim=lim)
    try:
        p = planner.plan(instr, pworld)
        plan = {"skill": p.skill, "args": ids.map_args(_jsonable(p.args), to="id"), "why": p.why}
    except Exception as e:  # noqa: BLE001
        plan = {"skill": "error", "args": {}, "why": f"{type(e).__name__}: {e}"}
    return {"id": f"{task}:{lid}", "task": task, "layout_id": lid, "layout_file": lay_path.name,
            "settled": lay_path.name.endswith(".settled.json"), "instruction": instr, "step_lim": lim,
            "scene": text, "labels": ids.to_label, "planner": plan,
            "objects": {oid: {"label": lab, "category": objs[lab].category, "idx": int(objs[lab].idx),
                              "type": objs[lab].obj_type, "clutter": bool(objs[lab].clutter),
                              "center": [round(float(v), 4) for v in objs[lab].center_w]}
                        for oid, lab in ids.to_label.items()}}


def _jsonable(v):
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def main(argv=None) -> None:
    import os
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layouts", nargs="+", required=True, help="layout roots, searched in order per task")
    ap.add_argument("--per-config", type=int, default=3)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene-version", type=int, default=2,
                    help="1: first caption name only (run 1); 2: all names; 3: + frames (first 8); 4: every frame")
    ap.add_argument("--planner-meshes", action="store_true",
                    help="the planner reference on a world with meshes (runtime colours), not the scene's")
    a = ap.parse_args(argv)
    rd_root = Path(os.environ.get("RD_ROOT", str(Path(os.environ.get("RD_HOME", "~/dojo-robodojo")).expanduser() / "RoboDojo")))
    tasks = a.tasks or sorted(p.stem for p in (rd_root / "task" / "RoboDojo" / "config").glob("*.yml")
                              if not p.stem.startswith("_"))
    assets = AssetDB(rd_root / "Assets" / "Object" / "RoboDojo", catalog_root=[])
    roots = [Path(os.path.expanduser(r)) for r in a.layouts]
    out = Path(os.path.expanduser(a.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for task in tasks:
            files = pick_layouts(task, roots, a.per_config)
            if not files:
                print(f"[dataset] {task}: no layouts", flush=True)
                continue
            for lp in files:
                it = build_item(task, lp, rd_root, assets, a.scene_version, a.planner_meshes)
                f.write(json.dumps(it) + "\n")
                n += 1
                print(f"[dataset] {it['id']} {it['planner']['skill']}: {it['instruction'][:90]}", flush=True)
    print(f"[dataset] {n} items -> {out}")


if __name__ == "__main__":
    main()
