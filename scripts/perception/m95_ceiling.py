"""Stream M: the ceiling of a PERFECT per-object selector over every frame-0 perception source we have (PERCEPTION_95.md).

    python scripts/perception/m95_ceiling.py --g1 NAME=DIR[@SET] ... --pmodels CASES_DIR --pm-tags b1,son8_ref_arbR,...
        --pm-lists PMODELS_DIR [--targets EVENTS.jsonl] [--out OUT.json]

Sources:
* ``--g1``: eval_g1-format ``objects.jsonl`` (the SAM bake-off arms, Identity's g1 runs); per object ``ok`` = eval_g1
  ``failure_class(row, TOLERANCE[task]) == "ok"`` (within tolerance, right asset). ``@SET`` names the frame set when
  the rows carry none.
* ``--pmodels``: the perceiver-models cases (``<task>__<id>__f0/{b1,agent_<tag>}.json`` + ``gt.json``) scored with
  ``dev_d0.score_case``; ``ok`` = within tolerance and not a wrong asset. The case's frame set comes from the
  ``only_<set>.txt`` lists (hard19 / d0hard / d0rest -> d0 (6.0 proxy), exthard / ext -> d0_ext, coin, laptop).

Objects are keyed (set, task, episode, label): the same rendered frame and the same GT object. Per frame set: per
source the share of objects ok and of scenes with every object ok; the union (an object is ok if ANY source has it
right); the objects no source gets right, by eval_g1 class (missed / gross / overshoot / identity / rot) and flags
(hidden, occluded, thin). ``--targets``: the grasp-metric events (``m95_score.py score``) name, per task, the labels
the skills grasp or place on; the "grasp-relevant" columns keep only those labels' groups. Ids 100000-09 /
100020-49 only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SLICE = [(100000, 100009), (100020, 100049)]
THIN = {"coin", "pen", "key", "card", "toothbrush", "screwdriver", "spoon", "knife", "fork", "chopsticks", "mallet",
        "brush", "pencil", "marker", "ruler"}


def on_slice(ep) -> bool:
    try:
        i = int(str(ep).split("_")[0])
    except ValueError:
        return False
    return any(a <= i <= b for a, b in SLICE)


def g1_rows(path: Path, set_name: str | None) -> list[dict]:
    from dojo.perception.eval_g1 import TOLERANCE, failure_class
    out = []
    for ln in open(path / "objects.jsonl"):
        r = json.loads(ln)
        if str(r.get("frame", 0)) not in ("0",):
            continue
        tol = TOLERANCE.get(r["task"], (None, None, ""))
        cls = failure_class(r, tol)
        out.append({"set": r.get("set") or set_name, "task": r["task"], "episode": str(r["episode"]), "label": r["label"],
                    "category": r.get("category"), "cls": cls, "ok": cls == "ok", "xy_mm": r.get("xy_err_mm"),
                    "hidden_frac": r.get("hidden_frac"), "occluded": r.get("occluded")})
    return out


def pm_sets(lists_dir: Path) -> dict:
    m = {"hard19": "d0", "d0hard": "d0", "d0rest": "d0", "exthard": "d0_ext", "ext": "d0_ext", "coin": "coin",
         "laptop": "laptop_pm", "main": None, "mainA": None, "mainB": None, "probe": None, "x2": None}
    out = {}
    for f in sorted(lists_dir.glob("only_*.txt")):
        s = m.get(f.stem[len("only_"):])
        if s is None:
            continue
        for n in f.read_text().split():
            out.setdefault(n, s)
    return out


def pm_rows(cases: Path, tag: str, sets: dict) -> list[dict]:
    from dojo.perception.agentic.dev_d0 import score_case
    out = []
    for d in sorted(cases.iterdir()):
        if not d.is_dir() or d.name not in sets:
            continue
        for r in score_case(d, tag):
            if not r["found"]:
                cls = "missed"
            elif r["gross"]:
                cls = "gross"
            elif not r["within_tol"]:
                cls = "overshoot"
            elif r.get("id_ok") is False:
                cls = "identity"
            else:
                cls = "ok"
            ep = d.name.split("__")[1]
            out.append({"set": sets[d.name], "task": r["task"], "episode": ep, "label": r["label"],
                        "category": r.get("category"), "cls": cls, "ok": cls == "ok", "xy_mm": r.get("xy_mm")})
    return out


def grasp_labels(events: list[Path]) -> dict:
    """task -> {labels} the skills grasped or placed on (targets, held objects, references) in the scored episodes."""
    out = defaultdict(set)
    for p in events:
        for ln in open(p):
            e = json.loads(ln)
            for r in e.get("required") or []:
                for k in ("label", "gt_label"):
                    if r.get(k):
                        out[e["task"]].add(r[k])
    return out


def ceiling(by_src: dict[str, list[dict]], relevant: dict | None = None) -> dict:
    """Per set: per source (objects ok, scenes all ok), the union, and what no source fixes."""
    keyed = defaultdict(dict)                         # (set, task, ep, label) -> {src: row}
    for src, rows in by_src.items():
        for r in rows:
            if not on_slice(r["episode"]):
                continue
            keyed[(r["set"], r["task"], r["episode"], r["label"])][src] = r
    out = {}
    sets = sorted({k[0] for k in keyed})
    for s in sets:
        keys = [k for k in keyed if k[0] == s]
        allsrc = sorted({src for k in keys for src in keyed[k]})
        cov = {src: sum(src in keyed[k] for k in keys) / len(keys) for src in allsrc}
        srcs = [src for src in allsrc if cov[src] >= 0.9] or allsrc     # the columns: sources covering the set
        part = [src for src in allsrc if src not in srcs]                 # the rest join the union where they scored
        full = [k for k in keys if all(src in keyed[k] for src in srcs)]
        res = {"objects": len(full), "scenes": len({k[1:3] for k in full}), "partial": {p: round(cov[p], 2) for p in part}}
        for variant, flt in (("all", lambda k: True),
                             ("grasp", lambda k: relevant is None or k[3] in relevant.get(k[1], ()) or not relevant.get(k[1]))):
            ks = [k for k in full if flt(k)]
            scenes = defaultdict(list)
            for k in ks:
                scenes[k[1:3]].append(k)
            row = {}
            for src in srcs:
                row[src] = {"obj_ok": sum(keyed[k][src]["ok"] for k in ks),
                            "scene_ok": sum(all(keyed[k][src]["ok"] for k in v) for v in scenes.values())}
            any_ok = {k: any(keyed[k][x]["ok"] for x in srcs) for k in ks}
            any_all = {k: any(r["ok"] for r in keyed[k].values()) for k in ks}
            row["UNION"] = {"obj_ok": sum(any_ok.values()), "scene_ok": sum(all(any_ok[k] for k in v) for v in scenes.values())}
            if part:
                row["UNION+partial"] = {"obj_ok": sum(any_all.values()),
                                        "scene_ok": sum(all(any_all[k] for k in v) for v in scenes.values())}
            res[variant] = {"objects": len(ks), "scenes": len(scenes), "by_source": row}
            per_task = {}
            for t in sorted({k[1] for k in ks}):
                sc = {kk: v for kk, v in scenes.items() if kk[0] == t}
                obj = [k for k in ks if k[1] == t]
                best = max(srcs, key=lambda x: sum(all(keyed[k][x]["ok"] for k in v) for v in sc.values()))
                per_task[t] = {"scenes": len(sc), "objects": len(obj),
                               "b1_scene_ok": sum(all(keyed[k][srcs[0]]["ok"] for k in v) for v in sc.values()),
                               "best": best, "best_scene_ok": sum(all(keyed[k][best]["ok"] for k in v) for v in sc.values()),
                               "union_scene_ok": sum(all(any_all[k] for k in v) for v in sc.values()),
                               "union_obj_ok": sum(any_all[k] for k in obj)}
            res[variant]["per_task"] = per_task
            if variant == "all":
                never = [k for k in ks if not any_all[k]]
                cls, task, flags = Counter(), Counter(), Counter()
                for k in never:
                    best = Counter(keyed[k][x]["cls"] for x in srcs).most_common(1)[0][0]
                    cls[best] += 1
                    task[k[1]] += 1
                    r0 = next((keyed[k][x] for x in srcs if keyed[k][x].get("hidden_frac") is not None), None)
                    if r0 and (r0.get("hidden_frac") or 0) >= 0.5:
                        flags["hidden>=50%"] += 1
                    if r0 and r0.get("occluded"):
                        flags["occluded"] += 1
                    if any(t in str(keyed[k][srcs[0]].get("category")) for t in THIN):
                        flags["thin"] += 1
                res["never"] = {"n": len(never), "class": dict(cls), "task": dict(task.most_common()), "flags": dict(flags),
                                "examples": [f"{k[1]} {k[2]} {k[3]} ({keyed[k][srcs[0]]['cls']} "
                                             f"{keyed[k][srcs[0]].get('xy_mm')})" for k in never[:40]],
                                "objects": [{"task": k[1], "episode": k[2], "label": k[3],
                                             "category": keyed[k][srcs[0]].get("category"),
                                             "cls": {x: keyed[k][x]["cls"] for x in keyed[k]},
                                             "xy_mm": {x: keyed[k][x].get("xy_mm") for x in keyed[k]},
                                             "hidden_frac": next((keyed[k][x].get("hidden_frac") for x in keyed[k]
                                                                  if keyed[k][x].get("hidden_frac") is not None), None),
                                             "occluded": any(keyed[k][x].get("occluded") for x in keyed[k])} for k in never]}
        out[s] = res
    return out


def srcs_name(a: dict) -> str:
    return next(iter(a["by_source"]))


def to_markdown(res: dict) -> str:
    lines = []
    for s, r in res.items():
        lines.append(f"\n**{s}**: {r['scenes']} scenes / {r['objects']} objects scored by every source")
        lines.append("| source | objects ok | scenes all ok | grasp-relevant objects ok | grasp-relevant scenes all ok |")
        lines.append("|---|---|---|---|---|")
        a, g = r["all"], r["grasp"]
        for src in a["by_source"]:
            x, y = a["by_source"][src], g["by_source"][src]
            pct = lambda n, d: f"{100 * n / d:.0f} % ({n})" if d else "-"   # noqa: E731
            name = {"UNION": "**UNION** (perfect selector)", "UNION+partial": f"**UNION + partial sources** {r['partial']}"}.get(src, src)
            lines.append(f"| {name} | {pct(x['obj_ok'], a['objects'])} | {pct(x['scene_ok'], a['scenes'])} | "
                         f"{pct(y['obj_ok'], g['objects'])} | {pct(y['scene_ok'], g['scenes'])} |")
        lines.append(f"| task | scenes | objects | {srcs_name(a)} scenes all ok | best single source | perfect selector: "
                     "scenes all ok | objects ok |")
        lines.append("|---|---|---|---|---|---|---|")
        for t, d in a["per_task"].items():
            lines.append(f"| {t} | {d['scenes']} | {d['objects']} | {d['b1_scene_ok']} | {d['best']} {d['best_scene_ok']} | "
                         f"{d['union_scene_ok']} | {d['union_obj_ok']} |")
        nv = r["never"]
        lines.append(f"no source right: {nv['n']} objects; class {nv['class']}; flags {nv['flags']}; by task {nv['task']}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", action="append", default=[], help="NAME=DIR[@SET]")
    ap.add_argument("--pmodels", default=None)
    ap.add_argument("--pm-tags", default="b1")
    ap.add_argument("--pm-lists", default=None)
    ap.add_argument("--targets", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--combine", default=None, help="SET,SET...: also one combined set over these (the bake-off's 222 "
                    "scenes: d0_51,d0_ext,coin,laptop)")
    ap.add_argument("--identity-ok", default="", help="TASK,...: count a within-tolerance wrong asset as ok there "
                    "(make_kong: tile faces no source reads)")
    a = ap.parse_args(argv)
    by_src = {}
    for spec in a.g1:
        name, rest = spec.split("=", 1)
        path, _, set_name = rest.partition("@")
        by_src[name] = g1_rows(Path(path), set_name or None)
        print(f"{name}: {len(by_src[name])} rows", file=sys.stderr)
    if a.pmodels:
        sets = pm_sets(Path(a.pm_lists or Path(a.pmodels).parent))
        for tag in a.pm_tags.split(","):
            by_src[f"pm_{tag}"] = pm_rows(Path(a.pmodels), tag, sets)
            print(f"pm_{tag}: {len(by_src[f'pm_{tag}'])} rows", file=sys.stderr)
    loose = set(filter(None, a.identity_ok.split(",")))
    for rows in by_src.values():
        for r in rows:
            if r["task"] in loose and r["cls"] == "identity":
                r["cls"], r["ok"] = "ok", True
    if a.combine:
        keep = set(a.combine.split(","))
        for name in list(by_src):
            extra = [{**r, "set": "+".join(sorted(keep))} for r in by_src[name] if r["set"] in keep]
            by_src[name] = by_src[name] + extra
    rel = grasp_labels([Path(p) for p in a.targets]) if a.targets else None
    res = ceiling(by_src, rel)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(to_markdown(res))


if __name__ == "__main__":
    main()
