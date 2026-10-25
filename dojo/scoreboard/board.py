"""RoboDojo-Sim leaderboard, per task: fetch the site's JS bundle, extract {entry: {task: {score, successRate}}},
recompute the official 5-dimension averages, and rank tasks by points available.

    uv run python -m dojo.scoreboard.board [--bundle FILE] [--md OUT.md] [--ref PyRUA-Lean]

Average = mean over 5 dimensions; Generalization = mean(std half, random half), each half a mean over 12 tasks.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request

SITE = "https://robodojo-benchmark.com"

# RoboDojo scripts/internal/summarize_result.py DIMENSIONS
DIMENSIONS = {
    "Generalization": ["stack_bowls", "push_T", "pack_objects_into_box", "fold_clothes", "hang_mugs",
                       "sweep_blocks", "pour_liquid_into_cup", "make_toast", "arrange_largest_number",
                       "sort_nesting_dolls_by_size", "store_laptop_and_headphones", "stack_blocks"],
    "Precision": ["fasten_screws", "plug_in_charger", "insert_tubes", "pour_balls_into_vase", "play_Xylophone",
                  "deposit_coin", "insert_key", "build_tower"],
    "Long-Horizon": ["put_bottles_into_dustbin", "fill_pen_holder", "classify_objects", "play_tic_tac_toe",
                     "fill_egg_holder", "organize_table", "make_kong", "play_stacking_toy"],
    "Memory": ["cover_blocks", "match_and_pick_from_conveyor", "swap_blocks", "swap_T", "press_by_number",
               "imitate_sorting_sequence"],
    "Open": ["align_blocks", "general_pickup", "stack_blocks_by_language", "solve_equation",
             "classify_objects_by_language", "pick_from_conveyor_by_image", "store_tools_in_toolbox",
             "pour_by_language"],
}


def cells() -> list[tuple[str, str, float]]:
    """(cell, dimension, weight in the Average); a Generalization task is two cells (std, _random)."""
    out = []
    for dim, tasks in DIMENSIONS.items():
        if dim == "Generalization":
            for t in tasks:
                out.append((t, dim, 1 / 5 / 2 / len(tasks)))
                out.append((t + "_random", dim, 1 / 5 / 2 / len(tasks)))
        else:
            out += [(t, dim, 1 / 5 / len(tasks)) for t in tasks]
    return out


def fetch_bundle() -> str:
    html = urllib.request.urlopen(f"{SITE}/leaderboard", timeout=30).read().decode()
    js = sorted(set(re.findall(r"assets/index-[A-Za-z0-9_-]+\.js", html)))
    text = ""
    for path in js:
        text += urllib.request.urlopen(f"{SITE}/{path}", timeout=60).read().decode()
    return text


def _balanced(s: str, i: int) -> int:
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced")


def extract(bundle: str) -> dict[str, dict[str, dict[str, float]]]:
    """Entries whose value is a JSON object of per-task {score, successRate}."""
    entries: dict[str, dict] = {}
    for m in re.finditer(r'"([^"]{1,80})":\{"[a-zA-Z_]+":\{"score"', bundle):
        start = m.end() - len(m.group(0)) + len(m.group(1)) + 3
        end = _balanced(bundle, start)
        try:
            obj = json.loads(bundle[start:end])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and len(obj) >= 20 and all(isinstance(v, dict) and "score" in v for v in obj.values()):
            entries.setdefault(m.group(1), obj)
    return entries


def average(per_task: dict, key: str = "score") -> dict[str, float]:
    def val(t: str) -> float:
        return float(per_task.get(t, {}).get(key, 0.0))

    dims = {}
    for dim, tasks in DIMENSIONS.items():
        if dim == "Generalization":
            std = sum(val(t) for t in tasks) / len(tasks)
            rnd = sum(val(t + "_random") if t + "_random" in per_task else val(t) for t in tasks) / len(tasks)
            dims["Gen-std"], dims["Gen-rand"] = std, rnd
            dims[dim] = (std + rnd) / 2
        else:
            dims[dim] = sum(val(t) for t in tasks) / len(tasks)
    dims["Average"] = sum(dims[d] for d in DIMENSIONS) / len(DIMENSIONS)
    return dims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle")
    ap.add_argument("--md")
    ap.add_argument("--json")
    ap.add_argument("--ref", default="PyRUA-Lean")
    a = ap.parse_args()
    bundle = open(a.bundle).read() if a.bundle else fetch_bundle()
    E = extract(bundle)
    lines = [f"# RoboDojo-Sim per-task board ({len(E)} entries)", "",
             "| Entry | Gen-std | Gen-rand | Prec | Long-H | Memory | Open | **Average** |", "|---|---|---|---|---|---|---|---|"]
    ranked = sorted(E, key=lambda n: -average(E[n])["Average"])
    for n in ranked[:20]:
        d = average(E[n])
        lines.append(f"| {n} | " + " | ".join(f"{d[k]:.1f}" for k in
                     ("Gen-std", "Gen-rand", "Precision", "Long-Horizon", "Memory", "Open")) + f" | **{d['Average']:.2f}** |")
    ref = E.get(a.ref, {})
    rows = []
    for cell, dim, w in cells():
        scores = {n: float(E[n].get(cell, {}).get("score", 0.0)) for n in E}
        best = max(scores, key=scores.get)
        r = float(ref.get(cell, {}).get("score", 0.0))
        rows.append((cell, dim, w, r, scores[best], best, w * (100 - r), w * (scores[best] - r)))
    lines += ["", f"## Cells ranked by points left vs {a.ref} (points = weight × (100 − ref score), in Average units)", "",
              "| Cell | Dim | Weight | Ref | Best | Best entry | Pts to 100 | Pts to best |", "|---|---|---|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda r: -r[6]):
        lines.append(f"| {row[0]} | {row[1]} | {row[2]*100:.2f}% | {row[3]:.1f} | {row[4]:.1f} | {row[5]} | {row[6]:.2f} | {row[7]:.2f} |")
    best_any = sum(r[2] * r[4] for r in rows)
    lines += ["", f"Best-of-any-entry per cell, combined: **{best_any:.2f}**. {a.ref}: **{average(ref)['Average']:.2f}**."]
    out = "\n".join(lines)
    if a.md:
        open(a.md, "w").write(out + "\n")
    if a.json:
        json.dump(E, open(a.json, "w"))
    print(out)


if __name__ == "__main__":
    main()
