"""Replay logged agentic perceive calls offline (no LLM): the agent's logged answer through today's post-processing
(``j1.finish_answer``: snap, H2, arbitration) and ``record.merge_into_record``, scored against our dev-layout truth.

Inputs: the runtime's breadth run dirs (``<run>/<task>_..._<layout>_report.json.gz`` + ``<run>/agentic/call_*/<ws>/``:
frames, ``calib0.json``, ``b1.json`` = the hint as sent, ``transcript.jsonl`` = the agent's replies, the scan
observation as ``cam_*_scan.png`` + ``calib_scan.json``). The B1 record the answer was merged into is the report's
record before it (agentic0: the scan record, else initial; agentic2: the record before it).

    python -m dojo.perception.agentic.replay --runs RUN_DIR... --layouts ~/dojo-data/simlab/layouts \\
        --out rows.jsonl [--set arb_margin=0.02]

Per truth label (dev truth; identical assets matched by position): the pre-agent error, the logged post-agent error and
the replayed one without (arb0) and with (arb1) the arbitration (arb1_nopin: no higher bar for config-slot objects),
all from the same H2 output. Scoring only: truth never enters the pipeline. DOJO_BG_FOLDS / DOJO_COLOR_CALIB / DOJO_CATALOG as for
the perception service (the renderer the frames came from).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path

TYPES = ("Rigid", "Geometry", "Articulation")
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _yaw(q) -> float:
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def record_by_label(rec: dict) -> dict:
    return {o["label"]: dict(o, _key=f"{t}/{c}/{int(o['category_idx']):05d}") for t in TYPES
            for c, lst in (rec.get(t) or {}).items() for o in lst if o.get("label")}


def _sym(key: str, index: dict) -> int:
    return int((index.get(key) or {}).get("symmetry_z") or 1)


def error(o: dict | None, g: dict, index: dict) -> dict | None:
    if o is None:
        return None
    per = 2 * math.pi / max(1, min(_sym(o["_key"], index), 36))
    dy = (_yaw(o["default_ori"]) - _yaw(g["default_ori"])) % per
    return {"xy_mm": round(1000 * math.dist(o["default_pos"][:2], g["default_pos"][:2]), 1),
            "yaw_deg": round(math.degrees(min(dy, per - dy)), 1), "key_ok": o["_key"] == g["_key"],
            "z_mm": round(1000 * (float(o["default_pos"][2]) - float(g["default_pos"][2])), 1)}


def match(found: dict, truth: dict) -> dict:
    """truth label -> found label; labels of identical assets are interchangeable (matched by position)."""
    by_key: dict = {}
    for lab, g in truth.items():
        by_key.setdefault(g["_key"], []).append(lab)
    out = {}
    for labs in by_key.values():
        if len(labs) == 1:
            if labs[0] in found:
                out[labs[0]] = labs[0]
            continue
        cand = [lab for lab in found if lab in labs]
        k = min(len(cand), len(labs))
        best = None
        for ta in itertools.permutations(labs, k):      # which truth labels, in which order ...
            for fb in itertools.combinations(cand, k):  # ... against which found ones
                c = sum(math.dist(truth[x]["default_pos"][:2], found[y]["default_pos"][:2]) for x, y in zip(ta, fb))
                if best is None or c < best[0]:
                    best = (c, dict(zip(ta, fb)))
        if best:
            out.update(best[1])
    return out


def gross(e: dict | None, xy_mm: float = 15.0, yaw_deg: float = 15.0) -> bool:
    return e is None or e["xy_mm"] > xy_mm or e["yaw_deg"] > yaw_deg


def verdict(a: dict | None, b: dict | None) -> str:
    ga, gb = gross(a), gross(b)
    return "better" if ga and not gb else "worse" if gb and not ga else "both_gross" if ga else "both_ok"


def calls(runs: list[Path]) -> list[dict]:
    out = []
    for run in runs:
        for f in sorted(glob.glob(str(run / "*_report.json.gz"))):
            r = json.load(gzip.open(f))
            recs = r.get("records") or {}
            names = list(recs)
            for i, n in enumerate(names):
                if not n.startswith("agentic"):
                    continue
                ag = (recs[n].get("_perception") or {}).get("agentic") or {}
                ws = ag.get("ws")
                local = run / "agentic" / ws.split("/agentic/")[-1] if ws else None
                pre = ("scan" if "scan" in names else "initial") if n == "agentic0" else names[i - 1]
                out.append({"run": str(run), "report": f, "task": r.get("vocabulary"), "layout": int(Path(f).name.split("_")[-2]),
                            "rec": n, "pre": pre, "ws": str(local) if local and local.is_dir() else None,
                            "result": ag.get("result")})
    return out


def _calib(path: Path) -> tuple[dict, dict]:
    d = json.loads(path.read_text())
    flange = {a: list(e["flange"]["pos"]) + list(e["flange"]["quat"]) for a, e in (d.get("eef") or {}).items()}
    return d["calib"], flange


def load_call(ws: Path) -> dict | None:
    end, first = None, None
    for line in (ws / "transcript.jsonl").read_text().splitlines():
        ev = json.loads(line)
        if ev.get("event") == "start":
            first = ev
        elif ev.get("event") == "end":
            end = ev
    if end is None or not end.get("answer"):
        return None
    text = " ".join(x for x in (first or {}).get("first_user") or [] if isinstance(x, str))
    grip = {a: float(v) for a, v in re.findall(r"(left|right) ([01](?:\.\d+)?)", text.split("Measured")[0])}
    instr = re.search(r"Instruction: (?:np\.str_\()?'(.*?)'\)?\n", text)
    calib, flange = _calib(ws / "calib0.json")
    extra = []
    if (ws / "calib_scan.json").exists() and (ws / "cam_head_scan.png").exists():
        c2, f2 = _calib(ws / "calib_scan.json")
        extra.append({"tag": "scan", "frames": {c: str(ws / f"{c}_scan.png") for c in CAMS if (ws / f"{c}_scan.png").exists()},
                      "calib": c2, "flange": f2})
    return {"answer": end["answer"], "frames": {c: str(ws / f"{c}.png") for c in CAMS if (ws / f"{c}.png").exists()},
            "calib": calib, "flange": flange, "grip": grip, "instruction": instr.group(1) if instr else "",
            "b1": json.loads((ws / "b1.json").read_text()), "extra_obs": extra}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--layouts", default=str(Path.home() / "dojo-data" / "simlab" / "layouts"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[], help="finish_answer cfg override k=v (e.g. arb_margin=0.02)")
    ap.add_argument("--tasks", nargs="*", default=None)
    a = ap.parse_args(argv)

    import cv2
    import numpy as np

    from ..segment import load_backgrounds
    from . import j1
    from .record import merge_into_record
    from ..inverse_layout import fixed_slot_labels
    from .seed_refine import arbitrate, build_views

    bg = Path(os.environ["DOJO_BG_FOLDS"]) / "fold0"
    bgs = load_backgrounds(bg)
    um = bg / "cam_head_clean_unseen.png"
    if um.exists():
        bgs["cam_head_clean_unseen"] = (cv2.imread(str(um), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    index = json.loads((Path(os.environ["DOJO_CATALOG"]) / "index.json").read_text())
    cfg = j1._cfg({"keep_unmentioned": "0"})  # the runtime's record path (agentic_estimate) keeps them out
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = float(v) if k in ("arb_margin", "gate_thr") else v not in ("0", "false", "False")
    rows = []
    with open(a.out, "w") as fo:
        for c in calls([Path(r) for r in a.runs]):
            if a.tasks and c["task"] not in a.tasks:
                continue
            row = {k: c[k] for k in ("run", "task", "layout", "rec", "pre", "result")}
            r = json.load(gzip.open(c["report"]))
            pre_rec = dict(r["records"][c["pre"]], task=c["task"])
            post_logged = r["records"][c["rec"]]
            ld = Path(a.layouts) / c["task"]
            lf = ld / f"{c['layout']}.settled.json"
            lay = json.loads((lf if lf.exists() else ld / f"{c['layout']}.json").read_text())
            truth = record_by_label(lay)
            call = load_call(Path(c["ws"])) if c["ws"] else None
            if call is None:
                row["skip"] = "no workspace / no answer"
                fo.write(json.dumps(row) + "\n")
                continue
            try:
                # H2 once; the arbitration on its output (exactly what finish_answer does with arbitrate=1), so the
                # two variants differ only by the arbitration
                cfg0 = dict(cfg, arbitrate=False)
                objs, _, src, probs, info = j1.finish_answer(
                    call["answer"], call["b1"], call["frames"], call["calib"], call["flange"], call["grip"], cfg0,
                    instruction=call["instruction"], task=c["task"], bgs=bgs, extra_obs=call["extra_obs"], keep_markers=True)
                st, _ = j1._b1._state(call["flange"], call["grip"])
                views = build_views(j1._b1._load_frames(call["frames"]), st, j1._b1._cameras(call["calib"]), bgs,
                                    j1._extra_views(call["extra_obs"], call["grip"]))
                # the hint as the runtime builds it now: config-slot objects carry flags.slot (record_objects); the
                # logged b1.json predates that flag
                slots = fixed_slot_labels(c["task"])
                b1o = [dict(b, flags=dict(b.get("flags") or {}, slot=True)) if b.get("label") in slots else b
                       for b in call["b1"].get("objects") or []]
                objs1, arb = arbitrate(views, objs, b1o, margin=cfg.get("arb_margin", 0.0),
                                       pinned_margin=cfg.get("arb_pinned_margin", 0.15))
                objs2, arb2 = arbitrate(views, objs, b1o, margin=cfg.get("arb_margin", 0.0), pinned_margin=0.0)
                new0, log0 = merge_into_record(pre_rec, objs)
                new1, log1 = merge_into_record(pre_rec, objs1)
                new2, _ = merge_into_record(pre_rec, objs2)
            except Exception as e:  # noqa: BLE001
                row["skip"] = f"{type(e).__name__}: {e}"
                fo.write(json.dumps(row) + "\n")
                continue
            row.update(problems=probs, info=info, arbitration=arb, arbitration_nopin=arb2, merge0=log0, merge1=log1)
            A, L, N0, N1, N2 = (record_by_label(pre_rec), record_by_label(post_logged), record_by_label(new0),
                                record_by_label(new1), record_by_label(new2))
            ma, ml, m0, m1, m2 = match(A, truth), match(L, truth), match(N0, truth), match(N1, truth), match(N2, truth)
            row["labels"] = {}
            for lab, g in truth.items():
                if not any(lab in m for m in (ma, ml, m0, m1, m2)):
                    continue  # never estimated (e.g. a fixture outside the vocabulary)
                e = {name: error(R.get(m.get(lab)), g, index) for name, R, m in
                     (("pre", A, ma), ("logged", L, ml), ("arb0", N0, m0), ("arb1", N1, m1), ("arb1_nopin", N2, m2))}
                row["labels"][lab] = {**e, **{f"{k}_v": verdict(e["pre"], e[k]) for k in ("logged", "arb0", "arb1", "arb1_nopin")}}
            fo.write(json.dumps(row) + "\n")
            fo.flush()
            rows.append(row)
            print(c["task"], c["layout"], c["rec"], {lab: (v["logged_v"], v["arb0_v"], v["arb1_v"]) for lab, v in
                                                   row["labels"].items() if len({v["logged_v"], v["arb0_v"], v["arb1_v"]}) > 1},
                  flush=True)


if __name__ == "__main__":
    sys.exit(main())
