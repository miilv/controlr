"""Offline evaluation of the FK agent on recorded demos (no arm).

For every held-out session and every query view (a key point on the way to the grasp:
``approach`` ~150 mm, ``pre_grasp`` ~60 mm from it), the agent gets that pose's joints and wrist
photo as its only observation and the task "pick up the <object>", and answers as in a live step.
Ground truth = the demo's grasp TCP (FK of the offset-corrected joints at jaw contact).

Conditions (few-shot sessions shown):
* ``zero``      — none;
* ``loo``       — every other session (other cycles of the same recording included: same scene,
                  so the other objects' positions may hint);
* ``loo_rec``   — only sessions from OTHER recordings (strict leave-one-recording-out).

Metrics per query: object estimate error (3D and horizontal), FK(q target) - grasp TCP (3D,
horizontal, height), |expected_tcp - FK(q)| (the model's own kinematics), the action, and the
baseline "stay at the query pose" (query TCP - grasp TCP).
"""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from controlr.fkagent import geometry as G
from controlr.fkagent import prompt as Pm
from controlr.fkagent.agent import ask

CONDITIONS = ("zero", "loo", "loo_rec")
OFFLINE_LIMITS = dict(box_hi_mm=(320.0, 150.0, 260.0))   # the demos grasped up to x ~300 mm


def queries(sessions: list[dict], views=("approach", "pre_grasp")) -> list[dict]:
    out = []
    for s in sessions:
        for kp in s["keypoints"]:
            if kp["name"] in views and kp.get("photo"):
                out.append({"session": s, "view": kp})
    return out


def fewshot_for(cond: str, held: dict, sessions: list[dict]) -> list[dict]:
    if cond == "zero":
        return []
    if cond == "loo":
        return [s for s in sessions if s["session_id"] != held["session_id"]]
    if cond == "loo_rec":
        return [s for s in sessions if s["recording"] != held["recording"]]
    raise ValueError(f"unknown condition {cond!r}")


def build_query(q: dict, cond: str, sessions: list[dict], root, *, lim: G.Limits, max_steps: int = 6,
                long_edge: int = 640) -> tuple[str, list]:
    s, kp = q["session"], q["view"]
    shots = fewshot_for(cond, s, sessions)
    parts = Pm.fewshot_parts(shots, root, long_edge=long_edge) if shots else []
    step = Pm.Step(0, "observe", kp["q_deg"], kp.get("gripper"), (Path(root) / kp["photo"]).read_bytes(),
                   where="the arm's current pose (the arm moves from here)")
    parts += Pm.history_parts(f"pick up the {s['object']}", [step], long_edge=long_edge, max_steps=max_steps)
    return Pm.system_text(lim, max_steps=max_steps), parts


def score(q: dict, d: Pm.Decision | None) -> dict:
    s, kp = q["session"], q["view"]
    gt = np.asarray(s["grasp"]["tcp_mm"], float)
    view_tcp = np.asarray(kp["tcp_mm"], float)
    out = {"session": s["session_id"], "object": s["object"], "view": kp["name"],
           "gt_tcp_mm": gt.round(1).tolist(), "view_tcp_mm": view_tcp.round(1).tolist(),
           "baseline_stay_mm": round(float(np.linalg.norm(view_tcp - gt)), 1),
           "baseline_stay_xy_mm": round(float(np.linalg.norm(view_tcp[:2] - gt[:2])), 1)}
    if d is None:
        out["ok"] = False
        return out
    out.update({"ok": True, "action": d.action, "visible": d.visible, "uncertainty_mm": d.uncertainty_mm,
                "q_deg": d.q_deg, "object_xyz_mm": d.object_xyz_mm, "gripper_open_pct": d.gripper_open_pct})
    if d.object_xyz_mm is not None:
        e = np.asarray(d.object_xyz_mm) - gt
        out["est_err_mm"] = round(float(np.linalg.norm(e)), 1)
        out["est_err_xy_mm"] = round(float(np.linalg.norm(e[:2])), 1)
    if d.q_deg is not None:
        p = G.tcp_mm(d.q_deg)
        e = p - gt
        out["target_tcp_mm"] = p.round(1).tolist()
        out["target_pitch_deg"] = round(G.tool_pitch_deg(d.q_deg), 1)
        out["target_err_mm"] = round(float(np.linalg.norm(e)), 1)
        out["target_err_xy_mm"] = round(float(np.linalg.norm(e[:2])), 1)
        out["target_dz_mm"] = round(float(e[2]), 1)
        if d.grasp_q_deg is not None:
            gp = G.tcp_mm(d.grasp_q_deg)
            ge = gp - gt
            out["grasp_tcp_mm"] = gp.round(1).tolist()
            out["grasp_err_mm"] = round(float(np.linalg.norm(ge)), 1)
            out["grasp_err_xy_mm"] = round(float(np.linalg.norm(ge[:2])), 1)
            out["grasp_dz_mm"] = round(float(ge[2]), 1)
            out["grasp_pitch_deg"] = round(G.tool_pitch_deg(d.grasp_q_deg), 1)
        if d.expected_tcp_mm is not None:
            out["expected_vs_fk_mm"] = round(float(np.linalg.norm(np.asarray(d.expected_tcp_mm) - p)), 1)
    return out


def run(sessions: list[dict], root, out_dir, llm, *, model: str, conditions=CONDITIONS, views=("approach", "pre_grasp"),
        max_tokens: int = 16000, workers: int = 4, long_edge: int = 640, only: list[str] | None = None) -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lim = G.Limits(**OFFLINE_LIMITS)
    qs = [q for q in queries(sessions, views) if not only or q["session"]["session_id"] in only]
    jobs = [(q, c) for q in qs for c in conditions]

    def one(job):
        q, c = job
        system, parts = build_query(q, c, sessions, root, lim=lim, long_edge=long_edge)
        t0 = time.monotonic()
        d, calls, text = ask(llm, model, system, parts, max_tokens=max_tokens, retries=1)
        r = score(q, d)
        r.update({"condition": c, "n_fewshot": len(fewshot_for(c, q["session"], sessions)),
                  "wall_s": round(time.monotonic() - t0, 1), "llm": calls})
        return r

    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for r in ex.map(one, jobs):
            results.append(r)
            with open(out / "results.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{r['condition']:8s} {r['session']:24s} {r['view']:10s} est {r.get('est_err_mm', '-'):>6} "
                  f"target {r.get('target_err_mm', '-'):>6} (xy {r.get('target_err_xy_mm', '-')}, dz "
                  f"{r.get('target_dz_mm', '-')}) grasp {r.get('grasp_err_mm', '-')} action {r.get('action', '-')} "
                  f"stay {r['baseline_stay_mm']}",
                  flush=True)
    return results


def _med(xs):
    xs = [x for x in xs if x is not None]
    return (statistics.median(xs), len(xs)) if xs else (None, 0)


def report(results: list[dict]) -> str:
    keys = [("est_err_mm", "object est 3D"), ("est_err_xy_mm", "object est xy"), ("target_err_mm", "target TCP 3D"),
            ("target_err_xy_mm", "target TCP xy"), ("grasp_err_mm", "proposed grasp TCP 3D"),
            ("grasp_err_xy_mm", "proposed grasp TCP xy"), ("expected_vs_fk_mm", "own FK |expected-FK|"),
            ("baseline_stay_mm", "baseline: stay")]
    conds = [c for c in CONDITIONS if any(r["condition"] == c for r in results)]
    lines = ["| metric (median mm, n) | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for k, name in keys:
        row = []
        for c in conds:
            m, n = _med([r.get(k) for r in results if r["condition"] == c])
            row.append("-" if m is None else f"{m:.0f} (n={n})")
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    row = []
    for c in conds:
        rs = [r for r in results if r["condition"] == c]
        row.append(f"{sum(1 for r in rs if r.get('action') == 'grip')}/{len(rs)}")
    lines.append("| action = grip | " + " | ".join(row) + " |")
    row = []
    for c in conds:
        rs = [r for r in results if r["condition"] == c and r.get("target_err_mm") is not None]
        row.append(f"{sum(1 for r in rs if r['target_err_mm'] <= 30)}/{len(rs)}")
    lines.append("| target within 30 mm | " + " | ".join(row) + " |")
    row = []
    for c in conds:
        rs = [r for r in results if r["condition"] == c and r.get("grasp_err_mm") is not None]
        row.append(f"{sum(1 for r in rs if r['grasp_err_mm'] <= 30)}/{len(rs)}")
    lines.append("| proposed grasp within 30 mm | " + " | ".join(row) + " |")
    lines.append("")
    lines.append("| session | view | " + " | ".join(f"{c} est / grasp" for c in conds) + " | stay |")
    lines.append("|---|---|" + "---|" * (len(conds) + 1))
    seen = []
    for r in results:
        key = (r["session"], r["view"])
        if key in seen:
            continue
        seen.append(key)
        cells = []
        for c in conds:
            rr = next((x for x in results if (x["session"], x["view"], x["condition"]) == (*key, c)), None)
            cells.append("-" if rr is None else f"{rr.get('est_err_mm', '-')} / {rr.get('grasp_err_mm', '-')}")
        lines.append(f"| {r['session']} ({r['object']}) | {r['view']} | " + " | ".join(cells)
                     + f" | {r['baseline_stay_mm']} |")
    return "\n".join(lines)
