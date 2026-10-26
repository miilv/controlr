"""DEV evaluation of ``held.held_pose`` on rendered held-object frames (GT read only here, for priors and scoring).

    python -m dojo.perception.held_eval variants --root ~/dojo-data/perc_held/cap [--tasks ...]
    python -m dojo.perception.held_eval run --root ROOT --out OUT.jsonl [--tasks ...] [--n-priors 3] [--workers 4]
        [--views all|own|own+head|head+observer] [--frames C,V]   # frame tag prefixes
    python -m dojo.perception.held_eval report OUT.jsonl [...]

Frames: ``<root>/<task>/<seed>/frame_<tag>.json`` with a ``held`` block (``held_data`` captures / variants) or a
``held_gt`` block (perception's earlier synthetic stack-block frames, ``heldsyn_synth``), rendered by the Isaac 6.0
proxy (``meta.renderer`` set).

Priors (mode B's grasp-time belief error, our model of it): the GT in-hand transform moved by a turn about the
closing axis (TCP y) of 1-5 deg, turns about x / z of N(0, 1 deg), a slide along x / z of 1-4 mm and N(0, 0.5 mm)
along y; ``--prior plan`` also uses the skills' planned transform (G3's belief = GT, so it is close to GT).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from .held_data import R_rotvec, inv_T, lead_axis_from_target, pose7_to_T, pose_errors, rotvec_R, tcp_from_flange_T

VIEW_SETS = {"all": (), "own": ("cam_head", "OBSERVER"), "own+head": ("OBSERVER",), "head+observer": ("OWN",),
             "own+observer": ("cam_head",)}


def load_frame(path: Path) -> dict | None:
    """Images + state + held info + GT in-hand transform + scoring geometry of one rendered frame."""
    import cv2

    from .robot_model import X5Model
    from .render import load_asset

    rec = json.loads(Path(path).read_text())
    if not (rec.get("meta") or {}).get("renderer"):
        return None
    st = rec["obs"]["state"]
    if "held" in rec:
        h = rec["held"]
        arm, label, key = h["arm"], h["label"], h["key"]
        row = next(r for r in rec["gt"]["objects"] if r.get("label") == label)
        T_obj = pose7_to_T(list(row["pos"]) + list(row["quat"]))
        place = h.get("place")
    elif rec.get("held_gt") and rec["held_gt"].get("held"):
        hg = rec["held_gt"]
        arm, label, key = hg["arm"], hg["held"]["label"], hg["held"]["key"]
        T_obj = pose7_to_T(hg["held"]["pose7"])
        place = None
    else:
        return None
    model = X5Model.default()
    q = st[f"{arm}_arm_joint_state"]
    T_tcp = tcp_from_flange_T(model.fk_world(arm, q)["link6"])
    T_gt = inv_T(T_tcp) @ T_obj
    a = load_asset(key)
    if place and place.get("T_obj_target") is not None:
        lead, axis = lead_axis_from_target(a.V, place["T_obj_target"])
        geo = "target"
    else:
        from .held import _lead_axis
        lead, axis = _lead_axis(a, T_gt, {})
        geo = "default"
    d = Path(path).parent
    images = {}
    for c, f in rec["obs"]["images"].items():
        img = cv2.imread(str(d / f), cv2.IMREAD_COLOR)
        if img is None:
            return None
        images[c] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    state = {k: np.asarray(v, float) for k, v in st.items()}
    meta = rec.get("meta") or {}
    var = meta.get("variant") or {}
    return {"path": str(path), "tag": rec.get("frame"), "task": Path(path).parent.parent.name, "seed": Path(path).parent.name,
            "frame": {"images": images, "state": state}, "arm": arm, "label": label, "key": key, "T_gt": T_gt,
            "T_plan": np.asarray(rec["held"]["T_tcp_obj_plan"]) if "held" in rec and rec["held"].get("T_tcp_obj_plan") else None,
            "lead": lead, "axis": axis, "geo": geo, "kind": var.get("kind", "hover"), "variant": var,
            "plan_costs": (rec.get("held") or {}).get("plan_costs")}


def perturb(T_gt, rng) -> tuple[np.ndarray, dict]:
    ry = math.radians(rng.uniform(1.0, 5.0)) * rng.choice([-1, 1])
    rx, rz = (math.radians(v) for v in rng.normal(0, 1.0, 2))
    t = np.array([rng.uniform(1.0, 4.0) * rng.choice([-1, 1]), rng.normal(0, 0.5), rng.uniform(1.0, 4.0) * rng.choice([-1, 1])])
    t *= rng.uniform(0.3, 1.0, 3) * np.array([1, 1, 1])
    D = np.eye(4)
    D[:3, :3] = rotvec_R([rx, ry, rz])
    c = T_gt[:3, 3]
    D[:3, 3] = c - D[:3, :3] @ c + t / 1000
    return D @ T_gt, {"rot_deg": np.degrees([rx, ry, rz]).tolist(), "t_mm": t.tolist()}


def _exclude(view_set: str, arm: str) -> tuple:
    ex = VIEW_SETS[view_set]
    other = "right" if arm == "left" else "left"
    out = []
    for e in ex:
        out.append({"OBSERVER": f"cam_{other}_wrist", "OWN": f"cam_{arm}_wrist"}.get(e, e))
    return tuple(out)


def eval_frame(path: str, n_priors: int, view_sets: list[str], seed: int, prior_kinds=("perturb",)) -> list[dict]:
    from .held import held_pose

    F = load_frame(Path(path))
    if F is None:
        return []
    rng = np.random.default_rng(seed + zlib.crc32(F["path"].encode()) % 100000)
    priors = []
    if "perturb" in prior_kinds:
        for i in range(n_priors):
            Tp, info = perturb(F["T_gt"], rng)
            priors.append(("perturb", i, Tp, info))
    if "plan" in prior_kinds and F["T_plan"] is not None:
        priors.append(("plan", 0, F["T_plan"], {}))
    if "gt" in prior_kinds:
        priors.append(("gt", 0, F["T_gt"], {}))
    rows = []
    for vs in view_sets:
        for kind, i, Tp, info in priors:
            t0 = time.time()
            try:
                r = held_pose(F["frame"], F["arm"], F["key"], Tp,
                              {"lead_obj": F["lead"], "axis_obj": F["axis"], "exclude_cams": _exclude(vs, F["arm"])})
            except Exception as e:  # noqa: BLE001
                import traceback
                rows.append({"path": F["path"], "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
                continue
            e_prior = pose_errors(Tp, F["T_gt"], F["lead"], F["axis"])
            e_fit = pose_errors(r.get("fit_T_tcp_obj", r["T_tcp_obj"]), F["T_gt"], F["lead"], F["axis"])
            e_out = pose_errors(r["T_tcp_obj"], F["T_gt"], F["lead"], F["axis"])
            rows.append({"path": F["path"], "task": F["task"], "seed": F["seed"], "tag": F["tag"], "kind": F["kind"],
                         "variant": F["variant"], "key": F["key"], "arm": F["arm"], "geo": F["geo"], "views_set": vs,
                         "prior": kind, "prior_i": i, "prior_info": info, "ok": r["ok"], "why": r["why"],
                         "views": r.get("views"), "view_diag": r.get("view_diag"), "iou": r.get("iou"),
                         "sigma": r.get("sigma"), "err_prior": e_prior, "err_fit": e_fit, "err_out": e_out,
                         "residual": r.get("residual"), "t_s": round(time.time() - t0, 2)})
    return rows


def _paths(root: Path, tasks, frames) -> list[str]:
    out = []
    for t in sorted(p for p in root.iterdir() if p.is_dir()):
        if tasks and t.name not in tasks:
            continue
        for d in sorted(p for p in t.iterdir() if p.is_dir()):
            for f in sorted(d.glob("frame_*.json")):
                tag = f.stem[len("frame_"):]
                if frames and not any(tag.startswith(p) for p in frames):
                    continue
                out.append(str(f))
    return out


def _worker(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    path, n, vs, seed, pk = args
    return eval_frame(path, n, vs, seed, pk)


def cmd_variants(a) -> None:
    from .held_data import make_variants

    root = Path(a.root)
    for f in _paths(root, a.tasks, ["C"]):
        if not f.endswith(".json") or "frame_C" not in f:
            continue
        rec = json.loads(Path(f).read_text())
        if "held" not in rec:
            continue
        k = rec["frame"][1:]
        if list(Path(f).parent.glob(f"frame_V{k}[gos]*.json")) and not a.force:
            continue
        w = make_variants(Path(f), n_grasp=a.n_grasp, n_looks=a.n_looks, n_shows=a.n_shows)
        print(f, "->", len(w), flush=True)


def cmd_run(a) -> None:
    root = Path(a.root)
    paths = _paths(root, a.tasks, a.frames)
    if a.limit:
        paths = paths[: a.limit]
    vs = a.views.split(",")
    pk = tuple(a.prior.split(","))
    jobs = [(p, a.n_priors, vs, a.seed, pk) for p in paths]
    out = open(a.out, "a")
    n = 0
    if a.workers > 1:
        from multiprocessing import get_context

        with get_context("spawn").Pool(a.workers) as pool:
            for rows in pool.imap_unordered(_worker, jobs):
                for r in rows:
                    out.write(json.dumps(r) + "\n")
                out.flush()
                n += 1
                print(f"[held_eval] {n}/{len(jobs)}", flush=True)
    else:
        for j in jobs:
            for r in _worker(j):
                out.write(json.dumps(r) + "\n")
                print(json.dumps({k: r.get(k) for k in ("tag", "views_set", "prior", "ok", "why")}),
                      {k: round(r["err_out"][k], 2) for k in ("lead_norm_mm", "axis_deg", "rot_norm_deg")} if "err_out" in r else r.get("error"), flush=True)
            out.flush()


def _q(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def summarize(rows: list[dict], key=lambda r: (r["task"], r["kind"], r["views_set"])) -> list[dict]:
    groups: dict = {}
    for r in rows:
        if "err_out" not in r:
            continue
        groups.setdefault(key(r), []).append(r)
    out = []
    for k, rs in sorted(groups.items()):
        ok = [r for r in rs if r["ok"]]
        lead = np.array([r["err_fit"]["lead_norm_mm"] for r in rs])
        ax = np.array([r["err_fit"]["axis_deg"] for r in rs])
        lead_ok = np.array([r["err_out"]["lead_norm_mm"] for r in ok])
        ax_ok = np.array([r["err_out"]["axis_deg"] for r in ok])
        pri = np.array([r["err_prior"]["lead_norm_mm"] for r in rs])
        hit = [r for r in ok if r["err_out"]["lead_norm_mm"] <= 0.5 and r["err_out"]["axis_deg"] <= 1.0]
        out.append({"group": k, "n": len(rs), "ok": len(ok), "prior_lead_med": _q(pri, 50),
                    "fit_lead_med": _q(lead, 50), "fit_lead_p90": _q(lead, 90), "fit_axis_med": _q(ax, 50),
                    "fit_axis_p90": _q(ax, 90), "ok_lead_med": _q(lead_ok, 50), "ok_lead_p90": _q(lead_ok, 90),
                    "ok_lead_max": float(lead_ok.max()) if len(lead_ok) else float("nan"),
                    "ok_axis_med": _q(ax_ok, 50), "ok_axis_p90": _q(ax_ok, 90),
                    "within_0.5mm_1deg": len(hit), "t_med_s": _q([r["t_s"] for r in rs], 50)})
    return out


def coverage(rows: list[dict]) -> dict:
    """Calibration of the reported sigma on ok fits: share of |error| <= 2 sigma per lead axis and for the axis angle,
    and the scale k that would make 95 % of max_i(|e_i| / sigma_i) <= 2 k."""
    ok = [r for r in rows if r.get("ok") and r.get("sigma") and "err_out" in r]
    if not ok:
        return {}
    e = np.abs(np.array([r["err_out"]["lead_mm"] for r in ok]))
    s = np.maximum(np.array([r["sigma"]["lead_mm"] for r in ok]), 1e-6)
    ea = np.array([r["err_out"]["axis_deg"] for r in ok])
    sa = np.maximum(np.array([r["sigma"]["axis_deg"] for r in ok]), 1e-6)
    z = np.max(e / s, axis=1)
    return {"n": len(ok), "lead_within_2s": float(np.mean(np.all(e <= 2 * s, axis=1))),
            "axis_within_2s": float(np.mean(ea <= 2 * sa)), "k_lead_95": float(np.percentile(z, 95) / 2),
            "k_axis_95": float(np.percentile(ea / sa, 95) / 2)}


def cmd_report(a) -> None:
    rows = []
    for f in a.files:
        rows += [json.loads(l) for l in open(f) if l.strip()]
    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"{len(errs)} errors, first: {errs[0]['error']}")
    keyf = {"task": lambda r: (r["task"], r["kind"], r["views_set"], r["prior"]),
            "views": lambda r: (r["views_set"], r["kind"]),
            "kind": lambda r: (r["kind"],)}[a.by]
    S = summarize(rows, keyf)
    cols = ["n", "ok", "prior_lead_med", "fit_lead_med", "fit_lead_p90", "fit_axis_med", "fit_axis_p90", "ok_lead_med",
            "ok_lead_p90", "ok_lead_max", "ok_axis_med", "ok_axis_p90", "within_0.5mm_1deg", "t_med_s"]
    print("| group | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for s in S:
        print("| " + " / ".join(map(str, s["group"])) + " | " + " | ".join(
            (f"{s[c]:.2f}" if isinstance(s[c], float) else str(s[c])) for c in cols) + " |")
    print("sigma coverage (ok fits):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in coverage(rows).items()})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("variants")
    v.add_argument("--root", required=True)
    v.add_argument("--tasks", nargs="*", default=None)
    v.add_argument("--n-grasp", type=int, default=2)
    v.add_argument("--n-looks", type=int, default=2)
    v.add_argument("--n-shows", type=int, default=1)
    v.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--tasks", nargs="*", default=None)
    r.add_argument("--frames", nargs="*", default=None)
    r.add_argument("--n-priors", type=int, default=2)
    r.add_argument("--views", default="all")
    r.add_argument("--prior", default="perturb")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    p.add_argument("--by", default="task")
    a = ap.parse_args(argv)
    {"variants": cmd_variants, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
