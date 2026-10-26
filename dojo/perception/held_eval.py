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

from .held_data import R_rotvec, inv_T, lead_axis_from_target, pose7_to_T, pose_errors, rotvec_R, tcp_from_flange_T  # noqa: F401

AUTO_VIEWS = {"hover": ["own", "all"], "grasp": ["own", "all"], "observer_look": ["all"], "look_ranked": ["all"],
              "show": ["own+head", "all"], "show_ranked": ["own+head", "all"]}
VIEW_SETS = {"all": (), "own": ("cam_head", "OBSERVER"), "own+head": ("OBSERVER",), "head+observer": ("OWN",),
             "own+observer": ("cam_head",)}


_R51_INDEX: dict = {}


def _r51_record(path: Path) -> dict | None:
    """Bridge-isr's genuine Isaac 5.1 held_look frames (``<root>/frames/<task>/<layout>_s<step>/state.json`` + the
    three PNGs; ``<root>/held_index.json`` holds the true object poses): a record in the captures' format. The held
    object's catalogue key comes from our dev layout (labels -> category / idx)."""
    root = path.parents[3]
    if root not in _R51_INDEX:
        idx = json.loads((root / "held_index.json").read_text())
        _R51_INDEX[root] = {i["dir"].split("/all/")[1]: i for i in idx}
    e = _R51_INDEX[root].get(f"{path.parent.parent.name}/{path.parent.name}")
    if e is None:
        return None
    lay_d = Path(os.environ.get("HELD_LAYOUTS", os.path.expanduser("~/dojo-data/simlab/layouts"))) / e["task"]
    lp = lay_d / f"{e['layout']}.settled.json"
    lay = json.loads((lp if lp.exists() else lay_d / f"{e['layout']}.json").read_text())
    key = None
    for typ in ("Rigid",):
        for cat, recs in (lay.get(typ) or {}).items():
            for r in recs:
                if r.get("label") == e["physical_object"]:
                    key = f"{typ}/{cat}/{int(r['category_idx']):05d}"
    if key is None:
        return None
    arm = e["check"]["arm"]
    st = json.loads(path.read_text())
    return {"frame": path.parent.name, "obs": {"state": st, "images": {c: f"{c}.png" for c in ("cam_head", "cam_left_wrist",
                                                                                               "cam_right_wrist")}},
            "meta": {"renderer": "isaac-5.1 (RoboDojo native, Bridge-isr replay)"},
            "held_gt": {"arm": arm, "held": {"label": e["physical_object"], "key": key,
                                             "pose7": e["objects_true"][e["physical_object"]]}}}


def load_frame(path: Path) -> dict | None:
    """Images + state + held info + GT in-hand transform + scoring geometry of one rendered frame."""
    import cv2

    from .robot_model import X5Model
    from .render import load_asset

    path = Path(path)
    rec = _r51_record(path) if path.name == "state.json" else json.loads(path.read_text())
    if rec is None:
        return None
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
            "renderer": meta.get("renderer"),
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


def align_spin(T_est, T_gt, axis_obj_sym) -> np.ndarray:
    """``T_est`` turned about the object's own symmetry axis (object frame, through the bbox-centre line the mesh is
    symmetric about) to the turn nearest ``T_gt``: a symmetric object is the same object at every turn, so only the
    rest of the pose is an error (the skills take the nearest symmetric target, ``Skills._nearest_spin``)."""
    from .render import load_asset  # noqa: F401
    T_est, T_gt = np.asarray(T_est, float), np.asarray(T_gt, float)
    a = np.asarray(axis_obj_sym, float)
    def score(th):
        return float(np.trace((T_est[:3, :3] @ rotvec_R(a * th)).T @ T_gt[:3, :3]))
    th = max(np.radians(np.arange(0.0, 360.0, 0.5)), key=score)
    return max(th + np.radians(np.arange(-0.5, 0.5, 0.01)), key=score)


def _spin_aligned(T_est, T_gt, key):
    from .held import spin_axis
    from .render import load_asset

    sp = spin_axis(key)
    if sp is None:
        return np.asarray(T_est, float)
    a = load_asset(key)
    c = (np.asarray(a.info["bbox_min"]) + np.asarray(a.info["bbox_max"])) / 2
    th = align_spin(T_est, T_gt, sp)
    S = np.eye(4)
    S[:3, :3] = rotvec_R(sp * th)
    S[:3, 3] = c - S[:3, :3] @ c                    # about the symmetry line through the bbox centre
    return np.asarray(T_est, float) @ S


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
    if view_sets == ["auto"]:                   # per frame kind: what each option can use
        view_sets = AUTO_VIEWS.get(F["kind"], ["own", "all"])
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
            T_fit = np.asarray(r.get("fit_T_tcp_obj", r["T_tcp_obj"]), float)
            e_prior = pose_errors(_spin_aligned(Tp, F["T_gt"], F["key"]), F["T_gt"], F["lead"], F["axis"])
            e_fit = pose_errors(_spin_aligned(T_fit, F["T_gt"], F["key"]), F["T_gt"], F["lead"], F["axis"])
            e_out = pose_errors(_spin_aligned(r["T_tcp_obj"], F["T_gt"], F["key"]), F["T_gt"], F["lead"], F["axis"])
            rows.append({"path": F["path"], "task": F["task"], "seed": F["seed"], "tag": F["tag"], "kind": F["kind"],
                         "variant": F["variant"], "key": F["key"], "arm": F["arm"], "geo": F["geo"], "views_set": vs,
                         "prior": kind, "prior_i": i, "prior_info": info, "ok": r["ok"], "why": r["why"],
                         "views": r.get("views"), "view_diag": r.get("view_diag"), "iou": r.get("iou"),
                         "sigma": r.get("sigma"), "err_prior": e_prior, "err_fit": e_fit, "err_out": e_out,
                         "T_fit": T_fit.tolist(), "T_gt": np.asarray(F["T_gt"]).tolist(),
                         "residual": r.get("residual"), "t_s": round(time.time() - t0, 2)})
    return rows


def _paths(root: Path, tasks, frames) -> list[str]:
    if (root / "held_index.json").exists():                    # Bridge-isr's 5.1 replay frames
        return sorted(str(p) for p in (root / "frames").glob("*/*/state.json")
                      if not tasks or p.parent.parent.name in tasks)
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
        if list(Path(f).parent.glob(f"frame_V{k}[{a.kinds}]*.json")) and not a.force:
            continue
        w = make_variants(Path(f), n_grasp=a.n_grasp, n_looks=a.n_looks, n_shows=a.n_shows, kinds=a.kinds)
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


def step_cost(r: dict) -> int:
    """Env steps the option costs beyond the hover: observer look = its move (parallel to the approach when the skill
    schedules it so); show pose = out (the observer in parallel) + back."""
    v = r.get("variant") or {}
    k = r.get("kind")
    if k in ("observer_look", "look_ranked", "show_ranked"):
        return int(v.get("steps") or 0)
    if k == "show":
        return int(max(v.get("steps_out") or 0, v.get("look_steps") or 0) + (v.get("steps_back") or 0))
    return 0


_CLASS: dict = {}


def geometry_class(key: str) -> str:
    if key not in _CLASS:
        from .held import spin_axis
        _CLASS[key] = "spin" if spin_axis(key) is not None else "rigid"
    return _CLASS[key]


def rescale(rows: list[dict], scales: dict) -> list[dict]:
    """Rows as if the estimator had used ``scales`` {class: {"pos", "rot"}} (the rows' sigma are raw, scale 1): sigma
    scaled, ok re-decided with held's bars (fit_poor / out_of_range stay not ok), err_out = the fit when ok else the
    prior."""
    from . import held as H

    out = []
    for r in rows:
        if "err_fit" not in r or not r.get("sigma"):
            out.append(r)
            continue
        k = scales[geometry_class(r["key"])]
        s = dict(r["sigma"])
        for f in ("pos_mm", "lead_mm"):
            s[f] = [v * k["pos"] for v in s[f]]
        for f in ("lead_lat_mm", "lead_along_mm"):
            if f in s:
                s[f] = s[f] * k["pos"]
        s["rot_deg"] = [v * k["rot"] for v in s["rot_deg"]]
        s["axis_deg"] = s["axis_deg"] * k["rot"]
        ok = r["why"] in ("", "ambiguous") and s.get("lead_lat_mm", 1e9) <= H.OK_MAX_LEAD_SIGMA_MM \
            and s["axis_deg"] <= H.OK_MAX_AXIS_SIGMA_DEG
        out.append({**r, "sigma": s, "ok": ok, "why": "" if ok else (r["why"] or "ambiguous"),
                    "err_out": r["err_fit"] if ok else r["err_prior"]})
    return out


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
        hit = [r for r in ok if r["err_out"]["lead_lat_mm"] <= 0.5 and r["err_out"]["axis_deg"] <= 1.0]
        hit_all = [r for r in rs if r["err_fit"]["lead_lat_mm"] <= 0.5 and r["err_fit"]["axis_deg"] <= 1.0]
        lat = np.array([r["err_fit"]["lead_lat_mm"] for r in rs])
        lat_ok = np.array([r["err_out"]["lead_lat_mm"] for r in ok])
        along = np.array([abs(r["err_fit"]["lead_along_mm"]) for r in rs])
        ry = np.array([abs(r["err_fit"]["rot_deg"][1]) for r in rs])
        ry_ok = np.array([abs(r["err_out"]["rot_deg"][1]) for r in ok])
        out.append({"group": k, "n": len(rs), "ok": len(ok), "steps_med": _q([step_cost(r) for r in rs], 50),
                    "prior_lead_med": _q(pri, 50),
                    "fit_lead_med": _q(lead, 50), "fit_lead_p90": _q(lead, 90), "fit_axis_med": _q(ax, 50),
                    "fit_axis_p90": _q(ax, 90), "ok_lead_med": _q(lead_ok, 50), "ok_lead_p90": _q(lead_ok, 90),
                    "ok_lead_max": float(lead_ok.max()) if len(lead_ok) else float("nan"),
                    "ok_axis_med": _q(ax_ok, 50), "ok_axis_p90": _q(ax_ok, 90),
                    "fit_ry_med": _q(ry, 50), "fit_ry_p90": _q(ry, 90), "ok_ry_p90": _q(ry_ok, 90),
                    "fit_lat_med": _q(lat, 50), "fit_lat_p90": _q(lat, 90), "fit_along_med": _q(along, 50),
                    "ok_lat_med": _q(lat_ok, 50), "ok_lat_p90": _q(lat_ok, 90),
                    "ok_lat_max": float(lat_ok.max()) if len(lat_ok) else float("nan"),
                    "hit_all": len(hit_all), "within_0.5mm_1deg": len(hit), "t_med_s": _q([r["t_s"] for r in rs], 50)})
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


def calibrate(rows: list[dict], q: float = 95.0) -> dict:
    """SIGMA_SCALE factors from fits that pass every gate but the sigma one (why in "", "ambiguous"), 2-fold by layout:
    the factor k (on the reported sigma) that puts q % of max_i |e_i| / (k sigma_i) <= 2, fitted on one half and
    checked on the other."""
    use = [r for r in rows if "err_fit" in r and r.get("sigma") and r["why"] in ("", "ambiguous")]

    def k_of(rs):
        e = np.abs(np.array([r["err_fit"]["lead_mm"] for r in rs]))
        s = np.maximum(np.array([r["sigma"]["lead_mm"] for r in rs]), 1e-6)
        ea = np.array([r["err_fit"]["axis_deg"] for r in rs])
        sa = np.maximum(np.array([r["sigma"]["axis_deg"] for r in rs]), 1e-6)
        return float(np.percentile(np.max(e / s, 1), q) / 2), float(np.percentile(ea / sa, q) / 2)

    def k_lat(rs):
        rs = [r for r in rs if "lead_lat_mm" in r["sigma"]]
        if not rs:
            return None
        e = np.array([r["err_fit"]["lead_lat_mm"] for r in rs])
        s = np.maximum(np.array([r["sigma"]["lead_lat_mm"] for r in rs]), 1e-6)
        return round(float(np.percentile(e / s, q) / 2), 2)

    def cover(rs, kl, ka):
        e = np.abs(np.array([r["err_fit"]["lead_mm"] for r in rs]))
        s = np.array([r["sigma"]["lead_mm"] for r in rs]) * kl
        ea = np.array([r["err_fit"]["axis_deg"] for r in rs])
        sa = np.array([r["sigma"]["axis_deg"] for r in rs]) * ka
        return float(np.mean(np.all(e <= 2 * s, 1))), float(np.mean(ea <= 2 * sa))

    out = {"n": len(use)}
    if len(use) < 20:
        return out
    halves = [[r for r in use if int(str(r["seed"]).split("_")[0][-1]) % 2 == h] for h in (0, 1)]
    for h in (0, 1):
        kl, ka = k_of(halves[h])
        out[f"fit_on_{h}"] = {"k_lead": round(kl, 2), "k_axis": round(ka, 2),
                              "held_out_cover_lead_axis": [round(v, 3) for v in cover(halves[1 - h], kl, ka)]}
    kl, ka = k_of(use)
    out["all"] = {"k_lead": round(kl, 2), "k_axis": round(ka, 2), "k_lead_lat": k_lat(use)}
    return out


def calibrate_classes(rows: list[dict], q: float = 95.0) -> dict:
    """Per geometry class: the lateral-lead and axis factors (fit on all), and their 2-fold held-out coverage."""
    out = {}
    for cls in ("rigid", "spin"):
        use = [r for r in rows if "err_fit" in r and r.get("sigma") and r["why"] in ("", "ambiguous")
               and geometry_class(r["key"]) == cls and "lead_lat_mm" in r["sigma"]]
        if len(use) < 20:
            continue

        def ks(rs):
            el = np.array([r["err_fit"]["lead_lat_mm"] for r in rs])
            sl = np.maximum(np.array([r["sigma"]["lead_lat_mm"] for r in rs]), 1e-6)
            ea = np.array([r["err_fit"]["axis_deg"] for r in rs])
            sa = np.maximum(np.array([r["sigma"]["axis_deg"] for r in rs]), 1e-6)
            return float(np.percentile(el / sl, q) / 2), float(np.percentile(ea / sa, q) / 2)

        def cover(rs, kp, kr):
            el = np.array([r["err_fit"]["lead_lat_mm"] for r in rs])
            sl = np.array([r["sigma"]["lead_lat_mm"] for r in rs]) * kp
            ea = np.array([r["err_fit"]["axis_deg"] for r in rs])
            sa = np.array([r["sigma"]["axis_deg"] for r in rs]) * kr
            return [round(float(np.mean(el <= 2 * sl)), 3), round(float(np.mean(ea <= 2 * sa)), 3)]

        halves = [[r for r in use if int(str(r["seed"]).split("_")[0][-1]) % 2 == h] for h in (0, 1)]
        kp, kr = ks(use)
        out[cls] = {"n": len(use), "pos": round(kp, 2), "rot": round(kr, 2),
                    "held_out": [cover(halves[1 - h], *ks(halves[h])) for h in (0, 1) if len(halves[h]) >= 10]}
    return out


def cmd_report(a) -> None:
    rows = []
    for f in a.files:
        rows += [json.loads(l) for l in open(f) if l.strip()]
    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"{len(errs)} errors, first: {errs[0]['error']}")
    keyf = {"task": lambda r: (r["task"], r["kind"], r["views_set"], r["prior"]),
            "task_kind": lambda r: (r["task"], r["kind"], r["views_set"]),
            "views": lambda r: (r["views_set"], r["kind"]),
            "kind": lambda r: (r["kind"],)}[a.by]
    if a.scales:
        sc = json.loads(a.scales)
        rows = rescale(rows, {c: sc.get(c, {"pos": 1.0, "rot": 1.0}) for c in ("rigid", "spin")})
    S = summarize(rows, keyf)
    cols = {"short": ["n", "steps_med", "prior_lead_med", "fit_lat_med", "fit_lat_p90", "fit_along_med", "fit_axis_med",
                      "fit_axis_p90", "hit_all", "ok", "ok_lat_med", "ok_lat_p90", "ok_lat_max", "ok_axis_med", "ok_axis_p90",
                      "within_0.5mm_1deg"],
            "long": ["n", "ok", "steps_med", "prior_lead_med", "fit_lead_med", "fit_lead_p90", "fit_axis_med", "fit_axis_p90",
                     "fit_ry_med", "fit_ry_p90", "ok_lead_med", "ok_lead_p90", "ok_lead_max", "ok_axis_med", "ok_axis_p90",
                     "ok_ry_p90", "within_0.5mm_1deg", "t_med_s"]}[a.cols]
    print("| group | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for s in S:
        print("| " + " / ".join(map(str, s["group"])) + " | " + " | ".join(
            (f"{s[c]:.2f}" if isinstance(s[c], float) else str(s[c])) for c in cols) + " |")
    print("sigma coverage (ok fits):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in coverage(rows).items()})
    print("sigma calibration (2-fold by layout):", calibrate([r for r in rows if "err_fit" in r]))
    if a.classes:
        print("per-class calibration (lateral lead, axis):", calibrate_classes(rows))


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
    v.add_argument("--kinds", default="gosrt", help="g grasp, o naive look, s naive show, r ranked look, t ranked show")
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
    p.add_argument("--cols", default="short")
    p.add_argument("--scales", default=None, help='json {"rigid": {"pos": k, "rot": k}, "spin": {...}} applied to raw sigma')
    p.add_argument("--classes", action="store_true", help="per geometry class calibration (needs the catalogue)")
    a = ap.parse_args(argv)
    {"variants": cmd_variants, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
