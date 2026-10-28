"""Stream C, X2: the write re-runner. A recorded episode's belief writes (pick / place / held refines, post-place
re-measures) re-run offline on its stored per-step frames (``seq_capture``), with the runtime's own inputs: the
recorded prior (or any other, e.g. an A1 arm's pose: ``inputs(..., prior7=...)``), the other believed objects at that
step and the object in the jaws (``RecordedBelief``: the runtime's ``dev_belief_probe`` snapshots, taken at every
gripper change, carried to the step with the logged writes in between), and the runtime config's search radius /
empty hand / identity candidates. The perception call is the service's own (``_Perception.refine``, in process); no
LLM anywhere.

Fidelity mode (this script's main): every recorded write re-run from its recorded prior on the stored frame of its step,
diffed against the logged result. The replay's pixels are not the recorded run's (RTX differs run to run), so the
diffs are the render-noise floor of a re-computed write, per writer: ok agreement, position / rotation of the adopted
pose (both ok), fit shift (not ok), p50 / p95 / max.

    PYTHONPATH=<the recorded run's code dir> PYOPENGL_PLATFORM=egl <its venv python> \\
        scripts/perception/bgperc/write_rerun.py SEQ_ROOT --out OUT.json [--valid VALID.json] [--names a,b]

Run it with the recorded runs' code (their perception) and the same GL backend (EGL on the recording box's GPU), and
the integ_eval environment (DOJO_CATALOG, DOJO_BG_FOLDS, DOJO_COLOR_CALIB, DOJO_CALIB_SET): ``write_rerun.sh``.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

WRITERS = ("refine_pick", "refine_place", "refine_held", "post_place")


def _seq_data():
    """``dojo.perception.seq_data`` from this checkout (the recorded runs' code may predate it)."""
    p = Path(__file__).resolve().parents[3] / "dojo" / "perception" / "seq_data.py"
    spec = importlib.util.spec_from_file_location("c_seq_data", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_src(seq) -> tuple[dict, list]:
    d = os.path.dirname(seq.meta["actions"])
    rep = glob.glob(os.path.join(d, f"{seq.task}_*_{seq.layout_id}_report.json.gz"))
    report = json.load(gzip.open(rep[0])) if rep else {}
    acts = json.load(gzip.open(seq.meta["actions"], "rt"))
    return report, acts["actions"] if isinstance(acts, dict) else acts


def writer(ev: dict) -> str | None:
    if ev.get("kind") == "refine" and ev.get("for") in ("pick", "place", "held") and ev.get("label"):
        return f"refine_{ev['for']}"
    if ev.get("kind") == "post_place" and ev.get("label"):
        return "post_place"
    return None


def quat_angle_deg(q1, q2) -> float:
    q1, q2 = np.asarray(q1, float), np.asarray(q2, float)
    d = abs(float(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))


def _obj(lab, o, pos=None, quat=None) -> dict:
    return {"label": lab, "key": o["key"], "pos": [float(v) for v in (o["pos"] if pos is None else pos)],
            "quat": [float(v) for v in (o["quat"] if quat is None else quat)]}


def _movable(snap: dict) -> list[str]:
    return [lab for lab, o in snap.items() if o.get("type") != "Geometry" and not str(o.get("key", "")).startswith("Geometry/")]


def target_candidates(report: dict, label: str, vocabulary: str) -> list[str] | None:
    """The runtime's identity candidates of a pick refine (``policy._refine_hook``): the planning record's
    instruction-ranked top-k, else the label's own config group."""
    from dojo.runtime import policy as P
    from dojo.perception.grasp_metric import PLANNING_ORDER
    recs = report.get("records") or {}
    rec = next((recs[k] for k in PLANNING_ORDER if recs.get(k)), None) or {}
    rank = ((rec.get("_perception") or {}).get("target_ranking") or {}).get(label)
    if rank:
        return [k for _, k in rank[:P.IDENTITY_TOPK]]
    return P.label_candidates(vocabulary, label)


def adopted(e: dict):
    """The pose7 a logged write put into the runtime's belief, None when it changed nothing (M's ``_applied`` rule)."""
    from dojo.perception.grasp_metric import _applied
    k = e.get("kind")
    if k == "refine" and e.get("for") in ("pick", "place", "held") and _applied(e):
        p7 = [float(v) for v in e["pose7"]]
        if e.get("for") != "held" and "orientation kept" in str(e.get("why") or "") and e.get("prior7"):
            p7[3:7] = [float(v) for v in e["prior7"][3:7]]
        return p7
    if k == "post_place" and e.get("ok") and e.get("pose7") is not None:
        return [float(v) for v in e["pose7"]]
    return None


class RecordedBelief:
    """The runtime's belief right before its i-th logged event, from its ``dev_belief_probe`` (a snapshot before
    every gripper change: objects, attached per arm, the joints the belief last stepped). Between two snapshots P
    (step < s) and N (step >= s) the jaws do not change, so: an attached object rides the commanded TCP (from P, from
    its last adopted in-hand refine, or from N when the close was P's change); a free object is at its last adopted
    write since P, else at the prior its next pick / place refine logged before N (no adopted write before that
    refine), else at N's pose when nothing wrote it up to N and either there is no P or P's change released it, else
    where the jaws let it go (released), else at P's, else (no P) the planning record + the writes (BeliefTracker).
    Belief changes no event logs (a divergence's re-estimate, a scan) are not seen."""

    def __init__(self, report: dict, actions: list[dict], revs: list[dict]):
        from dojo.perception.grasp_metric import BeliefTracker, _R_quat, pose_T, probe_belief
        from dojo.skills.kinematics import X5Kin
        self.kin = X5Kin()
        self.tracker = BeliefTracker(report, actions, self.kin)
        self.track = self.tracker.track
        self.snaps = sorted(((report.get("dev_belief_probe") or {}).get("snaps") or []), key=lambda x: x["step"])
        self.revs = revs
        self._pb, self._pose_T, self._R_quat = probe_belief, pose_T, _R_quat

    def _T(self, arm, step):
        tr = self.track[arm]
        return np.asarray(tr[min(max(int(step), 0), len(tr) - 1)], float)

    def before(self, i: int, step: int | None = None) -> tuple[dict, dict, str]:
        """(belief {label: {key, pos, quat}}, attached {arm: label}, source) right before ``revs[i]`` (events
        ``[0, i)`` applied), after ``step`` actions (default: the event's own step; ``i == len(revs)``: all events)."""
        s = int(self.revs[i]["step"]) if step is None else int(step)
        P = next((x for x in reversed(self.snaps) if x["step"] < s), None)
        N = next((x for x in self.snaps if x["step"] >= s), None)
        if P is None and N is None:
            return self.tracker_snapshot(i, s), {}, "rebuilt"
        p_step = P["step"] if P else -1
        att = dict((N or P).get("attached") or {})
        base = P or N
        bP = self._pb(P, self.track, s, self.kin) if P else {}
        bN = self._pb(N, self.track, s, self.kin) if N else {}
        writes = [(j, e) for j, e in enumerate(self.revs) if e.get("label") and adopted(e) is not None]
        out = {}
        for lab, o0 in base["objects"].items():
            arm = next((a for a, x in att.items() if x == lab), None)
            last = [(j, e) for j, e in writes if e["label"] == lab and p_step < int(e["step"]) <= s and j < i]
            if arm is not None:
                held = [(j, e) for j, e in last if e.get("for") == "held"]
                if held:
                    j, e = held[-1]
                    p7 = adopted(e)
                    T = self._T(arm, s) @ np.linalg.inv(self._T(arm, int(e["step"]))) @ self._pose_T(p7[:3], p7[3:7])
                    o = {**o0, "pos": T[:3, 3].tolist(), "quat": self._R_quat(T[:3, :3]).tolist()}
                elif P and (P.get("attached") or {}).get(arm) == lab:
                    o = bP[lab]
                else:
                    o = bN.get(lab, o0)
            elif last:
                p7 = adopted(last[-1][1])
                o = {**o0, "pos": p7[:3], "quat": p7[3:7]}
            else:
                o = self._free(lab, o0, i, s, P, N, writes)
            out[lab] = {"key": o.get("key", o0.get("key")), "pos": [float(v) for v in o["pos"]],
                        "quat": [float(v) for v in o["quat"]]}
        return out, att, "probe"

    def _free(self, lab, o0, i, s, P, N, writes):
        """A free object with no adopted write between P and this event (see the class docstring)."""
        hi = N["step"] if N else 10 ** 9
        later = [(j, e) for j, e in writes if j >= i and e["label"] == lab and int(e["step"]) <= hi]
        nxt = next(((j, e) for j, e in enumerate(self.revs) if j >= i and e.get("label") == lab
                    and e.get("kind") == "refine" and e.get("for") in ("pick", "place") and e.get("prior7")
                    and int(e["step"]) <= hi), None)
        if nxt is not None and not any(j < nxt[0] for j, _e in later):
            return {**o0, "pos": nxt[1]["prior7"][:3], "quat": nxt[1]["prior7"][3:7]}   # its next refine's prior
        released = P is not None and lab in (P.get("attached") or {}).values()
        if (P is None or released) and N is not None and lab in N["objects"] and not later:
            return N["objects"][lab]                    # unchanged up to N: N's pose (where the runtime put it)
        if released and any(e.get("kind") in ("grip_check", "held_look") and e.get("dropped") == "air"
                            and e.get("label") == lab and int(e["step"]) <= s for e in self.revs[:i]):
            # empty jaws: the runtime put it back where it was before the grasp (BeliefTracker's "held_dropped")
            pre = next((x for x in reversed(self.snaps) if x["step"] < P["step"]
                        and lab not in (x.get("attached") or {}).values()), None)
            if pre is not None and lab in pre["objects"]:
                return pre["objects"][lab]
        if released:
            return self._pb(P, self.track, P["step"] + 1, self.kin).get(lab, o0)   # where the jaws let it go
        if P is not None:
            return P["objects"].get(lab, o0)
        return self.tracker_snapshot(i, s).get(lab, o0)  # before the first gripper change: record + writes

    def tracker_snapshot(self, i: int, step: int) -> dict:
        tr = self.tracker
        stop = self.revs[i] if i < len(self.revs) else None
        while tr._i < len(tr.revs) and tr.revs[tr._i] is not stop:
            tr._apply(tr.revs[tr._i])
            tr._i += 1
        return tr.snapshot(int(step))


def inputs(ev: dict, snap: dict, attached: dict, cfg: dict, report: dict, prior7=None) -> dict:
    """The perception ``refine`` request the runtime sent for this write, with ``prior7`` (default: the recorded one)."""
    w, lab, arm = writer(ev), ev["label"], ev["arm"]
    if lab not in snap:
        raise KeyError(f"{lab} not in the belief")
    if prior7 is None:
        if w == "post_place":
            o = snap[lab]
            pos = ev.get("from") or o["pos"]                     # the released pose (logged when the re-measure moved it)
            prior7 = list(pos)[:3] + list(o["quat"])
        else:
            prior7 = ev["prior7"]
    held_lab = attached.get(arm)
    target = _obj(lab, snap[lab], prior7[:3], prior7[3:7])
    req = {"arm": arm, "max_shift_m": float(cfg.get("refine_max_shift_m", 0.03)), "held": None, "kw": {},
           "identity_vocabulary": None}
    if w in ("refine_pick", "refine_place"):
        others = [x for x in _movable(snap) if x not in (lab, held_lab)]
        if w == "refine_place" and held_lab in snap:
            h = snap[held_lab]
            req["held"] = (h["key"], [float(v) for v in list(h["pos"]) + list(h["quat"])])
        sr = float(cfg.get("refine_search_radius") or 0)
        if sr > 0:
            req["kw"]["search_radius"] = sr
            req["max_shift_m"] = max(req["max_shift_m"], sr + 0.01)
        if float(cfg.get("grip_miss_search_m") or 0) > sr and w == "refine_pick":
            raise NotImplementedError("grip_miss_search_m > refine_search_radius: the grip-miss radius is not replayed")
        if cfg.get("refine_empty_hand") and w == "refine_pick" and not attached:
            req["kw"]["empty_hand"] = True
        if cfg.get("refine_identity") and w == "refine_pick":
            req["identity_vocabulary"] = target_candidates(report, lab, report.get("vocabulary") or "")
    elif w == "refine_held":
        others = [x for x in _movable(snap) if x != lab]
    else:                                                       # post_place: the checker's refine
        others = [x for x in _movable(snap) if x != lab]
        for _a, hl in attached.items():
            if hl in snap:
                h = snap[hl]
                req["held"] = (h["key"], [float(v) for v in list(h["pos"]) + list(h["quat"])])
    req["objects"] = [target] + [_obj(x, snap[x]) for x in others]
    return req


def rerun(P, frame: dict, req: dict) -> dict:
    from dojo.runtime.perception_service import encode_frame
    out = P.refine(encode_frame(frame), req["objects"], 0, req["max_shift_m"], arm=req["arm"],
                   identity_vocabulary=req["identity_vocabulary"], held=req["held"], **req["kw"])
    return out.get(req["objects"][0]["label"]) or {}


def rot_deg(key: str | None, q_rep, q_rec) -> float:
    """The rotation between two fits modulo the object's symmetry: M's scorer's rule (``grasp_metric._rot_err``:
    ``eval_g1``'s symmetry of the placement), the raw angle without the catalogue."""
    from dojo.perception.grasp_metric import _rot_err
    d = _rot_err(key, q_rep, q_rec) if key else None
    return float(d) if d is not None else quat_angle_deg(q_rep, q_rec)


def compare(ev: dict, r: dict, key: str | None = None) -> dict:
    ok_a, ok_b = bool(ev.get("ok")), bool(r.get("ok"))
    out = {"ok_rec": ok_a, "ok_rep": ok_b, "iou_rec": ev.get("iou"), "iou_rep": r.get("iou"),
           "shift_rec": ev.get("shift_mm"), "shift_rep": r.get("shift_mm")}
    pa = ev.get("pose7")
    if ok_a and ok_b and pa is not None and r.get("pos") is not None:
        pb = list(r["pos"]) + list(r["quat"])
        out["d_mm"] = round(float(np.linalg.norm(np.subtract(pa[:3], pb[:3]))) * 1000, 2)
        out["d_xy_mm"] = round(float(np.linalg.norm(np.subtract(pa[:2], pb[:2]))) * 1000, 2)
        out["d_deg_raw"] = round(quat_angle_deg(pa[3:7], pb[3:7]), 2)
        out["d_deg"] = round(rot_deg(key, pb[3:7], pa[3:7]), 2)
        out["pose7_rep"] = [round(float(v), 5) for v in pb]
    elif ok_a and ok_b and ev.get("shift_mm") is not None and r.get("shift_mm") is not None:
        # no logged pose (a post-place re-measure that moved nothing): the shifts from the same prior differ at least by
        out["d_shift_mm"] = round(abs(float(ev["shift_mm"]) - float(r["shift_mm"])), 2)
    if not ok_a and not ok_b and ev.get("fit_shift_mm") is not None and r.get("fit_shift_mm") is not None:
        out["d_fit_shift_mm"] = round(abs(float(ev["fit_shift_mm"]) - float(r["fit_shift_mm"])), 2)
    return out


def episode(P, seq, log) -> list[dict]:
    report, actions = load_src(seq)
    cfg = report.get("config") or {}
    revs = sorted((e for e in report.get("events") or [] if isinstance(e.get("step"), (int, float))),
                  key=lambda e: e["step"])
    rb = RecordedBelief(report, actions, revs)
    rows = []
    for i, ev in enumerate(revs):
        w = writer(ev)
        if w is None:
            continue
        step = int(ev["step"])
        row = {"episode": seq.dir.name, "writer": w, "step": step, "label": ev["label"], "arm": ev.get("arm")}
        if step not in seq.by_step:
            rows.append({**row, "skip": "no stored frame at the step"})
            continue
        try:
            snap, att, src = rb.before(i)
            row["belief"] = src
            req = inputs(ev, snap, att, cfg, report)
            t0 = time.time()
            r = rerun(P, seq.frame(step), req)
            row.update(compare(ev, r, req["objects"][0]["key"]), key=req["objects"][0]["key"], time_s=round(time.time() - t0, 2),
                       n_cands=len(req["identity_vocabulary"] or []) or 1)
        except Exception as e:  # noqa: BLE001 (one write's failure is a row, not the run's end)
            row["skip"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        log(row)
    return rows


def summary(rows: list[dict]) -> dict:
    out = {}
    for w in WRITERS:
        rs = [r for r in rows if r["writer"] == w and "skip" not in r]
        if not rs:
            continue
        both = [r for r in rs if "d_mm" in r]

        def q(vals):
            v = np.asarray(vals, float)
            return ({"p50": round(float(np.percentile(v, 50)), 2), "p95": round(float(np.percentile(v, 95)), 2),
                     "max": round(float(v.max()), 2)} if len(v) else None)
        out[w] = {"n": len(rs), "skipped": sum(1 for r in rows if r["writer"] == w and "skip" in r),
                  "ok_agree": sum(r["ok_rec"] == r["ok_rep"] for r in rs),
                  "ok_flips": [f"{r['episode']}@{r['step']} {r['label']} {r['ok_rec']}->{r['ok_rep']}"
                               for r in rs if r["ok_rec"] != r["ok_rep"]],
                  "both_ok": len(both), "mm": q([r["d_mm"] for r in both]), "xy_mm": q([r["d_xy_mm"] for r in both]),
                  "deg": q([r["d_deg"] for r in both if "d_deg_raw" in r]),     # symmetry-aware rows only
                  "deg_raw": q([r.get("d_deg_raw", r["d_deg"]) for r in both]),
                  "gt_2mm": sum(r["d_mm"] > 2.0 for r in both),
                  "shift_only": q([r["d_shift_mm"] for r in rs if "d_shift_mm" in r]),
                  "fit_shift_mm": q([r["d_fit_shift_mm"] for r in rs if "d_fit_shift_mm" in r])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--valid", default=None)
    ap.add_argument("--names", default="")
    ap.add_argument("--order", default="name", choices=("name", "task_rr"))
    ap.add_argument("--per-task", type=int, default=0, help="with task_rr: at most this many episodes per task")
    a = ap.parse_args()
    sd = _seq_data()
    from dojo.runtime.perception_service import _Perception
    P = _Perception()
    names = sorted(os.listdir(os.path.join(a.root, "seq")))
    if a.valid:
        ok = {r["name"] for r in json.load(open(a.valid)) if r.get("valid")}
        names = [n for n in names if n in ok]
    if a.names:
        names = [n for n in names if n in set(a.names.split(","))]
    if a.order == "task_rr":                    # one episode per task in turn: every writer's floor covers every task
        by: dict = {}
        for n in names:
            by.setdefault(json.load(open(os.path.join(a.root, "seq", n, "meta.json")))["task"], []).append(n)
        lists = [by[t][:a.per_task] if a.per_task else by[t] for t in sorted(by)]
        names = [lst[i] for i in range(max(map(len, lists), default=0)) for lst in lists if i < len(lst)]
    rows = json.load(open(a.out)).get("rows", []) if os.path.exists(a.out) else []     # resume
    done = {r["episode"] for r in rows}

    def log(r):
        print(json.dumps(r), flush=True)
    for n in names:
        if n in done:
            continue
        seq = sd.Sequence(os.path.join(a.root, "seq", n))
        rows += episode(P, seq, log)
        json.dump({"summary": summary(rows), "rows": rows}, open(a.out, "w"), indent=1)
    res = {"summary": summary(rows), "rows": rows}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
