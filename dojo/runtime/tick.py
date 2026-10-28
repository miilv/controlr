"""A1 (stream W): the belief's tick, runtime side (numpy only; the perception is ``dojo.perception.redetect``).

    st = TickState()
    req = build_request(objs, held, triggers, st, ...)          # who is active, which seeds per trigger kind
    resp = <perception service op "redetect"> (req)              # detections, association, hypotheses
    writes, discrepancies = decide(resp, objs, triggers, st, TickConfig(), step)

``decide`` is deterministic in (resp, belief, triggers, state, cfg): no wall time, no randomness, sorted labels.

Write rules (reviewer, 28 Oct; rendered arm 9's fasten_screws loss):
- the incumbent competes: the believed pose is a hypothesis scored on the same evidence; a candidate writes only when its
  support beats the incumbent's by ``margin``. A near tie never writes;
- support bar: a touched object needs one camera at IoU >= ``touched_min_iou`` (>= ``touched_min_px`` px) or two at
  ``two_cam_iou``; an untouched one (a "missing" promotion, a contact suspect) a wrist at ``untouched_wrist_iou`` or two
  cameras; a small object (tiny / thin mesh) two cameras, or the same hypothesis on two ticks;
- resolution floor: no write below max(``floor_mm``, ``floor_px`` x the deciding view's mm/px); below ``near_move_mm``
  the fit IoU must reach ``near_move_iou``;
- fixture-held objects (a nut on its thread, a mug on its hook: ``fixture``) keep the kinematic belief unless the error
  is gross (> ``gross_mm``) and two views / two ticks agree;
- a collapsed object flips again only at 2 x margin over 2 ticks;
- the pad volume of an empty close is falsified: no hypothesis inside it is taken;
- a candidate whose footprint overlaps a believed, not-active object by > ``max_overlap`` of the smaller is rejected
  (an elongated neighbour; a look-alike already believed there);
- an active object with no supported hypothesis is marked uncertain (never re-asserted at a falsified pose).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

WRISTS = ("cam_left_wrist", "cam_right_wrist")


@dataclass(frozen=True)
class TickConfig:
    mode: str = "log"                  # off | log | apply
    margin: float = 0.10
    touched_min_iou: float = 0.60
    touched_min_px: int = 400
    two_cam_iou: float = 0.50
    untouched_wrist_iou: float = 0.70
    floor_mm: float = 3.0
    floor_px: float = 1.5
    near_move_mm: float = 10.0
    near_move_iou: float = 0.80
    gross_mm: float = 45.0
    flip_mult: float = 2.0
    confirm_mm: float = 15.0           # two ticks agree when their hypotheses are this close
    lift_m: float = 0.03
    added_views: int = 2
    added_miss_drop: int = 2
    max_replans: int = 2
    max_overlap: float = 0.20          # a candidate whose footprint covers this share of a believed, not-active object

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trigger:
    kind: str                          # close_empty | release | slip | lift | contact | hover
    step: int
    arm: str | None = None
    label: str | None = None
    data: dict = field(default_factory=dict)


@dataclass
class Discrepancy:
    kind: str                          # relocated | uncertain | phantom_flag | added | removed | attach_relabel
    label: str | None
    step: int
    trigger: str
    before7: list | None = None
    after7: list | None = None
    evidence: dict = field(default_factory=dict)
    applied: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TickState:
    """What the tick keeps between ticks (per episode)."""
    pending: dict = field(default_factory=dict)       # label -> (step, pose7) of a supported hypothesis awaiting a 2nd tick
    collapsed: dict = field(default_factory=dict)     # label -> pose7 the tick last wrote
    uncertain: set = field(default_factory=set)
    falsified: dict = field(default_factory=dict)     # label -> [(centre xyz, half extents xyz)] pad volumes of empty closes
    added: dict = field(default_factory=dict)         # label -> {key, pose7, misses}
    added_pending: list = field(default_factory=list)  # [(step, key, pose7)]
    n_added: int = 0
    log: list = field(default_factory=list)


# --------------------------------------------------------------------------------------------- the request
def seeds_for(trigger: Trigger, obj: dict, meta: dict) -> list[dict]:
    """Seeds per trigger kind (reviewer B3 / the fasten_screws correction):
    - close_empty: none here (the pad volume is falsified; the near-miss ring is the radius search / the detections);
    - release / slip: under the jaws (``meta["drop7"]``), and "never left" (``meta["grasp7"]``) only after a real lift
      (``meta["lift_m"]`` >= TickConfig.lift_m) with no evidence the object rose with the jaws (``meta["rose"]``)."""
    out = []
    if trigger.kind in ("release", "slip"):
        if meta.get("drop7") is not None:
            out.append({"src": "under_jaws", "pose7": list(meta["drop7"])})
        if (meta.get("grasp7") is not None and float(meta.get("lift_m") or 0.0) >= TickConfig.lift_m
                and not meta.get("rose")):
            out.append({"src": "never_left", "pose7": list(meta["grasp7"])})
    return out


def build_request(objs: list[dict], held: dict, triggers: list[Trigger], meta: dict | None = None,
                  touched: set | None = None, q_path: dict | None = None, corridor=None) -> dict:
    """``objs`` [{label, key, pose7, small?, fixture?}] (the belief, held ones included), ``meta`` {label: {...}} the seed
    inputs. Active: the triggers' labels + ``touched`` (objects the robot moved, the twin's > 4 mm moves included)."""
    meta = meta or {}
    act: dict = {}
    seeds: dict = {}
    for t in sorted(triggers, key=lambda t: (t.step, t.kind, t.label or "")):
        if t.label and t.kind in ("close_empty", "release", "slip", "contact", "hover"):
            act.setdefault(t.label, t.kind)
            seeds.setdefault(t.label, []).extend(seeds_for(t, {}, meta.get(t.label, {})))
    for lab in sorted(touched or ()):
        act.setdefault(lab, "touched")
    held_labels = set(held.values())
    out = []
    for o in sorted(objs, key=lambda o: o["label"]):
        lab = o["label"]
        r = {"label": lab, "key": o["key"], "pose7": [float(v) for v in o["pose7"]],
             "active": None if lab in held_labels else act.get(lab), "seeds": seeds.get(lab, [])}
        out.append(r)
    lift = sorted({t.arm for t in triggers if t.kind == "lift" and t.arm})
    return {"objects": out, "held": dict(held), "q_path": q_path or {}, "lift": lift,
            "corridor": None if corridor is None else [list(map(float, p)) for p in corridor]}


# --------------------------------------------------------------------------------------------- deciding
def _ev(h: dict, min_px: int = 30) -> dict:
    """Support summary: best IoU, its camera, cams at >= 0.5, best wrist IoU."""
    s = {c: v for c, v in (h.get("support") or {}).items() if v[1] >= min_px}
    if not s:
        return {"best": 0.0, "cam": None, "px": 0, "n50": 0, "wrist": 0.0}
    cam = max(sorted(s), key=lambda c: s[c][0])
    return {"best": float(s[cam][0]), "cam": cam, "px": int(s[cam][1]),
            "n50": sum(1 for v in s.values() if v[0] >= TickConfig.two_cam_iou),
            "wrist": max((float(v[0]) for c, v in s.items() if c in WRISTS), default=0.0)}


def _inside(p, boxes) -> bool:
    p = np.asarray(p[:3], float)
    return any(np.all(np.abs(p - np.asarray(c, float)) <= np.asarray(hh, float)) for c, hh in boxes or ())


def _mm_per_px(cam_name: str | None) -> float:
    # head ~2.4 mm/px at ~0.7 m; a wrist at a hover ~0.4 mm/px (conservative constants: the floor, not a fit)
    return 2.4 if cam_name == "cam_head" else 0.5


def decide(resp: dict, objs: list[dict], triggers: list[Trigger], st: TickState, cfg: TickConfig,
           step: int) -> tuple[dict, list[Discrepancy]]:
    """Returns ({label: pose7 to write}, discrepancies). ``objs`` [{label, key, pose7, small, fixture, touched}]."""
    by_label = {o["label"]: o for o in objs}
    trig = {}
    for t in sorted(triggers, key=lambda t: (t.step, t.kind)):
        if t.label:
            trig.setdefault(t.label, t.kind)
    writes: dict = {}
    out: list[Discrepancy] = []
    assignment = resp.get("assignment") or {}
    for lab in sorted(resp.get("hyps") or {}):
        o = by_label.get(lab)
        if o is None:
            continue
        hs = resp["hyps"][lab]
        reason = (resp.get("active") or {}).get(lab, trig.get(lab, "touched"))
        inc = hs[0]
        e_inc = _ev(inc)
        # candidates: the seeds and the detection hypothesis this object won in the joint assignment (B4)
        won = assignment.get(lab)
        cands = [(i, h) for i, h in enumerate(hs[1:], 1)
                 if not h["src"].startswith("det:") or (won is not None and won[2] == i)]
        cands = [(i, h) for i, h in cands if not _inside(h["pose7"], st.falsified.get(lab))
                 and float((h.get("overlap") or [None, 0.0])[1]) <= cfg.max_overlap]
        touched = bool(o.get("touched")) or reason in ("close_empty", "release", "slip", "touched", "hover")
        best = None
        for i, h in cands:
            e = _ev(h)
            key = (e["n50"], e["best"])
            if best is None or key > best[0]:
                best = (key, i, h, e)
        evid = {"reason": reason, "incumbent": e_inc, "n_hyps": len(hs), "dets_hash": resp.get("dets_hash")}
        wrote = False
        if best is not None:
            _, i, h, e = best
            evid["cand"] = {"src": h["src"], **e, "fit_iou": h.get("fit_iou")}
            move = 1000.0 * float(np.linalg.norm(np.asarray(h["pose7"][:3]) - np.asarray(inc["pose7"][:3])))
            evid["move_mm"] = round(move, 1)
            if touched:
                ok = (e["best"] >= cfg.touched_min_iou and e["px"] >= cfg.touched_min_px) or e["n50"] >= 2
            else:
                ok = e["wrist"] >= cfg.untouched_wrist_iou or e["n50"] >= 2
            why = None if ok else "support"
            margin = cfg.margin * (cfg.flip_mult if lab in st.collapsed else 1.0)
            if why is None and e["best"] < e_inc["best"] + margin:
                why = "incumbent"
            floor = max(cfg.floor_mm, cfg.floor_px * _mm_per_px(e["cam"]))
            if why is None and move < floor:
                why = "floor"
            if why is None and move < cfg.near_move_mm and e["best"] < cfg.near_move_iou:
                why = "near_move_iou"
            if why is None and o.get("fixture") and move <= cfg.gross_mm:
                why = "fixture"
            two_tick = False
            if why is None and (o.get("small") or o.get("fixture") or lab in st.collapsed) and e["n50"] < 2:
                prev = st.pending.get(lab)
                two_tick = prev is not None and prev[0] < step and \
                    1000.0 * float(np.linalg.norm(np.asarray(prev[1][:3]) - np.asarray(h["pose7"][:3]))) <= cfg.confirm_mm
                if not two_tick:
                    st.pending[lab] = (step, list(h["pose7"]))
                    why = "second_tick"
            evid["why"] = why
            if why is None:
                apply = cfg.mode == "apply"
                out.append(Discrepancy("relocated", lab, step, reason, list(inc["pose7"]), list(h["pose7"]), evid, apply))
                if apply:
                    writes[lab] = list(h["pose7"])
                st.collapsed[lab] = list(h["pose7"])
                st.pending.pop(lab, None)
                st.uncertain.discard(lab)
                wrote = True
        if not wrote:
            sup = (e_inc["best"] >= cfg.touched_min_iou and e_inc["px"] >= cfg.touched_min_px) or e_inc["n50"] >= 2
            if sup:
                st.uncertain.discard(lab)
            elif reason in ("close_empty", "slip", "release") and not _supported_anywhere(inc):
                if lab not in st.uncertain:
                    st.uncertain.add(lab)
                    out.append(Discrepancy("uncertain", lab, step, reason, list(inc["pose7"]), None, evid,
                                           cfg.mode == "apply"))
            elif reason == "missing":
                out.append(Discrepancy("phantom_flag", lab, step, reason, list(inc["pose7"]), None, evid, False))
    out += _added(resp, st, cfg, step)
    for lf in resp.get("lift") or []:
        d = lf.get("det")
        if d is not None and not lf.get("exact"):
            out.append(Discrepancy("attach_relabel", lf.get("label"), step, "lift", None, None, dict(lf), False))
    st.log += [d.to_dict() for d in out]
    return writes, out


def _supported_anywhere(h: dict) -> bool:
    return any(v[0] >= 0.3 and v[1] >= 30 for v in (h.get("support") or {}).values())


def _added(resp: dict, st: TickState, cfg: TickConfig, step: int) -> list[Discrepancy]:
    """Unexplained detections near the corridor: an obstacle once two views (or two ticks) agree."""
    out = []
    for a in resp.get("added") or []:
        n_views = sum(1 for v in (a.get("support") or {}).values() if v[0] >= cfg.two_cam_iou)
        prev = next((p for p in st.added_pending if p[1] == a["key"] and p[0] < step and
                     1000.0 * math.dist(p[2][:3], a["pose7"][:3]) <= 20.0), None)
        if n_views >= cfg.added_views or prev is not None:
            st.n_added += 1
            lab = f"det_{a['key'].split('/')[1] if a['key'].count('/') >= 2 else a['key']}_{st.n_added}"
            st.added[lab] = {"key": a["key"], "pose7": list(a["pose7"]), "misses": 0}
            out.append(Discrepancy("added", lab, step, "hover", None, list(a["pose7"]),
                                   {"key": a["key"], "views": n_views, "corridor_m": a.get("corridor_m")},
                                   cfg.mode == "apply"))
            if prev is not None:
                st.added_pending.remove(prev)
        else:
            st.added_pending.append((step, a["key"], list(a["pose7"])))
    return out


__all__ = ["TickConfig", "Trigger", "Discrepancy", "TickState", "build_request", "decide", "seeds_for"]
