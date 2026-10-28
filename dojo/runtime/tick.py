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
- an object with a same-key twin in the belief moves farther than ``twin_far_mm`` only on ``twin_min_cams`` agreeing
  cameras (the mask may be the twin's);
- the pad volume of an empty close is falsified: no hypothesis inside it is taken;
- a candidate whose footprint overlaps a believed, not-active object by > ``max_overlap`` of the smaller is rejected,
  and so is a write onto another object's final footprint (``blocking_overlap``: same height layer, not a static
  support / container, not resting one on the other)
  (an elongated neighbour; a look-alike already believed there);
- an active object with no supported hypothesis is marked uncertain (never re-asserted at a falsified pose).

Trigger kinds: close_empty / slip / release / lift (``triggers_from_events``), hover (pick / place, with a corridor),
contact (the perception's swept-volume suspects), and ``periodic`` (label None: an always-on schedule's tick between
events). A periodic tick activates no object and seeds nothing: only the touched objects, the contact suspects and
"missing" promotions are active, under the same write rules, so it can confirm (a second tick), flag or relocate only
what any tick could.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

WRISTS = ("cam_left_wrist", "cam_right_wrist")
MIN_VIS_TO_MOVE = 0.9   # = RuntimeConfig.place_min_vis: a view showing less of the object cannot move a belief


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
    min_vis_to_move: float = MIN_VIS_TO_MOVE   # deciding views must show this share of the object (low-visibility
                                               # fits slide along it: M's arm-8 audit, 30-47 mm at 0.53-0.63)
    pending_max_age: int = 60          # steps: a second-tick confirmation / pending obstacle older than this expires
    max_overlap: float = 0.20          # a candidate whose footprint covers this share of a believed, not-active object
    # identical twins (the lead, a priori): a relocation farther than twin_far_mm of an object with a same-key twin in
    # the belief needs twin_min_cams cameras agreeing on the candidate (IoU >= two_cam_iou, views >= min_vis_to_move);
    # a wrist at the touched bar plus the head is such a pair. Association between identical objects is ambiguous and
    # breaking a correct belief costs more than not fixing a wrong one
    twin_far_mm: float = 50.0
    twin_min_cams: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trigger:
    kind: str                          # close_empty | release | slip | lift | contact | hover | periodic
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
def _ev(h: dict, min_px: int = 30, min_vis: float = 0.0) -> dict:
    """Support summary over the cameras with >= ``min_px`` px and visible fraction >= ``min_vis``: best IoU, its
    camera, cams at >= 0.5, best wrist IoU, the scores of the detections used."""
    s = {c: v for c, v in (h.get("support") or {}).items()
         if v[1] >= min_px and (len(v) < 3 or v[2] is None or float(v[2]) >= min_vis)}
    if not s:
        return {"best": 0.0, "cam": None, "px": 0, "n50": 0, "wrist": 0.0, "scores": {}}
    cam = max(sorted(s), key=lambda c: s[c][0])
    return {"best": float(s[cam][0]), "cam": cam, "px": int(s[cam][1]),
            "n50": sum(1 for v in s.values() if v[0] >= TickConfig.two_cam_iou),
            "wrist": max((float(v[0]) for c, v in s.items() if c in WRISTS), default=0.0),
            "scores": {c: (v[3] if len(v) > 3 else None) for c, v in sorted(s.items())}}


def _agreeing_cams(h: dict, cfg: "TickConfig", min_score: float = 0.5) -> int:
    """Distinct cameras supporting this one hypothesis at IoU >= two_cam_iou, on views showing >= min_vis_to_move of
    it, from detections scoring >= ``min_score`` (the seed threshold; the perception's support already uses only
    those)."""
    n = 0
    for v in (h.get("support") or {}).values():
        vis_ok = len(v) < 3 or v[2] is None or float(v[2]) >= cfg.min_vis_to_move
        score_ok = len(v) < 4 or v[3] is None or float(v[3]) >= min_score
        if float(v[0]) >= cfg.two_cam_iou and int(v[1]) >= 30 and vis_ok and score_ok:
            n += 1
    return n


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
    final_fp: dict = {}                                   # footprints of this tick's writes
    for lab_, (step_, _) in list(st.pending.items()):     # S4: a confirmation never older than pending_max_age
        if step - step_ > cfg.pending_max_age:
            st.pending.pop(lab_)
    st.added_pending = [p for p in st.added_pending if step - p[0] <= cfg.pending_max_age]
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
        # the touched bar only for objects the robot moved (reviewer M1: an untouched pick target at its hover needs a
        # wrist at 0.7 or two cameras, like any untouched object)
        touched = bool(o.get("touched")) or reason in ("close_empty", "release", "slip")
        best = None
        for i, h in cands:
            e = _ev(h, min_vis=cfg.min_vis_to_move)   # a write rests only on views that show the object (>= 0.9)
            key = (e["n50"], e["best"])
            if best is None or key > best[0]:
                best = (key, i, h, e)
        evid = {"reason": reason, "incumbent": e_inc, "n_hyps": len(hs), "dets_hash": resp.get("dets_hash")}
        wrote = False
        if best is not None:
            _, i, h, e = best
            evid["cand"] = {"src": h["src"], **e, "fit_iou": h.get("fit_iou"), "overlap": h.get("overlap"),
                            "all_views": _ev(h)}
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
            if why is None and move > cfg.twin_far_mm and _agreeing_cams(h, cfg) < cfg.twin_min_cams \
                    and any(x["key"] == o["key"] and x["label"] != lab for x in objs):
                why = "twin_far"
            two_tick = False
            if why is None and (o.get("small") or o.get("fixture") or lab in st.collapsed) and e["n50"] < 2:
                prev = st.pending.get(lab)
                two_tick = prev is not None and prev[0] < step and step - prev[0] <= cfg.pending_max_age and \
                    1000.0 * float(np.linalg.norm(np.asarray(prev[1][:3]) - np.asarray(h["pose7"][:3]))) <= cfg.confirm_mm
                if not two_tick:
                    st.pending[lab] = (step, list(h["pose7"]))
                    why = "second_tick"
            if why is None:
                # S1: the write's footprint against every other object's FINAL pose (this tick's writes included): two
                # active identical pieces must not end up on one another
                fp = h.get("footprint")
                for lab2 in sorted(set((resp.get("footprints") or {})) | set(final_fp)):
                    if lab2 == lab or fp is None:
                        continue
                    fp2 = final_fp.get(lab2, (resp.get("footprints") or {}).get(lab2))
                    if fp2 is not None and blocking_overlap(fp, fp2) > cfg.max_overlap:
                        why = f"overlap:{lab2}"
                        break
            evid["why"] = why
            if why is None:
                apply = cfg.mode == "apply"
                out.append(Discrepancy("relocated", lab, step, reason, list(inc["pose7"]), list(h["pose7"]), evid, apply))
                if apply:                     # S5: log mode keeps no state as if it had written
                    writes[lab] = list(h["pose7"])
                    st.collapsed[lab] = list(h["pose7"])
                    if h.get("footprint") is not None:
                        final_fp[lab] = h["footprint"]
                st.pending.pop(lab, None)
                st.uncertain.discard(lab)
                wrote = True
        if not wrote:
            sup = (e_inc["best"] >= cfg.touched_min_iou and e_inc["px"] >= cfg.touched_min_px) or e_inc["n50"] >= 2
            if sup:
                st.uncertain.discard(lab)
            elif (reason in ("close_empty", "slip", "release") and not _supported_anywhere(inc)
                  and not (resp.get("assoc") or {}).get(lab)):        # S2: explained at the 0.25 level: seen
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


def _clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: the convex polygon ``subject`` clipped by the convex polygon ``clip`` (both CCW or both CW)."""
    out = [np.asarray(p, float) for p in subject]
    c = np.asarray(clip, float)
    sgn = 1.0 if _area(c) >= 0 else -1.0
    for i in range(len(c)):
        a, b = c[i], c[(i + 1) % len(c)]
        inp, out = out, []
        if not inp:
            break

        def inside(p):
            return sgn * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= 0

        def cross(p, q):
            d1, d2 = q - p, b - a
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-15:
                return p
            t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den
            return p + t * d1
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            if inside(q):
                if not inside(p):
                    out.append(cross(p, q))
                out.append(q)
            elif inside(p):
                out.append(cross(p, q))
    return np.asarray(out) if out else np.zeros((0, 2))


def _area(p) -> float:
    p = np.asarray(p, float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _poly_overlap(a, b) -> float:
    """Intersection area / the smaller area of two convex xy footprints."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    sm = min(abs(_area(a)), abs(_area(b)))
    return abs(_area(_clip(a, b))) / sm if sm > 1e-12 else 0.0


REST_TOL_M = 0.005          # a top this close under the other's bottom: one rests on the other
LAYER_FRAC = 0.5            # two objects share a layer when their heights overlap by this share of the smaller


def blocking_overlap(a, b) -> float:
    """How much candidate ``a`` blocks on object ``b`` (footprints {"xy": hull, "z": [lo, hi] | None, "static": bool};
    a bare hull = xy only): 0 when ``b`` is a static support / container (Geometry, Articulation: a bottle into the
    bin, a nut on its bolt), when one rests on the other (a block on a block) or when their heights overlap by less
    than ``LAYER_FRAC`` of the smaller; else the xy overlap / the smaller footprint. Two identical pieces in one layer
    and a lock-on to a neighbour stay refused. Shared by the perception's per-hypothesis guard and ``decide``'s
    final-footprint guard."""
    a = a if isinstance(a, dict) else {"xy": a}
    b = b if isinstance(b, dict) else {"xy": b}
    if a.get("xy") is None or b.get("xy") is None or b.get("static"):
        return 0.0
    za, zb = a.get("z"), b.get("z")
    if za is not None and zb is not None:
        if zb[1] <= za[0] + REST_TOL_M or za[1] <= zb[0] + REST_TOL_M:
            return 0.0
        ov = min(za[1], zb[1]) - max(za[0], zb[0])
        if ov < LAYER_FRAC * min(za[1] - za[0], zb[1] - zb[0]):
            return 0.0
    return _poly_overlap(a["xy"], b["xy"])


# --------------------------------------------------------------------------------------------- the shared entry point
def run(obs: dict, objs: list[dict], held: dict, triggers: list[Trigger], meta: dict, st: TickState, cfg: TickConfig,
        client, step: int, touched=(), q_path: dict | None = None, held_path: dict | None = None, corridor=None,
        close_tcp: dict | None = None) -> tuple[dict, list[Discrepancy], dict]:
    """One tick, the same lines online (``RuntimeProvider.a1_tick``) and offline (stream C's X2 harness):
    the falsified pad volumes of this tick's empty closes (centred on the TCP at the close: ``close_tcp`` {label: xyz},
    else the object's pose), the request, ``client(obs, req)`` (perception op "redetect"), ``decide``.
    Returns (writes, discrepancies, the perception's response)."""
    for t in triggers:
        if t.kind == "close_empty" and t.label:
            c = (close_tcp or {}).get(t.label)
            if c is None:
                o = next((x for x in objs if x["label"] == t.label), None)
                c = None if o is None else o["pose7"][:3]
            if c is not None:
                st.falsified.setdefault(t.label, []).append(([float(v) for v in c[:3]], [0.02, 0.02, 0.03]))
    req = build_request(objs, held, triggers, meta=meta, touched=set(touched), q_path=q_path, corridor=corridor)
    if held_path:
        req["held_path"] = held_path
    resp = client(obs, req) or {}
    writes, ds = decide(resp, objs, triggers, st, cfg, step)
    return writes, ds, resp


# --------------------------------------------------------------------------------------------- inputs from a log
def triggers_from_events(new_events: list[dict], released: list[tuple], attached: list[tuple], lifted: set,
                         step: int, lift_m: float = 0.03) -> list[Trigger]:
    """The tick's triggers, as ``RuntimeProvider._a1_triggers`` derives them (offline harnesses rebuild them from a
    recorded report with this, so offline triggers equal online ones):
    - ``new_events``: the provider's events since the last tick (grip_check "dropped" air / slip; held_look "held"
      False that was not "kept");
    - ``released``: [(label, arm)] released since the last tick;
    - ``attached``: [(arm, label, grasp n, TCP z at the close, TCP z now)]; a lift fires once per (label, grasp n) when
      the TCP rose >= ``lift_m`` (``lifted`` is updated in place)."""
    trig = []
    for e in new_events:
        k = e.get("kind")
        if k == "grip_check" and e.get("dropped") in ("air", "slip"):
            trig.append(Trigger("close_empty" if e["dropped"] == "air" else "slip", step, e.get("arm"), e.get("label")))
        elif k == "held_look" and e.get("held") is False and not e.get("kept") and e.get("label"):
            trig.append(Trigger("close_empty", step, e.get("arm"), e["label"]))
    for lab, arm in released:
        trig.append(Trigger("release", step, arm, lab))
    for arm, lab, n, z0, z in sorted(attached):
        if (lab, n) in lifted or z - z0 < lift_m:
            continue
        lifted.add((lab, n))
        trig.append(Trigger("lift", step, arm, lab))
    return trig


def held_path_from_log(log: list[tuple], sampled: dict, keys: dict) -> dict:
    """Which object each arm carried at each sampled q of the contact sweep, so a release tick still sweeps the carry:
    ``log`` [(idx, arm, label | None, T_tcp_obj | None)] in idx order (from executed action idx on, the arm holds
    ``label`` at that in-hand pose; None: nothing), ``sampled`` {arm: [executed index of each q in q_path[arm]]},
    ``keys`` {label: asset key}. Returns {arm: [{"i0", "i1", "label", "key", "T_tcp_obj"}]}: q_path[arm][i0:i1] carried
    it. Online: ``RuntimeProvider`` logs the kinematic belief's attachments; offline: the recorded grasps / releases."""
    out: dict = {}
    for arm, idxs in sorted(sampled.items()):
        ent = [e for e in log if e[1] == arm]
        segs: list = []
        j, cur = 0, None
        for i, k in enumerate(idxs):
            while j < len(ent) and ent[j][0] <= k:
                cur, j = j, j + 1
            e = ent[cur] if cur is not None else None
            if e is None or e[2] is None or keys.get(e[2]) is None:
                continue
            if segs and segs[-1]["i1"] == i and segs[-1]["_e"] == cur:
                segs[-1]["i1"] = i + 1
            else:
                segs.append({"i0": i, "i1": i + 1, "label": e[2], "key": keys[e[2]],
                             "T_tcp_obj": np.asarray(e[3], float).reshape(4, 4).tolist(), "_e": cur})
        if segs:
            out[arm] = [{k: v for k, v in sg.items() if k != "_e"} for sg in segs]
    return out


def fixture_labels(events: list[dict], held_now=(), grasps=()) -> set:
    """Labels a fixture holds now: the online rule (``RuntimeProvider._fixture_held``) on a recorded run. A label is in
    when its latest ``belief_drop`` event with how "held by a fixture" / "on a fixture" is later than its latest grasp
    (``grasps``: [(label, step)], the steps it was attached) and it is not held now."""
    last_drop: dict = {}
    for e in events:
        if e.get("kind") == "belief_drop" and e.get("how") in ("held by a fixture", "on a fixture") and e.get("label"):
            last_drop[e["label"]] = max(last_drop.get(e["label"], -1), int(e.get("step", 0)))
    last_grasp: dict = {}
    for lab, st in grasps:
        last_grasp[lab] = max(last_grasp.get(lab, -1), int(st))
    return {lab for lab, sd in last_drop.items() if sd > last_grasp.get(lab, -1)} - set(held_now)


def rose_evidence(events: list[dict], label: str, grasp_step: int, release_step: int) -> bool:
    """A grip_check "held" verdict or a held_look "held" (>= 0.5) for ``label`` between its close and its release: the
    object rose with the jaws (``_grip_held_seen`` / ``_vision_held_seen``), so no "never left" seed."""
    for e in events:
        s = e.get("step")
        if e.get("label") != label or not isinstance(s, (int, float)) or not grasp_step <= s <= release_step:
            continue
        if e.get("kind") == "grip_check" and e.get("verdict") == "held":
            return True
        if e.get("kind") == "held_look" and e.get("held") is True and float(e.get("confidence") or 1.0) >= 0.5:
            return True
    return False


__all__ = ["TickConfig", "Trigger", "Discrepancy", "TickState", "build_request", "decide", "seeds_for", "run",
           "triggers_from_events", "held_path_from_log", "fixture_labels", "rose_evidence", "blocking_overlap"]
