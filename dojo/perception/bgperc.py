"""Background (continuous) perception: re-measure believed objects from the frames the episode produces anyway.

The runtime perceives at a few discrete points (frame 0, scan, looks, the pre-grasp refine); between them the robot
moves and both wrist cameras sweep the table at every step. ``BackgroundPerceiver`` consumes those frames:

1. **visibility**: per believed object and wrist camera, the belief pose's silhouette (decimated mesh): area, share
   inside the image, share under the robot (arm, jaws, a held object) and camera distance;
2. **priority**: objects REQUIRED for the next event (its target, the place reference, neighbours within ``R`` of the
   approach point) first, then the unconfirmed / low-confidence ones; at most ``max_refines`` wrist refines per
   processed step (``every`` steps), the best view first;
3. **refine**: perception's local wrist render-and-compare (``refine.refine``) from the belief pose;
4. **fusion** per object, since the robot last touched it (``epoch``): an ok fit close to the belief confirms it; a fit
   elsewhere is a candidate that moves the belief only when ``min_views`` fits from different steps agree within
   ``agree_mm`` (or one fit is very strong); fits that see the place empty count as absence evidence (``doubt``);
5. **look request**: a required object still unconfirmed with low confidence, or in doubt, at decision time
   (``needs_look``) is what a targeted look should be spent on.

Optional (default off): periodic re-reads (B1 on a later head view, ``reread_every``: gross errors), held checks during
a carry (perception's B1 held_look, ``held_every``: the jaws measured closed on nothing) and contact suspects (a jaw
inside an untouched object's footprint: searched for at its next view, ``contact``).

The perceiver never moves the robot and never writes GT; it reads only the official observation (images + state) and
the belief (``sync``). ``touch(label)`` (a grasp / release / push of that object) starts a new epoch. Offline replay
and scoring: ``scripts/perception/bgperc/``; in the runtime: ``RuntimeConfig.bg_perception`` (service op ``bg_step``).
Silhouettes are GL when a context exists (the service); CPU silhouettes of 25k-face meshes cost ~13 s per refine.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BgConfig:
    every: int = 2                 # process every k-th observation step
    max_refines: int = 1           # wrist refines per processed step (the CPU budget: ~0.05 s a block, ~0.6 s a mug)
    min_px: int = 250              # belief silhouette area in a wrist view (full-resolution pixels)
    max_trunc: float = 0.15        # share of the object's vertices outside the image
    max_robot: float = 0.4         # share of the silhouette under the robot mask (arm, jaws, held object)
    max_dist: float = 0.65         # m, wrist camera to object
    accept_iou: float = 0.7        # an ok fit counts as an observation from this iou
    agree_mm: float = 6.0          # fits within this of each other (xy) agree; within this of the belief confirm it
    max_shift_mm: float = 60.0     # fits farther from the belief are ignored (a neighbour, not this object)
    min_views: int = 2             # agreeing fits from different steps that move a belief
    step_gap: int = 3              # steps between two fits that count as different views
    strong_iou: float = 0.9        # ... or one fit this good (and visible_frac >= strong_vis) within strong_mm
    strong_vis: float = 0.8
    strong_mm: float = 15.0
    move_turn: bool = False        # a move takes the fits' orientation too (default: position only, the belief's yaw
                                   # stays; build_tower 100002: a fit's 7 deg turned a right block "rot" wrong)
    absent_iou: float = 0.25       # a not-ok fit in clear view (vis >= absent_vis) below this iou: nothing there
    absent_vis: float = 0.7
    search_radius: float = 0.06    # m: objects in doubt are refined with perception's radius search
    revisit_steps: int = 80        # a confirmed, untouched object is re-measured at most this often
    req_revisit_steps: int = 30    # ... a required one (the next event's target / reference / neighbours)
    low_conf: float = 0.6          # B1 confidence below this: unconfirmed objects rank first
    R: float = 0.10                # m, neighbours of the approach point (the grasp-time metric's R)
    refine_iters: int = 60
    # periodic re-reads (B1 on a later frame), for gross errors / phantoms / identity
    reread_every: int = 0          # steps between re-reads (0: off)
    reread_views: str = "head"     # head (B1 on the head view: ~2-6 s) | all (+ both wrists: 40+ s on mug scenes)
    reread_max_robot: float = 0.18 # ... on a frame whose head view the robot covers at most this much (home: 0.153)
    reread_conf: float = 0.5       # a reading this confident counts
    reread_agree_mm: float = 15.0  # two re-reads within this of each other (and off the belief) move it
    reread_match_m: float = 0.35   # a reading pairs with a believed object of its category within this
    reread_held_m: float = 0.10    # readings this close to a held object are dropped (the jaws' object, not a free one)
    wide_search: bool = False      # perception's wide search after a failed fit (slow; the absence evidence replaces it)
    max_s: float = 0.0             # 0: no wall budget per step; else stop refining in a step after this many s
    # held checks during a carry (perception's B1 held_look on the holding wrist, no LLM): is it still in the jaws?
    held_every: int = 0            # steps between two checks of a held object (0: off)
    held_after: int = 8            # ... from this many steps after the close (the lift)
    held_drop_conf: float = 0.8    # a "not held" this sure counts
    held_drop_reads: int = 2       # ... and this many in a row drop it
    held_jaws_closed: bool = True  # ... and only reads that MEASURED the jaws closed past half the object's narrowest
                                   # width (the runtime's own sure-drop rule, ``RuntimeProvider._jaws_closed``)
    # contacts: a jaw passing through an untouched object's footprint below its top may have moved it
    contact: bool = False
    contact_margin_m: float = 0.015


@dataclass
class Track:
    label: str
    key: str
    pose: np.ndarray               # (7,) pos + quat wxyz: the belief
    conf: float = 0.5              # the estimate's own confidence
    epoch: int = 0
    confirmed: int | None = None   # step of the last confirmation (this epoch)
    moved: int | None = None       # step the background moved the belief (this epoch)
    obs: list = field(default_factory=list)   # this epoch's fits: dict(step, arm, xy, pose, iou, vis, ok)
    absent: int = 0                # absence evidences since the last ok fit
    last_try: int = -10 ** 6
    held: bool = False
    rereads: list = field(default_factory=list)   # this epoch's re-read readings off the belief: (step, pose7, key)
    suspect: int | None = None     # step a jaw passed through it (it may have moved): searched at the next view
    held_arm: str | None = None
    held_since: int | None = None
    not_held: int = 0              # consecutive confident "not held" reads
    last_held_check: int = -10 ** 6

    DOUBT_AFTER = 2                # absence evidences (no ok fit since) that put an object in doubt

    @property
    def doubt(self) -> bool:
        return self.absent >= self.DOUBT_AFTER and self.confirmed is None


def reread_estimate(images: dict, state: dict, task: str, instruction: str, bgs: dict, cfg, views: str = "head") -> dict:
    """B1 on one later frame for a re-read: the head view alone (``views="head"``) or every camera."""
    import dataclasses

    from .inverse_layout import estimate
    if views == "head":
        cfg = dataclasses.replace(cfg, use_wrists=False)
    return estimate(images, state, task, instruction, bgs=bgs, cfg=cfg)


def record_readings(rec: dict | None) -> dict:
    """A layout record as re-read readings: label -> {"key", "pos", "quat", "conf"} (``BackgroundPerceiver.reread``)."""
    out = {}
    for typ in ("Rigid", "Geometry", "Articulation", "Dynamic"):
        for cat, lst in ((rec or {}).get(typ) or {}).items():
            for r in lst or []:
                if not r.get("label") or r.get("default_pos") is None:
                    continue
                out[r["label"]] = {"key": f"{typ}/{cat}/{int(r.get('category_idx', 0)):05d}",
                                   "pos": [float(v) for v in r["default_pos"]][:3],
                                   "quat": [float(v) for v in (r.get("default_ori") or [1, 0, 0, 0])][:4],
                                   "conf": float((r.get("_perception") or {}).get("confidence", 0.5))}
    return out


class BackgroundPerceiver:
    def __init__(self, cfg: BgConfig | None = None, refine_fn=None, log=None, writes_back: bool = False):
        """``writes_back``: the caller writes every returned move into the belief it ``sync``s from (the runtime), so a
        synced pose equal to our own move is not someone else's change (offline arms keep the moves as overrides)."""
        self.cfg = cfg or BgConfig()
        self.writes_back = writes_back
        self._synced: dict = {}
        self._sources: dict = {}
        self.tracks: dict[str, Track] = {}
        self.stats = {"steps": 0, "processed": 0, "vis_s": 0.0, "refine_s": 0.0, "refines": 0, "ok": 0,
                      "confirms": 0, "moves": 0, "absent": 0, "searches": 0}
        self.events: list[dict] = []
        self._refine = refine_fn
        self.log = log or (lambda *_: None)

    # ------------------------------------------------------------------ belief interface
    def init(self, objects: dict, step: int = 0) -> None:
        """``objects``: label -> {"key", "pos", "quat", "conf"} (the belief the episode starts from)."""
        self.tracks = {}
        for lab, o in objects.items():
            self.tracks[lab] = Track(lab, o["key"], np.r_[np.asarray(o["pos"], float)[:3], np.asarray(o["quat"], float)[:4]],
                                     float(o.get("conf", 0.5)))
        self._synced = {lab: t.pose[:3].copy() for lab, t in self.tracks.items()}

    def sync(self, objects: dict, held=()) -> None:
        """Follow the caller's belief (label -> {"key", "pos", "quat"[, "conf"]}; ``held``: labels in the jaws). A pose
        that changed by more than 1 mm since the last sync was moved by someone else (the runtime's own refine, a
        place, a re-estimate): the track takes it and starts a new epoch. Held objects follow the jaws, skipped.
        The background's own moves are not echoed back as changes."""
        held = set(held)
        if not self.tracks:
            self.init({lab: o for lab, o in objects.items() if o.get("pos") is not None})
        synced = self._synced
        for lab, o in objects.items():
            if o.get("pos") is None:
                continue
            p = np.asarray(o["pos"], float)[:3]
            t = self.tracks.get(lab)
            if t is None:                                   # a new label (a re-estimate added it)
                self.tracks[lab] = Track(lab, o["key"], np.r_[p, np.asarray(o["quat"], float)[:4]],
                                         float(o.get("conf", 0.5)))
                synced[lab] = p
                t = self.tracks[lab]
            if lab in held or t.held:                       # in the jaws / just released: a new epoch, no evidence
                if (lab in held) != t.held:
                    self.touch(lab, held=lab in held)
                t.pose = np.r_[p, np.asarray(o["quat"], float)[:4]]
                synced[lab] = p
                if o.get("source") is not None:
                    self._sources[lab] = o["source"]
                continue
            src = o.get("source")
            if lab not in synced or float(np.linalg.norm(p - synced[lab])) > 0.001 or \
                    (src is not None and src != self._sources.get(lab, src)):
                # moved, or re-measured by someone else (a refine that kept the pose within 1 mm still supersedes our
                # older evidence: build_tower 100002, a pick refine 0 mm off after a background move)
                self.set_pose(lab, p, o["quat"], touched=True)
                if o.get("key"):
                    t.key = o["key"]
                synced[lab] = p
            if src is not None:
                self._sources[lab] = src
        self._synced = synced

    def set_pose(self, label: str, pos, quat, touched: bool = True, step: int | None = None) -> None:
        """The runtime moved the belief (its own refine, a place, the held object): a new epoch when ``touched``."""
        t = self.tracks.get(label)
        if t is None:
            return
        t.pose = np.r_[np.asarray(pos, float)[:3], np.asarray(quat, float)[:4]]
        if touched:
            self.touch(label)

    def touch(self, label: str, held: bool | None = None) -> None:
        t = self.tracks.get(label)
        if t is None:
            return
        t.epoch += 1
        t.obs, t.absent, t.confirmed, t.moved, t.rereads, t.suspect = [], 0, None, None, [], None
        if held is not None:
            t.held = held
            t.not_held, t.last_held_check = 0, -10 ** 6

    def belief(self) -> dict:
        return {lab: {"key": t.key, "pos": t.pose[:3].tolist(), "quat": t.pose[3:].tolist(), "conf": t.conf,
                      "confirmed": t.confirmed, "moved": t.moved, "doubt": t.doubt, "n_obs": len(t.obs)}
                for lab, t in self.tracks.items()}

    def required(self, point_xy, target: str | None = None, reference: str | None = None) -> list[str]:
        """The next event's required objects from the belief: target, reference, believed objects within R of the
        approach point (their footprint)."""
        from .render import load_asset

        req = [x for x in (target, reference) if x and x in self.tracks]
        if point_xy is None:
            return req
        p = np.asarray(point_xy, float)[:2]
        for lab, t in self.tracks.items():
            if lab in req or t.held:
                continue
            try:
                r = 0.5 * float(np.max(load_asset(t.key).extents[:2]))
            except Exception:  # noqa: BLE001
                r = 0.03
            if float(np.hypot(*(t.pose[:2] - p))) - r <= self.cfg.R:
                req.append(lab)
        return req

    def needs_look(self, labels) -> list[dict]:
        """Required objects a targeted look should be spent on: in doubt, or never confirmed and low-confidence."""
        out = []
        for lab in labels:
            t = self.tracks.get(lab)
            if t is None or t.held:
                continue
            if t.doubt:
                out.append({"label": lab, "why": "doubt", "absent": t.absent})
            elif t.confirmed is None and t.moved is None and t.conf < self.cfg.low_conf:
                out.append({"label": lab, "why": "unconfirmed_low_conf", "conf": round(t.conf, 2)})
        return out

    # ------------------------------------------------------------------ per frame
    def visibility(self, frame: dict, labels) -> list[dict]:
        """Per (label, arm): the belief silhouette in that wrist view. Only views passing the gates are returned."""
        from .cameras import cameras_from_state, pose_to_T
        from .refine import truncated_frac
        from .render import load_asset, robot_mask, silhouette, transform

        cfg = self.cfg
        cams = cameras_from_state(frame["state"])
        out = []
        for arm in ("left", "right"):
            cam = cams.get(f"cam_{arm}_wrist")
            if cam is None:
                continue
            rob = None
            for lab in labels:
                t = self.tracks[lab]
                if t.held:
                    continue
                T = pose_to_T(t.pose[:3], t.pose[3:])
                if float(np.linalg.norm(T[:3, 3] - cam.center)) > cfg.max_dist:
                    continue
                a = load_asset(t.key)
                Vw = transform(a.Vs, T)
                tr = truncated_frac(cam, Vw)
                if tr > cfg.max_trunc:
                    continue
                sil = silhouette(cam, Vw, a.Fs)
                n = int(sil.sum())
                if n < cfg.min_px:
                    continue
                if rob is None:
                    rob = robot_mask(cam, frame["state"], dilate_px=3, closed_jaws=True)
                    for hl, ht in self.tracks.items():          # a held object hides what is behind it
                        if ht.held:
                            ha = load_asset(ht.key)
                            rob |= silhouette(cam, transform(ha.Vs, pose_to_T(ht.pose[:3], ht.pose[3:])), ha.Fs)
                under = float((sil & rob).sum()) / n
                if under > cfg.max_robot:
                    continue
                out.append({"label": lab, "arm": arm, "px": n, "trunc": round(tr, 3), "robot": round(under, 3),
                            "dist": round(float(np.linalg.norm(T[:3, 3] - cam.center)), 3)})
        return out

    def _priority(self, lab: str, step: int, required: set) -> float | None:
        """None = not now. Unconfirmed objects first (required ones, low-confidence ones, a candidate waiting for its
        second view, objects in doubt); a confirmed object only when required and its confirmation is older than
        ``req_revisit_steps`` (or any object after ``revisit_steps``)."""
        t = self.tracks[lab]
        cfg = self.cfg
        if t.held or step - t.last_try < cfg.step_gap:
            return None
        req = lab in required
        if t.confirmed is not None:
            age = step - t.confirmed
            if age < (cfg.req_revisit_steps if req else cfg.revisit_steps):
                return None
            return 1.5 if req else 0.5
        p = 1.0
        if req:
            p += 4.0
        if t.doubt or (req and t.absent) or t.suspect is not None:   # empty place / a jaw went through it
            p += 2.0
        if t.conf < cfg.low_conf:
            p += 1.0
        if t.obs:                                        # a candidate waiting for its second view
            p += 1.5
        return p

    # ------------------------------------------------------------------ held checks and contacts
    def held_due(self, step: int, holds: dict) -> list[tuple[str, str]]:
        """(arm, label) pairs whose held check is due (``holds``: arm -> label in its jaws, with the close step in
        ``Track.held_since``)."""
        cfg = self.cfg
        out = []
        if not cfg.held_every:
            return out
        for arm, lab in holds.items():
            t = self.tracks.get(lab)
            if t is None:
                continue
            if t.held_arm != arm:
                t.held_arm, t.held_since, t.not_held, t.last_held_check = arm, step, 0, -10 ** 6
            if step - t.held_since >= cfg.held_after and step - t.last_held_check >= cfg.held_every:
                out.append((arm, lab))
        return out

    def held_result(self, step: int, arm: str, label: str, r: dict) -> dict | None:
        """A held_look read for ``label`` in ``arm``'s jaws. Returns a drop ({"label", "arm", "step", "reads"}) when
        ``held_drop_reads`` confident "not held" reads came in a row; a confident "held" resets the count."""
        cfg = self.cfg
        t = self.tracks[label]
        t.last_held_check = step
        self.stats["held_checks"] = self.stats.get("held_checks", 0) + 1
        held, conf = r.get("held"), float(r.get("confidence") or 0.0)
        ev = {"step": step, "label": label, "arm": arm, "held_look": held, "conf": round(conf, 2)}
        closed = self._jaws_closed(r, t.key) if cfg.held_jaws_closed else True
        ev["jaws_closed"] = closed
        if held is False and conf >= cfg.held_drop_conf and closed:
            t.not_held += 1
        elif held is True and conf >= 0.6:
            t.not_held = 0
        ev["not_held"] = t.not_held
        self.events.append(ev)
        if t.not_held >= cfg.held_drop_reads:
            self.stats["held_drops"] = self.stats.get("held_drops", 0) + 1
            t.not_held = 0
            return {"label": label, "arm": arm, "step": step, "reads": cfg.held_drop_reads}
        return None

    @staticmethod
    def _jaws_closed(r: dict, key: str) -> bool:
        """held_look measured the jaw opening (evidence "opening") narrower than half the object's narrowest width
        (capped at 10 mm): nothing that size is between the pads."""
        op = r.get("opening") or {}
        if r.get("evidence") != "opening" or op.get("measured") is None:
            return False
        try:
            from .render import load_asset
            w = float(np.min(load_asset(key).extents))
        except Exception:  # noqa: BLE001
            w = 0.02
        try:
            from dojo.skills.kinematics import grip_to_gap
            gap = grip_to_gap(float(op["measured"]))
        except Exception:  # noqa: BLE001 (no skills package: the command scale, 0 = closed)
            gap = 0.1 * float(op["measured"])
        return gap < max(0.001, min(0.01, 0.5 * w))

    def contacts(self, step: int, tcps: list) -> list[str]:
        """``tcps``: TCP positions (xyz) of both jaws at this step. Untouched objects whose footprint a jaw entered
        below their top (+ ``contact_margin_m``) become suspects: searched at their next clear view."""
        from .render import load_asset

        cfg = self.cfg
        if not cfg.contact:
            return []
        out = []
        for lab, t in self.tracks.items():
            if t.held:
                continue
            try:
                ext = load_asset(t.key).extents
            except Exception:  # noqa: BLE001
                continue
            r = 0.5 * float(np.max(ext[:2])) + cfg.contact_margin_m
            top = float(t.pose[2]) + 0.5 * float(ext[2]) + cfg.contact_margin_m
            for p in tcps:
                p = np.asarray(p, float)
                if float(np.hypot(*(p[:2] - t.pose[:2]))) < r and p[2] < top:
                    if t.suspect is None:
                        t.suspect = step
                        t.confirmed = None
                        self.stats["contacts"] = self.stats.get("contacts", 0) + 1
                        self.events.append({"step": step, "label": lab, "contact": True})
                        out.append(lab)
                    break
        return out

    # ------------------------------------------------------------------ re-reads
    def reread_due(self, step: int, frame: dict | None = None) -> bool:
        """A re-read is due (``reread_every`` steps since the last) and the frame's head view is clear enough."""
        cfg = self.cfg
        if not cfg.reread_every or step - getattr(self, "_last_reread", -10 ** 6) < cfg.reread_every:
            return False
        if frame is None:
            return True
        from .cameras import cameras_from_state
        from .render import robot_mask
        cam = cameras_from_state(frame["state"])["cam_head"]
        cover = float(robot_mask(cam, frame["state"], dilate_px=0, closed_jaws=True).mean())
        return cover <= cfg.reread_max_robot

    def reread(self, step: int, readings: dict) -> list[dict]:
        """A full re-read (label -> {"key", "pos", "quat", "conf"}) against the belief. Readings pair with believed
        objects of the same category (nearest pairs first, 3-D distance <= ``reread_match_m``; labels of interchangeable objects
        may differ between estimates). A paired reading within ``agree_mm`` confirms an unconfirmed object; one
        farther off is kept, and a second re-read within ``reread_agree_mm`` of it moves the belief there (objects nobody
        touched this epoch only; held ones are skipped). Returns the moves."""
        cfg = self.cfg
        self._last_reread = step
        self.stats["rereads"] = self.stats.get("rereads", 0) + 1
        cat = lambda key: str(key).split("/")[1] if str(key).count("/") == 2 else str(key)   # noqa: E731
        changes = []
        free = {lab: t for lab, t in self.tracks.items() if not t.held}
        # a reading at a held object is that object in the jaws (or under them), never a free one of its category
        held_p = [t.pose[:3] for t in self.tracks.values() if t.held]
        readings = {lab: r for lab, r in readings.items()
                    if all(float(np.linalg.norm(np.asarray(r["pos"], float)[:3] - p)) > cfg.reread_held_m for p in held_p)}
        for c in sorted({cat(t.key) for t in free.values()}):
            T = [lab for lab, t in free.items() if cat(t.key) == c]
            Rr = [lab for lab, r in readings.items() if cat(r["key"]) == c and float(r.get("conf", 1.0)) >= cfg.reread_conf]
            if not T or not Rr:
                continue
            C = np.array([[float(np.linalg.norm(np.asarray(readings[r]["pos"], float) - self.tracks[t].pose[:3]))
                           for r in Rr] for t in T])
            pairs, used_i, used_j = [], set(), set()       # greedy nearest pairs (a few objects per category)
            for i, j in sorted(np.ndindex(*C.shape), key=lambda ij: C[ij]):
                if i not in used_i and j not in used_j and C[i, j] <= cfg.reread_match_m:
                    pairs.append((i, j))
                    used_i.add(i)
                    used_j.add(j)
            for i, j in pairs:
                t, r = self.tracks[T[i]], readings[Rr[j]]
                pose = np.r_[np.asarray(r["pos"], float)[:3], np.asarray(r["quat"], float)[:4]]
                d = float(np.hypot(*(pose[:2] - t.pose[:2]))) * 1000
                ev = {"step": step, "label": t.label, "reread": True, "d_mm": round(d, 1), "key": r["key"]}
                if d <= cfg.agree_mm:
                    if t.confirmed is None:
                        t.confirmed = step
                    t.rereads = []
                    ev["action"] = "confirm"
                else:
                    near = [x for x in t.rereads if float(np.hypot(*(x[1][:2] - pose[:2]))) * 1000 <= cfg.reread_agree_mm]
                    t.rereads.append((step, pose, r["key"]))
                    if near:
                        pts = np.stack([x[1] for x in near] + [pose])
                        old = t.pose.copy()
                        t.pose = np.r_[np.median(pts[:, :3], axis=0), pose[3:] if cfg.move_turn else old[3:]]
                        if r["key"] != t.key and all(x[2] == r["key"] for x in near):
                            ev["key_from"] = t.key
                            t.key = r["key"]
                        t.confirmed, t.moved = step, step
                        t.rereads = []
                        if self.writes_back:
                            self._synced[t.label] = t.pose[:3].copy()
                        self.stats["moves"] += 1
                        ev["action"] = "move"
                        changes.append({"label": t.label, "from": old.tolist(), "to": t.pose.tolist(), "step": step,
                                        "n_views": len(near) + 1, "reread": True, "key": t.key})
                    else:
                        ev["action"] = "candidate"
                self.events.append(ev)
        return changes

    def step(self, frame: dict, step: int, required=(), others_extra=()) -> list[dict]:
        """Consume one observation. ``required``: labels the next event needs. Returns the belief changes made."""
        cfg = self.cfg
        self.stats["steps"] += 1
        if step % max(1, cfg.every):
            return []
        self.stats["processed"] += 1
        req = set(required)
        pri = {lab: self._priority(lab, step, req) for lab in self.tracks}
        cands = [lab for lab, p in pri.items() if p is not None and p > 0]
        if not cands:
            return []
        t0 = time.perf_counter()
        views = self.visibility(frame, cands)
        self.stats["vis_s"] += time.perf_counter() - t0
        views.sort(key=lambda v: (-pri[v["label"]], -v["px"]))
        done, changes, n = set(), [], 0
        t_step = time.perf_counter()
        for v in views:
            if n >= cfg.max_refines or v["label"] in done:
                continue
            if cfg.max_s and time.perf_counter() - t_step > cfg.max_s:
                break
            done.add(v["label"])
            n += 1
            ch = self._observe(frame, step, v, v["label"] in req)
            if ch:
                changes.append(ch)
        return changes

    def _refine_call(self, key, img, state, prior, arm, others, radius):
        if self._refine is not None:
            return self._refine(key, img, state, prior, arm, others, radius)
        from .refine import refine
        kw = {"search_radius": radius} if radius else {}
        return refine(key, img, state, list(map(float, prior)), arm=arm, others=others or None,
                      iters=self.cfg.refine_iters, search=self.cfg.wide_search, **kw)

    def _observe(self, frame: dict, step: int, v: dict, required: bool = False) -> dict | None:
        cfg = self.cfg
        t = self.tracks[v["label"]]
        t.last_try = step
        others = [(o.key, o.pose.tolist()) for lab, o in self.tracks.items() if lab != t.label and not o.held]
        # an object in doubt (or a required one whose place just looked empty) is searched for around its belief
        radius = cfg.search_radius if (t.doubt or (required and t.absent) or t.suspect is not None) else None
        if radius:
            self.stats["searches"] += 1
        t0 = time.perf_counter()
        try:
            r = self._refine_call(t.key, frame["images"][f"cam_{v['arm']}_wrist"], frame["state"], t.pose, v["arm"],
                                  others, radius)
        except Exception as e:  # noqa: BLE001 (one bad view never stops the episode)
            self.events.append({"step": step, "label": t.label, "error": f"{type(e).__name__}: {str(e)[:120]}"})
            return None
        dt = time.perf_counter() - t0
        self.stats["refine_s"] += dt
        self.stats["refines"] += 1
        ok = bool(r.get("ok"))
        iou, vis = float(r.get("iou") or 0.0), float(r.get("visible_frac") or 0.0)
        pose = np.asarray(r.get("pose") if ok else (r.get("fit_pose") or r.get("pose")), float)
        shift = float(np.hypot(*(pose[:2] - t.pose[:2]))) * 1000
        ev = {"step": step, "label": t.label, "arm": v["arm"], "ok": ok, "iou": round(iou, 3), "vis": round(vis, 3),
              "shift_mm": round(shift, 1), "px": v["px"], "s": round(dt, 3)}
        change = None
        max_shift = cfg.max_shift_mm + (1000 * radius if radius else 0.0)
        if ok and iou >= cfg.accept_iou and shift <= max_shift:
            self.stats["ok"] += 1
            t.absent = 0
            t.suspect = None
            t.obs.append({"step": step, "arm": v["arm"], "xy": pose[:2].copy(), "pose": pose.copy(), "iou": iou,
                          "vis": vis})
            if shift <= cfg.agree_mm:
                t.confirmed = step
                self.stats["confirms"] += 1
                ev["action"] = "confirm"
            else:
                change = self._try_move(t, step, ev)
        elif not ok and vis >= cfg.absent_vis and iou < cfg.absent_iou:
            t.absent += 1
            self.stats["absent"] += 1
            ev["action"] = "absent"
        self.events.append(ev)
        return change

    def _try_move(self, t: Track, step: int, ev: dict) -> dict | None:
        cfg = self.cfg
        last = t.obs[-1]
        cl = [o for o in t.obs if float(np.hypot(*(o["xy"] - last["xy"]))) * 1000 <= cfg.agree_mm]
        steps = sorted({o["step"] for o in cl})
        distinct = 1 + sum(1 for a, b in zip(steps, steps[1:]) if b - a >= cfg.step_gap)
        strong = (last["iou"] >= cfg.strong_iou and last["vis"] >= cfg.strong_vis
                  and float(np.hypot(*(last["xy"] - t.pose[:2]))) * 1000 <= cfg.strong_mm)
        if distinct < cfg.min_views and not strong:
            ev["action"] = "candidate"
            return None
        best = max(cl, key=lambda o: o["iou"])
        xy = np.median(np.stack([o["xy"] for o in cl]), axis=0)
        old = t.pose.copy()
        t.pose = np.r_[xy, best["pose"][2], best["pose"][3:] if cfg.move_turn else old[3:]]
        t.confirmed, t.moved = step, step
        t.obs = cl
        if self.writes_back:
            self._synced[t.label] = t.pose[:3].copy()
        self.stats["moves"] += 1
        ev["action"] = "move"
        ev["n_views"] = distinct
        return {"label": t.label, "from": old.tolist(), "to": t.pose.tolist(), "step": step, "n_views": distinct,
                "strong": strong}


__all__ = ["BgConfig", "Track", "BackgroundPerceiver", "record_readings", "reread_estimate"]
