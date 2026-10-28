"""Grasp-time wrist check (R1; behind ``RuntimeConfig.wrist_check``, default off).

At the pre-grasp / pre-place hover the arm's wrist camera sees the work area at ~0.3-0.5 mm/px (the head ~2.4 mm/px):
the only BOP-resolution view we have. The check verifies, in that frame, the belief the manipulator is about to act on:
the target, a place's reference, and every believed object whose footprint lies within ``radius_m`` of the approach
corridor (the grasp / place point plus the last approach segment), i.e. exactly the objects stream M's grasp-time
metric requires (docs: /tmp/dojo-team/PERCEPTION_95.md).

    from dojo.perception.wrist_check import CheckConfig, WristCheck
    wc = WristCheck(CheckConfig(), sam=sam_masks.client())          # sam / embed optional
    out = wc.check(frame, belief, arm="left", kind="pick", target="mug0", corridor=path, held=None)
    out["belief"]     # the belief after the check, same format ({label: {"key", "pos", "quat", ...}})
    out["objects"]    # per checked label: role, verdict, fit iou, sam_iou, fg_frac, moved_mm, ...
    out["found"]      # wrist-view objects no belief explained, fitted to a believed label (relocations)

``belief`` is {label: {"key": "Rigid/mug/00007", "pos": xyz, "quat": wxyz}} in the asset (mesh) frame (a layout
record's frame, ``grasp_metric.flatten_record``); ``frame`` = {"images": {cam: HxWx3 uint8}, "state": official state}.

Per checked object (its belief pose projected into the wrist camera, the robot and a held object masked):
- **existence**: the hover refine (``refine.refine``: local table-colour foreground + silhouette fit) and a SAM 2.1-S
  click at the projected belief. ``ok`` = the fit holds; ``present`` = no fit, but a SAM mask agrees with the
  predicted silhouette; ``phantom`` = no fit, no agreeing mask and the predicted silhouette is table-coloured
  (``fg_frac`` < ``empty_fg_frac``); ``unseen`` = too little of it in view; else ``unsure``. A phantom neighbour /
  reference leaves the belief (``drop_phantoms``) unless a relocation finds it; the target is only flagged.
- **pose**: an ok fit moves a neighbour / reference (``refine_neighbours``); the target's pose is the runtime's own
  hover refine (``refine_target`` re-applies the fit here, for offline evals that start before it).
- **unexplained objects** (``relocate``): wrist-view foreground near the corridor that no believed object explains
  (silhouettes at the checked poses) is a missed or misplaced object: every believed label that is a phantom here, or
  that was not verified in this view, is fitted at the blob; the best ok fit (iou >= ``relocate_min_iou``) moves it.
- **identity + flip** (``embed_model``, default None = off): the label group's candidate assets and the 180-degree
  flip are rendered at the checked pose and scored against the observed crop with DINOv2 (MUSE's recipe: CLS cosine +
  GeM-pooled patch-token cosine, relative to the best rival); a change needs ``identity_margin``; the flip is scored
  always but applied only with ``flip_check`` (E1: no flip was needed, the wrong ones scored up to +0.07).

E1 (offline, stream C's 13 replayed episodes / 64 grasp + place events, stream M's metric): clean at R = 10 cm
38 -> 39 of 64 (+1 / -0); non-target phantom verdicts 7 / 7 right (all through the head view); median 0.21 s per
hover, p90 4.4 s (the radius search). Most failures there are held-object state or neighbours the wrist cannot see
from the hover (docs/journal/2026-10-28-dojo-wrist-check.md).

The check never steps the env and never reads GT. Costs are reported per stage (``time_s``).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

ROLES = ("target", "reference", "neighbour")
_GALLERY = None


@dataclass
class CheckConfig:
    radius_m: float = 0.10              # neighbours: believed footprints within this of the approach corridor
    min_vis_px: int = 150               # predicted visible silhouette below this: unseen (no verdict)
    min_vis_frac: float = 0.3           # ... or less than this share of it visible (robot / held object in front)
    max_trunc: float = 0.5              # ... or more than this share of its vertices outside the image
    exist_iou: float = 0.3              # a SAM mask agreeing this much with the predicted silhouette: present
    empty_fg_frac: float = 0.25         # local foreground below this share of the predicted silhouette: empty (table)
    present_fg_frac: float = 0.6        # the target (no fit of its own here): this much foreground = present
    head_min_px: int = 40               # an object the wrist does not see: head-view phantom needs this many pixels ...
    head_empty_prec: float = 0.15       # ... of which less than this share is static-background foreground
    head_present_prec: float = 0.5      # ... present at this share
    drop_phantoms: bool = True          # a phantom neighbour / reference leaves the belief
    refine_neighbours: bool = True      # an ok fit moves a neighbour / the reference
    refine_target: bool = False         # an ok fit moves the target too (the runtime's hover refine does it already)
    max_shift_m: float = 0.03           # a fit moving an object more than this is not applied (a neighbour grab)
    neighbour_iters: int = 40           # refine iterations for neighbours (the target's hover refine uses 60)
    relocate: bool = True               # fit believed labels at unexplained wrist-view blobs near the corridor
    relocate_radius_m: float = 0.07     # a phantom in view: refine's radius search around its belief (0 = off)
    phantom_max_iou: float = 0.3        # a phantom's own fit explains less than this of its foreground
    relocate_min_px: int = 600          # an unexplained blob smaller than this is ignored
    relocate_min_iou: float = 0.7       # a relocation's fit must explain its blob this well
    relocate_max_cands: int = 3         # believed labels tried per blob (phantoms first, then the nearest)
    relocate_size_ratio: float = 3.0    # a candidate's expected silhouette area within this factor of the blob's
    sam_model: str = "sam2.1_small"
    embed_model: str | None = None      # e.g. "dinov2_b": identity + flip scoring on the wrist crop
    identity_margin: float = 0.05       # MUSE score margin over the believed asset to change identity (E1, 24 objects:
    # 0.03 -> 2 right / 1 wrong changes, 0.05 -> 1 right / 0 wrong)
    flip_check: bool = False            # score the 180-degree flip (E1: 0 flips needed, 4-5 wrong at margins <= 0.05)
    flip_margin: float = 0.1            # ... a flip needs this margin over the believed yaw
    identity_min_px: int = 900          # identity / flip only on objects at least this big in the wrist view
    gem_p: float = 3.0
    cls_weight: float = 0.5             # MUSE score = w * cos(CLS) + (1 - w) * cos(GeM(patches))

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------------------------- geometry
def footprint(key: str, pos, quat) -> np.ndarray:
    """World-xy convex hull (K, 2) of an asset's mesh at a pose."""
    import cv2

    from .cameras import pose_to_T
    from .render import load_asset, transform

    W = transform(load_asset(key).Vs, pose_to_T(pos, quat))[:, :2].astype(np.float32)
    return cv2.convexHull(W).reshape(-1, 2).astype(float)


def corridor_dist(poly: np.ndarray, path, step: float = 0.005) -> float:
    """Distance (m) in xy between a footprint polygon and a TCP path (N, 3) sampled every ``step``; 0 if they touch."""
    import cv2

    P = np.asarray(path, float).reshape(-1, 3)[:, :2]
    pts = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = int(np.ceil(np.linalg.norm(b - a) / step))
        pts += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]] if n else []
    c = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    d = max(cv2.pointPolygonTest(c, (float(p[0]), float(p[1])), True) for p in pts)
    return max(0.0, -float(d))


def _pose7(o: dict) -> list:
    return [float(v) for v in list(o["pos"])[:3] + list(o["quat"])[:4]]


def _interior_point(mask: np.ndarray):
    """The pixel deepest inside a mask (distance transform argmax): a click that lands on the object, not its edge."""
    import cv2

    d = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    v, u = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(u), float(v)


def _iou(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None = None) -> float:
    a, b = a > 0, b > 0
    if valid is not None:
        a, b = a & (valid > 0), b & (valid > 0)
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def yaw_flip(pos, quat, centre) -> tuple[list, list]:
    """The pose turned 180 degrees about world z through ``centre`` (the body's bbox centre)."""
    from .cameras import R_to_quat_wxyz, pose_to_T

    T = pose_to_T(pos, quat)
    Rz = np.diag([-1.0, -1.0, 1.0])
    T2 = np.eye(4)
    T2[:3, :3] = Rz @ T[:3, :3]
    c = np.asarray(centre, float)
    T2[:3, 3] = Rz @ (T[:3, 3] - c) + c
    return T2[:3, 3].tolist(), R_to_quat_wxyz(T2[:3, :3]).tolist()


# ------------------------------------------------------------------------------------------------------ appearance
class MuseScorer:
    """DINOv2 render-vs-crop similarity, MUSE-style (arXiv 2510.17866, recipe only): cosine of the CLS tokens plus
    cosine of GeM-pooled patch tokens (patches inside the object's mask), then each candidate's score relative to its
    best rival. ``embedder`` = ``embed_id.Embedder`` (needs ``tokens``)."""

    def __init__(self, embedder, cls_weight: float = 0.5, gem_p: float = 3.0, size: int = 224):
        self.emb, self.w, self.p, self.size = embedder, cls_weight, gem_p, size

    def _feats(self, imgs: list[np.ndarray], masks: list[np.ndarray]):
        import cv2

        cls, pat = self.emb.tokens(imgs)                       # (N, D), (N, g, g, D)
        g = pat.shape[1]
        out = []
        for i, m in enumerate(masks):
            mm = cv2.resize((m > 0).astype(np.float32), (g, g), interpolation=cv2.INTER_AREA) > 0.3
            P = pat[i][mm] if mm.any() else pat[i].reshape(-1, pat.shape[-1])
            # GeM (p > 1: clamped at 0 as for CNN features; p = 1: the plain mean of the signed tokens)
            gem = P.mean(0) if self.p == 1 else (np.clip(P, 1e-6, None) ** self.p).mean(0) ** (1.0 / self.p)
            gem = gem / max(float(np.linalg.norm(gem)), 1e-9)
            out.append((cls[i], gem))
        return out

    def scores(self, obs: np.ndarray, obs_mask: np.ndarray, renders: list[tuple[np.ndarray, np.ndarray]]) -> list[float]:
        """Absolute MUSE score of each render (rgb, alpha) against the observed crop."""
        from .embed_id import masked

        imgs = [masked(obs, obs_mask)] + [masked(r, a) for r, a in renders]
        f = self._feats(imgs, [obs_mask] + [a for _, a in renders])
        c0, g0 = f[0]
        return [float(self.w * (c @ c0) + (1 - self.w) * (g @ g0)) for c, g in f[1:]]


# ------------------------------------------------------------------------------------------------------- the check
@dataclass
class _Obj:
    label: str
    role: str
    key: str
    pose: list
    dist_m: float = 0.0
    out: dict = field(default_factory=dict)


class WristCheck:
    """One configured check; ``sam`` = anything with ``points(img, uv, model, multimask) -> [SamMask]`` (the
    ``sam_masks`` client, or ``sam_masks.LocalSam`` offline), ``embed`` = an ``embed_id.Embedder`` (identity / flip)."""

    def __init__(self, cfg: CheckConfig | None = None, sam=None, embed=None, candidates=None, refine_fn=None):
        self.cfg = cfg or CheckConfig()
        self.sam = sam
        self.muse = MuseScorer(embed, self.cfg.cls_weight, self.cfg.gem_p) if embed is not None else None
        self.candidates = candidates or (lambda label, key: [key])   # identity candidates of a label
        self._refine = refine_fn

    # ---- pieces (overridable in tests)
    def refine(self, key, img, state, prior, arm, **kw) -> dict:
        if self._refine is not None:
            return self._refine(key, img, state, prior, arm, **kw)
        from .refine import refine

        return refine(key, img, state, prior, arm=arm, **kw)

    @staticmethod
    def camera(arm: str, state: dict):
        from .cameras import wrist_camera

        return wrist_camera(arm, np.asarray(state[f"{arm}_arm_joint_state"], float))

    @staticmethod
    def silhouette(cam, key: str, pose) -> np.ndarray:
        from .cameras import pose_to_T
        from .render import load_asset, silhouette, transform

        a = load_asset(key)
        return silhouette(cam, transform(a.Vs, pose_to_T(pose[:3], pose[3:])), a.Fs)

    @staticmethod
    def truncated(cam, key: str, pose) -> float:
        from .cameras import pose_to_T
        from .refine import truncated_frac
        from .render import load_asset, transform

        return truncated_frac(cam, transform(load_asset(key).Vs, pose_to_T(pose[:3], pose[3:])))

    @staticmethod
    def robot(cam, state: dict, closed_jaws: bool) -> np.ndarray:
        from .render import robot_mask

        return robot_mask(cam, state, dilate_px=3, closed_jaws=closed_jaws)

    @staticmethod
    def foreground(img, rob) -> np.ndarray:
        from .refine import frame_foreground

        return frame_foreground(img, rob)

    @staticmethod
    def fg_frac(img, cam, sil, occ, others, belief_key=None, pose=None) -> float | None:
        """Share of the predicted silhouette that is local foreground (``refine.local_foreground``: table colours from a
        ring around it; other believed objects are not table)."""
        import cv2

        from .refine import local_foreground

        excl = None
        if others:
            excl = np.zeros_like(sil)
            for k, p in others:
                try:
                    excl |= WristCheck.silhouette(cam, k, p)
                except Exception:  # noqa: BLE001
                    continue
            excl = cv2.dilate(excl, np.ones((7, 7), np.uint8)) & (1 - sil)
        fg, known = local_foreground(img, cam, sil, occ, exclude=excl)
        s = cv2.erode(sil.astype(np.uint8), np.ones((5, 5), np.uint8)) & known
        n = int(s.sum())
        return float((fg & s).sum()) / n if n >= 30 else None

    @staticmethod
    def _head_view(frame: dict, bgs: dict, thr: float):
        from .cameras import head_camera
        from .segment import segment_views

        cam = head_camera()
        v = segment_views({"cam_head": frame["images"]["cam_head"]}, {"cam_head": cam}, frame["state"], bgs, thr=thr)
        return v["cam_head"]

    def _head_evidence(self, v, ob: _Obj) -> dict | None:
        sil = self.silhouette(v.cam, ob.key, ob.pose)
        vis = (sil > 0) & (v.known > 0) & (v.robot == 0)
        n = int(vis.sum())
        if n == 0:
            return None
        return {"px": n, "prec": round(float((vis & (v.fg > 0)).sum()) / n, 3)}

    # ---- the check
    def required(self, belief: dict, target: str | None, reference: str | None, corridor, exclude=()) -> list[_Obj]:
        """Target, reference, then believed objects within ``radius_m`` of the corridor (nearest first)."""
        out, seen = [], set(exclude)
        for lab, role in ((target, "target"), (reference, "reference")):
            if lab and lab in belief and belief[lab].get("pos") is not None and lab not in seen:
                out.append(_Obj(lab, role, belief[lab]["key"], _pose7(belief[lab])))
                seen.add(lab)
        nb = []
        for lab, o in belief.items():
            if lab in seen or o.get("pos") is None or not o.get("key"):
                continue
            try:
                d = corridor_dist(footprint(o["key"], o["pos"], o["quat"]), corridor)
            except Exception:  # noqa: BLE001 (no mesh for this key: not checkable)
                continue
            if d <= self.cfg.radius_m:
                nb.append(_Obj(lab, "neighbour", o["key"], _pose7(o), dist_m=round(d, 4)))
        return out + sorted(nb, key=lambda x: x.dist_m)

    def _sam_best(self, img, uv, ref_mask, valid):
        """(best IoU of a SAM click's masks with ``ref_mask`` over ``valid`` pixels, that mask) | (None, None)."""
        if self.sam is None:
            return None, None
        try:
            ms = self.sam.points(img, np.asarray([uv], float), model=self.cfg.sam_model, multimask=True)
        except Exception:  # noqa: BLE001 (server gone: no SAM evidence)
            return None, None
        best = (0.0, None)
        for m in ms:
            v = _iou(m.mask, ref_mask, valid)
            if v > best[0]:
                best = (v, m.mask)
        return round(best[0], 4), best[1]

    def check(self, frame: dict, belief: dict, arm: str, kind: str, target: str | None = None,
              reference: str | None = None, corridor=None, held: tuple | None = None, held_label: str | None = None,
              empty_hand: bool | None = None, head_bgs: dict | None = None, head_thr: float = 18.0) -> dict:
        """See the module doc. ``held`` = (key, pose7) of the object in this arm's jaws (a place), ``held_label`` its
        label (never checked); ``corridor`` (N, 3) TCP path (default: the target's position). ``head_bgs``: the
        perception backgrounds (``segment.load_backgrounds``): objects the wrist does not see are checked in the head
        frame's static-background foreground (``prec`` = share of the predicted silhouette that is foreground)."""
        import copy

        import cv2

        cfg = self.cfg
        t0 = time.perf_counter()
        tm: dict = {}
        state = frame["state"]
        img = frame["images"][f"cam_{arm}_wrist"]
        cam = self.camera(arm, state)
        if corridor is None:
            c = belief.get(target or reference or "", {}).get("pos")
            corridor = np.asarray([c if c is not None else [0, 0, 0]], float)
        empty = (kind == "pick" and held is None) if empty_hand is None else empty_hand
        rob = self.robot(cam, state, closed_jaws=not empty)
        occ = rob.copy()
        if held is not None:
            occ |= cv2.dilate(self.silhouette(cam, held[0], list(held[1])), np.ones((9, 9), np.uint8))
        new = copy.deepcopy(belief)
        objs = self.required(belief, target, reference, corridor, exclude={held_label} if held_label else ())
        tm["select"] = time.perf_counter() - t0
        others_all = [(o["key"], _pose7(o)) for lab, o in belief.items()
                      if lab != held_label and o.get("pos") is not None and o.get("key")]
        # ---- existence + pose, per required object
        t1 = time.perf_counter()
        hv = self._head_view(frame, head_bgs, head_thr) if head_bgs else None
        relocs: list[dict] = []
        for ob in objs:
            r = ob.out
            r.update(role=ob.role, key=ob.key, dist_m=ob.dist_m)
            sil = self.silhouette(cam, ob.key, ob.pose)
            vis = sil & (1 - occ)
            n_sil, n_vis = int(sil.sum()), int(vis.sum())
            trunc = self.truncated(cam, ob.key, ob.pose)
            r.update(sil_px=n_sil, vis_px=n_vis, trunc=round(trunc, 3))
            if n_vis < cfg.min_vis_px or n_vis < cfg.min_vis_frac * max(n_sil, 1) or trunc > cfg.max_trunc:
                r["verdict"] = "unseen"
                if hv is not None:              # the head view's static-background foreground at the belief
                    h = self._head_evidence(hv, ob)
                    if h is not None:
                        r["head"] = h
                        if h["px"] >= cfg.head_min_px and h["prec"] < cfg.head_empty_prec:
                            r["verdict"] = "phantom"
                            r["via"] = "head"
                        elif h["prec"] >= cfg.head_present_prec:
                            r["verdict"] = "present"
                            r["via"] = "head"
                continue
            others = [(k, p) for k, p in others_all if not (k == ob.key and np.allclose(p, ob.pose))]
            do_fit = ob.role != "target" or cfg.refine_target
            ok = False
            fpose = list(ob.pose)
            if do_fit:
                kw = dict(others=others or None, held=held, empty_hand=empty)
                if ob.role == "neighbour":
                    kw.update(iters=cfg.neighbour_iters, search=False, edge_fallback=False)
                fit = self.refine(ob.key, img, state, ob.pose, arm, **kw)
                op = fit.get("obs_prior") or {}
                fg_frac = op.get("fg_px", 0) / op["sil_px"] if op.get("sil_px") else None
                fpose = list(fit.get("pose") or ob.pose)
                shift = float(np.hypot(fpose[0] - ob.pose[0], fpose[1] - ob.pose[1]))
                r.update(iou=fit.get("iou"), fit_ok=bool(fit.get("ok")), moved_mm=round(1000 * shift, 1),
                         turned_deg=fit.get("turned_deg"), fit_why=fit.get("why"))
                if fit.get("fit_pose") is not None:
                    r["fit_pose7"] = [round(float(v), 5) for v in fit["fit_pose"]]
                ok = bool(fit.get("ok")) and shift <= cfg.max_shift_m
            else:                               # the target: the runtime's hover refine owns its pose
                fg_frac = self.fg_frac(img, cam, sil, occ, others, belief_key=ob.key, pose=ob.pose)
            r["fg_frac"] = None if fg_frac is None else round(float(fg_frac), 3)
            ref_sil = self.silhouette(cam, ob.key, fpose) & (1 - occ) if ok else vis
            uv = _interior_point(ref_sil if ref_sil.any() else vis)
            sam_iou, _ = self._sam_best(img, uv, ref_sil, 1 - occ)
            r["sam_iou"] = sam_iou
            if ok:
                r["verdict"] = "ok"
                if (ob.role != "target" and cfg.refine_neighbours) or (ob.role == "target" and cfg.refine_target):
                    if r.get("moved_mm") or r.get("turned_deg"):
                        new[ob.label].update(pos=fpose[:3], quat=fpose[3:7], source="wrist_check")
                        r.update(applied=True, pose7=[round(float(v), 5) for v in fpose])
                ob.pose = fpose
            elif sam_iou is not None and sam_iou >= cfg.exist_iou:
                r["verdict"] = "present"
            elif not do_fit and fg_frac is not None and fg_frac >= cfg.present_fg_frac:
                r["verdict"] = "present"
            elif (fg_frac is not None and fg_frac < cfg.empty_fg_frac and (sam_iou is None or sam_iou < cfg.exist_iou)
                  and float(r.get("iou") or 0.0) < cfg.phantom_max_iou):
                r["verdict"] = "phantom"
            else:
                r["verdict"] = "unsure"
            if r["verdict"] == "phantom" and cfg.relocate and cfg.relocate_radius_m > 0:
                # not where believed, but maybe nearby in this view: the hover refine's radius search (PR #183)
                fit = self.refine(ob.key, img, state, ob.pose, arm, others=others or None, held=held, empty_hand=empty,
                                  search_radius=cfg.relocate_radius_m)
                fpose = list(fit.get("pose") or ob.pose)
                shift = float(np.hypot(fpose[0] - ob.pose[0], fpose[1] - ob.pose[1]))
                rs = fit.get("radius_search") or {}
                r["radius"] = {"ok": bool(fit.get("ok")), "iou": fit.get("iou"), "moved_mm": round(1000 * shift, 1),
                               "used": rs.get("used"), "why": fit.get("why")}
                if fit.get("ok") and float(fit.get("iou") or 0) >= cfg.relocate_min_iou and shift > 1e-4:
                    r["verdict"] = "relocated"
                    new[ob.label].update(pos=fpose[:3], quat=fpose[3:7], source="wrist_relocate")
                    relocs.append({"label": ob.label, "via": "radius", "applied": True, "iou": fit.get("iou"),
                                   "pose7": [round(float(v), 5) for v in fpose], "moved_mm": round(1000 * shift, 1)})
                    ob.pose = fpose
        tm["exist_pose"] = time.perf_counter() - t1
        # ---- unexplained wrist-view objects near the corridor -> relocate a believed label there
        found = list(relocs)
        if cfg.relocate:
            t2 = time.perf_counter()
            found += self._relocate(img, cam, state, arm, new, objs, occ, rob, corridor, held, held_label, empty)
            tm["relocate"] = time.perf_counter() - t2
        relocated = {f["label"] for f in found if f.get("applied")}
        for ob in objs:
            if ob.out.get("verdict") == "phantom" and ob.role != "target" and cfg.drop_phantoms \
                    and ob.label not in relocated:
                new.pop(ob.label, None)
                ob.out["dropped"] = True
        # ---- identity + flip on the target / reference
        if self.muse is not None:
            t3 = time.perf_counter()
            for ob in objs:
                if ob.role in ("target", "reference") and ob.out.get("verdict") in ("ok", "present"):
                    idr = self._identity(img, cam, ob, occ)
                    if idr:
                        ob.out["identity"] = idr
                        if idr.get("change_key") and ob.label in new:
                            new[ob.label]["key"] = idr["change_key"]
                        if idr.get("flip") and ob.label in new:
                            new[ob.label].update(pos=idr["flip_pose"][:3], quat=idr["flip_pose"][3:7],
                                                 source="wrist_flip")
            tm["identity"] = time.perf_counter() - t3
        tm["total"] = time.perf_counter() - t0
        return {"belief": new, "objects": {ob.label: ob.out for ob in objs}, "found": found,
                "time_s": {k: round(v, 3) for k, v in tm.items()}, "camera": cam.name}

    def _relocate(self, img, cam, state, arm, new, objs, occ, rob, corridor, held, held_label, empty) -> list[dict]:
        import cv2

        cfg = self.cfg
        from .cameras import TABLE_Z

        fg = self.foreground(img, rob) & (1 - occ)
        expl = np.zeros_like(fg)
        verdict = {ob.label: ob.out.get("verdict") for ob in objs}
        for lab, o in new.items():
            if lab == held_label or o.get("pos") is None or verdict.get(lab) == "phantom":
                continue
            try:
                expl |= self.silhouette(cam, o["key"], _pose7(o))
            except Exception:  # noqa: BLE001
                continue
        expl = cv2.dilate(expl, np.ones((15, 15), np.uint8))
        resid = (fg & (1 - expl)).astype(np.uint8)
        resid = cv2.morphologyEx(resid, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab_img, stats, cents = cv2.connectedComponentsWithStats(resid, connectivity=8)
        blobs = []
        P = np.asarray(corridor, float).reshape(-1, 3)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < cfg.relocate_min_px:
                continue
            m = (lab_img == i).astype(np.uint8)
            uv = _interior_point(m)
            X = cam.ray_plane(np.asarray([uv]), TABLE_Z + 0.02)[0]
            if not np.isfinite(X).all():
                continue
            d = float(np.min(np.linalg.norm(P[:, :2] - X[None, :2], axis=1)))
            if d > cfg.radius_m + 0.03:
                continue
            blobs.append((area, m, uv, X, d))
        if not blobs:
            return []
        # candidate labels: phantoms here first, then believed objects not verified in this view, nearest first
        checked_ok = {lab for lab, v in verdict.items() if v in ("ok", "present", "relocated")}
        out = []
        taken = set()
        for area, m, uv, X, d in sorted(blobs, key=lambda b: -b[0]):
            cands = []
            for lab, o in new.items():
                if lab == held_label or lab in checked_ok or lab in taken or o.get("pos") is None or not o.get("key"):
                    continue
                pri = 0 if verdict.get(lab) == "phantom" else 1
                cands.append((pri, float(np.hypot(o["pos"][0] - X[0], o["pos"][1] - X[1])), lab))
            rec = {"area_px": area, "uv": [round(uv[0], 1), round(uv[1], 1)], "xy": [round(float(X[0]), 4),
                   round(float(X[1]), 4)], "corridor_m": round(d, 4), "tried": []}
            best = None
            for pri, dist, lab in sorted(cands)[: max(cfg.relocate_max_cands * 3, 6)]:
                if len(rec["tried"]) >= cfg.relocate_max_cands:
                    break
                o = new[lab]
                p0 = _pose7(o)
                # the prior moved onto the blob: its xy at the blob, its rest height kept when it rests on the table
                prior = [float(X[0]), float(X[1]), p0[2]] + p0[3:]
                try:
                    sil = self.silhouette(cam, o["key"], prior)
                except Exception:  # noqa: BLE001
                    continue
                ratio = (sil.sum() + 1) / (area + 1)
                if not (1 / cfg.relocate_size_ratio <= ratio <= cfg.relocate_size_ratio):
                    rec["tried"].append({"label": lab, "skip": f"size x{ratio:.2f}"})
                    continue
                others = [(o2["key"], _pose7(o2)) for l2, o2 in new.items()
                          if l2 not in (lab, held_label) and o2.get("pos") is not None and o2.get("key")]
                fit = self.refine(o["key"], img, state, prior, arm, others=others or None, held=held,
                                  empty_hand=empty, search=False)
                fp = list(fit.get("pose") or prior)
                fs = self.silhouette(cam, o["key"], fp) & (1 - occ)
                biou = _iou(fs, m)
                t = {"label": lab, "phantom": pri == 0, "ok": bool(fit.get("ok")), "iou": fit.get("iou"),
                     "blob_iou": round(biou, 3), "from_mm": round(1000 * dist, 1)}
                rec["tried"].append(t)
                if fit.get("ok") and (fit.get("iou") or 0) >= cfg.relocate_min_iou and (best is None or
                                                                                         fit["iou"] > best[0]):
                    best = (float(fit["iou"]), lab, fp)
            if best is not None:
                _, lab, fp = best
                rec.update(label=lab, pose7=[round(v, 4) for v in fp], applied=True,
                           moved_mm=round(1000 * float(np.hypot(fp[0] - new[lab]["pos"][0], fp[1] - new[lab]["pos"][1])), 1))
                new[lab].update(pos=fp[:3], quat=fp[3:7], source="wrist_relocate")
                taken.add(lab)
            out.append(rec)
        return out

    def _gallery(self):
        """One process-wide gallery: a pyrender renderer that is garbage-collected calls eglTerminate on the shared
        EGL display and breaks every other GL context of the process (GLError 12289 on the next render)."""
        global _GALLERY
        if _GALLERY is None:
            from .embed_id import Gallery
            from .identity import load_color_M

            _GALLERY = Gallery(lit=True, color_M=load_color_M())
        return _GALLERY

    def _identity(self, img, cam, ob: _Obj, occ) -> dict | None:
        """Candidates (the label's group + the 180-degree flip of the believed asset) rendered at the checked pose,
        scored against the wrist crop (MUSE); a change needs the margin over the belief's own render."""
        import cv2

        from .embed_id import crop, square_box
        from .render import load_asset

        cfg = self.cfg
        sil = self.silhouette(cam, ob.key, ob.pose)
        vis = sil & (1 - occ)
        if vis.sum() < cfg.identity_min_px:
            return None
        ys, xs = np.nonzero(sil)
        box = square_box((xs.min(), ys.min(), np.ptp(xs) + 1, np.ptp(ys) + 1), pad=0.15)
        obs = crop(img, box)
        keys = list(dict.fromkeys([ob.key] + [k for k in self.candidates(ob.label, ob.key) if k != ob.key]))
        a = load_asset(ob.key)
        centre = (a.V.min(0) + a.V.max(0)) / 2
        from .cameras import pose_to_T

        T = pose_to_T(ob.pose[:3], ob.pose[3:])
        fp, fq = yaw_flip(ob.pose[:3], ob.pose[3:], T[:3, :3] @ centre + T[:3, 3])
        items = [(k, ob.pose, "id") for k in keys] + [(ob.key, fp + fq, "flip")]
        gal = self._gallery()
        renders = []
        for k, p, _ in items:
            rgb, alpha = gal.render_box(cam, k, pose_to_T(p[:3], p[3:]), box, lit=True)
            renders.append((rgb, alpha))
        occ_c = crop(occ.astype(np.uint8) * 255, box, interp=cv2.INTER_NEAREST) > 127
        # the observed crop is masked by the union of the candidates' silhouettes minus the occluders
        union = np.zeros(obs.shape[:2], bool)
        for _, al in renders:
            union |= al > 0
        obs_mask = (union & ~occ_c).astype(np.uint8)
        renders = [(rgb, ((al > 0) & ~occ_c).astype(np.uint8)) for rgb, al in renders]
        s = self.muse.scores(obs, obs_mask, renders)
        sc = {f"{tag}:{k}": round(v, 4) for (k, _, tag), v in zip(items, s)}
        s_id = s[: len(keys)]
        best_i = int(np.argmax(s_id))
        out = {"scores": sc, "n_keys": len(keys), "box_px": round(float(box[2] - box[0]), 1)}
        if best_i != 0 and s_id[best_i] - s_id[0] >= cfg.identity_margin:
            out["change_key"] = keys[best_i]
            out["identity_margin"] = round(s_id[best_i] - s_id[0], 4)
        else:
            rival = max(s_id[1:], default=None)
            out["identity_margin"] = None if rival is None else round(s_id[0] - rival, 4)
        sym = self._yaw_period(ob.key)
        if cfg.flip_check and sym > 1.5 * np.pi and s[-1] - s[0] >= cfg.flip_margin and "change_key" not in out:
            out["flip"] = True
            out["flip_pose"] = [round(v, 5) for v in fp + fq]
        out["flip_margin"] = round(s[-1] - s[0], 4)
        return out

    @staticmethod
    def _yaw_period(key: str) -> float:
        try:
            from .inverse_layout import shape_models

            return float(shape_models(key)[0].yaw_period)
        except Exception:  # noqa: BLE001
            return 2 * np.pi
