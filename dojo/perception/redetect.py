"""A1 (stream W): region-free re-detection at the runtime's decision points (perception side).

At a tick (``dojo.runtime.tick``: after an empty close, after a release, at a lift, after a contact, at a pick / place
hover) the twin-trained segmenter (``seg_twin``: class = catalogue asset) runs on all three cameras, its masks are
associated with the belief, and the objects the runtime is unsure of get pose hypotheses scored on that evidence:

    out = redetect(frame, req, DetSource(seg_twin.client()), RDConfig())

- **detections** (``DetSource``): sorted by (score desc, x0, y0), masks below ``min_det_px`` dropped, cached by a content
  hash of the image bytes + the model id + the input size (the offline replays and the online runs see the same inputs);
- **prediction**: z-buffered silhouettes of the believed objects per camera, the robot (FK) as occluder;
- **association**: per camera, one Hungarian over believed objects x detections, gated by the asset category (an exact
  asset key gets a bonus) and IoU >= ``gate_iou`` or a centroid within ``gate_mm`` at the object's depth. Observation
  only: no task-config pools, label groups, generator regions or counts;
- **active objects** (the request's, plus promotions): an untouched object predicted clearly visible (>= 32 x 32 px in a
  camera, or >= 16 x 16 in two) that no camera associates is "missing"; an object the arm swept (links, jaws, the held
  object, since the last tick) is a contact suspect;
- **hypotheses** per active object: the incumbent (its believed pose), the runtime's seeds, and fits inside the
  unexplained detections of its category (``fit_in_mask``: every catalogue resting pose, full yaw grid, no region,
  bound or yaw prior), each scored by its support in every camera (IoU with the best same-category mask there). One
  assignment over all active objects x detection-derived hypotheses (a mask feeds at most one object);
- **added**: unexplained detections near the request's corridor, fitted with their own class, as obstacle proposals;
- **lift**: the class of the mask between the holding arm's jaws in its wrist view (log-only until measured).

Deciding what to write is the runtime's (``dojo.runtime.tick.decide``). This module never reads GT, a layout id or a
task name; ``redetect`` raises on a GT key in the frame.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass

import numpy as np

CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
GT_KEYS = ("gt", "_dev_gt", "objects_gt", "dev_gt", "layout_id", "task")


@dataclass(frozen=True)
class RDConfig:
    det_conf: float = 0.25
    explain_min_score: float = 0.25   # a detection that explains a believed object (blocks a "missing" promotion)
    seed_min_score: float = 0.50      # a detection that seeds / supports a write or an added object (C: ~7 % of wrist
                                      # detections at 0.25-0.46 flip between JPEG and raw frames)
    min_det_px: int = 25
    gate_iou: float = 0.10
    exact_bonus: float = 0.10
    promote_min_px: int = 1024        # "missing": predicted visible area in one camera (32 x 32 px) ...
    promote_two_cam_px: int = 256     # ... or this much (16 x 16) in two cameras
    max_robot_occ: float = 0.4        # ... with at most this share of the silhouette under the robot
    min_vis_assoc_px: int = 20        # an object below this visible area takes no part in a camera's association
    max_fits: int = 8                 # fits in masks per tick (count cap, never time)
    nm_iters: int = 60
    yaw_step_deg: float = 15.0
    contact_margin_m: float = 0.01
    contact_pts_per_link: int = 200
    added_max_corridor_m: float = 0.10
    max_fit_range_m: float = 1.5      # a mask's table-plane point farther than this from its camera is not fitted
    explained_max_jump_m: float = 0.05   # an associated object takes no unexplained mask farther than this

    def to_dict(self) -> dict:
        return asdict(self)


def category(key: str) -> str:
    return key.split("/")[1] if isinstance(key, str) and key.count("/") >= 2 else str(key)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def pose7(T: np.ndarray) -> list:
    from .cameras import R_to_quat_wxyz

    return [float(v) for v in T[:3, 3]] + [float(v) for v in R_to_quat_wxyz(T[:3, :3])]


# ------------------------------------------------------------------------------------------------- detections
def image_hash(img: np.ndarray, model_id: str = "", imgsz: int = 0) -> str:
    h = hashlib.sha1(np.ascontiguousarray(img).tobytes())
    h.update(f"|{model_id}|{imgsz}|{img.shape}".encode())
    return h.hexdigest()[:24]


def sort_dets(dets: list) -> list:
    """Deterministic order: score desc, then the mask's top-left corner."""
    def k(d):
        ys, xs = np.nonzero(d.mask)
        return (-round(float(d.score), 4), int(xs.min()) if len(xs) else 0, int(ys.min()) if len(ys) else 0, d.key)
    return sorted(dets, key=k)


class DetSource:
    """The segmenter (``seg_twin`` client or anything with ``detect(img, conf) -> [Det]``) behind a content-hash cache.
    ``cache``: a dict {hash: [Det]} (offline: loaded from / saved to disk, see ``save`` / ``load``)."""

    def __init__(self, seg=None, model_id: str = "", imgsz: int = 0, cache: dict | None = None, conf: float = 0.25,
                 min_px: int = 25):
        self.seg, self.model_id, self.imgsz = seg, model_id, int(imgsz)
        self.cache = {} if cache is None else cache
        self.conf, self.min_px = conf, min_px
        self.hits = self.misses = 0

    def detect(self, img: np.ndarray) -> list:
        k = image_hash(img, self.model_id, self.imgsz)
        if k in self.cache:
            self.hits += 1
            return self.cache[k]
        self.misses += 1
        if self.seg is None:
            return []
        dets = [d for d in self.seg.detect(img, conf=self.conf) if int(np.asarray(d.mask).sum()) >= self.min_px]
        dets = sort_dets(dets)
        self.cache[k] = dets
        return dets

    def save(self, path: str) -> None:
        import pickle

        from .sam_masks import pack
        rows = {k: [(pack(d.mask), d.key, float(d.score), tuple(d.box)) for d in v] for k, v in self.cache.items()}
        with open(path, "wb") as f:
            pickle.dump({"model_id": self.model_id, "imgsz": self.imgsz, "rows": rows}, f)

    @classmethod
    def load(cls, path: str, seg=None, **kw) -> "DetSource":
        import pickle

        from .seg_twin import Det
        from .sam_masks import unpack
        with open(path, "rb") as f:
            d = pickle.load(f)
        cache = {k: [Det(unpack(*m), key, sc, box) for m, key, sc, box in v] for k, v in d["rows"].items()}
        return cls(seg, d["model_id"], d["imgsz"], cache, **kw)


def dets_hash(dets_by_cam: dict) -> str:
    h = hashlib.sha1()
    for cam in sorted(dets_by_cam):
        for d in dets_by_cam[cam]:
            h.update(f"{cam}|{d.key}|{float(d.score):.4f}|{int(np.asarray(d.mask).sum())}".encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------- prediction
class Scene:
    """The cameras of one frame and the believed objects' silhouettes in them (z-buffered; the robot from FK is an
    occluder). Pieces are overridable (tests)."""

    def __init__(self, frame: dict, cams: dict | None = None, require_gl: bool = True):
        from .cameras import cameras_from_state
        from .render import gl

        if require_gl and gl() is None:
            # the CPU fallback orders overlapping objects differently (constant depth): online and offline must both
            # z-buffer on GL (reviewer M5)
            raise RuntimeError("redetect needs GL silhouettes (DOJO_GL=1, EGL); refusing the CPU fallback")
        self.frame = frame
        self.state = frame["state"]
        self.cams = cams if cams is not None else cameras_from_state(self.state)
        self._robot: dict = {}

    def robot(self, cam_name: str) -> np.ndarray:
        if cam_name not in self._robot:
            from .render import robot_mask

            self._robot[cam_name] = robot_mask(self.cams[cam_name], self.state, dilate_px=2, closed_jaws=False)
        return self._robot[cam_name]

    def depth(self, cam_name: str, key: str, p7) -> np.ndarray:
        """Depth of the object's surface per pixel (0 = not covered)."""
        from .cameras import pose_to_T
        from .render import gl, load_asset, silhouette, transform

        cam = self.cams[cam_name]
        a = load_asset(key)
        T = pose_to_T(p7[:3], p7[3:7])
        g = gl()
        if g is not None:
            return g.render(cam, [(key, a.Vg, a.Fg, T)], depth=True).astype(np.float32)
        s = silhouette(cam, transform(a.Vs, T), a.Fs)
        d = float(np.linalg.norm(T[:3, 3] - cam.center))
        return (s > 0).astype(np.float32) * d

    def silhouette(self, cam_name: str, key: str, p7) -> np.ndarray:
        return (self.depth(cam_name, key, p7) > 0).astype(np.uint8)

    def visible(self, objs: list[dict]) -> dict:
        """{cam: {label: (visible mask, n_sil, n_vis, robot occlusion share)}}: front-most per pixel among ``objs``."""
        out = {}
        for cam in sorted(self.cams):
            if cam not in self.frame["images"]:
                continue
            rob = self.robot(cam)
            ds = {}
            for o in objs:
                try:
                    ds[o["label"]] = self.depth(cam, o["key"], o["pose7"])
                except Exception:  # noqa: BLE001 (no mesh for this key: invisible to the check)
                    continue
            if not ds:
                out[cam] = {}
                continue
            labs = sorted(ds)
            D = np.stack([np.where(ds[lab] > 0, ds[lab], np.inf) for lab in labs])
            front = np.argmin(D, axis=0)
            anyc = np.isfinite(D.min(axis=0))
            per = {}
            for i, lab in enumerate(labs):
                s = ds[lab] > 0
                n_sil = int(s.sum())
                if n_sil == 0:
                    continue
                vis = s & anyc & (front == i) & (rob == 0)
                occ = float((s & (rob > 0)).sum()) / n_sil
                per[lab] = (vis.astype(np.uint8), n_sil, int(vis.sum()), round(occ, 3))
            out[cam] = per
        return out


# ------------------------------------------------------------------------------------------------- association
def associate(objs: list[dict], dets_by_cam: dict, vis: dict, scene: Scene, cfg: RDConfig) -> tuple[dict, dict]:
    """Per camera, one Hungarian over objects x detections. Returns ({label: {cam: (det index, iou, exact)}},
    {cam: set(used det indices)})."""
    from scipy.optimize import linear_sum_assignment

    by_label = {o["label"]: o for o in objs}
    assoc: dict = {o["label"]: {} for o in objs}
    used: dict = {}
    for cam in sorted(dets_by_cam):
        dets = dets_by_cam[cam]
        pv = vis.get(cam, {})
        labs = sorted(lab for lab, v in pv.items() if v[2] >= cfg.min_vis_assoc_px and lab in by_label)
        used[cam] = set()
        if not labs or not dets:
            continue
        BIG = 1e3
        C = np.full((len(labs), len(dets)), BIG)
        info = {}
        for i, lab in enumerate(labs):
            o = by_label[lab]
            m, n_sil, n_vis, occ = pv[lab]
            for j, d in enumerate(dets):
                if category(d.key) != category(o["key"]) or float(d.score) < cfg.explain_min_score:
                    continue
                v = _iou(m, d.mask)
                # mask IoU only, never a centroid distance (reviewer: an elongated object's far centre with an
                # overlapping footprint locked the recentre chain onto its neighbour)
                if v < cfg.gate_iou:
                    continue
                exact = d.key == o["key"]
                C[i, j] = 1.0 - v - (cfg.exact_bonus if exact else 0.0)
                info[(i, j)] = (round(v, 4), exact)
        r, c = linear_sum_assignment(C)
        for i, j in zip(r, c):
            if C[i, j] >= BIG:
                continue
            v, exact = info[(i, j)]
            assoc[labs[i]][cam] = (int(j), v, bool(exact))
            used[cam].add(int(j))
    return assoc, used


# ------------------------------------------------------------------------------------------------- fits
def fit_in_mask(key: str, cam_name: str, scene: Scene, mask: np.ndarray, cfg: RDConfig) -> list[tuple[list, float]]:
    """The asset inside one detection mask, region-free: every catalogue resting shape (lying too), a full yaw grid at
    the mask's ray-plane centre, then a refine on the mask (no origin bounds, no yaw range, no group prior). Returns
    [(pose7, mask IoU)] best first, one per resting shape."""
    from .inverse_layout import Blob, coarse_fit, refine, shape_models
    from .segment import ViewMasks

    cam = scene.cams[cam_name]
    rob = scene.robot(cam_name)
    M = ((np.asarray(mask) > 0) & (rob == 0)).astype(np.uint8)
    n = int(M.sum())
    if n < cfg.min_det_px:
        return []
    ys, xs = np.nonzero(M)
    blob = Blob(cam_name, M, n, np.array([xs.mean() + 0.5, ys.mean() + 0.5]),
                (int(xs.min()), int(ys.min()), int(np.ptp(xs) + 1), int(np.ptp(ys) + 1)))
    kn = (1 - rob).astype(np.uint8) | M
    views = {cam_name: ViewMasks(cam, scene.frame["images"][cam_name], M, kn, np.zeros(M.shape, np.float32), rob)}
    out = []
    try:
        sms = shape_models(key)
    except Exception:  # noqa: BLE001
        return []
    from .cameras import TABLE_Z
    for sm in sms:
        # the coarse fit assumes the object on the table plane: a ray that meets it far away (a wrist looking along
        # the table) gives a huge ROI (fasten_screws: minutes per fit), so such a view does not fit this shape
        X0 = cam.ray_plane(blob.centroid[None, :], TABLE_Z + sm.height / 2)[0]
        if not np.isfinite(X0).all() or float(np.linalg.norm(X0 - cam.center)) > cfg.max_fit_range_m:
            continue
        ext = float(np.max(sm.asset.extents))
        uv, _ = cam.project(np.array([X0, X0 + [ext, 0, 0], X0 + [0, ext, 0]]))
        if float(np.max(np.abs(uv[1:] - uv[0]))) > 2 * max(cam.w, cam.h):
            continue
        best = None
        for x, y, yaw, _ in coarse_fit(sm, blob, views, cfg.yaw_step_deg, None)[:2]:
            x2, y2, yaw2, _, _ = refine(sm, x, y, yaw, views, None, xy_step=0.004, yaw_step=0.08, maxiter=cfg.nm_iters,
                                        bounds=None)
            T = sm.pose(x2, y2, yaw2)
            v = _iou(sm.sil(cam, T), M)
            if best is None or v > best[1]:
                best = (pose7(T), round(v, 4))
        if best is not None:
            out.append(best)
    return sorted(out, key=lambda t: -t[1])


def support(key: str, p7, scene: Scene, dets_by_cam: dict, exclude: dict | None = None,
            min_score: float = 0.5) -> dict:
    """{cam: [best IoU of the pose's silhouette with a same-category detection there, silhouette px, visible fraction,
    that detection's score]} for every camera where the pose projects. Only detections at ``min_score`` or above count
    (a write never rests on one a re-encoding can flip); ``exclude`` {cam: det indices explained by other objects}.
    Visible fraction: the silhouette's share outside the robot times its vertices' share inside the image."""
    from .cameras import pose_to_T
    from .refine import truncated_frac
    from .render import load_asset, transform

    out = {}
    for cam in sorted(scene.cams):
        if cam not in scene.frame["images"]:
            continue
        try:
            s_full = scene.silhouette(cam, key, p7)
            a = load_asset(key)
            trunc = truncated_frac(scene.cams[cam], transform(a.Vs, pose_to_T(p7[:3], p7[3:7])))
        except Exception:  # noqa: BLE001
            continue
        n_full = int(s_full.sum())
        s = s_full & (1 - scene.robot(cam))
        n = int(s.sum())
        if n == 0:
            continue
        vis = (n / max(n_full, 1)) * (1.0 - trunc)
        best, sc = 0.0, None
        for j, d in enumerate(dets_by_cam.get(cam, [])):
            if (category(d.key) != category(key) or j in (exclude or {}).get(cam, ())
                    or float(d.score) < min_score):
                continue
            v = _iou(s, d.mask)
            if v > best:
                best, sc = v, round(float(d.score), 3)
        out[cam] = [round(best, 4), n, round(vis, 3), sc]
    return out


def footprint(key: str, p7) -> np.ndarray:
    """World-xy convex hull of the asset's mesh at a pose (float32, K x 2)."""
    import cv2

    from .cameras import pose_to_T
    from .render import load_asset, transform

    W = transform(load_asset(key).Vs, pose_to_T(p7[:3], p7[3:7]))[:, :2].astype(np.float32)
    return cv2.convexHull(W).reshape(-1, 2)


def footprint_overlap(key: str, p7, others: list[dict]) -> list:
    """[label, overlap / the smaller footprint's area] of the most overlapped other object (footprint hulls in xy)."""
    import cv2

    try:
        a = footprint(key, p7)
    except Exception:  # noqa: BLE001
        return [None, 0.0]
    aa = float(cv2.contourArea(a))
    best = [None, 0.0]
    for o in sorted(others, key=lambda x: x["label"]):
        try:
            b = footprint(o["key"], o["pose7"])
        except Exception:  # noqa: BLE001
            continue
        inter, _ = cv2.intersectConvexConvex(a, b)
        sm = min(aa, float(cv2.contourArea(b)))
        f = float(inter) / sm if sm > 1e-9 else 0.0
        if f > best[1]:
            best = [o["label"], round(f, 3)]
    return best


# ------------------------------------------------------------------------------------------------- contact
def contact_suspects(q_path: dict, objs: list[dict], held: dict, cfg: RDConfig, model=None,
                     held_path: dict | None = None) -> list[str]:
    """Labels whose bbox (+ ``contact_margin_m``) the arm swept since the last tick: every link's mesh points (FK on the
    joints of ``q_path`` {arm: [q6, ...]}: the executed actions since the last tick) and the held object's mesh points
    riding the TCP (``held_path`` {arm: {"key", "T_tcp_obj"}}). Held objects are not suspects."""
    from .cameras import pose_to_T
    from .render import load_asset, robot_link_meshes
    from .robot_model import X5Model

    model = model or X5Model.default()
    meshes = robot_link_meshes()
    rng = np.random.default_rng(0)
    pts_local = {ln: V[rng.choice(len(V), min(len(V), cfg.contact_pts_per_link), replace=False)]
                 for ln, (V, F) in sorted(meshes.items())}
    P = []
    hp = {}
    for arm, h in sorted((held_path or {}).items()):
        try:
            Vh = np.asarray(load_asset(h["key"]).Vs, float)
            hp[arm] = (Vh[rng.choice(len(Vh), min(len(Vh), cfg.contact_pts_per_link), replace=False)],
                       np.asarray(h["T_tcp_obj"], float).reshape(4, 4))
        except Exception:  # noqa: BLE001
            continue
    from .held import tcp_T
    for arm in sorted(q_path):
        for q in q_path[arm]:
            q6 = np.asarray(q, float)[:6]
            links = model.fk_world(arm, q6)
            for ln, Vl in pts_local.items():
                if ln in links:
                    T = links[ln]
                    P.append(Vl @ T[:3, :3].T + T[:3, 3])
            if arm in hp:
                Vh, T_rel = hp[arm]
                T = tcp_T(arm, q6, model) @ T_rel
                P.append(Vh @ T[:3, :3].T + T[:3, 3])
    if not P:
        return []
    P = np.concatenate(P)
    held_labels = set(held.values())
    out = []
    for o in objs:
        if o["label"] in held_labels:
            continue
        try:
            a = load_asset(o["key"])
        except Exception:  # noqa: BLE001
            continue
        T = pose_to_T(o["pose7"][:3], o["pose7"][3:7])
        lo, hi = np.asarray(a.info["bbox_min"], float), np.asarray(a.info["bbox_max"], float)
        loc = (P - T[:3, 3]) @ T[:3, :3]
        inside = np.all((loc >= lo - cfg.contact_margin_m) & (loc <= hi + cfg.contact_margin_m), axis=1)
        if inside.any():
            out.append(o["label"])
    return sorted(out)


# ------------------------------------------------------------------------------------------------- lift identity
def lift_identity(scene: Scene, arm: str, held_label: str, objs: list[dict], dets_by_cam: dict) -> dict:
    """The detection between the holding arm's jaws in its wrist view (the mask covering the image region below the
    fingertips the most): its class against the held label's asset. Log-only (the runtime decides)."""
    cam = f"cam_{arm}_wrist"
    dets = dets_by_cam.get(cam, [])
    by_label = {o["label"]: o for o in objs}
    if held_label not in by_label or not dets:
        return {"arm": arm, "label": held_label, "det": None}
    held = by_label[held_label]
    try:
        s = scene.silhouette(cam, held["key"], held["pose7"])
    except Exception:  # noqa: BLE001
        s = None
    best = None
    for j, d in enumerate(dets):
        v = _iou(s, d.mask) if s is not None else 0.0
        if best is None or v > best[1]:
            best = (j, v)
    j, v = best
    d = dets[j]
    other = [lab for lab, o in sorted(by_label.items()) if lab != held_label and o["key"] == d.key]
    return {"arm": arm, "label": held_label, "det": {"key": d.key, "score": round(float(d.score), 3),
            "px": int(np.asarray(d.mask).sum()), "iou_with_belief": round(v, 3)},
            "agrees": category(d.key) == category(held["key"]), "exact": d.key == held["key"], "other_labels": other}


# ------------------------------------------------------------------------------------------------- the tick
def _check_no_gt(frame: dict, req: dict) -> None:
    for k in GT_KEYS:
        if k in frame or k in (frame.get("state") or {}) or k in req:
            raise ValueError(f"redetect: '{k}' in its input (GT / scenario keys never reach the tick)")


def redetect(frame: dict, req: dict, src: DetSource, cfg: RDConfig | None = None, scene: Scene | None = None) -> dict:
    """One tick's perception. ``req``:
      objects: [{label, key, pose7, active: reason | None, seeds: [{pose7, src}], small: bool}] (held ones too),
      held: {arm: label}, q_path: {arm: [q6, ...]} since the last tick, lift: [arm], corridor: [[x, y, z], ...] | None
    Returns {dets_hash, n_dets, assoc, active, hyps, assignment, added, lift, contact, time_s}."""
    cfg = cfg or RDConfig()
    _check_no_gt(frame, req)
    t0 = time.perf_counter()
    scene = scene or Scene(frame)
    held = dict(req.get("held") or {})
    held_labels = set(held.values())
    objs = sorted((dict(o) for o in req["objects"]), key=lambda o: o["label"])
    free = [o for o in objs if o["label"] not in held_labels]
    dets = {cam: src.detect(frame["images"][cam]) for cam in sorted(frame["images"]) if cam in CAMS}
    tm = {"detect": time.perf_counter() - t0}
    vis = scene.visible(objs)
    tm["visible"] = time.perf_counter() - t0
    assoc, used = associate(free, dets, vis, scene, cfg)
    tm["assoc"] = time.perf_counter() - t0
    active = {o["label"]: o["active"] for o in free if o.get("active")}
    contact = (contact_suspects(req.get("q_path") or {}, free, held, cfg, held_path=req.get("held_path"))
               if req.get("q_path") else [])
    tm["contact"] = time.perf_counter() - t0
    for lab in contact:
        active.setdefault(lab, "contact")
    # "missing": clearly visible where believed, associated nowhere
    for o in free:
        lab = o["label"]
        if lab in active or assoc.get(lab):
            continue
        big = [cam for cam, pv in vis.items() if lab in pv and pv[lab][3] <= cfg.max_robot_occ
               and pv[lab][2] >= cfg.promote_min_px]
        mid = [cam for cam, pv in vis.items() if lab in pv and pv[lab][3] <= cfg.max_robot_occ
               and pv[lab][2] >= cfg.promote_two_cam_px]
        if big or len(mid) >= 2:
            active[lab] = "missing"
    # unexplained detections (not associated with any object, held ones excluded by their own association below)
    held_objs = [o for o in objs if o["label"] in held_labels]
    if held_objs:
        h_assoc, h_used = associate(held_objs, dets, vis, scene, cfg)
        for cam, s in h_used.items():
            used.setdefault(cam, set()).update(s)
    free_dets = {cam: [j for j in range(len(dets[cam])) if j not in used.get(cam, set())] for cam in dets}
    # hypotheses per active object
    hyps: dict = {}
    n_fit = 0
    det_hyps = []                                         # (label, cam, det j, hyp index)
    by_label = {o["label"]: o for o in free}
    # the fit budget by priority (reviewer M3): the triggers' objects, then "missing", contact, touched
    prio = {"hover": 0, "close_empty": 0, "release": 0, "slip": 0, "missing": 1, "contact": 2, "touched": 3}
    for lab in sorted(active, key=lambda lb: (prio.get(active[lb], 4), lb)):
        o = by_label[lab]
        hs = [{"src": "incumbent", "pose7": list(o["pose7"]),
               "support": support(o["key"], o["pose7"], scene, dets, used_others(used, assoc, lab),
                                  cfg.seed_min_score)}]
        for sd in o.get("seeds") or []:
            hs.append({"src": sd.get("src", "seed"), "pose7": list(sd["pose7"]),
                       "support": support(o["key"], sd["pose7"], scene, dets, used_others(used, assoc, lab),
                                          cfg.seed_min_score)})
        # local only: an object a camera associates, and a "touched" object at a later tick (its release-time read was
        # the moment to find where it went; later it moves only by a contact, which has its own trigger)
        explained = bool(assoc.get(lab)) or active.get(lab) == "touched"
        # an object a camera associates is near its belief: refit inside its own mask (a few mm .. cm), and never jump
        # to a far unexplained mask (ttt: a piece placed 24 mm off was "relocated" 270 mm onto another piece of the pool)
        for cam, (j, v_, _) in sorted((assoc.get(lab) or {}).items()):
            if n_fit >= cfg.max_fits:
                break
            if float(dets[cam][j].score) < cfg.seed_min_score:
                continue
            n_fit += 1
            fits = fit_in_mask(o["key"], cam, scene, dets[cam][j].mask, cfg)
            if fits:
                p7, v = fits[0]
                if float(np.linalg.norm(np.asarray(p7[:2]) - np.asarray(o["pose7"][:2]))) > cfg.explained_max_jump_m:
                    continue                    # its own mask is near its belief: a far fit is the plane's error
                hs.append({"src": f"own:{cam}:{j}", "pose7": p7, "fit_iou": v,
                           "support": support(o["key"], p7, scene, dets, used_others(used, assoc, lab),
                                              cfg.seed_min_score)})
        for cam in sorted(free_dets):
            for j in free_dets[cam]:
                d = dets[cam][j]
                # an unexplained mask seeds a relocation only with the object's own asset class: the category gates the
                # association, but a same-category mask of another asset is another object (ttt: a player piece moved
                # onto an opponent's piece of the pool)
                if d.key != o["key"] or float(d.score) < cfg.seed_min_score or n_fit >= cfg.max_fits:
                    continue
                n_fit += 1
                fits = fit_in_mask(o["key"], cam, scene, d.mask, cfg)
                if not fits:
                    continue
                p7, v = fits[0]
                if explained and float(np.linalg.norm(np.asarray(p7[:2]) - np.asarray(o["pose7"][:2]))) > cfg.explained_max_jump_m:
                    continue
                hs.append({"src": f"det:{cam}:{j}", "pose7": p7, "fit_iou": v, "det_key": d.key,
                           "det_score": round(float(d.score), 3),
                           "support": support(o["key"], p7, scene, dets, used_others(used, assoc, lab),
                                              cfg.seed_min_score)})
                det_hyps.append((lab, cam, j, len(hs) - 1))
        for h in hs[1:]:                                    # footprint overlap with a believed, not-active object
            h["overlap"] = footprint_overlap(o["key"], h["pose7"], [x for x in free if x["label"] != lab
                                                                    and x["label"] not in active])
            try:
                h["footprint"] = [[round(float(a), 5), round(float(b), 5)] for a, b in footprint(o["key"], h["pose7"])]
            except Exception:  # noqa: BLE001
                h["footprint"] = None
        hyps[lab] = hs
    tm["fits"] = time.perf_counter() - t0
    assignment = joint_assignment(hyps, det_hyps)
    # added: unexplained detections near the corridor no active object took
    taken = {(cam, j) for lab, (cam, j, _) in assignment.items()}
    added = []
    cor = req.get("corridor")
    if cor is not None:
        P = np.asarray(cor, float).reshape(-1, 3)
        for cam in sorted(free_dets):
            for j in free_dets[cam]:
                if (cam, j) in taken or n_fit >= cfg.max_fits or float(dets[cam][j].score) < cfg.seed_min_score:
                    continue
                d = dets[cam][j]
                n_fit += 1
                fits = fit_in_mask(d.key, cam, scene, d.mask, cfg)
                if not fits:
                    continue
                p7, v = fits[0]
                dist = float(np.min(np.linalg.norm(P[:, :2] - np.asarray(p7[:2])[None], axis=1)))
                if dist > cfg.added_max_corridor_m:
                    continue
                added.append({"cam": cam, "det": j, "key": d.key, "score": round(float(d.score), 3), "pose7": p7,
                              "fit_iou": v, "corridor_m": round(dist, 4),
                              "support": support(d.key, p7, scene, dets, None, cfg.seed_min_score)})
    lift = [lift_identity(scene, arm, held[arm], objs, dets) for arm in sorted(req.get("lift") or []) if arm in held]
    tm["total"] = time.perf_counter() - t0
    fps = {}
    for o in free:                                        # every object's footprint now (decide's overlap with writes)
        try:
            fps[o["label"]] = [[round(float(a), 5), round(float(b), 5)] for a, b in footprint(o["key"], o["pose7"])]
        except Exception:  # noqa: BLE001
            continue
    return {"dets_hash": dets_hash(dets), "n_dets": {c: len(v) for c, v in dets.items()}, "footprints": fps,
            "assoc": {lab: {c: list(v) for c, v in a.items()} for lab, a in assoc.items()},
            "active": active, "hyps": hyps, "assignment": {lab: list(v) for lab, v in assignment.items()},
            "added": added, "lift": lift, "contact": contact, "n_fits": n_fit,
            "time_s": {k: round(v, 3) for k, v in tm.items()}}


def used_others(used: dict, assoc: dict, label: str) -> dict:
    """Detections explained by OTHER objects (not available as this object's support)."""
    mine = {cam: j for cam, (j, _, _) in (assoc.get(label) or {}).items()}
    return {cam: {j for j in s if mine.get(cam) != j} for cam, s in used.items()}


def joint_assignment(hyps: dict, det_hyps: list) -> dict:
    """One assignment over active objects x detection-derived hypotheses (a detection feeds at most one object), by the
    hypothesis' best support; an object may take none. Returns {label: (cam, det j, hyp index)}."""
    from scipy.optimize import linear_sum_assignment

    if not det_hyps:
        return {}
    labs = sorted({lab for lab, *_ in det_hyps})
    keys = sorted({(cam, j) for _, cam, j, _ in det_hyps})
    BIG = 1e3
    C = np.full((len(labs), len(keys) + len(labs)), BIG)
    pick = {}
    for lab, cam, j, hi in det_hyps:
        h = hyps[lab][hi]
        s = max((v[0] for v in h["support"].values()), default=0.0)
        i, k = labs.index(lab), keys.index((cam, j))
        if 1.0 - s < C[i, k]:
            C[i, k] = 1.0 - s
            pick[(i, k)] = hi
    for i in range(len(labs)):                              # "none" columns
        C[i, len(keys) + i] = 0.95
    r, c = linear_sum_assignment(C)
    out = {}
    for i, k in zip(r, c):
        if k < len(keys) and C[i, k] < BIG:
            cam, j = keys[k]
            out[labs[i]] = (cam, int(j), int(pick[(i, k)]))
    return out


__all__ = ["RDConfig", "DetSource", "Scene", "redetect", "fit_in_mask", "associate", "support", "contact_suspects",
           "joint_assignment", "category", "image_hash"]
