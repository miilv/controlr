"""Imitation twins by appearance: a joint one-to-one assignment of the demonstrator's objects to ours from several
image cues, with explicit ties (IMIT2, ``RuntimeConfig.imitate_twin_cues``; default off).

    twin_costs(frame, objects, demo, ours, cfg=None) -> {"cost": {demo: {ours: c}}, "cues": {...}, "assign": {d: o},
                                                         "margin": {d: m}, "ties": [d, ...], "views": {...}}

``objects``: the frame-0 belief, [{"label", "key", "pos", "quat"}]; ``demo`` / ``ours``: labels on each side of the
table. Every cue comes from the frame's own pixels, read at each object's believed pose (its rendered silhouette,
front-most among the believed objects, our robot removed, table-coloured pixels dropped):

- colour: an L histogram and an (a, b) chroma histogram of the object's pixels (Bhattacharyya distance), per camera
  that sees both objects with at least ``min_px`` pixels, averaged over those cameras;
- size: the visible area corrected to a common viewing distance (pixels x depth^2 / (fx fy), m^2), log-ratio;
- category: the believed identity of each side (same asset 0, same category 0.5, else 1), a weak prior;
- height: the believed top height above the table (m), a weak cue for dark objects whose colour is near the table's.

The assignment minimises the total cost over all pairs at once (Hungarian). A pair's margin is the least extra cost of
any two-pair swap that moves it; a pair whose margin is under ``tie_eps`` is a tie: the caller decides what to do with
it (look again, or stop the copy before it). The correspondence is decided by appearance only (D-C17): no layout
relation, quota or GT name is read.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULTS = {
    "min_px": 40,            # a camera's read of an object counts with at least this many object pixels
    "w_color": 1.0,          # Bhattacharyya distance (0..1) of the colour histograms
    "w_area": 0.2,           # |log(area_d / area_o)|
    "w_cat": 0.0,            # believed identity prior: 0 / 0.5 / 1 (off: it hurt on the dev frames)
    "w_height": 0.0,         # |top_d - top_o| in m, believed tops (off: circular with the believed identity)
    "tie_eps": 0.05,         # a pair whose best swap costs less extra than this is a tie
    "table_dist": 14.0,      # Lab distance to the local table colour under which a pixel is table, not object
    "ring_px": 6,            # width of the ring around a silhouette that samples the local table colour
    "dilate_px": 3,          # the believed silhouette grown by this much (a pose a few mm off)
    "cams": ("cam_head", "cam_left_wrist", "cam_right_wrist"),
    "cam_mode": "best_all",  # best_all: mean of best and all | shared: mean over cameras seeing both | best: each object's largest view, across cameras
}

L_BINS = 8
AB_BINS = 6
AB_RANGE = (-48.0, 48.0)


def _lab(img: np.ndarray) -> np.ndarray:
    import cv2

    lab = cv2.cvtColor(np.ascontiguousarray(img[..., :3]).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def histogram(lab_px: np.ndarray) -> np.ndarray:
    """L (``L_BINS``) and (a, b) (``AB_BINS`` x ``AB_BINS``) histograms of N x 3 Lab pixels, each normalised to 1,
    concatenated (sum 2)."""
    if len(lab_px) == 0:
        return np.zeros(L_BINS + AB_BINS * AB_BINS)
    hl, _ = np.histogram(np.clip(lab_px[:, 0], 0, 99.999), bins=L_BINS, range=(0.0, 100.0))
    a = np.clip(lab_px[:, 1], AB_RANGE[0], AB_RANGE[1] - 1e-3)
    b = np.clip(lab_px[:, 2], AB_RANGE[0], AB_RANGE[1] - 1e-3)
    hab, _, _ = np.histogram2d(a, b, bins=AB_BINS, range=(AB_RANGE, AB_RANGE))
    hl = hl / max(hl.sum(), 1)
    hab = hab.ravel() / max(hab.sum(), 1)
    return np.concatenate([hl, hab])


def bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    """Mean Bhattacharyya distance (sqrt(1 - BC), 0 same .. 1 disjoint) of the L part and the (a, b) part."""
    out = []
    for s in (slice(0, L_BINS), slice(L_BINS, None)):
        bc = float(np.sum(np.sqrt(np.clip(h1[s], 0, None) * np.clip(h2[s], 0, None))))
        out.append(math.sqrt(max(0.0, 1.0 - min(1.0, bc))))
    return 0.5 * (out[0] + out[1])


def _cat(key: str) -> str:
    parts = str(key).split("/")
    return parts[1] if len(parts) > 2 else str(key)


def prior_cost(ka: str, kb: str) -> float:
    return 0.0 if ka == kb else (0.5 if _cat(ka) == _cat(kb) else 1.0)


def object_views(images: dict, state: dict, objects: list[dict], cfg: dict, cams=None) -> dict:
    """Per object label and camera: {"n_px", "area_m2", "hist"} from its believed silhouette (front-most among the
    believed objects, our robot removed, table pixels dropped)."""
    import cv2

    from .cameras import cameras_from_state, pose_to_T
    from .inverse_layout import shape_models
    from .render import gl, robot_mask

    st = {k: np.asarray(v, float) for k, v in state.items()}
    cams = cams or {c: cam for c, cam in cameras_from_state(st).items() if c in images and c in cfg["cams"]}
    sms, Ts = [], []
    for o in objects:
        try:
            sms.append(shape_models(o["key"])[0])
        except Exception:  # noqa: BLE001 (not in the catalogue: no read)
            sms.append(None)
        Ts.append(pose_to_T(np.asarray(o["pos"], float), np.asarray(o.get("quat") or [1, 0, 0, 0], float)))
    g = gl()
    out: dict = {o["label"]: {} for o in objects}
    k_d = np.ones((2 * cfg["dilate_px"] + 1,) * 2, np.uint8)
    k_r = np.ones((2 * cfg["ring_px"] + 1,) * 2, np.uint8)
    for cname, cam in cams.items():
        img = np.asarray(images[cname])
        lab = _lab(img)
        rob = robot_mask(cam, st, dilate_px=3).astype(bool)
        depth = []
        for sm, T in zip(sms, Ts):
            if sm is None:
                depth.append(np.full((cam.h, cam.w), np.inf))
            elif g is not None:
                d = g.render(cam, [(sm.key, sm.asset.Vg, sm.asset.Fg, T)], depth=True)
                depth.append(np.where(d > 0, d, np.inf))
            else:
                m = sm.sil(cam, T).astype(bool)
                z = float(np.linalg.norm(T[:3, 3] - cam.center))
                depth.append(np.where(m, z, np.inf))
        if not depth:
            continue
        D = np.stack(depth)
        front = np.argmin(D, axis=0)
        has = np.isfinite(D).any(axis=0)
        any_obj = has.astype(np.uint8)
        for i, o in enumerate(objects):
            if sms[i] is None:
                continue
            own = (has & (front == i)).astype(np.uint8)
            if not own.any():
                continue
            grown = cv2.dilate(own, k_d).astype(bool) & ~(any_obj.astype(bool) & ~own.astype(bool))
            ring = cv2.dilate(own, k_r).astype(bool) & ~cv2.dilate(any_obj, k_d).astype(bool) & ~rob
            m = grown & ~rob
            if ring.sum() >= 10:
                tab = np.median(lab[ring], axis=0)
                m &= np.linalg.norm(lab - tab, axis=-1) > cfg["table_dist"]
            n = int(m.sum())
            if n == 0:
                continue
            z = float(np.median(D[i][own.astype(bool) & np.isfinite(D[i])])) if np.isfinite(D[i]).any() else 1.0
            out[o["label"]][cname] = {"n_px": n, "area_m2": n * z * z / float(cam.K[0, 0] * cam.K[1, 1]),
                                      "hist": histogram(lab[m])}
    return out


def _top(o: dict) -> float | None:
    try:
        from .cameras import pose_to_T
        from .inverse_layout import shape_models

        sm = shape_models(o["key"])[0]
        T = pose_to_T(np.asarray(o["pos"], float), np.asarray(o.get("quat") or [1, 0, 0, 0], float))
        V = np.asarray(sm.asset.V if getattr(sm.asset, "V", None) is not None else sm.asset.Vg)
        return float((V @ T[:3, :3].T + T[:3, 3])[:, 2].max())
    except Exception:  # noqa: BLE001
        return None


def cost_matrix(views: dict, objects: list[dict], demo: list[str], ours: list[str], cfg: dict,
                tops: dict | None = None) -> tuple[np.ndarray, dict]:
    by = {o["label"]: o for o in objects}
    C = np.zeros((len(demo), len(ours)))
    cues = {"color": {}, "area": {}, "cat": {}, "height": {}}
    for i, d in enumerate(demo):
        for j, o in enumerate(ours):
            vd, vo = views.get(d, {}), views.get(o, {})
            shared = [c for c in cfg["cams"] if c in vd and c in vo
                      and vd[c]["n_px"] >= cfg["min_px"] and vo[c]["n_px"] >= cfg["min_px"]]
            if cfg.get("cam_mode") == "best_all":
                bd = max((vd[c] for c in cfg["cams"] if c in vd and vd[c]["n_px"] >= cfg["min_px"]),
                         key=lambda v: v["n_px"], default=None)
                bo = max((vo[c] for c in cfg["cams"] if c in vo and vo[c]["n_px"] >= cfg["min_px"]),
                         key=lambda v: v["n_px"], default=None)
                ds = [vd[c]["hist"] for c in cfg["cams"] if c in vd and vd[c]["n_px"] >= cfg["min_px"]]
                os_ = [vo[c]["hist"] for c in cfg["cams"] if c in vo and vo[c]["n_px"] >= cfg["min_px"]]
                col = 0.5 * (bhattacharyya(bd["hist"], bo["hist"]) + float(np.mean([bhattacharyya(a, b) for a in ds for b in os_]))) \
                    if bd is not None and bo is not None else 0.5
            elif cfg.get("cam_mode") == "best":
                bd = max((vd[c] for c in cfg["cams"] if c in vd and vd[c]["n_px"] >= cfg["min_px"]),
                         key=lambda v: v["n_px"], default=None)
                bo = max((vo[c] for c in cfg["cams"] if c in vo and vo[c]["n_px"] >= cfg["min_px"]),
                         key=lambda v: v["n_px"], default=None)
                col = bhattacharyya(bd["hist"], bo["hist"]) if bd is not None and bo is not None else 0.5
            elif cfg.get("cam_mode") == "all":
                ds = [vd[c]["hist"] for c in cfg["cams"] if c in vd and vd[c]["n_px"] >= cfg["min_px"]]
                os_ = [vo[c]["hist"] for c in cfg["cams"] if c in vo and vo[c]["n_px"] >= cfg["min_px"]]
                col = float(np.mean([bhattacharyya(a, b) for a in ds for b in os_])) if ds and os_ else 0.5
            else:
                col = float(np.mean([bhattacharyya(vd[c]["hist"], vo[c]["hist"]) for c in shared])) if shared else 0.5
            ad = [vd[c]["area_m2"] for c in cfg["cams"] if c in vd and vd[c]["n_px"] >= cfg["min_px"]]
            ao = [vo[c]["area_m2"] for c in cfg["cams"] if c in vo and vo[c]["n_px"] >= cfg["min_px"]]
            area = abs(math.log(max(ad) / max(ao))) if ad and ao and max(ad) > 0 and max(ao) > 0 else 0.0
            cat = prior_cost(by[d]["key"], by[o]["key"])
            hgt = 0.0
            if tops and tops.get(d) is not None and tops.get(o) is not None:
                hgt = abs(tops[d] - tops[o])
            C[i, j] = cfg["w_color"] * col + cfg["w_area"] * min(area, 1.5) + cfg["w_cat"] * cat + cfg["w_height"] * min(hgt, 0.05)
            k = f"{d}|{o}"
            cues["color"][k], cues["area"][k], cues["cat"][k], cues["height"][k] = (
                round(col, 3), round(area, 3), cat, round(hgt, 4))
    return C, cues


def assign(C: np.ndarray) -> list[tuple[int, int]]:
    try:
        from scipy.optimize import linear_sum_assignment

        ri, ci = linear_sum_assignment(C)
        return [(int(i), int(j)) for i, j in zip(ri, ci)]
    except ImportError:
        import itertools

        n = min(C.shape)
        best = min(itertools.permutations(range(C.shape[1]), n), key=lambda p: sum(C[i, j] for i, j in enumerate(p)))
        return list(enumerate(best))


def margins(C: np.ndarray, pairs: list[tuple[int, int]]) -> dict[int, float]:
    """Per assigned row: the least extra total cost of a two-pair swap (or a move to an unassigned column)."""
    col_of = dict(pairs)
    used = set(col_of.values())
    out = {}
    for i, j in pairs:
        best = math.inf
        for k, l in pairs:
            if k != i:
                best = min(best, C[i, l] + C[k, j] - C[i, j] - C[k, l])
        for l in range(C.shape[1]):
            if l not in used:
                best = min(best, C[i, l] - C[i, j])
        out[i] = float(best) if math.isfinite(best) else 1.0
    return out


def twin_costs(frame: dict, objects: list[dict], demo: list[str], ours: list[str], cfg: dict | None = None) -> dict:
    c = dict(DEFAULTS)
    c.update(cfg or {})
    by = {o["label"]: o for o in objects}
    demo = [d for d in demo if d in by]
    ours = [o for o in ours if o in by]
    if not demo or not ours:
        return {"cost": {}, "assign": {}, "margin": {}, "ties": [], "cues": {}, "views": {}}
    views = object_views(frame["images"], frame["state"], objects, c)
    tops = {o["label"]: _top(o) for o in objects} if c["w_height"] > 0 else None
    C, cues = cost_matrix(views, objects, demo, ours, c, tops)
    pairs = assign(C)
    mg = margins(C, pairs)
    return {"cost": {d: {o: round(float(C[i, j]), 4) for j, o in enumerate(ours)} for i, d in enumerate(demo)},
            "assign": {demo[i]: ours[j] for i, j in pairs},
            "margin": {demo[i]: round(mg[i], 4) for i, _ in pairs},
            "ties": [demo[i] for i, _ in pairs if mg[i] < c["tie_eps"]],
            "cues": cues,
            "views": {l: {cn: int(v["n_px"]) for cn, v in vv.items()} for l, vv in views.items()}}
