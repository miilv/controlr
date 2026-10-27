"""Hybrid H2: the agent says WHAT and roughly WHERE, B1's render-and-compare says exactly where.

For every object of the agent's answer that is new or moved by the agent (no B1 object of the same category within
``same_mm``), or that lacks an orientation, fit the catalogue shape locally in the current frames (head + wrists,
robot self-masked) from the agent's xy: Nelder-Mead over (x, y, yaw) from several yaw starts (the agent's yaw and its
flip when given), over the candidate assets of the object's category in the task's vocabulary. Accept the best fit when
its silhouette loss is below ``max_loss`` and it stays within ``max_shift_mm`` of the agent's point; else keep the
agent's pose. Worker side (needs the perception extra + catalogue + GL or the CPU silhouettes).
"""
from __future__ import annotations

import math

import numpy as np


def _cat(o: dict) -> str | None:
    a = str(o.get("asset") or "").split("/")
    return a[1] if len(a) == 3 else o.get("category")


def _extents(key: str):
    from ..render import load_asset

    try:
        return sorted(load_asset(key).info["extents"])
    except Exception:
        return None


def _candidates(o: dict, b1_objs: list[dict], task: str | None, k_max: int = 3) -> list[str]:
    if o.get("asset"):
        return [o["asset"]]
    keys = _candidates_all(o, b1_objs, task)
    size = o.get("size")
    if size and len(size) == 3 and len(keys) > k_max:  # the agent's measured size ranks the shapes
        sz = sorted(float(v) for v in size)

        def d(k):
            e = _extents(k)
            return 9.0 if e is None else sum(abs(a - b) / max(b, 1e-3) for a, b in zip(sz, e))
        keys = sorted(keys, key=d)
    return keys[:k_max]


def _candidates_all(o: dict, b1_objs: list[dict], task: str | None) -> list[str]:
    cat = _cat(o)
    lab = str(o.get("label") or "")
    keys = []
    for b in b1_objs:  # B1's own reading of the same label / category
        if b.get("asset") and (b.get("label") == lab or (cat and _cat(b) == cat)) and b["asset"] not in keys:
            keys.append(b["asset"])
    if task:
        from ..inverse_layout import label_groups

        for g in label_groups(task):
            for k in g["candidates"]:
                c = k.split("/")[1]
                if (cat and c == cat) or (lab and (lab in g["labels"] or c in lab.lower())):
                    if k not in keys:
                        keys.append(k)
    return keys


def seeded_refine(images: dict, state: dict, cams: dict, task: str | None, objects: list[dict], b1_objs: list[dict],
                  bgs: dict, same_mm: float = 8.0, max_shift_mm: float = 40.0, max_loss: float = 0.5,
                  n_yaw: int = 8, orient_loss: float = 0.75, orient_shift_mm: float = 15.0, shift_frac: float = 0.45) -> tuple[list[dict], list[dict]]:
    """-> (objects with refined poses where accepted, per-object info)."""
    from ..cameras import R_to_quat_wxyz, pose_to_T
    from ..inverse_layout import EstimatorConfig, _views_subset, refine as nm_refine, shape_models
    from ..segment import segment_views

    cfg = EstimatorConfig()
    views = _views_subset(segment_views(images, cams, state, bgs, thr=cfg.fg_thr), cfg)
    out, infos = [dict(o) for o in objects], []

    def todo(o):
        """New or moved by the agent: no B1 object (any label) within same_mm. Orientation-less objects that sit on
        a B1 object take that object's asset and pose (the agent re-labelled it, it did not move it)."""
        for b in b1_objs:
            if math.hypot(b["pos"][0] - o["pos"][0], b["pos"][1] - o["pos"][1]) * 1000 <= same_mm:
                if not o.get("quat") and b.get("quat"):
                    o.update({k: b[k] for k in ("pos", "quat", "asset", "size", "top_z") if b.get(k) is not None})
                return False
        return True

    shapes = {}

    def sm_of(o):
        k = o.get("asset")
        if not k or not o.get("quat"):
            return None
        try:
            T = pose_to_T(o["pos"], o["quat"])
            from ..refine import _shape_for

            return _shape_for(k, T[:3, :3], None), T
        except Exception:
            return None

    for i, o in enumerate(out):
        if not todo(o):
            continue
        info = {"i": i, "label": o.get("label"), "accepted": False}
        others = {}
        for name, v in views.items():
            m = np.zeros((v.cam.h, v.cam.w), np.uint8)
            for j, o2 in enumerate(out):
                if j == i:
                    continue
                s = shapes.get(j) or sm_of(o2)
                if s is not None:
                    shapes[j] = s
                    m |= s[0].sil(v.cam, s[1])
            others[name] = m
        x0, y0 = float(o["pos"][0]), float(o["pos"][1])
        best = None
        for key in _candidates(o, b1_objs, task):
            try:
                sms = shape_models(key)
            except Exception as e:
                info.setdefault("errors", []).append(f"{key}: {type(e).__name__}")
                continue
            for sm in sms:
                per = sm.yaw_period
                if o.get("quat"):
                    w, _, _, z = [float(v) for v in o["quat"]]
                    y_world = 2 * math.atan2(z, w)
                    yaws = [y_world, y_world + math.pi] if per > math.pi + 1e-6 else [y_world]
                else:
                    yaws = [k * per / n_yaw for k in range(n_yaw)] if per > 0.2 else [0.0]
                # the seed is the object's centre; the shape's origin can sit off it
                for yaw in yaws:
                    c_off = sm.pose(0, 0, yaw)[:3, :3] @ ((sm.asset.V.min(0) + sm.asset.V.max(0)) / 2)
                    try:
                        x, y, yy, loss, _ = nm_refine(sm, x0 - c_off[0], y0 - c_off[1], yaw, views, others, xy_step=0.006,
                                                      yaw_step=0.15, maxiter=150, fix_yaw=per < 0.2)
                    except Exception as e:
                        info.setdefault("errors", []).append(f"{key}: {type(e).__name__}: {e}")
                        continue
                    if best is None or loss < best[0]:
                        best = (loss, key, sm, x, y, yy)
        if best is not None:
            loss, key, sm, x, y, yy = best
            T = sm.pose(x, y, yy)
            c = T[:3, :3] @ ((sm.asset.V.min(0) + sm.asset.V.max(0)) / 2) + T[:3, 3]
            shift = math.hypot(c[0] - x0, c[1] - y0) * 1000
            info.update(loss=round(float(loss), 3), key=key, shift_mm=round(shift, 1))
            V0 = sm.asset.V @ sm.R0.T
            foot = float(min(np.ptp(V0[:, 0]), np.ptp(V0[:, 1])))
            cap = min(max_shift_mm, shift_frac * foot * 1000)  # a block may move ~half its width, a board more
            near = [j for j, o2 in enumerate(out) if j != i and math.hypot(c[0] - o2["pos"][0], c[1] - o2["pos"][1]) * 1000 < shift]
            info.update(cap_mm=round(cap, 1), neighbour=bool(near))
            if loss < max_loss and shift < cap and not near:
                V = sm.asset.V @ sm.R0.T
                o.update(pos=[round(float(v), 5) for v in T[:3, 3]], quat=[round(float(v), 6) for v in R_to_quat_wxyz(T[:3, :3])],
                         asset=key, category=key.split("/")[1], size=np.round(V.max(0) - V.min(0), 4).tolist(),
                         top_z=round(float((sm.asset.V @ T[:3, :3].T + T[:3, 3])[:, 2].max()), 4),
                         confidence=round(float(max(0.05, min(0.99, 1 - loss))), 3), source="refined")
                o.pop("_agent_point", None)
                shapes[i] = (sm, T)
                info["accepted"] = True
            elif loss < orient_loss and shift < min(orient_shift_mm, cap) and not near and not o.get("quat"):
                # a poor fit that agrees with the agent's point: take its shape and yaw, keep the agent's xy
                c_off = T[:3, :3] @ ((sm.asset.V.min(0) + sm.asset.V.max(0)) / 2)
                T2 = sm.pose(x0 - c_off[0], y0 - c_off[1], yy)
                V = sm.asset.V @ sm.R0.T
                o.update(pos=[round(float(v), 5) for v in T2[:3, 3]], quat=[round(float(v), 6) for v in R_to_quat_wxyz(T2[:3, :3])],
                         asset=key, category=key.split("/")[1], size=np.round(V.max(0) - V.min(0), 4).tolist(),
                         confidence=round(float(min(0.5, 1 - loss)), 3), source="refined")
                shapes[i] = (sm, T2)
                info["accepted"] = "orientation_only"
        infos.append(info)
    return out, infos


def build_views(images: dict, state: dict, cams: dict, bgs: dict, extra: list | None = None) -> dict:
    """Segmented views of this frame (``<cam>``) and of the extra observations of the same scene (``<cam>@<k>``, the
    names B1's extra frames use: ``roi_loss`` counts them only where they see the object well)."""
    from ..inverse_layout import EstimatorConfig, _views_subset
    from ..segment import segment_views

    cfg = EstimatorConfig()
    views = _views_subset(segment_views(images, cams, state, bgs, thr=cfg.fg_thr), cfg)
    for k, (im_k, st_k, cams_k) in enumerate(extra or [], start=1):
        vk = _views_subset(segment_views(im_k, cams_k, st_k, bgs, thr=cfg.fg_thr), cfg)
        views.update({f"{name}@{k}": v for name, v in vk.items()})
    return views


def _quat_angle_deg(a, b) -> float:
    d = abs(sum(float(x) * float(y) for x, y in zip(a, b)))
    return math.degrees(2 * math.acos(min(1.0, d)))


def _hyp(o: dict):
    """(shape, T) of an object reading with an asset and an orientation, else None."""
    from ..cameras import pose_to_T
    from ..refine import _shape_for

    k, q = o.get("asset"), o.get("quat")
    if not k or not q:
        return None
    try:
        T = pose_to_T(o["pos"], q)
        return _shape_for(k, T[:3, :3], None), T
    except Exception:  # noqa: BLE001 (an asset outside the catalogue)
        return None


def pose_loss(o: dict, views: dict, others: dict | None = None) -> float | None:
    """Silhouette loss (``roi_loss``: weighted 1 - soft IoU over the views that see it) of one reading as given (asset,
    pos, quat), with ``others`` (per view: the other objects' silhouettes) taken out of the foreground. None when no
    view sees it."""
    from ..inverse_layout import make_roi_views, roi_loss

    h = _hyp(o)
    if h is None:
        return None
    sm, T = h
    rvs = make_roi_views(sm, T, views, others)
    if not rvs:
        return None
    loss, per = roi_loss(sm, T, rvs)
    return float(loss) if per else None


def _bottom(h) -> float:
    """Lowest point (world z) of a (shape, T) hypothesis."""
    sm, T = h
    V = getattr(sm.asset, "Vs", None)
    V = sm.asset.V if V is None else V
    return float((V @ T[:3, :3].T + T[:3, 3])[:, 2].min())


def arbitrate(views: dict, objects: list[dict], b1_objs: list[dict], margin: float = 0.0, move_mm: float = 3.0,
              turn_deg: float = 5.0, pair_m: float = 0.15, rest_z: bool = True,
              pinned_margin: float = 0.15, refined_turn_deg: float = 30.0) -> tuple[list[dict], list[dict]]:
    """The agent says WHAT is where; it replaces a pose B1 fitted only with evidence. Agent objects are paired with B1's
    objects of the same category (Hungarian on xy within ``pair_m``, the pairing ``record.merge_into_record`` makes).

    - Height (``rest_z``): an unrefined agent point (``_agent_point``) is a head-plane hit at a height the agent chose
      (often the top face): breadth replays had it 15-154 mm off for 30 of 99 agent-changed objects (a garage figure
      100 mm under the table). Its asset is set down so its lowest point is at its paired B1 object's (the same
      physical object), else on the table.
    - Pose: a pair whose agent pose moves B1's by more than ``move_mm`` or turns it by more than ``turn_deg`` is scored
      in every view (this frame + the extra observations), each inside its own scene (B1's pose with B1's other objects
      masked out, the agent's with the agent's). The agent's pose stands when its loss is below B1's by ``margin``
      (``pinned_margin`` for an object B1 placed at its config slot, ``flags.slot``: make_kong's tiles, the tower's base
      boards); else the object takes B1's pose, asset and confidence back (``arbitrated: "b1"``). A pose B1's views
      cannot score (hidden everywhere) leaves the agent's reading. An H2 fit (``source: refined``) that turns B1's pose
      by at most ``refined_turn_deg`` stands without the comparison (H2's own fit is the evidence).

    Breadth-2 (proxy B): the agent turned confident, multi-view-consistent charger poses by 80-180 deg (yaw typed in its
    own convention, or H2 fitted from it into a flip) and slid a make_kong tile row 23 mm off its config slots."""
    from scipy.optimize import linear_sum_assignment

    from ..cameras import TABLE_Z

    out = [dict(o) for o in objects]
    infos: list[dict] = []
    b1_cats = {_cat(b) for b in b1_objs}

    def cat_of(o):  # merge_into_record's rule: a free-text kind ("figurine") is the nearest B1 object's category
        c = _cat(o)
        if c in b1_cats or not b1_objs:
            return c
        return _cat(min(b1_objs, key=lambda b: math.hypot(b["pos"][0] - o["pos"][0], b["pos"][1] - o["pos"][1])))

    by_cat: dict = {}
    for i, o in enumerate(out):
        by_cat.setdefault(cat_of(o), [[], []])[0].append(i)
    for j, b in enumerate(b1_objs):
        if _cat(b) in by_cat:
            by_cat[_cat(b)][1].append(j)
    pairs = []
    for cat, (A, B) in by_cat.items():
        if not A or not B:
            continue
        C = np.array([[math.hypot(out[i]["pos"][0] - b1_objs[j]["pos"][0], out[i]["pos"][1] - b1_objs[j]["pos"][1])
                       for j in B] for i in A])
        ri, ci = linear_sum_assignment(np.minimum(C, pair_m + 1.0))
        pairs += [(A[r], B[c]) for r, c in zip(ri, ci) if C[r, c] <= pair_m]
    pair_of = dict(pairs)
    h_b1 = {j: _hyp(b) for j, b in enumerate(b1_objs)}
    if rest_z:
        for i, o in enumerate(out):
            if not o.get("_agent_point") or o.get("source") == "refined":
                continue
            j = pair_of.get(i)
            cand = dict(o)
            if j is not None:  # merge keeps B1's turn / asset when the agent gives none
                cand = dict(o, quat=o.get("quat") or b1_objs[j].get("quat"), asset=o.get("asset") or b1_objs[j].get("asset"))
            h = _hyp(cand)
            target = _bottom(h_b1[j]) if j is not None and h_b1.get(j) is not None else TABLE_Z
            if h is None:
                continue
            dz = target - _bottom(h)
            o["pos"] = [float(o["pos"][0]), float(o["pos"][1]), float(o["pos"][2]) + dz]
            o["rest_dz_mm"] = round(dz * 1000, 1)
    # each pose is scored inside its own scene: B1's pose with B1's other objects (+ the agent's new ones) masked out,
    # the agent's with the agent's others (a row of tiles shifted together must not mask the true neighbours)
    paired_a = {i for i, _ in pairs}
    h_ag = {i: _hyp(o) for i, o in enumerate(out)}

    sil_cache: dict = {}

    def masks(hs: list) -> dict:
        m_out = {}
        for name, v in views.items():
            m = np.zeros((v.cam.h, v.cam.w), np.uint8)
            for key, h in hs:
                if h is not None:
                    if (key, name) not in sil_cache:
                        sil_cache[(key, name)] = h[0].sil(v.cam, h[1])
                    m |= sil_cache[(key, name)]
            m_out[name] = m
        return m_out

    for i, j in pairs:
        o, b = out[i], b1_objs[j]
        moved = math.hypot(o["pos"][0] - b["pos"][0], o["pos"][1] - b["pos"][1]) * 1000
        turn = _quat_angle_deg(o["quat"], b["quat"]) if o.get("quat") and b.get("quat") else 0.0
        if moved <= move_mm and turn <= turn_deg and (not o.get("asset") or o.get("asset") == b.get("asset")):
            continue
        if o.get("source") == "refined" and turn <= refined_turn_deg and not (b.get("flags") or {}).get("slot"):
            # H2 fitted it in these views from the agent's point (loss < 0.5, a capped shift): that is the evidence. A
            # frame-0-only comparison reverted good H2 fits of blocks half under the home grippers, where B1's wrong
            # fit to the visible half scores lower. A flipped H2 fit (plug 100003, 180 deg) is still compared
            continue
        # merge keeps B1's turn / asset where the agent gives none
        cand = dict(o, quat=o.get("quat") or b.get("quat"), asset=o.get("asset") or b.get("asset"))
        oth_a = masks([(("a", k), h) for k, h in h_ag.items() if k != i])
        oth_b = masks([(("b", k), h) for k, h in h_b1.items() if k != j] + [(("a", k), h) for k, h in h_ag.items() if k not in paired_a])
        l_a, l_b = pose_loss(cand, views, oth_a), pose_loss(b, views, oth_b)
        m = pinned_margin if (b.get("flags") or {}).get("slot") else margin  # the config pins it there: a higher bar
        keep_b1 = l_a is None or (l_b is not None and not (l_a < l_b - m))
        info = {"label": o.get("label"), "b1_label": b.get("label"), "moved_mm": round(moved, 1), "turn_deg": round(turn, 1),
                "loss_agent": None if l_a is None else round(l_a, 4), "loss_b1": None if l_b is None else round(l_b, 4),
                "agent_source": o.get("source"), "margin": m, "kept": "b1" if keep_b1 else "agent"}
        if keep_b1:
            o.update({k: b[k] for k in ("pos", "quat", "asset", "size", "top_z", "category") if b.get(k) is not None})
            o["confidence"] = float(b.get("confidence", o.get("confidence", 0.5)))
            o["source"] = b.get("source", "head")
            o["arbitrated"] = "b1"
            o.pop("rest_dz_mm", None)
        infos.append(info)
    return out, infos
