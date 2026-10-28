"""Belt reads for the runtime's tracker: the riders of a moving base (a conveyor), one best single-object fit per
head-view blob.

B1 (``inverse_layout.estimate``) assigns the task's label counts: on a belt the five riders' group "needs" five objects
while most riders are still upstream, out of view. It then splits the visible blobs into extra objects (2-3 phantoms
per read on our recorded belt frames, the frame-0 camera of breadth-2 100005 read as puppet + puppet + bottle) and
mislabels the crowd. The tracker only needs, per visible rider: where it is, what it is, how sure. Here every head-view
foreground blob (against the scene background, ``segment.with_scene``) gets inverse layout's own proposals for the
riders' candidate assets (``_propose``: area / colour prefilter, yaw-grid coarse fit, quick multi-view refine with
appearance), and keeps its single best hypothesis. No label counts, no splitting.

    belt_read(images, state, task, instruction="", bgs=None, cfg=None) -> record
        record["Rigid"][cat] = [{category, category_idx, label "rider<k>", default_pos, default_ori,
                                 _perception {confidence, sil_loss, blob_px, alternative}}]

A read with the table background where the belt should have its own (``blind_reason``: no scene background for
the moving base in the fold) sees no rider; the perception service marks such a read ``_perception.blind`` and the
runtime's tracker skips it (it neither confirms nor drops anything).

With ``cfg.embed_model`` set (default None: off), each blob's asset is picked by learned image embeddings instead
(``embed_id``: the head crop against catalogue renders of every rider candidate, similarity minus ``embed_weight`` x
our fit loss); a picked asset our proposals did not fit is fitted on the blob. ``_perception.embed`` records the pick.

GT is never read here.
"""
from __future__ import annotations

import time

import numpy as np

MIN_RIDER_CONF = 0.25     # a blob whose best fit explains less is not reported (an arm's edge, a shadow)


def rider_groups(groups: list[dict], task_labels: dict) -> list[int]:
    """Indices of the label groups that ride a Dynamic base (their placement's ``relative_plane`` names a Dynamic
    label)."""
    out = []
    for gi, g in enumerate(groups):
        rp = g["placement"].get("relative_plane", "Table")
        bases = rp if isinstance(rp, (list, tuple)) else [rp]
        if g["type"] == "Rigid" and any((task_labels.get(str(b).split("/")[0]) or {}).get("type") == "Dynamic" for b in bases):
            out.append(gi)
    return out


def moving_bases(task: str) -> list[str]:
    """Asset keys of the Dynamic bases (belts) the task's Rigid labels ride (their placement's ``relative_plane``);
    [] when nothing rides one, or for an unknown task."""
    from .inverse_layout import task_map

    try:
        labels = task_map()[task]["labels"]
    except Exception:  # noqa: BLE001 (an unknown set, or no catalogue)
        return []
    out: set = set()
    for d in labels.values():
        if d.get("type") != "Rigid":
            continue
        rp = (d.get("placement") or {}).get("relative_plane", "Table")
        for b in rp if isinstance(rp, (list, tuple)) else [rp]:
            base = labels.get(str(b).split("/")[0]) or {}
            if base.get("type") == "Dynamic":
                out.update(base.get("candidates") or [])
    return sorted(out)


def blind_reason(task: str, scene_bg: str | None) -> str | None:
    """Why a belt read of ``task`` with the scene background ``scene_bg`` (``segment.scene_bg_key``; None = the
    table's) cannot see the riders, or None. Without its belt's scene background the dark belt segments against the
    table background as foreground: genuine-5.1 compute4 2026-10-28 (bg_folds_51 had no ``scene/``), 125 of 127 reads
    found nothing and the rest were static fixture blobs, so no rider got a track and the skill read "nothing moves"."""
    bases = moving_bases(task)
    if bases and scene_bg not in bases:
        return f"no scene background for {', '.join(bases)} in the fold (the belt segments as foreground)"
    return None


def belt_read(images: dict, state: dict, task: str, instruction: str = "", bgs: dict | None = None, cfg=None) -> dict:
    from .cameras import R_to_quat_wxyz, cameras_from_state
    from .inverse_layout import (EstimatorConfig, _geometry_groups, _placement_prior, _propose, _views_subset, blobs,
                                 label_groups, task_map)
    from .segment import load_backgrounds, segment_views

    import dataclasses

    t0 = time.time()
    # one group, one object per blob: let more of the riders' candidates reach the refined fit (identity)
    cfg = dataclasses.replace(cfg or EstimatorConfig(), max_candidates_per_blob=10, refine_per_blob=6, refine_per_group=6)
    bgs = bgs if bgs is not None else load_backgrounds()
    views = _views_subset(segment_views(images, cameras_from_state(state), state, bgs, thr=cfg.fg_thr), cfg)
    head_blobs = blobs(views["cam_head"], "cam_head")
    groups = label_groups(task)
    labels = task_map()[task]["labels"]
    gis = rider_groups(groups, labels)
    record: dict = {"task": task, "instruction": instruction, "Rigid": {}}
    if not gis or not head_blobs:
        record["_perception"] = {"version": "belt-v0", "n_head_blobs": len(head_blobs), "t_s": round(time.time() - t0, 3)}
        return record
    pending = [(gi, groups[gi], _placement_prior(groups[gi])) for gi in gis]
    shapes_by_group = {gi: _geometry_groups(g, prior) for gi, g, prior in pending}
    hyps = _propose(head_blobs, pending, shapes_by_group, views, cfg, groups)
    best: dict[int, list] = {}
    for h in hyps:
        best.setdefault(h.blob, []).append(h)
    picks: dict[int, dict] = {}
    if getattr(cfg, "embed_model", None):
        best, picks = _embed_pick(head_blobs, best, gis, groups, views, bgs, cfg)
        if getattr(cfg, "embed_merge", False):
            head_blobs, best, picks = _merge_fragments(head_blobs, best, picks, gis, groups, views, cfg)
    k = 0
    for bi in sorted(best):
        hs = sorted(best[bi], key=lambda h: h.loss)
        if bi in picks:  # the embedding's pick first, then our other fits
            hs = sorted(hs, key=lambda h: (h.key != picks[bi]["key"], h.loss))
        h = hs[0]
        conf = float(np.clip(1 - h.per_view.get("sil_loss", h.loss), 0, 1))
        if conf < MIN_RIDER_CONF:
            continue
        alt = next((x for x in hs[1:] if x.key != h.key), None)
        T = h.T()
        cat, idx = h.key.split("/")[1], int(h.key.split("/")[2])
        record["Rigid"].setdefault(cat, []).append({
            "category": cat, "category_idx": idx, "label": f"rider{k}", "default_pos": T[:3, 3].tolist(),
            "default_ori": R_to_quat_wxyz(T[:3, :3]).tolist(), "scale": [1.0, 1.0, 1.0],
            "_perception": {"confidence": round(conf, 3), "sil_loss": h.per_view.get("sil_loss"), "cost": round(float(h.loss), 4),
                            "blob_px": int(head_blobs[bi].area) if 0 <= bi < len(head_blobs) else None,
                            "alternative": None if alt is None else {"key": alt.key, "margin": round(float(alt.loss - h.loss), 4)},
                            "source": "belt_read", **({"embed": picks[bi]} if bi in picks else {})}})
        k += 1
    record["_perception"] = {"version": "belt-v0", "n_head_blobs": len(head_blobs), "t_s": round(time.time() - t0, 3)}
    return record


def _fit_key(key: str, gi: int, bi: int, blob, views, cfg, groups):
    """Our best fit of one asset on one blob (coarse yaw grid over its place orientations, then the quick refine), or
    None."""
    from .inverse_layout import Hypothesis, _score_hyp, coarse_fit, shape_models, yaw_range

    best = None
    for sm in shape_models(key):
        for x, y, yaw, l in coarse_fit(sm, blob, views, cfg.yaw_step_deg, yaw_range(groups[gi], sm.yaw_period))[:1]:
            if best is None or l < best.loss:
                best = Hypothesis(gi, key, sm.tag, x, y, yaw, l, blob=bi)
                best.per_view = {"geom_group": [key]}
    return None if best is None else _score_hyp(best, views, None, cfg, groups, cfg.quick_iters)


def _embed_pick(head_blobs, best: dict, gis: list, groups: list, views, bgs, cfg):
    """Per blob (>= ``cfg.embed_min_px``): the asset the embedding fused with our losses picks; a fit of it is added to
    the blob's hypotheses when our proposals had none. Returns (hypotheses per blob, {blob: pick record})."""
    from .embed_id import fuse, identity

    emb = identity(cfg.embed_model, color_M=cfg.color_M, lit=cfg.embed_lit, crop_mode=cfg.embed_crop)
    key_group = {}
    for gi in gis:
        for key in groups[gi]["candidates"]:
            key_group.setdefault(key, gi)
    keys = sorted(key_group)
    head = views["cam_head"]
    bg = (bgs or {}).get("cam_head")
    best = {bi: list(hs) for bi, hs in best.items()}
    picks = {}
    for bi, b in enumerate(head_blobs):
        if b.area < cfg.embed_min_px:
            continue
        sims = emb.scores(head.img, head.cam, b.mask, b.bbox, b.centroid, keys, bg=bg)
        if not sims:
            continue
        hs = best.get(bi, [])
        losses: dict[str, float] = {}
        for h in sorted(hs, key=lambda h: h.loss):
            losses.setdefault(h.key, float(h.loss))
        ranked = fuse(sims, losses, cfg.embed_weight)
        key = ranked[0][0]
        if key not in losses:
            h = _fit_key(key, key_group[key], bi, b, views, cfg, groups)
            if h is None:
                continue
            best.setdefault(bi, []).append(h)
        picks[bi] = {"key": key, "sim": round(sims[key], 4), "margin": round(ranked[0][1] - ranked[1][1], 4) if len(ranked) > 1 else None,
                     "ours": min(losses, key=losses.get) if losses else None}
    return best, picks


def _merge_fragments(head_blobs, best: dict, picks: dict, gis: list, groups: list, views, cfg):
    """Blobs the embedding gives the same asset, whose centres lie within half that asset's largest extent of each other
    (a dark figure split into pieces on the dark belt), become one blob fitted again as that asset. Returns new
    (blobs, hypotheses per blob, picks); merged-away blobs drop out."""
    from .cameras import TABLE_Z
    from .inverse_layout import Blob, shape_models

    cam = views["cam_head"].cam
    key_group = {}
    for gi in gis:
        for key in groups[gi]["candidates"]:
            key_group.setdefault(key, gi)
    pts, reach = {}, {}
    for bi, p in picks.items():
        sm = shape_models(p["key"])[0]
        pts[bi] = cam.ray_plane(np.asarray(head_blobs[bi].centroid, float)[None, :], TABLE_Z + sm.height / 2)[0][:2]
        reach[bi] = max(0.04, 0.5 * float(np.max(sm.asset.extents)))
    parent = {bi: bi for bi in picks}

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    ids = sorted(picks)
    for x, i in enumerate(ids):
        for j in ids[x + 1:]:
            if picks[i]["key"] == picks[j]["key"] and np.isfinite(pts[i]).all() and np.isfinite(pts[j]).all() \
                    and float(np.hypot(*(pts[i] - pts[j]))) <= reach[i]:
                parent[root(j)] = root(i)
    comps: dict[int, list] = {}
    for bi in ids:
        comps.setdefault(root(bi), []).append(bi)
    blobs_out, best, picks = list(head_blobs), dict(best), dict(picks)
    for r, members in comps.items():
        if len(members) < 2:
            continue
        m = np.zeros_like(head_blobs[r].mask)
        for bi in members:
            m |= head_blobs[bi].mask
        ys, xs = np.nonzero(m)
        merged = Blob("cam_head", m, int(m.sum()), np.array([xs.mean(), ys.mean()]),
                      (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
        key = picks[r]["key"]
        h = _fit_key(key, key_group[key], r, merged, views, cfg, groups)
        if h is None:
            continue
        blobs_out[r] = merged
        best[r] = [h] + [x for x in best.get(r, []) if x.key != key]
        picks[r] = {**picks[r], "merged": len(members)}
        for bi in members:
            if bi != r:
                best.pop(bi, None)
                picks.pop(bi, None)
    return blobs_out, best, picks


__all__ = ["belt_read", "blind_reason", "moving_bases", "rider_groups"]
