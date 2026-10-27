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
    k = 0
    for bi in sorted(best):
        hs = sorted(best[bi], key=lambda h: h.loss)
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
                            "source": "belt_read"}})
        k += 1
    record["_perception"] = {"version": "belt-v0", "n_head_blobs": len(head_blobs), "t_s": round(time.time() - t0, 3)}
    return record


__all__ = ["belt_read", "rider_groups"]
