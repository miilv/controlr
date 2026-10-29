"""Stream C: SAM 3.1's video tracker (Comfy-Org/sam3.1 multiplex checkpoint, Meta's sam3 code) as the frame-to-frame
mask tracker on a per-step sequence, against GT visible silhouettes; and the per-frame alternative (perception's wrist
refine from a perturbed prior, its fitted silhouette as the mask) on the same frames.

    python scripts/perception/bgperc/sam_track_eval.py SEQ_DIR --cam cam_left_wrist --start 0 --n 120 [--stride 1]
        [--click oracle|belief] [--refine] [--out OUT.json]

GT (scorer side): every label object's silhouette at its GT pose, painted far to near, minus the robot mask.
Click "oracle": the deepest pixel of the object's visible GT mask on the first frame it shows >= 400 px; "belief":
the same on B1's frame-0 pose (what a runtime would click). One SAM session per sequence; all objects at once.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np


def gt_objects(rows):
    """GT rows (scorer side) in the asset frame, the fixtures perception never estimates dropped."""
    from dojo.perception.grasp_metric import IGNORE_LABELS, gt_rows_asset
    out = []
    for r in gt_rows_asset(rows):
        if r["label"] in IGNORE_LABELS or r.get("clutter"):
            continue
        out.append({"label": r["label"], "key": r["key"], "pos": np.asarray(r["pos"], float),
                    "quat": np.asarray(r["quat"], float), "type": r.get("type", "Rigid")})
    return out


def visible_gt(seq, step, cam_name, gts):
    """label -> visible GT mask (painter's order by mean depth, robot removed)."""
    from dojo.perception.cameras import cameras_from_state, pose_to_T
    from dojo.perception.render import load_asset, robot_mask, silhouette, transform
    st = seq.state(step)
    cam = cameras_from_state(st)[cam_name]
    items = []
    for g in gts:
        try:
            a = load_asset(g["key"])
        except Exception:  # noqa: BLE001
            continue
        T = pose_to_T(g["pos"], g["quat"])
        V = transform(a.Vs, T)
        _, z = cam.project(V)
        if (z <= 0.01).all():
            continue
        m = silhouette(cam, V, a.Fs)
        if m.sum() == 0:
            continue
        items.append((float(np.median(z[z > 0.01])), g["label"], m))
    lab = np.full((cam.h, cam.w), -1, np.int32)
    names = []
    for i, (_, name, m) in enumerate(sorted(items, key=lambda x: -x[0])):   # far first, near paints over
        lab[m > 0] = i
        names.append(name)
    rob = robot_mask(cam, st, dilate_px=0, closed_jaws=True)
    lab[rob > 0] = -1
    return {name: (lab == i) for i, name in enumerate(names)}


def belief_visible(seq, steps, cam_name):
    """Per frame: GT label -> the visible mask of the runtime's planning-record object nearest that GT object (same
    group, frame 0): what a runtime would click from its belief."""
    import glob
    import gzip

    from dojo.perception.grasp_metric import Groups, flatten_record
    from dojo.runtime.estimate import planning_record
    d = os.path.dirname(seq.meta["actions"])
    rep = json.load(gzip.open(glob.glob(os.path.join(d, f"{seq.task}_*_{seq.layout_id}_report.json.gz"))[0]))
    bel = flatten_record(planning_record(rep["records"])[1])
    groups = Groups(seq.task)
    g0 = gt_objects(seq.gt(steps[0]))
    pairs = {}
    for g in g0:
        cand = [(float(np.linalg.norm(np.asarray(b["pos"]) - g["pos"])), lab) for lab, b in bel.items()
                if groups.of(lab) == groups.of(g["label"])]
        if cand and min(cand)[0] < 0.08:
            pairs[g["label"]] = min(cand)[1]
    objs = [{"label": gl, "key": bel[bl]["key"], "pos": np.asarray(bel[bl]["pos"], float),
             "quat": np.asarray(bel[bl]["quat"], float), "type": "Rigid"} for gl, bl in pairs.items()]
    return {i: visible_gt(seq, s, cam_name, objs) for i, s in enumerate(steps)}


def deepest(mask):
    import cv2
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    v, u = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(u), float(v)


def patch_edt():
    """sam3 imports a triton kernel for its euclidean distance transform (``sam3.model.edt``); the perception venv has
    no triton (and a stub breaks torch's inductor). Register OpenCV's distanceTransform under that module name before
    sam3 is imported (it is used only to sample correction clicks, never on the point-prompt inference path)."""
    import types

    import cv2
    import torch

    def edt_triton(data):
        x = data.detach().to("cpu").numpy().astype(np.uint8)
        out = np.stack([cv2.distanceTransform(m, cv2.DIST_L2, 0) for m in x.reshape(-1, *x.shape[-2:])])
        return torch.from_numpy(out.reshape(x.shape)).to(data.device)

    m = types.ModuleType("sam3.model.edt")
    m.edt_triton = edt_triton
    sys.modules["sam3.model.edt"] = m


@contextlib.contextmanager
def safetensors_torch_load():
    """Meta's builder torch.load()s its checkpoint path; read a .safetensors one instead (no 1.7 GB .pt copy)."""
    import torch
    from safetensors.torch import load_file
    orig = torch.load

    def load(f, *a, **k):
        p = f if isinstance(f, str) else getattr(f, "name", "")
        if str(p).endswith(".safetensors"):
            return load_file(str(p))
        return orig(f, *a, **k)
    torch.load = load
    try:
        yield
    finally:
        torch.load = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--cam", default="cam_left_wrist")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--click", default="oracle", choices=("oracle", "belief"))
    ap.add_argument("--refine", action="store_true", help="also the per-frame wrist refine baseline (wrist cams)")
    ap.add_argument("--text", default="object", help="the detector prompt of the first full pass ('' = none: points only)")
    ap.add_argument("--ckpt", default=os.path.expanduser("~/dojo-data/models/sam3.1/sam3.1_multiplex_fp16.safetensors"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from dojo.perception.seq_data import Sequence
    seq = Sequence(a.seq)
    steps = [s for s in seq.steps if s >= a.start][:: a.stride][: a.n]
    tmp = tempfile.mkdtemp(prefix="bgperc_sam_")
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)   # the frame copies (13 dirs had piled up in /tmp)
    for i, s in enumerate(steps):
        shutil.copy(seq.dir / "frames" / f"{s:04d}_{a.cam}.jpg", os.path.join(tmp, f"{i}.jpg"))
    gtm = {}
    t = time.perf_counter()
    for i, s in enumerate(steps):
        gtm[i] = visible_gt(seq, s, a.cam, gt_objects(seq.gt(s)))
    t_gt = time.perf_counter() - t
    # prompts: per label, the first frame where it shows >= 400 px (oracle: its visible GT mask; belief: the runtime's
    # planning-record pose of the label matched to that GT object, projected, robot removed)
    prompts = {}
    belief_masks = None
    if a.click == "belief":
        belief_masks = belief_visible(seq, steps, a.cam)
    for i in range(len(steps)):
        for lab, m in gtm[i].items():
            if lab in prompts or m.sum() < 400:
                continue
            if belief_masks is None:
                prompts[lab] = (i, deepest(m))
            else:
                bm = belief_masks[i].get(lab)
                if bm is not None and bm.sum() >= 400:
                    prompts[lab] = (i, deepest(bm))
    import torch
    patch_edt()
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    t = time.perf_counter()
    with safetensors_torch_load():
        pred = build_sam3_multiplex_video_predictor(checkpoint_path=a.ckpt, use_fa3=False, use_rope_real=False,
                                                    async_loading_frames=False)
    t_load = time.perf_counter() - t
    H, W = 480, 640
    ids = {lab: 100 + k for k, lab in enumerate(sorted(prompts))}
    out_masks = {}
    t = time.perf_counter()
    m_ = pred.model              # the model API directly (the predictor's start_session passes kwargs this init_state lacks)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = m_.init_state(resource_path=tmp, async_loading_frames=False)
        t_start = time.perf_counter() - t
        # SAM 3.1's multiplex model drops a point-prompted object after its prompt frame unless a detector (text)
        # propagation ran first (its partial SAM2 propagation merges into the detector's per-frame outputs): one text
        # prompt + a full pass, then the clicks (2026-10-28)
        t_f = time.perf_counter()
        n_full = 0
        if a.text:
            m_.add_prompt(st, frame_idx=0, text_str=a.text)
            for _fi, _o in m_.propagate_in_video(st, start_frame_idx=0, reverse=False):
                n_full += 1
        t_full = time.perf_counter() - t_f
        for lab, (fi, (u, v)) in sorted(prompts.items(), key=lambda x: x[1][0]):
            m_.add_prompt(st, frame_idx=fi, points=torch.tensor([[u / W, v / H]], dtype=torch.float32),
                          point_labels=torch.tensor([1], dtype=torch.int32), obj_id=ids[lab])
        t1 = time.perf_counter()
        for fi, o in m_.propagate_in_video(st, start_frame_idx=0, reverse=False):
            if not isinstance(o, dict) or "out_obj_ids" not in o:
                continue
            oids = [int(x) for x in np.asarray(o["out_obj_ids"].cpu() if hasattr(o["out_obj_ids"], "cpu")
                                                else o["out_obj_ids"]).reshape(-1)]
            ms = o["out_binary_masks"]
            ms = np.asarray(ms.cpu() if hasattr(ms, "cpu") else ms)
            out_masks[int(fi)] = {k: ms[j].astype(bool) for j, k in enumerate(oids)}
        t_prop = time.perf_counter() - t1
    vram = torch.cuda.max_memory_allocated() / 1e9
    per = {lab: {"iou": [], "fp": 0, "fn": 0, "n": 0} for lab in ids}
    for i in range(len(steps)):
        for lab in ids:
            g = gtm[i].get(lab)
            gv = g is not None and g.sum() >= 100
            m = out_masks.get(i, {}).get(ids[lab])
            mv = m is not None and m.sum() >= 50
            if i < prompts[lab][0]:
                continue
            if gv:
                per[lab]["n"] += 1
                if mv:
                    inter = float((m & g).sum())
                    per[lab]["iou"].append(inter / float((m | g).sum()))
                else:
                    per[lab]["fn"] += 1
                    per[lab]["iou"].append(0.0)
            elif mv:
                per[lab]["fp"] += 1
    res = {"seq": str(seq.dir), "cam": a.cam, "steps": [steps[0], steps[-1]], "n_frames": len(steps), "stride": a.stride,
           "load_s": round(t_load, 1), "start_session_s": round(t_start, 2), "propagate_s": round(t_prop, 2),
           "text": a.text, "full_pass_s": round(t_full, 2), "full_pass_frames": n_full,
           "ms_per_frame": round(1000 * t_prop / max(1, len(steps)), 1), "n_objects": len(ids), "vram_gb": round(vram, 2),
           "gt_s": round(t_gt, 1),
           "per_object": {lab: {"n": p["n"], "miou": round(float(np.mean(p["iou"])), 3) if p["iou"] else None,
                                "recall50": round(float(np.mean([x >= 0.5 for x in p["iou"]])), 3) if p["iou"] else None,
                                "fn": p["fn"], "fp": p["fp"]} for lab, p in per.items()}}
    if a.refine and "wrist" in a.cam:
        res["refine"] = refine_baseline(seq, steps, a.cam, gtm)
    shutil.rmtree(tmp, ignore_errors=True)
    s = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    print(s)


def refine_baseline(seq, steps, cam_name, gtm, noise_mm=8.0, max_frames=60):
    """Per frame: refine every GT object visible >= 400 px from GT + noise_mm; mask = fitted silhouette minus the robot
    (ok fits only; a not-ok fit = no mask)."""
    from dojo.perception.cameras import cameras_from_state, pose_to_T
    from dojo.perception.refine import refine
    from dojo.perception.render import load_asset, robot_mask, silhouette, transform
    arm = "left" if "left" in cam_name else "right"
    rng = np.random.default_rng(0)
    per: dict = {}
    t_all, n_calls = 0.0, 0
    for i, s in list(enumerate(steps))[:max_frames]:
        fr = seq.frame(s, cams=(cam_name,))
        st = fr["state"]
        cam = cameras_from_state(st)[cam_name]
        rob = robot_mask(cam, st, dilate_px=0, closed_jaws=True)
        for g in gt_objects(seq.gt(s)):
            gm = gtm[i].get(g["label"])
            if gm is None or gm.sum() < 400 or g["type"] != "Rigid":
                continue
            d = rng.normal(size=2)
            prior = list(g["pos"] + np.r_[d / np.linalg.norm(d) * noise_mm / 1000, 0.0]) + list(g["quat"])
            t = time.perf_counter()
            try:
                r = refine(g["key"], fr["images"][cam_name], st, prior, arm=arm)
            except Exception:  # noqa: BLE001
                r = {"ok": False}
            t_all += time.perf_counter() - t
            n_calls += 1
            p = per.setdefault(g["label"], {"iou": [], "ok": 0, "n": 0, "err_mm": []})
            p["n"] += 1
            if r.get("ok"):
                p["ok"] += 1
                a = load_asset(g["key"])
                m = silhouette(cam, transform(a.Vs, pose_to_T(r["pose"][:3], r["pose"][3:])), a.Fs).astype(bool) & ~rob.astype(bool)
                p["iou"].append(float((m & gm).sum()) / float((m | gm).sum()))
                p["err_mm"].append(1000 * float(np.hypot(*(np.asarray(r["pose"][:2]) - g["pos"][:2]))))
            else:
                p["iou"].append(0.0)
    return {"ms_per_call": round(1000 * t_all / max(1, n_calls), 1), "calls": n_calls,
            "per_object": {lab: {"n": p["n"], "ok": p["ok"], "miou": round(float(np.mean(p["iou"])), 3),
                                 "recall50": round(float(np.mean([x >= 0.5 for x in p["iou"]])), 3),
                                 "med_err_mm": round(float(np.median(p["err_mm"])), 1) if p["err_mm"] else None}
                           for lab, p in per.items()}}


if __name__ == "__main__":
    main()
