"""Offline SAM task-bank/v2b mask comparison. DEV ONLY; never a runtime input.

prepare ROOT renders settled GT silhouettes with the existing seg_twin projector.
pred ROOT OUT --model sam3 --checkpoint CKPT --weights V2B --classes CLASSES.
Evaluation uses seg_twin's >=30px visibility, greedy one-to-one IoU and phantom definition.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from dojo.perception.seg_twin import (
    Det,
    asset_key,
    category_of,
    instances,
    iou,
    match_greedy,
)

BANKS = json.loads(Path(__file__).with_name("task_prompt_banks.json").read_text())
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def phrase_categories(task, p):
    """Explicit semantic mapping, independent of GT poses/identities. Empty = unmapped.

    Catalogue calls plush dolls `garage`, striped bowls `plate`, cups `ice_cream`.
    Ambiguous pen prompts remain ambiguous and cannot count as correct identity.
    """
    if task == "pour_by_language":
        return ["wine_bowl" if "bowl" in p else "wine_bottle"]
    rules = [
        ("pen holder|container|hexagonal cup|pencil cup", "pen_holder"),
        ("egg holder", "egg_holder"),
        ("basket", "egg_basket" if task == "fill_egg_holder" else "basket_grasp"),
        ("egg", "egg"),
        ("pig|bank", "piggy_bank"),
        ("gold|coin", "coin"),
        ("mouse.*(mat|pad)|mouse mat|mouse pad", "mousemat"),
        ("mouse", "mouse"),
        ("keyboard", "keyboard"),
        ("clock|beige box|white box", "alarm"),
        (
            "figurine|statue|toy figure|cactus|circular base",
            "garage" if task == "organize_table" else "figurine",
        ),
        (
            r"\bcar\b|vehicle|tank|van|bus",
            "car"
            if task in ("classify_objects_by_language", "pick_from_conveyor_by_image")
            else "toy_car",
        ),
        (
            "camera|black rectangular|small black box",
            "action_camera"
            if task in ("imitate_sorting_sequence", "classify_objects")
            else "camera",
        ),
        ("phone|nokia", "phone"),
        ("watch|bracelet|silver ring", "watch"),
        (
            "plush|stuffed|doll|rabbit",
            "toy" if task == "pick_from_conveyor_by_image" else "garage",
        ),
        ("wooden", "wooden_toy"),
        ("pepper", "pepper"),
        ("chocolate|candy", "chocolate_bar"),
        ("console|screen", "game_machine"),
        ("headphone", "headset"),
        ("bowl", "plate"),
        ("rattle|action figure", "puppet"),
        ("handbag|^bag$", "bag"),
        ("Red Bull|drink can", "bottle"),
        ("bottle", "bottle"),
        ("aerosol", "can"),
        ("correction tape", "correction_tape"),
        ("donut", "donut"),
        ("glue", "glue"),
        ("noodle", "noodles"),
        ("remote", "remote"),
        ("sachet", "sachet"),
        ("peeler", "peeler"),
        ("yogurt", "yogurt"),
        ("cup", "cup" if task == "pick_from_conveyor_by_image" else "ice_cream"),
    ]
    import re

    if (
        task in ("fill_pen_holder", "classify_objects")
        and re.search("pen|marker|highlighter", p)
        and "holder" not in p
        and "cup" not in p
    ):
        return ["pen", "oil_pen"] if task == "fill_pen_holder" else ["pen"]
    for pat, cat in rules:
        if re.search(pat, p, re.I):
            return [cat]
    return []


def dedup(dets, threshold=0.5):
    out = []
    for d in dets:
        if not any(iou(d.mask, p.mask) > threshold for p in out):
            out.append(d)
    return out


def samples(root):
    for p in sorted(Path(root).glob("*/*/source.json")):
        rec = json.loads(p.read_text())
        lid = rec["layout_id"]
        if not (100000 <= lid <= 100009 or 100020 <= lid <= 100029):
            raise ValueError("non-dev id")
        # 100025–29 are allowed dev IDs but overlap the old segmenter training
        # range. Keep this component comparison on the historical held-out slice.
        if lid >= 100025:
            continue
        yield p.parent, rec


def prepare(root):
    from dojo.perception.cameras import cameras_from_state
    from dojo.perception.seg_twin import gt_code_map

    for d, src in samples(root):
        if (d / "gt.json").exists():
            continue
        settled = json.loads((d / "settled.json").read_text())
        objs = []
        for typ in ("Rigid", "Geometry", "Articulation", "Dynamic"):
            for arr in (settled.get(typ) or {}).values():
                for o in arr:
                    objs.append(
                        {
                            **o,
                            "type": typ,
                            "pos": o["default_pos"],
                            "quat": o["default_ori"],
                        }
                    )
        if src.get("belt_phase"):
            shift = src["belt_phase"]["native_steps"] * 0.004
            for o in objs:
                if o["type"] == "Rigid" and o.get("relative_plane") == "conveyor":
                    o["pos"] = [o["pos"][0] + shift, *o["pos"][1:]]
            # Frame-zero is before any policy action. Only untouched, on-belt
            # riders are eligible; never reuse this for picked/touched states.
            objs = [
                o
                for o in objs
                if not (
                    o["type"] == "Rigid"
                    and o.get("relative_plane") == "conveyor"
                    and (o["pos"][0] > 1.88 or o["pos"][2] < 0.75)
                )
            ]
        state = json.loads((d / "state.json").read_text())
        cams = cameras_from_state(state)
        # Use the saved joint state for each view, including pre-grasp surveys.
        for cam in CAMS:
            codes, ok = gt_code_map(cams[cam], objs, state)
            Image.fromarray(codes).save(d / f"gt_{cam}.png")
        (d / "gt.json").write_text(json.dumps({"objects": objs, "rendered": ok}))
        print("GT", src["task"], src["layout_id"], sum(ok), len(objs), flush=True)


def score(dets, codes, objs):
    gt = instances(codes, 30)
    matches = match_greedy(gt, dets, thr=0.0)
    # Keep the historical >=30px matching unchanged; audit smaller objects
    # afterwards against unused predictions, outside the primary denominator.
    tiny = {gi: gm for gi, gm in instances(codes, 1).items() if gi not in gt}
    unused = [
        i for i in range(len(dets)) if i not in {pi for pi, _ in matches.values()}
    ]
    tiny_matches = match_greedy(tiny, [dets[i] for i in unused], thr=0.0)
    matches.update({gi: (unused[pi], v) for gi, (pi, v) in tiny_matches.items()})
    rows = []
    for gi, gm in {**gt, **tiny}.items():
        o = objs[gi]
        key = asset_key(o)
        p = None
        v = 0.0
        if gi in matches:
            pi, v = matches[gi]
            p = dets[pi]
        cats = p.key.split("|") if p else []
        cat_ok = (
            (category_of(p.key) == category_of(key))
            if p and "/" in p.key
            else (len(cats) == 1 and category_of(key) in cats)
        )
        rows.append(
            {
                "gi": gi,
                "label": o.get("label"),
                "key": key,
                "px": int(gm.sum()),
                "clutter": bool(o.get("clutter")),
                "iou": v,
                "pred_key": p.key if p else None,
                "class_ok": bool(p and p.key == key),
                "cat_ok": bool(cat_ok),
            }
        )
    robot = codes == 65534
    phantom = sum(
        max([iou(p.mask, g) for g in gt.values()] + [0.0]) < 0.3
        and (p.mask & robot).sum() < 0.5 * p.mask.sum()
        for p in dets
    )
    return rows, int(phantom)


def pred(a):
    import torch
    from dojo.perception.seg_twin import _SegModel
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    import shutil
    import subprocess

    used = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    )
    if used > 24 * 1024 or shutil.disk_usage(a.root).free < 10 * 1024**3:
        raise RuntimeError(
            "compute admission requires used VRAM <=24GiB and free disk >=10GiB"
        )
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(0.30)
    classes = json.loads(Path(a.classes).read_text())["names"]
    seg = _SegModel(a.weights, classes)
    # Separate process invocations keep the checkpoint arms' peaks independent.
    if a.model == "sam3.1":
        from safetensors.torch import load_file
        import sam3.model_builder as mb
        from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
        import pkg_resources

        # Construct Meta's multiplex detector, without its unused video tracker.
        # Components/arguments follow sam3.model_builder at f1647e8.
        tri = mb._create_multiplex_tri_backbone(
            compile_mode=None, use_fa3=False, use_rope_real=False
        )
        text = mb._create_text_encoder(
            pkg_resources.resource_filename(
                "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
            )
        )
        model = Sam3MultiplexDetector(
            num_feature_levels=1,
            backbone=mb.SAM3VLBackboneTri(scalp=0, visual=tri, text=text),
            transformer=mb._create_sam3_transformer(use_fa3=False),
            segmentation_head=mb._create_segmentation_head(use_fa3=False),
            semantic_segmentation_head=None,
            input_geometry_encoder=mb._create_geometry_encoder(),
            use_early_fusion=True,
            use_dot_prod_scoring=True,
            dot_prod_scoring=mb._create_dot_product_scoring(),
            supervise_joint_box_scores=True,
            is_multiplex=True,
        )
        ck = load_file(a.checkpoint)
        detector = {
            k.removeprefix("detector."): v
            for k, v in ck.items()
            if k.startswith("detector.")
        }
        missing, unexpected = model.load_state_dict(detector, strict=False)
        # Official loader also permits the unused CLIP projection and computed RoPE buffers.
        if unexpected or any(
            not (k.endswith("freqs_cis") or k.endswith("text_projection"))
            for k in missing
        ):
            raise RuntimeError((missing, unexpected))
        model = model.eval().cuda()
        del ck, detector
    else:
        model = build_sam3_image_model(
            checkpoint_path=a.checkpoint,
            load_from_HF=False,
            enable_inst_interactivity=False,
        )
    processor = Sam3Processor(model, confidence_threshold=0.05)
    timings = []
    with (
        open(a.out, "w") as out,
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        for d, src in samples(a.root):
            objs = json.loads((d / "gt.json").read_text())["objects"]
            task = src["task"]
            bank = BANKS[task]
            for cam in CAMS:
                img = np.asarray(Image.open(d / f"{cam}.png").convert("RGB"))
                codes = np.asarray(Image.open(d / f"gt_{cam}.png")).astype(np.int64)
                torch.cuda.synchronize()
                t = time.perf_counter()
                vd = [
                    Det(m.astype(np.uint8), key, sc, tuple(b))
                    for m, key, sc, b in seg.detect(img)
                ]
                torch.cuda.synchronize()
                tv = time.perf_counter() - t
                t = time.perf_counter()
                st = processor.set_image(Image.fromarray(img))
                sd = []
                for p in bank["phrases"]:
                    processor.reset_all_prompts(st)
                    st = processor.set_text_prompt(p, st)
                    for m, sc in zip(
                        st["masks"].squeeze(1).cpu().numpy(),
                        st["scores"].float().cpu().numpy(),
                    ):
                        if sc >= bank["api_min"] and m.any():
                            sd.append(
                                Det(
                                    m.astype(np.uint8),
                                    "|".join(phrase_categories(task, p)),
                                    float(sc),
                                    (0, 0, 0, 0),
                                )
                            )
                torch.cuda.synchronize()
                ts = time.perf_counter() - t
                sd = dedup(sorted(sd, key=lambda x: -x.score))
                union = dedup(vd + sd)
                for arm, dets in [("v2b", vd), (a.model, sd), ("union", union)]:
                    rows, ph = score(dets, codes, objs)
                    out.write(
                        json.dumps(
                            {
                                **src,
                                "image_sha256": hashlib.sha256(
                                    (d / f"{cam}.png").read_bytes()
                                ).hexdigest(),
                                "cam": cam,
                                "arm": arm,
                                "rows": rows,
                                "n_pred": len(dets),
                                "phantoms": ph,
                                "latency_s": tv
                                if arm == "v2b"
                                else ts
                                if arm == a.model
                                else tv + ts,
                            }
                        )
                        + "\n"
                    )
                out.flush()
                timings.append((tv, ts))
                print(
                    task,
                    src["layout_id"],
                    cam,
                    len(vd),
                    len(sd),
                    round(ts, 2),
                    flush=True,
                )
                del st
                gc.collect()
    Path(a.out + ".meta.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": hashlib.file_digest(
                    open(a.checkpoint, "rb"), "sha256"
                ).hexdigest(),
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                "torch": torch.__version__,
                "timings": timings,
            },
            indent=2,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "pred"])
    ap.add_argument("root")
    ap.add_argument("--out")
    ap.add_argument("--model", default="sam3")
    ap.add_argument("--checkpoint")
    ap.add_argument("--weights")
    ap.add_argument("--classes")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.root)
    else:
        pred(a)


if __name__ == "__main__":
    main()
