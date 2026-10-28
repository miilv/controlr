# 2026-10-28 — Learned multi-view 3D from our calibrated cameras (VGGT, MASt3R, MapAnything), and what it fixes at grasp time (stream V)

**Question.** The plane sweep (2026-10-27-stereo-height) measures object tops to 0.5-1.3 mm, but only where two views
overlap: 35 % coverage at home, 75 % at the scan pose. It fails on vertical faces, textureless tops and the glossy belt.
- Do learned multi-view geometry models give dense, reliable 3D from our 3 calibrated RGB cameras, plus D0's look
  frames?
- Does that help the perception failures that matter at grasp time (hidden or partly visible objects, thin objects,
  textureless or look-alike objects, height errors, shadow phantoms)?

**Data** (dev only). Frame 0 of d0_51 (genuine 5.1) and d0_ext (6.0 proxy): 207 scenes, 1043 objects. The same frames
as the sweep's report, ids 100000-09 / 100020-49.
- "home" = the 3 frame-0 cameras;
- "+looks" = plus the wrist views of D0 frames 1-3 (the scripted looks; objects untouched), median 5 views;
- deposit_coin: 10 frame-0 scenes from the perceiver stream's d0_coin;
- B1's own frame-0 records: the SAM stream's dump (current-main B1; same frames, same scorer).

GT = surfaces ray-cast from our layout records (`mv3d_eval.gt_surface`), read only by the scorers, after every
prediction.

**Spend.** 0 LLM calls. GPU: compute3 light3 (`~/dojo-gpu-light3.lock`), one job at a time, <= 4 GPU jobs on the box,
about 1.5 GPU-hours in total. CPU <= 8 cores. Disk < 8 GB: weights as bf16 / fp16 under `~/dojo-data/pbake-mv`, torch
read from the finished identity stream's venv through a `.pth`.

## Method

- **The model's own cameras are discarded; ours are used.**
  - VGGT's predicted cameras, aligned to ours by a similarity, are off by 46 mm rms (centres) and up to 3.7 deg
    (frame 0, median per scene).
  - With its global scale the head depth is off by 29 mm.
  - So each view keeps only its predicted depth map. That depth goes through our exact K and pose (head constants,
    wrists from FK), `mv3d.unproject`.
- **Scale from the table.** The table is at a known plane and covers most of every view.
  - `table`: one scale per view, the mode of (known-plane depth / predicted depth).
  - `poly`: a smooth per-view field, a 2nd-order polynomial in (u, v) of that log-ratio. It is fitted robustly (objects
    stand closer than the plane, so they are outliers on one side). It corrects the model's low-frequency warp.
- **Fused.** Every view's points are z-buffered into the head camera (`mv3d.splat`), so the nearest surface any view
  sees wins.
- **Top-down.** A 4 mm grid of max heights over the object region, from all views.
- **Scoring.** The sweep report's scorer (`stereo_eval.score`), so the numbers compare: object top = 90th percentile of
  heights over its visible pixels; top faces = pixels with |n_z| > 0.9; "detected" = >= 30 % of its raised pixels read
  > 1 cm; phantom blobs; B1's colour foreground on bare table, rejected = reads on the plane.

## Results

### Depth (207 scenes, 1043 objects)

| arm | objects measured | object top height abs err (median / p90) | within 5 / 10 mm | raised objects detected | phantom blobs / frame | B1 table-FG pixels rejected / confirmed | s / scene, VRAM |
|---|---|---|---|---|---|---|---|
| plane sweep, home (stereo report, `home_v5lr`) | 722 / 1044 | 4.4 / 79.2 mm | 0.51 / 0.60 | 449 / 583 | 1.4 | 0.75 / 0.12 | 4.7 s CPU |
| VGGT home, the model's own sim3 scale | 1026 / 1043 | 16.9 / 43.0 | 0.10 / 0.22 | 678 / 796 | 1.0 | 0.13 / 0.36 | |
| VGGT home, head depth only, poly | 1026 | 6.0 / 13.9 | 0.41 / 0.81 | 782 / 796 | 0.36 | 0.69 / 0.22 | |
| **VGGT home, fused, poly** | 1026 | **3.5 / 12.3** | **0.60 / 0.85** | 785 / 796 | 0.61 | 0.67 / 0.25 | 0.3 s GPU, 6.9 GB |
| VGGT home, fused, poly, bf16 trunk | 1026 | 3.6 / 12.4 | 0.60 / 0.86 | 785 / 796 | 0.59 | 0.67 / 0.25 | **0.2 s, 3.5 GB** |
| VGGT + looks (median 5 views), fused, poly | 1026 | 3.0 / 10.2 | 0.71 / 0.89 | 785 / 796 | 0.61 | 0.66 / 0.25 | 0.5 s, 8.1 GB |
| MapAnything (our K + poses as inputs), fused, poly | 1026 | 7.4 / 19.3 | 0.32 / 0.68 | 783 / 796 | 2.0 | 0.46 / 0.37 | 0.2 s, 8.0 GB |
| MASt3R (metric) raw depth | | about 2x too far (abs rel 1.0-1.1) | | | | | 0.8 s + 3.2 s matching, 4.4 GB |
| MASt3R dense, fused, poly | 1026 | 5.5 / 21.1 | 0.47 / | 783 / 796 | 10.0 | 0.51 / 0.39 | |
| MASt3R matches triangulated with our cameras | 481 | 1.8 / 8.6 | 0.83 / | 338 / 382 | 0.0 | 0.61 / 0.25 | |

**Paired with the sweep** (the objects it measures):

| objects | sweep (median / p90) | VGGT home fused poly |
|---|---|---|
| all 716 | 4.5 / 79.7 mm | 2.7 / 10.0 |
| top faces, 626 | 1.2 / 52.8 | 2.4 / 10.4 |
| +looks, 332 (the sweep with frames 1-3) | 2.0 / 43.9 | 3.0 / 11.1 |

- **Where the sweep measures a top it is more precise.** VGGT measures everything, and its tail is 5-8x shorter.
- **Bias.**
  - Head depth alone flattens objects by about 10 % (-5 mm median, -12 mm on objects > 10 cm).
  - Fusing in the wrist views brings the bias to -1 mm.
- **Per-pixel depth after the fit.** Table 0.9-1.0 mm median in every view. Object surfaces: 6 mm in the head, 2.5 mm
  in the wrists.
- **Per task** (median / p90 of the object top).
  - Worst: push_T 6.8 / 120 (the T under the grippers), press_by_number 9.4 / 13, put_bottles 5.8 / 25,
    general_pickup 5.8 / 27.
  - Best: store_tools 1.9 / 4.0, fill_pen 1.6 / 5.0.
- **Top-down fusion.** 905 of 917 raised objects found, 3.8 / 14.6 mm, but 3.7 phantom blobs per frame (fringes at
  object edges).
- **MASt3R** (ViT-L 512 metric, pairs head <-> wrists both ways + wrist <-> wrist; 0.8 s + 3.2 s matching per scene,
  4.4 GB).
  - Dense depth: its metric scale is about 2x too far on our scenes. Table-fitted (poly), fused: 5.5 / 21.1 mm with
    10 phantom blobs per frame, worse than VGGT.
  - Reciprocal matches triangulated with OUR exact cameras (reprojection <= 1 px, ray angle >= 2 deg) give a precise
    sparse map:
    - 481 / 1043 objects measured; top 1.8 / 8.6 mm, 83 % within 5 mm; 0 phantoms.
    - Paired with the sweep on 475 objects: 1.7 / 8.3 vs 1.9 / 88.7 mm.
  - It is the most precise of the learned arms where it measures, but the coverage is 46 % of objects.
- **MapAnything.** Given our intrinsics and metric poses, it is worse than VGGT on every metric. Its output cameras keep
  a 9 % scale error.

### B1's misses at frame 0 (matched by GT position)

- **Missed (18).**
  - 11 are visible in the head; VGGT reads all 8 tall ones as raised (fused top error 2.6 / 13.9 mm).
  - 4 are invisible to every home camera (stack_blocks 100020-22, stack_by_language 100002, under the arms). No 3D
    method fixes those; they need a look.
- **Gross (50).** 48 are visible in the head, and 31 of the 32 tall ones read as raised.
- So the geometry is there for most of B1's failures. What a consumer can do with it is below.

### Consumer 1: the height check of B1's own records (`stereo.check_heights`, paired, same frames)

Setup: B1's frame-0 records (the dump), 207 scenes, 1025 B1 objects. "Bad" = B1's predicted surface is > 10 mm off the
GT surface on its head pixels (the sweep report's protocol); 88 are bad. Every source goes through the same
`check_heights` at full resolution.

| source (home views) | objects measured | flag at 10 mm: precision / recall / false flags on right objects | flag at 15 mm | dh error on right objects (median / p90) | unexplained raised blobs / frame |
|---|---|---|---|---|---|
| plane sweep (check_record's) | 730 | 50/228, 50/88, 19 % | 44/202, 44/88, 17 % | 1.9 / 51.7 mm | 2.0 |
| **VGGT fused poly** | 1025 | 80/182, 80/88, 10.9 % | **65/107, 65/88, 4.5 %** | 3.2 / 10.1 | 1.0 |
| hybrid: sweep wherever it measured | 1025 | 80/190, 80/88, 11.7 % | 61/111, 61/88, 5.3 % | 2.9 / 10.3 | 3.2 |
| hybrid: sweep where it agrees with VGGT within 10 mm | 1025 | 79/182, 79/88, 11.0 % | 64/106, 64/88, 4.5 % | 2.8 / 10.1 | 1.0 |

- **deposit_coin** (10 scenes, 21 B1 objects, 13 bad).
  - At 15 mm: VGGT 7/8 precision, 7/13 recall; sweep 4/4, 4/13 (the sweep measures 14 of 21 at home).
  - VGGT flags the 3 misplaced piggy banks (40-60 mm off); the sweep flags 1.
  - VGGT does not resolve the thin stand: about 11 px wide in the head, smoothed away at 504 px. It misses the stands
    that are 11-14 mm off.
  - The scan-pose sweep (stereo_reject slot_bases) keeps the stand.

### Consumer 2: raised-but-unexplained regions -> candidates (`mv3d.propose_unexplained`)

Pipeline:
- VGGT raised blobs that no B1 object explains (dilated by 11 px), >= 150 px;
- candidates: B1's missing labels, plus its height-check suspects (|dh| > 15 mm);
- each fitted at the blob by B1's render-and-compare (`inverse_layout.refine`), with the blob's raised pixels added to
  the head foreground;
- accepted if the loss is <= the cap, the fit is within 40 mm of the blob, and no other object is within 30 mm. A
  suspect moves only if it beats its own pose's loss by 0.05.

eval_g1 before / after on the same records:

| arm | blobs tried (frames) | accepted | missed -> ok | missed -> overshoot | gross -> overshoot | missed -> gross | broken (ok -> worse) | phantoms |
|---|---|---|---|---|---|---|---|---|
| loss <= 0.6, >= 150 px | 28 (18 of 217 scenes incl. coin) | 12 | 2 | 1 | 2 | 4 (2 coin) | **0** | 49 -> 48 |
| loss <= 0.5, >= 60 px | 37 (22 of 217) | 9 | 2 | 1 | 1 | 4 (2 coin) | **0** | 43 -> 43 |

- **Positions are good when the candidate is right.** Bottles land at 1-3 mm, the hammer at 13 mm, the organize alarm
  at 5-11 mm.
- **Failures.**
  - Yaw: the alarm is placed 180 deg / 85 deg flipped, and a silhouette cannot tell.
  - Moved suspects: the shoe and the general_pickup target went to another object's blob.
  - Piggy banks: 37-71 mm and 120 deg off, a poor silhouette fit.
- **The real limit is how often it engages.** Only 18 of 207 frames have an unexplained raised blob. Most B1 gross
  objects sit on another object's blob: a swap, not a gap.

### Consumer 3: FoundationPose (model-based, catalogue meshes) on fused VGGT depth

**Setup** (`scripts/perception/mv/fpose_eval.py`).
- What runs: FoundationPose's refiner and scorer networks with its own crop / batch code. Weights from HF
  `gpue/foundationpose-weights`.
- Its `nvdiffrast_render` is replaced by our pyrender per-face-colour renderer on the crop camera, lit the same way.
  pytorch3d / open3d are stubbed. Nothing is compiled, and compute3 has no CUDA toolkit.
- Inputs: RGB = the head image; depth = the fused VGGT depth in the head camera; hypotheses = B1's pose and 8 yaw turns.
- Then 5 refine iterations; the scorer picks one.
- About 1 s per object.

| objects refined | -> within (fixed) | within -> worse (broken) | net within |
|---|---|---|---|
| every B1 object (1003) | gross -> ok 17, gross -> identity 6, overshoot -> ok / identity 15 | ok -> gross 21, ok -> overshoot 15, identity -> overshoot / gross 7 | **-5** |
| only those the VGGT height check flags at 15 mm (91) | 11 | 24 (ok -> gross 19) | **-13** |
| only those flagged at 10 mm (150) | 16 | 31 (ok -> gross 19) | **-15** |

- It finds real fixes, but on our resolution and with learned depth it breaks as many right objects. Gating by the
  height check does not help: the flagged right objects are exactly the ones with ambiguous geometry.
- This agrees with the research report (FoundationPose is a later step, for stacked / held objects with better depth).
  **Dropped.**

## Verdict

- **ADOPT, flag default off: VGGT-1B depth through our cameras as the height-check source** (`height_source="mv3d"`).
  - Use the bf16 trunk: 0.2 s, 3.5 GB.
  - It needs a GPU server next to the perception service (`python -m dojo.perception.mv3d serve`, `DOJO_MV3D_ADDR`).
  - It replaces the sweep at home: 15 mm flags 65/107 vs 44/202 precision, 65/88 vs 44/88 recall, 4.5 % vs 17 % false
    flags.
  - The coin stand stays with the scan-pose sweep.
  - Next: a rendered A/B (Integration) with `{"stereo_check": true, "height_source": "mv3d"}`. Consumers of the flag
    beyond `stereo_reject="slot_bases"` are not written yet; the annotation alone changes nothing downstream.
- **NEEDS an appearance score first: the raised-but-unexplained proposer** (`mv3d_propose`, default off).
  - Result: +2 within, +2-3 overshoot, 0 broken.
  - It is correct in position and wrong in yaw, and it engages in only about 20 of 217 scenes. B1's gross objects are
    mostly swaps, not gaps.
- **DROP:**
  - FoundationPose on learned depth: net -5 / -13 / -15 within, blanket / gated;
  - MapAnything with our calibration as input: worse than VGGT on every metric;
  - MASt3R dense depth: 2x scale, phantoms;
  - the sweep + VGGT hybrids: no gain;
  - the learned map as a general foreground detector: the top-down map has 3.7 phantom blobs per frame;
  - B1's colour shadows: VGGT rejects 0.67 and confirms 0.25, against the sweep's 0.75 / 0.12. The fringes at object
    edges are where shadows sit.
- **Keep as an option:** MASt3R matches triangulated with our cameras (1.8 / 8.6 mm, 0 phantoms, 46 % of objects) as a
  precise sparse check. It is 4 s per scene, and VGGT covers the same need.
- **What it does not fix:**
  - objects no home camera sees (4 of B1's 18 misses): they need a look;
  - thin parts below the model's resolution (coin, stand);
  - identity among look-alikes: not measured here. A 3D extent at 3-12 mm (median / p90) can only separate look-alikes
    whose heights differ by more than that.

## Reproduce

```bash
# compute3, GPU under ~/dojo-gpu-light3.lock; weights: VGGT-1B as bf16 safetensors + config.json in a directory
python -m dojo.perception.mv3d_eval infer --model vggt --bf16-trunk --weights DIR --root ~/dojo-data/d0_51 ~/dojo-data/d0_ext --views home --out OUT
python -m dojo.perception.mv3d_eval eval --out OUT --views home --tag home --workers 6       # CPU, GT after
python -m dojo.perception.mv3d_eval summary --out OUT --tag home --stereo ~/dojo-data/pbake-stereo/home_v5lr/frames.jsonl
python -m dojo.perception.mv3d_eval check --out OUT --views home --mode poly                 # B1 dump records
python -m dojo.perception.mv3d_eval check-summary --out OUT
python -m dojo.perception.mv3d_eval propose --out OUT --kw '{"max_loss": 0.5}'
python scripts/perception/mv/fpose_eval.py --pred OUT --out OUT_FP --classes gross overshoot identity ok missed
```
