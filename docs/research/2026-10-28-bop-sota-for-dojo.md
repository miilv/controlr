# 2026-10-28 — BOP state of the art (2023-2026) for unseen-object pose, mapped onto RoboDojo perception

Stream: Research (desk only). No downloads, no installs, no LLM API calls. Every number below comes from the cited
page; anything I could not check is marked *unverified*. Leaderboard numbers are a snapshot of bop.felk.cvut.cz taken
on 2026-10-28.

Question: our perception problem is close to BOP's "model-based 6D localization of unseen objects". What is the 2026
state of the art, and what should we adopt?

Our setting, for reference:
- we have the CAD mesh of every object (the public catalogue);
- 3 calibrated RGB cameras: head 640x480, fx ~288; two wrist cameras, fx ~397, with poses exact from FK;
- no depth sensor; our plane-sweep stereo gives heights (0.5 mm on covered tops);
- B1 (`inverse_layout.py` + `refine.py`) is a multi-view silhouette render-and-compare with table contact, task counts
  and colour identity;
- an agentic LLM perceiver.

## Summary

1. **Our task is BOP's "6D localization", not "6D detection".** BOP tells localization methods which objects are in
   the image and how many; our task config gives B1 the same (labels and counts). The closest BOP setups are the
   model-based unseen-object track (CAD only, onboarding in 5 min per object on one GPU), plus the 2025 multi-view
   track on BOP-Industrial.
2. **BOP and our bake-off agree on the bottleneck: finding and identifying objects, not refining them.**
   - The BOP 2024 report calls 2D detection of unseen objects "the primary bottleneck" of these pipelines.
   - The best unseen-object 2D detector today scores AP_Core 0.580 (3PT, CVPR 2026). A detector trained on synthetic
     renders of the same objects scores 0.769 (GDet2023-PBR).
   - Our bake-off found the same: B1's refine is already 0.3-1.5 mm once seeded, and its misses are found / identity /
     visibility.
3. **Most of BOP's RGB vs RGB-D gap does not apply to objects resting on our table.**
   - Best RGB-D: 0.845 AR (WAPR.v2). Best RGB-only: 0.682 (Co-op).
   - Depth mostly buys translation along the camera ray. We get that for free from the table-plane contact that B1
     already uses.
   - The gap does apply to stacked, elevated and held objects, which is where our stereo heights belong.
4. **The clearest 2025-2026 result is multi-view.**
   - BOP 2025: "multi-view clearly outperforms single-view".
   - AlignPose (CVPR 2026, code public) refines one world-frame pose by DINOv2 feature-metric alignment across
     calibrated RGB views: Co-op 69.7 -> 83.8 AR on YCB-V and 68.2 -> 89.6 on T-LESS (4 views); FoundPose 31.4 ->
     79.8 AP on IPD.
   - Its main assumption is exact extrinsics, which we meet better than BOP does (FK, no calibration error).
5. **Resolution is our specific handicap.** At about 2.4 mm per head pixel, a 4 cm block is about 17 px and a 2 cm coin
   about 8 px. BOP objects are typically 60-300 px. DINOv2-based template matching uses 14 px patches, so it has
   almost nothing to work with in our head view. At a 12-20 cm hover the wrist camera gives about 0.3-0.5 mm/px, which
   is BOP-like or better.
   - This plausibly (not measured) explains why DINOv2 identity helped the belt but not table B1 on head crops
     (303 -> 300 within tolerance).
   - So appearance features belong in the wrist views.
6. **Most of the top methods are not usable.**
   - No public code found (*unverified* where noted): GenFlow, Co-op, MUSE, 3PT, FreeZeV2.
   - Depth required: FoundationPose, SAM-6D's pose stage, FreeZe, WAPR.
   - Of the open RGB methods, the best (PicoPose / FoundPose / GigaPose + MegaPose, ~0.58-0.60 AR_Core) are not better
     than what B1's refine does once seeded.
7. **Recommendation, ranked by expected gain per engineering hour (details in §5):**
   1. **R1 — a grasp-time check of identity, flip and existence in the wrist view.** It uses CNOS / MUSE-style DINOv2
      scoring against renders at the believed pose, on SAM 2.1 masks: 1-1.5 days, 0 LLM calls.
   2. **R2 — make the twin-trained detector (stream S) the main proposer.** This is BOP's "seen-object" track, and our
      sim-to-sim domain gap is about zero.
   3. **R3 — add AlignPose-style multi-view feature-metric refinement to B1's joint stage.** It covers flips, merged
      blobs and yaw, with head + scan / look wrists: 2-4 days.

   Not now: learned single-view refiners (MegaPose / GenFlow / FoundationPose) on our partial stereo depth. They come
   after R1-R3, and only for stacked or held objects.

## 1. How BOP maps onto our problem

| BOP concept | BOP setting | ours |
|---|---|---|
| task | 6D localization: object ids and instance counts given; 6D detection: not given | ids / counts from the task config, so localization. `_random` held-out instances are closer to detection |
| onboarding | CAD only, <= 5 min per object on 1 GPU (since 2023) | public catalogue meshes, unlimited offline time |
| test images | real photos; training data = synthetic PBR (BlenderProc) | Isaac renders of the same assets: the PBR-to-real gap that BOP fights does not exist for us |
| resolution | e.g. YCB-V fx ~1066, LM-O fx ~572, objects ~60-300 px | head fx 288, ~2.4 mm/px at ~0.68 m (4 cm = ~17 px); wrist fx 397, ~0.3-0.5 mm/px at a hover |
| depth | top methods all RGB-D | none; plane-sweep stereo heights where 2 views overlap (coverage 0.35 at home, 0.75 at the scan pose) |
| views | single view, except BOP-Industrial 2025 (multi-view, poses given) | 3 calibrated views per frame, plus looks / scan and every frame of the episode |
| priors | none | table plane, task counts, slots, robot geometry from FK (known occluder), only touched objects move |
| metric | AR = mean of AR_VSD, AR_MSSD, AR_MSPD over thresholds of 5-50 % of the object diameter (MSPD in px) | strict tolerance per task (7-23 mm, 5-7 deg) plus identity, at grasp time (stream M) |

Reading AR: 0.80 AR does not mean "80 % within our 15 mm". AR averages recall from very tight to loose thresholds. It
also does not score identity confusions the way our tasks do: in localization the id is given.

## 2. Survey

### 2.1 Leaderboard state (BOP-Classic-Core = LM-O, T-LESS, TUD-L, IC-BIN, ITODD, HB, YCB-V)

**Model-based 6D localization of unseen objects** (AR_Core; time = s per image, on the participant's own GPU):

| method | date | input | AR_Core | s/img | code |
|---|---|---|---|---|---|
| WAPR.v2 (multi 2D detections) | 2026-03 | RGB-D | **0.845** | 50.8 | yes (WAPR, see 2.2) |
| FRTPose-WAPR.v2 | 2025-10 | RGB-D | 0.844 | 212.7 | not found |
| FreeZeV2.2 | 2025-05 | RGB-D | 0.833 | 21.7 | not found (*unverified*) |
| FreeZeV2.1 (2024 winner) | 2024-11 | RGB-D | 0.821 | 24.9 | not found (*unverified*) |
| WAPR.v2.1 (MUSE detections, fast mode) | 2026-02 | RGB-D | 0.806 | 0.72 | yes |
| Co-op (F3DT2D, 1 hypothesis) | 2024-11 | RGB-D | 0.759 | 0.77 | not found |
| FoundationPose | 2024-08 | RGB-D | 0.734 | 29.3 | yes |
| SAM-6D | 2023-12 | RGB-D | 0.704 | 4.4 | yes |
| GenFlow (2023 winner) | 2023 | RGB-D | 0.674 | 34.6 | not found |
| **best RGB-only: Co-op (F3DT2D, 5 hypotheses)** | 2024-11 | RGB | **0.682** | 3.9 | not found |
| Co-op (CNOS, 1 hypothesis) | 2024-11 | RGB | 0.640 | 1.7 | not found |
| GigaPose + GenFlow | 2023-11 | RGB | 0.605 | 10.6 | GigaPose yes, GenFlow no |
| PicoPose + MegaPose (5 hypotheses) | 2025-07 | RGB | 0.599 | n/a | yes |
| FoundPose + FeatRef + MegaPose | 2024-01 | RGB | 0.596 | 20.5 | yes (without FeatRef) |
| OPFormer + MegaPose | 2025-01 | RGB | 0.598 | 1.7 | not found |
| GigaPose + MegaPose | 2024-02 | RGB | 0.578 | 7.7 | yes |
| B2TFPose (training-free, DINOv3; paper, not on the board) | 2026-09 | RGB | 0.564 | n/a | not found |
| MegaPose (CNOS-FastSAM) | 2023 | RGB | 0.509 | 31.7 | yes |
| GigaPose coarse only | 2024 | RGB | 0.268 | 0.4 | yes |

For comparison, seen objects (trained on renders of the target objects), 6D localization:
- best RGB-D: Flose 0.865, GPose2023 0.856, GPose2023-PBR (synthetic only) 0.844;
- best RGB-only: HccePose 0.773 (0.1 s);
- synthetic-only RGB: ZebraPose 0.720, GDRNPP-PBR-RGB 0.702.

**Model-based 2D detection of unseen objects** (AP_Core):

| method | date | input | AP_Core | s/img | code |
|---|---|---|---|---|---|
| 3PT-Detection (CVPR 2026) | 2025-10 / 11 | RGB | **0.580** / 0.579 | 7.5 / 1.6 | N/A on the board |
| CLON | 2026-06 | RGB-D | 0.552 | 5.5 | unknown |
| MUSE (WACV 2026) | 2025-08 | RGB | 0.533 | 0.5 | not found |
| F3DT2D | 2024-09 | RGB | 0.509 | 0.43 | unknown |
| SAM6D-FastSAM (ISM) | 2023-12 | RGB-D | 0.471 | 0.45 | yes |
| NIDS-Net (WA + Sappe) | 2024-05 | RGB | 0.469 | 0.49 | yes, MIT |
| CNOS (FastSAM) | 2023-08 | RGB | 0.428 | 0.22 | yes, MIT |
| CNOS (SAM) | 2023-08 | RGB | 0.361 | 1.85 | yes, MIT |

Seen objects (trained detector), same metric:
- GDet2023 0.798 (PBR + real);
- **GDet2023-PBR 0.769 (synthetic only)**;
- GDRNPPDet_PBR (YOLOX) 0.738 at 0.08 s.

So training on synthetic renders of the exact objects beats the best unseen-object detector by about 19 AP points on
real images. For us, where the test frames come from the same renderer, it should be at least that.

**BOP-H3 (HOT3D / HOPEv2 / HANDAL, 2024):** everything drops.
- MUSE: 39.0 AP (2D detection).
- GigaPose + GenFlow: 31.2 AP (6D detection).
- Model-free GFreeDet (3D Gaussians from onboarding videos): 31.9 AP.

**BOP-Industrial, multi-view (2025):** the challenge page states "multi-view clearly outperforms single-view". The BOP
2025 final awards are a Google Drive PDF that I could not read. Who won which track is taken from the winners' own
papers (*unverified* against the award list):
- 3PT says it ranked first in 7 of 11 tracks;
- MUSE says it was first in 2D detection on Classic-Core, H3 and Industrial.

The current board shows 3PT ahead of MUSE on Classic-Core.

**Related 2025 challenge (OpenCV Bin-Picking Challenge, real robot validation).** The winner was a per-class
YOLOv12-X detector, then SAM 2 masks, then a trained "SAM6D-industrial" pose model with multi-view fusion. Two of their
findings:
- object-specific training did worse than unseen-object training, on real industrial data;
- the metric choice changed the ranking.

The first finding comes from a real-data domain gap that we do not have. It is still a warning for `_random`
held-out instances.

**BOP 2026 (ECCV 2026; final submissions due 27 Nov 2026) switches to language grounding (BOP-Refer).** The organisers'
first evaluation of frontier VLMs found:
- the best 2D grounding at 53.7 % AP;
- Qwen3-VL at 2.6 % AP in 3D on the same images and queries.

This independently matches our measurement that the LLM perceiver is a good finder and a poor pose estimator, and that
B1-style geometry must supply the pose.

### 2.2 Method cards

VRAM is rarely reported. Where a number is ours it comes from the 10-27 bake-off; "n/r" = not reported.

| method | role | inputs | best number (BOP) | runtime | VRAM | code / weights | licence | fit for us |
|---|---|---|---|---|---|---|---|---|
| **CNOS** (ICCVW 2023) | unseen 2D detection / segmentation | RGB + templates rendered from CAD (count configurable; 42 by default, from memory, *unverified*) | 0.428 AP (FastSAM), 0.361 (SAM) | 0.22 / 1.85 s per image | n/r | github.com/nv-nguyen/cnos | MIT. FastSAM proposals are AGPL-3.0 (Ultralytics); SAM / DINOv2 Apache-2.0 | reference for template scoring; swap the proposals for our SAM 2.1 |
| **MUSE** (arXiv 2510.17866, WACV 2026) | unseen 2D detection | RGB; SAM2 + Grounding DINO proposals; DINOv2 class + patch tokens (GeM), absolute + relative similarity, uncertainty-aware prior | 0.533 AP_Core; 0.39 AP on H3 | 0.5 s per image | n/r | no code found | paper CC BY 4.0 | **its scoring recipe** is what to copy (+21 % over CNOS) |
| **3PT** (CVPR 2026) | unified detection / segmentation / pose; multi-view refinement, no depth | RGB, multi-view | 0.580 AP_Core; first in 7 of 11 BOP 2025 tracks (its claim) | 1.6-7.5 s per image | n/r | code N/A | — | not adoptable; shows that multi-view RGB without depth can lead |
| **NIDS-Net** (IROS 2025) | unseen detection | RGB; Grounding DINO + SAM ViT-H + averaged DINOv2 patch features + a small weight adapter | 0.469 AP | 0.49 s | n/r | github.com/YoungSean/NIDS-Net, weights downloadable | MIT (deps: Grounding DINO Apache-2.0, SAM Apache-2.0) | alternative to CNOS; the adapter idea suits our unlimited renders |
| **SAM-6D** (CVPR 2024) | detection (ISM) + pose (PEM) | RGB-D (PEM needs depth) | 0.704 AR; ISM 0.471 AP | 4.4 s (FastSAM 1.4 s) | n/r | github.com/JiehongLin/SAM-6D | no licence file shown (*unverified*) | ISM ≈ CNOS + geometric score; PEM needs depth |
| **FoundationPose** (CVPR 2024) | pose estimation + tracking, model-based or model-free | **RGB-D required** | 0.734 AR | paper: 1.3 s per object registration (0.88 refine + 0.42 select), tracking ~32 Hz; TensorRT library: 148 ms register / 2 ms track | n/r | NVlabs/FoundationPose (weights on Google Drive); TensorRT inference library nvidia-isaac/foundation-pose-inference-library; HF nvidia/foundationpose | NVlabs repo: NVIDIA Source Code License (non-commercial). Inference library: Apache-2.0, weights NVIDIA Open Model License (commercial OK) | only with our stereo depth; strongest open tracker |
| **WAPR** (arXiv 2610.09535, ECCV 2026) | RGB-D refiner with a ±90 deg basin; FoundationPose ScoreNet picks the pose | RGB-D crop 160x160 + mask, metric mesh | 0.845 AR (multi-det, 50.8 s); 0.806 fast mode (MUSE, 0.72 s); LM-O / YCB-V 75.5 / 89.3 vs FoundationPose 67.4 / 80.8 with the same 12 hypotheses | ~40 ms per instance on an RTX 5090 (paper); 88-90 instances / s TensorRT FP16 (README) | n/r | `pip install wapr`, github.com/WangYuLin-SEU/WAPR, 4 checkpoints downloaded at first run | code LGPL-2.1; weights CC BY-ND 4.0 | best open refiner; needs depth in the crop |
| **FreeZe / FreeZeV2.x** (ECCV 2024; arXiv 2506.09784) | training-free; DINOv2 + GeDi 3D descriptors, RANSAC registration | RGB-D | 0.821-0.833 AR | 17-25 s per image | n/r | FreeZe GitHub is the website only; no V2 code found (*unverified*) | paper CC BY 4.0 | no |
| **GenFlow** (CVPR 2024, NAVER LABS) | recurrent-flow refiner | RGB or RGB-D | 2023 winner: 0.674 RGB-D / 0.576 RGB | 34.6-40.5 s | n/r | no public code found | — | no |
| **Co-op** (CVPR 2025, NAVER LABS) | template correspondences + PnP, then probabilistic-flow refiner | RGB (or RGB-D) | best RGB 0.682 AR; RGB-D 0.759 at 0.8 s | 1.7-3.9 s (RGB) | n/r | no public code found | — | no |
| **GigaPose** (CVPR 2024) | fast coarse pose (out-of-plane from templates + 1 correspondence) | RGB + templates | 0.268 coarse; 0.578 + MegaPose; 0.605 + GenFlow | 0.4 s coarse | n/r ("memory issues fixed") | github.com/nv-nguyen/gigapose, ckpt + templates on HF nv-nguyen/gigaPose | MIT | a coarse seed for refiners; B1 already seeds better on the plane |
| **MegaPose** (CoRL 2022) | coarse classifier + render-and-compare refiner | RGB (optional depth), mesh, rough box | 0.509 AR (RGB, CNOS); 0.628 RGB-D + Teaser | 31.7 s per image (RGB); 142 s (RGB-D + Teaser) | n/r | megapose6d/megapose6d (weights on an Inria server); maintained fork agimus-project/happypose | Apache-2.0; happypose BSD-2 | old and slow; B1's refine is better once seeded |
| **FoundPose** (ECCV 2024, Meta) | DINOv2 patch features, template retrieval, PnP-RANSAC, feature-metric refinement | RGB | 0.373 coarse (1.7 s); 0.426 + FeatRef; 0.596 + MegaPose | 1.7-20.5 s | n/r | facebookresearch/foundpose (released without the feature-metric stage) | CC BY-NC 4.0 | candidate generator inside AlignPose |
| **PicoPose** (CoRL 2025) | 3-stage pixel-to-pixel correspondence | RGB + templates | 0.599 + MegaPose | n/r | n/r | github.com/foollh/picopose, ckpt on Google Drive | MIT (badge) | similar to GigaPose |
| **AlignPose** (CVPR 2026, CTU / CIIRC) | **multi-view feature-metric refinement** of single-view candidates into one world-frame pose, 3D NMS before and after | RGB, ≥ 2 calibrated views, mesh; DINOv2-L layer 18 (DINOv3-L ≈ same) | YCB-V 69.7 -> 83.8 AR, T-LESS 68.2 -> 89.6 (Co-op init, 4 views; CosyPose MV reaches 81.0 / 78.7); IPD 31.4 -> 79.8 AP, ITODD-MV 41.2 -> 76.8 | 0.56 s per detection (L40S), linear in candidates; onboarding < 5 min per object (788 templates -> PCA) | n/r | github.com/mikestikova/alignpose_project (code public: templates, representations, refinement, BOP eval) | **no licence file** (treat as research-only); deps FoundPose (CC BY-NC), PixLoc (*licence unverified*), DINOv2 (Apache-2.0) | **closest published method to B1**; failure cases: views too similar, a candidate off in depth (thin cutlery), near-symmetric texture |
| **CosyPose** (ECCV 2020) | object-level multi-view RANSAC + bundle adjustment (unknown camera poses) | RGB; detector and pose nets trained per dataset | T-LESS seen 4 views 72.8 AR (synthetic only) vs 64.0 for 1 view | ~0.3-0.4 s per scene (bundle adjustment, from AlignPose) | n/r | ylabbe/cosypose (unmaintained) -> happypose | MIT | superseded by AlignPose for our case (we know the camera poses) |
| **SAM3-FS-FP** (BOP entry, Sep-Oct 2026) | SAM 3 text masks + FoundationStereo depth from 1-6 extra views (baselines 0.10-0.24 m) + FoundationPose; sensor depth ignored | multi-view RGB | T-LESS 0.803 AR (172 s / img); YCB-V 0.531 (27 s / img); RTX PRO 6000 | as left | n/r | nvidia-isaac/foundationpose_perception_pipeline: Ubuntu >= 24.04 (glibc 2.38), driver >= 580, CUDA >= 12.8 | pipeline Apache-2.0; SAM 3 = Meta SAM License (gated); FoundationStereo research weights non-commercial (TAO / NGC version commercial) | proof that RGB stereo can feed FoundationPose; the result depends on the dataset |
| **GoTrack** (CVPRW 2025, Meta) | RGB refiner / tracker: DINOv2-based model-to-frame flow + PnP-RANSAC | RGB, mesh, initial pose | refinement AR LM-O 56.5 / T-LESS 50.4 / YCB-V 63.1 | n/r | n/r | facebookresearch/gotrack (checkpoint in the repo); frame-to-frame tracking "currently not implemented" | CC BY-NC 4.0 | possible RGB tracker; non-commercial |
| **M3T / ICG / SRT3D** (DLR, 2020-2023) | region (silhouette / colour) + depth + texture tracking; multi-body (kinematic chains), multi-camera; known occlusions via renderers | RGB and / or depth, mesh, initial pose | tracking benchmarks (RBOT, YCB-V; numbers not checked) | real time on CPU (*unverified*: ~1 ms per frame from memory) | CPU | DLR-RM/3DObjectTracking (C++); pym3t binding is single-camera | MIT | the reference for what our refine already does; articulated (Mb-ICG) |
| **BundleSDF** (CVPR 2023) | model-free RGB-D tracking + neural reconstruction | RGB-D video | — | "near real-time" | n/r | NVlabs/BundleSDF | NVIDIA licence (restrictive) | no: we have the meshes |
| **OVE6D** (CVPR 2022) | depth-based viewpoint encoding | depth + mask | old | — | — | dingdingcai/OVE6D-pose | licence not shown | no |
| **SAM 3D Objects** (Meta, Nov 2025) | single-image generative shape + pose | RGB | — | — | — | facebookresearch/sam-3d-objects | SAM License | no: we have the meshes |
| **Detectors for the twin** | trained 2D detection (BOP's seen track) | RGB | GDRNPPDet YOLOX: 0.738 AP synthetic only (0.08 s) | ~0.1 s | small | YOLOX (Megvii, Apache-2.0); GDRNPP shanice-l/gdrnpp_bop2022 (Apache-2.0); RT-DETR (Apache-2.0); D-FINE (Apache-2.0, *verify*) | Ultralytics YOLO / FastSAM are AGPL-3.0 (matters only if the system is distributed) | stream S |
| **FoundationStereo / Fast-FoundationStereo** (CVPR 2025 / 2026) | learned stereo (stream V's domain) | rectified pair | — | Fast: 23-49 ms at 640x480 on a 3090, 0.65 GB; full ViT-L ~2 s, 8-12 GB (community numbers, *unverified*) | as left | NVlabs | research weights non-commercial; TAO / NGC commercial | the matching degrades beyond ~45 deg of rectification rotation (pipeline README), so head <-> wrist pairs are a poor fit; wrist <-> wrist across two arm poses is better |

Feature backbones: DINOv2 (Apache-2.0, ungated) is enough. AlignPose's ablation puts DINOv3-L within a point of DINOv2-L,
and DINOv3 has a custom licence and gated downloads.

## 3. Mapping onto our failure classes

Our numbers come from the 10-25 options report (625 objects, D0 frame 0) and the 10-27 bake-off (1098 objects, B1
961 within / 85 gross / 25 missed / 49 phantoms).

| class | our data | what the 2026 SOTA offers | verdict |
|---|---|---|---|
| (a) not found / low contrast (dark camera on the dark belt, edge blocks) | missed 10 + 7 of 625; SAM 2.1-S proposer 28 fixed / 1 broken; text masks miss tiny objects | trained detector (0.769 AP synthetic only) >> unseen detectors (0.58 best, 0.43-0.47 open). SAM-family proposals + template scoring (CNOS / MUSE) are the unseen fallback | **R2** (stream S) first; CNOS / MUSE-style scoring as the fallback for `_random` held-out instances |
| (b) look-alike identity (camera / phone / garage, mugs, headset, tower shapes) | identity 36 of 625 (22 % of failures); DINOv2 head crops: belt 0.46 -> 0.83, table B1 303 -> 300 | MUSE: class + patch tokens with GeM and relative similarity, +21 % over CNOS's class token. Every method matches at 60-300 px; ours are 8-40 px in the head view | **R1**: score identity on the wrist view at hover, against renders at the refined pose. Not in the head view |
| (c) shadows read as objects (piggy-bank shadow -> coin / stand) | stereo rejects 0.75 of B1's table pixels; slot_bases 11/11 | template matching rejects shadows (no template matches); SAM 3 text masks had 0 shadow px | covered by our stereo check; BOP adds nothing new |
| (d) moving belt | background subtraction fails; belt stereo ~20 phantoms per frame (glossy, striped) | BOP is static. Trackers: FoundationPose (needs depth: unreliable on the belt), GoTrack (RGB, non-commercial, no frame-to-frame), M3T region tracking (RGB, CPU, MIT) | stream C: B1 refine seeded by the previous pose + belt velocity (= the M3T idea), plus the adopted DINOv2 belt identity |
| (e) heights / z (place dz -38..-62 mm, stacked, held, tilted toast) | stereo tops 0.5 mm with scan coverage | the RGB -> RGB-D gap (0.68 -> 0.85 AR) is mostly translation along the ray. SAM3-FS-FP: stereo depth feeding FoundationPose works on T-LESS (0.803) but not YCB-V (0.531) | stereo heights into B1's z for elevated / stacked objects. WAPR / FoundationPose (R4) only where stereo coverage is good |
| (f) occlusion by grippers / outside the camera overlap | 63 of 162 failures (39 %); 6 of the 11 residual gross after the agent | AlignPose refines anything seen in ≥ 1 view, but fails when every view sees it the same way or occluded. M3T models known occluders by rendering (B1 already masks the robot via FK) | information-limited: scan pose / looks and the tracker (both measured). Multi-view fusion (R3) uses the views once they exist |
| (g) thin objects / 180 deg flips | 15 flips, all thin pack items | thin cutlery is AlignPose's own failure case; depth is poor on thin objects; WAPR's basin is ±90 deg, not 180 | **R1** at hover: score the flip hypothesis with appearance (2 renders). R3 for frame 0 / scan |
| (h) articulated (laptop on its stand) | B1 flip fixed by `on_base_refit` (PR #200) | BOP is rigid-only. Mb-ICG / M3T track kinematic chains from a URDF; FoundationPose does not | keep the per-configuration rigid models; Mb-ICG only if lid angles start to matter |

## 4. Constraints

- **GPU** (5090 32 GB or 4090 24 GB, shared with Isaac at ~7-8 GB per rendered episode). R1-R3 fit:
  - DINOv2-B/L: ~0.3-1.1 GB of weights;
  - SAM 2.1-S: 0.55 GB, measured;
  - AlignPose: renderer + DINOv2-L.

  These do not fit, or are unknown:
  - FoundationPose / WAPR: n/r, *unverified*;
  - SAM 3.1: 4.7 GB, measured;
  - the full FoundationStereo ViT-L: 8-12 GB, *unverified*.
- **Wall time.** The simulator is paused while we perceive, so only wall time counts (J1_TIMEOUT_S 300):
  - B1: 1-10 s;
  - agent: 31-47 s median;
  - AlignPose: ~0.56 s per object, so ~5 s for a 10-object scene;
  - R1 at a hover: a few renders + DINOv2 on 2-4 crops, < 1 s.
- **RGB only.** Every depth-hungry method needs our stereo:
  - FoundationPose, SAM-6D's PEM, FreeZe, WAPR, OVE6D;
  - coverage is 0.35 at home and 0.75 at the scan pose, and the belt is unreliable.
- **Calibration** is exact (published head constants + FK). This is the main assumption of AlignPose and CosyPose-style
  fusion, and the thing that BOP's real rigs get wrong.

## 5. Recommendation

### What to keep from the sketched pipeline, and what to drop

The sketched pipeline was:
1. SAM 3.1 masks + DINOv2 template matching against catalogue renders;
2. MegaPose / GenFlow / FoundationPose refinement fed by stereo depth;
3. multi-view fusion across head + wrist;
4. B1 as the fallback.

What to do with each part:
- **Keep 1**, but on the **wrist** views, with renders at the believed pose and SAM 2.1-S clicks. The bake-off
  measured SAM 2.1-S clicks above SAM 3.1 text: 28 fixed / 1 broken vs 28 / 5.
- **Keep 3**, as a feature-metric term inside B1 (AlignPose).
- **Make B1 the core, not the fallback.** Its refine is already at mm level once seeded, it uses priors that no BOP
  method has, and every learned refiner with code is either RGB at ~0.5-0.6 AR or needs depth.
- **Drop 2 for now:**
  - GenFlow has no code;
  - MegaPose is a slow 2022 refiner;
  - FoundationPose / WAPR need depth that we have only on covered tops.

  Revisit 2 (as R4) for stacked / elevated / held objects once stereo depth in crops is measured.

### Ranked list (expected gain per engineering hour)

| rank | what | why (evidence) | targets | engineering | runtime / VRAM | repos / weights (licence) |
|---|---|---|---|---|---|---|
| **R1** | **Grasp-time wrist check**: at the pre-grasp / pre-place hover, for the target, its reference and every neighbour within R. (a) **Existence**: a SAM 2.1-S mask at the projected belief, or a flag (re-look). (b) **Identity + flip**: render every candidate asset of the label group, and the 180 deg flip, at the refined pose in the wrist camera, then score observed vs render with DINOv2 (MUSE recipe: CLS + GeM-pooled patch cosine + relative similarity across candidates, swap only with a margin). Extend `refine.refine_identity`, which today scores only silhouette residual and mean colour | the metric is grasp-time; the hover view is the only one at BOP-like resolution (~0.3-0.5 mm/px); with the pose known, matching is render-vs-render at the same viewpoint, which CNOS has to search over 40+ views to get; MUSE's scoring is +21 % over CNOS on BOP | identity (22 % of failures), flips (15), phantoms within R | 1-1.5 days (`embed_id.py`, the SAM server and `refine.py` exist) | < 1 s per hover; DINOv2-B ~0.3 GB or -L ~1.1 GB, SAM 2.1-S 0.55 GB | facebookresearch/dinov2: `dinov2_vitb14` / `vitl14` (HF facebook/dinov2-base / -large), Apache-2.0. facebook/sam2.1-hiera-small (on the boxes), Apache-2.0. Reference code: nv-nguyen/cnos (MIT). MUSE arXiv 2510.17866: recipe only |
| **R2** | **Twin-trained detector as the main proposer** (stream S): boxes + masks + asset class (+ 2 heading keypoints) on all 3 cameras, including home-gripper occlusions and hover / scan wrist views. Feed B1 like the SAM proposer. Keep CNOS / MUSE scoring only as the `_random` fallback | BOP's largest documented gap: 0.769 AP synthetic-only trained vs 0.580 best unseen (0.43-0.47 open). Our sim-to-sim gap is about zero. The BPC 2025 counter-result (trained < unseen) is a real-data effect | missed, identity, flips: upper bound 11 % of objects (10-25 report) | stream S's plan (3-4 days + ~12 GPU-h); my input is the design and eval only | ~0.1 s per image, small | YOLOX: Megvii-BaseDetection/YOLOX, or GDRNPP's YOLOX config (shanice-l/gdrnpp_bop2022, 73.8 AP synthetic only), both Apache-2.0. Or RT-DETR (lyuwenyu/RT-DETR) / D-FINE (Apache-2.0). Masks from SAM 2.1 box prompts (the BPC winner's YOLO -> SAM 2 pattern). Not GDRNPP's pose head: synthetic-only RGB 0.702 AR is no better than B1 seeded |
| **R3** | **Multi-view feature-metric refinement in B1's joint stage** (AlignPose): add a DINOv2-L layer-18 feature-metric residual (PCA-reduced, robust loss, features rendered online) over head + any wrist / scan / look views to B1's silhouette loss. Fuse candidates from B1, SAM, the agent and the detector by 3D NMS before and after refinement | BOP 2025: multi-view wins. AlignPose: +14-21 AR over single view on YCB-V / T-LESS and IPD 31 -> 80 AP, more than CosyPose MV everywhere (YCB-V +14 vs +11, T-LESS +21 vs +11, IPD +48 vs +5 AP). Appearance resolves what silhouettes cannot (flips, merged blobs, T / pad yaw) | visible-but-wrong 48 (30 % of failures), flips 15, yaw | 2-4 days: reimplement the loss in our renderer; run their code on a few scenes as the reference | ~0.56 s per object (L40S); DINOv2-L ~1.1 GB | github.com/mikestikova/alignpose_project (no licence file: research use; ask the authors before copying code). Deps: facebookresearch/foundpose (CC BY-NC 4.0), PixLoc (*licence unverified*), DINOv2 (Apache-2.0). A reimplementation avoids the entanglement |
| R4 | Stereo-depth-fed RGB-D refiner, only for stacked / elevated / held objects and where scan-pose coverage is good | the RGB-D lead is translation along the ray, which the table prior already gives resting objects. SAM3-FS-FP shows stereo -> FoundationPose varies by dataset (0.80 vs 0.53) | z errors, tilted pieces | 3-5 days | WAPR ~40 ms per instance (5090); FoundationPose TRT 148 ms | WAPR (`pip install wapr`; code LGPL-2.1, weights CC BY-ND 4.0). Or nvidia-isaac/foundation-pose-inference-library (Apache-2.0) + HF nvidia/foundationpose (NVIDIA Open Model License). compute4 has driver 580; the Ubuntu >= 24.04 requirement is stated for the Intrinsic pipeline, *unverified* for the library alone |
| R5 | Belt / held tracking | M3T-style region tracking = our refine seeded by the last pose and the belt speed. FoundationPose tracking needs depth | belt (d), held | stream C | CPU | DLR-RM/3DObjectTracking (MIT) as the reference |

**Don't adopt:**
- GenFlow, Co-op, MUSE (as code), 3PT, FreeZeV2: no code found;
- MegaPose: 0.51 AR RGB at 32 s per image, unmaintained since 2023;
- CosyPose MV: per-dataset training; AlignPose supersedes it when the camera poses are known;
- SAM-6D PEM, OVE6D, BundleSDF, SAM 3D Objects: they need depth, or solve reconstruction we don't need;
- DINOv3 instead of DINOv2: same accuracy in AlignPose's ablation, gated, custom licence.

## 6. First experiments (on our data; 0 LLM calls)

**E1 — R1 offline on grasp events (light GPU slot, ~1 day).**
- Data:
  - stream M's grasp / place events from the saved rendered runs: the wrist frame at the hover / pre-close, plus the
    runtime's belief at that moment;
  - where M's runs lack hover frames: the 71 scan-pose layouts and the dev_d0 look frames, with the believed pose
    perturbed by B1's own error distribution.
- Arms:
  1. today's `refine_identity` (silhouette residual + colour + `look_alike`);
  2. + DINOv2-B CLS cosine against renders at the refined pose;
  3. + the MUSE recipe (CLS + GeM patch + relative similarity);
  4. as 3 with DINOv2-L;

  plus the flip hypothesis and the SAM 2.1-S existence check in arms 2-4.
- Metrics:
  - identity top-1 per object kind;
  - flips fixed / broken;
  - phantom rejection (precision / recall);
  - events turned clean / broken on M's metric at R = 5 / 10 / 15 cm;
  - all binned by the crop's pixel size, which gives the accuracy-vs-pixels curve we never measured.
- Adopt if it fixes events with ≤ 1 broken per 100 events, n ≥ 10 paired per task.

**E2 — R2 vs an unseen-object baseline on the same frames (with stream S, ~0.5 day of my side).**
- Data: d0_51 + d0_ext frame 0 (207 frames) plus scan / look wrist frames, ids 100000-09 / 100020-49.
- Arms:
  - B1 foreground blobs;
  - CNOS-style (our SAM 2.1 automatic masks + DINOv2 against catalogue templates; CNOS's code, proposals swapped);
  - S's twin detector.
- Metrics: recall@IoU 0.5, identity top-1, phantoms, by view and by object size in px (8-16 / 16-32 / 32-64 / 64+),
  then eval_g1 end to end with each as B1's proposer.
- This decides whether an unseen-object detector earns its place at all (only for `_random` held-out instances), and on
  which views.

**E3 — R3: feature-metric multi-view refinement vs B1's silhouette refine (light GPU, 2-3 days).**
- Data: d0_51 + d0_ext frame 0, and the 71 scan layouts (head + scan wrists).
- Seeds: B1's poses, and B1 + the SAM proposer.
- Arms:
  1. B1;
  2. B1 + feature-metric term, head only;
  3. B1 + feature-metric term, head + wrists;
  4. arm 3 + 3D NMS fusion of B1 / SAM / agent candidates.
- Metrics: eval_g1 within / gross / flip / identity per task, fixed / broken, s per scene, VRAM.
- Reference: AlignPose's own code on 2-3 scenes (private venv), as a correctness check of the reimplementation.

## 7. Not verified

- **The BOP 2025 final award list** (Google Drive PDF) and the workshop recording (YouTube): not read. Track winners are
  from the winners' own papers and the live leaderboard.
- **Code availability:**
  - "No code found" for GenFlow, Co-op, MUSE and FreeZeV2 means none was found in searches and on the BOP method pages.
  - 3PT's board entry says "Implementation: N/A".
- **Licences:**
  - SAM-6D: no licence file shown;
  - OVE6D: not shown;
  - AlignPose: no LICENSE file;
  - PixLoc and D-FINE: not checked.

  For an internal benchmark entry the non-commercial licences (FoundationPose NVlabs, FoundPose, GoTrack,
  FoundationStereo research weights) are probably fine. Re-check them if the system is ever distributed.
- **VRAM** for every learned method, except our own measurements and the Fast-FoundationStereo README.
- **Unchecked figures:** CNOS's default template count; M3T's CPU speed; full FoundationStereo's speed / VRAM
  (community wrapper).
- **The SAM3-FS-FP entry:** its board columns were read via the per-submission pages (T-LESS 0.803, YCB-V 0.531). It
  has no publication.
- **All expected gains in §5 are mechanisms with upper bounds** from our failure counts, not measurements. E1-E3 are
  there to measure them.

## Sources

- BOP 2024 report: [arXiv 2504.02812](https://arxiv.org/abs/2504.02812) (tables of tracks 1-7); BOP 2023 report:
  [arXiv 2403.09799](https://arxiv.org/abs/2403.09799).
- Leaderboards (snapshot 2026-10-28):
  - [6D localization, unseen, Core](https://bop.felk.cvut.cz/leaderboards/pose-estimation-unseen-bop23/core-datasets/);
  - [2D detection, unseen, Core](https://bop.felk.cvut.cz/leaderboards/detection-unseen-bop23/core-datasets/);
  - [6D localization, seen](https://bop.felk.cvut.cz/leaderboards/pose-estimation-bop19/bop-classic-core/None/);
  - [2D detection, seen](https://bop.felk.cvut.cz/leaderboards/detection-bop22/bop-classic-core/None/).
- BOP 2025: [challenge page](https://bop.felk.cvut.cz/challenges/bop-challenge-2025/),
  [R6D 2025](https://cmp.felk.cvut.cz/sixd/workshop_2025/). BOP 2026 / BOP-Refer:
  [R6D 2026](https://cmp.felk.cvut.cz/sixd/workshop_2026/).
- BPC 2025 winners: [Huang et al., ICCVW 2025](https://openaccess.thecvf.com/content/ICCV2025W/R6D/papers/Huang_Lessons_and_Winning_Solutions_in_Industrial_Object_Detection_and_Pose_ICCVW_2025_paper.pdf)
  (abstract and figure text via search; the PDF itself returned 403).
- Detection / segmentation:
  - [CNOS](https://github.com/nv-nguyen/cnos), [arXiv 2307.11067](https://arxiv.org/abs/2307.11067);
  - [MUSE arXiv 2510.17866](https://arxiv.org/abs/2510.17866);
  - [3PT, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Kalra_3D-Object_Perception_Transformer_3PT_CVPR_2026_paper.html);
  - [NIDS-Net](https://github.com/YoungSean/NIDS-Net);
  - [FastSAM licence](https://github.com/CASIA-LMC-Lab/FastSAM/issues/185).
- Pose:
  - [FoundationPose](https://github.com/NVlabs/FoundationPose), [arXiv 2312.08344](https://arxiv.org/html/2312.08344),
    [inference library](https://github.com/nvidia-isaac/foundation-pose-inference-library),
    [HF nvidia/foundationpose](https://huggingface.co/nvidia/foundationpose);
  - [WAPR arXiv 2610.09535](https://arxiv.org/html/2610.09535), [WAPR code](https://github.com/WangYuLin-SEU/WAPR);
  - [SAM-6D](https://github.com/JiehongLin/SAM-6D);
  - [FreeZe arXiv 2312.00947](https://arxiv.org/abs/2312.00947), [FreeZeV2 arXiv 2506.09784](https://arxiv.org/abs/2506.09784);
  - [GenFlow, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Moon_GenFlow_Generalizable_Recurrent_Flow_for_6D_Pose_Refinement_of_Novel_CVPR_2024_paper.html);
  - [Co-op arXiv 2503.17731](https://arxiv.org/abs/2503.17731);
  - [GigaPose](https://github.com/nv-nguyen/gigapose);
  - [MegaPose](https://github.com/megapose6d/megapose6d), [happypose](https://github.com/agimus-project/happypose);
  - [FoundPose](https://github.com/facebookresearch/foundpose);
  - [PicoPose](https://github.com/foollh/picopose);
  - [B2TFPose arXiv 2609.06726](https://arxiv.org/abs/2609.06726);
  - [OVE6D](https://github.com/dingdingcai/OVE6D-pose);
  - [GDRNPP](https://github.com/shanice-l/gdrnpp_bop2022).
- Multi-view:
  - [AlignPose arXiv 2512.20538](https://arxiv.org/abs/2512.20538), [project page](https://mikestikova.github.io/alignpose/),
    [code](https://github.com/mikestikova/alignpose_project);
  - [CosyPose](https://github.com/ylabbe/cosypose);
  - SAM3-FS-FP: [method](https://bop.felk.cvut.cz/method_info/1534/),
    [T-LESS submission](https://bop.felk.cvut.cz/sub_info/41978/), [YCB-V submission](https://bop.felk.cvut.cz/sub_info/43028/),
    [pipeline](https://github.com/intrinsic-opensource/foundationpose_perception_pipeline).
- Tracking:
  - [GoTrack](https://github.com/facebookresearch/gotrack), [arXiv 2506.07155](https://arxiv.org/abs/2506.07155);
  - [DLR 3DObjectTracking (M3T / ICG)](https://github.com/DLR-RM/3DObjectTracking);
  - [BundleSDF](https://github.com/NVlabs/BundleSDF).
- Other:
  - [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects);
  - [FoundationStereo](https://github.com/NVlabs/FoundationStereo), [Fast-FoundationStereo](https://github.com/nvlabs/fast-foundationstereo);
  - DINOv3 licence: [issue #28](https://github.com/facebookresearch/dinov3/issues/28);
  - detector licences: [RT-DETR / D-FINE / Ultralytics comparison](https://www.eurthtech.com/post/is-yolo-free-for-commercial-use-agpl-apache-alternatives).
- Ours:
  - `docs/experiments/2026-10-25-perception-options.md`, `2026-10-27-identity-embeddings.md`,
    `2026-10-27-sam-bakeoff.md`, `2026-10-27-stereo-height.md`, `2026-10-27-perceiver-models.md`;
  - `/tmp/dojo-team/PERCEPTION_95.md`, `PERCEPTION_BAKEOFF.md`;
  - `dojo/perception/inverse_layout.py`, `refine.py`, `cameras.py` (the head and wrist intrinsics and poses behind the
    mm/px figures).

Spend: 0 LLM API calls (this desk session's own model only), no downloads, no installs. About 45 web searches and fetches.
