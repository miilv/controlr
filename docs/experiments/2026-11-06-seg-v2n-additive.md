# Independent new-class segmenter with an immutable v2b base

**Hypothesis:** a separate new-only network can improve variant naming without the old-class
losses caused by fine-tuning one shared network. **PASS at the selected IoU 0.1**: zero
category/asset losers, 24/49 naming hits, and unchanged 35 phantoms. Camera success and
production adoption remain unmeasured. All three registered IoUs pass with identical metrics.

## Protocol and data

[Pre-registration](data/seg-v2n/recipe.json), initially committed as `a0b45237` before training
or additive measurements. No relabel arm. Extra confidence 0.5; dev mask-IoU grid 0.1/0.3/0.5.
Zero old-class category **and asset-index** losers in found, identity, pose10 and identity+10.
New naming target ≥ 18/49. Phantom tolerance declared before measurement: ≤ baseline +2 overall,
≤ +1 in any (set, task). Qualify old/phantom rules first, then maximize naming; ties choose
lowest IoU. Same five dev-only sets, frame 0, IDs 100000–04 and 100030–49; 162 frames / 975 objects.
No measurement, sealed or original gate image is training data.

`new_only_export.py` projects the complete round-2 export onto 32 disjoint new classes.
All 5,067 validated images retained: 4,816 train / 251 val (100049), with 3,346 / 171 deliberate
old-only negatives (empty labels). Every new class has positive training support.
RFS t=0.05 yields 5,842 training lines. All source layouts are positive variants of 100030–49;
the dev gate uses original draws. RGB remains the explicitly registered Isaac 6.0 proxy
with 5.1 physics; no genuine-domain substitution or new render is hidden here.
[Support and sampling hashes](data/seg-v2n/support.json), [class order](data/seg-v2n/classes.json).

The new network initializes from the 32 selected classifier rows of parked v2e, loads 561/561
weights, and trains independently. Maximum 40 epochs, patience 10, imgsz 960, seed 0,
close_mosaic 5, automatic AdamW. AutoBatch 39 hit the allocator cap and retried at 19;
≥ 6 GiB free VRAM remained. Early stopping at 37; selected epoch 27. The original registration
called best.pt a mask-metric selection; before any dev measurement we clarified that this
Ultralytics version actually sums box+mask mAP50–95. Both criteria selected epoch 27, so this
description correction had no checkpoint effect. [Training curve](data/seg-v2n/training-results.csv),
[resolved summary](data/seg-v2n/training-summary.json).

## Construction and base preservation

V2b's checkpoint/model is unchanged. Separate v2n detections are appended only when their
class belongs to its disjoint vocabulary and their mask IoU with **every** base detection
is at most theta. Every base mask/key/score/box remains an unchanged ordered prefix; no
combined NMS or detection cap can remove one. Extras carry provenance, require exact
public task candidates, and have lower priority than base entries during consumer dedup,
matching and proposal consumption. Wrist pools and pad extension remain base-only.
`EstimatorConfig.seg_additive` defaults false. Server options `--extra-weights/--extra-classes`
load the second network; enabled requests without it fail loudly.

The raw detector guarantee does not prove downstream pose/count invariance when additional
masks enter joint fitting. We verify the complete estimator separately below.
Serialized parity: **486 dev images / 2,055 base detections**, zero mismatches for standalone
v2b vs disabled-composition response and enabled-composition base prefix. Including 22
hitcheck images × 3 IoUs: 552 checks / 2,772 base detections, zero mismatches.

A fresh disabled baseline on the new code has zero errors and reproduces all 975 object
rows in found, pred_idx, identity, 3-D/XY/Z error, yaw and confidence exactly. All category
and asset metrics match, phantoms 35. [Category table](data/seg-v2n/v2b-off-per-class.md),
[asset table](data/seg-v2n/v2b-off-per-asset.md).

## Paired estimator gate

All 162 frames / 975 objects complete with zero errors at IoU 0.1. Found / identity / pose10 /
identity+10: **913 / 639 / 768 / 572**, versus **913 / 637 / 767 / 571**. There are zero
losers across all 90 category and 292 asset groups, including 68 category / 220 asset
groups containing old-class objects. All 756 old-class rows reproduce the checked detection,
identity, confidence and pose fields exactly; old totals stay **739 / 507 / 622 / 447**.

| changed asset group | n | found A→B | identity A→B | pose10 A→B | identity+10 A→B |
|---|---:|---:|---:|---:|---:|
| p60 / sort_nesting_dolls_by_size_random / matryoshka_dolls/00017 | 3 | 3→3 | 1→2 | 1→2 | 1→2 |
| p60 / sort_nesting_dolls_by_size_random / matryoshka_dolls/00019 | 3 | 3→3 | 1→2 | 1→1 | 1→1 |

Every other asset group's four metrics is unchanged. Phantoms remain **35**, with zero
change in each of 32 (set, task) groups; the declared tolerance is not needed.
[Full category table](data/seg-v2n/add-0.1-per-class.md),
[full per-asset table](data/seg-v2n/add-0.1-per-asset.md).

| mask IoU | category / asset losers | old category / asset losers | found / identity / pose10 / identity+10 | naming | phantoms |
|---|---:|---:|---|---:|---:|
| **0.1 selected** | 0 / 0 | 0 / 0 | 913 / 639 / 768 / 572 | 24/49 | 35 |
| 0.3 | 0 / 0 | 0 / 0 | 913 / 639 / 768 / 572 | 24/49 | 35 |
| 0.5 | 0 / 0 | 0 / 0 | 913 / 639 / 768 / 572 | 24/49 | 35 |

Full tables: [0.3 categories](data/seg-v2n/add-0.3-per-class.md),
[0.3 assets](data/seg-v2n/add-0.3-per-asset.md),
[0.5 categories](data/seg-v2n/add-0.5-per-class.md),
[0.5 assets](data/seg-v2n/add-0.5-per-asset.md),
[machine-readable gate](data/seg-v2n/gate-summary.json).
All naming scores tie, so the registered lowest-qualifying-IoU rule selects **0.1**.
The 756 old-class rows are exactly unchanged at every threshold, not just in aggregate.

## Genuine-dev naming

Unit: image-level GT scene membership, **not localisation, visibility filtering or independent
objects**. The 49 presences occupy 22 archived genuine-dev images and 16 asset keys. Base conf 0.25,
extra conf 0.5, inference 960. Historical v2d/v2e counts are references, not reruns this round.

| arm | named | merged | missed |
|---|---:|---:|---:|
| v2b |0|7|42|
| historical v2d |13|3|33|
| round-2 v2e |10|3|36|
| additive v2b+v2n, each registered IoU |**24**|13|12|

All 24 hits are on the common43-presence vocabulary; brick2/paper_ball0 remain0/6. Dolls 15–19
improve to8/15 and laptop 1 to2/3. Some wrong base names remain alongside correct extra names;
this is why naming is separate from count and phantom checks.

| asset | presences | named | merged | missed |
|---|---:|---:|---:|---:|
| Articulation/laptop/00001 | 3 | 2 | 0 | 1 |
| Articulation/toaster/00002 | 1 | 0 | 1 | 0 |
| Articulation/toaster/00004 | 6 | 5 | 0 | 1 |
| Geometry/cup_holder/00001 | 3 | 3 | 0 | 0 |
| Rigid/brick/00002 | 3 | 0 | 0 | 3 |
| Rigid/headset/00008 | 3 | 1 | 0 | 2 |
| Rigid/matryoshka_dolls/00015 | 3 | 1 | 2 | 0 |
| Rigid/matryoshka_dolls/00016 | 3 | 2 | 1 | 0 |
| Rigid/matryoshka_dolls/00017 | 3 | 2 | 1 | 0 |
| Rigid/matryoshka_dolls/00018 | 3 | 1 | 2 | 0 |
| Rigid/matryoshka_dolls/00019 | 3 | 2 | 1 | 0 |
| Rigid/mug/00012 | 3 | 0 | 3 | 0 |
| Rigid/mug/00014 | 3 | 1 | 2 | 0 |
| Rigid/mug/00017 | 3 | 2 | 0 | 1 |
| Rigid/paper_ball/00000 | 3 | 0 | 0 | 3 |
| Rigid/wuliangye/00001 | 3 | 2 | 0 | 1 |

[Complete per-image predictions and per-class counts](data/seg-v2n/hitcheck.json).

## Reproducibility and limits

Weights stay on compute3, `~/dojo-data/s-v2n-seg/runs/v2n/weights/best.pt`,
MD5 **be0817051b8efd0565218033cb647cec**. Immutable base
`~/dojo-w-ext/seg_v2b.pt`, MD5 **2f546196145fc424f76a5e2e47dd3210**.
[Checkpoint MD5s](data/seg-v2n/weights.md5), [inference/training source hashes](data/seg-v2n/source-sha256.json).
Torch 2.9.1+cu128 / Ultralytics 8.4.191. Heavy jobs serialize under `~/dojo-gpu.lock`,
four physical cores excluded from affinity, training 8 GiB reserve / 6 GiB hard floor,
10 GB disk guard retained. No light locks, system changes, compute4 access, weight upload,
LLM calls/tokens or episodes. Training occupied about 22 minutes; four complete bakes took 26.4 minutes (397/397/402/389s).
The complete GPU-lock occupancy was about 50 minutes, including data setup and detector checks.
All SEG launchers/servers exited; heavy lock free, 110 GB disk free, colleague PID3613 retained
its 2,418 MiB. All 650 frozen inputs/calibrations rehashed unchanged after the final gate.
[Final provenance and output hashes](data/seg-v2n/final-provenance.json).

Validation: local 34 passed / 8 optional skips; compute3 perception suite 59 passed with no
skips, including SAM/proposer consumers and classifier-row selection. CI passed for the
implementation and recorded detector results. The final documentation commit has its own
check. The round-2 required Isaac simulation suite passed 19 tests; no Isaac/robot source
changed in round 3.

[Camera A/B pre-registration](2026-11-06-seg-v2n-camera-ab.md): 50 gain pairs across the
five new-asset variant cells, 20 pairs on strong stack_blocks/insert_tubes guards, default
Camera A/B decision rule. The specification contains explicit estimator overlays and the
private-server requirement; it is unexecuted.
No episode videos exist because no episodes ran. Production defaults remain unchanged.
