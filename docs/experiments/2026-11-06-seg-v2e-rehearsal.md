# Segmenter v2e: complete rehearsal support, zero-loser dev gate FAIL

Continues the [dev-gate integrity report](2026-11-05-seg-v2e-dev-gate.md).
**v2b remains active. v2e is parked; no camera A/B is qualified.**
Full recipe, recovery history and checkpoints: [journal](../journal/2026-11-06-dojo-seg-v2e-round2.md).

## Setup and spend

compute3 only, seed 0, expanded v2b head with its exact 511-name/id prefix plus
32 classes. RGB is explicitly **Isaac 6.0 proxy**, with 5.1 physics and Isaac
instance labels; genuine 5.1 RTX still crashes at startup. This domain change
was registered before rendering/training. No compute4 access or model promotion.

5067 labelled images: 4816 train / 251 val (100049); new variant draws on
100030–49 only. All 511 old classes have ≥1 train image: min / median / max
1 / 19 / 1627. Coverage is not equivalent to adequate diversity. The control
keeps 1723 images, shares every appended-class image and its RFS repetitions,
and omits only old-only rehearsal images. It has 324 unsupported old classes;
it does **not** reproduce v2d's deleted data (140 unsupported old classes).

[Recipe](data/seg-v2e-dev/round2-recipe.json),
[class order](data/seg-v2e-dev/classes-v2e.json),
[full support](data/seg-v2e-dev/support-full-round2.json),
[control support](data/seg-v2e-dev/support-narrow-round2.json),
[manifest/sampling hashes](data/seg-v2e-dev/training-data-sha256.json),
[final code/checkpoint/gate provenance](data/seg-v2e-dev/final-provenance.json).

Both short arms: two epochs, batch 16, imgsz 960, seed 0, same expanded init.
v2e: fresh eight epochs from that init, close_mosaic 3, RFS 0.02, **last.pt**.
AutoBatch selected 16 under an 8 GiB allocator reserve; actual training never
triggered the 6 GiB free-memory stop. GPU training was ~0.22 h total, with no
power-throttle stalls. Three candidate gates took ~19.2 min. LLM calls/tokens:
**0 / 0**. No episodes ran, so there are no success/failure episode videos.

## Paired dev-only gate

All arms: same 162 frame-0 inputs / 975 objects; ids 100000–04 and 100030–49,
five sets; zero frame errors. A clean v2b bake replaced the invalid archive;
explicit PyOpenGL ctypes-handler registration fixed the five press_by_number
errors. No failed task was dropped. Rule, frozen before training: zero losers
in found, identity, pose ≤10 mm and identity+10, at category **and asset-index**
resolution, plus positive genuine-dev new-class naming gain.

| arm | found | identity | pose10 | identity+10 | category / asset losers vs v2b | phantoms |
|---|---:|---:|---:|---:|---:|---:|
| v2b | 913 | 637 | 767 | 571 | — | 35 |
| narrow2 | 900 | 628 | 754 | 560 | 10 / 15 | 36 |
| rehearsal2 | 909 | 635 | 764 | 570 | 4 / 5 | 36 |
| **v2e** | **912** | **637** | **763** | **569** | **6 / 7 — FAIL** | **37** |

Full tables: [v2b](data/seg-v2e-dev/v2b-clean-per-asset.md),
[narrow categories](data/seg-v2e-dev/narrow2-per-class.md),
[narrow assets](data/seg-v2e-dev/narrow2-per-asset.md),
[rehearsal categories](data/seg-v2e-dev/rehearsal2-per-class.md),
[rehearsal assets](data/seg-v2e-dev/rehearsal2-per-asset.md),
[v2e categories](data/seg-v2e-dev/v2e-per-class.md),
[v2e assets](data/seg-v2e-dev/v2e-per-asset.md).

The seven v2e asset losers (unchanged metrics omitted):

| set/task/asset | n | loss vs v2b |
|---|---:|---|
| d0_51/put_bottles_into_dustbin/bottle/00000 | 3 | pose10 and identity+10 2→1 |
| p60/play_Xylophone/mallet/00000 | 5 | pose10 and identity+10 3→2 |
| p60/play_Xylophone/mallet_stand/00000 | 5 | pose10 and identity+10 3→2 |
| series/classify_objects/toy_car/00004 | 2 | identity, pose10 and identity+10 2→1 |
| w51/arrange_largest_number/cube_cushion/00001 | 24 | found, identity, pose10 and identity+10 22→20 |
| w51/arrange_largest_number/number/00006 | 1 | found and pose10 1→0 |
| w51/arrange_largest_number/number/00026 | 1 | found and pose10 1→0 |

## New-class hitcheck

Genuine archived dev frame-0, conf ≥0.25, imgsz 960. Unit: image-level GT asset
presence from scene membership, **not localisation, visibility filtering or independent physical objects**. All 49 presences
(16 asset keys) are within v2e's vocabulary; none is within v2b's.

| arm | named | merged | missed |
|---|---:|---:|---:|
| v2b | 0 | 7 | 42 |
| v2e | 10 | 3 | 36 |

On the 43 presences covered by v2d's 30-class increment, v2e names **10/43**;
historical v2d named 13/43 (not re-run here). The added brick/paper-ball six
presences contribute no hits. v2e hits: toaster 4 **5/6**, cup_holder 1 **1/3**,
headset 8 **1/3**, mug 14 **2/3**, mug 17 **1/3**. Other classes remain at zero,
including dolls 15–19 **0/15** and laptop 1 **0/3** despite additional data.
Training head-view support exists: dolls 15–19 have 43–52 images each; laptop 1
has 39. [View support](data/seg-v2e-dev/new-focus-view-support.json).
All per-asset counts: [text](data/seg-v2e-dev/hitcheck-v2e.txt),
[JSON](data/seg-v2e-dev/hitcheck-v2e.json).

## Interpretation and disposition

Adding old-only rehearsal reduces short-run regressions, but the same arrange
number/cushion detection loss remains in both short arms and v2e. Nonzero
support alone does not repair it. The dev garage identity/pose10 is retained
3→3 by both rehearsal arms, versus 3→2 in the narrow control. The excluded
historical garage frame 100009 was never used for tuning or this decision.
This single-seed, short proxy ablation supports rehearsal as a contributing
mechanism; it does not prove why historical v2d failed or establish generalisation.

v2e fails the registered rule. No camera A/B spec is issued, no weights become
default, and PR #387 remains draft for lead review. Follow-up: isolate mask /
series-count sensitivity or incremental optimization; do not just append more
classes or relax the gate. No further checkpoint was selected using these outcomes.

Weights remain on compute3 under `~/dojo-data/s-v2e-seg/runs/NAME/weights/last.pt`:

| NAME | MD5 |
|---|---|
| narrow2 | `23d5ab8ebafff047a16145e0faeda229` |
| rehearsal2 | `63137e506be837d13b7a6150c7afffcd` |
| v2e | `9de6416f2e517e1ff91f56cbc5b47237` |

Validation: 53 focused tests passed / 14 optional-dependency skips locally;
compute3's earlier focused suite passed all 60 tests. Required Isaac simulation
checks: **19 passed in 313.74 s**, private port 17821; server stopped. CI passed
for code/result commit `926bf1d1`. The earlier bounded local broad suite did not
complete and had one fkagent failure that passed independently on retry; no
local full-suite pass is claimed. Heavy GPU lock released, all our jobs/servers
exited, colleague's original GPU service and desktop remained alive. Disk free
at cleanup: 110 GB; no system changes or disk-floor override.
