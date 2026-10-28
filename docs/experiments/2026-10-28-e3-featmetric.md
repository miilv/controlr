# 2026-10-28 — E3 (research R3): a multi-view feature metric for B1's poses: a doubt signal, a continuous refine, candidate fusion (stream V)

Follows [2026-10-28-mv3d-learned-depth](2026-10-28-mv3d-learned-depth.md) and the research report
[2026-10-28-bop-sota-for-dojo](../research/2026-10-28-bop-sota-for-dojo.md), whose R3 / E3 asks for an AlignPose-style
multi-view feature-metric term in B1, with candidates fused by 3D NMS.

**Question.** Does appearance, compared at the pose across the calibrated views, help in three ways?
- Does it tell which of B1's objects are wrong?
- Does it refine them?
- Does it pick the right pose among candidates from other sources?

**Metric** (`dojo/perception/featmetric.py`). For an asset at a pose, and every view that sees it:
- crop: a square around the projection, rendered at 448 px;
- render: the catalogue per-face-colour mesh, lit from the camera, mapped through the renderer colour calibration of
  the frame's fold, composited over the observed crop, robot pixels kept in front;
- features: DINOv2-B patch tokens (224 px input, 16 x 16 patches) of the observed crop and of the composite.

The distance is mean (1 - cosine) over the patches the object covers, weighted by patch count across views. Candidates
compared together use one crop and their union of patches (`score`), or each its own crop when far apart
(`score_each`). Views: frame 0's head + both wrists.

**Data.** d0_51 + d0_ext frame 0, 207 scenes, ids 100000-09 / 100020-49. The B1 records are the SAM stream's dump
(current-main B1), and the B1 objects are scored with eval_g1. "Within" = ok + identity (pose within tolerance).

**Spend.** 0 LLM calls. GPU on compute2's light slot (DINOv2-B 0.5-0.9 GB; with VGGT 3.5 GB); CPU <= 4 cores.

## 1. Discrete yaw turns of B1's own pose: no gain

Candidates = B1's pose and its 90 / 180 / 270 deg turns about the object's centre. Switching to the best one when it
beats B1 by a margin:

| margin | 0 | 0.01 | 0.02 | 0.04 | 0.08 |
|---|---|---|---|---|---|
| within (930 before) | 915 | 923 | 927 | 928 | 930 |

Only 10 of B1's failures are rotation-only. The D0 look wrists (frames 1-3) do not change this: 917-929.

## 2. Diagnostic: with the right pose among the candidates, the metric picks it

The GT pose of the same asset was added as one more candidate. This is a dev diagnostic and is never applied.

| B1's class for the object | n | GT ranked better than B1's pose | ... by more than 0.02 |
|---|---|---|---|
| gross | 40 | 39 | 39 |
| overshoot | 12 | 12 | 9 |
| ok | 688 | 523 (76 %) | 194 (28 %) |

So for B1's gross errors the bottleneck is generating the right candidate, not scoring it.

## 3. A doubt signal: B1's own pose's distance (+ the learned height check)

The distance of B1's pose scored alone (its own patches; what `b1_doubt` ships) detects B1's wrong objects (gross or
overshoot: 95 of 1016) with **AUC 0.91**. Scored together with its yaw turns (their union of patches; the first
measurement) the AUC is 0.85.

| signal (pose alone) | flagged | precision | recall | false flags on right objects |
|---|---|---|---|---|
| d > 0.55 | 112 | 68 / 112 | 68 / 95 | 4.8 % |
| d > 0.6 | 82 | 60 / 82 | 60 / 95 | 2.4 % |
| learned height check \|dh\| > 15 mm | 95 | 42 / 95 | 42 / 95 | 5.8 % |
| d > 0.55 or \|dh\| > 15 mm | 165 | 73 / 165 | 73 / 95 | 10.0 % |
| **d > 0.6 or \|dh\| > 15 mm (the default)** | 138 | 67 / 138 | **67 / 95** | **7.7 %** |

- Shipped as `EstimatorConfig.b1_doubt` (`perception/doubt.py`, mv3d server op `featdist`), default off.
  - Defaults: `doubt_feat_d` 0.6, `doubt_dh_mm` 15, mode `or`.
  - For a cheaper gate: d > 0.6 alone flags 60 of 95 at 2.4 %.
- Consumers behind their own flags:
  - `AGENTIC_DOUBT_RISK` floors `j1.risk` for doubted objects. The gate, `needs_agent` and `risky_b1` all go through
    `j1.risk`.
  - `RuntimeConfig.look_doubt`: a targeted look.
- End to end on the real server:
  - organize 100002 doubts the alarm and the garage (both B1-gross);
  - pack 100000 doubts the shoe (B1-gross);
  - nothing else is doubted;
  - about 0.2 s per object.
- With D0's look wrists as extra views (frames 1-3, the union variant) the AUC does not improve (0.74 vs 0.78 on the
  same objects). The scripted looks aim at one object each and add mostly views that cut objects at the crop edge.
  Wrist frames taken at grasp time (stream C's sequences) are not measured.

## 4. The continuous stage: Nelder-Mead on the feature distance

`featmetric.refine`:
- parameters: x, y, yaw;
- each view's crop is fixed at the start pose (padded), and the observed features are computed once;
- 40 evaluations.

Run on B1 objects whose own distance exceeds a gate. The refined pose is kept when the distance drops by more than the
minimum gain.

| gate / objects | refined, kept | -> within | within -> worse | gross before -> after | net within |
|---|---|---|---|---|---|
| d > 0.4, all types, gain 0.02 | 431, 352 | 15 | 10 (ok -> overshoot 8) | -11 | +5 |
| **d > 0.45, Rigid only, gain 0.05** | 281, 134 | 14 | 5 (ok / identity -> overshoot) | **-9** | **+9** |
| d > 0.55, Rigid only, gain 0.05 | 88, 57 | 10 | 1 | -4 | +9 |

- No right object is broken into gross in any arm. The breaks are ok -> overshoot: the feature optimum sits a few mm off
  GT, enough to cross a tight tolerance.
- Fixes include:
  - the organize alarm, 46 mm moved (gross -> ok);
  - push_T's T, 16 mm (gross -> ok);
  - plug_in_charger's charger, 18 mm (gross -> ok);
  - stack_blocks blocks.
- Fixtures are excluded: the dustbin moved 23-25 mm in the all-types arm.
- Shipped as `EstimatorConfig.fm_refine` (0.45 / 0.05, Rigid only; server op `fmrefine`), default off.
- End-to-end smoke: § 6.

## 5. Arm 4: candidate fusion -> 3D NMS -> the feature metric

Gate: the doubt rule (d > 0.55 or |dh| > 15 mm). Seed sources:
- swaps: the other B1 objects of the same label group;
- raised regions of the learned height map, with their measured top height;
- B1's own head foreground blobs.

Each seed is fitted by B1's render-and-compare (up to 6 seeds, 4 yaw starts; silhouette loss <= the cap, lifted onto
the measured top when it stands on something). Candidates go through 3D NMS at 15 mm; the feature metric picks one
(each candidate scored on its own crop). Policies are applied offline (`scripts/perception/mv/e3_policy.py`) on the same
207 scenes, where B1 has 73 gross and 930 within:

| candidate sources | margin | silhouette loss cap | switched | within | gross | changes |
|---|---|---|---|---|---|---|
| foreground + raised (+ swaps: identical) | 0.1 | 0.6 | 8 | **930 -> 934** | **73 -> 68** | gross -> ok 3, -> identity 1, -> overshoot 1; 0 broken |
| foreground + raised | 0.05 | 0.6 | 14 | 930 -> 935 | 73 -> 68 | gross -> ok 4, -> identity 1, -> overshoot 1; overshoot -> gross 1 |
| foreground + raised | 0.02 / 0.05 | 0.3 | 5 | 930 -> 931 | 73 -> 72 | gross -> ok 1 |
| + yaw turns of B1's pose | 0.05 | 0.6 | 32 | 930 -> 933 | 73 -> 70 | ok -> gross 2 (yaw turns break right objects) |

- Swap seeds never won: the same-group objects' places fit the moved shape worse than B1's own pose.
- Phantoms: 37 -> 35-37.
- About 2-20 s per scene in one process (the seed fits dominate, about 0.3 s each with GL).
- **The candidates are the limit.** B1's own foreground and the learned raised regions rarely hold the right pose of a
  gross object; most of them are swaps (hidden tower blocks, pack items inside the box) and occlusions.
- Stream S's twin-trained detector (key + mask per detection) plugs in as one more seed source (`--dets`). Its GT-mask
  oracle through B1's proposer fixed 59 / 1013. That run follows when its frame-0 detections are ready.

## 6. End to end (B1 + `fm_refine` through the server)

Current-main B1 on compute2's CPU, with the mv3d server (VGGT bf16 + DINOv2-B) in the light slot, and
`EstimatorConfig(fm_refine=True)` vs off on the same frames:

| scene | off | fm_refine | refined / kept | cost |
|---|---|---|---|---|
| organize_table 100000 | gross 1 (alarm), missed 1 | **gross 0** (alarm -> ok) | 1 / alarm | +0.3 s |
| plug_in_charger 100004 | gross 1 (charger) | **gross 0** (charger -> ok) | 1 / charger | +0.7 s |
| push_T 100001 | gross 1 (t) | unchanged (its distance stays below 0.45) | 0 | 0 |

## Verdict

- **ADOPT, flags default off:**
  - `b1_doubt`, the signal: 67 / 95 of B1's wrong objects at 7.7 % false flags (AUC 0.91), no LLM.
  - Its two consumers: `AGENTIC_DOUBT_RISK` for the gate and `risky_b1`; `look_doubt` for the look macro.
  - `fm_refine` (0.45 / 0.05, Rigid only): gross -9, within +9 net, 0 right objects into gross.
  - Next: a rendered A/B, Integration.
- **Arm 4 candidate fusion with B1-foreground / raised seeds: small** (+4 within, gross -5, 0 broken at margin 0.1).
  Keep it as the frame for detector candidates; not wired into B1.
- **DROP:** yaw-turn candidates. They break right objects at every margin.
- **NEEDS:** better candidates for gross objects (S's detector), and a view of the hidden ones. The diagnostic says
  the metric will pick them once they exist (39 / 40).
