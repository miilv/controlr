# 2026-10-27 — SAM bake-off: SAM 3.1 / SAM 2.1 as mask proposers for B1, and the numbered-series count bug

Stream SAM of the perception bake-off (`/tmp/dojo-team/PERCEPTION_BAKEOFF.md`). Journal:
[../journal/2026-10-27-dojo-sam.md](../journal/2026-10-27-dojo-sam.md). Code: `dojo/perception/sam_masks.py`,
`EstimatorConfig.sam_fg` / `sam_propose` / `series_count`, scripts in `scripts/perception/sam/`.

## Question

Does a SAM mask proposer improve analysis-by-synthesis perception on these failure classes:

- (a) not found / low contrast
- (c) shadow phantoms
- (d) clutter

Models tested:

- SAM 3.1: Comfy-Org's fp16 repack of `facebook/sam3.1`, with text (concept) and point prompts.
- SAM 2.1 hiera small / large: point prompts, REAL_RSI's pattern.

## Data and protocol

- **Frame-0 scenes:** only ids 100000-09 and 100020-49. No 100010-19, nothing >= 100100, no seed 2, no
  Eval_Layout / Traj.

  | set | scenes | renderer | backgrounds / colour map |
  |---|---|---|---|
  | `d0_51` (6 tasks) | 92 | genuine 5.1 | `bg_folds_51` / `color_calib_51` |
  | `d0_ext` (12 tasks) | 115 | 6.0 proxy | `bg_folds` / `color_calib` |
  | deposit_coin (perceiver stream's `d0_coin`) | 10 | 6.0 proxy | `bg_folds` / `color_calib` |
  | store_laptop_and_headphones 100000-04 | 5 | genuine 5.1 (Integration's run `c4_laptop_v3`) | `bg_folds_51` / `color_calib_51` |
  | sweep_blocks 100000-03 (`c4_sweep_v3`) | 4 | genuine 5.1 | `bg_folds_51` / `color_calib_51` |
  | rendered for the series question: sweep_blocks 100000-09 + 100020-24, classify_objects 100000-09 | 25 | 6.0 proxy (`render_series.sh`) | `bg_folds` / `color_calib` |

  - For the laptop and sweep runs, the images come from the agentic call dirs and GT from our settled records
    (`frames_from_run.py`).
- **B1:** main (da46879 + this branch's flags off), 2-fold backgrounds, wrists on. `scripts/perception/sam/bake.py`
  runs `inverse_layout.estimate` per frame and scores it with `eval_g1.score_record`. Classes follow
  `eval_g1.failure_class`: "within" = pose within tolerance (15 mm where a task has none), identity included.
  "Phantoms" = predictions matched to nothing, or more than 40 mm from every GT object.
- **Paired:** same frames, same code, one flag changed. 8 CPU workers (4 after 22:05Z, box memory). One SAM server
  process in the light1 slot.
- **Masks:** GT visible silhouettes rendered from the GT poses with the robot as occluder (`bake.py --dump`), then
  `masks.py`.

## 1. Masks: SAM vs B1's static-background foreground (scored objects with >= 30 visible head pixels)

Cells: mean IoU / recall@IoU 0.5 (n).

Columns:
- `B1 blob`: the foreground component overlapping the object most.
- `B1 fit`: B1's rendered silhouette.
- `fit`: REAL_RSI's proposer. Clicks at B1's centre + 4 offsets; the mask agreeing best with B1's model silhouette.
- `@B1 top`: SAM's top-score mask at B1's centre.
- `@GT best3`: an oracle click at the object's most interior pixel, best of 3 masks.
- `text`: best SAM 3.1 text mask for the scene vocabulary.

| class | n | B1 blob | B1 fit | 2.1-S fit | 2.1-L fit | 3.1 fit | 2.1-S @B1 top | 2.1-L @GT best3 | 2.1-S @GT best3 | 3.1 @GT | 3.1 text |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 1115 | 0.58 / 0.63 | 0.77 / 0.86 | 0.82 / 0.87 | 0.83 / 0.86 | 0.83 / 0.88 | 0.71 / 0.73 | 0.91 / 0.96 | 0.89 / 0.96 | 0.86 / 0.91 | 0.51 / 0.54 |
| (a) B1 fg covers < 50 % or missed | 99 | 0.26 / 0.19 | 0.33 / 0.41 (63) | 0.21 / 0.16 (63) | 0.21 / 0.17 | 0.19 / 0.13 | 0.19 / 0.16 | 0.60 / 0.66 | 0.61 / 0.61 | 0.57 / 0.52 | 0.31 / 0.34 |
| B1 found, >= 15 mm off | 108 | 0.44 / 0.43 | 0.24 / 0.18 | 0.37 / 0.39 | 0.38 / 0.40 | 0.40 / 0.43 | 0.32 / 0.33 | 0.81 / 0.89 | 0.80 / 0.89 | 0.75 / 0.81 | 0.58 / 0.66 |
| (c) deposit_coin | 29 | 0.39 / 0.48 | 0.28 / 0.21 (24) | 0.41 / 0.25 | 0.45 / 0.46 | 0.47 / 0.33 | 0.35 / 0.21 | 0.69 / 0.76 | 0.69 / 0.83 | 0.57 / 0.38 | 0.33 / 0.34 |
| (d) clutter scenes (pack_random targets) | 10 | 0.73 / 0.90 | 0.57 / 0.70 | 0.62 / 0.70 | 0.60 / 0.60 | 0.63 / 0.70 | 0.54 / 0.50 | 0.85 / 0.90 | 0.88 / 1.00 | 0.89 / 1.00 | 0.73 / 0.80 |
| sweep_blocks cubes (genuine 5.1) | 14 | 0.86 / 0.93 | - | - | - | - | - | 0.89 / 1.00 | 0.91 / 1.00 | 0.85 / 1.00 | 0.89 / 1.00 |

| scenes | n | B1 fg pixels off every object, median / mean | SAM 3.1 text pixels off every object | text masks | on nothing | on clutter |
|---|---|---|---|---|---|---|
| coin | 10 | 1645 / 1174 (the piggy bank's shadow) | 0 / 0 | 10 | 0 | 0 |
| clutter | 10 | 17025 / 15673 (distractors) | 0 / 526 | 17 | 3 | 1 |
| other | 206 | 176 / 1423 | 0 / 38 | 819 | 15 | 0 |

- **Given a right click, SAM's masks are far better segments than our foreground:** IoU 0.89-0.91 vs 0.58. This holds
  in every class. On the scenes where B1 is wrong, it's 0.60-0.81 vs 0.26-0.44.
- **The runtime has no right click.** Clicks at B1's own point inherit B1's error. Where B1's foreground fails (a),
  SAM at B1's point is no better (0.19-0.21), and where B1 is >= 15 mm off, 0.37-0.40.
- **REAL_RSI's model-fit selection picks the right mask:** "fit" 0.82-0.83 equals the best of 3 at B1's point.
- **SAM 3.1 text masks are clean** (0 pixels off-object in the medians, no shadows, 18 of 846 masks on nothing). But
  the vocabulary misses many objects (recall 0.54). Catalogue names like "t-block" and "desk card stand" don't hit,
  and the tiny coin and its stand are never found.

## 2. End to end (eval_g1 scorer), paired against B1

222 scenes, 1098 objects. sweep_blocks is left out here (see §3); the laptop scenes are included.

| arm | within tol | gross | missed | phantoms | fixed / broken (objects) |
|---|---|---|---|---|---|
| B1 | 961 | 85 | 25 | 49 | - |
| `sam_fg=union` (fg OR text masks) | 970 | 85 | 17 | 50 | 15 / 6 |
| `sam_fg=trim` without the cover guard (blobs touching a mask cut to it) | 973 | 79 | 28 | 38 | 32 / 20 (organize_table -9: merged blobs cut) |
| `sam_fg=gate` (fg only near a mask) | 871 | 82 | 128 | 41 | 36 / 126 (objects outside the vocabulary removed) |
| `sam_propose=sam2.1_small`, flip by IoU | 986 | 74 | 25 | 42 | 29 / 4 (4 lying bottles flipped) |

The fixes these arms led to (the proposer's flip only with texture agreement; trim only for blobs the masks cover by
>= 50 %) and the SAM 2.1-L / SAM 3.1 click arms are measured in a follow-up report.

Per class:

- **(a) low contrast / not found:** union recovers 8 of 25 missed (put_bottles 3, build_tower 2). The text vocabulary
  is what limits it.
- **(c) deposit_coin (30 objects):** B1 within 15. Union 15, trim 17, proposer 17.
  - The coin + stand phantoms 580-805 mm away (100000, 100008) are unchanged by every arm. SAM 3.1 text never finds the
    ~10 px assembly, and the proposer only refits locally.
  - What SAM fixes is the piggy bank, whose shadow biased its fit. Proposer: 100003 52 -> 1 mm, 100002 15 -> 2 mm.
    Trim also finds it on 2 scenes B1 missed.
- **(d) clutter (pack_objects_into_box_random):** union 52 -> 52, trim 52 -> 53, proposer 52 -> 55.
- **Moving belt:** not measured. The belt frames are on compute3 only.

## 3. The numbered-series count (sweep_blocks' "missed cubes", classify_objects)

B1 looked for one object per config label. sweep_blocks' `rigid2` becomes 3-5 cubes, classify_objects' `rigid0`
becomes 3-9 items; details in the journal. 29 scenes (sweep 19 = 15 rendered 6.0 + 4 genuine 5.1; classify 10),
eval_g1 scorer with series labels scored.

| `series_count` | sweep_blocks within / missed (107) | classify_objects within / missed (69) | phantoms | fixed / broken vs one |
|---|---|---|---|---|
| one (old) | 56 / 50 | 8 / 59 | 1 | - |
| min (config minimum, no SAM; the new default) | 92 / 14 | 25 / 39 | 1 | 46 / 0 |
| sam (SAM 3.1 text count, whole category sets, clamped) | 103 / 3 | 30 / 33 | 1 | 60 / 0 |

- On the 4 genuine-5.1 sweep scenes alone, within tolerance goes 12 / 22 (one), 19 (min), 21 (sam).
- classify_objects keeps headroom: SAM 3.1 text counts 1-5 of its small toys where 3-9 are there.
- `min` on d0_51 + d0_ext + coin + laptop (1098 objects): identical to B1, except one identity margin at the 4th
  decimal.

## 4. Laptop and headset (store_laptop_and_headphones, genuine 5.1, 5 scenes, 25 objects)

- **Proposer (SAM 2.1-S):** within 16 -> 18.
  - Headset: identity and pose fixed on 2 of 3 wrong ones. 21 mm / wrong variant -> 1.7 mm / right; 9 mm / wrong ->
    2.0 mm / right. The third is 222 mm off (a wrong blob) and stays.
  - Headphone stand: 8-17 mm / 4-10 deg -> 1-4 mm / < 1 deg on 5 of 5.
- **Laptop:** SAM 3.1 text segments it cleanly (score 0.94), but no arm changes B1's laptop. My GT for the articulated
  laptop (settled-record root pose) disagrees with the image by 150-270 mm / 175 deg even where B1's silhouette sits on
  it, so the laptop is not scored here. The flip is a shape-model question (open lid vs the catalogue mesh), not a mask
  question.

## 5. Speed and VRAM (RTX 5090, bf16)

Server timings over the mask runs:
- SAM 3.1 text: ~0.3-0.55 s per scene (image + 10-14 prompts).
- SAM 2.1 click: 26-39 ms, 3 masks.
- SAM 3.1 click: 180 ms through the tracker (a one-frame session per call).

The proposer adds ~0.5-2 s per scene on top of B1's 1-70 s. All three models in one process take 6.2 GB peak; SAM 3.1
alone 3.9-4.4 GB.

## Verdict

- **Adopt the count fix** (`series_count=min` by default, `sam` as an option): +46 / +60 objects, 0 broken. It's a bug
  fix, and SAM 3.1's count adds +14 on top.
- **SAM proposer (`sam_propose=sam2.1_small`):** an adopt candidate behind its flag (default off). It fixes the pose and
  identity of found objects (+25 net before the flip guard) but doesn't find missed ones. The guarded rerun is in the
  follow-up report.
- **SAM as a foreground replacement:** drop `gate` / `replace`. The text vocabulary misses too many objects. `union`
  and the guarded `trim` are small, task-dependent gains.
- **Needs:**
  - a better vocabulary than catalogue captions, plus a size prior for the tiny coin / stand;
  - a SAM server started next to the perception service. The flags go through the existing `DOJO_PERCEPTION_CFG`
    (DEVELOPMENT.md pitfalls).

## Spend and reproduction

- 0 LLM calls.
- GPU: the SAM server in light1 (one process), and 6 min of Isaac renders (25 scenes, 49 MB).
- Weights: 2.8 GB in `~/dojo-data/models`.
- Runs: `~/dojo-data/sam/<arm>/` on compute4 (`objects.jsonl`, `frames.jsonl`; `dump/` for the mask metrics).
- Setup: `scripts/perception/sam/setup_venv.sh`.
- Arms:
  - `python -m dojo.perception.sam_masks serve --sock ...`
  - `DOJO_SAM_ADDR=... python scripts/perception/sam/bake.py run --out DIR --cfg '{"sam_propose": "sam2.1_small"}'`
  - `bake.py summary DIR BASE`
- Masks: `masks.py run DUMP OUT.jsonl`, then `masks.py table OUT.jsonl`.
