# 2026-10-26 — what IS there: identity from held_look and the post-place refine

**Question.** The runtime asks held_look "is block_2 in the jaws?" and the post-place refine "is block_1 where I put
it?". When the answer is no, it needs to know which interchangeable candidate it holds or placed, so it can fix the
labels instead of undoing progress. How reliable is that identity reading, how sure must it be before the runtime
acts on it (the bar is conf >= 0.7 and margin >= 0.15), and does the wrist hover settle a block's size?

**Code.**
- `dojo/perception/identity.py`: `rank`, `pick`, `refine_what`, and the confidence tables.
- `j1_adapter.held_what`: the held_look `what` / `asked_held` / `scored` fields.
- `refine.refine_identity`: `look_alike`, and the bar for an identity change.
- `dojo/perception/val_what.py`: the calibration.
- No LLM calls. All GT is used only to build priors and to score.

## Setup

- **Data.**
  - **Wrist looks:** stack_blocks, 25 layouts, D0 frames 1-2, the left and right wrist looks. The same physics
    states are rendered by Isaac 6.0 (the proxy, `~/dojo-data/d0`) and by Isaac 5.1 (`~/dojo-data/d0_51`).
  - **Held frames:** a synthetic set, 6.0 proxy renders. 10 stack layouts × the D0 look poses, with each block in
    turn between the fingers, plus closed empty jaws: 90 held frames and 30 empty ones. Each held frame is asked
    three ways (asked = the held block, or one of the two others).
  - **5.1 replays:** Bridge-isr's held_look frames from 100001 / 100004 / 100007 / 100008.
- **Refine priors.** GT ±20 mm and ±10°, with the GT neighbours.
- **Candidates.**
  - "twins": the episode's same-category objects, which is what the runtime sends.
  - "group": all 5 assets of the episode's size group.
- **Score.** Colour: `1 - appearance_distance(observed mean colour, catalogue colour)`.
  - The refine reads the colour at an ok fit, else at the prior. It uses the renderer's colour map (6.0 or 5.1),
    with the fold held out by episode.
  - held_look reads the colour of the asked block's silhouette at its expected in-jaw pose, minus the fingers. It
    uses unmapped catalogue colours, with lightness weight 1.

## Results

### refine.best (post-place / any refine)

| colour source | renderer | n | twins top-1 | margin >= 0.15: share, precision |
|---|---|---|---|---|
| ok fit or prior | 6.0 | 89 | 0.989 | 0.83, 1.000 |
| ok fit or prior | 5.1 | 92 | 0.967 | 0.84, 0.974 |
| ok fit only | both | 154 | — | 132 / 132 right |
| prior (fit not ok) | both | 27 | — | 17 / 19 right |

- Texture discrimination on 5.1 wrist views is fine.
- The 5.1 errors are fits that were not ok (iou 0.25-0.45), read at the prior. That case gets its own, lower
  confidence table (`CONF_PRIOR`, at most 0.85).

### held_look.what (in-jaw view)

| colour statistic, map | twins top-1 | runtime bar (conf >= 0.7, margin >= 0.15): share, precision |
|---|---|---|
| mean, no map, lightness weight 1 (shipped) | 0.893 (n 270) | 0.71, 0.979 |
| mean, 6.0 head map, lightness 0.3 | 0.878 | — |
| mean, held map fitted on 6.0 (2-fold) | 0.911 | — (not shipped: fitted on one renderer) |

- held recall 270/270; empty jaws read not-held 90/90.
- **Label permutation, asked != held (180 asks).**
  - asked_held=False on 71 %, None on 24 %, True (wrong) on 5 %.
  - The runtime would repair 127 times, and the repair target is right 99.2 % of the time.
- **asked == held (90 asks).** asked_held=True 90 %, None 9 %, False (a wrong repair) 1 of 90.
- **5.1 replay held frames** (dark idx 12 / 13 blocks): what was right 6/6 at conf 0.97 (margin 0.67-0.75).
  asked_held was right for every asked label, and empty jaws gave what=None.
- `CONF_HELD`: margin < 0.25 → 0.6 (14/22 right); margin >= 0.25 → 0.97 (67/68 right).

### Size at the hover (`refine_identity`, the three size variants of the same texture, belief = a wrong size)

| adopt rule | adopted / 196 | precision |
|---|---|---|
| margin >= 0.05, ok (before) | 155 | 0.845 |
| margin >= 0.05, iou >= 0.85 | 135 | 0.919 |
| **margin >= 0.05, iou >= 0.9 (now)** | **115** | **1.000** |

- Most errors read a 30 mm block as 35 mm (24 of 41 confusions) on mediocre fits.
- Changing identity now takes iou >= 0.9. Otherwise the current identity's own fit is returned with
  `identity_ok=False` and `identity_rival`.
- `look_alike` keeps the texture when the size changes. Shape classes are represented by one key each, so the old
  code swapped textures too: 5.1 100006 went 15 → 7 instead of 16.

## Field case: 5.1 build_tower s0L9 (Bridge-isr strict, practice layout)

- **The hover refine_identity created the failure.**
  - Frame 0 / scan / agent gave all 8 physical objects their correct assets.
  - At the pick hover, a pillar (00002, 38 × 62 × 38 mm) became the prism (00003, 61 × 37.5 × 46) at iou 0.67,
    turned 91°. Their footprints from above are nearly equal.
  - The pillar then went into the tower standing on its 62 mm end. The board above it tilted, and the checker said
    "not built" while the belief said built.
- **The iou >= 0.9 bar blocks this change.**
- **The runtime should also restrict the candidates to the label's own config group.** In build_tower each label
  has exactly one index. That fix belongs to integration-2.

## Bridge-isr 5.1 calibration set (186 check frames, 20 dev-eval stack / tower episodes)

- **Set and scripts.** The frames are genuine Isaac 5.1 renders of the runtime's own checks:
  - 94 held_look frames, with truth "object in the jaws";
  - 92 post-place frames, with every object's true pose.
  - Scored by `dojo/perception/val_replay51.py`. "old" = main before #44, "new" = this branch.
  - No LLM calls.

### held_look

| | stack: in the jaws (31) held / not held / no verdict | stack: empty (5) not held / held | tower: in the jaws (58) held / not held / no verdict |
|---|---|---|---|
| runtime at the time (B1 + agent) | 77 / 12 over both tasks | 5 / 0 | (agent answered the B1 gaps, 6 "not held" at 0.9) |
| old B1 | 23 / 8 (6 at conf >= 0.5) / 0 | 5 / 0 | 31 / 0 / **27** |
| **new B1** | **31 / 0 / 0** | **5 / 0** | **58 / 0 / 0** |

- **Tower "no verdict" fix.** B1 skipped every arm commanded above 0.5. A board or pillar grasp commands 0.55-0.58, so
  27 of 58 tower holds had no B1 answer, and the agent then said "not held" on 6 of them. held_look now checks any arm
  commanded below 0.95 (`HELD_OPEN_CMD`).
- **The 3 unmatched 100006 holds count as in the jaws.** Bridge-isr could not match them by asset key, because the
  belief had the wrong size group. The object was 7 mm from the TCP.
- **what.**
  - stack: 28/28 right; 27/27 at the runtime bar.
  - tower: 58/58 "right", but only because the asked one ranks first on ties. Every tower piece has the same
    catalogue colour, so margin = 0 and nothing passes the bar, which is correct.

### Post-place refine

**Truth rule (target vs true pose, per task; `val_replay51.py`).**
- stack_blocks: the block is level (≤ 5°), and either rests on the table (the stack's base) or on another block with
  the height step within 5 mm and the xy offset within the task's stack threshold (15 / 17.5 / 20 mm).
- build_tower:
  - the checker's up axis is within 5°;
  - the piece rests on the table or on a top under it (5 mm);
  - block7 is within 43 mm of block0, block3 within 23 mm of block7, block4 within 23 mm of block3;
  - a pillar on block0 lies inside block0's footprint.
- Result: 90 well placed, 2 misplaced (a pillar off block0, a pillar tilted 56°).

**Prior** (the runtime's expected pose is not logged):
- well placed: the true pose ± 3 mm / ± 3°;
- misplaced: upright, on its support.
- The same prior is used for old and new.

| | stack: well (23) ok / not ok | tower: well (67) ok / not ok | ok but > 5 mm off | tower not-ok reasons |
|---|---|---|---|---|
| runtime at the time | 6 / 17 | 10 / 57 | — | — |
| old | 5 / 18 | 10 / 57 | 2 | truncated 26, fit 20, neighbour 11 |
| **new** | **20 / 3** | **19 / 48** | 4 | truncated 26, fit 15, slid out of view 7 |

**What changed on the way.**
- The table colour model excludes known non-support objects (`local_foreground(exclude=)`). A light-wood board in
  the ring made a white pillar's lit face "table".
- A support's pixels stay foreground only around the target's expected silhouette (15 px).
- A fit that leaves more than 10 % more of the object outside the view than the prior did is rejected (block3 slid
  36-44 mm along the same-colour block7, iou 0.7-0.8).

**Remaining tower not-oks.**
- Boards (24-37 cm) larger than the wrist view (`truncated`, unverifiable from the wrist).
- Pillars standing on the same-colour board block0 (fit, 14). Colour segmentation cannot separate them; that needs an
  edge or depth-like cue.

**D0 regression of these refine changes** (`val_refine.py`, 352 rows, priors GT ±20 mm, `WITH_OTHERS=1`):

| | ok | ok within 5 mm | ok > 10 mm off | ok xy p90 | worse than prior + 10 mm |
|---|---|---|---|---|---|
| main | 284 | 0.919 | 17 | 3.1 mm | 6 |
| this branch | 306 | 0.951 | 5 | 2.3 mm | 2 |

## Not measured / next

- held what on 5.1 beyond the 6 replay frames: needs more of Bridge-isr's regenerated held frames.
- Tasks other than stack_blocks: for example build_tower pillars, which are identical keys (`label` = None by
  design).
