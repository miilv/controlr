# 2026-10-27 — Height from calibrated views: a plane-sweep height map and a height check (perception bake-off, Stereo stream)

**Question.** We know every camera pose exactly: the head camera is fixed, and the wrists come from FK, which is exact in
sim. Can the official RGB plus the public calibration give us depth, as a stereo rig would? Would that depth help with
the bake-off's failure classes:
- (a) objects not found or low in contrast;
- (c) shadows and phantoms read as objects;
- (e) heights?

**Method** (`dojo/perception/stereo.py`, CPU only, no learned parts).
- **Plane sweep.** The head view is the reference. For each candidate height h above the table (-1 to +30 cm: 2 mm
  steps up to 6 cm, then 4 mm; 95 planes), each other view is warped onto the head through the plane z = table + h.
- **Cost.** (1 - ZNCC) / 2 over a 7x7 grey window, then a 5x5 box over the cost. Per height, every pixel keeps its
  best view, so a point hidden in one view can still match in another.
- **Per-pixel outputs.**
  - `height`: the best h with a sub-step parabola, NaN unless the minimum is unique.
  - `on_table`: the h = 0 plane explains the pixel.
  - `proud`: h = 0 fails and a plane >= 1.2 cm explains it.
- **Left-right consistency** (`lr`). Every wrist also sweeps with itself as the reference against the head. A proud
  pixel then needs a view whose own height, at the matched pixel, agrees. This removes occlusion halos: table next to
  an object, hidden in a wrist view, otherwise matches the object's texture at some raised plane.
- **Height check** (`check_record`). Each record object's mesh at its pose gives a predicted surface height on its
  front-most head pixels (GL depth per object, else a CPU ray cast).
  - `dh` = median of measured minus predicted over its pixels with a measured height.
  - `mismatch` = |dh| > 4 mm with >= 8 measured pixels. With no measured height, the object's decided pixels lying on
    the table counts as a mismatch.

**Data** (dev only; GT = our own layout records, read by the scorers only; ids 100000-09 / 100020-49).
- d0_51 + d0_ext frame 0 (207 frames, arms home).
- The saved frames of the proxy integ runs: frame 0, the scan pose and the look pose. That is 71 layouts over 15 tasks;
  B1's records come from the same episodes.
- deposit_coin, 10 layouts, from those runs.
- sweep_blocks, rendered 5.1 on compute4, 5 layouts (the agentic workspaces' frame 0).
- belt51, genuine 5.1 conveyor frames (67).

**Spend.** 0 LLM calls, 0 GPU jobs, ≤ 8 cores on compute3, < 0.2 GB of data.

## Results

**Height accuracy, top-facing surfaces** (|n_z| > 0.9, median of estimate - GT per object):

| data / views | frames | coverage of the object region | objects measured | median abs err | within 5 mm | s / frame |
|---|---|---|---|---|---|---|
| d0_51 + d0_ext, head + home wrists | 207 | 0.35 | 631 | 1.3 mm | 0.69 | 1.4 sweep (3.8 incl. CPU robot masks) |
| integ, frame 0 (home) | 71 | 0.34 | 270 | 0.9 mm | 0.76 | 5.3 (with lr) |
| integ, scan pose (head + scan wrists) | 71 | 0.75 | 372 | 0.5 mm | 0.87 | 5.5 |
| integ, frame-0 head + home, scan and look wrists | 71 | 0.76 | 354 | 0.5 mm | 0.89 | 13.3 |

- **Tops.** Blocks, tower pieces, charger and tools: 0.1-0.9 mm median.
- **Failures.**
  - Vertical or thin features (bottles -35 mm, press_by_number's upright cards, insert_key) fail: a horizontal-plane
    window cannot match a vertical face.
  - So do textureless tops (the white block's interior, the piggy bank): those pixels get no verdict.
- **Coverage.** The home wrists see only the table beyond y ≈ -0.15. The zone in front of the grippers, where most
  objects sit, needs the scan pose. The scan pose gives 0.75 coverage at no extra steps when the scan triggers.

**Foreground "proud > 1 cm" vs GT, d0 home** (pixels, frames with complete GT):

| variant | recall | precision | phantom blobs / frame | B1 colour-FG table pixels: rejected / wrongly confirmed |
|---|---|---|---|---|
| plain | 0.75 | 0.46 | 2.6 | 0.75 / 0.20 |
| + lr (tol 6 mm + 15 %) | 0.39 | 0.63 | 0.9 | 0.75 / 0.04 |
| + lr (tol 10 mm + 25 %) | 0.63 | 0.54 | 1.4 | 0.75 / 0.12 |

- **Detection.** Colour already finds 687 of the 690 covered objects in d0, and stereo adds 0.
- **Stereo as a detector.** Across the 15 integ tasks it gives 9 phantom blobs per layout at the scan pose: too many.

**(c) deposit_coin phantoms** (10 layouts, 11 phantom coin + stand events from the runs' frame-0 and scan records and
from current-main B1).
- **Rule.** Check the stand and slave the coin. The 2 mm coin, edge-on, is too thin to measure.

| views | phantom stands rejected | true stands (GT pose) flagged | B1 stands <= 7 mm off flagged | B1 stands 10-20 mm off flagged |
|---|---|---|---|---|
| home (frame 0) | 4/11 (7 unseen) | 1/10 | 0 | 2 |
| **scan pose** | **11/11** | **1/10** | **0/6 (all \|dh\| <= 1.3 mm)** | **4/4 (\|dh\| 6.7-53 mm)** |

- **Plane test alone.** The on-table test without heights rejects 5 of the 11 events. It also rejects 3 of 10 true
  stands: the stand is ~11 px wide, smaller than the matching window.
- **Phantoms on the piggy bank** (100002, 100008) are "proud", correctly, because something real is there. Only the
  height mismatch catches them (+65..+100 mm).
- **The one wrongly flagged true stand** (100002) is hidden behind the piggy bank in the head view.

**(a) sweep_blocks** (rendered 5.1, frame 0, home).
- B1 missed 12 of 17 cubes. SAM's stream found the cause: a count bug, B1 searched for one cube.
- The sweep marks 9 of the 12 as proud; the other 3 are outside home coverage. All 5 found cubes that are covered are
  also proud.

**Belt** (belt51, 67 frames, arms home).
- 94 of the 102 belt objects inside coverage are detected, including the black camera in 17 of 18 frames. Top error is
  0.6 mm median.
- **But the belt is unusable as a detector.** The belt is glossy and striped along its motion, so it gives ~20 phantom
  blobs per frame (pixel precision 0.17).
- A uniqueness gate on the raised minimum (`proud_unique` 0.03) does not fix it: precision 0.25, 24 blobs per frame.
  The streaks are view-dependent reflections, not an ambiguity along the epipolar lines.

**Height check vs B1's own records** (integ, 71 layouts, all tasks; "truth" = GT minus B1's predicted surface height
on the same pixels > 10 mm):

| views | flag threshold | flagged | precision | recall | false flags on right objects |
|---|---|---|---|---|---|
| scan | 10 mm | 46 / 395 | 11/46 | 11/13 | 9 % |
| scan | 15 mm | 32 / 395 | 10/32 | 10/13 | 6 % |
| frame-0 head + all wrists | 15 mm | 31 / 373 | 21/31 | 21/33 | 3 % |

- **False flags** concentrate on concave or vertical shapes: hang_mugs, organize_table's cabinet and drawer, kong tiles
  at the edge of coverage.
- **As a general reject**, across tasks, it is too noisy. As a verification signal it is useful, and so is a consumer
  narrowed to small, flat-topped fixtures.

**End to end: current-main B1 with the check** (`scripts/perception/stereo_coin_e2e.py`).
- **Setup.** deposit_coin 100000-09, the proxy runs' saved frames, the perception service's settings (fold backgrounds,
  scan-aware ones for the scan read, colour map). Frame 0 = `estimate`; scan read = `estimate_with_scan`, with the check
  on its final record.
- **Pairing.** Off vs `slot_bases` are the same annotated record with and without the consumer.
- **Cost.** B1 itself took 28-139 s (frame 0) and 116-285 s (scan read) on CPU.

| layout | frame 0: stand err (off -> on) | scan read: stand err (off -> on) | stand dh at the scan pose |
|---|---|---|---|
| 100000 | 580 mm phantom (unseen at home, kept) | **580 mm phantom -> not found** | -5.2 mm |
| 100001 | 20 mm -> not found | 20 mm -> not found | -6.7 mm |
| 100002 | **187 mm phantom (on the piggy) -> not found** | 187 mm phantom, kept: fully hidden behind the piggy in the head view, no pixel to check | — |
| 100003 | 0.2 mm, kept | 0.2 mm, kept | -0.8 mm |
| 100004 | 6.8 mm, kept (unseen) | 6.8 mm, kept | +0.7 mm |
| 100005 | 10 mm (coin not found) | stand 10 mm + coin 501 mm phantom -> both not found | +11.4 mm |
| 100006 | not found | not found | — |
| 100007 | not found | **274 mm phantom -> not found** | -5.3 mm |
| 100008 | 866 mm phantom, kept (unseen) | 866 mm phantom, kept: hidden behind the piggy | — |
| 100009 | 445 mm phantom (unseen at home, kept) | **445 mm phantom -> not found** | +31.7 mm |

Scan read summary:
- Phantom stands: 3 of 5 dropped. The other 2 were placed where the head sees only the piggy bank, so the check
  abstains.
- Stands within 7 mm: 2 of 2 kept.
- Stands 10-20 mm off: 2 of 2 dropped (100005 also had a phantom coin).
- Check time: 3.2-6.0 s per frame, CPU.
- What a drop does downstream (search, agent, look) is what Integration's rendered A/B measures.

## Verdict

- **Adopt as a verification signal, default off.**
  - `EstimatorConfig.stereo_check` annotates every object with `_perception.stereo` (`dh_mm`, `mismatch`, shares) and
    the record with `stereo_unexplained`.
  - The first consumer is `stereo_reject="slot_bases"`: a support-slot base whose slot children are all thin, i.e.
    deposit_coin's stand + coin. On a mismatch it goes to `_not_found` with its children.
  - The scan read runs the check once, on its final record.
  - A/B arm: `DOJO_PERCEPTION_CFG='{"stereo_check": true, "stereo_reject": "slot_bases"}'`. Integration A/Bs it on
    rendered deposit_coin.
- **Drop as a general foreground detector**: on the belt and across tasks it gives too many phantom blobs.
- **Needs** more views for coverage in front of the grippers: the scan pose works. A dedicated stereo pose would cost a
  look move, ~25 steps.
- **Motion stereo (rectified SGBM on two wrist poses) is not done.** The plane sweep over exact poses already gives
  0.5 mm on tops where covered.
