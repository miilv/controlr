# 2026-10-27 — Perception bake-off, Stereo stream: height from the calibrated views

Stream: Stereo, in the evening perception bake-off. Brief (Ilia's idea): every camera pose is known exactly (head fixed,
wrists from FK), so the official RGB can give depth the way a stereo rig does. Measure whether it helps with:
- (a) objects not found;
- (c) shadows and phantoms;
- (e) heights.

Machines: compute3 only, CPU (≤ 8 cores, `taskset`), own worktree `~/controlr-stereo` with its own venv, data in
`~/dojo-data/pbake-stereo` (< 0.2 GB). No GPU job, 0 LLM calls. Report: experiments/2026-10-27-stereo-height.md.

## 1. Plane sweep (`dojo/perception/stereo.py`)

- **Method.** Horizontal planes, homography warps of every other view onto the head, ZNCC 7x7 plus a 5x5 cost box, best
  view per plane.
- **Speed.** The plain sweep is 1.4 s per 640x480 frame for 95 planes and 2 views. The left-right check adds a reverse
  sweep per view, ~5 s per frame.
- **Accuracy and coverage.** Top-facing surfaces are 0.5-1.3 mm median where covered. The home wrists see only the table
  beyond y ≈ -0.15, so coverage is 0.35. The scan pose gives 0.75.
- **Landmines.**
  - Warping a wrist that is 2.4x finer than the head is fine with ZNCC; no pre-blur was needed.
  - The far table is seen at grazing angles by the home wrists, which is noise, so the reference is cut to the object
    region (`REGION`).
  - The belt is glossy: view-dependent streaks make ~20 phantom blobs per frame, and a uniqueness gate does not help.
  - A 2 mm coin edge-on and objects ~11 px wide are below the window. The on-table test rejects 3 of 10 true stands, but
    the median height over the stand's pixels is right to ±1.5 mm.

## 2. Occlusion halos

Next to an object, the table the head sees can be hidden in a wrist view. h = 0 then fails, and a raised plane matches
the object's texture. With the left-right check (each wrist sweeps against the head, and a proud pixel needs a view
whose own height agrees), B1's colour-FG table pixels wrongly confirmed as proud drop from 20 % to 4 %. Pixel recall
drops from 0.75 to 0.39.

## 3. What it is good for

- **deposit_coin** (the lead's first ask). Height check of the stand at the scan pose: phantom stands rejected 11/11, right
  stands flagged 0/6, stands 10-20 mm off flagged 4/4.
  - Phantoms on the piggy bank are "proud", correctly. Only the height mismatch (+65..+100 mm) catches them.
  - The coin follows its stand: too thin to measure.
- **sweep_blocks.** 9 of 12 B1-missed cubes are proud. The misses were SAM's count bug, so this confirms the cubes are
  there.
- **Across 15 tasks** (71 integ layouts, B1's own records), the 10 mm flag catches 11/13 real > 1 cm height errors at
  the scan pose. But 9 % of right objects are flagged: mugs, organize_table's cabinet, kong tiles at the coverage edge.
  So it is a verification signal, not a general reject.

## 4. Shipped (PR, default off)

- **The flag.** `EstimatorConfig.stereo_check` writes `_perception.stereo` per object (`dh_mm`, `mismatch`, `known` /
  `on_table` / `proud`) and `_perception.stereo_unexplained`. The scan read runs it once, on its final record.
- **The consumer.** `stereo_reject="slot_bases"`: a support-slot base whose slot children are all thin, on a mismatch,
  goes to `_not_found` with its children. Today that is only deposit_coin's stand + coin. Kong tiles, the bread shelf
  and the mallet stand are never dropped.
- **A/B arm.** `DOJO_PERCEPTION_CFG='{"stereo_check": true, "stereo_reject": "slot_bases"}'`. Integration runs it on
  rendered deposit_coin.

## 5. Not done

- **Motion stereo** (rectified SGBM on two wrist poses) needs new Isaac renders. Two looks cost ~25 steps each.
- **Both wrists at the same instant for the belt.** The belt's gloss is the blocker, not the timing.
