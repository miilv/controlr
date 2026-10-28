# 2026-10-28 — stream C: continuous (background) perception

Brief: PERCEPTION_95.md, stream C. The question: the robot moves, and the wrist cameras see new views at every step.
Would perception that consumes those frames between the discrete perceive points (frame 0, scan, looks, pre-grasp
refine) make the belief right at grasp / place time more often than the runtime's own belief does? The score is
stream M's metric. A second question: is SAM 3.1's video tracker a better frame-to-frame mask source than re-running
refine per frame? Report: [../experiments/2026-10-28-bgperc.md](../experiments/2026-10-28-bgperc.md).

## 1. Per-step frame sequences (`dojo.perception.seq_capture`, `scripts/perception/seq_render.sh`)

- **How they are made.** A recorded proxy run saves only the frames it asked for. Its executed action list is
  complete, though (`<run>/<tag>_<lid>_actions.json.gz`). `SeqReplayPolicy` replays that list through
  `dojo.runtime.eval` in mode gt: no twin, no perception service, only the 6.0 render service. It sends one action per
  `next()` (`DOJO_SEQ_STRIDE`) and keeps every observation: the three cameras, the official state, and the dev-only
  GT rows (scorer side) per step.
- **The replay is exact.** hang_mugs 100000's `dev_gt_probe` is identical to the recorded P3r3 run's: 4 closes with
  the same labels, jaw distances 49.2 / 30.2 / 175.7 / 51.5 mm, every rise the same, score 0.40 in both.
- **Cost.** About 0.12 s of render per step. A 360-650-step episode takes 65-135 s on compute3 heavy1 (one job per
  episode, then a 90 s yield).
- **Storage.** JPEG q95 4:4:4 takes ~0.3 MB per step for three cameras; PNG took 1.1 MB. Frame 0 is also kept as PNG.
- **JPEG vs PNG** (build_tower 100004): B1 at frame 0 gives the same class and identity for 8/8 labels, |dxy| <= 2 mm.
  Wrist refines at steps 16 / 30 / 45 give the same ok-status on 14/14, with errors within 1.5 mm.
- **What was rendered.** 19 episodes, 6,798 observations, 2.0 GB, ids 100000-09 only:
  - hang_mugs 100000-04 (P3r3);
  - organize_table 100000-04 (B_breadth2);
  - stack_bowls 100000-01 (AGB);
  - deposit_coin 100006 (S79) and 100001-03 (B_breadth2);
  - build_tower 100000 / 02 / 04 (P3r3).
- **For stream W** (the lead's request), at stride 2: play_Xylophone, fasten_screws, fill_pen_holder and
  store_tools_in_toolbox, 3 episodes each.

## 2. Landmines found on the way

- **Pre-shaped grippers broke the grasp events.** hang_mugs commands 1.0 -> 0.43 on the approach (the wall grasp's
  pre-shape), then 0.0 at the close. The rule "first command below 0.5 that closes further" (`GTProbe`, and the first
  `grasp_metric.grip_events`) fired at the pre-shape, so the real grasp was never an event: 3 of 5 hang_mugs episodes
  scored 0 events. Stream M fixed `grip_events` within the hour; `GTProbe` still has the bug (BACKLOG).
- **CPU silhouettes are unusable for wrist refines of real meshes.** With `DOJO_GL=0`, one refine of a mug takes 13 s
  (a 25k-face silhouette mesh, ~550 silhouettes per Nelder-Mead fit). With GL (EGL) it takes 0.4-1.5 s. Only
  build_tower's blocks (12 faces) are cheap on CPU, so offline evaluation is a (light) GPU job.
- **Full B1 re-reads of later frames are expensive.** On hang_mugs (GL) a re-read takes 43 s with head + wrists, and
  19 s with the head view alone. The head view on its own is also never clear of the arms: robot coverage is 0.153 at
  home and 0.155-0.25 during the episode.
- **SAM 3.1's video tracker** (the Comfy-Org multiplex checkpoint, Meta's sam3 at f1647e8):
  - Point-only prompts are dropped after the prompt frame. Masks were empty on every later frame of a wrist and a head
    sequence, because the multiplex model merges its SAM2-style partial propagation into the detector's per-frame
    outputs.
  - A text prompt with one full detector pass first does make the clicked object propagate (20/20 frames).
  - That detector pass needs 18-21 GB of VRAM for 60 frames at 640x480, even with a text that matches nothing. Twice
    it hit the shared 32 GB card for under a minute before my VRAM guard killed it; the lead was told.
  - Dropped. The per-frame wrist refine stays the frame-to-frame source: 0.63 s per call, mIoU against the GT visible
    masks 0.93 / 0.65 / 0.71 for the three mugs of hang_mugs 100000's left wrist, 65 calls.
- `gpu_run.sh` now refuses to start below its VRAM floor. It used to start anyway after a 10-minute wait.

## 3. The perceiver (`dojo.perception.bgperc.BackgroundPerceiver`)

- **Scope.** The perceiver works on objects nobody has touched since their last epoch start. An epoch starts at a
  grasp, a release, or any re-measure by the runtime (a pose change > 1 mm, or a new belief source).
- **Each processed step:**
  - checks the visibility gates per wrist camera: silhouette >= 250 px, >= 85 % inside the image, <= 40 % under the
    robot or held-object mask, <= 65 cm away;
  - ranks the objects: required by the next event > in doubt / suspect > unconfirmed / low confidence. Confidence is
    lowered by B1's hidden_frac, so mostly hidden objects come first when a wrist view shows them;
  - runs at most one wrist refine.
- **Fusion.**
  - A fit within 6 mm confirms the object.
  - A fit elsewhere moves only the position, and only on two agreeing views >= 3 steps apart, or on one very strong fit.
  - An empty place in clear view is absence evidence: the object is searched (radius 6 cm) at its next view, and two
    such views put it in doubt, which raises a look request.
- **Moves keep the belief's orientation.** On build_tower 100002 a fit's 7-degree turn made a correct block "rot"
  wrong, so `move_turn` is off by default.
- **Optional parts:**
  - head re-reads (`reread_every`);
  - held checks during a carry (B1 `held_look` on the holding wrist, measured closed jaws, two reads). Now stream H's
    lane;
  - contact suspects (a jaw inside an untouched object's footprint below its top).
- **In the runtime.** `RuntimeConfig.bg_perception` (off | refine | reread, default off), `bg_obs_every`, `bg_cfg`.
  The perception service op `bg_step` keeps the perceiver's state between calls. `dojo.runtime.bg.BgHook` writes
  moves of free objects into the belief; objects at a closed jaw count as held. `Bridge.on_obs` / `obs_every` give
  extra observations in long segments. There is no change with the flag off; the runtime tests are green.

## 4. Results (proxy tier, M's metric, R = 10 cm; tables in the report)

- **Baseline (19 episodes, 118 events).** The runtime's own belief is clean on 89 (75 %), frame-0 perception alone on
  77. The rt failures are phantoms 12, held not_held 11, neighbours not found 9, neighbours off 6, target gross 4.
- **Background refines.** Alone and with contact suspects: 0 fixed / 0 broken on 8 episodes (64 events) and on 4
  tic-tac-toe episodes (87 events).
- **Head checks + doubt-triggered head re-reads.** 3 fixed / 1 broken on tic-tac-toe. organize / hang / bowls are in
  the report.
- **Knocked objects are found, but late.** hang_mugs 100000: knocked at ~300, the place seen empty at 316, found 51 mm
  away at 354 / 358. The re-grasp came at 338.
- **Look-alikes can mislead the background.** Two agreeing fits once put a chessman on its neighbour's spot (ttt).
- **Tic-tac-toe opponent.** The opponent never moved a piece in 100000-02. In 100004 it moved opponent_piece0 to the
  centre, and the belief never followed (5 events).
  - The head check sees the emptied start place: precision 0.0 on 11/11 checks, while present pieces never read below
    0.70 (0/675).
  - B1 re-reads cannot re-find the piece: its region priors keep the opponent's pieces on their side.
  - Fixed at the skill instead (PR #215, a watch-confirmed landing into the belief, gated on that head check + the
    provider's confirm_pose).

## 5. Verdict

- **Background refines / re-reads as a belief writer: dropped for now.** No net gain where the runtime is already right,
  and the remaining failures need region-free re-detection, which neither local refines nor B1 provide.
- **Kept as default-off infrastructure.** The per-step sequences are already used by W, H and V. `BackgroundPerceiver`
  plus `RuntimeConfig.bg_perception` is where between-event signals fuse with epochs: V's `b1_doubt` (from frames
  where the gripper is not over the object), SAM 2.1 clicks, S's detector.
- **SAM 3.1 video tracker: dropped** (§2).

## 6. Loose ends

- GTProbe's first-close rule (BACKLOG, M's item).
- general_pickup's clutter distractors near the corridor: an item for S.
- B1's placement priors make it blind to actor-moved objects off their generator region: a re-detector should not
  carry them.

## 7. X2: genuine-5.1 replays of Integration's arm-8 episodes (the review's test set)

- **compute2 cannot reproduce compute4's recordings** (now in BACKLOG):
  - c4b_bottles_gcon_100000 replayed on compute2 drifts from compute4's physics from the settled start: 0.002 mm at
    step 0, 2 mm by step 161, 77 mm later. Its dev_gt_probe fails from the 2nd close, and the 6th close names another
    bottle.
  - compute2 is self-consistent: its rendered replay is bitwise equal to its physics-only replay, so rendering does not
    perturb physics.
  - compute3's physics-only replays reproduce compute4 at 0.0 mm (stream M).
  - The boxes have different CPUs (13900KF vs 14900KF), and their Robots asset trees checksum differently.
- **So the 70 replays render on compute4 itself**, one lane on ~/dojo-gpu.lock, ON first. `scripts/perception/seq_render.sh`
  with `SEQ_RENDERER=isaac51 SEQ_NAME_BY_SRC=1 DOJO_SEQ_STRIDE=2`. Each replay is gated by
  `scripts/perception/bgperc/replay_check.py`: dev_gt_probe closes reproduced at 0.0 mm and the same step count,
  otherwise dropped.
- **The reviewer's test-set hygiene, in the capture** (`seq_capture`, `replay_check.py`):
  - decision frames are observed whatever the stride: every step the recorded run logged (report events, decisions),
    k-1 of every grip event (M's last-closing-command rule), release+1..5. `valid.json` counts them present vs
    expected per episode;
  - the inputs carry no GT: `steps.jsonl` holds step, state and grip commands; the dev GT is in `gt.jsonl`.
- **The offline arms read one frozen detection cache** (`scripts/perception/bgperc/seg_cache.py`): v2b (weights sha1
  1a8bc5d4522c, imgsz 960, conf 0.25) on every stored JPEG, keyed by the JPEG bytes' sha1, masks lossless (bit-exact
  against a live run). The online arm runs v2b live on raw frames and logs its own detections.
- **JPEG vs PNG for the segmenter** (`seg_jpeg_check.py`): v2b on frame 0's PNG and JPEG of every valid sequence,
  matched by mask IoU >= 0.5.
  - The first 8 JPEG episodes give 24 frame-0 views: 70/71 detections matched, IoU mean 0.986, 2 class flips,
    1 lost, 2 gained. That is 7 % of detections differing.
  - Every difference is a wrist detection at score 0.25-0.46. At score >= 0.5: 0 of 74 differ.
  - Borderline wrist detections are exactly where extra reads between events would show. So **the X2 frames are
    lossless**: PNG, compress level 3, about 0.42 GB per episode. The 9 JPEG bottles episodes are re-rendered.
  - `DOJO_SEQ_FMT` now defaults to png.
- **RTX frames are not bit-identical from run to run.**
  - The recorded run's saved frame 0 (look_frames, PNG) and the replay's differ on about 65 % of pixels, mean |Δ| 0.45
    levels.
  - v2b on the two, over 33 views: 2.9 % of detections differ, all below score 0.5. At >= 0.5: 0 / 97 differ.
    Matched IoU is 0.997, against 0.986 for JPEG.
  - So a replay never sees the recorded run's pixels, and re-running the runtime's perception on replay frames
    would add render noise to rt.
- **X2 protocol** (the reviewer's design, adopted by the lead):
  - rt = the recorded belief (probe plus logged writes). It is exact, consistent with the replayed actions, and
    needs 0 LLM calls.
  - Each A1 arm = the recorded belief plus that arm's writes on the stored PNGs. Once the arm moves an object, that
    object's later recorded writes (refine, post_place, held) are re-computed on the same step's stored frame from
    the arm's pose. Untouched objects replay as recorded.
  - Agentic records replay as recorded in every arm.
  - A fidelity check bounds the render noise: the runtime's refines re-run offline at their recorded steps from
    their recorded priors, diffed against the logged results.
  - The detection cache is keyed by the sha1 of the stored PNG bytes; `replay_check.py --seg-cache` records its
    hash per episode in `valid.json`.
