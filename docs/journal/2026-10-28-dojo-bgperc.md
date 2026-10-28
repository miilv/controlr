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
