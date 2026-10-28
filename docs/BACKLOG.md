# Backlog

> The task queue. Take an item → branch → PR linking the item → strike it here in the same PR
> (`- [x] ~~…~~ (journal/… or experiments/…)`). Found a landmine — add it here.
> Sources: [experiments/](experiments/), [reviews/](reviews/), [journal/](journal/).

## P1 — before the next round (carry failures and perception)

- [ ] **dojo perception: store_laptop headset identity 00002 vs 00004** (5.1 c4 frames; B1 picks 00002 with the alternative
  00004 at +0.03..+0.08, so `identity_margin_swap` never fires): try the embedding path (`embed_b1`) on these frames.
  The laptop flip: `on_base_refit` (off) fixes 7 of 17 estimates; A/B after the root-vs-mesh offset is resolved.
- [ ] **dojo skills: store_laptop's headset hang never plans: G3 mode A 0/10 stops before the laptop.** The planner
  picks `hang_on_hooks` for the headset (its `headband_bridge` on `stand`'s `support_cradle`) first, and finds
  "0 grasp-place pairs" on all of 100000-09. 8-99 of the 96-405 headset grasps are reachable per arm, but none pairs
  with a reachable hang. The episode then finishes with the laptop untouched (`not_home`, 0.0). Identical on main
  22f6339 and with #198's frame fix (journal 2026-10-27-dojo-integ-c4 §5; runs compute4
  `~/dojo-data/integ-c4/g3/laptop_A_{main,pr}`). The laptop stage runs only once this plans.
- [ ] **dojo: the rest of the articulation root frame (laptop).** `artroot` converts at objects_from_gt,
  estimate_to_layout, the twin's verify rows and eval_g1 (journal 2026-10-27-dojo-integ-c4 §4). Left: (1) dev tools
  that pair GT rows with meshes on their own (`color_calib`, `val_refine`, `val_what`, `review_exp`, `held_data`,
  runtime `GTProbe`'s nearest object at a close) still read the root pose: use `artroot.row_to_asset`;
  (2) presettle v2 never writes an articulation back, and the laptop settles ~8 mm down after spawn (probe: GT - record
  = (0, +1, -8) mm), so settled records keep the spawn z; (3) `Rigid/factory_nut` carries an ArticulationRootAPI on
  `/root/visual` (15.6 mm off) but loads as a rigid prim: G3 fasten_screws 10/10 says harmless, unprobed;
  (4) the record's `scale` is not applied to the root offset (every record: 1).

- [ ] **dojo perception: learned identity on the conveyors needs torch in the perception venvs.** The belt read with
  `EstimatorConfig.embed_model="dinov2_s"` (5.1 belt reads: identity 79/116 -> 104/116, phantoms 77 -> 47; report
  2026-10-27-identity-embeddings) is off by default. The rendered A/B arm sets `DOJO_PERCEPTION_CFG='{"embed_model":
  "dinov2_s"}'` and needs torch + transformers + `$DOJO_EMBED_MODELS/dinov2-small` (DEVELOPMENT §4). What is left on
  the belt: puppet -> camera when their blobs touch (5 of 23 puppet reads). B1's path (`embed_b1`) stays off: neutral
  on the table except fill_pen (-3 pens). pick_from_conveyor_by_image: the picture board is hidden behind the right arm
  at home; after a look, CLIP-L picks the pictured rider 33/33 (picture = the public photo.png).
- [ ] **dojo perception: laptop root vs catalogue mesh frame** (store_laptop, perceiver-models report §5): the catalogue
  mesh `Articulation/laptop/00000` drawn at the sim's laptop pose (`objects_gt`, the settled record) sits ~80 mm in front
  of the rendered laptop; at +(0, 0.08, -0.025) m in the object frame it fits all 10 genuine-5.1 frames. Perception's
  mesh-frame pose vs the twin / skills' root frame: trace through catalogue, record conversion, twin, skills (Integration).
- [ ] **dojo perception: arbitration reverts agent fixes** (perceiver-models report §3): an unrefined agent point loses
  to B1's refined fit to the wrong blob; `arbitrate(risky_b1=0.2)` (+7..+12 fixed of 84 on every model, broken +0..+2)
  needs a rendered A/B before it becomes the default (`AGENTIC_*` wiring not done).
- [ ] **dojo agentic: the first message shows `np.str_('...')` as the instruction** when the runtime passes a numpy
  string (`j1._first_message` formats `instruction!r`; c4_laptop transcripts). Harmless to the models so far; `str()` it.
- [ ] **dojo skills: a silent in-hand slip in mode B** (store_tools 5.1 B, 6/10 "stages_left 1"; journal
  2026-10-27-dojo-longh §8). A physics replay of 100008: the wrench slid out of an off-centre grasp ~15 steps after the
  lift, after held_look had said "held"; nothing re-checks before the release and the skill places an empty gripper.
  The strict v1 contract gives no finger gap; a vision check (held_look) right before the release of slip-prone grasps.
- [ ] **dojo runtime: a Kit hang on a GUI dialog holds the heavy lock** (compute8, organize_table 100008 on 8c845c9):
  `zenity ... Failed to open display` every 2 min in integ_eval.log, no env step for 45 min; integ_eval's TERM path hung
  in `systemctl --user stop` until the eval python got `kill -9`. Needs a hard watchdog (no step for N min -> kill -9).
- [ ] **dojo perception: the robot mask reads a pre-shaped gripper as a grip** (`render.robot_items`): any command below
  0.5 masks the jaw sweep and a 65 mm held box between the fingertips (the strict tier has no finger measurement). A
  pre-grasp opening for a narrow object (`gap_to_grip`, Memory's conveyor approach at 0.444) is not a grip: at a 15 cm
  hover the box hid 197k of 307k wrist pixels. `refine(empty_hand=True)` / the service's `empty_hand` opt out; B1's wrist
  frames, looks and `held_look` still mask it. A caller that knows the hand is empty should say so.
- [ ] **dojo perception: deposit_coin's coin + stand read at a phantom near a home gripper** (5.1 coinmain, 6 of 10 layouts, at frame 0 or in the scan read). The real assembly is ~10 px in the head view; B1 fits the stand onto the piggy bank or gripper area instead. #153 makes the agent's correction stand the coin up; the B1 miss is the root. Also: coin not found when the stand fit is 19-115 mm off (the slot check refutes it).
  Candidate fix, off by default: the stereo height check at the scan pose rejects 11/11 phantom stands and flags 0 of 6 right ones
  (`DOJO_PERCEPTION_CFG='{"stereo_check": true, "stereo_reject": "slot_bases"}'`; experiments/2026-10-27-stereo-height);
  the rendered deposit_coin A/B decides. Gap: current-main scan reads put 2 of 5 phantom stands where the head sees only
  the piggy bank (behind or inside it), and the check abstains on an object no head pixel shows.
- [ ] **dojo perception: stereo place-z check (needs post-place frames).** Place misses were dz -38..-62 mm. The plane
  sweep measures covered tops to 0.5 mm median at the scan pose (0.9-1.3 mm at home, 87 % within 5 mm; experiments/
  2026-10-27-stereo-height), so a placed object's top vs its believed z is 10-100x above the noise. But no saved frame
  shows a placed object: `save_look_frames` keeps only frame0 / scan / look (and belt track frames). None of the 1158 integ
  look_frames dirs on compute3 has one; the `post_place` events log only shift / iou. Next: a dev `_keep_frame` at
  place_confirm / post_place, one rendered run (tower, stacking toy, toast), then `stereo.check_record` on the placed
  label.
- [ ] **dojo perception: plane-sweep stereo is blind on the conveyor belt and in front of the home grippers.** The belt is
  glossy and striped: its view-dependent streaks give ~20 phantom "proud" blobs per frame (a uniqueness gate does not
  help), so stereo there can only confirm colour candidates. The home wrists see only the table beyond y ≈ -0.15, the
  zone in front of the grippers needs the scan pose (coverage 0.34 -> 0.75). Vertical faces and textureless tops get no
  height (experiments/2026-10-27-stereo-height).
- [ ] **dojo perception: classify_objects' series stay under the config label** (`rigid0` x n). Its per-category
  prefixes (`cat0_*`, `cat1_*`, `cat2_*`) follow the generator's random category order, which can't be observed, so
  `series_count` doesn't rename them. Check what the checker / twin need there. SAM 3.1 text also undercounts its
  small toys (1-5 masks for 3-9 objects): within 30 / 69 with `sam`. arrange_largest_number's `linked_count` mats are
  not handled at all.
- [ ] **dojo perception: a "held" refine can fit the object 80 mm below the jaws** (breadth3 store_tools 100000 step 67: fit 78 mm low, 43 deg off, iou 0.73). The runtime's in-hand bounds (#99) reject it now; perception could reject fits whose centre is outside the jaws itself.
- [ ] **dojo skills: organize_table, three skill-side gaps (G3 mode A 0.44: alarm 2/9, figurine 0/9, keyboard 4/9).**
  (journal 2026-10-27-dojo-longh §5) The drawer's top is a tray 23 mm under a rim; the place targets use the bbox top,
  so the alarm lands across the rim (17-19 deg) or falls off its front edge. The figurine has no collision-free pair within
  12 mm of its 40 mm stand next to the drawer (with the drawer ignored there is one). The keyboard, pushed away from the
  robot first, ends beside the drawer where no sideways stroke fits; pushes turn it 15-26 deg and it slides 20-30 % past
  a stroke. Tried on `feat/dojo-longh-organize` without a gain: mesh-height-map targets (their full plans collide at the
  pick), stand offsets to its half-size (the screen's 6000-IK budget runs out over 85 targets), a cover-the-frame push
  goal, a stroke-order lookahead, off-centre strokes.
- [ ] **dojo perception: tic-tac-toe cell reads flicker on genuine 5.1 frames.** `watch_read` (kind cells) on compute8's
  rendered tic-tac-toe (Long-Horizon, journal 2026-10-27-dojo-longh): a board with our two pieces and one of the
  opponent's read "........." for 200 steps (100005), pieces appear and vanish between reads 1-2 steps apart
  ("....x...." / "........." / "....x..o."). `play_board_game` counts the opponent's moves from these reads (the
  belief never moves the opponent's pieces), so a blind read ends the game in "wait opponent (timeout)".
- [ ] **dojo runtime: a held_look in-hand refine moved a held tic-tac-toe piece 40-42 mm (iou 0.66-0.67, visible_frac
  0.53) and was adopted** (rendered 5.1 100003 / 100004, breadth2's 41.7 / 42.1 mm place_miss rows). The place is planned
  from that transform; the belief then says "42 mm from the cell" while the piece is in it. A piece in closed jaws cannot
  shift more than about its half-width: reject such held refines (or require iou >= 0.85 above 10 mm).
- [ ] **refine() reports ok for a wrong fit, 20-63 mm off.** 10 of 291 cases in the D0 wrist validation
  (`dojo/perception/val_refine.py`, priors ±20 mm, `WITH_OTHERS=1`), IoU 0.62-0.86, mostly an identical
  pillar / block nearby:
  - build_tower 100003 f1 block1, 100004 L1 block1, 100007 L1 block1, 100009 L1 block2, 100018 f1 + f2 block1,
    100022 L1 block6;
  - stack_blocks 100002 f2 block_2, 100015 L1 block_1, 100017 f2 block_2.
  
  Candidate fix: score the prior's own position against the best fit (prefer the prior unless the fit gains
  clearly), or require the neighbour guard against all same-shape objects of the belief. This is the next tower risk
  after the scan.
  Re-measured on main 2026-10-27 (Perception-2 journal §9).
  - Same set: 3 ok fits of 271 are beyond 15 mm. Only build_tower 100009 L1 block2 is gross (62 mm, iou 0.67), and the
    runtime's big-shift bar (iou >= 0.7 above 30 mm) already refuses it.
  - Genuine 5.1 d0_51 (frames 1/2, 4 tasks): 0 of 321 ok fits beyond 15 mm.
  - Keep open only as a watch item.
- [ ] **Runtime: a confident flip of a near-symmetric object is a coin toss.** A pick / place refine that turns an
  object more than `refine_max_turn_deg` (90) at iou >= `refine_turn_min_iou` (0.9) keeps the prior orientation and the
  fit's centre (`policy.keep_centre`); below that iou nothing is applied. The charger's flips at iou 0.92-0.98 were
  right in breadth2 100003 / 100004 (the prior was the flipped one) and wrong in breadth 100002 / 100004 and breadth2
  100002, the same mode on every re-hover either way: neither the iou nor a second view decides. Needs a feature cue
  (the prong side) from perception, or the in-hand review after the lift; then an A/B on plug_in_charger.
  In-hand half done (journal 2026-10-27-dojo-precision.md §5): an in-hand flip is adopted only when a second view agrees (`held_flip_confirm`; rendered plug 4/10 -> 8/10). Left: pick views alone still never flip (a wrong frame-0 yaw without a relay stays wrong); Perception-2's paired-texture pick refine separates 5/5 on the 100004 replay.
- [ ] **Runtime post-place drops z-only corrections.** `belief.post_place` adopts a refine only when its xy
  `shift_mm` >= `post_place_min_mm` (2 mm). A refine that lowered a floating belief onto its support
  (`z_dropped_mm`, 5.1 stack 100007 s142: 56 mm) with < 2 mm xy shift is ignored. Adopt when `z_dropped_mm` is
  set (or use a 3-D shift) (journal/2026-10-26-held-look-stack-refine.md).
- [ ] **Hover identity candidates = the whole task vocabulary.** `perception_service.shape_classes` /
  `policy._refine_hook` offer `refine_identity` every distinct shape of the task, so a label can become another
  label's asset: 5.1 build_tower s0L9, a pillar (00002) became the prism (00003) at the pick hover and was placed
  standing on its 62 mm end. Offer only the label's own config group (in build_tower, one index per label: no
  change possible). Perception now also requires iou >= 0.9 for an identity change
  (experiments/2026-10-26-what-is-there.md).
- [ ] **Runtime: a `partial` post-place refine confirms the across offset only.** For a board wider than the wrist
  view, refine fits the offset across the long axis and the yaw. `partial.unobserved_axis` keeps the prior's
  position along the axis. Check that axis with the head-view render-compare or a second wrist pose at a board end;
  do not read `ok` as a full confirmation (experiments/2026-10-26-tower-post-place.md).
- [ ] **Save the scan frame with `save_look_frames`.** The runtime keeps frame 0 and the last look only, so a
  scan-primary artefact cannot be replayed. Integration's tower scan A/B phantoms in 100011 / 100014 could not be
  re-run (experiments/2026-10-26-phantom-stack.md).
- [ ] **Run `catalog_check --sync --assets <RoboDojo Assets>` on every catalogue install.** It annotates the slot
  bases (`passive.support`); without it, `catalog_check` now exits 1 with an ERROR. make_toast's bread and
  play_Xylophone's mallet are placed from `Geometry/bread_shelf/00000`'s and `Rigid/mallet_stand/00000`'s slots.
  Annotated on compute3 only. Then settle the other slot tasks' pairs: deposit_coin, fill_egg_holder
  (experiments/2026-10-26-slot-objects.md). The check also fails on an egg holder / toaster / laptop without its
  `links.json` (the skills' lid hinge and lever; `--sync --sources <skills catalogue>` copies it, `--assets` writes it):
  compute3's perception catalogue got `Articulation/egg_holder/00000/links.json` from skills6's on 2026-10-27.
- [ ] **Corner look when the expected count is not met after scan + agent.** If the scene's vocabulary expects more
  objects than the scan-primary estimate + agentic answer report, look at the head-view regions the scan pose still
  cannot resolve: the table corners and edges, |x| >= 0.4. The case is build_tower `block4`, a green prism at
  x = -0.43 that no reader found in D0 100024. A perception-side `look_plan` request
  (docs/experiments/2026-10-25-scan-pose.md). D0 100024's three missed pieces were blobs merged with a pinned base
  board or with each other; B1 now finds them at frame 0 (experiments/2026-10-27-tower-merged-blobs.md), so the case
  for this item is open again.
- [ ] **Decision head, first light** (`configs/sim_waffle_yaw_luna_dec.yaml`, journal/2026-10-07-luna-decisions.md):
  (1) ~~image format probe~~ done: `parts` is seen, `field` is not (journal/2026-10-07-luna-decisions.md);
  drop `field` or make its live test expect blindness; (2) one mock episode
  (`configs/mock.yaml --set llm.backend=decisions ...`); (3) the 4 rotation seeds with the pinned
  plans vs Sonnet 5.5 / Luna chat — success, turns, wall time per episode, cost. Then axes:
  `levels` (resolution), `reduce`, `history`, `include_manual`.
  → Tuning round done (journal/2026-10-07-decision-head-tuning.md): mock reach 11/12 with the split head,
  3 cameras, fovea and a named target. Open: (a) **top up the OpenRouter credit** (402 at ~$1.9);
  (b) visible, concretely named targets on Isaac: the planner names the object per phase and a
  `phase` choice picks it (one request can carry every phase's questions); (c) the status question
  (DONE 2/12) — or end on a sensor-checkable condition; (d) scene/top left/right on photoreal
  frames still leans "left" even in the fovea; (e) yaw for the split head.

- [ ] **Real-sensor feedback (`feedback.sensing: real | privileged`, default real).** Today every
  STOP names what was hit ("the held packet pushed against the box wall (the box moved)"), the box
  proximity WARNs need the box's exact current pose, and a wrong DONE gets "task not complete yet" —
  all simulator ground truth no real sensor provides ([journal](journal/2026-10-03-feedback-realism.md)).
  Emulate the real rig instead: one contact detector = net external force at the wrist relative to
  the start of the move (+ UR protective stop on hardware), message `STOP: contact force at the
  wrist[, toward ±axis], motion stopped` — the model works out from the image what it hit; box
  WARNs off by default; no goal-check feedback (DONE ends the episode, scored afterwards); manual
  text updated. The arm is a **CB3**: the sim detector must emulate a coarse joint-current force
  estimate (threshold well above the noise floor, direction unreliable) — so the default message
  carries no direction; an FT-300S would allow `toward ±axis`.
- [ ] **Luna's grasps blow up PhysX:** 4 of 7 Luna failures (both efforts) were `unstable` at `GRIP close` after small low descents next to the mat; Sonnet's final round had none ([experiment](experiments/2026-10-04-luna-control.md)). Replay those closes on a fresh server and find what differs (finger depth, packet offset, closing onto the mat).
- [ ] **Release false stops:** never apply the robot-vs-packet stop while the gripper is opening
  (3 STOPs in a row while releasing, contacts-and-speed first pass), and don't count harness-caused
  stops toward `episode.max_stops`.
- [ ] **Effort experiment:** `llm.extra_body.reasoning_effort: low` (≈2 s/turn, 0 thinking) vs the
  default (≈high, 3.3 s mean) on the same 4 rotation seeds with the same plans (~90 calls)
  ([journal](journal/2026-10-03-thinking-controls.md)).
- [ ] **Planner on vs off vs pinned** on the same seeds — the planner's single-view 3-D guesses were
  confidently wrong (rim 110–130 mm vs 180 mm) and it costs 57–111 s per episode.
- [ ] **Repeats per seed** (≥ 3) before claiming any success rate; 4/4 on 4 seeds says the stack works, not which change mattered.
- [ ] **Cameras (hypothesis: the main failure cause is a single angled D435).** Add a virtual
  wrist camera and a second fixed one (top/side) to the Isaac scene; experiment 1 camera vs
  + wrist vs + top (tiled and separate). The renderer and config already support it
  (`observation.cameras`, `observation.tile`); the server needs the cameras + `CameraInfo`.
  The real rig would need a physical wrist camera — decide after sim.
- [ ] **Box geometry and carry corridor in the manual/task:** near wall at y ≈ −85 mm, rim at
  z = 180 mm, the packet hangs ≈30 mm below the TCP; rise while moving toward +y; cross the wall
  at z ≥ 260 (5 of 8 failures in the rotation round). Re-run seeds 0–3 with the same plans
  ([rotation-yaw](experiments/2026-10-02-rotation-yaw.md) §6.1).
- [x] ~~**Arm links vs box in `SafetyEnvelope`**~~ — `safety.box_collision: block | warn | off` (default warn), the box is a dynamic body, a light touch stops the arm (journal/2026-10-03-contacts-and-speed.md, experiments/2026-10-02-contacts-and-speed.md)
- [x] ~~**Object-force stop fires falsely**~~ — the held-packet stop uses a packet-vs-environment contact view (journal/2026-10-03-contacts-and-speed.md, experiments/2026-10-02-contacts-and-speed.md)
- [ ] **Joint-space escape/home**: the arm gets stuck at the edge of reach and almost every move
  is refused (Sonnet seed 0 — 17 turns wasted).
- [x] ~~**Feedback noise**~~ — `feedback.level: short` (TASK / STATE / WARN / STOP), no tactile, no settle notices (journal/2026-10-03-contacts-and-speed.md, experiments/2026-10-02-contacts-and-speed.md)
- [ ] **`grid` / `ee_marker` overlays in a live run** — implemented and projecting correctly,
  never tested live; 80–200 mm depth error is the main source of mistakes.
- [x] ~~**Reach target is ambiguous in depth**~~ — the marker is gone; reach / push targets are text relative to visible objects (journal/2026-10-03-contacts-and-speed.md, experiments/2026-10-02-contacts-and-speed.md)
- [ ] **Reach / push with text targets never run live** (only CPU tests + the Isaac reset test).
- [ ] **Grasp yaw window −40…+30°** at the current tilt: a `yaw_min/max` clip param or `rotation=full`.
- [ ] **The planner should always write "packet heading h, grasp yaw = h ± 90"** — when it does, models hit it exactly.
- [x] ~~**`no-think/claude/claude-sonnet-5-5`** in an episode~~ — `no-think/` still thinks for Sonnet 5.5; `reasoning_effort: low` is the working switch (journal/2026-10-03-thinking-controls.md); the episode-level comparison is the effort experiment above.

- [x] ~~**dojo agent: the scene must list functional frames** (pegs, slots, press tags) for `stack_on_pegs` /
  `slots_and_press`~~ — scene v3 (`AgentConfig.scene_version=3`) lists the asset's named support / functional frames.
- [x] ~~**dojo agent: AgentConfig ran scene v2 by default**, so the frames above never reached the chooser (Sonnet
  wrote code for hang_on_hooks / slots_and_press / strike_in_order / stack_on_pegs, 19/93 items), and v3 cut every
  object's frames at 8 (no `hit_*` xylophone keys, no ninth board cell)~~ — scene v4 (every frame, numbered series
  with positions) is the default; `stack_on_pegs` doc v8 says which group fits which peg
  ([routing report](experiments/2026-10-27-routing-sonnet.md)).
- [x] ~~**dojo agent: registry path lags SKILL_DOCS v8**~~ — `skills/registry.yaml`'s `stack_on_pegs` doc carries
  the same "most pieces -> highest peg" sentence (test `test_registry_peg_doc_says_which_group_fits_which_peg`).
- [ ] **dojo agent: agent_v5's retry line, episode check.** In the replay probe (journal 2026-10-27-dojo-official
  §2.3) no retry dropped ids under agent_v5 (v4: Sol 9/10, Sonnet 1/10 did), but 8 of 10 Sol samples re-measured
  first and their retry was not observed. Confirm with an episode A/B on an every-object skill (make_kong 100002 rendered).
- [ ] **dojo agent: wrong asset captions mislead the chooser.** The mug rack (cup_holder 0) is captioned
  "smartphone", the bread and the coin "smartphone", both vertical stands "business card holder". With scene v2
  Sonnet called the rack "a phone-shaped holder, not a rack" and planned a `robo.search` for a missing rack. A
  caption-sanity pass (category vs caption) in `scene.caption` would remove the trap.
- [ ] **dojo agent: `robo.obj()` has no frames.** Code cannot look a frame name up (Run A plans "print the toaster's
  frames" would have found nothing); since scene v4 the frames are in the scene text only.
- [x] ~~**dojo agent: routing of the default model is unmeasured on the new skills**~~ — Sol 6.1 (`AgentConfig.model`)
  on the same 93 items, scene v4 + docs v8: 90/93, 0 harmful; insert_tubes 3/3 `insert_each`
  ([routing report](experiments/2026-10-27-routing-sonnet.md)).
- [x] ~~**dojo agent: Sol routes fill_pen_holder to code (0/3).** It reads "Hold the pen holder with one hand, place
  all pens into it with the other hand" literally; the checker judges only the end state~~ — SKILL_DOCS v10:
  `fill_container`'s doc says the hold is judged by the end state and the skill fills the holder standing. Sol
  0/3 -> 3/3, Sonnet 3/3 -> 3/3 (journal 2026-10-27-dojo-official §2.1).
- [x] ~~**dojo agent: check pick_from_conveyor_by_image routing under SKILL_DOCS v10**~~ (the basket must stay lifted
  there) — Sol 3/3, Sonnet 3/3 `pick_into_held`, unchanged from docs v8 (journal 2026-10-27-dojo-official §2.1).
- [ ] **dojo skills: no laptop skill** (close the lid, stand it in the vertical rack): store_laptop_and_headphones is
  code for the agent; the rule planner runs `hang_on_hooks` for the headphones only (the paired 20 of 100).
- [ ] **dojo agent: perception is the bottleneck in mode B** (Phase 2, 2026-10-25): stack_bowls 0/6 and build_tower
  100000 0/3 were perception_gross for every model; put_bottles 0/6 were put_into drop misses / pick collisions
  (verify off). The model choice did not change these outcomes ([report](experiments/2026-10-25-agent-e2e.md)).
- [ ] **dojo agent: align_blocks' set square has no grasp** (skills: 0 feasible / no top candidates); Astra solved
  one layout by sweeping the blocks with a fingertip `move_to` line. A tool-push skill would make it a skill task.

## P2 — harness and infrastructure

- [ ] **A G2 replay record on an allowed layout (100000-09)** for the bootstrap smokes: smoke_twin's contact-restore check, smoke_dev_eval and smoke_proxy_eval replay G2 on 110007, inside the reserved >= 100100 range (D-C10a), so they are skipped unless `SMOKE_RESERVED_OK=1` (journal/2026-10-28-dojo-compute2-bootstrap.md).
- [ ] **Code defaults `RD_HOME=/root/controlr-robodojo` on root boxes** (`rdpy.sh`, `dev_eval.sh`, `paths.py`, `world.py`, `interactive.py`): a job that forgets `source ~/dojo-env.sh` uses whatever that path is. On compute2 it is a symlink to `~/dojo-robodojo` now; the old install there had Eval_Layout.

- [x] ~~**Turn latency, harness side:** execute on the STATUS word with the note + usage read in the background; keep-alive 120 s~~ — `llm.overlap_tail` (default on), `KEEPALIVE_S` (journal/2026-10-04-turn-latency.md)
- [ ] **`observation.size: 336` A/B for Sonnet** (≈ −0.2 s/turn; 224 px makes Sonnet think — journal/2026-10-04-sonnet-latency.md).
- [ ] **`observation.jpeg_quality: 75` A/B** (−41 % upload bytes, same image tokens) — success and latency vs 90 on the same seeds.
- [ ] **Execute each action line as soon as it is complete** (before the STATUS word): another ~0.1–0.2 s/turn; changes behaviour on replies that later fail to parse — needs a rule for that first.
- [ ] **Direct Codex WebSocket transport for the loop** (`llm.transport: router | codex_ws`): measured −0.3–0.6 s/turn vs the router and 57 KB instead of up to 0.7 MB per turn (journal/2026-10-04-turn-latency.md). Needs: manual in `instructions`; `previous_response_id` incremental input; on any error drop the socket and resend in full; **retry on `invalid_prompt` moderation false positives (≈ 3–4 % of direct calls)**; reconnect at the 60-min limit; token refresh with a single holder (where the loop runs: compute3 — then the dev-box copy must go); from compute3 through its VPN (direct is blocked).
- [x] ~~**Router-side latency: split OmniRoute vs Codex time**~~ — router telemetry: parse+validate+policy ≈ 8 ms, upstream (`connect`) is the rest; reconnect to chatgpt.com ≈ 60 ms (journal/2026-10-04-turn-latency.md). Left unchanged on purpose: a keep-alive/dispatcher change saves ≤ 60 ms but needs a restart of the shared gate; `codexTransport=websocket` is per shared connection. Revisit only with the other clients' owners.
- [ ] **Cache probe through the router (~6 calls):** do `llm.cache` / `cache_ttl` matter at all on `claude/` / `cc/` (the router seems to set the boundary) — don't sweep these axes until then.
- [ ] **Pin one upstream account** (a lease/session header in `llm.extra_headers`) — cache drops on turns 4, 5, 7 in one run.
- [ ] **Haiku 4.5 doesn't cache the first turns** (manual < 4096 tokens) — a meaningful appendix (kinematics/examples), not filler.
- [ ] **`action.format=tool`** (actions as tool calls instead of the text grammar) — an experiment axis, not implemented.
- [ ] **Continuous video recording in Isaac** (`log.record_video`); today videos are per-turn frames only.
- [x] ~~**Isaac runs at ~0.24× real time**~~ — ~0.7× with the new defaults (journal/2026-10-03-contacts-and-speed.md, experiments/2026-10-02-contacts-and-speed.md)
- [ ] **Isaac ≥ 1× real time:** settling is ~45 % of sim time (fixed thresholds, ≤ 2 s); self-collisions off gives 1.02× (not default: the force stop does not see self-contact either); 24 iterations failed 1 of 4 expert scenes, 16 failed 1/2, dt 2 ms lost the grasp. Next: settle criteria per move, contact-report only during motion.
- [ ] **5 ms drive-target ticks blow up some grasps** (2 of 4 live closes near the mat; replay-confirmed, 1 ms is fine) although the scripted expert passes — find why before using `substeps > 1`.
- [ ] **Force-stop overshoot is an impact spike, not sampling:** driving the fingers into the mat at 0.15 m/s peaks at 85–350 N against an 80 N limit at every sampling rate (1 ms included). Slow the last ~20 mm near contact, or a compliant wrist.
- [ ] **Usage grace is not a real deadline** (deferred in review, L8): up to 0–0.2 s per turn. On `cx/` (Luna) the usage chunk trails by up to ~0.5 s — the Luna config sets 0.6 s; a usage-less turn makes `summary.json`'s cache share wrong (count usage-less turns in the summary). Better: execute the parsed action at `t_complete` and collect usage concurrently — the wait was 0.1–0.7 s (~0.3 s mean, ~10 % of a Luna call) in the Luna round.
- [ ] **Every turn re-uploads the whole transcript** (all images as base64: ~42 KB per image, ~730 KB by turn 16). Harmless on a fast link, 10–17 s per call on compute3's slow days ([journal](journal/2026-10-04-luna-latency.md)). Options: the router's `/responses` with server-side state (only the new turn uploaded — check cache and the append-only invariant), or run the loop where the uplink is fast. Also: ~1.4 s of each call is router + upstream overhead (Luna 1.15–1.4 s TTFT on a 31-token prompt).
- [ ] **Arm self-collisions** aren't detected by the force stop (S5, partial).
- [ ] **Mock: table_z 0.053 vs Isaac −0.0095** — unify in one spec (mock only, harmless).
- [ ] **Archive of `runs/`** (722 MB locally, not in git): where to keep runs that reports reference (HF bucket/dataset or compute3).
- [ ] **CI for Isaac tests** — a self-hosted runner on compute3 or a nightly run; manual for now.
- [ ] **Skill `.claude/skills/run-experiment`**: the EXPERIMENTS.md protocol as a skill (config → budget → run → report → videos → index).

- [ ] **First real run of `scripts/bootstrap_gpu_box.sh` / `bootstrap_fleet.sh`** on the new 4090 / 5090s: record the
  per-step wall times, the copy speeds (HF, compute3 relay, box-to-box direct vs relayed) and the smoke results in the
  journal, and add the hosts to ENVIRONMENT.md. Isaac Sim 5.1 RTX on a 5090 with driver 580 has never been tried: with
  role auto a failed `smoke_render` turns the box into `physics` by itself (RUNBOOK-new-gpu-box §7). Never run end to end:
  the robodojo step, the Isaac smokes, the isaac6 copy (31 GB) and `smoke_proxy_eval`, the fallback to physics.
- [ ] **CLAUDE.md rule 5 still says `.env` lives only locally and on compute3.** The owner approved `~/controlr/.env`
  (OMNIROUTE_* only, mode 600) on our own GPU boxes, and the bootstrap writes it (ENVIRONMENT.md says so); the rule's
  text is the owner's to update.
- [ ] **Isaac Sim 6.0 render proxy without compute3:** physics boxes copy compute3's `6.0.0-rc.59` build (31 GB, the
  one the proxy calibration was fitted on). If compute3 is gone, `isaacsim==6.0.x` from pypi.nvidia.com (cp312) is the
  fallback, but `scripts/simlab/isaac6.sh` expects the standalone layout, and the render would have to be re-checked
  against genuine 5.1 frames (perception calibration).
- [ ] **Fleet: the box-to-box key** (`bootstrap_fleet.sh --p2p direct`) is removed after a run. Streams that want to sync
  data between boxes later need a deliberate, documented key (or the relay through the local box).
- [ ] **5.1-rendering boxes and the 6.0-proxy calibration:** `integ_eval.sh` defaults to `bg_folds` / `color_calib.json`
  (fitted on the 6.0 proxy); on a box that renders genuine 5.1 frames it should take `bg_folds_51` / `color_calib_51.json`.
- [ ] **One default for RD_HOME / DOJO_DATA:** `rdpy.sh`, `dev_eval.sh` and `dojo/simlab/paths.py` default to compute2's
  `/root/controlr-robodojo`, `/root/dojo-data`, while `proxy_eval.sh` / `integ_eval.sh` use `~/dojo-*`. New boxes need
  `source ~/dojo-env.sh` (or the `/root/controlr-robodojo` link as root) until the defaults agree.

## P3 — next directions

- [x] ~~**RoboDojo:** wrap controlr as an XPolicyLab policy server~~ — done: `controlr robodojo-serve` + the shim, both arms ([ROBODOJO.md](ROBODOJO.md)).
- [ ] **RoboDojo runs:** after the `general_pickup` pilot, 6 tasks × 10 layouts (general_pickup, stack_blocks, stack_bowls, push_T, press_by_number, plug_in_charger), then a 5-dimension profile vs Astra's 22.48 % (that needs all 42 tasks × 3 seeds).
- [ ] **RoboDojo: faster motion per env step makes grasps slip** (0.12 rad + gripper 0.5: 0/3, 9 closes per episode; [report](experiments/2026-10-04-robodojo-fast-ab.md)). Separate arm vs gripper speed; fast only while the gripper is open.
- [ ] **Sonnet `reasoning_effort: low` writes its reasoning as prose in the reply** (51 vs 18 words median on RoboDojo) despite the manual's "no prose" — 2.2× faster turns anyway.
- [ ] **RoboDojo: `episode.end_on_fail: false` A/B** — the model declares FAIL with 7–25 env steps left (enough for one more grasp cycle); see [tool-direction](experiments/2026-10-05-robodojo-tool-direction.md).
- [ ] **RoboDojo: the env-step budget binds** (pilot: every failure ran out of `general_pickup`'s 200 steps during re-grasps; [report](experiments/2026-10-04-robodojo-pilot.md)). Levers: `max_chunk > 1`, coarser approach moves, `reasoning_effort: low` for latency.
- [ ] **dojo runtime: one perception vocabulary per instruction, including `_random`.** `stack_blocks` and `stack_blocks_random` (and the other G tasks) share the instruction, so the instruction-matched candidate set must be the union of the std and held-out instances (`dojo/runtime/supervisor.py` VOCABULARIES); today it is the std set only.
- [x] ~~**dojo runtime: mode perception needs catalogue assets for every task the planner serves.**~~
  `python -m dojo.runtime.catalog_check [--sync --sources <skills catalogues> --assets <RoboDojo Assets>]` reports
  and fills the gaps: it copies from the skills catalogues (same builder format), else builds, then runs annotate /
  index / precompute. compute3 (2026-10-25): 314 -> 407 index entries, existing entries unchanged; every table
  vocabulary is complete except sweep_blocks' `Rigid/broom_shovel/00001`, `00004`, which RoboDojo's Assets do not ship
  (only 00000). A test checks an installed catalogue when present.
- [ ] **isr-lab's perception catalogue**: Bridge-isr merged make_toast's assets there. The xylophone / mallet set,
  the mugs + cup_holder set and the other 75 assets added on compute3 are probably still missing: run
  `catalog_check --sync` there before mode-B runs of those tasks.
- [x] ~~**dojo bridge: the official-client path needs robot-harness's `joint_steps` verb**~~: robot-harness PR #9 (J2 server + shim `harness_strict_j2` + `bin/robot-start-j2`); verified offline by `tests/dojo/test_bridge_rh.py`. The Isaac path runs first on our GPUs.
- [x] ~~**dojo bridge: robot-harness's launchers hard-code isr-lab paths**~~: the J2 starter / launcher read `HARNESS_ENV` (robot-harness PR #9); BRIDGE-RUNBOOK §2.
- [x] ~~**dojo bridge: spawn the twin after the runtime's vocabulary choice.**~~ The driver hands the policy a lazy
  `_twin_factory`; `RuntimePolicy.twin(scene)` starts the twin for the scene the episode chose (`bridge.json` →
  `twin_scene`, `twin_start_s`).
- [ ] **dojo skills: twin verification for the library skills that run their own picks / places.** `verify_choose`
  (k candidates rolled in the twin, the first robust one runs) covers only `run_jobs` (stack, tower, swap, put_into).
  insert_into, place_oriented, slots_and_press, hang_on_hooks, fill_container, place_in_slots and stack_on_pegs plan
  and execute their primitives unverified, even with `verify=bundle` (5.1 plug_in_charger s0L9: 0 verified segments).
  - Size of the gap, from the latest G3 run per task on compute3 (`~/dojo-data/skills*/g3`, 2026-10-25), in those
    tasks' failed episodes:
    - physical failures (slip / place miss / loose / fell) 41;
    - checker disagreements 6 (make_toast);
    - planning failures 20, which verification cannot fix (it chooses among feasible candidates).
  - Per task (physical / failed / n): hang_mugs 8/9/10, hang_rand 6/10/10, organize 7/8/8, fill_pen 4/5/5,
    store_tools 3/5/5, deposit_coin 3/4/5, laptop 3/10/10, make_toast 2/7/25 (+2 checker), toast_rand 2/7/16
    (+4 checker), stacking 2/5/5, insert_tubes 1/4/5.
  - Upper bound on what a k-candidate twin check could catch: 47 of ~120 episodes. The realistic share is lower: the
    twin must reproduce the failure and an alternative candidate must pass. In run_jobs tasks verify rescued about 1
    in 10 episodes (mode A stack: 100009's slip), so expect roughly 10-20 episodes.
  - Effort: about 1 day on the skills side (route the shared pick / place / transfer primitives' candidate lists
    through `verify_choose`, keep the twin chain), no runtime change; then a G3 A/B per task.
  - Idea, not needed for plug (Skills-5: in-jaw drift 0.0-0.4 mm): an in-hand transform from the twin's nominal
    replay of an unverified grasp.
- [ ] **dojo runtime: a second look at a weakly measured support before the next object covers it.** 5.1 stack
  100006: block_0's post-place refine was borderline (iou 0.70). The next block hid it completely (visible 0.0 at all
  3 place hovers), and the stack missed by 14-18 mm. Proposal: when a support's post-place iou is below 0.8, the
  releasing arm looks again from 3 cm higher before the next carry, behind a config field (default off), then
  measured on the proxy. It needs a motion step at release time, i.e. a skills hook; the provider cannot move arms.
- [ ] **dojo runtime: a partial refine is not a full confirmation.** perception #52 marks a board fit that cannot see
  along one axis as `partial.unobserved_axis`. Such an ok must not confirm or apply a correction along that axis:
  post-place, place_confirm's second view and the recentre revert should project the correction onto the observed
  axis only, or treat it as unconfirmed.
- [ ] **dojo runtime: label repair after a confident "what is there" (deferred, designed).** held_look already
  returns `what` / `asked_held` when given `candidates` (`j1.held_look_impl`), and `identity.refine_what` returns
  `best` for the post-place. The runtime service passes no candidates yet. Design:
  - candidates = the labels of the asked label's config group, with belief poses (the asked one at its in-jaw /
    released pose);
  - repair when `asked_*` is false and `what` / `best` names another label at conf >= 0.7 and margin >= 0.15:
    1. swap the two labels' belief states (pose, attachment, touched) and the skills' `Held` label;
    2. log `label_repair`;
    3. continue (no drop or re-pick);
  - below the bar, keep the grasp / pose. Same-key twins cannot be repaired by design.
  - Deferred: today's failures don't need it. Stack labels are exchangeable, and tower pieces of one shape share a
    colour.
- [ ] **dojo perception: a multi-view estimate can stack a duplicate pillar on a real one.** In 3 of 23 tower
  scan-primary records (100011, 100014, 100022):
  - a key-00002 pillar sits on top of a real right-side pillar, at z +38 mm (one pillar height) with conf 0.57-0.62;
  - it carries the label of a real left-side pillar, which thereby counts as found while it is not;
  - all 3 episodes failed: a tower "built" on the belief, or picks with 0 feasible grasps next to the phantom;
  - the search's full fuse produced the same kind of stacked find in 100022 (conf 0.80).

  At frame 0 nothing is stacked on build_tower. Perception should reject a piece resting on an identical piece
  unless the views support both. The runtime could also distrust a frame-0 "stacked" piece when the task starts flat.
  Journal 2026-10-25-dojo-integ2 §8.
- [ ] **dojo runtime: search after the scan.** The search regions come from the record only (perception
  `look.search_regions(record)`), and each region's full-fuse fallback fuses frame 0 + that region's wrist view.
  - After a scan, the regions come from the scan-fused record, so the search visits different places.
  - In 100024 the search found 2 of 3 missing pieces in the trigger arm vs 3 of 3 in the off arm.
  - The start pose is not the cause: the looks home every arm they move before the search.
  - Cheap candidate: add the scan observation to the full fuse (`estimate([frame0, scan, region view])`) as a config
    switch, A/B on the tower layouts with not-found pieces.
- [ ] **dojo runtime: fixture-aware place refine.** Under perception cb5c3a4 the place refines on the tower's base board
  are all "not ok" (the static fixture keeps its prior), and each place still spends 8-10 steps on 2 re-hovers (~50
  steps per 8-piece tower episode, journal 2026-10-25-dojo-integ2 §5). A stationary support whose prior perception
  kept + a not-ok place refine -> accept the prior, no re-hover.
- [x] ~~**dojo bridge: two instruction routers for perception.** `j1_adapter.infer_task` (the server's `perceive` route) knows 6 tasks; the runtime's `supervisor.perception_vocabulary` knows more. One function should serve both.~~ j1-adapter-3: `task_for` falls back to `perception_vocabulary`; `backgrounds_for` applies the scene backgrounds (journal 2026-10-27-dojo-official §3.2).
- [ ] **dojo runtime: the twin service is started per task by the launcher** (`--task`). A submission must start / load it from the instruction-matched scene; the policy already refuses to verify when the twin's task differs from that scene. Done for the official-client path: `dojo.bridge.twin` spawns it for the instruction's scene (`perception_vocabulary`); the proxy launcher still takes `--task`.
- [x] ~~**dojo bridge: the twin-layout template needs RoboDojo's generator deps in OUR venv**~~ (`tqdm`, `transforms3d`, imported by `utils.cluttered_generator`). Without them every twin verify of a bridge episode failed (`ModuleNotFoundError`, verify disabled). Now in the perception extra, with a CI guard (journal 2026-10-26-bridge-isr).
- [x] ~~**dojo bridge: twin requests carried numpy-2 objects into RoboDojo's numpy 1.26**~~: they failed to unpickle (`numpy._core`) and the service died. `TwinClient` sends plain lists and numbers now (`twin_service.plain`).
- [x] ~~**dojo bridge: `bridge_episode.sh` orphaned the perception service**~~: it killed only the subshell's bash. The orphan held fd 9, the session lock, and the next episode waited on it. The service is now exec'd, with fd 9 closed.
- [ ] **dojo runtime: no perception vocabulary for make_toast, make_toast_random, store_laptop_and_headphones, hang_mugs** (`supervisor.VOCABULARIES`), so they cannot run in mode perception, i.e. not on the strict official path at all. With no rule skill, push_T and cover_blocks end at once under RuntimePolicy (`no skill for`); they need the agent.
- [ ] **dojo twin: stack_bowls verify fails in the twin service**: `Pickling of "pxr.Usd.Stage" instances is not enabled` (isr-lab, ilbr-rt-bowls-s0L9-1). Probably a deepcopy of something holding the env during load / verify. It never showed in the proxy, where stack_bowls ran with verify off.
- [ ] **dojo perception: swap_blocks' target cube (Rigid/cube/00003) has geometry only in the catalogue** (no `visual_lo.glb` / silhouettes), so B1 finds no foreground and the episode ends at frame 0 (strict s0L9: "estimate lacks target0, target1").
- [x] ~~**dojo perception: held_look / post-place false negatives on genuine 5.1 frames**~~: 12 of 17 "not held" verdicts in the dev eval had the block in the jaws, and false post-place failures hit correct stacks; the runtime then undid the progress. Fixed by main #41-#46 (jaw opening + absence cap, head-confirmed release, never undo, post-place z-only adoption). Measured on the same genuine-5.1 layouts: stack 6/10 -> 9/10 (journal 2026-10-26-bridge-isr A5-A6).
- [ ] **dojo perception: a placement-truth rule for post_place calibration.** 186 regenerated 5.1 frame sets with truth are in `isr-lab:~/data2/ilia/dojo-data/bridge/replay_frames/all/` (index matched by asset key + position). The 92 post_place checks still need a target-vs-truth rule before ok / not-ok can be scored.
- [ ] **dojo perception: build_tower 100008 identity gross at planning** (block3 read as block1, 12 mm) on genuine 5.1 frames. It is the only tower loss on main f9b0d75 (9/10), and the same mode cost it on b8acf78. The dev-eval report and actions are in `isr-lab:~/data2/ilia/dojo-data/bridge/dev_eval/dev51-build_tower-dc63679/` (journal 2026-10-26-bridge-isr A9).
- [ ] **dojo runtime: s0L9 build_tower "believed built", checker says not** (`ilbr-scan-tower-s0L9-1`, 3/8 identity-gross at planning). The eval layout cannot be replayed on our side; the kept frames are in `runs/<id>/frames/`.
- [ ] **isr-lab: Isaac's texture cache under our shim HOME grows ~0.1-0.4 GB per new task's assets** (1.6 -> 2.4 GB over 10 tasks); data2 has ~7-9 GB free. Never prune it (or the shader cache): the rebuild spiked the box to ~20 cores. The block gate stops below 7 GB free instead; free space must come from elsewhere (run dirs, or Mikhail's side).
- [x] ~~**dojo skills: the perception catalogue lacks the skills2 tasks' meshes**~~: it has them. The files are md5-identical to `skills2/catalog`, and G3 has read perception's catalogue all along. The skills3 tasks' new assets (coin, coin stand, piggy bank, stacking toy, toaster, bread, bread shelf) are in `compute3:~/dojo-data/skills3/catalog` (`DOJO_CATALOG`); perception should add them.
- [ ] **dojo skills: thin irregular tools still slip out** (store_tools). Two causes are fixed. First, every box grasp of a thin lying tool failed the table margin (the fingertip estimate sat 1 mm low; now taken from the finger model). Second, there were no grasps near the centre of mass (now stations every 6 mm). The hammer at its CoM lifts with 11-13 mm drift and is now carried on. Still open: the pliers' outline is rounded everywhere, so the patch is <= 0.08 at the joint. Try the handles' outer faces, or collider-based grasps; the colliders are convex decompositions of `/root/collision/model`, not the visual mesh. Skills-8 (journal 2026-10-26-dojo-skills8): the tools spawn lying as their slots want them, so a level carry suffices. A straight-down pinch at the centre of mass holds (hammer: 4-7 mm drift over a 15 cm carry); a 25-deg tilted one turns 37-81 mm. Straight down does not reach most slots, though: the IK at the slot fails for 28 of 40 hammer, 34 of 36 wrench and 16 of 16 tape-measure grasps. PhysX's centre of mass differs from the visual mesh's: the hammer's is 12.4 mm away, because the colliders are inflated to 20 mm `minThickness`. Read it in the twin (`RigidPrim.get_com`), e.g. into the catalogue, before ranking grasps by lever.
- [ ] **dojo skills: planning time on irregular meshes**. Done: meshes are clustered under 12k faces, hold quality is cached per grasp in the object frame, and the relay search is capped at 150 s (one store_tools episode had spent 21 min). Open: IK per (arm, grasp) is recomputed for every relay leg.
- [x] ~~**dojo skills: with a refine hook, a transfer's pick is re-planned with any grasp** (skills-8, mode-B store_tools). `Skills.execute_pick` hovers, refines, then calls `plan_pick(label, arm, "auto")` with the default candidates, so the executed grasp can differ from the one `plan_transfer` paired with the place. Seen in mode B: a tape measure was planned with a 25-deg pinch, executed straight down, and the place re-planned for that grasp found no IK at the slot; the nominal plan put it 33 deg tilted. A level carry's custom grasp source is lost the same way. Fix: re-plan with the planned grasp moved by the refined pose (`grasps=[...]`), falling back to the defaults; regression-check every `plan_transfer` user in mode B (plug, tubes, coin, slots, fill).~~ Fixed (skills-9, journal/2026-10-26-dojo-skills9.md): the same grasp in the object frame at the refined pose, re-paired with its place only when infeasible; mode B store_tools 0.10 -> 0.40, deposit_coin 0.00 -> 0.68.
- [x] ~~**dojo skills: insert_key's handover giver.**~~ Pegs (journal 2026-10-27-dojo-pegs §2-4): `regrasp.rest_handover`. The giver pinches the key lengthwise (bow end + tip: 0.1 mm / 0.1 deg in hand) and hands it over still lying. The taker's bow-face grasp is chosen at the lock. The key is inserted and let go; a fresh grasp in the lock turns it.
- [ ] **dojo skills: insert_key's lock is narrower than the key's teeth.** In the collision geometry the rotation cavity leaves r 12.0-12.5 mm between -15 and -36 deg, against the teeth's outer corners at r 12.7-12.9. A key turned on the floor catches at -14 deg and tilts. Turned 0.3-1.2 mm above the seat, it snaps through at -30..-36 commanded with 5-7 deg of tilt for a step or two, which is when the checker can pass (`regrasp.TURN_LIFT`). Mode B has no measurement of that window.
- [ ] **dojo skills: IK's wrist +-pi seeds flip stepped rotations.** `X5Kin.ik` falls back to the wrist +-pi seeds when the local solution runs out (joint 6 at its limit). A stepped turn then jumps half a turn: the key spun 117 deg out of its lock. `regrasp` rejects steps that move a joint > 0.3 rad. Other stepped rotations (screw sweeps, the place tail's turn) may need the same guard.
- [ ] **dojo skills: insert_key mode B.** With #131's keyhole read wired (`regrasp._look_over`): rendered 5.1 100003 **1.00** (submit_v1, the first). G3 (lock believed 10 deg off, key 2 mm, held-pose noise) 7/14 with the read vs 0/14 without. Learned in the twin (Pegs journal §4, §9): the taker's bow grasp is a hinge about y_key (a joint-space carry swung the key 8 deg; turned up about x_key + an eased Cartesian carry: 0.1-3.5 deg); a key on the lock's top slides and pivots in the pads instead of stopping the arm (0.5-1 mm of lag: read in fine steps, not `GUARD_LAG`); the look's arm must be home before the taker moves in (its unchecked pre-grasp move knocked the key out). Open: the step budget with an in-hand read (4 of 14 out of the 300 at the turn); the key's place opting into #136's `held_read`, judged on rendered episodes; 100010 lost after the release.
- [ ] **dojo skills: mode B's planned place tail never turns.** Only `_servo_tail` honours `turn`. A direct `place_oriented` with `turn` and no in-hand read inserts and lets go unturned (rendered 5.1 insert_key 100003: 0.15). `library._turn_if_not_turned` turns such a key with a fresh grasp (`regrasp.turn_in_lock`). Any other user of `turn` gets nothing.
- [ ] **dojo skills: insert_key mode A, 4 of 14 left** (10/14 with the slip fallback). 100010 / 100023: the key is let go 6 deg tilted (xy 3.3-3.5 mm). No grip in the lock plans the turn from the target upright, and turning it as it stands jams (14.6 deg). Holding on to turn with the handover's grasp lost more than it won (k5 / k6). 100001: no handover pose. 100013: the place's approach hits the table. (100024's slipped direct pick now goes to the rest handover: 1.0.)
- [x] ~~**dojo skills: fill_pen_holder, the collider's opening is not the visual cup's**~~ (skills-8). It is open: pens let go 4 mm over the rim within 18 mm of the axis fell to the floor 39/41 in the twin. The jam came from carrying each pen down to the floor with the holder ignored by the collision model and grasps near the tip catching the rim. `fill_container` now drops each pen over the opening.
- [ ] **dojo skills: parked tasks** (lead, 2026-10-25). insert_key: lift + return banks 15; the in-air handover lost the key on the carry 0/31. store_tools: keep v2 (0.20); the pliers slip, and the hammer and wrench are lost on the carry. fill_pen: unparked by skills-8 (drops). insert_key: skills-8's stand-up against the table also failed (see above). Revisit after the Generalization tasks.
- [x] ~~**dojo: make_toast_random's slot checks cannot pass in our twin.**~~ Resolved: the official rendered kit writes physics to USD (live link poses), and the physics-only runner now reads link poses from physics by default (`usd_sync_links`, journal 2026-10-25-usd-sync-default). RoboDojo's `get_functional_points` returns toaster 00002/00004's frames untransformed (object frame, z ≈ -0.03). Those assets lack ArticulationRootAPI, and `Articulation.get_link_pose` reads the USD xform. Skills place physically; score rand with `scripts/skills/phys_slots.py`. Behaviour in the official env is unknown upstream. Evidence: `g3 checks` (journal 2026-10-25-dojo-skills4 §3).
- [ ] **dojo skills: store_laptop's laptop half (80) unexplained.** In our twin, the checker's `bottom` frame ends 78 mm above the table for any +y-up laptop, while the rack's bbox is 49 mm tall. The public demo's end pose is upright in the rack. Find what the official checker does differently before building the lid-close + stand skill. Headset half: `hang_on_hooks`.
- [ ] **dojo runtime: watch reads for observation-driven skills** (skills-6, journal 2026-10-26-dojo-skills6 §6).
  `play_board_game` and `copy_demonstration` call `provider.watch_read(region, prior)` (perception-R&D's dict shape:
  `cells` / `container`) and `provider.actors_moving()` while they hold still. `match_and_pick` needs object tracks on
  the belt. G3 answers all three from GT (`GTProvider`); `RuntimeProvider` has none of them yet. Without
  `actors_moving` the skills hold a fixed 60 / 110 steps, tuned on our own demos.
- [x] ~~**dojo skills: fill_egg_holder, a 0.3 kg egg falls out of the pads while the hand turns it upright**~~
  (twin v3-v6). Not needed: `fill_holes` mode "drop" (tools stream) carries the egg lying as picked and lets it go low
  over a hole, off its centre, so it tips in (journal 2026-10-27-dojo-tools). The stand-up remains as the fallback for
  the back holes under the open lid.
- [ ] **dojo skills: fill_egg_holder, eggs against the basket wall.** Most dev eggs rest on the wall's foot: no
  top-down grasp keeps the finger's root block (38 x 56 mm) out of the wall. `nudge_from_wall` pushes one off when its
  fingertip path fits the gap on the wall side (some do not). Eggs on the table under the basket's flared rim have no
  grasp at all. Next: a push along the wall to a freer spot, tilted grasps from the basket's middle.
- [ ] **dojo skills: fill_egg_holder, an egg can land lying across a hole** (its centre 2-5 mm over the plate, the
  checker wants it in). Two presses from above did not free it; ordering the drops so the far end rests on the outer
  rim (not on the wall between two holes) did not help (G3, 11 layouts). `slide_in` pushes it along its axis until
  one end is over the hole (G3 mode B 8 -> 12 eggs); mode B still lands half as many as mode A, because the egg's
  in-hand position along its axis (10-14 mm lower than believed) is not measured: a held-pose look would close it.
- [x] ~~**dojo runtime: an egg let go over the egg holder's hole is predicted on the table**~~ (and a bread slice let
  go in the toaster's slot: breadth3's make_toast "dz -62.6 mm"). An articulation's static links are now a fixture
  (`static_link_mesh`: the egg holder's body, the toaster's base; not the lid or the lever). Its moving links are
  left out at any joint state; a fixture rule for them needs the joint's current position in the belief.
- [ ] **dojo skills: hang_mugs, a mug lying on its side cannot be stood up or hung** (G3). No pick + upright-place
  pair exists: the place puts the X5's wrist horizontal low over the table (IK fails within 13 deg of horizontal).
  Only a rim-to-base pinch plans a direct hang, and it slips. Next: an in-air handover that re-orients it, or a
  handle grasp. In mode A 14 of 62 mugs fall at the release after a good place (journal 2026-10-27-dojo-tools §8):
  preventing that comes first.
- [ ] **dojo simlab: pick_from_conveyor_by_image's photo board and basket placement is a public guess.** v12 put the
  basket's 22 cm body inside the conveyor (ejected at load). v13 (skills-6) follows the public PhysicalRSI sampler's
  `place_fixture` (world xy from the conveyor's root, z from its world bbox top): the basket stands on the belt in front
  of the left arm, the board at the robot side, 1.02 m up. PhysicalRSI marks that convention "native scene not
  validated"; check it against the official videos / a rendered episode when one is available. Skills-6's layouts are
  in `compute3:~/dojo-data/skills6/layouts/pick_from_conveyor_by_image` (simlab's shared copies are still v12).
  - The breadth3 by_image episodes ran on v12: its board sits behind our right arm's links in the head view (not
    readable); v13's board is fully in view (72 x 93 px). Public code cannot settle it (RoboDojo only loads layout
    files; PhysicalRSI's own docstring: "no historical evaluation layout supplies board/basket transforms"), so
    perception should find the board by appearance in whatever view shows it, not at a pinned pose (journal
    2026-10-27-dojo-perception3).
- [ ] **dojo perception: imitate_sorting_sequence, what frame 0 still gets wrong** (experiments/2026-10-27-imitate-frame0,
  plan right 8 / 13 offline): category swaps among the demonstrator's small far objects (action camera / garage figure
  / watch: 100016, 100018, 100023) and objects not found (a far phone, a near watch, a far watch hidden behind a figure
  at frame 0). Candidate: pair each far object with the near objects' assets (same asset per pair, the near side's
  categories are 64 / 65 right) by render-and-compare at the far blob.
- [ ] **dojo perception: no 6.0-proxy scene background for imitate.** The genuine-5.1 one is in every box's
  `bg_folds_51/fold{0,1}/scene/Geometry_basket_00005/` (`scripts/perception/scene_bg_frame0.py`); compute3's 6.0
  `bg_folds` has none, so proxy imitate runs still see the demonstrator's arm as foreground. Build it from 6.0 renders
  of the non-demo layouts' frame 0 when proxy imitate runs matter.
- [ ] **dojo perception: the scan estimate assumes nothing moved since frame 0** (`scan.estimate_with_scan` fuses frame
  0 as extra views, then `fill_from_frame0` fills a not-found label from frame 0's reading). On a conveyor the riders
  move 4 mm / step, so a rider can appear twice: at its scan-time pose and as a frame-0 ghost under another label.
  The belt tracker drops belief objects no re-read ever confirmed (`RuntimeProvider._track`, 2 reads), but the scan
  itself (14 steps + a fused estimate) buys nothing there. Candidate fix: no scan (or no frame-0 fusion) when the
  scene has a Dynamic base (journal 2026-10-27-dojo-memory).
- [ ] **dojo twin: swap_blocks' cube/00003 never moves in any pick rollout.**
  - G3 `--verify 2`: all 48 pick rollouts of 100001 were rejected with "rise 0 mm". Both cubes stay at their start
    poses in every rollout: an identical-asset label swap was ruled out with a re-assignment by position, which never
    fired.
  - stack_blocks verifies fine (6/6 first_ok).
  - Official-Path saw none_ok on every segment on the rendered path too (~72 s of twin time per episode, no
    selection). The rendered outcomes are 10/10, so it costs time, not points.
  - Suspects: a non-simulated pose read for that asset in the twin envs, or the cube held by the cushion mat.
  - Tool: the judges' reject reasons (`Skills.vreasons`, `rejects` / `reject_examples` in G3 and runtime verify
    reports).
- [ ] **dojo skills: match_and_pick, objects wider than the hand at the top** (ice cream 104 mm on a 69 mm cone, 3 of
  25 dev targets) have no intercept grasp within 40 deg of vertical; a side grasp of the cone on the moving belt.
  Memory (2026-10-27): `plan_intercept` falls back to `waist_grasps` (level, the pad gap from the mesh at that height);
  G3 result in journal 2026-10-27-dojo-memory.
- [ ] **dojo skills: sweep_blocks** (lead: a real broom sweep only, no pick-into-pan; low weight, after the others) and ~~**fold_clothes** (parked: no cloth state in the skills layer)~~ — `fold_garment` (#103); Sol, Sonnet and the planner route fold_clothes (+ `_random`) to it 6/6 each (journal 2026-10-27-dojo-official §2.2).
- [x] ~~**dojo skills: deposit_coin.** The coin starts out of the inserting arm's reach in about half the layouts.~~ Rim grasps tilted in the coin's plane reach 12/25 with one arm; the rest go through an in-air rim handover (the giver turns the coin on its side, the taker takes its top rim 8 mm off-centre). The coin is 28.4 mm x 1.9 mm (the collider; `physics.size` in its metadata says 21 x 1 mm). Two grippers can share it this way (journal/2026-10-26-dojo-skills5.md).
- [ ] **dojo skills: a rim-held coin slides in the pads as it leaves its stand** (G3 2/25, journal
  2026-10-27-dojo-precision.md §3). 100002: the coin slid 5.2 mm down in the pads during the lift out of the stand
  (cover 9 -> 3.8 mm) and fell out early in the carry; 100019: it slid back into the stand. Ideas: a longer squeeze, a
  re-grasp when the measured cover is under `DISC_MIN_COVER` while the coin is still over its stand. Also note: a hand
  on its side at the slot (finger roots over the bank) is now avoided only for the handover's taker (`hand_down_first`).
- [ ] **dojo skills: a flat coin can be pinched but not inserted.** Pinched across a diameter with the fingertips 0.6 mm over the table, it lifts and turns on edge (3/3). The hand then points sideways, and the X5's 37 mm finger roots hang 28 mm below the TCP over the bank's back, so the coin stays 7 mm above the slot (table relay 0/7, `DISC_TABLE_RELAY = False`).
- [ ] **dojo: the insert_tubes depth term is met by a tube lying on the table.** `is_A_depth_in_B` is rack bbox top - tube bbox bottom > 45 mm, and a lying tube's bottom is 50 mm below the rack's top. Not a gaming risk (the skill inserts), but any score from a "partial" episode must be read with that in mind. The rack's holes are 54 mm deep.
- [ ] **dojo: asset metadata bboxes are not the colliders.** The test tube's metadata bbox is 42 x 140 mm, its mesh 35 x 117 mm (the checker reads a 42 mm square). The coin's checker bbox is +-20 mm against a 14.2 mm radius. A coin resting tilted inside the bank can fail the bank's 8 mm bottom bound with its box corners. The skills' collision model uses the metadata box: standing tubes are therefore registered with their mesh height map (`Collision.heightmaps`).
- [ ] **dojo: RoboDojo Geometry objects are static colliders** (`GeometryObject._remove_rigid_body_if_exists`): a coin bank, a coin stand, a bolt, a rack, a xylophone cannot be moved or pushed. Plans that move a receptacle closer only work for Rigid ones.
- [ ] **dojo runtime + skills: measure the in-hand pose at the pre-insert hover in mode B** (perception #59,
  `PerceptionClient.held_pose` / `.show_pose`, experiments/2026-10-26-held-pose.md). Integration-2:
  `provider.held_pose(arm)` (frame, held key, `Held.T_tcp_obj` as the prior, the place target in cfg). Skills: in the
  shared insertion tail, adopt `T_tcp_obj` when `ok` and the sigma suits the fit; otherwise an observer look
  (`show_pose` hover_look, ~14-19 steps, can overlap the approach), then a show pose; protocol per object class in the
  report.
- [ ] **dojo perception: the own wrist camera cannot separate a turn about the closing axis from a slide along the
  approach** for box-like held objects (the charger; a flat valley of the cost). `held_pose` reports it (sigma along the
  approach 3-7 mm, `why=ambiguous`); a second view fixes it. Ideas: a photometric term on the shading of the faces
  (tried flat shading + NCC: no gain on 5.1 blocks), a learned keypoint for the leading end.
- [ ] **dojo perception: `show_pose`'s predicted sigma is relative.** The silhouette information assumes independent
  pixel noise and is far below the measured error (ranking only). Calibrate `SIGMA_PX` against `held_eval` errors per
  option before a skill compares options by it.
- [ ] **dojo skills: plug_in_charger in mode B is bounded by the in-hand pose** (journal/2026-10-26-dojo-skills5.md).
  The blades are 0.8 mm in 1.0 mm slots (about +-0.1 mm), and a 3.5 mm belief error at grasp time leaves the blades on the
  strip. A discrete probe grid would need about 0.2 mm spacing (hundreds of probes); a pressed spiral to 5 mm is about 260
  steps, against ~170 left. Even with an exact belief (no offset) mode B goes 0/4: G3's servo works because it re-measures
  the held object 3 mm over the strip. Needs that measurement (a held-object view at the hover, about 0.3 mm), then a
  short spiral. The contact-probing tail (`Skills._guarded_tail`) is in place for mode B and is harmless (the place
  completes, nothing drops).
- [x] ~~**dojo skills: insert_tubes, the second tube from the same side.** After the first tube stands, the same arm often has no plan into the remaining holes of its half of the rack (no grasp-place pairs, see the journal), and the relay fails too.~~ Fixed (journal 2026-10-27-dojo-precision.md §4): an as-is relay of the lying tube (`place_oriented(as_is_lying=)`), one-arm tubes first, places within 0.56 m, off-CoM plans after the relay; G3 mode A 16 -> 24/25.
- [ ] **dojo skills: insert_tubes in mode B runs out of steps on relays.** A relay leg costs ~110 steps (put-down + re-pick); G3 mode B relay episodes end at 471-485 of 500, and proxy B spends ~440 on three direct tubes. Levers: a cheaper put-down (lower pre-place, faster descent), fewer mode-B refine hovers per pick, or a hole assignment that never needs a relay.
- [ ] **dojo skills: the right arm into the rack's far-left front hole** (0.562-0.567 m from its base) is held 46-56 mm off its command and misses (0/4), while the left arm's mirror place at 0.596 m goes in (2/2): `EACH_REACH` 0.56 m avoids it; the asymmetry is not understood.
- [ ] **dojo: hang_mugs judges only the FIRST frame on each side** (`is_axis_aligned` takes `fp_poses[0]` of the mug's
  `handle` and of the rack's `handle_support_point`; journal/2026-10-26-dojo-skills7.md).
  - Only the hooks oriented like the rack's hook 0 can score: 4 of 6, the ±x arms at two heights. The ±y middle arms
    never pass the x-axis test.
  - Mug 14 (the `_random` set) has handle frame 0 = identity, so its z is the mug's axis. Hung by its handle that axis
    is horizontal; the test wants it within 60 deg of the hook's z. A mug 14 hung on a hook can never pass.
  - 17 of our 25 rand dev layouts hold at least one mug 14, which caps them at 40 or 15. The rand ceiling on our set is
    about 55 (official draws: about 68). This is part of why the board's rand numbers are low (PyRUA 28.4).
  - `hang_on_hooks` uses only the hooks oriented like the first one, and skips handle frames whose x does not run
    through the handle's hole (`threads_along_x`).
- [ ] **dojo: play_stacking_toy's 1 mm pair check is tighter than the pegs' clearance.** The holes are 5.2-5.55 mm in
  radius (from the meshes) and the pegs 4.83 mm, so a piece moves up to 0.65 mm off-axis. Let go at the peg's top, the
  pieces of one peg ended 1.0-1.14 mm apart (twin v4.1/v4.2: 4 of 25 episodes at 60 instead of 100). `stack_on_pegs`
  now seats each piece on the one below, against the same side of the peg, servoed to 0.3 mm.
- [ ] **dojo skills: `plan_transfer`'s `yaw_sym` turns targets about the bbox centre.** A place whose symmetry axis is not
  the bbox centre (a star-shaped ring's hole is 2.9 mm off it) must pass `yaw_about`. Before that, 3 of 4 yaw targets
  missed the peg by 4-6 mm, and every tilted piece in skills-3/4 was a star.
- [ ] **dojo skills: `free_spots` ignores Geometry fixtures** unless they are passed in `avoid=`: a relay spot can land
  on a fixture between the arms (a peg base). `place_oriented(spot_avoid=)` passes them.
- [ ] **dojo skills: mode B and pieces on pegs.** Without an object measurement the guarded tail is all there is. A ring
  stopped by the peg's rounded top slid 2.6 mm in the pads, and the arm lagged only 1.7 mm against `GUARD_LAG` (4 mm),
  so the ring was let go tilted on the top (G3 `--mode-b`, 2 mm belief offsets: 0.6 / 0.6 / 0.1). Pegs now pass
  `guard_lag` 1.2 mm / `guard_seated` 1.5 mm and retry a piece left tilted. The real fix is a held-object or peg
  measurement at the hover, like plug_in_charger's.
  - Seated blind (v4.6), a ring pushed 30-50 mm down the peg on a predicted pose binds and tilts: 0/5, 0.30. v4.7 threads
    4 mm blind instead: 1/3, 0.73.
  - ~~Open gap for the insertion-tail stream: the seat target ran blind when the `held_pose` read failed.~~ Closed
    (skills-9): in mode B the pieces stay threaded (`pegs.PEG_SEAT_MODE_B` off) and the guarded tail runs on the
    held-object read when there is one; seated on a read they did worse in G3 (0.78 / 0.36 against threaded 0.92 /
    0.76, pieces / pieces + base offset: the read fixes the hand, not the peg, and the servo's held pose cannot see a
    ring caught on a peg top). If seating is turned on, a seat without a read <= 0.5 mm aims `blind_up` above it (the
    threaded depth), never pushes down blind (journal 2026-10-26-dojo-skills9.md).
  - Hang has the same gap: mode B 0/3, 0.10 against mode A 1.0 / 1.0 / 0.4 on the same layouts. The mug's 2-10 mm slip
    in the wall pinch is never re-measured, so the handle misses the hook's arm (journal/2026-10-26-dojo-skills7.md).
- [ ] **dojo skills: planning time with mesh-box obstacles.** `hang_on_hooks` plans 200-450 s per episode: rack and hung mugs
  as `Collision.parts`, two handle frames x two lift heights per hook, and the relay search. The batched parts check
  (3 ms) made it about 10x faster; most of the time left is IK screening.
- [ ] **dojo skills: the X5's practical limits are tighter than the IK's.** Places 0.66 m out, and places with the wrist at j6's stop, stalled in the twin. `PLACE_REACH` 0.60 m and `wrist_safe` (j4 <= 1.6, abs(j6) <= 3.0) filter places only. Picks and handovers still use the full IK range.
- [ ] **RoboDojo: the envelope knows neither the objects nor the other arm**; cuRobo plans each arm against the table and itself only. Arm-arm collisions are possible (sim only; never on hardware like this).
- [ ] **Real UR3 backend** via PHANTOM drivers + `SafetyMonitor` (RTDE, Robotiq, RealSense).
- [x] ~~**A real arm**~~ — SO-101 with a wrist camera: `robot.backend: so101` ([SO101.md](SO101.md)); first motion and live runs pending.
- [x] ~~**SO-101: first motion**~~ — jog ±20 mm per axis within ~1 mm after sag compensation; VIEW directions match the image; no stop on free moves (journal 2026-10-14-so101-backend).
- [ ] **SO-101: stop thresholds** — set `current_stop_ma` above the peak currents of free moves (`turns.jsonl` → `backend.runs`); `track_stop_deg` 12 never fired.
- [ ] **SO-101: only one jaw in the wrist image** at home — calibrate, then pick `tool_roll_deg` / home so both jaw tips are visible.
- [ ] **SO-101: hand-eye calibration** (`scripts/so101_calibrate.py`) — until then the stock URDF camera mount; then enable `grid` / `ee_marker` overlays and compare against `raw` (a config axis).
- [ ] **SO-101: gripper width** — the 0..100 % → 0..70 mm map is linear and unmeasured; measure the jaw opening at a few settings.
- [ ] **SO-101: camera-frame moves** (`MOVE cam_delta right forward up`): with a wrist camera the base axes rotate in the image as the arm pans; a move in image directions may suit the model better than base axes + VIEW. New grammar mode = a config axis.
- [ ] **SO-101 on a robot vacuum** (owner's goal: the arm on a Roomba-like base) — after the fixed-base rig works; needs a mobile-base backend (base pose in the frame chain, `cmd_vel`-like moves).
- [x] ~~**Which UR3 is it — CB3 or e-Series?**~~ — **CB3** (Ilia, 2026-10-03). No built-in wrist F/T: contact sensing on hardware = `getActualTCPForce` (estimated from joint currents, coarse) + UR protective stops, unless a Robotiq FT-300S is added. Note: PHANTOM's `configs/hardware.yaml` still says `generation: e-series` (500 Hz rates, `wrist_ft.source: ur_internal`) — PHANTOM's to fix, not ours.
- [ ] **Ideas from Waddle Labs** ([research/sources/waddle-labs.md](../research/sources/waddle-labs.md)): a recovery rule (when already out of bounds, accept a command that reduces one violation without worsening others — generalises the straight-elbow fix); tag every executed action by source (model / safety clamp / human) and log holds and e-stops (needed for the real UR3); success-vs-cost curves across effort levels with n and CIs; a cross-episode "lessons" appendix in the cached prefix as an ablation; their 32 MuJoCo task environments (Apache-2.0) as a task catalogue.
- [ ] **Export runs to LeRobotDataset** — data for a future light action head.
- [ ] **Action head** (transformer/diffusion) closing the ~1 s between model turns.

## compute3 infrastructure

- [x] ~~**Slow VPN on compute3**~~ — Happ/AmneziaVPN/OpenVPN removed, machine-wide `controlr-vpn` (sing-box tun, router direct: 740 KB in 0.43–0.48 s) (journal/2026-10-04-compute3-network.md). Remaining: ethernet on `eno1` (Ilia).
- [ ] The meta package `linux-modules-nvidia-595-open-generic-hwe-24.04` lags the kernel → the next kernel update will drop the GPU again ([incident](incidents/2026-10-02-compute3-nvidia-driver.md)). Upgrade it together with the kernel (agree with the machine's owner).
- [ ] Delete the old `~/controlr-dev-isaac`, `~/controlr-dev-loopcli` (≈100 MB; their `.env` copies are already deleted).
- [ ] **dojo runtime: wire `provider.show_pose`** (skills-9's held-pose protocol, `Skills._held_protocol` / `_pre_look`).
  The skills call `provider.show_pose(arm, T_tcp_obj, asset_key, cfg_)` and expect perception's `show_pose.propose`
  dict; `cfg_["q_hold"]` (when present) = the holding arm's joints to plan for (the hover it is approaching: put them in
  `state`). `held_pose`'s `cfg_` also carries `view` (hover / look / show; perception may ignore it). Until it is wired
  the protocol reads at the hover only.
- [ ] **dojo skills: G3 mode B keeps a released object's belief offset forever** (`g3.OffsetBeliefProvider`). A support or
  a piece already placed stays off by its offset in the skills' own checks (`pegs.on_peg`, `verify_place`), and the
  null refine hook never corrects it (the runtime's place refine would). Pieces 2 mm off against a base 2 mm off the
  other way read 4 mm off their peg ("not on") while the checker counts them. Read G3 mode-B numbers with that in mind;
  `--held-pose` changes only the held object.
- [ ] **dojo skills: in G3 mode B a record's `final_objects` is the BELIEF** (the provider's refresh), not the truth;
  compare `final_true` (g3 hook runs record it). A skill that measures its own progress on an object the belief cannot
  see must model it instead: fasten_screws mode B read "final poses equal to mode A's" while the belief had turned each
  nut down with the hand (2.6-2.8 mm a sweep against a true 0.7-0.9 then 2.2-2.5) and the loop stopped a sweep early,
  ~1.5 mm short of the checker's 9 mm (`screw_on`'s depth model, journal 2026-10-27-dojo-precision.md). Others that
  count on a re-measure after a release onto a fixture: insert_each's "in a hole" check, `disc_into_slot`'s inside test.
- [ ] **dojo skills: held-pose protocol, per class** (journal 2026-10-26-dojo-skills9.md). With the merged estimator's
  calibrated sigma a test tube's read is almost never ok (sigma about 3x its error, perception-held's report), so the
  tubes stay on the fallback; a coin's own-camera read under-covers (perception-held: never adopt it without a second
  view), which the protocol does not enforce yet (`disc_into_slot` places pass no `held_tol`).
