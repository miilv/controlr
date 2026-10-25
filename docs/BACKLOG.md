# Backlog

> The task queue. Take an item → branch → PR linking the item → strike it here in the same PR
> (`- [x] ~~…~~ (journal/… or experiments/…)`). Found a landmine — add it here.
> Sources: [experiments/](experiments/), [reviews/](reviews/), [journal/](journal/).

## P1 — before the next round (carry failures and perception)

- [ ] **refine() reports ok for a wrong fit, 20-63 mm off.** 10 of 291 cases in the D0 wrist validation
  (`dojo/perception/val_refine.py`, priors ±20 mm, `WITH_OTHERS=1`), IoU 0.62-0.86, mostly an identical
  pillar / block nearby:
  - build_tower 100003 f1 block1, 100004 L1 block1, 100007 L1 block1, 100009 L1 block2, 100018 f1 + f2 block1,
    100022 L1 block6;
  - stack_blocks 100002 f2 block_2, 100015 L1 block_1, 100017 f2 block_2.
  
  Candidate fix: score the prior's own position against the best fit (prefer the prior unless the fit gains
  clearly), or require the neighbour guard against all same-shape objects of the belief. This is the next tower risk
  after the scan.
- [ ] **Corner look when the expected count is not met after scan + agent.** If the scene's vocabulary expects more
  objects than the scan-primary estimate + agentic answer report, look at the head-view regions the scan pose still
  cannot resolve: the table corners and edges, |x| >= 0.4. The case is build_tower `block4`, a green prism at
  x = -0.43 that no reader found in D0 100024. A perception-side `look_plan` request
  (docs/experiments/2026-10-25-scan-pose.md).
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
- [ ] **dojo agent: perception is the bottleneck in mode B** (Phase 2, 2026-10-25): stack_bowls 0/6 and build_tower
  100000 0/3 were perception_gross for every model; put_bottles 0/6 were put_into drop misses / pick collisions
  (verify off). The model choice did not change these outcomes ([report](experiments/2026-10-25-agent-e2e.md)).
- [ ] **dojo agent: align_blocks' set square has no grasp** (skills: 0 feasible / no top candidates); Astra solved
  one layout by sweeping the blocks with a fingertip `move_to` line. A tool-push skill would make it a skill task.

## P2 — harness and infrastructure

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
- [ ] **dojo runtime: mode perception needs catalogue assets for every task the planner serves.** compute3's perception catalogue holds the D0 tasks only; swap_blocks / press_by_number (mats, buttons, cards) cannot run in mode perception until their assets are converted.
- [x] ~~**dojo bridge: the official-client path needs robot-harness's `joint_steps` verb**~~: robot-harness PR #9 (J2 server + shim `harness_strict_j2` + `bin/robot-start-j2`); verified offline by `tests/dojo/test_bridge_rh.py`. The Isaac path runs first on our GPUs.
- [x] ~~**dojo bridge: robot-harness's launchers hard-code isr-lab paths**~~: the J2 starter / launcher read `HARNESS_ENV` (robot-harness PR #9); BRIDGE-RUNBOOK §2.
- [x] ~~**dojo bridge: spawn the twin after the runtime's vocabulary choice.**~~ The driver hands the policy a lazy
  `_twin_factory`; `RuntimePolicy.twin(scene)` starts the twin for the scene the episode chose (`bridge.json` →
  `twin_scene`, `twin_start_s`).
- [ ] **dojo runtime: fixture-aware place refine.** Under perception cb5c3a4 the place refines on the tower's base board
  are all "not ok" (the static fixture keeps its prior), and each place still spends 8-10 steps on 2 re-hovers (~50
  steps per 8-piece tower episode, journal 2026-10-25-dojo-integ2 §5). A stationary support whose prior perception
  kept + a not-ok place refine -> accept the prior, no re-hover.
- [ ] **dojo bridge: two instruction routers for perception.** `j1_adapter.infer_task` (the server's `perceive` route) knows 6 tasks; the runtime's `supervisor.perception_vocabulary` knows more. One function should serve both.
- [ ] **dojo runtime: the twin service is started per task by the launcher** (`--task`). A submission must start / load it from the instruction-matched scene; the policy already refuses to verify when the twin's task differs from that scene. Done for the official-client path: `dojo.bridge.twin` spawns it for the instruction's scene (`perception_vocabulary`); the proxy launcher still takes `--task`.
- [x] ~~**dojo bridge: the twin-layout template needs RoboDojo's generator deps in OUR venv**~~ (`tqdm`, `transforms3d`, imported by `utils.cluttered_generator`). Without them every twin verify of a bridge episode failed (`ModuleNotFoundError`, verify disabled). Now in the perception extra, with a CI guard (journal 2026-10-26-bridge-isr).
- [x] ~~**dojo bridge: twin requests carried numpy-2 objects into RoboDojo's numpy 1.26**~~: they failed to unpickle (`numpy._core`) and the service died. `TwinClient` sends plain lists and numbers now (`twin_service.plain`).
- [x] ~~**dojo bridge: `bridge_episode.sh` orphaned the perception service**~~: it killed only the subshell's bash. The orphan held fd 9, the session lock, and the next episode waited on it. The service is now exec'd, with fd 9 closed.
- [ ] **dojo runtime: no perception vocabulary for make_toast, make_toast_random, store_laptop_and_headphones, hang_mugs** (`supervisor.VOCABULARIES`), so they cannot run in mode perception, i.e. not on the strict official path at all. With no rule skill, push_T and cover_blocks end at once under RuntimePolicy (`no skill for`); they need the agent.
- [ ] **dojo twin: stack_bowls verify fails in the twin service**: `Pickling of "pxr.Usd.Stage" instances is not enabled` (isr-lab, ilbr-rt-bowls-s0L9-1). Probably a deepcopy of something holding the env during load / verify. It never showed in the proxy, where stack_bowls ran with verify off.
- [ ] **dojo perception: swap_blocks' target cube (Rigid/cube/00003) has geometry only in the catalogue** (no `visual_lo.glb` / silhouettes), so B1 finds no foreground and the episode ends at frame 0 (strict s0L9: "estimate lacks target0, target1").
- [ ] **isr-lab: Isaac's texture cache under our shim HOME grows ~0.1-0.4 GB per new task's assets** (1.6 -> 2.4 GB over 10 tasks); data2 has ~7-9 GB free. Never prune it (or the shader cache): the rebuild spiked the box to ~20 cores. The block gate stops below 7 GB free instead; free space must come from elsewhere (run dirs, or Mikhail's side).
- [x] ~~**dojo skills: the perception catalogue lacks the skills2 tasks' meshes**~~: it has them. The files are md5-identical to `skills2/catalog`, and G3 has read perception's catalogue all along. The skills3 tasks' new assets (coin, coin stand, piggy bank, stacking toy, toaster, bread, bread shelf) are in `compute3:~/dojo-data/skills3/catalog` (`DOJO_CATALOG`); perception should add them.
- [ ] **dojo skills: thin irregular tools still slip out** (store_tools). Two causes are fixed. First, every box grasp of a thin lying tool failed the table margin (the fingertip estimate sat 1 mm low; now taken from the finger model). Second, there were no grasps near the centre of mass (now stations every 6 mm). The hammer at its CoM lifts with 11-13 mm drift and is now carried on. Still open: the pliers' outline is rounded everywhere, so the patch is <= 0.08 at the joint. Try the handles' outer faces, or collider-based grasps; the colliders are convex decompositions of `/root/collision/model`, not the visual mesh.
- [ ] **dojo skills: planning time on irregular meshes**. Done: meshes are clustered under 12k faces, hold quality is cached per grasp in the object frame, and the relay search is capped at 150 s (one store_tools episode had spent 21 min). Open: IK per (arm, grasp) is recomputed for every relay leg.
- [ ] **dojo skills: insert_key's handover giver.** The key's round bow (a cylinder collider) lifts reliably but turns about the closing axis on the carry to the handover pose. The 5 mm shaft slips at the lift. Candidates: hand the key over with less turn (an intermediate orientation), or have the giver hold the bow while the taker takes the shaft's flat faces.
- [ ] **dojo skills: fill_pen_holder, the collider's opening is not the visual cup's** (parked at 0.08). The holder's collider is a convex decomposition of a 453-vertex `/World/collision` mesh. Pens centred within 3 mm of the axis jam about 6 mm into the opening and push the holder across the table, while the fit (visual mesh, 30 mm inner radius) says they clear. Fit against the collision mesh, or the decomposition's hulls, and re-run.
- [ ] **dojo skills: parked tasks** (lead, 2026-10-25). insert_key: lift + return banks 15; the in-air handover lost the key on the carry 0/31. store_tools: keep v2 (0.20); the pliers slip, and the hammer and wrench are lost on the carry. fill_pen: see above. Revisit after the Generalization tasks.
- [x] ~~**dojo: make_toast_random's slot checks cannot pass in our twin.**~~ Resolved: the official rendered kit writes physics to USD (live link poses), and the physics-only runner now reads link poses from physics by default (`usd_sync_links`, journal 2026-10-25-usd-sync-default). RoboDojo's `get_functional_points` returns toaster 00002/00004's frames untransformed (object frame, z ≈ -0.03). Those assets lack ArticulationRootAPI, and `Articulation.get_link_pose` reads the USD xform. Skills place physically; score rand with `scripts/skills/phys_slots.py`. Behaviour in the official env is unknown upstream. Evidence: `g3 checks` (journal 2026-10-25-dojo-skills4 §3).
- [ ] **dojo skills: store_laptop's laptop half (80) unexplained.** In our twin, the checker's `bottom` frame ends 78 mm above the table for any +y-up laptop, while the rack's bbox is 49 mm tall. The public demo's end pose is upright in the rack. Find what the official checker does differently before building the lid-close + stand skill. Headset half: `hang_on_hooks`.
- [ ] **dojo skills: sweep_blocks** (lead: a real broom sweep only, no pick-into-pan; low weight, after the others) and **fold_clothes** (parked: no cloth state in the skills layer).
- [ ] **dojo skills: deposit_coin.** The coin bank's `slot_opening` frames now give the target, but the coin (21 mm, 1 mm thick) starts out of the inserting arm's reach. The as-is relay lays it flat, and two grippers cannot share it in the air. It needs a relay that keeps it on its edge (back into a stand-like holder, or leaned against something).
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
