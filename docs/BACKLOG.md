# Backlog

> The task queue. Take an item → branch → PR linking the item → strike it here in the same PR
> (`- [x] ~~…~~ (journal/… or experiments/…)`). Found a landmine — add it here.
> Sources: [experiments/](experiments/), [reviews/](reviews/), [journal/](journal/).

## P1 — before the next round (carry failures and perception)

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

## P2 — harness and infrastructure

- [ ] **Cache probe through the router (~6 calls):** do `llm.cache` / `cache_ttl` matter at all on `claude/` / `cc/` (the router seems to set the boundary) — don't sweep these axes until then.
- [ ] **Pin one upstream account** (a lease/session header in `llm.extra_headers`) — cache drops on turns 4, 5, 7 in one run.
- [ ] **Haiku 4.5 doesn't cache the first turns** (manual < 4096 tokens) — a meaningful appendix (kinematics/examples), not filler.
- [ ] **`action.format=tool`** (actions as tool calls instead of the text grammar) — an experiment axis, not implemented.
- [ ] **Continuous video recording in Isaac** (`log.record_video`); today videos are per-turn frames only.
- [x] ~~**Isaac runs at ~0.24× real time**~~ — ~0.7× with the new defaults (journal/2026-10-03-contacts-and-speed.md, experiments/2026-10-02-contacts-and-speed.md)
- [ ] **Isaac ≥ 1× real time:** settling is ~45 % of sim time (fixed thresholds, ≤ 2 s); self-collisions off gives 1.02× (not default: the force stop does not see self-contact either); 24 iterations failed 1 of 4 expert scenes, 16 failed 1/2, dt 2 ms lost the grasp. Next: settle criteria per move, contact-report only during motion.
- [ ] **5 ms drive-target ticks blow up some grasps** (2 of 4 live closes near the mat; replay-confirmed, 1 ms is fine) although the scripted expert passes — find why before using `substeps > 1`.
- [ ] **Force-stop overshoot is an impact spike, not sampling:** driving the fingers into the mat at 0.15 m/s peaks at 85–350 N against an 80 N limit at every sampling rate (1 ms included). Slow the last ~20 mm near contact, or a compliant wrist.
- [ ] **Usage grace is not a real deadline** (deferred in review, L8): up to 0–0.2 s per turn.
- [ ] **Arm self-collisions** aren't detected by the force stop (S5, partial).
- [ ] **Mock: table_z 0.053 vs Isaac −0.0095** — unify in one spec (mock only, harmless).
- [ ] **Archive of `runs/`** (722 MB locally, not in git): where to keep runs that reports reference (HF bucket/dataset or compute3).
- [ ] **CI for Isaac tests** — a self-hosted runner on compute3 or a nightly run; manual for now.
- [ ] **Skill `.claude/skills/run-experiment`**: the EXPERIMENTS.md protocol as a skill (config → budget → run → report → videos → index).

## P3 — next directions

- [ ] **RoboDojo:** wrap controlr as an XPolicyLab policy server; a reduced run (10 episodes/task, 1 seed, ≈$300 on Sonnet with caching working) for a 5-dimension profile vs Astra's 22.48 %. Isaac 5.1 is on compute2.
- [ ] **Real UR3 backend** via PHANTOM drivers + `SafetyMonitor` (RTDE, Robotiq, RealSense).
- [x] ~~**Which UR3 is it — CB3 or e-Series?**~~ — **CB3** (Ilia, 2026-10-03). No built-in wrist F/T: contact sensing on hardware = `getActualTCPForce` (estimated from joint currents, coarse) + UR protective stops, unless a Robotiq FT-300S is added. Note: PHANTOM's `configs/hardware.yaml` still says `generation: e-series` (500 Hz rates, `wrist_ft.source: ur_internal`) — PHANTOM's to fix, not ours.
- [ ] **Ideas from Waddle Labs** ([research/sources/waddle-labs.md](../research/sources/waddle-labs.md)): a recovery rule (when already out of bounds, accept a command that reduces one violation without worsening others — generalises the straight-elbow fix); tag every executed action by source (model / safety clamp / human) and log holds and e-stops (needed for the real UR3); success-vs-cost curves across effort levels with n and CIs; a cross-episode "lessons" appendix in the cached prefix as an ablation; their 32 MuJoCo task environments (Apache-2.0) as a task catalogue.
- [ ] **Export runs to LeRobotDataset** — data for a future light action head.
- [ ] **Action head** (transformer/diffusion) closing the ~1 s between model turns.

## compute3 infrastructure

- [ ] The meta package `linux-modules-nvidia-595-open-generic-hwe-24.04` lags the kernel → the next kernel update will drop the GPU again ([incident](incidents/2026-10-02-compute3-nvidia-driver.md)). Upgrade it together with the kernel (agree with the machine's owner).
- [ ] Delete the old `~/controlr-dev-isaac`, `~/controlr-dev-loopcli` (≈100 MB; their `.env` copies are already deleted).
