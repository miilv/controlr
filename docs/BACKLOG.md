# Backlog

> The task queue. Take an item → branch → PR linking the item → strike it here in the same PR
> (`- [x] ~~…~~ (journal/… or experiments/…)`). Found a landmine — add it here.
> Sources: [experiments/](experiments/), [reviews/](reviews/), [journal/](journal/).

## P1 — before the next round (carry failures and perception)

- [ ] **Cameras (hypothesis: the main failure cause is a single angled D435).** Add a virtual
  wrist camera and a second fixed one (top/side) to the Isaac scene; experiment 1 camera vs
  + wrist vs + top (tiled and separate). The renderer and config already support it
  (`observation.cameras`, `observation.tile`); the server needs the cameras + `CameraInfo`.
  The real rig would need a physical wrist camera — decide after sim.
- [ ] **Box geometry and carry corridor in the manual/task:** near wall at y ≈ −85 mm, rim at
  z = 180 mm, the packet hangs ≈30 mm below the TCP; rise while moving toward +y; cross the wall
  at z ≥ 260 (5 of 8 failures in the rotation round). Re-run seeds 0–3 with the same plans
  ([rotation-yaw](experiments/2026-10-02-rotation-yaw.md) §6.1).
- [ ] **Arm links vs box in `SafetyEnvelope`** (wrist/gripper housing against the box body,
  ≈50 mm inflation, like `tasks.body_box_clearance`) — block before execution instead of
  catching PhysX blow-ups.
- [ ] **Object-force stop fires falsely** on the gripper's own grip force (85–89 N in free air
  vs real jams of 47–222 N). Needs a "packet vs environment" contact view in the server and a
  stop based on it, not on pad force.
- [ ] **Joint-space escape/home**: the arm gets stuck at the edge of reach and almost every move
  is refused (Sonnet seed 0 — 17 turns wasted).
- [ ] **Feedback noise:** a repeated "touched the packet 14 N" while holding, "not settled"
  warnings → wasted HOLDs.
- [ ] **`grid` / `ee_marker` overlays in a live run** — implemented and projecting correctly,
  never tested live; 80–200 mm depth error is the main source of mistakes.
- [ ] **Reach target is ambiguous in depth:** a 12 mm ball on an invisible 1.5 mm pole that
  projects onto the packet; every planner put it on the mat (0/4 reach in
  [smoke-v0](experiments/2026-10-02-smoke-v0.md) §3.1). Give it a visible base/pole and state its
  height in the instruction; check that `setup.json` now records the marker position.
- [ ] **Grasp yaw window −40…+30°** at the current tilt: a `yaw_min/max` clip param or `rotation=full`.
- [ ] **The planner should always write "packet heading h, grasp yaw = h ± 90"** — when it does, models hit it exactly.
- [ ] **`no-think/claude/claude-sonnet-5-5`** in an episode: Sonnet 5.5 spends 3–5× more reasoning than Opus for the same behaviour.

## P2 — harness and infrastructure

- [ ] **Cache probe through the router (~6 calls):** do `llm.cache` / `cache_ttl` matter at all on `claude/` / `cc/` (the router seems to set the boundary) — don't sweep these axes until then.
- [ ] **Pin one upstream account** (a lease/session header in `llm.extra_headers`) — cache drops on turns 4, 5, 7 in one run.
- [ ] **Haiku 4.5 doesn't cache the first turns** (manual < 4096 tokens) — a meaningful appendix (kinematics/examples), not filler.
- [ ] **`action.format=tool`** (actions as tool calls instead of the text grammar) — an experiment axis, not implemented.
- [ ] **Continuous video recording in Isaac** (`log.record_video`); today videos are per-turn frames only.
- [ ] **Isaac runs at ~0.24× real time** — wire up PHANTOM's faster scenes (4 ms sliding-pad) or speed up execution.
- [ ] **Usage grace is not a real deadline** (deferred in review, L8): up to 0–0.2 s per turn.
- [ ] **Arm self-collisions** aren't detected by the force stop (S5, partial).
- [ ] **Mock: table_z 0.053 vs Isaac −0.0095** — unify in one spec (mock only, harmless).
- [ ] **Archive of `runs/`** (722 MB locally, not in git): where to keep runs that reports reference (HF bucket/dataset or compute3).
- [ ] **CI for Isaac tests** — a self-hosted runner on compute3 or a nightly run; manual for now.
- [ ] **Skill `.claude/skills/run-experiment`**: the EXPERIMENTS.md protocol as a skill (config → budget → run → report → videos → index).

## P3 — next directions

- [ ] **RoboDojo:** wrap controlr as an XPolicyLab policy server; a reduced run (10 episodes/task, 1 seed, ≈$300 on Sonnet with caching working) for a 5-dimension profile vs Astra's 22.48 %. Isaac 5.1 is on compute2.
- [ ] **Real UR3 backend** via PHANTOM drivers + `SafetyMonitor` (RTDE, Robotiq, RealSense).
- [ ] **Export runs to LeRobotDataset** — data for a future light action head.
- [ ] **Action head** (transformer/diffusion) closing the ~1 s between model turns.

## compute3 infrastructure

- [ ] The meta package `linux-modules-nvidia-595-open-generic-hwe-24.04` lags the kernel → the next kernel update will drop the GPU again ([incident](incidents/2026-10-02-compute3-nvidia-driver.md)). Upgrade it together with the kernel (agree with the machine's owner).
- [ ] Delete the old `~/controlr-dev-isaac`, `~/controlr-dev-loopcli` (≈100 MB; their `.env` copies are already deleted).
