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

## P3 — next directions

- [x] ~~**RoboDojo:** wrap controlr as an XPolicyLab policy server~~ — done: `controlr robodojo-serve` + the shim, both arms ([ROBODOJO.md](ROBODOJO.md)).
- [ ] **RoboDojo runs:** after the `general_pickup` pilot, 6 tasks × 10 layouts (general_pickup, stack_blocks, stack_bowls, push_T, press_by_number, plug_in_charger), then a 5-dimension profile vs Astra's 22.48 % (that needs all 42 tasks × 3 seeds).
- [ ] **RoboDojo: faster motion per env step makes grasps slip** (0.12 rad + gripper 0.5: 0/3, 9 closes per episode; [report](experiments/2026-10-04-robodojo-fast-ab.md)). Separate arm vs gripper speed; fast only while the gripper is open.
- [ ] **Sonnet `reasoning_effort: low` writes its reasoning as prose in the reply** (51 vs 18 words median on RoboDojo) despite the manual's "no prose" — 2.2× faster turns anyway.
- [ ] **RoboDojo: the env-step budget binds** (pilot: every failure ran out of `general_pickup`'s 200 steps during re-grasps; [report](experiments/2026-10-04-robodojo-pilot.md)). Levers: `max_chunk > 1`, coarser approach moves, `reasoning_effort: low` for latency.
- [ ] **RoboDojo: the envelope knows neither the objects nor the other arm**; cuRobo plans each arm against the table and itself only. Arm-arm collisions are possible (sim only; never on hardware like this).
- [ ] **Real UR3 backend** via PHANTOM drivers + `SafetyMonitor` (RTDE, Robotiq, RealSense).
- [x] ~~**Which UR3 is it — CB3 or e-Series?**~~ — **CB3** (Ilia, 2026-10-03). No built-in wrist F/T: contact sensing on hardware = `getActualTCPForce` (estimated from joint currents, coarse) + UR protective stops, unless a Robotiq FT-300S is added. Note: PHANTOM's `configs/hardware.yaml` still says `generation: e-series` (500 Hz rates, `wrist_ft.source: ur_internal`) — PHANTOM's to fix, not ours.
- [ ] **Ideas from Waddle Labs** ([research/sources/waddle-labs.md](../research/sources/waddle-labs.md)): a recovery rule (when already out of bounds, accept a command that reduces one violation without worsening others — generalises the straight-elbow fix); tag every executed action by source (model / safety clamp / human) and log holds and e-stops (needed for the real UR3); success-vs-cost curves across effort levels with n and CIs; a cross-episode "lessons" appendix in the cached prefix as an ablation; their 32 MuJoCo task environments (Apache-2.0) as a task catalogue.
- [ ] **Export runs to LeRobotDataset** — data for a future light action head.
- [ ] **Action head** (transformer/diffusion) closing the ~1 s between model turns.

## compute3 infrastructure

- [x] ~~**Slow VPN on compute3**~~ — Happ/AmneziaVPN/OpenVPN removed, machine-wide `controlr-vpn` (sing-box tun, router direct: 740 KB in 0.43–0.48 s) (journal/2026-10-04-compute3-network.md). Remaining: ethernet on `eno1` (Ilia).
- [ ] The meta package `linux-modules-nvidia-595-open-generic-hwe-24.04` lags the kernel → the next kernel update will drop the GPU again ([incident](incidents/2026-10-02-compute3-nvidia-driver.md)). Upgrade it together with the kernel (agree with the machine's owner).
- [ ] Delete the old `~/controlr-dev-isaac`, `~/controlr-dev-loopcli` (≈100 MB; their `.env` copies are already deleted).
