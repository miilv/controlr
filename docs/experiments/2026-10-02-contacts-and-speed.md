# 2026-10-02/03 — contacts, short feedback and Isaac speed: does the pick-to-box get more robust, and how fast can the sim run?

**Hypothesis.** Most rotation-round failures came from the box, not from the grasp: an immovable
box blew PhysX up, the held-packet stop fired on the grip's own force, and the model had no idea
where the walls were. A light dynamic box, a stop on packet-vs-environment force and contact
reported as a short STOP should turn those failures into recoverable events. Separately, the sim
should run at ≥ 1× real time.

**Setup.** `configs/sim_waffle_yaw.yaml` with the new defaults (journal
[2026-10-03-contacts-and-speed](../journal/2026-10-03-contacts-and-speed.md)): dynamic box 0.4 kg,
`safety.box_collision: warn`, box stop 30 N + push stop, held-packet stop on packet-vs-environment
force, `feedback.level: short` (TASK / STATE without `holding` / WARN / STOP), `observation.tactile:
false`, speed defaults below. Control model `claude/claude-sonnet-5-5` only (Ilia). Seeds 0–3, the
rotation round's pinned plans (`planner.plan_file=runs/rot_plans/seed<k>.md`, identical to the
plans the Sonnet 5.5 rotation episodes used), so the comparison is control-side. Code: commit
d3a4bd0 + the speed-default fix in this PR. compute3, RTX 5090, CPU governor `powersave`, other
users' jobs running (load 8–13).

**Budget and spend.** Budget ≤ 150 control calls, ≤ 4 planner calls. Spent: **139 control
calls** (no retries), **0 planner calls**. Tokens: 1.34 M prompt (1.22 M cache reads, 91 %), 28.3 k
completion (23.2 k of it reasoning). GPU time ≈ 2.5 h (profiling, Isaac tests, live round). The
Isaac server was stopped after every step.

## 1. Speed (scripted, no LLM: `scripts/isaac_profile.py`, `scripts/isaac_profile_round.sh`)

Scenario `expert` = the closed-loop scripted yaw expert (`tasks.run_scripted_yaw_pick_place`)
through the yaw envelope on two scenes (packet −15° / start 0°, packet +15° / start −20°), 47
actions, one render per action — the moves a model makes, ~58 s of sim time. Ratio = sim time /
wall time of execute (render included in the second number). Wall-time split from the server's
per-execute `profile`.

| setting (cumulative where noted) | ratio exec / +render | drive targets s | PhysX s | contact reads s | expert |
|---|---|---|---|---|---|
| PHANTOM's settings (64 it, `World.step` per 1 ms, PHANTOM per-body contact views, USD write-back) | **0.18** / 0.18 | 8.0 | 240.7 | 67.5 | (run before the containment fix, see §4) |
| + one contact matrix, PhysX stepped directly (`reader: matrix`, `direct`) | 0.24 / 0.24 | 3.6 | 230.0 | 8.8 | 2/2 |
| + no USD write-back per step (`usd_writeback: false`; transforms written before each render) | 0.68 / 0.66 | 2.8 | 72.2 | 8.4 | 2/2 |
| + PHANTOM's per-body views not built (`legacy_contact_views: false`) — **chosen default** | ≈ 0.70 (0.67 expert, 0.73–0.79 low grasps / replays) | | | | 2/2, low grasps 3/3 held |
| + 5 ms target ticks (`substeps: 5`, predictive stop) | 0.79 / 0.77 | 0.6 | 68.8 | 2.5 | 2/2 — **but 2 of 4 live grasps blew up**, §3 |
| + 10 ms ticks | 0.81 / 0.79 | 0.3 | 67.8 | 1.7 | 2/2 |
| 32 solver iterations, 5 ms ticks | 0.97 / 0.94 | 0.7 | 56.5 | 2.8 | 2/2; a low grasp lost the packet (also at 1 ms ticks) |
| 32 it + velocity iterations 4 | 0.97 / 0.94 | | | | 2/2 |
| 32 it + self-collisions off | 1.06 / **1.02** | | | | 2/2 (not taken: see below) |
| 32 it + 4 PhysX threads | 0.94 / 0.91 | | | | 2/2 |
| 24 iterations | 1.01 / 0.98 | | | | 3/4 |
| 16 iterations | 1.14 / 1.04 | | | | 1/2 |
| 48 iterations | 0.84 / 0.81 | | | | 2/2 |
| forearm `convexHull` instead of PHANTOM's convex decomposition (32 / 64 it) | 0.92 / 0.77 | | | | 0/2 (climb blocked) |
| dt 2 ms, 32 it (PHANTOM's validator rejects > 1 ms for the W2L gripper) | 2.17 / 1.61 | | | | 0/2 (packet not lifted) |

What costs time: with PHANTOM's settings 76 % of the wall time is PhysX's step **including USD
write-back of every body transform** (disabling it alone took PhysX from 230 s to 72 s), 21 % is
reading ~30 per-body contact views every step, 3 % setting drive targets through the
`isaacsim.core` wrappers. Rendering is negligible (0.04 s per observe). GPU PhysX is not an option:
GPU contact filters on collider prims (the box parts) are unsupported and one articulation does not
profit (read-only source check; not measured). `World.step` already never renders; Kit's
substepping only happens inside `app.update()` (which renders), so "several substeps per Python
call" = holding the drive target for k physics steps and calling `simulate` k times — the
`substeps` setting. PhysX drives have no target ramping, so per-waypoint targets would jump.

Settling is ~45 % of sim time in the expert (25 of 58 s): the arm settles up to 2 s per action
(BACKLOG).

**Chosen default** (`controlr/robot/isaac/client.py` `DEFAULTS`): PHANTOM's physics (1 ms, 64 / 8
iterations, self-collisions on), no per-step USD write-back, no per-body diagnostic views, one
contact matrix, PhysX stepped directly, drive target + contact sample every 1 ms. **≈ 0.7× real
time (0.65–0.71 in the live episodes), up from 0.18×; ≥ 1× was NOT reached with stable physics.**
Real time is only reached by settings that each failed a check: 32 iterations (a low grasp dropped
the packet), 5 ms ticks (live grasp blow-ups), 24/16 iterations (expert failures), dt 2 ms (no
grasp) or self-collisions off (1.02×, expert fine, but then the gripper can pass through the
forearm and nothing would notice: the force stop does not see self-contact either).

Frames: with and without USD write-back the rendered D435 frames after a fixed sequence differ by
0.55/255 on average (RTX noise level) — the camera sees the same scene.

**Stop overshoot** (`mat_push`: the fingers driven 30 mm into the mat at 0.15 m/s, no envelope):
the first stop is mostly robot-vs-packet (the packet is near), the mat stop peaks at 85–350 N
against 80 N at every setting, including 1 ms sampling (351 N). The overshoot is the impact spike
of a stiff drive hitting a rigid surface, not sampling: the "< 1.5× threshold" goal holds only for
some settings by chance (BACKLOG: slow down near contact). The box is different: the 0.4 kg box
slides at 2–4 N, so a 160 mm push never crossed the 30 N box limit — hence the push stop (touching
the box while it moved > 5 mm), which stopped every box push at 5–6 mm of box travel.

## 2. Contacts (scripted, Isaac tests on compute3)

- `test_dynamic_box_slides_when_pushed_stops_the_arm_and_reset_restores_it`: gripper driven into the
  near wall → STOP "the gripper pushed the blue box (it moved)", box moved a few mm, no `unstable`,
  reset restores the box within 2 mm.
- `test_held_packet_free_air_never_stops_but_a_wall_jam_does`: grasp, then lift + ±25° yaw turns +
  climb + diagonal carry in free air → no STOP (the rotation round stopped at 85–89 N grip force);
  lowering the held packet onto the near wall → STOP "the held packet pushed against the box wall
  (the box moved)".
- Full suite: `CONTROLR_ISAAC=1 pytest tests/` 408 passed, 4 skipped (live), **6 min** (was ~19 min).

## 3. Live round (Sonnet 5.5, short feedback, seeds 0–3, pinned plans)

Two passes: the first ran with the faster speed candidate (32 iterations, 5 ms ticks); two of its
four grasps blew up the physics at `GRIP close`. Replaying the logged actions on a fresh server
reproduced the blow-up with 5 ms ticks at both 32 and 64 iterations and NOT with 1 ms ticks
(legacy or new path: an ordinary 40 N object stop instead). The default was changed to 1 ms ticks /
64 iterations and all four seeds re-run (seeds 1 and 2 at `max_turns 26` to stay within budget —
both finished before that).

| seed | rotation round (Sonnet 5.5, full feedback, static box) | this round, final settings | turns | STOPs (what) | box moved max | LLM p50 s | exec s/turn (mean) | turn cycle p50 s | cache share |
|---|---|---|---|---|---|---|---|---|---|
| 0 | max_turns (straight-elbow trap) | **success** | 23 | 1: held packet vs near wall | 5.6 mm | 3.4 | 2.2 | 6.0 | 0.94 |
| 1 | safety_stop (held packet jammed on wall ×3) | **success** | 22 | 1: held packet vs near wall | 5.5 mm | 2.9 | 2.3 | 6.3 | 0.94 |
| 2 | safety_stop (627 N at lift; wall jam) | **success** | 20 | 0 | 1.4 mm | 2.2 | 3.0 | 6.2 | 0.94 |
| 3 | success (24 turns) | **success** | 18 | 2: held packet vs near wall; arm pushed the box while lowering | 10.5 mm | 2.9 | 2.0 | 5.5 | 0.93 |

First pass (rejected speed setting, same plans): seed 0 `unstable` at the first close (turn 7),
seed 1 safety_stop (three "the gripper pushed against the packet" STOPs while releasing — the 32-
iteration grip let one pad unload, so the held rule no longer applied), seed 2 success (22 turns,
one held-packet wall STOP), seed 3 `unstable` at the first close (turn 8). Runs:
`runs/20261003T0{05821,05934,10155,10358}Z_sim_waffle_yaw` (first pass),
`runs/20261003T01{2436,2711,2917,3137}Z_sim_waffle_yaw` (final).

**4/4 vs 1/4 in the rotation round — n = 4, a description, not a finding**; the round also changed
several things at once (dynamic box, stops, feedback, speed), so it cannot say which mattered.

**How the model used contact.** In all four final episodes the carry over the near wall was the
critical moment, as in the rotation round. Three episodes touched the wall with the held packet;
each time the single line `STOP: the held packet pushed against the box wall (the box moved)` was
enough, and the model's next two replies were identical in shape:

> `MOVE ee_delta 0 60 20 0` / `STATUS OK carrying packet toward box, staying high` →
> `STOP: the held packet pushed against the box wall (the box moved)` →
> `MOVE ee_delta 0 -15 50 0` / `STATUS OK backing off wall and lifting higher to clear rim` →
> `MOVE ee_delta 0 40 10 0` / `STATUS OK carrying packet over the near wall, higher now` (seed 0)

In the rotation round the same situation was a jam at 73–222 N repeated three times (Sonnet seed 1)
or a PhysX blow-up (Opus seeds 1/3). With a light box the touch moves the box 5–10 mm and stops the
arm at a few newtons, so it is information, not damage. In seed 3 the arm brushed the box while
lowering into it ("STOP: the arm pushed the blue box (it moved)"); the model released right there
and the goal check, which follows the moved box, passed. The predictive `warn` fired once (seed 3,
turn 4, while descending to the grasp next to the near wall); the model continued and nothing
touched. No episode needed `holding` or force numbers; the model checked grasps by lifting and
looking ("lifting to check the grasp").

**Time per turn.** Execution 2.0–3.0 s per turn (mean) vs 5.6–11.3 s in the rotation round; full
turn cycle p50 5.5–6.3 s vs 9.7–17.9 s. The LLM is now about half of a turn (p50 2.2–3.4 s; the
rest is execution and settling). Sonnet's reasoning: 2.3–4.6 k tokens per episode.

**Cache.** 0.93–0.94 read share in the final episodes (0.66–0.91 in the first pass; one cache
regression at seed 0's first-pass turn 3).

## 4. Found along the way

- **Containment in a dynamic box.** Resting partly on the 3 mm mat the box tips 0.65°, and the old
  yaw-only interior test put a packet standing on the floor 0.6 mm "below" it: the scripted expert
  "failed" 2/2. Now tested in the box's own frame (`tasks.to_box_interior`). The baseline row of
  §1 ran before this fix.
- **The held packet's contacts jitter with the grip** (22–34 N against the mat while closing): the
  held-packet rule is suspended while the gripper closes, re-baselined after each grasp, and never
  extrapolated.
- **Force-stop overshoot is an impact spike** (85–350 N against 80 N at any sampling rate) → BACKLOG.
- **Release can trip the object stop** when the grip is weak (first pass, seed 1) → BACKLOG.

## 5. Conclusions (n = 4)

- The dynamic box + packet-vs-environment stop + one-line STOP turned the near-wall carry from the
  main failure into a recoverable event; all three wall touches in the final round were recovered
  with the same back-off-and-lift reply. No `unstable` episode at the final settings.
- The short feedback (TASK / STATE / WARN / STOP) did not visibly hurt: 4/4, no parse errors, the
  model never asked for `holding`.
- Speed: 0.18× → ≈ 0.7× with stable physics; turns are ~2× shorter. ≥ 1× needs either a weaker
  solver / coarser control (which broke grasps here) or dropping self-collisions.

## 6. Next

- The same round with `feedback.level: full` on the same seeds (budget did not allow the optional
  2-seed arm) and with `box_dynamic: false` — one variable at a time.
- More seeds (n ≥ 20) before calling 4/4 anything.
- Speed: shorter settling (45 % of sim time), contact reports only during motion, then re-test
  32 iterations with a stronger grip latch.

Videos (`docs/video/`): `contacts_sonnet55_seed3_success.mp4` (wall touch, back off, release
after brushing the box), `contacts_sonnet55_seed0_success.mp4`,
`contacts_sonnet55_seed3_fast_unstable.mp4` (the rejected 5 ms setting: the grasp blows up).
