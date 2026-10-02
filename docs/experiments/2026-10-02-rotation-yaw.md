# Gripper rotation (`action.rotation: yaw`) — report (2026-10-02)

The model now commands the tool heading as well as the position: `MOVE ee_delta dx dy dz dyaw`.
The tilt stays fixed at the demo tilt (fingers about 35° below horizontal). The waffle task now
turns the packet up to ±40° and the start heading up to ±20°, so the jaws must be turned across
the packet before the grasp. Config: `configs/sim_waffle_yaw.yaml`.

**Model change:** Sonnet 5.5 (`claude/claude-sonnet-5-5`) replaced Sonnet 5 as the Sonnet control
model, at Ilia's request. It is also the new default `llm.model` in `configs/base.yaml` and
`controlr/config.py`. Its prompt caching shows up in usage from the first episode on: the cache
read share was 0.90 per episode, with reads from turn 1 on.

## 1. What changed

| area | change |
|---|---|
| logging | `setup.json` now has `scene` (`Robot.scene_record()`; Isaac: `tasks.scene_record`, returned by the server's `reset`). It records the packet position (sampled and settled), yaw, yaw offset from nominal, tilt, size and settle drift; the box centre, yaw and interior; the start joints, start TCP, start tool yaw and start yaw offset; and the marker / zone. Units are mm and deg. The mock robot records its target / cube too. Tests: `test_isaac_tasks::test_scene_record_*`, `test_loop::test_setup_records_the_sampled_scene`, `test_isaac_client::test_scene_record_*`. |
| task params | New waffle / reach / push params: `start_yaw_deg` (v → U(−v, v), [lo, hi] → U(lo, hi); default 0). `tasks.rotate_start_q` follows the start TCP in 2° IK steps on the recorded branch. New `yaw_dist: normal\|uniform` (default normal = PHANTOM) and `yaw_offset_deg` (a fixed packet yaw). All new draws come after the packet draws, so with defaults every seed gets the same packet pose as before (tested). |
| envelope | `SafetyEnvelope(..., rotation=cfg.action.rotation)` is set by the loop and by the mock / Isaac fallback envelopes. With rotation=yaw only the reference roll and pitch are held. The heading is the current yaw + dyaw, with dyaw clamped to 30°/line with a CLAMP line. Before this, `Rz(dyaw) @ R_measured` locked any contact tilt in; now a contact tilt is undone ("yaw is kept"). |
| envelope messages | A shortened move that turns now reports the turned angle ("moved 87 % of the way (0 mm, turned 26.2 deg of 30.0 deg)"). It also names the kinematic reason: "the arm is stretched to the edge of its reach: elbow almost straight; come closer to the robot base or lower", or "wrist singularity". |
| envelope bug (live) | **Straight-elbow trap.** In Sonnet seed 0 (run 184337) a reach-clamped lift parked the arm at elbow = 0.0°. From that pose every move, inward ones included, needs a large first-step joint change. The swing guard refused them all for 17 turns, and the episode ran out of turns. Fix: a move never ends with the elbow straighter than 8°. A move that bends the elbow out of the stretch, on the episode's branch, may make a large first step. Regression tests use the logged joints. This was deployed after Sonnet seed 0; every later episode had it. |
| Isaac server / client | Two fixes. (1) PhysX NaN / zero-quaternion transforms crashed the server in `_tcp_measured` (Haiku seed 1 ended as `error`). They now return NaN and STOP as `unstable`. (2) When an ordinary stop was followed by a solver blow-up (contact peaks up to 1e17 N, arm teleported), the run continued (Opus seed 1, Haiku seed 1). The client now turns absurd or non-finite peaks into an `unstable` STOP, which ends the episode. Also, an object STOP while holding now says "the held packet (pushed against an obstacle such as the box wall, or swung hard …)". These fixes were deployed after the live round. |
| manual | With rotation=yaw the manual used to say nothing about the tool. It now has a yaw tool paragraph (see §4). The STATE example and Appendix B use the rig's real heading, and Appendix B turns the jaws +15° in its first move. The worked example includes a pure turn, `MOVE ee_delta 0 0 0 20`. The grammar line says the turn pivots about the TCP and keeps the tilt. Appendix A explains heading wrap-around. The workspace z bound with the fingertip drop also holds for yaw, because the drop depends only on the tilt. |
| tools | `scripts/turn_video.py`: per turn, the frame the model saw, the STATE it saw, its reply, the LLM latency (`turns.jsonl` `llm.t_end`, with `timings.llm_wall` as fallback; the old script read `timings.llm` and printed nothing), first-token time, cache share and thinking tokens, then the feedback, and a final end-state card. `tasks.run_scripted_yaw_pick_place` is a closed-loop expert in the model's own action space. `tasks.body_box_clearance` measures how close the wrist and housing come to the box. |

## 2. Scripted Isaac check (no LLM; `tests/test_isaac_sim.py`, compute3)

- **Yaw tracking** (`test_yaw_delta_moves_track_on_the_tilted_tool`): 8 ee_delta moves through
  the yaw envelope from the sim_waffle start, covering pure turns of ±15–40° (40 is clamped to 30)
  and mixed moves with up to 60 mm of travel. Worst errors: **yaw 0.007°, TCP 0.19 mm, roll/pitch
  0.04°**. The limits were 1°, 2 mm and 1°.
- **Packet at ±45°** (`test_packet_stands_still_at_large_yaw`): the packet stands upright (tilt
  < 1°, drift < 3 mm) and settles at the commanded yaw.
- **Scripted expert** (`test_scripted_yaw_expert_succeeds`, all passed). It hovers 20 mm above the
  target-heading pregrasp, turns the jaws across the packet, takes the pregrasp 80 mm back along
  the tool axis, approaches, closes, lifts 30 mm, turns back to the demo heading while still low,
  then lifts, climbs and carries to the box, lowers, releases and retreats.

| packet yaw offset | start yaw offset | actions | result |
|---|---|---|---|
| −40° | +20° | 26 | success |
| −15° | 0° | 23 | success |
| +15° | −20° | 24 | success |
| +30° | 0° | 23 | success |

**+40° does not work, and rotation alone cannot fix it.** I tested it and it failed. With the
fixed demo tilt, jaws turned +40° put the gripper housing within about 46 mm of the box's near
wall at the grasp. In the sim, 47 mm already meant contact: two earlier expert versions touched the
box at 93 N and 257 N. Moving the grasp 50 mm along the packet away from the box runs out of arm
reach. The CPU twin (`test_isaac_tasks::test_scripted_yaw_expert_is_feasible_through_the_envelope`,
3 start yaws each) shows:

- graspable: −40 to +30° at the nominal packet position, and +35° when the packet lies 20 mm
  toward −y;
- not graspable: +40°, and +25° when the packet lies 20 mm toward the box
  (`test_large_positive_packet_yaw_is_boxed_in`);
- at −45° the pregrasp is out of reach (elbow straight);
- at the high start pose a negative turn runs out of reach after about 27°, so start_yaw ±20 is
  safe and larger negative turns must be made lower down.

So "±40° graspable" is not true for this rig. The graspable window is about −40…+30°. The seeds
used live (−36.7, −28.5, +25.1 and +24.1 at dy = −5 / −20 mm) all fall inside it.

## 3. Prompt (rendered with `controlr prompt -c configs/sim_waffle_yaw.yaml --setup <run>`)

This is the yaw paragraph the model reads. The numbers come from the reference pose and the
calibration:

> Tool orientation: you command the heading (yaw) only; the tilt is fixed. `yaw` is the direction
> of the jaw line — the line along which the two fingertips close — in the base x-y plane,
> measured from +x toward +y … The fingers point 35 deg below horizontal, toward heading yaw + 104
> deg … A dyaw turn pivots the whole gripper about the vertical line through the TCP … To grasp an
> object, set yaw = (heading of the object's long side) ± 90 deg … In the `scene` image … a line at
> heading h on the table appears at about h counter-clockwise from the image's rightward direction
> (within 3 deg here) … a positive dyaw turns the jaw line counter-clockwise in the image … Large
> turns are not reachable everywhere … Turn the jaws close to working height … and turn back toward
> the start heading before long carries.

The image-angle rule is computed from the calibration, not hard-coded. It is printed only if the
error stays under 10° for every heading (here it is under 3°). The paragraph does not change with
the start yaw (tested), but the joint table's "home" column and the example STATE yaw do.

## 4. Live round (compute3, Isaac Sim 6.0, `configs/sim_waffle_yaw.yaml`, max_turns 30)

Planner (opus-5-5-xhigh) calls: one per seed during the Sonnet runs, 57–79 s each. Opus and Haiku
ran on the same plan text through `planner.plan_file`, so differences come from the control
model. Sonnet seed 0 ran before the straight-elbow fix; all other episodes ran with it. The
server-side fixes in §1 came after the round.

| seed | packet yaw offset | start yaw offset | model | outcome | turns | LLM p50 s | cache share | jaw turn / jaw error at 1st close |
|---|---|---|---|---|---|---|---|---|
| 0 | −36.7 | −19.3 | sonnet-5-5 | max_turns (straight-elbow trap) | 30 | 5.7 | 0.90 | −7 / +12° |
| 1 | −28.5 | +17.9 | sonnet-5-5 | safety_stop (held packet jammed on box wall ×3) | 15 | 3.7 | 0.79 | −42 / +6° |
| 2 | +25.1 | −16.3 | sonnet-5-5 | safety_stop (627 N at lift; jam on wall) | 30 | 6.7 | 0.89 | +40 / −0° |
| 3 | +24.1 | +3.3 | sonnet-5-5 | **success** | 24 | 3.7 | 0.91 | +14 / −6° |
| 0 | −36.7 | −19.3 | opus-5-5 | **success** | 18 | 9.6 | 0.81 | −5 / +14° |
| 1 | −28.5 | +17.9 | opus-5-5 | safety_stop (solver blow-up when the held packet hit the box) | 23 | 7.1 | 0.88 | −42 / +6° |
| 2 | +25.1 | −16.3 | opus-5-5 | safety_stop (85–89 N held-packet spikes; then wall jam) | 19 | 4.0 | 0.89 | +40 / −0° |
| 3 | +24.1 | +3.3 | opus-5-5 | safety_stop (`unstable`: arm hit box at 136 kN while reaching over it) | 18 | 3.4 | 0.88 | +17 / −2° |
| 0 | −36.7 | −19.3 | haiku-4.5 (no-think) | safety_stop (released at the near wall, box contacts while regrasping) | 25 | 1.7 | 0.87 | −7 / +12° |
| 1 | −28.5 | +17.9 | haiku-4.5 (no-think) | error (PhysX NaN after hitting the box; crashed the server, now fixed) | 16 | 1.5 | 0.87 | −30 / +18° |

"Jaw error" is the STATE yaw at the first `GRIP close` minus (packet yaw + 90°), taken modulo
180°. Run dirs: `runs/20261002T18{4337,5710}Z`, `19{0212,1222,1825,2358,3020,3513,3856,4251}Z_sim_waffle_yaw`
(not committed).

**Successes: 2/10 overall.** Sonnet 5.5 had 1/4 (seed 3), Opus 5.5 had 1/4 (seed 0) and Haiku 4.5
had 0/2. **The jaw-alignment part of the task worked.** Every episode turned the jaws in the right
direction, at most 2 turns into the episode. 8 of 10 first closes were within 12° of the packet's
thin axis, and all 10 within 18°. The +25° seeds needed a +40° turn, and both Sonnet and Opus hit it exactly (0° error).
All 10 first closes reported `holding: yes` (grip 35–46 mm). **The failures came after the grasp:** carrying the packet
past the box's near wall.

Usage (control calls only): 222 calls, 2.44 M prompt tokens of which about 88 % were cache reads,
and 52.6 k completion tokens.

- Sonnet 5.5: p50 3.7–6.7 s, p90 7–11 s, 4.2–14.2 k reasoning tokens per episode. It thinks much
  more than Opus.
- Opus 5.5: p50 3.4–9.6 s, 1.9–4.1 k reasoning tokens per episode.
- Haiku 4.5 (no-think): p50 1.5–1.7 s.

Cache regressions (a turn reading less than the previous call cached) appeared on 8 of 10
episodes: 0–6 per episode, more often on Sonnet.

## 5. Failure modes (with quoted replies)

1. **Carrying too low past the near wall (5 episodes).** Lifting above the packet runs into the
   arm's reach at about z 235–260. The models then carry in +y at that height. The packet hangs
   below the TCP, about 30 mm lower than the TCP, so its bottom is near z ≈ 150–230, while the
   wall top is at z = 180. The models did not know the wall height; the planner guessed "Rim height
   is uncertain, about 130-180 mm".
   - Sonnet seed 1 carried at z = 238 ("MOVE ee_delta 10 90 0 0 / STATUS OK carrying toward the
     box, staying high") and got "STOP: motion stopped: contact force 73 N against the packet". It
     then lifted, got a second STOP, and opened the gripper ("releasing the jammed packet over the
     wall to relieve force"), which was the third STOP.
   - Sonnet seed 2 carried at z = 181 and also jammed.
   - Opus seed 0 succeeded because it rose while moving toward the box ("MOVE ee_delta 0 50 80 0
     / STATUS OK packet lifted, rising and heading toward box"). That is the expert's "climb", and
     the reach allows it closer to the base.
2. **Solver blow-ups on box contact (3 episodes).**
   - Opus seed 1 ("MOVE ee_delta 0 90 0 0 / STATUS OK carrying over box near wall") and Haiku
     seed 1 hit the wall; PhysX diverged, with contact peaks up to 1e20 N and the arm teleported
     to x = +270, z = −62.
   - The harness reported only "71 N against the packet" and kept going. Opus then wrote "HOLD /
     STATUS STUCK abnormal contact forces, simulation glitch suspected" and "STATUS FAIL sim
     contact blow-up threw packet out of view". Haiku tried to recover from a pose 400 mm outside
     the workspace.
   - Haiku seed 1 then crashed the server. Both cases are now `unstable` STOPs (§1).
   - Opus seed 3 reached over the box at x = −416 and y = +46 near the edge of its reach and drove
     the arm into the box at 136 kN ("STATUS OK moving further over box interior centre"). The
     envelope has no link-vs-box check.
3. **Held-packet force STOPs with no obstacle.** In Opus seed 2, a turn plus lift near the reach
   edge ("MOVE ee_delta 0 0 50 -20 / STATUS OK packet held, lifting higher and turning back toward
   90") gave 89 N, and the next carry gave 85 N, both at z ≈ 200, far from any wall. The 40 N
   object stop (baseline floor capped at 80 N) counts the grip's own dynamic forces.
   - Genuine wall jams sampled 47–222 N, so the two ranges overlap and no single threshold
     separates them. A fix needs packet-vs-environment contact data, which the server does not
     observe yet.
   - Opus then crept toward the box in 30 mm steps ("gentle steps toward the box to avoid a third
     stop"), and its third STOP was the wall.
4. **The straight-elbow trap (Sonnet seed 0, fixed).** After "MOVE ee_delta 0 0 60 0 / STATUS OK
   lifting further", the reach clamp left the elbow at 0.0°. From then on every move came back
   "not reachable without a large joint swing (the arm is stretched to the edge of its reach …)".
   The model probed for 17 turns ("STATUS STUCK probing a tiny upward move to find any reachable
   direction"). It wrongly concluded "packet did not lift", even though STATE said `holding: yes`,
   and opened the gripper.
5. **Small turns when the plan said so.** For seed 0 (needed jaw 68°, start 87°) the planner wrote
   "yaw 87, which is already about right for the grasp" and "Set yaw to about 80-85". All three
   models turned only −5° to −7° and closed 12–14° off the thin axis. The grasp still held, so the
   35 mm packet tolerates about 15° of misalignment. Where the plan computed the heading ("Grasp
   yaw: 127 (37+90)"), the models followed it exactly.
6. **A wrong DONE (Haiku seed 0).** Haiku opened the gripper at z = 223, y = −89, over the near
   wall, and then said "STATUS DONE wafer packet placed in box". The goal check said not reached,
   and the regrasp attempts hit the box twice.

## 6. Recommendations

1. **Put the box geometry in the task or manual:** near wall at y ≈ −85 mm, rim at z = 180 mm, and
   the held packet hangs about 30 mm below the TCP. Also state the carry corridor: rise while
   moving toward +y, cross the wall with the TCP at z ≥ 260, and that height was reachable at x ≈ −415
   in Opus seed 0's successful carry, but not further out. This would have saved 5 of the 8 failures. Re-run seeds 0–3 for
   Sonnet 5.5 and Opus 5.5 with the same pinned plans.
2. **Add a link-vs-box check to the envelope** (wrist and housing against the box solid, as in
   `tasks.body_box_clearance`, inflated by about 50 mm). Blocking a move before it runs is better
   than PhysX blow-ups.
3. **Observe packet-vs-environment contacts in the server** (a contact view on the packet), then
   stop on "held packet against box/table" instead of on pad force, which also includes the grip's
   own dynamic forces.
4. **Keep the graspable yaw window at −40…+30°** for this tilt, e.g. `yaw_dist: uniform`,
   `yaw_max_deg: 40` plus a new `yaw_min_max` clip. For the full ±40°, allow a pitch change
   (rotation=full) or move the packet off the box side.
5. The planner computed headings well when it stated them; ask it to always write "packet heading
   h, grasp yaw = h ± 90". Sonnet 5.5 spends 3–5× more reasoning tokens than Opus for the same
   behaviour; try `no-think/claude/claude-sonnet-5-5` next.

## 7. Tests and spend

| where | command | result |
|---|---|---|
| local | `uv run pytest -q` | 360 passed, 21 skipped |
| compute3 (after all fixes) | `CONTROLR_ISAAC=1 .venv/bin/python -m pytest -q tests/` | 377 passed, 4 skipped (live LLM), 19 min |
| compute3, yaw Isaac tests | `-k "yaw or packet_stands"` | 7 passed (14.5 min) |

Spend: 222 control-model calls (budget 250): 100 Sonnet 5.5, 81 Opus 5.5, 41 Haiku 4.5. 4 planner
calls (budget 6), plus one fake-LLM dry run (0 calls). Isaac GPU time was about 1.5 h. The server
was stopped after the round and after the tests.

Videos (`docs/video/`, 0.7 frames/s, per-turn LLM latency in the panel):

- `yaw_sonnet55_seed3_success.mp4`
- `yaw_opus55_seed0_success.mp4`
- `yaw_sonnet55_seed1_failure.mp4`: a good −42° turn and grasp, then the held packet jams on the
  near wall, three STOPs.
