# 2026-11-07 — MATCH2 camera screen (diagnostic; NOT an adoption test)

Prospective registration, committed before any row is queued. No camera data exist for this block.

**This screen is diagnostic.** It measures, per stage, what the MATCH2 switches do on camera. It cannot adopt a
lever, change a default or a candidate, and no verdict is drawn from its score. The adoption test is a separate
follow-up: a 20-pair panel on a NEW salt over ids 100000-09 + 100020-29 under the current Camera A/B rule
(EXPERIMENTS.md), registered only after this screen. This block's layouts and ids are never reused for it.

Background: [MATCH2 report](2026-11-07-match2.md), [journal](../journal/2026-11-07-dojo-match2.md), PR #420.

## Arms

Both arms: merged main `ec865e93f7ac691376b579b6083bfbec75793e83` (contains #420), deployed on compute6 at
`~/dojo-integ-c6-ec865e93` by the KEY / KONG / POUR recipe (isolated Python 3.12.13 venv, candidate / package parity,
catalogue gate), family `c6b_matchpicksg_m2scr`, `submit_v3`, Isaac 5.1 rendered camera-only belief,
RuleSupervisor, `AGENTIC_MODEL=claude/claude-sonnet-5-5` pinned on every row, and the frozen candidate
`scripts/dojo/queues/2026-11-06-candidate.txt` (host paths relocated; it has `belt_wrist_veto: on`).

- **A** = candidate + `{"save_belt_look_frames": 6}` (a dev frame dump; no behaviour change).
- **B** = A + `{"belt_ref": "tracked", "belt_veto": "evidence", "belt_speed": "lock", "belt_high_lift": "on",
  "belt_retry": "window", "belt_wrist_id": "on"}`.

`belt_grasp=solid` and `belt_ref=similar` are excluded: they measured worse in G3 / offline.

## Inputs

Block `match2-screen-20261107-r1` (salt SHA256 `da470f5188828cb418da034630fea2c1e4171d86e4d3a5161c4012203c38c133`),
match_and_pick_from_conveyor ids **100000-100009** (all in `OUR_MEASUREMENT_SLICES`), unchanged v15 generator,
generated and raw-checked on compute3 (10 / 10 raw ok, G3 8 / 10 with today's skills). Copied to
`compute6:~/dojo-data/match2-cam/screen1` and verified byte for byte against the frozen SHA256s; raw-check repeated
on compute6 before queuing. The conveyor runs its RAW record (`settled: false`), so raw-check is its stability test.
Record hashes, design, rows and manifest: [data/match2-screen/](data/match2-screen/).

Frozen before insertion: manifest SHA256 `d5b56cc848e86f4c4d90a38449c1ae663cf9c8b65639d02476abd66885d6968c`, rows SHA256
`52ad063e1bb8f66973954745a9d00a443951a0608a1603bfc6824b420da24e3b` (20 rows). Deployment checks: package parity
`diff={}`, candidate parity true, catalogue gate OK; resolved RuntimeConfig against the previous compute6 deployment
`066e9c8f`: all 203 existing fields unchanged, the 8 new fields default off, A differs from the defaults only in
`save_belt_look_frames`, B from A in exactly the six switches. compute6 raw-check: 10 / 10 raw ok, records unchanged.
All preparation made zero model calls.

## Sample and order

10 pairs, one repeat per arm, AB for even pair ordinals and BA for odd ones. The 20 rows are inserted under the
admission and queue locks immediately before the first not-yet-started tie row of `~/dojo-integ-c6-state/queue2.txt`;
started pairs are never split, and every other row keeps its bytes and order. Infrastructure failures permit one
identical-input retry per arm; a missing terminal row is reported, never refilled. No selection, tuning or stopping
on outcomes. Stop dispatch on any HTTP 429.

## What is measured (per arm, per episode; GT for scoring only)

GT: physics-only replays of each episode's recorded actions on compute6 (`scripts/perception/m95_replay.py`, frames
every 5 steps plus every gripper change), plus the layout. No policy reads GT.

1. **Reference identity:** the remembered first object's key vs target0's asset.
2. **Match decision:** the picked track's nearest GT rider at the decision (target1 / another rider / none).
3. **Wrist vetoes:** every pick look's `ok`, `iou`, `visible_frac`, head-track age, the decision (veto / descend) and
   whether a real rider was within 60 mm of the track; the saved look frames (first 6 per episode) inspected for
   why a look sees nothing.
4. **Wrist identity (B):** `obj_id`, `identity_ok`, `identity_margin` against GT; competing-category vetoes and
   whether they were right.
5. **Belt speed at the pick:** the velocity used vs 4.0 mm / step; `belt_lock` events.
6. **Intercept error:** the wrist look's offset from the track, and GT TCP-to-object offset at the close.
7. **Holds by category:** each close on target1, its lift (GT) and slip, by target category.
8. **Score** per arm (mean, SD) and per-pair differences, reported descriptively only.

**Spend estimate:** 20 episodes x 2-4 logical Claude calls (Sonnet frame-0 agentic estimate): 40-80 calls,
~0.3-0.6 M prompt tokens; about 2 lane-hours. Actual calls / tokens are reported from the call logs.

## What would justify registering the panel

Descriptive, not a gate on adoption: B shows fewer reference errors and fewer vetoes of real riders than A on the
same layouts, without a new failure class (e.g. wrong-category picks after `belt_wrist_id`, or air closes after
evidence vetoes are skipped). If a switch shows harm in the stage data, it is dropped from the panel's B arm, and
the panel's registration says so.
