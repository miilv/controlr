# 2026-10-28: consumer-tolerance events (stream H, the reviewer's item)

**Question.** Does an in-hand pose error at a place ("class B" in M's metric) matter?

**Arm 8 ON, M's score_inbox:** 22 class-B events (rot 12, overshoot 4, gross 3, identity 3).
- **13 are probably harmless:**
  - 7 bottles dropped into a bin, where the release tolerance is the opening;
  - 4 bowls: 2 in episodes that succeed, 2 in episodes that failed on target perception;
  - a sweep's broom_shovel (115 mm), a belief error, not in-hand.
- **At most ~6 plausibly matter:**
  - store_tools: 8 events of 10-42 deg rotation. The tools end 2-8 mm from plan, but the score arithmetic caps the
    damage: 100003 had 3 rot-wrong tools and only 1 tool failed.
  - the alarm (37 mm / 43 deg; it fell 148 mm), maybe the broom.

So the metric over-counts B. It should judge a belief error against the precision the consuming action needs.

**Built.** Every contact-sensitive skill action logs one event into the run's event log (`Skills._consume`;
`RuntimeProvider.on_consume` -> report events):

    {"kind": "consume", "step", "skill", "label", "key", "pose7_used", "need": {"xy_mm", "z_mm", "yaw_deg"},
     "purpose", "arm", ...}

- `pose7_used` is the belief pose the action acts on: world frame, of the object's asset (mesh) frame, as grasp_metric
  and B1 records use it.
- A held object at a place also logs `T_tcp_obj7`, the in-hand transform the belief used, so M can split an in-hand
  error from a target-pose error. `key` is the believed asset.
- `need`, from the skill's own constants / geometry, never a task name; null = the axis does not matter:

| purpose | emitted at | need |
|---|---|---|
| grasp | `execute_pick`, before the descent | xy = jaw clearance per side, (`Grasp.open_gap` - width) / 2 |
| place | `run_place`, before the descent (held object + its reference) | the place's `tol_xy` / `tol_z` / `tol_yaw` |
| insert | a servo tail | xy = `held_gate` (default half of `tol_xy`), z = `seat_tol` |
| hang, thread | `hang_on_hooks`, `screw_on` (the place's `purpose`) | as place / insert |
| press | `press_sequence`, `press_once`, `press_joint` | xy = half the cap's / part's smaller horizontal size; z = `CAP_DEPTH` - 5 mm (the checker's 5 mm), or the stroke's `over` |

**Behaviour.** `RuntimeConfig.consume_log`, on by default because it is logging only.
`tests/dojo/test_consume_events.py` runs one kinematic episode with it on and off, in mode perception and mode gt, and
asserts identical action streams.

**Schema agreed with M.** M scores each event per axis against GT at step - 1, ignoring null axes, and reports a
per-purpose clean rate next to the grasp-time one.
