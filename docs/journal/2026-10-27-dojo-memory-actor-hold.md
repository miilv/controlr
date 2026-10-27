# 2026-10-27: no arm motion before a skill that waits on a scripted actor (Memory stream)

**Finding.**
- imitate_sorting_sequence's checker has a failure query with aim 0, so one occurrence fails the env:
  `is_robot_not_back_to_origin(left OR right, 0.3 m / 30 deg) AND is_robot_not_back_to_origin(support_arm0, 0.3 m / 40 deg)`.
  Our arms away from home while the demonstrator's arm is away from its own ends the episode.
- make_kong carries the same query (Long-Horizon checked). Its discard starts at the first step.
- submit_v1 scans at step 0 when a label is not found, then looks, and the actor starts at step 1.
- Genuine-5.1 B, imitate 100002 (a demo layout): the episode ended at step 27, success False.
- breadth-2 never saw this: its imitate layout 100000 has no demonstration (the no-demo stub never moves), and
  make_kong survived by luck.
- play_tic_tac_toe has the query too, but its opponent starts only after our first piece. `play_board_game` banks and
  homes each turn.

**Change (default behaviour of submit_v1 for these skills).**
- `library.WAITS_ON_ACTOR = {copy_demonstration, declare_kong}` (a skill property).
- `planner.waits_on_actor(instruction, world)` returns the start plan's skill when it is one of them. Without the
  objects the route needs, it falls back to the route's own instruction patterns (shared with `plan`).
- When it applies, `RuntimePolicy._setup` skips the scan, the looks and the search (event `actor_hold`). Perception
  from frame 0 and the agentic gate still run.
- Both skills already hold at home from their start: `copy_demonstration` watches with `ex.hold`, `declare_kong`
  holds >= 64 steps.

Tests: `test_no_arm_moves_before_a_skill_that_waits_on_a_scripted_actor` (kong, imitate, and stack as the control),
`test_waits_on_actor_from_the_plan_or_the_words`.
