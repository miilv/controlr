# 2026-11-08 — IMIT2 round 3: audit, general handover rule, route, demo timing

Lead's go after the [camera verdict](../experiments/2026-11-08-imit2-camera.md). Report with all numbers:
[experiments/2026-11-08-imit2-round3](../experiments/2026-11-08-imit2-round3.md).

- **Registered write audit (B vs C):** 32 physics replays on compute3 (2 light-lock jobs, nice), every replay
  reproduced its episode's score. `perception` writer +7.1 pp harmful rate in B (24 / 42 vs 13 / 26); the one adopted
  `refind` write harmful by the registered association → **B vs C does not pass.** 129 / 133 associations were nearest
  fallbacks (imitate's perception labels share the GT name space); BACKLOG item.
- **`CROSS_TRIES_KEYS` gone** (local branch `ilia/imit2-general`, `44bd0caa`, not pushed: the repo moves). Evidence
  alone (a slip / air close of this object's direct pick) lost 100029 in G3; evidence + a flat-long perceived box
  (`REACH_FLAT` 0.28) equals current B on all 17 dev layouts. A test checks every IMIT2 function's code against all
  task names, vocabularies and catalogue categories. The converged legality engine shows zero new findings, after the
  new lines were rewritten without unresolved calls and a global name clash with the belt's `FLAT_RATIO` was renamed.
- **Route:** written up (BACKLOG, report §3): a planner-callable `imitate(demonstrator, demo_container, our_container)`
  verb, what in IMIT2 already fits, what is still task-shaped.
- **Demo timing:** from public code only (sim dt, collect_freq, the planner's interpolation, the Franka's limits, the
  gripper's 100 sub-steps). Our demos 570–900 env steps vs ~410–550 at public timing for the same waypoints; the
  re-timing caps cause it. Proposed `demo_gen --timing public`, not implemented. `demo_gen`'s old pace comments came
  from official make_kong / tic-tac-toe `Traj` files (25 Oct): flagged.
- Spend: 0 LLM calls, 0 camera jobs; compute3 32 replays + 51 G3 episodes. No other process touched.
