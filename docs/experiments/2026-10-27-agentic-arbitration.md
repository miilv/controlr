# 2026-10-27 — Agentic perceiver: replace a B1 pose only with evidence (replay of 243 logged calls, no LLM)

**Question.** In breadth-2 plug_in_charger (proxy mode B, submit_v1), the agentic perceiver (Sonnet 5.5) replaced a
correct B1 / scan charger pose with a wrong one in 3 of 5 episodes: turned 80-180 deg, 23-47 mm off (Precision's
read). Which rule lets it, how often does the agent step help or hurt across tasks, and does an evidence check fix
the harm without losing the help?

**Data.**
- **Breadth runs.** Every logged agentic call of Integration's breadth-1 and breadth-2 runs:
  - `compute3:~/dojo-data/integ/runs/B_<task>_breadth{,2}/`;
  - 17 tasks × 5 dev layouts (100000-100004), 243 calls;
  - 6.0 proxy frames.
- **What each call's workspace holds.**
  - Frame 0, or the agentic2 observation.
  - The scan observation, when the scan ran.
  - The B1 hint as sent (`b1.json`).
  - The agent's replies (`transcript.jsonl`).
- **The merge target.** The B1 record the answer was merged into is the report's record before the agentic one.
- **Truth.** Our settled dev layouts. Identical assets are matched by position.
- **Caveat.** For make_kong's agentic2 calls the initial truth is stale: the Franka knocks the discard tile over
  before the call.
- **D0 hard set.** Perception-options' 28 hard D0 scenes, 102 objects, frame 0 only (no scan): Sonnet's saved answers
  after H2 (`sonnet_m_ref`, `sonnet_m2_ref`).

**Method.**
- `dojo.perception.agentic.replay` runs each logged answer through today's post-processing (`j1.finish_answer`: snap,
  H2) once.
- It then merges the result into the pre-agent record twice: without the new step ("arb0") and with it ("arb1").
- CPU silhouettes (`DOJO_GL=0`). The runtime used GL, so arb0 is close to, not identical to, the logged run.
- D0 hard set: `dev_d0 arbitrate` + `dev_d0 score`.
- 0 LLM calls.

## Cause

- The agent's charger point lies 23 mm from B1's, and B1's calibrated risk is >= 0.2, so `snap_to_b1` does not apply.
- H2 then does one of two things:
  - rejects its own fit, which keeps the agent's raw reading. The reading carries a yaw the agent typed in its own
    convention ("the prong direction as the heading": -118.8 deg against B1's correct -15.3);
  - fits from that yaw into a flip (100003: 180 deg, 47 mm).
- `merge_into_record` adopts any agent pose unconditionally.
- Raw agent points also carry the agent's head-plane height (often the top face): 30 of 99 agent-changed objects were
  15-154 mm off in z.

## Fix

`seed_refine.arbitrate`, run by `j1.finish_answer` after H2 (`AGENTIC_ARBITRATE`, default on):
- **Height.** An unrefined agent point rests where its paired B1 object rests (same lowest point), else on the table.
- **Pose.** An agent pose that moves (> 3 mm) or turns (> 5 deg) one of B1's same-kind objects is scored against B1's
  pose by silhouette loss in this frame + the scan views. Each pose is scored inside its own scene (the other objects
  of the same reading masked out). The agent's pose stands only if its loss is lower; else the object takes B1's
  pose, asset and confidence back.
  - An object B1 placed at its config slot needs a 0.15 lower loss (`AGENTIC_ARB_PINNED_MARGIN`).
  - An H2 fit that turns B1's pose by <= 30 deg stands without the comparison: H2's own fit is the evidence. Without
    this, the D0 hard set lost good H2 fits of blocks half under the home grippers.
- **Pairing.** Pairs are within 15 cm, as the merge pairs them. A free-text kind ("figurine") pairs with the nearest
  B1 object's kind; a catalogue kind B1 has none of (a missed alarm clock) pairs with nothing and fills its label.

## Results

**Breadth replay, 243 calls.** Labels whose pose the agent step changed vs the B1 record it was given (better = gross
-> not gross, worse = the reverse; gross = > 15 mm or > 15 deg):

| arm | better | worse | both ok | both gross | changed and \|z\| > 15 mm |
|---|---|---|---|---|---|
| logged runtime | 26 | 35 | 14 | 25 | 27 |
| replay, no arbitration (arb0) | 30 | 33 | 13 | 23 | 30 |
| **replay, arbitration (arb1)** | **24** | **5** | 4 | 15 | **14** |
| arb1 without the config-slot bar | 24 | 7 | 4 | 15 | 16 |

The 5 remaining "worse" are make_kong agentic2 discard tiles, where the truth is stale (the Franka knocked the tile
over). Without those calls: arb0 30 better / 27 worse -> arb1 24 / 0.

| task | arb0 better / worse | arb1 better / worse |
|---|---|---|
| plug_in_charger | 0 / 5 | 0 / 0 |
| make_kong | 2 / 20 | 2 / 5 (stale truth) |
| organize_table | 7 / 3 | 2 / 0 |
| deposit_coin | 15 / 0 | 14 / 0 |
| store_tools_in_toolbox | 0 / 2 | 0 / 0 |
| insert_tubes | 1 / 1 | 1 / 0 |
| play_Xylophone | 2 / 1 | 2 / 0 |
| make_toast | 0 / 1 | 0 / 0 |
| swap_blocks | 2 / 0 | 2 / 0 |
| imitate_sorting_sequence | 1 / 0 | 1 / 0 |

- **plug_in_charger.** The five breadth-2 chargers end at B1's pose: 0.5-0.9 mm, <= 1.2 deg. The logged runtime had
  them 23-47 mm off, turned 80-180 deg.
- **make_kong 100004 agentic0.** The row slide is gone: 14 config-slot tiles moved 21-28 mm without arbitration, 0
  with it (all 14 pairs kept B1).
- **Gains lost (6).**
  - organize_table: the keyboard x2 (B1 16-21 mm, agent 4 mm) and the garage figure x3 (B1 21-96 mm with the wrong
    figure, agent 4-8 mm). The silhouette loss prefers B1's pose. The garage's raw agent point is scored with B1's
    (wrong) figure, since the agent names no asset.
  - deposit_coin breadth-1 100001's stand (16 -> 1 mm).

**D0 hard set (28 scenes, 102 objects, frame 0 only, no scan).**

| tag | within tol | gross |
|---|---|---|
| B1 | 63 | 24 |
| Sonnet + H2 (`sonnet_m_ref`) | 67 | 11 |
| + arbitration | 67 | 11 |
| Sonnet + H2, B1 unmentioned kept (`sonnet_m2_ref`) | 66 | 11 |
| + arbitration | 66 | 12 |

- Kept: the arbitration reverts two agent flips (pack hammer and shoe, 171-179 deg).
- Lost: it reverts three raw agent points that were better: two stack blocks half under the home grippers (B1 17-22
  mm, agent 2-4 mm) and the m2 toothbrush (73 -> 13 mm).
- Frame 0 alone is a weaker check than the runtime's frame 0 + scan, under which the breadth numbers above were
  measured.

**Paths unchanged.** B1-only paths and G3 are unchanged: the step runs only in `finish_answer`, on an agent answer.
The new `flags.slot` in `record_objects` is not read by `risk()`.

## Caveats
- 6.0 proxy frames; the replay's silhouettes are CPU (`DOJO_GL=0`), the runtime's GL. n = 243 calls over 17 tasks ×
  5 layouts, one repeat each; the per-task rows are small.
- Not yet measured end to end: a breadth-3 run carries it.
- Spend: 0 LLM calls.
