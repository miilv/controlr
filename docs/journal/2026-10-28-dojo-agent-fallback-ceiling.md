# 2026-10-28: the LLM agent as the "Plan none" fallback: a planning ceiling (verdict: stays off)

**Question.** 19 board cells (32.5 Average points) printed "Plan none": the rule planner names no skill. Can the hybrid
LLM agent (`AgentPolicy` with `planner_first`: the planner's skill first, the agent only when it names none) score
there while streams P / N build routes?

**Setup.** GT-scene-text planning ceiling: `scripts/agent_fallback/` (#263), i.e. `dojo.runtime.eval._run` on a
physics-only `ProxyEpisode` with a null renderer (no frames: the agent reads the GT scene text only), mode gt, verify
off.
- Model cx/gpt-6.1-sol, manual_v6 (the compliant manual: no checker numbers, judging phrases or per-task recipes;
  journal 2026-10-28-dojo-agent-manual-v6), `planner_first`, at most 30 calls.
- Settled layouts 100000-04 (planning ceilings).
- Light lanes: compute7 light8 until the lead's emergency stop, then compute3 light7 / light3, one job at a time,
  started at MemAvailable >= 12 GB. Episode wall cap 5.5 min (one 5-layout job per cell within the 30-min job timeout).
- The planner named no skill in every episode below, so the agent ran every time.
- Scope was trimmed mid-run: compute7's stop left 9 queued cells unmeasured, and the lead dropped the cells that got
  P / N routes (dolls x2, arrange x2, pour_liquid x2).

## Results

| cell | result | Average pts | calls / ep | prompt k (cached k) / out k per ep | wall / ep |
|---|---|---|---|---|---|
| cover_blocks | 4/4, score 100 (n = 4: compute7 stopped before 100004) | +3.33 | 6 | (compute7 off-limits after the stop) | 67-120 s |
| pack_objects_into_box | 0/5, score 4 | +0.03 | 25 | 203 (188) / 2.2 | 188 s |
| pack_objects_into_box_random | 0/5, score 4 | +0.03 | 26 | 214 (200) / 2.3 | 188 s |
| pour_by_language | 0/5, score 0 | 0 | 13 | 100 (90) / 1.5 | 214 s |
| ref: sort_nesting_dolls_by_size_random | 2/5, score 40 | (route exists) | 18 | 163 (146) / 1.6 | 135 s |
| ref: sort_nesting_dolls_by_size | 0/2 | (route exists) | 10 | 67 (62) / 0.9 | 98 s |

On the cells still Plan-none (pack x2, pour_by_language) the fallback adds about 0.07 points against 0 today. cover's
+3.33 holds only until P's ported skill lands.

## Failure modes (22 failed of 26 episodes)

1. **Reach / IK into the box:** 9 of 10 pack episodes ("target unreachable (no seed converged)", "joint move hits
   box").
2. **Place miss:** 5. A bottle 55-70 mm off after a pour; dolls tipped 21-84 deg, twice off the table.
3. **Grasp collision / slip:** 4 (0 feasible grasps; an object that never rose).
4. **The agent declares done wrongly:** 2 (dolls: `finish('done')` with a doll on top of another).
5. **Other:**
   - one pour episode hit the 5.5-min wall cap at step 750 / 800;
   - one pour episode refused, because without frames the instruction's colour words ("violet", "turquoise") did not
     match the catalogue's colour names (red / cyan / gray). This is a limit of the null renderer, not the agent.

## Verdict (the lead's)

- The fallback stays **default off and out of the preset**.
- The remaining Plan-none cells are skill-limited (reach, grasp, placement), not planning-limited: an LLM on top does
  not move them.
- Worth a re-test only in the camera dress rehearsal, where frames fix the colour grounding.

## Spend

- About 490 cx/ calls in all (subscription). compute3 alone: 431 calls, 3.53M prompt tokens (3.24M cached), 40k out,
  over 22 episodes.
- The rest went to the compute7 cover episodes, a manual_v5 pre-run that was set aside, and one probe.
- GPU: light lanes only.
