# 2026-10-27: plug_in_charger: an in-hand flip waits for a second view (rendered 5.1, 10 paired layouts)

**Question.** In rendered mode B the in-hand ("held") view and the pick view of the charger disagree by 180 deg about
half the time. Each adopted in-hand flip makes the skills put the charger down and re-plan, and the loop can repeat.
Does adopting an in-hand flip only when a second, different view agrees (`RuntimeConfig.held_flip_confirm`) fix it?

**Setup.**
- Genuine Isaac 5.1 rendered episodes (`integ_eval.sh --renderer isaac51`), compute6, preset `submit_v1`
  (`held_pose` off).
- Two arms: main a3dc7e1 vs + the fix (404cc5c). Settled dev layouts 100000-100009, chunks of 5, arms interleaved.
- LLM calls: the agentic perceiver's, ~3 per episode.

**Results.**

| layout | main | fix | in-hand flips (main / fix) |
|---|---|---|---|
| 100000 | 0 belief_mismatch | **1** | 1 adopted, put down / 1 pending |
| 100001 | 1 | 1 | - |
| 100002 | 0 plan | 0 plan | - (no relay plan in any mode) |
| 100003 | 0 plan | **1** | 3 adopted, 3 put-downs / 1 pending |
| 100004 | 1 | 1 | - |
| 100005 | 0 perception_gross | 0 perception_gross | - |
| 100006 | 1 | 1 | - |
| 100007 | 1 | 1 | - |
| 100008 | 0 belief_mismatch | **1** | 1 adopted, put down / 1 pending |
| 100009 | 0 grasp_slip | **1** | - / 1 pending, confirmed by the other arm's pick view |
| **total** | **4/10** | **8/10** | rescued 4, lost 0; wall median 171 -> 141 s |

**Conclusion.**
- Three rescues are the mechanism itself: the same layouts' in-hand flips no longer put the charger down.
- One shows the confirmation path: a flip a second view agreed with was applied.
- n = 10, 4 / 0, consistent with the replay of 100004 (Perception-2: the in-hand flipped side does not converge).
- `held_flip_confirm` defaults on (a default change, journal 2026-10-27-dojo-precision.md §5).

Data: compute6 `~/dojo-data/precision/runs/B_plug_flip{A,B}_1000{00,05}/`.
