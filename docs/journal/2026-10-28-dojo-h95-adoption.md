# 2026-10-28: the h95 bundle adopted into the candidate (rendered A/B; stream H, lead's decision)

**Decision (lead).** h95 joins the candidate preset, on condition that stream M's write audit (physics-only replays of
the 44 rendered episodes, harmful vs helpful belief writes per writer) shows no excess harmful writes. It is three
switches, all of them off by default in code:

| switch | from |
|---|---|
| `put_into_opts` [spot, patient, sequence] | #250 |
| `drop_verdict` "release" | #250 |
| `refine_recentre_guard` "vis" | #244 |

**The rendered A/B.** compute4 runs2, c4b_*_hon_* vs c4b_*_hoff_*, both arms on 1dc4b6a: submit_v3 + grip_check v2.
Numbers from M's north-star poller:

| task | n | result |
|---|---|---|
| put_bottles | 11 | score +19.6 (9 up, 2 down); success 0/12 in both arms |
| stack_bowls (guard) | 5 | 0/5 with the bundle vs 2/5 without (score -37) |
| organize (guard) | 4 | flat |
| build_tower (guard) | | still running when this was written |

**Deviation from the pre-registered rule.** The rule was "rescued - lost >= +2" in successes. It cannot be met when
both arms are 0/12 on put_bottles. The decision rests instead on:
- put_bottles' score (+19.6, n = 11, 9 up / 2 down), which is the board's partial-credit metric;
- the code-path analysis of the guard below.

**stack_bowls: no flag path ran, so the -37 is run-to-run noise.** From the reports and logs of 100005-09, both arms:
- `drop_verdict` and `put_into_opts` act only inside `library.put_into`. Its only caller is put_bottles;
  `stack_objects` never calls it.
- `refine_recentre_guard` "vis" leaves "prior kept, no recentre" in the refine event's `why` when it fires. That
  appears in none of the 10 episodes, and the provider has no `recentre_partial_view` count.
- The first divergence, in 100005 (1.0 -> 0.15) and 100009 (1.0 -> 0.15), is the first pick of bowl0:
  - Same grasp, same verify passes, but in-hand drift 3.9 vs 1.6 mm (100009: 5.6 vs 7.4 mm).
  - The next place's twin verify then draws different passes ([0,0,1,0] vs [1,0,1,0]) and picks another candidate.
  - The "on" arm ends in "no reachable stack site" (100005) and belief_mismatch (100009).
- 100007 (0.15 -> 0.0) diverges at the first place refine, a re-hover in one arm only, with no flag path either.
- So rendered physics noise, amplified by the verify's sampling, accounts for the whole difference. One pair of 5-episode
  runs cannot separate 2/5 from 0/5.

**Pending.** M's write audit decides the condition. A second hon / hoff pair on stack_bowls would firm up the guard
read.
