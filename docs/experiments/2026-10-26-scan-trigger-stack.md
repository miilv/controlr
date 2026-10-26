# 2026-10-26: the episode-start scan, gated by its trigger, on stack_blocks (mode B + verify, 25 paired layouts)

Second task of the scan A/B. The first is [scan-trigger-tower](2026-10-26-scan-trigger-tower.md), which has the
design, the code and the trigger. Same code (`~/dojo-integ3` = feat/dojo-integ2 3ac3597), same config (refine,
`refine_max_shift_m` 0.07, `max_rehovers` 2, post_place, search, place_confirm, watch, verify bundle), same proxy
(Isaac 5.1 physics + 6.0 render, canonical `bg_folds`).
- Arms: `scan: off` (`B_stack_scanoff_a-e`) vs `scan: trigger` (`B_stack_scantrigger_a-e`).
- Layouts: `stack_blocks/100000-100024.settled.json`, one repeat each.
- Spend: 0 LLM calls; GPU 0.6 h (off) + 0.5 h (trigger).

push_T was dropped from the A/B: the runtime has no push_T skill, so its episodes end at planning and measure only
planning perception.

## Result

| arm | success | score | failures | steps med (lim 550) | wall s / ep med |
|---|---|---|---|---|---|
| scan off | 17/25 | 71.0 | perception_gross 6, belief_mismatch 1, verify_reject 1 | 231 | 82.9 |
| **scan trigger** | **22/25** | **89.8** | verify_reject 2, belief_mismatch 1 | 254 | 74.8 |

**Paired over the same 25 layouts:**
- The trigger arm rescued 5 (100001, 100002, 100003, 100004, 100018) and lost 0 (sign test p = 0.06).
- The scan ran in 16 of 25 episodes; the other 9 had a frame0_clear verdict.
- 100003 was rescued **without** a scan, so it is run-to-run noise. Attributable to the scan: 4 vs 0.

**The off arm's failures.** All 6 perception_gross failures are frame-0 errors:
- blocks read under the wrong label or 30-110 mm off at confidence 0.4-0.5 (100001, 100002, 100014, 100020);
- blocks never found (100018, 100024).

100001 shows how fast a bad frame 0 ends an episode: the stack planner found "no reachable stack site" and the
episode finished after 26 steps.

**The trigger arm's remaining failures.**
- 100014 and 100020: verify_reject. The twin rejected every candidate and predicted the placed block 57 mm high
  ("dz +56.8 / +56.5 mm, verify passes [0, 0, 0, 0]").
  - This is the twin-load signature (journal 2026-10-25-dojo-integ2 §5c, §9).
  - Episodes with none_ok: off 5/25, trigger 6/25.
  - The twin-load A/B (`twin_load: consistent`, queued after breadth) measures exactly this.
- 100024: belief_mismatch (stages_left 1).

## Wall time (median s per episode; 25 paired layouts)

| phase | off | trigger | trigger - off (median of paired differences) |
|---|---|---|---|
| sim (5.1 physics steps) | 11.9 | 12.9 | +0.7 |
| policy worker, total | 65.8 | 59.5 | -4.4 |
| - scan (motion + scan-primary estimate) | 0.0 | 2.6 | +2.6 |
| - twin verify | 46.9 | 45.0 | -3.3 |
| **episode wall** | **82.9** | **74.8** | **-4.0** |
| env steps | 231 | 254 | +5 |

- On stack the scan costs 2.3-5.7 s per scan (vs ~18 s on the 8-piece tower) and 14 env steps.
- Trigger episodes were no slower overall: on the 17 layouts both arms solved, the median wall time is 74.7 s vs
  82.9 s.

## Both tasks together (tower + stack, 50 paired layouts)

| | tower | stack | total |
|---|---|---|---|
| success, off -> trigger | 9 -> 16 | 17 -> 22 | 26 -> 38 of 50 |
| rescued / lost | 8 / 1 | 5 / 0 | 13 / 1 (sign test p 0.002) |
| rescued without a scan (noise) | 1 (100006) | 1 (100003) | 2 |
| attributable to the scan / lost | 7 / 1 | 4 / 0 | 11 / 1 (p 0.006) |

**Noise.** The 2 rescues without a scan came from 11 paired episodes in which the scan did not run, so both arms ran
the same thing. Their outcomes flipped 2 times in 11: mode-B proxy episodes are not deterministic run to run.
A 2-in-11 flip rate is the scale of noise a single-repeat A/B on these tasks has to beat. The 11 / 1 split
above clears it.

## Conclusions (n = 25 + 25 paired, one repeat)
- On both tasks the trigger-gated scan raises mode-B success:
  - tower +7 (39 -> 65 score);
  - stack +5 (71 -> 90 score);
  - 1 loss in 50 paired layouts.
- Cost: 14 env steps when it runs. Wall time: +18 s on the tower, about +3 s on stack.
- The default stays `scan: off` until the lead decides. The evidence for switching the submit config to
  `scan: trigger` is 11 / 1 scan-attributable over two tasks.
- What the scan leaves: on stack, twin-load rejections (none_ok, the twin predicting a block 57 mm high); on the
  tower, missing pieces and the stacked phantom pillar (tower report, journal §8).

## Next
- The twin-load A/B (mode A + noise, raw vs consistent; mode B raw vs consistent), queued after breadth.
- A second repeat of both pairs, if the default is to change.
