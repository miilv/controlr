# 2026-10-26: the episode-start scan, gated by its trigger, on build_tower (mode B + verify, 25 paired layouts)

Follow-up to [scan-pose](2026-10-25-scan-pose.md). Offline, the fixed park-and-look pose fixed most of B1's occlusion
failures. This run asks whether that carries over to whole episodes: does running the scan when perception's trigger
asks for it (`RuntimeConfig.scan = "trigger"`) raise build_tower success in mode B against the same config without
it?

**Setup.**
- Code and data:
  - code `~/dojo-integ3` on compute3, frozen at feat/dojo-integ2 3ac3597 (perception 3ac3597, skills bcbe853,
    simlab ca5e406);
  - proxy eval: Isaac 5.1 physics + Isaac 6.0 render;
  - canonical backgrounds `bg_folds`;
  - dev layouts `build_tower/100000-100024.settled.json`, one repeat per layout and arm.
- Config, the same in both arms apart from `scan`:
  - mode perception, verify bundle, `refine`, `refine_max_shift_m` 0.07, `max_rehovers` 2;
  - `post_place`, `search`, `place_confirm`, `watch`, `save_look_frames`.
  - The scan arm: `scan: trigger`, at the defaults `scan_budget_frac` 0.12, `scan_min_step_lim` 300,
    `scan_step_rad` 0.16, `scan_settle` 2.
- Queue jobs: `B_build_scanoff_*` vs `B_build_scantrigger_*` (`a`-`e`, some split as `b1/b2` ... `e1/e2`, `a5`; the
  `__catalog_env_error` dirs are the quarantined first attempt, see
  [incident](../incidents/2026-10-26-catalog-env-crash.md)).
- Readouts: `python -m dojo.runtime.report --arm off:... --arm trigger:...`, plus
  `~/dojo-data/integ/scan_readout.sh` and `scan_phases.py`.

**Budget and spend.**
- 0 LLM calls: the agentic perceiver is off in both arms.
- GPU time: 2.1 h for the off arm, 2.4 h for trigger (one heavy lane, the GPU lock).

## Result

| arm | success | score | failures | steps med (lim 1050) | wall s / ep med |
|---|---|---|---|---|---|
| scan off | 9/25 | 39.2 | perception_gross 12, verify_reject 2, belief_mismatch 1, plan 1 | 643 | 328 |
| **scan trigger** | **16/25** | **65.2** | perception_gross 7, verify_reject 1, plan 1 | 664 | 361 |
| mode A (GT), reference | 25/25 | 100 | none | | |

The mode-A reference is `A_build_bundle3_*` (2855f2e, 25/25); the newer code (7f1b85d, `A_tower4`) is 10/10.

**Paired over the same 25 layouts:**
- The trigger arm rescued 8 layouts: 100003, 100005, 100006, 100013, 100017, 100018, 100019, 100021.
- It lost 1: 100024.
- A two-sided sign test on 8 vs 1 gives p = 0.04.

**Did the scan itself cause the rescues?**
- The trigger fired and the scan ran in 23 of 25 episodes. The 2 it skipped (100006, 100016) had a frame0_clear
  verdict.
- 100006 was rescued **without** a scan. In that episode the two arms ran the same thing, so this rescue measures
  run-to-run noise in mode B.
- That leaves 7 rescues attributable to the scan against 1 loss (p = 0.07).

**The flagged layouts** (the ones the frame-0 analysis singled out):
- 100003 (block1 / block3 shape swap) and 100005 (pieces missed) are fixed by the scan.
- 100000 (a piece never found) and 100008 (block3 read as block1, yaw 95 deg, z +12 mm) fail in both arms.

**The loss, 100024.**
- Frame 0 missed 3 pieces in both arms.
- The scan view found none of them.
- In the off arm the search from home found all 3. In the trigger arm the search ran from the scan pose and found 2
  (block4 stayed missing), so verify was disabled ("estimate lacks").
- Open question: does the search plan its regions as well from the scan pose as from home?

**Where the remaining failures are.**
- 7 of the trigger arm's 9 failures are still perception_gross at planning:
  - pieces never found: 100000, 100010, 100024, and 100022 (two missing);
  - one pillar / prism shape confusion: 100008 (yaw 95 deg, z +12 mm);
  - same-asset label permutations, 89-146 mm off at frame 0: 100011, 100014, 100022.
- The other two:
  - 100001: verify_reject, a place with a 90 deg relative tilt;
  - 100020: plan, a 12 mm place_miss.

## Wall time by phase (median s per episode; 25 paired layouts)

| phase | off | trigger | trigger - off (median of paired differences) |
|---|---|---|---|
| reset | 8.1 | 9.0 | +0.0 |
| sim (5.1 physics steps) | 33.9 | 44.6 | +2.7 |
| observe (6.0 render) | 7.3 | 8.9 | +0.2 |
| policy worker, total | 268.0 | 299.6 | +6.3 |
| - frame-0 estimate | 7.7 | 7.7 | -1.0 |
| - scan (motion + scan-primary estimate) | 0.0 | 17.9 | +17.9 |
| - looks | 10.6 | 8.9 | +0.6 |
| - twin verify | 164.7 | 210.0 | +16.6 |
| - skills + hooks (rest) | 31.0 | 37.5 | +5.4 |
| (refine ops, any phase) | 46.0 | 48.9 | +7.8 |
| **episode wall** | **328.0** | **361.0** | **+34.3** |
| env steps | 643 | 664 | +33 |

How to read the timings:
- The scan's direct cost is about 18 s and 14 env steps per episode. It is 14 of the 21 extra steps; the rest come
  from episodes that now get further.
- Most of the other extra time is survivorship: twin verify (+17 s) and sim (+3 s) grow because rescued episodes
  build more of the tower.
- On the 8 layouts both arms solved, wall time is the same (361 s vs 362 s), and steps go 654 -> 666.
- Accounting caveats:
  - refine ops run inside other phases, so that row is not additive;
  - the policy worker overlaps slightly with the eval loop, which is why "eval other" comes out at about -8 s.

## Found along the way
- Eval rows (`*_episodes.jsonl`) keep only the last 30 events, so the step-0 scan event drops out of long episodes.
  The first readout of this run counted 3/20 scans that ran; the reports say 18/20.
  - Fix (PR #56): rows carry `events_head` and `event_counts`.
  - For runs made before it, read the per-episode `*_report.json.gz`.

## Conclusions (n = 25 paired, one repeat)
- The scan gate is the clearest mode-B gain on build_tower so far: +7 successes (+26 score points), for about 18 s
  and 14 steps per episode.
- One rescue came without a scan, which shows the noise floor. The scan's net effect (7 vs 1) is significant only at
  about p 0.07, so it needs a repeat before it becomes the default. The defaults are unchanged: `scan` stays "off".
- What the scan does not touch:
  - pieces missing in every view;
  - same-asset permutations;
  - the pillar / prism shape confusion. That one needs identity at the hover (or label repair after the pick), not a
    better first view.

## Next
- stack_blocks and push_T scan A/B (queued next, 25 paired each), then scanFA (scan + one agentic estimate).
- A second repeat of the tower pair before changing the default.
- Search from the scan pose (100024).
