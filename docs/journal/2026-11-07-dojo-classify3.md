# 2026-11-07: CLASSIFY3, where the sort cells lose on camera (stream CLASSIFY3)

Report: [experiments/2026-11-07-classify3.md](../experiments/2026-11-07-classify3.md). Branch `feat/dojo-classify3`.

## What was done

1. **Stage audit of the 20 D-C19 candidate camera rows.**
   - Physics-only replays of their recorded actions on compute3 reproduce all 20 scores.
   - `stage_audit.py` joins the replays with the logged belief, the skill's own records and the layouts: per GT instance
     the first failing stage, per skill record its claim against physics.
2. **The container test.**
   - The skill's root-below-rim test read 31 of 128 placed objects as not placed; physics never had the reverse.
   - `release_rest_offline.py` tried three tests × two rest rules on 109 releases; the lowest-point test is the fix.
   - A container-floor rest rule in the belief (first hypothesis) was built, measured, and removed: 23 → 21 false "not
     in". The bottom test gives 23 → 2.
3. **General sort skill.** `sort_into_containers` wraps the vendored p2 mechanics with two options:
   - `contain=root|bottom`;
   - `park_other=off|home`: the idle arm goes home before a one-arm carry. G3 refused carries were blocked by the
     other arm 30 of 74 times.
   - The planner's sort route (instruction words + containers in the world) takes it when `RuntimeConfig.sort_contain`
     / `sort_park` are set.
   - This replaced a `classify_into_bins_p3` variant after the hard rule of the same day (no classify-specific paths).
4. **Persistent per-instance evidence** (`dojo.perception.evidence`, general, every episode).
   - Candidates: unexplained object-sized head regions, kept until a look resolves them.
   - Identity votes, kept apart from pose writes.
   - `RuntimeConfig.instance_evidence=log` records them from every estimate record.
   - Offline, candidates cover 12 of 29 missed language instances; the consumer (look, add, re-plan) is the next build.
5. **G3 true state on the same layouts:**
   - p2: 67.5 / 53.3;
   - contain=bottom: identical on every layout;
   - park home: 94.0 / 40.0 (classify 4 wins / 0 losses, language 2 / 4);
   - park blocked: 67.5 / 56.7.
   The camera A/B pre-registers B = A + `sort_contain=bottom` only.

## Landmines found

- The eval's per-episode `perception` block (and so `category: perception_gross`) matches one estimate per config label.
  With numbered series it reports correctly found instances as "not found". Score series cells from a replay instead.
- G3 `--belief-drop` (heldsim) does not fit the ported skills: p2's picks fail with it on and succeed with true state.
- The physics-only replay of the c4b rows took ~2.5 min per episode on a loaded compute3. `--full` is needed to check the
  final score.
- `np.asarray` on a pose that is later edited in place aliases the caller's array: an offline tool counted different
  numbers on a rerun until it copied.
- A Bash hook blocks greps over env files; one combined command was blocked whole, so a copy in it never ran.

## Not done

- The camera A/B: pre-registered in the report and parked for the lead's slot allocation.
- The candidate consumer.
- A general series count: SAMINT's general union is the natural source.
- The car carries refused by neighbouring objects and basket walls.
