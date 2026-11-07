# 2026-11-07 — CLASSIFY3 camera A/B pre-registration: the sort skill's lowest-point container test

**Status:** approved by the lead (7 Nov, slot on compute4 after the partner's slot, ~23:45Z). Registered on main before
any layout generation and before any queue write. Evidence: [CLASSIFY3 report](2026-11-07-classify3.md) §2–§3.

**Hypothesis.** On camera, the sort skill's root-below-rim container test reads placed objects as not placed. Judging
"in the container" on the object's lowest point (`sort_contain=bottom`) stops the re-dos and raises the score on the
two sort-into-containers cells. G3 cannot test this: with true state both tests agree (§4 of the report).

## Arms (one code, one stack)

Merged main `2f198642900d5b6b280b6ee17918ed72c94b9365` (contains #431), deployed to
`compute4:~/dojo-integ-c4-2f198642` by the MOVE34 / KONG compute4 recipe:
- source from `git archive`;
- an independent copy of the working `416a2a03` lane venv, re-pointed to the new source;
- package and candidate parity, catalogue gate;
- the per-job Kit environment (offline extension cache, 600 s watchdogs) through `BASH_ENV`.

| arm | config |
|---|---|
| **A** | the frozen candidate `scripts/dojo/queues/2026-11-06-candidate.txt` + `instance_evidence=log` |
| **B** | A + `sort_contain=bottom` |

- `instance_evidence=log` writes no belief and changes no action. It is on in both arms, so the evidence ledger's
  camera numbers come from the same episodes. `sort_park` stays `off` in both: neither park setting was a general
  gain in G3.
- Every row:
  - `--setenv AGENTIC_MODEL=claude/claude-sonnet-5-5`, checked on every row before queuing and on the live process;
  - a private salted `DOJO_DATA`, `--renderer isaac51`, 25-minute episode / job caps;
  - the existing lane wrapper and watchdogs.

## Layouts (fresh salt; prepared on compute3)

- Salt `classify3-cam-20261107-040c042653adcae94f9632b7c1f915f0`: `'classify3-cam-20261107-' +
  sha256('classify3-cam-20261107:' + code)[:32]`. Frozen in [design.json](data/classify3-camera/design.json) with
  [prep.py](data/classify3-camera/prep.py) before any generation.
- Ids **100000–100009 and 100020–100029 only**, never ≥ 100100, never sealed data or old salts.
- `dojo.simlab.gen_layouts --seed-salt`, then the default two-pass physics-only presettle. **Selection by stability
  only:** per cell, the first stable ids in this batch order:

| cell | role | pairs | batch order |
|---|---|---|---|
| classify_objects | gain | 10 | 100020–29, then 100000–09 |
| classify_objects_by_language | gain | 10 | 100020–29, then 100000–09 |
| stack_blocks | strong guard + parity guard | 5 | 100020–24, then 100025–29, then 100000–09 |

  A cell short of stable ids after its batches runs short, and that is reported. Every attempt and index is kept.
- Prep on compute3 light locks (≤ 2 jobs, admission GPU used ≤ 24 GiB and disk ≥ 10 GB, 8 G / 600 % scopes), zero
  model calls. The block is copied to `compute4:~/dojo-data/classify3-cam/` and SHA256-verified file by file. The input
  manifest is committed before the queue write.

## Panel and order

- 25 pairs = 50 episodes.
- Per id, the two arms run back to back, AB on even ordinals and BA on odd ones within each cell.
- Appended at the **end** of `compute4:~/dojo-integ-c4-state/queue2.txt` under `queue2.txt.lock`. Existing rows are
  never reordered. The lead's intentional STOP is not touched; the lead resumes the lanes.
- Family names: `c4b_<cell>sg_cls3{A,B}r1_<id>`.

## Decision (EXPERIMENTS.md §Camera A/B, as of 7 Nov)

1. **Gain (score-based):** over the two gain cells pooled (20 pairs), B's pooled score ≥ A's + 5 points.
   - Per-pair score wins (≥ 10 points) ≥ 2 × losses, with ≥ 2 net wins.
   - Success flips and per-cell results are reported alongside.
2. **Strong guard:** stack_blocks pooled score of B no more than 5 points below A's, and no more flip losses than wins.
3. **Parity guard (stack_blocks):** identical resolved config apart from `sort_contain`, and zero lever actions on every
   stack_blocks episode.
   - The sort route never fires there: no `ported:sort_into_containers` and no `classify_into_bins_*` skill in its
     history.
   - Checked from the logs after collection; its score is gated as the strong guard above.
4. **Write audit: N/A by proof** (the writer inventory below), checked after collection.
   - Every `belief_write` writer that appears in B must appear in A.
   - B's per-writer counts are reported. A writer present only in B voids the proof: B then fails the gate.

Infrastructure failures (rc 4, 0 steps) are retried once with identical inputs by the lane. A pair whose retry also
fails infrastructure is excluded and reported. Budget exhaustion, query endings and wall caps are outcomes. A safety
incident stops the batch and fails the A/B. No reruns on these ids.

## Writer inventory (why B writes no belief)

`sort_contain` is read in exactly one place: `planner.ported_plan`'s sort route. It hands `contain="bottom"` to
`sort_into_containers`, which replaces p2's module-level `in_bin` predicate with `in_bin_bottom`, a pure function of the
skill's `obs()` dict. Neither function calls the provider. The belief writers are:
- `perception` / `predicted` writes (`belief.RuntimeProvider`: refines, post-place, resync, adopt);
- `held_pose`;
- watch / relocate;
- `confirm_pose`.

They are reached only through Skills / provider calls that both arms make through the same p2 stage code. B changes
**which** objects the skill re-attempts, so writes of existing writers can differ in number (fewer re-dos). It
introduces no write path. `instance_evidence=log` is identical in both arms and never writes the world
(`dojo.perception.evidence` docstring, `tests/dojo/test_evidence.py`).

## Reported per episode

- score and steps;
- the skill's own `truth_in` records, and false "not in" against a physics replay of the recorded actions
  (`stage_audit.py`, dev scoring);
- re-picks from inside a basket; budget stops;
- frame-0 instances found / category-right; ledger candidates and hits (`candidates_offline.py` on the logged
  records);
- agentic calls and tokens.

## Budget

- **Episodes:** 50 camera episodes (40 sort at ~550 s, 10 stack ~160 s ≈ 6.6 lane-hours, ~3.3 h on two lanes).
- **Claude route** (`claude/claude-sonnet-5-5`): the candidate's agentic perceiver averaged ~6 calls per sort episode
  (frame 0 + after looks) and fewer on stack. That gives **≤ 300 calls** (`agentic_max` 3 estimates of ≤ 4 calls each
  bound it at ~600), ~4 M prompt tokens, mostly cache reads.
- Prep: zero calls. Stop at any HTTP 429.

## Reference

REPRO's PyRUA-RSI program on its first five salted layouts ([per_cell.json](data/rsi-yardstick/per_cell.json)): classify
100 (n = 5), language 88 (n = 5). Those are different layouts from this panel: context, not a gate.
