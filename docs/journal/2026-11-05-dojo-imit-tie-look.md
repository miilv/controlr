# IMIT: camera dev failures and twin ties

Scope: only old compute4 `~/dojo-data/integ-c4/runs2/c4b_imitate*` reports and episode JSONL for
100000-04. Files were selected by the row-directory id before reading their contents. No scoring ids, ids >=100100,
sealed directories, new camera runs, or compute4 writes/GPU work. Downloaded reports + rows total 496 KB.
No new LLM calls (0 tokens, 0 spend). No change to defaults. This is a diagnosis and candidate, not a board gain.

## What the rows establish

27 rows on five dev ids (100000: 2, 100001: 8, 100002: 5, 100003: 8, 100004: 4).
12 are watchdog infrastructure failures, 5 checker-invalid rows with no observed departure (old no-demo runs).
The 10 remaining rows have these outcomes; repeated layouts and different configs are not independent samples.

- `dev_100002`, `dev_r2_100002`, `dev_smask_100002`: score 0, ended at step 669. All five departures recorded by
  step 585; calm motion window returned at 664 while support was returning. The adopted home wait addresses this.
- `dev_home_100002`: 100, 1144 steps; `dev_wid_100002`: 100, 1257 steps. Both wait to 702 (38 extra home steps).
  The latter survives re-entry and copies all five. Demonstration reading and ending can work on camera.
- `dev_home_100001`: 0, 1065 steps. Five departures, start 838. Our watch is believed garage/00014 (`t1`),
  competing with the real garage (`t3`). Assignment gives the first garage departure to t1. Six empty closes over
  two attempts; no copy. Existing repair addresses the retry symptom; a correct identity is needed for the full order.
- `dev_smask_100001`: 30, 1069 steps. Two copied, false fallen read triggers re-entry; halves recomputed and
  next toy-car match lost. Adopted keep_sides addresses this (see wait-home journal).
- `dev_smask_100003__retry`: 0, 811 steps. Five departures, copy starts 808. First twin is a phantom;
  0/296 reachable grasps on all six attempts. Detection/identity, not a step-limit truncation.
- `dev_wid_100003`: 5, 907 steps. First twin eventually copied, second unreachable (32/306 grasps,
  no feasible pick). Repair drops it, then no matching phone. GT probe reports missing objects.
- `dev_100004__retry`: 0, 1179 steps. Four departures observed; camera identity/reach errors, relay and empty-jaw
  pick, a believed placement followed by false fallen/re-entry, then a post-place query failure (`queries[1][1]=1`).
  This is mixed perception/place/re-entry evidence, not evidence for a new tuned motion primitive.

Dominant historical classes: start gating (3/10), identity/detection/reach (at least 3/10), re-entry (1/10), mixed
place/identity/re-entry (1/10), successes (2/10). The first and re-entry already have fixes. Among remaining home-wait
examples, identity is the actionable gap. No diagnostic valid row reaches the 1600 step limit. More looks nevertheless
consume the spare budget, so budget regressions must be gated in A/B. A missed departure cannot be recovered by this
candidate. DEMO_KEY affects generator demo lookup and does not explain a camera pick closing on air.

## Two gaps in the existing look

`imitate_look=ambiguous` selects only pairs with different believed asset keys. In home 100001 that selects t3,
not the misread t1: the wrong pair has the *same* believed key. Swapping the garage and watch pairs costs exactly the
same total identity cost (0 + 1 either way). `tied_twins` detects both objects through this equal-cost swap.
A local row tie alone is insufficient: another row may uniquely need the alternative, so the whole swap must tie.
This detector intentionally does not handle larger cycles or unassigned candidates.

Tracing `_look_over` also revealed that its `look` refine kind does not request identity classification in the runtime:
only pick/size did. The new switch uses explicit `imitate` reads to enable that existing classifier under the same
candidate-group and acceptance checks. Ordinary looks retain their old behavior. No hidden correspondence, quotas,
GT names or task checker feedback are supplied to the pairing or tie detector.

## Candidate

`RuntimeConfig.imitate_tie_look=false` / `Skills.imitate_tie_look=False`; G3 `--skill-attr imitate_tie_look=on`.
With one_to_one pairing it enables the look pass after the demonstration wait, prioritizes equal-cost swap objects,
then mismatched pairs; max 3, existing copy-budget guard, return home and reassign. The flag also gates the runtime's
identity classifier for explicit imitate reads. `refine_identity` remains respected. No default changes.

Tests cover same-key wrong twins, unique-match protection, missing labels, budget, max looks, changed hidden relation,
identity correction/reassignment before picks, wait-before-look ordering, explicit refine kind/default preservation,
and runtime identity-vocabulary routing. Synthetic identity corrections test plumbing, not classifier accuracy.

## Validation and limits

Isolated deployment: compute3 `~/controlr-imit-ties`; outputs `~/dojo-data/imit-ties`. Shared `~/dojo-gpu.lock`,
normal disk guard (123 GB free at start), one Isaac job at a time. G3 dev 100002, pairing one_to_one, wait home,
keep_sides on: tie candidate scored 100, 1408 steps, wall 79.3 s. Perfect identities produce no ties; this validates
execution only, not the new camera classifier or any score increase. The off baseline also scored 100 at 1408 steps
(wall 82.1 s). Final code head `0ea46d8b`, clean deployment recorded in `.deployed.json`, scored 100 at 1408 steps
(wall 78.6 s), with no infrastructure failure or instability; each process exited and released the lock. All three
were physics-only, no rendering service or persistent server started. Focused imitation/skills/runtime tests:
212 passed, 1 skipped; regrasp tests: 5 passed. Changed-file pyflakes is clean (the pre-existing unused import in
`skills.py` is unchanged). Full-suite validation is recorded on [PR #383](https://github.com/miilv/controlr/pull/383).
No camera runs were justified before a paired
rendered experiment; flags remain off pending the [pre-registered A/B](../experiments/2026-11-05-imit-tie-look-ab.md).

The switch is a limited candidate: it neither repairs missing detections nor changes the conservative identity
classifier's IoU/margin thresholds. The 27-row audit does not identify the cause of every current board failure;
its historical rows contain superseded configs, repeated ids and generator/infra failures.

The A/B uses dev 100030-49 × 2 repeats, A existing ambiguous looks vs B plus tie flag, genuine 5.1 cameras,
cx/gpt-6-luna, frozen demos/config, guard imitation successes and two unrelated cells. It requires a >=10-point gain
with paired CI >0, no excess harmful writes and no guard/budget regressions. It starts only after explicit D-C19
release; no measurement artifacts or scoring ids are inputs.
