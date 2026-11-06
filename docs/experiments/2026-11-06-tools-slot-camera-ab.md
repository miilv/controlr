# TOOLS: Cartesian slot approach camera A/B pre-registration

Hypothesis: a Cartesian TCP transit to the matching-slot hover reduces tool loss during carry and raises
`store_tools_in_toolbox` score. This is a prospective specification, not a camera result or launch authorization.
No camera episodes or LLM calls were run to prepare it. Do not launch unless the completed paired G3 development
screen supports this lever and the lead releases camera capacity after reviewing the PR.

## Arms and freeze

Use the same reviewed PR SHA for both arms. A: `slot_approach=off`. B: `slot_approach=linear`.
Everything else is identical: preset, runtime JSON, planner, vocabulary, assets, generator, calibration,
backgrounds, segmenter, gripper source, held-pose policy, step limit and episode timeout. Start from the candidate
captured in `scripts/dojo/queues/2026-11-06-candidate.txt`; before launch save the exact resolved A/B configs,
environment, code SHA and asset/calibration/layout hashes. Do not reconstruct the candidate from old journal recipes.
In both arms explicitly pin **every** agentic/grounding route to `cx/gpt-6-luna`, with identical reasoning and call
limits. No `claude/` or `cc/` route is authorized. This is a cx-stack comparison, not a direct board update on a
different perception route.

Generate fresh v15 development records using the existing salted generator, IDs 100000–04 in each named block:

| Block | Task | Salt | Pairs per layout |
|---|---|---|---|
| tools_gain_a | store_tools_in_toolbox | tools-slot-linear-camera-a-20261106 | 2 |
| tools_gain_b | store_tools_in_toolbox | tools-slot-linear-camera-b-20261106 | 2 |
| tools_guard_tubes | insert_tubes | tools-slot-linear-guard-tubes-20261106 | 1 |
| tools_guard_bowls | stack_bowls | tools-slot-linear-guard-bowls-20261106 | 1 |

Primary: ten distinct dev layouts, two repeats each = **20 pairs**. Guards: five pairs per task. The guards test
accidental activation outside matching-slot placement after the generic planner parameter is added. No 100005–09,
100020–29, >=100100 or sealed records. Preserve the original training/other-stream refusals; do not bypass loaders.

Generate and two-pass presettle all blocks, require every status `ok`, then freeze a manifest of raw and settled
hashes before running either arm. A dropped layout blocks the schedule; do not replace it after seeing scores.
Use genuine Isaac 5.1 camera frames. Run both arms on the same box, alternating AB/BA by (block + id + repeat)
parity; save that complete schedule before launch. Sequential pairs under the assigned camera locks. This spec does
not authorize compute4 access, sealed-directory reads, system changes or extra concurrent jobs.

## Outcomes, audit and cost

Report official score (0–100), success, steps, pick/place statuses, tool loss phase, held reads and look steps,
planning time, LLM calls/tokens/cache reads and occupied GPU time. Keep all attempted rows. Retry a classified
infrastructure failure once with identical inputs; publish both attempts. A skill timeout, failed plan, slip or
budget exhaustion is an outcome. Unresolved infrastructure pairs leave the decision pending until the frozen
schedule can be completed. No score-based early stopping or optional removal of a losing layout.

Publish per-layout mean scores over repeats, paired deltas, n, score SD and success flips. Also give a paired
bootstrap 95% CI (10,000 draws, RNG seed 0), resampling distinct layouts with both repeats together. The CI is
descriptive; the adoption criteria below are frozen independently of it. Produce a success and typical failure
video from the completed camera runs using the existing replay/video tooling.

Run `scripts/perception/m95_score.py audit` on matched physics replays of both arms, with GT at every logged belief
write (including any held-pose, refine and post-place writer reached because the trajectory changed). Harmful means
>2 mm added 3-D error. Publish helpful/harmful counts by writer; missing write coverage blocks adoption. Audit GT
never enters the controller or threshold tuning.

Budget: 60 scheduled camera episodes (40 gain + 20 guard), plus at most one infrastructure retry per episode.
Preflight estimate: at most 50 perception/model calls per episode = 3,000 calls before retries; stop and report if
the frozen stack cannot respect that bound. No calls are spent by this specification. Report actual cost rather
than equating the bound to spend.

## Decision rule

All are required, using the repository's Camera A/B decision rule:

1. Primary pooled B score exceeds A by at least **5 points**.
2. Primary full-success flips have **wins >= 2 × losses**, with **at least two net wins**.
3. Each guard's pooled B score is no more than five points below A, and its flip losses do not exceed wins.
4. No excess harmful belief writes per writer versus A; full audit coverage and all scheduled pairs are resolved.

A safety incident stops the batch and fails adoption. A rejection with positive pooled delta permits one fresh,
separately pre-registered re-test on layouts never previously run with this lever. Never repeat these camera draws
for a new decision. Passing qualifies only the frozen stack; defaults remain off until a separate lead decision.
