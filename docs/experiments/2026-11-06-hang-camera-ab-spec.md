# HANG camera A/B pre-registration

Prospective protocol; no camera data have been collected for these levers.
Do not dispatch unless the final frozen paired G3 report shows at least a five-point mean score gain
in each cell, with no decrease in full successes, and the lead reviews the
candidate. A failed G3 prerequisite
means defer, not a camera adoption claim.

## Arms and inputs

Use one reviewed implementation commit in both arms. A disables
`hang_half_turn` and `hang_handle_grasp`; B enables both. This tests the bundle;
individual contributions require separate experiments. Preserve all other
runtime settings, step limits, perception, vocabulary, segmenter, background,
calibration and physics. Freeze complete configs, source, asset and layout
SHA-256 manifests before any outcomes. Use genuine Isaac 5.1 camera images.
Every model route must be `cx/gpt-6-sol`, with identical reasoning, retries and
limits across arms; explicitly pin agentic and held-look routes. No Claude
calls. No verifier, GT provider or G3 hooks in camera policies.

Gain cells: hang_mugs and hang_mugs_random, 10 distinct fresh layouts per cell,
one repeat per layout per arm. Generate variants in owned 100000-04 namespaces
(variants 21 then 22 per id, in increasing id order). If settling exclusions
leave fewer than ten, use variants 20 then 19 in the same order until ten
stable draws exist. Record all exclusions before G3 or camera inspection. Do not reuse layouts used in this investigation.
Select by settling/stability only; freeze the first ten stable draws per cell
and log every exclusion. No 100005-09, 100020-29, >=100100 or D-C19 inputs.

Guard cell: store_laptop_and_headphones, 10 fresh layouts, paired once. This
shares hang_on_hooks but uses the cradle path, on which both switches should
be inactive. Generate and freeze with the same rule. Alternate AB/BA ordering
by pair index; complete all 30 pairs before deciding. Only infrastructure
failures may retry once with identical inputs; retain both attempts. Skill,
checker, budget and timeout failures are outcomes. Unresolved infrastructure
pairs block the decision rather than silently reducing n.

## Decision rule

Apply the default Camera A/B rule independently to each gain cell: at least
10 usable pairs, B pooled score >= A +5 points, success flip wins >= twice
losses, and at least two net wins. On the guard, B pooled score >= A -5 and
flip losses <= wins. A positive-delta rejection permits one future retest
only, under a new registration on never-used layouts; never replay the same
ids to improve the verdict.

Report per-pair official scores, successes, steps, n and sample SD; paired
bootstrap 95% intervals over layout pairs (10,000 draws, RNG seed 0) are
descriptive, not a substitute decision rule. Report handle grasps, no-plan
exits, carry losses, insertion/release failures, model calls/tokens/spend,
transport attempts and lane time. Produce a success and typical-failure
video after camera data exist. Any perception/belief-writing change in the
frozen stack must also pass the repository harmful-write audit (>2 mm added
3-D error); these geometry switches add no writer. No policy access to audit
GT. Budget estimate: 60 camera episodes, roughly 10–20 lane-hours at 10–20
minutes per episode; record actual model usage rather than infer it from time.

This spec neither launches jobs nor authorizes compute4 access.
