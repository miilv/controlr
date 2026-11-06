# SEG additive v2n camera A/B pre-registration

Offline qualification: IoU 0.1 passed the registered dev gate with zero category/asset losers,
unchanged 35 phantoms and 24/49 naming hits. All three registered IoUs pass with identical
metrics; the lowest-qualifying-threshold rule selects 0.1.
[Completed offline evidence](2026-11-06-seg-v2n-additive.md).

Hypothesis: adding a separate new-class-only detector raises camera scores on variant
cells containing new assets while preserving strong old-class cells. This is a prospective
compute4 specification; SEG has made no compute4 connection, launched no episodes and
spent no LLM calls. Camera execution requires the lead's normal capacity release.

## Freeze and arms

Both arms use the same reviewed PR #387 deployment, same frozen candidate stack and
identical base v2b checkpoint. Record the exact code SHA and resolved runtime/perception
JSON before launch. Start with the tracked candidate in
`scripts/dojo/queues/2026-11-06-candidate.txt` (SHA256
`5cde3f9e684c0937f21a0f28609437e6d317a433e000bc171198349417cd398e`),
retaining its adopted settings, gripper source, readers, calibration and backgrounds.
Do not reconstruct the candidate from historical journal command snippets.
Set `AGENTIC_MODEL=cx/gpt-6-luna` in both arms and pin every explicit planner/agentic/grounding
model override to **cx/gpt-6-luna**, with identical reasoning,
limits and prompts across arms. No Claude/cc route is authorized by this specification.

A: immutable v2b, `seg_additive=false`. B: the same v2b plus v2n,
`seg_additive=true`, `seg_additive_iou=0.1`, `seg_additive_min_score=0.5`.
Explicit common estimator fields: `seg_propose=add`, `series_count=seg`,
`seg_add_move_weak=true`, `seg_add_refit_poor=true`; preserve candidate glyph/pad options.
There is no relabel arm. Extra proposals use exact public task candidates only.
Concrete estimator overlays: [A](data/seg-v2n/estimator-v2b.json) and
[B](data/seg-v2n/estimator-additive.json); merge the unchanged candidate reader settings
into both before launch, as described below.

Base: compute3 `~/dojo-w-ext/seg_v2b.pt`, MD5
`2f546196145fc424f76a5e2e47dd3210`, 511 classes.
Extra: compute3 `~/dojo-data/s-v2n-seg/runs/v2n/weights/best.pt`, MD5
`be0817051b8efd0565218033cb647cec`, 32 classes, names in this PR's
`docs/experiments/data/seg-v2n/classes.json`. Weight files remain on compute3 during
this task. The lead must stage and verify those exact bytes separately before dispatch.
Training selected epoch 27 by validation fitness; no camera score selects a checkpoint.

Load **both** checkpoints for both arms on one private server, changing only the request
config. Use `seg_twin serve --weights BASE --classes BASE_CLASSES --extra-weights EXTRA
--extra-classes EXTRA_CLASSES --sock PRIVATE_SOCKET`. A's disabled path returns v2b
unchanged. On compute4, candidate `--seg v2b` would redirect to the existing shared server,
which lacks v2n: remove that one queue option for these jobs, preserve the reader options,
and supply `--setenv DOJO_SEG_ADDR=PRIVATE_SOCKET` to both arms. Merge each arm's additive
fields into the candidate's complete `DOJO_PERCEPTION_CFG`; never replace its glyph/pad
settings with a partial object. Start/stop only the private server under assigned locks,
verify its info/weights and socket before episodes, and leave the shared servers alone.
The existing `with_seg_server.sh --extra-weights ... --extra-classes ... -- CMD` can own
that server's lifetime for a sequential block. This is a specification, not queue text
that can be pasted unchanged into lane2.

## Layouts and schedule

Use genuine Isaac 5.1 camera rendering on compute4. Generate **fresh salted v15 dev**
records: each task has two private blocks `a` and `b`, each IDs 100000–04, one repeat.
This gives ten distinct layouts / ten pairs per cell. Never use original gate records,
100005–29, training/calibration IDs 100030–49 or sealed IDs>=100100. Keep all loader refusals.

| role | task | salt prefix (append `-a` / `-b`) | pairs |
|---|---|---|---:|
| gain | make_toast_random | seg-v2n-camera-20261106-toast | 10 |
| gain | hang_mugs_random | seg-v2n-camera-20261106-mugs | 10 |
| gain | sort_nesting_dolls_by_size_random | seg-v2n-camera-20261106-dolls | 10 |
| gain | store_laptop_and_headphones_random | seg-v2n-camera-20261106-laptop | 10 |
| gain | pour_liquid_into_cup_random | seg-v2n-camera-20261106-pour | 10 |
| guard | stack_blocks | seg-v2n-camera-20261106-blocks | 10 |
| guard | insert_tubes | seg-v2n-camera-20261106-tubes | 10 |

These five gains exercise newly named toaster, holder/mug, dolls, laptop/headset and
wuliangye assets. Strong guards have recent paired camera scores 91.5→100 and 88→88
([public dev wave](../journal/2026-11-06-ab-wave-verdicts.md)); they are not selected from
new camera outcomes. Do not remove a gain because a downstream skill ceiling is low.
Report all five separately as well as the equal-weight pooled gain.

Generate and two-pass presettle each private block using existing tools; require every
scheduled layout's status `ok`. Freeze raw/settled SHA256+MD5, generator SHA, salts,
catalogue, calibration and backgrounds before running either arm. Before camera execution, verify each gain record has at least one of the 32 new assets;
otherwise revise the registration before running either arm. A failed settle blocks
the schedule; do not replace layouts after outcomes. Store an explicit complete schedule,
alternating AB/BA by (task index + block index + layout index) parity. Both members of a
pair run on the same box with identical frozen layouts. No score-based early stopping.
Use the lead's assigned camera locks, resource limits and admission/stagger policy.

## Outcomes and decision

Publish official score 0–100, success, steps, paired score deltas, per-cell n/spread,
success flips, perception errors, extra proposal counts, phantoms and runtime overhead.
Report LLM calls/tokens/cache and occupied GPU time. Keep every attempt. Retry a classified
infrastructure failure once with identical inputs and retain both attempts; skill errors,
timeouts, drops and budget exhaustion remain outcomes. Missing paired results leave the
verdict pending. Produce a success and a typical failure video from completed camera runs.

Apply the repository's **default Camera A/B decision rule**, without extra success-loss
requirements:

1. Pooled gain score B−A ≥5 points over the 50 scheduled gain pairs.
2. Gain success flips satisfy wins ≥2×losses and net wins ≥2.
3. On **each** strong guard cell, B−A ≥−5 points and flip losses ≤wins.

Report a paired bootstrap 95% CI (10,000 draws, seed 0), resampling distinct layouts within
cell; the CI is descriptive and does not replace the frozen rule. Report Generalization
contribution only as a descriptive weighted projection, never a new sealed board score.
A rejected positive pooled delta permits one fresh-layout re-test under a new registration,
as in EXPERIMENTS.md; never rerun these records for a new decision.

Budget: 70 pairs / **140 camera episodes**, plus at most one infrastructure retry each.
At 15 min/episode this is ~35 lane-hours (~17.5 wall-hours at two assigned lanes), before
settling/retries; the 1700 s episode cap bounds ~66.1 lane-hours. Preflight model-call bound:
50/episode =7,000 calls before retries, all cx route; stop and report if the common stack
cannot meet it. This spec has spent zero calls. Offline gate qualification authorizes
review of this A/B, not production adoption; defaults remain off pending the lead's verdict.
