# IMIT camera A/B pre-registration

Hypothesis: resolving equal-cost twin swaps before the first pick reduces wrong-identity grasp plans and raises
imitate_sorting_sequence's transition score. This is a proposed run, not a completed measurement.

Timing: only after the lead releases compute4 from D-C19 (nominally 6 Nov 06:00Z; time alone is not release).
No launch or preparation writes on compute4 while sealed. No sealed artifacts are inputs to this experiment.

## Arms and frozen inputs

Use one reviewed PR commit for both arms and the same rendered candidate stack, camera calibration, vocabulary,
segmenter, backgrounds, gripper source, planner, skill settings and step limit. Save their config JSONs and hashes
before launching. Both: `imitate_pairing=one_to_one`, `imitate_wait=home`, `imitate_keep_sides=true`,
`imitate_look=ambiguous`. Keep `imitate_repair` at the frozen candidate value in both arms.
A: `imitate_tie_look=false`. B: `imitate_tie_look=true`. No other change, no parallel repair ablation.
Use genuine Isaac 5.1 cameras, not 6.0 proxy images. Use `cx/gpt-6-luna` for every agentic/grounding call in both
arms, with the same reasoning setting and call limits; no Claude calls are authorized by this spec.

Primary ids: exactly 100030-100049, 2 repeats per id, 40 episodes per arm. All are dev/training, not measurement.
Freeze layouts and demonstration hashes before running either arm. Preflight demonstration existence and validity
with the policy arms held on compute3. If any of these ids lacks a valid demo, prepare and validate it on compute3
using the existing generator before the freeze; do not silently substitute ids, salt variants, or demo orders.
If this cannot be done, report the experiment blocked rather than changing its sample after seeing scores.
No 100005-09, 100020-29, or ids >=100100; no measurement episode diagnosis.

Within each id/repeat run A/B then B/A on alternating id+repeat parity, sequentially under the shared GPU lock.
Guard cells: imitate 100002 and 100004 (2 repeats each per arm), plus classify_objects and stack_blocks 100030-34
(1 repeat per arm). Guard ids are not in the primary statistic. Freeze guard layouts too. Missing guard demos block
launch. These detect damage to known imitation successes and accidental activation outside the copy skill.

## Audit and reporting

Record official transition score (0-100), success, env steps, first-copy step, support-home verdict, copied prefix,
look labels/count/steps, reassigned twins, failed picks, agentic calls/tokens and cache reads. Keep all rows, including
failed runs. Retry only a classified infrastructure failure once with identical inputs; publish both attempts and
exclude a pair from the primary paired estimate only if the retry also fails infrastructure. A checker-invalid,
wrong-order, timeout or budget exhaustion is an outcome, not infrastructure.

Before adoption run `scripts/perception/m95_score.py audit` on matched physics replays with GT-at-write for both
arms, including look/refine writers (>2 mm added 3-D error is harmful). Publish harmful/helpful counts by writer;
missing audit coverage fails the gate. Do not use audit GT to change policy or tune thresholds.

Publish per-id mean scores over repeats, paired differences, n, spread, successes, and a paired bootstrap 95% CI
(resample layout ids, keep repeats together; 10000 resamples, RNG seed 0). No optional early stopping for scores.
A safety incident stops the batch and fails adoption. Make a success and a typical failure video using the existing
video tooling when camera runs exist; no videos can be produced from this offline pre-registration.

## Adoption rule (all required)

1. B improves primary mean transition score by at least 10 points and the paired CI lower bound exceeds zero.
2. No excess harmful belief writes per writer versus A, per the repository's perception adoption rule.
3. Guard imitation mean score and full-success count do not decrease, and neither non-imitation guard task loses
   mean score. No increase in premature-start query failures or budget exhaustion counts.
4. All scheduled pairs have usable results; unresolved infrastructure pairs require completing the same frozen
   schedule before a decision. All resource/call costs are reported.

Passing qualifies this switch for the tested cx stack only. It does not establish an improvement on the 8.8 board
cell if that cell uses another model/config. The lead must separately authorize and pre-register a confirmation on
the intended production route after the reserved budget is released. No adoption or tuning from scoring ids.
