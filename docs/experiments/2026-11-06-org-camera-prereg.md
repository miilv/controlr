# ORG: low-hover and tilted-face camera A/B pre-registration

**Parked for lead review. No camera jobs or live model calls authorized here.**
The G3 development panel gains five score points, 3 score wins / 1 loss / 6 ties,
with no full successes and nearly doubled median steps. This is a transfer
hypothesis, not adoption. The separate lateral-route pilot is excluded.

## Frozen comparison

- A: the lead's frozen clean camera candidate, with `arrange_hover=off`,
  `push_approach=off`, `arrange_route=off`.
- B: identical except `arrange_hover=low`, `push_approach=tilted`.
  `arrange_route` remains off. No other runtime, perception or belief switch changes.
- Run both from one source containing the tested ORG implementation, freezing
  its exact SHA, full effective configs, candidate hash, layout/asset hashes,
  simulator/renderer/GPU versions and model settings before collection. A
  behavioral change requires a new registration before seeing camera outcomes.
- Official strict CAMERA path only. No GT/G3 poses or verifier oracle in camera
  control. RuleSupervisor executes skills; no planner/controller LLM. If agentic
  perception is part of the clean candidate, freeze `AGENTIC_MODEL=cx/gpt-6.1-sol`
  in both arms; no Claude route. Record actual calls/tokens/spend.

## Panel and execution

Gain cell **organize_table**, held-out **100020-29**, one repeat: **10 pairs**.
These records have not been inspected or tuned on by ORG. They are for future
measurement only. AB on even IDs and BA on odd IDs; fresh reset every episode.
Complete all pairs without early stopping, favorable repeats, substitutions of
low-scoring stable layouts, or retuning from outcomes. Keep the 1000-step limit.

Before any camera outcome, audit the frozen planner over all board cells for
other consumers of `arrange` / `push_to`. If any are found, revise this document
to name them and freeze **10 pairs per exposed guard cell** before collection.
Do not assume this instruction route is task-exclusive. Preserve the existing
belief-write audit in both arms; neither switch adds a belief writer.

Use the evaluator's existing infrastructure-failure classification. Repeat both
members of an affected pair once, retaining every attempt. A still-unavailable
pair blocks the decision until the complete registered panel is available.
The lead assigns a supported rendering host and normal camera lane separately;
ORG's physics light slots are not rendering authorization. No compute4 access
or model calls are performed by this implementation task.

## Decision rule: current score-based default, explicitly selected

This registration explicitly uses the current **score-based** rule despite its
November 6 filename; it does not use the grandfathered success-flip rule.

Require all of:

1. At least ten complete gain pairs and pooled B score **>=5 points above A**.
2. A score-pair win means B-A **>=10 points**, loss means A-B **>=10 points**.
   Require **wins >=2*losses** and **wins-losses >=2**.
3. Every exposed guard cell: B pooled score no more than five points below A,
   no more score-pair losses than wins, and no more full-success flip losses
   than wins. This states both interpretations of the guard flip protection.

Report n, per-arm mean/sample SD, paired score deltas and spread, score W/L/ties,
success counts/flips, steps/runtime, timeouts, infrastructure failures, actual
model spend, and harmful belief writes. More steps are an expected material
cost, not grounds to raise the task cap. A rejected positive-delta lever gets
at most one fresh preregistered retest on IDs it has never run on; no replay of
the rejected panel to seek adoption. Lead review remains required; no merge or
camera adoption is authorized by this document.
