# PEN: conditional-profile camera A/B pre-registration

**Parked. No rendered episodes or live model calls have been run by PEN.**
This replaces the proposed arm in the
[parked broad-profile specification](2026-11-06-pen-inset-camera-prereg.md).
It is a new pre-registration because the conditional arm differs. The lead
schedules any future camera run separately; physics light locks are not camera
lanes. No reserved measurement records have been inspected for tuning.

## Hypothesis and frozen arms

Holder pose and shaft geometry distinguish scenes that benefit from keeping the
holder stationary with inset insertion from scenes that need baseline packing
or relocation. The same condition evaluated on camera belief may improve score
without losing the baseline's successful paths.

- A: the lead's then-frozen baseline candidate, `fill_guard=off` and all four
  independent PEN switches off. Keep historical fill settings at that baseline.
- B: identical to A except `RuntimeConfig.fill_guard="geometry"`. The four
  independent PEN switches remain off in the resolved config. The scoped profile
  is `rolled_tilt` + `stay` + `inset` + `safe` only when the production geometry
  condition passes; no layout, asset or task identifiers select it.
- Freeze code SHA, candidate SHA256, complete effective configs, asset/layout
  hashes, renderer and GPU/runtime versions before either arm. Run both from
  the same deployed source. Any behavioral change to B requires another spec
  before its camera outcomes are inspected.
- Use the official strict CAMERA path and the same perception/belief settings
  in both arms. No GT, G3, refine oracle or replay poses enter camera control.
  RuleSupervisor executes the skill; no planner/controller LLM. If agentic
  perception is used, freeze `AGENTIC_MODEL=cx/gpt-6.1-sol` identically; no Claude
  route. Record actual perception calls and tokens, rather than estimating cost.

## Panel, execution and guards

Gain cell: **fill_pen_holder, IDs 100020–29, one repeat, 10 pairs**. These are
measurement-only records; no tuning or diagnosis on their individual episodes.
Alternate AB on even IDs and BA on odd IDs, fresh reset for each episode. Do not
stop early, add favorable repeats, substitute low-scoring stable layouts, or
retune from camera outcomes. Use the evaluator's existing infrastructure-failure
classification; repeat both members of an affected pair once, recording all
attempts. Unavailable pairs prevent adoption until the registered panel is complete.

The planner's “all things into holder” route is generic. Before launch, audit
all board task plans with the frozen planner/catalogue. If any other cell reaches
`fill_container`, register **10 paired measurement layouts per such guard cell**
under a revised specification before any camera outcome is inspected. Do not
assume guards are unnecessary merely because this development cohort contains
pens. No behavior in other skill entry points is changed by this switch.

Base budget is **20 rendered episodes**, plus any pre-registered shared-route
guards; zero planner/controller calls. Keep the existing perception configuration
and standard belief-write audit identical in both arms. This lever adds no belief
writer. Report condition activation and holder tilt as diagnostics; they do not
replace strict scoring or the acceptance rule.

## Decision: default Camera A/B rule

Adopt only if all conditions from `docs/EXPERIMENTS.md` hold:

1. At least 10 gain pairs; pooled B score is **at least 5 points above A**.
2. Success flip wins are **at least twice losses**, with **at least two net wins**.
3. Every guard cell loses at most 5 pooled points, with no more success flip
   losses than wins.

Report per-arm n, mean and sample SD, success counts, paired score wins/losses/
ties, success flips, runtime/step caps, infrastructure failures, actual model
calls/tokens and harmful belief writes. G3 score gains alone never pass this
rule. A rejected lever with positive pooled delta gets **at most one** fresh
pre-registered retest on layouts it has never run on; never rerun the same IDs
to seek adoption. No camera adoption or merge is authorized by this document.
