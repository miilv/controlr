# PEN: inset-packing camera A/B pre-registration

This supersedes the earlier centered-fit specification for the proposed camera run; neither specification has been run. It specifies a future rendered validation; no camera job is authorized or run by
this stream. G3 tuning uses only 100000–04 and private variant-2 draws 100030–39.
The frozen candidate is the combination below, not either rejected historical
lift-only arm. If its final paired G3 mean does not exceed A by 5 points, park this
specification without spending camera calls. Changing B requires a new spec before
camera data are inspected.

## Hypothesis and arms

Narrow horizontal shaft pinches with the original grasp fallback, a stationary
holder, inset-wall drop fits, and a corrected release inside measured joint limits
increase the fill_pen_holder ceiling and transfer to camera-only belief.

- A: all four new switches `off`, historical fill switches at defaults.
- B: `fill_grasp=rolled_tilt`, `fill_holder=stay`, `fill_drop=inset`,
  `fill_insert=safe`; every other field equal to A.
- Freeze the PR code commit, complete runtime configuration, camera/perception
  configuration, catalogue hashes, GPU/runtime versions and layout hashes in the
  launch manifest before either arm. Run both arms from that same code commit.
- Use camera-only runtime belief, no GT/refine oracle or G3 hook. RuleSupervisor
  runs the skill; no planner/controller LLM. If agentic camera perception is used,
  pin `AGENTIC_MODEL=cx/gpt-6.1-sol` identically in A/B; no Claude-route calls.
  This comparison does not establish parity with a differently configured board row.

## Layouts, order, budget

Gain cell: fill_pen_holder, 10 paired layouts, measurement IDs 100020–29,
one episode per arm per layout, existing measurement records unchanged. These
records are not inspected during tuning. Alternate AB/BA by even/odd ID. Reuse
neither this candidate's camera outcomes nor failures for tuning. No optional
stopping or replacement of low-scoring stable layouts. Infrastructure failures
repeat both arms on the same ID with the cause recorded; report all attempts.

There are no guard cells: these switches act only in fill_container. Run the
routing/skill unit tests and confirm only this task uses that route before launch;
if another task shares it, add at least 10 paired guard layouts for that cell in
a revised registration before any camera run.

Budget: 20 rendered episodes, zero planner/controller calls; perception calls and
tokens are reported from actual logs. The lead schedules camera lanes separately;
PEN's physics-only compute6 locks must not be repurposed for rendering.

## Decision (default Camera A/B rule)

Adopt only when B's pooled score is at least 5 points above A, success flip wins
are at least twice losses, and net success wins are at least two. Report mean,
sample SD, per-layout pairs, success counts, flips, runtime and failure classes.
Any guard cell must lose at most 5 pooled points and have no excess flip losses.
A rejected lever with positive pooled delta gets at most one fresh pre-registered
retest on previously unused layouts; never rerun the same IDs to seek adoption.
No belief writer is added by this lever; keep the existing perception configuration
and belief-write audit unchanged in both arms.
