# 2026-10-28: the grasp-time wrist check (research R1, experiment E1)

Stream W of the perception-to-95 % push (goal and metric: stream M's grasp-time metric, `dojo/perception/grasp_metric.py`).
The question: the wrist camera at the pre-grasp / pre-place hover is our only BOP-resolution view (~0.3-0.5 mm/px; the
head ~2.4 mm/px). Does checking the belief there turn grasp / place events clean?

## What was built (behind flags, default off)

- `dojo/perception/wrist_check.py`, `WristCheck.check(frame, belief, arm, kind, target, reference, corridor, held)`.
  - **Which objects:** the target, a place's reference, and every believed object whose footprint lies within 10 cm
    of the approach corridor.
  - **Existence (wrist view):** the hover refine (`refine.refine`) plus a SAM 2.1-S click at the projected belief.
    - `ok` = the fit holds; `present` = a mask agrees with the predicted silhouette.
    - `phantom` = no fit, no agreeing mask, and a table-coloured silhouette.
  - **Existence (head view):** objects the wrist does not see are checked against the head's static-background
    foreground, the same backgrounds B1 uses.
  - **Pose:** an ok fit moves a neighbour or the reference. The target keeps the runtime's own hover refine.
  - **Relocation:** a phantom in view gets the refine's radius search (PR #183). Unexplained foreground blobs in the
    corridor get believed labels fitted at them.
  - **Identity:** DINOv2 renders of the label group's candidates at the checked pose, scored MUSE-style (CLS + GeM
    patch-token cosine). A swap needs margin 0.05. The 180-degree flip is scored but not applied (`flip_check`).
- Supporting changes:
  - `embed_id.Embedder.tokens` (CLS + patch tokens);
  - `sam_masks.LocalSam` (the SAM server's models in process);
  - service op `wrist_check`;
  - `RuntimeConfig.wrist_check` = off | log | apply | drop, run in `_refine_hook` before the target's own refine.
- E1: `scripts/perception/wrist/e1.py` and `e1_report.py`.

## E1

Setup:
- Stream C's per-step replays of 13 recorded proxy episodes (hang_mugs 5, organize_table 5, stack_bowls 2,
  deposit_coin 1): 64 grasp / place events.
- The belief at each event is M's `BeliefTracker` over the recorded report and actions, scored with M's
  `score_event` against the replay's GT at k - 1.
- Hover frame: the runtime's own pick / place refine step (35 events), else the top of the final descent on the TCP
  track.
- Each event is scored independently: the corrected belief is not carried forward.
- Compute: compute3 light4, SAM 2.1-S + DINOv2-B in one process. 0 LLM calls.

Clean events, out of 64:

| arm | R5 | R10 | R15 | required | target | R10 fixed / broken |
|---|---|---|---|---|---|---|
| runtime belief | 41 | 38 | 38 | 43 | 43 | |
| + phantoms dropped | 41 | 39 | 38 | 43 | 43 | +1 / -0 |
| + neighbour / reference fits | 41 | 39 | 38 | 43 | 43 | +1 / -0 |
| + relocations | 41 | 39 | 38 | 43 | 43 | +1 / -0 |
| + DINOv2 identity and flip at margin 0.03 | 37 | 35 | 34 | 39 | 39 | +2 / -5 |

Failure reasons of the runtime belief at R10 (counts, out of the 26 failed events):

| reason | count |
|---|---|
| phantom | 14 |
| neighbour not found | 9 |
| neighbour off | 6 |
| held gross | 7 |
| held overshoot | 3 |
| held rot | 2 |
| held wrong object | 1 |
| target gross or overshoot | 6 |
| identity | 1 |
| other rotation | 3 |

### Why the gain is small

- **Held state:** 10 of the 26 failed events have a held-object failure (stream C's / H's lane), and most phantoms
  sit in those same events. Dropping the phantom leaves the event failed.
- **What the wrist cannot see:** at the place hovers of hang_mugs and stack_bowls, the neighbours project to
  0 px with > 90 % truncation: the held mug fills the view, or the rack is edge-on. All 6 "neighbour off 17-27 mm"
  are mugs on the rack after a release. Only the head sees them.

### What works

- Existence verdicts:
  - non-target phantom verdicts are 7 / 7 right, all through the head view (foreground share 0.005-0.07 at the
    belief);
  - the 3 wrong "phantom" verdicts are all targets, which are flagged, never dropped;
  - recall: 9 of 19 objects whose belief is > 50 mm from any GT object of their group.
- SAM 2.1-S clicks: 23 ms each, peak 0.8 GB VRAM. The CPU run without SAM called a real garage a phantom and relocated
  it 50 mm onto something else; with SAM that did not happen.
- Relocation by radius search found a 54 mm-off target mug at iou 0.91. It is still 18 mm off (tolerance 15).
- One neighbour bowl fit improved from 8.7 to 1.5 mm.

### Identity (24 objects with at least 2 candidates)

- The belief identity is right for 21. DINOv2 top-1 is also right for 21, with different misses.

| swap margin | changes | right | wrong |
|---|---|---|---|
| 0.03 | 3 | 2 | 1 |
| 0.05 | 1 | 1 | 0 |

- The one identity failure among the events (hang_mugs_100003 g236) is fixed only at margin 0.043: noise level.
- Flips: none of the 37 candidates needed one. Flips scoring +0.03..+0.07 were wrong 4-5 times (mouse, alarm), so
  `flip_check` is off.

### Cost per hover

| | median | p90 | max |
|---|---|---|---|
| total | 0.21 s | 4.4 s | 11.8 s |

- The tail is the radius search (up to 10.8 s) and the first DINOv2 / render warm-up.
- Without the radius search, every stage stays under 1 s.

## Decision (lead, 28 Oct)

- The hover check is not the lever on this set. It lands behind `wrist_check`, default off. `log` mode gives the
  verdicts as diagnostics.
- Stream W moves to the post-release pose of placed objects, the "neighbour off" class above.

## Landmine found

- A per-episode `Gallery` that got garbage-collected terminated the process-wide EGL display: `GLError 12289` on
  every later render. Keep pyrender renderers process-wide (DEVELOPMENT.md).

Data: compute3 `~/dojo-data/w-wrist/` (`e1_gpu_a.json`; `e1_cpu_v*.json` = no SAM / no GL).
