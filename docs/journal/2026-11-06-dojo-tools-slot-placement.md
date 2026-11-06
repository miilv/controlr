# TOOLS: slot placement on v15 dev layouts

## Setup

Baseline source: `ebcbca694616e452071793f46735325b891291a7`. Physics-only G3 on compute6 (robotics),
deployed to `~/controlr-tools`, outputs exclusively under `~/dojo-data/tools`.
No LLM calls. At most two light-lane jobs; no heavy lock. Admission checks used VRAM <=24576 MiB,
disk >=10 GB; jobs scoped to 8 GB RAM / 600% CPU. Other streams share the machine.

Fresh `simlab/v15` records: 100000–04 (variant 0), 100030–39 (variant 1).
Generated with `dojo.simlab.gen_layouts.generate`, presettled with the default two-pass validator.
All 15 passed. No 100005–09, 100020–29, >=100100, compute4, or sealed files were used.
These are development measurements, not an update to a camera board cell.

Baseline G3 command (blocks of five):

```bash
-m dojo.skills.g3 run --task store_tools_in_toolbox --seeds BLOCK \
  --layouts ~/dojo-data/tools/simlab/layouts/store_tools_in_toolbox \
  --trace --out ~/dojo-data/tools/g3/base_BLOCK
```

The remote `job.sh` takes a free `~/dojo-gpu-light*.lock` before checking VRAM and starting the scoped process.
The extra baseline chain yields 90 seconds between blocks and bounds each job with `timeout -k 60 1800`.
Raw layouts, settled records/index, traces, actions and logs remain on compute6; local writes stay below 1 GB.

## Prior evidence

The store-tools journals report pad slip and slot reach as the remaining classes. Re-gripping a loose pick and
slowing the carry did not improve paired v15 dev results (2026-10-29). The current place re-measures at hover;
if its corrected tail fails it can reuse the nominal transform, and if descent is blocked it can drop from hover.
The baseline trace audit compares tool pose relative to measured TCP, separating carry slip from joint lag.

## Results and disposition

Initial completed rows: 100000 = 1.00 (success), 100001 = 0.50, variant 100030 = 0.75.
Of the 12 tools, three failed: two slipped during approach (pliers 288 mm / 25 degrees; hammer
104 mm / 50 degrees), one wrench slipped during lift (rose 42/80 mm). Hammer approach TCP lag
was <=3 mm, with joint 5 near zero: this is not the known joint-stop failure.

First lever: `slot_approach=linear` (default `off`). Use the existing checked Cartesian planner
for the hover transit, preserving interpolated TCP orientation along the path instead of a
joint-space trajectory. No slower-carry/regrip change. Infeasible lines reject a candidate without
joint-path fallback. Seven CPU tests cover dispatch, collision-check arguments, refusal,
level/reorienting propagation, default and config validation.

Work in progress; final paired results will be recorded before review. No default is changed and no camera gain is claimed.

### Completed first ten baseline rows

| v15 draw | score | success |
|---|---:|---|
| 100000 | 1.00 | yes |
| 100001 | 0.50 | no |
| 100002 | 0.50 | no |
| 100003 | 0.75 | no |
| 100004 | 1.00 | yes |
| 100030 v1 | 0.75 | no |
| 100031 v1 | 0.75 | no |
| 100032 v1 | 1.00 | yes |
| 100033 v1 | 0.50 | no |
| 100034 v1 | 1.00 | yes |

Mean 0.775, 4/10 successes. Of 40 tool opportunities, nine miss score credit:

- **Carry loss, 4:** pliers 100001/02 (288/245 mm hover shifts), wrench 100003 (272 mm), hammer 100030 (104 mm).
- **Tail/contact loss, 1:** wrench 100033: hover shift 11 mm; another 38 mm / 31 degrees while still closed in the
  tail; final error 32 mm xy / 28 mm z / 69 degrees.
- **Pick slip, 2:** wrench 100001/100031 (rose 42/48 mm instead of 80 mm).
- **No usable slot transfer, 2:** tape measure 100002 (relay second pick slipped, then no successful completion),
  hammer 100033 (134/148 left grasps reachable, zero grasp-place pairs; no in-air handover for a 228 mm object).

These are credited-tool losses, not raw failed-attempt counts: e.g. 100003 hammer's 27-degree verification miss
is still credited. The tape-measure class includes both planning and failed relay execution, not a pure IK failure.
The first-ten ceiling on this box is higher than the historical ~0.55; do not pool them across box/layout revisions.

### Cartesian screen: negative

Candidate physics code `83c2f587` (record stamp `0706dad9`, docs/tests only above that), 100030 v1:
0.75 -> **0.50**, no success. Pliers newly slipped 122 mm at hover; hammer still slipped (154 mm).
No expanded Cartesian trial is justified from this screen. The flag remains default off pending disposition.

Next independent lever: `slot_hover=low` (default off), pre_h 15 mm / pre_clear 3 mm instead of
40 mm / 15 mm. Approach stays collision-checked and joint-space by default. The minimum is still computed
from the container top and tool geometry; this is not permission to traverse its wall. Hypothesis: a lower
hover admits more stable grasp/place pairs and shortens the fall when a corrected descent is refused.
