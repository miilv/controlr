# 2026-11-07 — ORG2 camera A/B pre-registration (organize_table gain, seven guards)

**Status:** camera slot approved by the lead (7 Nov); registered on main before any queue write. Evidence:
[ORG2 G3 report](2026-11-07-org2-g3.md) (registered fresh G3 panel, 18 pairs: 36.1 -> 58.3, 12 W / 2 L; rule passed,
bar 85.0 not reached).

**Hypothesis.** The ORG2 general capabilities raise organize_table's camera score over the frozen candidate without
costing any cell their code reaches.

## Arms

One code: merged main `83887dae2ee50d582520b60d13191279a5a58f32` (contains #430), deployed to
`compute6:~/dojo-integ-c6-83887dae` by the IMIT2 / MATCH2 recipe ([deploy.sh](data/org2-camera/deploy.sh): venv
parity `true`, empty diff; catalogue gate OK). Same stack, calibration, segmenter, backgrounds and step limit on every
row.

- **A**: the frozen candidate `scripts/dojo/queues/2026-11-06-candidate.txt`, unchanged.
- **B**: A + the eleven fields of [panel-B-attrs.txt](data/org2/panel-B-attrs.txt) (`push_contact=pair`,
  `push_plan=sequence`, `push_goal=clear`, `arrange_retry=once`, `place_surface=mesh`, `grasp_torso=on`,
  `stand_offset=base`, `retreat_axis=on`, `plan_world=objects`, `place_via=clear`, `slip_avoid=on`): the registered G3
  panel's B. The resolved RuntimeConfig differs from A in exactly these eleven fields (checked locally; both construct).

Every row: `--setenv AGENTIC_MODEL=claude/claude-sonnet-5-5` (checked on every row by `append.py` before the write), a
private salted `DOJO_DATA`, `--renderer isaac51`, 25-minute episode and job caps, the existing compute6 lane.

## Cells and reachability (why no parity guards)

| Cell | Role | Pairs | Changed code it reaches |
|---|---|---:|---|
| organize_table | gain | 12 | all eleven (multi-clause place / push instructions go to `arrange` through the general planner) |
| arrange_largest_number | guard | 10 | the general four (below) on its tile transfers; tiles are not irregular, so no torso grasps |
| arrange_largest_number_random | guard | 10 | the same |
| store_tools_in_toolbox | guard | 10 | the general four, torso grasps / squeeze on irregular tool meshes |
| fill_pen_holder | guard | 10 | the general four, torso grasps / squeeze on pens where irregular |
| insert_key | guard | 10 | the general four, torso grasps / squeeze on the key |
| classify_objects | guard | 10 | the general four, torso grasps / squeeze on irregular toys |
| build_tower | strong guard | 10 | `plan_world`, `slip_avoid`, `retreat_axis` |

The **general four** reach any cell: `plan_world=objects` replaces the free-space transit planner (`Motion.to_pose`)
in every motion; `slip_avoid` acts on any slipped pick; `retreat_axis` on any side-approach place; `grasp_torso` adds
candidates, the mesh-parts collision model and the squeeze only for irregular meshes. The **arrange-only** seven
(`push_contact`, `push_plan`, `push_goal`, `arrange_retry`, `place_surface`, `stand_offset`, `place_via`; callers
checked by grep on `83887dae`) act only inside `library.arrange`, which in this panel only organize_table's instruction
reaches. Because `plan_world` reaches every cell's motions, **no guard can be proven parity**: all seven are
camera-gated, as the lead specified.

## Layouts (fresh salt; prepared on compute3)

[design.json](data/org2-camera/design.json): salt `org2-cam-20261107-b5e37b4b7f230e7873ba6daa18ff1778` (SHA256
`2674f92c…c806cc73`), fixed in git before generation. Ids **100000-09 and 100020-29 only**. Unchanged
`dojo.simlab.gen_layouts --seed-salt` and default two-pass presettle from the same merged main, compute3 light7 /
light8 only, one job at a time ([prep.py](data/org2-camera/prep.py)). Per cell, the first `pairs` ids whose settled
status is `ok`, in ascending batch order 100000-09, then 100020-24, then 100025-29; a refill batch is generated only
when a cell is still short (the earlier refill rule; insert_key is expected to need it). No outcome-based selection.
A cell still short after 100029 runs with the ids it has, reported. The block is copied to
`compute6:~/dojo-data/org2-cam/` and SHA256-verified file by file before staging.

## Order and queue

Per id, the two arms back to back, AB on even ordinals and BA on odd ones (ordinal over all cells in the table order).
Appended at the **end** of `~/dojo-integ-c6-state/queue2.txt` under `wave-start.lock` then `queue2.txt.lock`, after
every row already queued (IMIT2's overnight block), never reordering ([append.py](data/org2-camera/append.py)); the
receipt `dispatch.json` is committed next to the manifest.

## Decision rule (EXPERIMENTS, 7 Nov, score-based)

1. **Gain** (organize_table): pooled B >= pooled A + 5 points; score wins (B - A >= 10) >= 2 x losses, net wins >= 2.
   Success flips reported alongside.
2. **Guards** (all seven camera-gated): B pooled no more than 5 points below A and no more success-flip losses than
   wins. build_tower is the strong guard (crashes and infrastructure regressions).
3. **Write audit: N/A by proof** (writer inventory). The only belief-touching call #430 adds is
   `Skills.confirm_moved(label, T_expected, "push")` in the planned push path, the same writer and the same call legacy
   `push_to` makes in A; the push planner's virtual moves (`o.pos` in `plan_sequence` / `covers`) are restored in
   `finally`, the pattern of the existing `_plan_leg`. Torso grasps, stand positions, vias, `slip_avoid` (a grasp
   memory), `Collision.parts` (a collision model) and `retreat_axis` write no belief. `base_refit` is not wired.
   The m95 audit is therefore not run for this A/B; any belief-write event kind seen only in B fails the gain.

Infrastructure failures (rc 4, 0 steps) are retried once with identical inputs by the lane; a pair whose retry also
fails infrastructure is excluded and reported. A safety incident stops the batch and fails the A/B. No reruns on
these ids.

## Reported

Per cell: n, mean / sample SD, paired deltas, score W / L / ties, success flips, steps, wall, infrastructure
exclusions; organize per item (mouse / keyboard / figure / alarm, `org2_stages.py` on physics-only replays where
available) and the camera census items (keyboard asset, mouse first read).

## Budget

**164 camera episodes** (82 pairs: 12 gain + 7 x 10 guards). Time: ~400 s per episode (organize ~420, others
300-420; `plan_world=objects` adds planning time), ~18 lane-hours, ~9 h on two lanes after the rows ahead.
Claude route (`claude/claude-sonnet-5-5`): at most 3 agentic calls per episode (`agentic_max` 3) -> **<= 492 Sonnet
calls**; expect ~250-400 (1-3 per episode). Prep: 0 calls. Stop at any HTTP 429 (the lane policy), and note that the
Claude route has its own weekly limit (incident 2026-10-27).

## Reference

REPRO on our canonical 100000-04 (camera, fixed programs): organize_table 85.0, 3/5 successes. Different layouts;
context, not a gate.

## Receipt (written after the queue write; nothing above was changed)

- Pre-registration commit on main: `4b2aa45f` (before generation of the camera inputs' staging and before the write).
- Inputs: [selection](data/org2-camera/selection.json) — organize_table 12 ids (15 of 20 settled ok), insert_key 10 (11
  of 20 ok, both refill blocks used), the other six cells 10 of 10 from 100000-09; 0 model calls in prep (compute3
  light7, one job). Block copied to compute6 and verified file by file: 195 files ([block.sha256](data/org2-camera/block.sha256)).
- Every row constructs a `RuntimeConfig` on the deployment's venv; one distinct config per arm; resolved B - A = the
  eleven registered fields. Route `claude/claude-sonnet-5-5` on all 164 rows.
- [dispatch.json](data/org2-camera/dispatch.json): inserted 2026-11-07T20:26:07Z, **positions 82-245** of 245, after
  all 81 existing rows (IMIT2 block), appended at the end; queue SHA256 before `f1f62e94…` / after `8c6ec76d…`; new
  rows `cdd572ca…` ([queue-org2.txt](data/org2-camera/queue-org2.txt)); manifest `0a15c5c5…`
  ([manifest.json](data/org2-camera/manifest.json)). Rough estimate: start ~03:07Z, finish ~10:42Z on 8 Nov (two
  lanes; admission, retries and power drains can delay).

## ORG2SPLIT preregistered amendment — 2026-11-08 05:05Z

The lead authorized moving the following **80 rows / 40 whole pairs** to compute4 to halve
remaining wall time when COIN2-CAM drains. Exact immutable row arguments and names:
[rows-source.txt](data/org2split/rows-source.txt). Largestnum, largestnumrand and storetools:
all A/B rows for 100000–100009; insertkey: both arms for 100003, 100004, 100005, 100007,
100009, 100020, 100022, 100024, 100025, 100026. No selected row has a start/done entry
in either compute6 lane log at selection; no episode data from these rows exists yet.
Pairs remain wholly on one box. Organize gain, buildtower strong guard, classify and fillpen
remain on compute6. No outcome-driven selection or inference/config change.

Commit this amendment before deployment/copy/queue mutation. Deploy the same merged code
83887dae to compute4 by CLASSIFY3/KONG, independent copied venv, source/archive identity,
identical freeze, candidate parity true/diff empty and matching catalogue gate.
Fail closed on **any** file-by-file SHA256 difference in calibration/background folds.
Copy only runtime-required input files for the selected layouts (no outputs), SHA256 both ends.
Only authorized row substitutions: --repo to /home/haptics/dojo-integ-c4-83887dae;
/home/robotics/ to /home/haptics/ in org2 DOJO_DATA, layout paths and DOJO_COLOR_CALIB.
Keep family names unchanged; all other bytes unchanged, verified by per-row normalized diff.
Retain compute4 per-job Kit environment and own physics kit as required by the existing recipe.
Append under both queue locks after COIN2-CAM, replacing an empty/blank queue if needed.
The lead owns lane starts. Stop without moving if checks fail or readiness misses 07:40Z.
The earlier blocker receipt is superseded only within the lead's explicit extended permissions.

### Operational overlay clarification — 2026-11-08 05:10Z (before queue mutation)

The lead additionally authorizes appending exactly
`--setenv BASH_ENV=/home/haptics/dojo-data/org2-cam/meta/kit-env.sh` to each moved row.
This is a box-level operational setting applied to **both arms of every moved pair**;
whole pairs stay on one box, so it cannot differ within a pair. The overlay is copied
byte-identically from CLASSIFY3 compute4's `meta/kit-env.sh`; its SHA256 is in the receipt.
The per-row normalized diff records this addition separately from authorized host-path changes.
No other row change is permitted.

## ORG2SPLIT dispatch receipt — 2026-11-08 05:11:12Z

Amendments `a43e1154` and `002f9426` were on main before queue mutation, well before 07:40Z.
**80 rows / 40 pairs moved**, exactly [source](data/org2split/rows-source.txt) →
[target](data/org2split/rows-target.txt); [per-row normalized diffs](data/org2split/row-diff.json)
are empty, with the separately authorized BASH_ENV addition on both arms. Family names unchanged.
No start/done entry for any selected row in either compute6 log, rechecked while holding its queue lock.

Deployment `/home/haptics/dojo-integ-c4-83887dae`: 1,808 source files match git archive
([source parity](data/org2split/source-parity.json)); copied 416a2a03 venv freeze identical;
imports resolve to the deployment. [Candidate parity](data/org2split/parity.json) true, diff {};
package versions identical. Both boxes' catalogue gates pass, md5
`2cd7455cbdce2ab38e4e9d586e28db61`. Compute6 candidate parity also passes with its explicit
lane background/calibration environment (an initial probe without DOJO_BG_FOLDS had a path-only diff).
All four model files SHA256-identical ([model parity](data/org2split/model-parity.json)).

All 96 background-folds/calibration files match byte hashes, no exceptions
([comparison](data/org2split/background-parity.json),
[compute4](data/org2split/compute4-backgrounds-sha256.json),
[compute6](data/org2split/compute6-backgrounds-sha256.json)).
Only the 40 referenced settled layout files were copied from compute6;
[per-file source/destination SHA256 manifest](data/org2split/inputs-sha256.json) matches.
Runtime eval reads each settled JSON directly (`dojo.runtime.eval._run`); these records are
self-contained layouts, with no per-layout companion files loaded from the stream. No run outputs copied.
compute4's own CLASSIFY3 `simlab.physics.kit` copied to the ORG2 stream's `simlab/kit/` and
`meta/`: SHA256 `03a4db8400b66195445548c88a1c8e2212487a3054ac1e1a95a02aa9e6172cd0`.
The CLASSIFY3 operational `kit-env.sh` is copied byte-identically into org2-cam/meta:
SHA256 `0023b748633125a441b1f4e1059b1675ccbb141faf35e7e0e289661a5a58634a`.
No environment/secret file was opened.

[Dispatch receipt](data/org2split/dispatch.json), by [move.py](data/org2split/move.py),
held **both queue2.txt.lock locks simultaneously** through both writes.
compute6: 147 → 67 rows; compute4: 3 → 83 rows, append after all three remaining COIN2-CAM
rows, no reordering. Full locked snapshots are saved next to the receipt.

- c6 before SHA256 `ff074a35d2550ba5ec0d1a4fe318d4eb66a6e01dd2c730956e7cf093dfd38503`; after `8d435ae651de2ab58ba9c84f99818f3b72b48a341b77daba1f2421f7cccf645a`.
- c4 before SHA256 `16ad83f6287b504d003d7346d27e0be5be3fc67ab857a3499fbe445d4e8bba4d`; after `00f2b0596f216fd013d338ec015e0275a260df317794901ab72a91f2d6663ffa`.

Route `AGENTIC_MODEL=claude/claude-sonnet-5-5` verified once on every moved row; actual call-log
route verification remains part of the episode audit when rows execute. All 80 configs construct.
Dispatch/deploy model spend **$0**. No lane was started, no running job stopped, no coord access.

**Lane handoff:** heavy1 (pidfile 1019493) is exited; log at 05:10:16Z says
`SHARED SIDE DOWN before c4b_tttsg_coin2camAr1_100003 (seg glyph glyphk): put back, every lane stops`.
Heavy2 (pidfile 1019925) is alive at inspection, on COIN2-CAM. Lead must address shared-side
readiness and owns restarts. Exact lead recipe (N = 1 or 2 as needed):

```bash
S=$HOME/dojo-integ-c4-state
LANE_REPO=$HOME/dojo-integ-c4-416a2a03 setsid nohup bash $S/lane2.sh heavyN $HOME/dojo-integ-c4-heavyN.lock >> $S/lane2_heavyN.log 2>&1 < /dev/null &
```
