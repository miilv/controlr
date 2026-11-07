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
