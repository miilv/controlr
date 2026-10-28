# 2026-10-28: put_bottles_into_dustbin deep-dive and the put_into options (stream H)

## Diagnosis (0 LLM)

**Rendered arm 8 (ON + OFF, 10 episodes): every episode fails step-bound** (634-682 of 700 steps; 0-3 of 4 bottles end
in the bin).
- **The drop check is wrong in mode B.** 13 of 18 releases over the bin put a bottle in, but 8 of 10 "place_miss"
  verdicts are false.
  - `library._inside` reads the belief after the drop rule.
  - `drop_rest` skips fixtures whose top is below the table (`TABLE_Z - 0.05`). The dustbin (top 0.675) is one, so
    the bottle is believed on the floor beside it, outside the bin's box.
  - The runtime then re-picks bottles that are already in (17 of 79 grasps). 3 of the 4 "no arm can pick" endings
    follow a false miss.
- 23 of 79 closes are on air: the belief is 40-180 mm off, repeated at the falsified pose (C1, W's).
- 7 harmful pick-refine recentres; see 2026-10-28-dojo-recentre-vis-guard.

**Mode gt (stream M, main 08b0dd7): only 1 of 5 succeeds**, so the skill is the ceiling.

**G3 mode gt on v1 draws of 100030-39** (`--trace`): 1 of 9 succeeds (100035 is unstable).
- **The grasp itself is not the problem.** A fresh layout, one pick per standing bottle
  (`scripts/skills/h95_pick_screen.py`): side grasps succeed 28/28 at h 0.5 / 0.65 and 27/27 at h 0.3 / 0.4, with no
  tips, no pushes and no neighbour moved.
- **In the episode the "slips" are knock-overs before the close.** The target moved 62-130 mm and tilted 21-74 deg
  during the approach. The picking arm lagged 34-96 mm behind its command: it was blocked by the other arm working
  100-150 mm away at the middle of the table, where `put_into` hands bottles over.
- **Early endings with the budget left:**
  - "handover spot" at steps 48 / 121: `free_spots` searches only a 30 x 30 cm patch, and with no spot the program
    fails;
  - the tries / round caps (5 of 10 episodes, 300-420 of 700 steps left).

## Built (flags, default off = today)

- `RuntimeConfig.put_into_opts` (`Skills.put_into_opts`; G3 `--put-into`):
  - `spot`: when the handover patch has no free spot, search spots anywhere both arms reach, else put the bottle back
    where it was picked (it was clear there; not counted as a failed try);
  - `patient`: 3 tries per object and 6 x n rounds instead of 2 and 3 x n, while the budget lasts;
  - `sequence`: when the two arms' pick plans of a round come closer than `Skills.parallel`'s 0.12 m plus the widest
    picked object's extent (fixed a priori), the helper waits a round.
- `RuntimeConfig.drop_verdict` (`Skills.drop_verdict`): `belief` (today) | `release`.
  - With `release`, a post-release measurement decides when there is one (belief source "perception").
  - Else the drop is in when the jaws held the object up to the release (`RuntimeProvider.holds`) and its centre was
    over the container's footprint at the release (`plan_drop` keeps `release_T`). The belief then puts it on the
    container's floor (`RuntimeProvider.place_inside`, event `drop_inside`).
  - Mode A / gt keeps the measured world. Unit tests use the false-miss numbers.
  - It only shows in mode B, so G3 cannot test it: the rendered arm does.
- The `grasp_slip` note now names its grasp, e.g. `[side h 0.50 head 49 pitch 20]`.

## G3 screen (mode gt, 0 LLM, compute3 light2): v1 draws of 100030-39, 9 stable layouts

| seed | base | all | no spot | no patient | no sequence |
|---|---|---|---|---|---|
| 100030 | 0.0 plan | 0.4 plan | 0.4 plan | 0.4 loop | **1.0** |
| 100031 | 0.1 handover spot @121 | **1.0** | **1.0** | **1.0** | **1.0** |
| 100032 | 0.25 loop | 0.25 plan | 0.25 plan | 0.25 loop | 0.25 plan |
| 100033 | 0.25 loop | 0.25 plan | 0.25 plan | 0.1 plan | 0.4 loop |
| 100034 | 0.0 handover spot @48 | **1.0** | 0.0 handover spot @48 | 0.4 loop | 0.25 loop |
| 100036 | 0.4 loop | **1.0** | **1.0** | **1.0** | **1.0** |
| 100037 | 0.25 loop | 0.25 plan | 0.25 plan | 0.25 plan | 0.4 plan |
| 100038 | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| 100039 | 0.4 loop | **1.0** | **1.0** | **1.0** | 0.4 loop |
| **success / mean** | **1/9, 0.29** | **5/9, 0.68** | 4/9, 0.57 | 4/9, 0.60 | 4/9, 0.63 |

- Each option rescues about one episode.
- `sequence` is mixed: it rescues 100039 / 100034 and costs 100030.
- n = 9 is a screen; the rendered arm is the evidence.

Spend: 0 LLM calls. GPU: about 6 G3 jobs of 3-8 min on compute3 light2, outputs < 50 MB.

## Other put_into callers, and the rendered arm

- Only put_bottles reaches `put_into`'s drop + handover in any saved G3 / integ run. The on-table baskets of the
  classify tasks have no off-table drop point (`drop_points`), so `put_into` returns at once there. The options touch no
  other task today, and no other G3 smoke was needed.
- Rendered arm (lead, passed to Integration): v0 of 100005-09 + 100020-29, one sha (main after #244 and #250).
  - BASE = the current candidate: submit_v3 + grip_check (DOJO_GRIP_SOURCE=v2, grip_official_key as in arm 8 ON).
  - ALL-ON = BASE + `put_into_opts` [spot, patient, sequence], `drop_verdict` "release", `refine_recentre_guard`
    "vis"; optionally ALL-ON minus sequence.
  - Tasks: put_bottles + guards organize_table, stack_bowls, build_tower.
