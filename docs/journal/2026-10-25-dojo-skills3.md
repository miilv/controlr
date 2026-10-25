# 2026-10-25: dojo skills3: grasps that hold, closed-loop insertion, PyRUA's holes

Stream: skills3 (branch `feat/dojo-skills3` from `origin/feat/dojo-skills2` 929742c; code `dojo/skills/`, tests
`tests/dojo/test_skills3.py`). Targets in the lead's order: insert_key, plug_in_charger, store_tools_in_toolbox,
fill_pen_holder, organize_table. G3 = the TRUE state in the physics-only twin, on simlab's settled dev layouts.
One entry point (`planner.solve`), with skills chosen from the instruction words plus object categories and
geometry. No task names, no LLM calls. Runs on compute3 go through `scripts/skills/g3_chunks.sh` from
`~/dojo-skills3`, with the deployed commit stamped by `scripts/skills/deploy_skills3.sh` (`.deployed_commit`;
runs named `<task>_vN_<commit>`). Data: `compute3:~/dojo-data/skills3/g3/`.

## 1. What the skills2 traces showed (before any change)

| task | what the trace shows | consequence |
|---|---|---|
| insert_key v2 (6/6 ik_fail / collision) | The key turned 50 deg relative to the giver's pads during the carry (10-17 deg per step), then fell out. The giver held the bow, which is a cylinder collider (r 9.7 mm, 9.4 mm thick), squeezed across its diameter. The carry turned the key about the closing axis at 0.14 rad/step. | The taker's "ik_fail" came after the key was already on the table. |
| plug_in_charger v2 (8/10 topple) | The charger was released on the strip's surface, 24 mm above its target. The blades sat 3-10 mm beside the slots. There are two causes. (1) Under load the arm settles a few mm off its commanded joints: the in-hand pose measured with the real joints was right, so the error was the arm's. (2) The outlet frames sit 0.5-1 mm off the slot pair, with the blade tips at the surface (~2 mm deep). The checker wants the blade tips >= 15 mm below the strip's bbox top. | The two successes were lucky alignments in which the charger fell into the slots by itself, 13-17 mm deeper than the plan. |
| store_tools v3 | The pliers and the hammer were picked with "slice" grasps across the whole outline (61 / 76 mm wide). The fingers then closed on the pivot's slanted edges or on a round handle above its widest line. The pliers rose 15 mm and dropped; the hammer never lifted. | Every box grasp of a thin lying tool failed the collision check: the box generator's analytic fingertip estimate sat 0.8-1 mm below the collision model's fingertips, inside the 2 mm table margin. Only slice grasps were left. |
| organize v1 | Keyboard: fingertips 11 mm up (half the bbox height) passed over its 5 mm front edge, 14 strokes, no motion. The alarm was carried with a 40-deg tilted grasp and tilted or slipped. The figurine had no full pick plan next to the drawer. | |
| fill_pen v2 | After the first pen, the pens landed 8-47 mm off target, on the rim or on the first pen. That size of error matches the arm's settle error seen on the plug. | |

## 2. Generic changes

| piece | where | what |
|---|---|---|
| centre of mass | `grasps.mesh_com` | Volume centroid of a closed mesh (PhysX takes mass properties from the colliders at uniform density), else the surface centroid. A key's sits at the bow/shaft junction, 5 mm above the surface centroid. |
| hold quality | `grasps.pad_contact`, `hold_quality`, `lift_hold` | Rays along the closing axis on a 5x5 pad grid give the contact patch and its centre. The hold-quality score adds the gravity lever about the pads to the missing patch. `lift_hold` combines the lever with the contact depth along the approach. |
| CoM stations | `grasps.box_grasps` | Uneven objects (irregular, or CoM > 8 mm off the bbox centre) get grasps every 6 mm within 4 cm of the CoM along the long axis, from the top only. Irregular objects also get quarter points. |
| fingertip floor | `grasps.box_grasps` | Computed from the same finger model the collision check uses. |
| width rays | `grasps.mesh_width_at` | Rays now reach down to the fingertips. Before, a TCP raised above a thin lying part missed it. |
| big meshes | `world.AssetDB` | Meshes over 12 k faces are clustered on load (1.5 mm cells, larger until under the limit). Irregularity is decided on the full mesh before clustering. Tools, figurines and mice had 0.1-1.2 M faces, and every ray cast is O(faces). |
| transfer ranking | `Skills.plan_transfer` | Uneven objects' grasp-place pairs are ranked by score + 0.5 x hold quality. |
| in-air handover | `library.plan_air_handover`, `run_air_handover` | Planning: the giver's picks are ranked by `lift_hold` and the taker's grasps by hold quality. The best of up to 12 complete plans wins, scored by giver lift_hold + taker hold quality + steps/100. The final-place check is cached per grasp in the object frame. The hands-apart check runs on grasp poses before any IK. Two lower handover points were added. Execution: the giver carries at 0.07 rad/step (was 0.14) and the carry is checked. The taker's grasp is re-planned on the measured pose, checked against the giver's hand along the whole approach. The take is measured. |
| servo insertion tail | `Skills._servo_tail` (`tail_kw["servo"]`) | Starts after the pre-place re-measure. Each move is commanded through the sag transform C = T_cmd · T_meas⁻¹, recomputed before every move. Sequence: descend to 3 mm above the receptacle's top, re-measure, descend to the target slowly, check the depth. If not seated, probe at 0.7 / 1.5 / 3 mm in 6 directions (seat tolerance per place: 4 mm, pens 2 cm). Then the turn and the release. Used by insert_into, fill_container, stack_on_pegs, slots_and_press. |
| fine insertion fit | `insertion.refine_insertion` | A sub-mm local fit of a nominal insert pose (asset frames) on the receptacle's holes: 0.1 mm grid, ±3 mm, ±3 deg. It is then lowered until the leading end meets the floor. For the charger on outlet 1 the shift is (-0.8, -0.6) mm (the twin's successes sat 1 mm off the frame) and it is lowered 16.5 mm. |
| turn | `Skills._place_tail` / servo | Runs at 0.05 rad/step (was 0.02: the key's turn alone cost 52 steps). If the wrist cannot finish it, the turn falls back to 85 / 70 / 60 % (the checker accepts -60 ± 30). |
| push | `library.push_to` | Pushes at half the real height of the trailing face (from the mesh). Falls back to the other axis, strokes up to 8 cm, stops after 2 strokes without motion. |
| refine guards | `Skills.no_refine`, `Skills._held_review` | For perception: kong tiles are not wrist-refined. After a re-orienting pick, the wrist re-views the held object; more than 60 deg off the plan counts as loose, and it is put down and re-planned. |
| receptacles | `library.insert_into` | Skips the receptacle the object is in (a coin's own stand). |
| new planner rules | `planner.plan` | Pieces onto pegs (`stack_on_pegs`: groups of identical pieces by size onto pegs by height). N objects one per slot, then press the lever (`slots_and_press`: N from the number word). Stack in a named colour order (`stack_objects(ordered=True)`). Colours come from `AssetInfo.rgb`, the catalogue's mean face colour. Both new skills are first versions; no twin layouts exist for them. |
| ordered stack | `stacking.plan_stack(ordered=True)` | objects[0] is the base and the rest go on in list order (agent layer's request). |
| practical limits | `skills.PLACE_REACH`, `wrist_safe` | Transfer and handover places at most 0.60 m from the arm base in the plane; place IK with j4 <= 1.6 and abs(j6) <= 3.0. |
| relay stability | `library.rests_stably` | The support footprint's half-width must be >= 0.2 x the CoM height. |
| loose grasps | `Skills.run_transfer` | Loose grasps are carried on (re-planned for the measured in-hand pose) when the place tolerates >= 2 cm. Uneven objects are carried at 0.09 rad/step. |
| planning time | various | Transfer screen budget of 2500 place IKs per arm (best grasps first). Relay search capped at 100 s, handover search at 80 s. No in-air handover for objects over 16 cm. |
| insertion frames | `library.insert_into` | Insert frames also match a receptacle's named openings (slot / opening / insert functional points: a coin bank's `slot_opening`). The fine fit lowers at most 0.8 of the object's height. |
| handover carry | `run_air_handover` | Carried along a straight line with the orientation slerped and eased, so the turn is monotonic and short-way. Planned with 5 mm between the hands, executed with 3 mm. |

## 3. G3 numbers

GT state, physics-only twin, settled dev layouts; `compute3:~/dojo-data/skills3/g3/<run>/`; every episode
record carries the deployed commit.

| task | run (commit) | layouts | success | mean score | steps median / max (limit) | main failure |
|---|---|---|---|---|---|---|
| insert_key | v1 (360279b) | 6 settled of 100000-08 | 0/6 | 0.075 | 64 / 96 (300) | grasp_slip 6. Giver on the 5 mm shaft (hold quality's pick): slid out at the lift (3/3) or fell during the carry (3/3). Regression vs skills2's 0.15 |
| insert_key | v2 (a84cccd: giver by lift_hold) | 11 settled of 100000-19 | 0/11 | 0.0 | 46 / 47 (300) | grasp_slip 11. The giver took the shaft/bow junction (contact at the pads' edge): nothing lifted |
| insert_key | v3 (5a28ddd: giver by lift_hold with contact depth + pad-edge terms -> the bow again) | 4 settled of 100000-04 (stopped: superseded) | 0/4 | 0.15 (lift credit, like skills2) | 72-110 (300) | Key fell from the giver during the 0.07 rad/step joint-space carry (3). One lengthwise giver grasp held (0.9 mm) but left no taker grasp on the measured pose (1). Next (5464468 / c834032): the carry is a slerped Cartesian line (monotonic, short-way turn); handovers planned at 5 mm hand clearance, executed at 3 mm |
| insert_key | v4 (000e5be: slerped-line carry + handover planned at 5 mm hand clearance) | 4 settled of 100000-04 (stopped) | 0/4 | 0.0 | 44-48 (300) | With a 5 mm clearance the bow-rim giver plus a rim taker no longer planned, so the giver took the shaft above the upper tooth again: it slid at the lift, 4/4. Fix cb4d133: giver picks need >= 6 mm contact depth along the approach; planning clearance back to 4 mm. Queued as v5 |
| plug_in_charger | v1 (a84cccd: servo + fine fit) | 100000-09 | 1/10 | 0.10 | 206 / 384 (400) | place_miss 9. 7 stalled: the placing arm was across the table, 0.66-0.67 m from its base. The measured TCP stopped 8 mm above the strip while the command went 33 mm lower; xy at the hover was right to 0.5 mm. 2 relays: the charger was stood on its blades on the table and toppled |
| plug_in_charger | v2 (5a28ddd: + place reach 0.60 m, rests_stably needs width for the CoM height, dense hand check) | 100000-09 | **6/10** | 0.60 | 296 / 398 (400); successes 145-381 | All 4 failures (place_miss 3, step_budget 1) used the untilted grasp. The wrist sat on j6's 3.14 stop (j4 1.7) and stalled 10 mm above the outlet; 8-12 probes changed nothing. All 6 successes used the 25-deg tilted grasp, plugged by the other arm after an as-is relay. Fix 6d6097b: places keep j4 <= 1.6, abs(j6) <= 3.0 |
| plug_in_charger | **v3 (c834032 x5 / cc626ee x20; same code for this task) + wrist_safe** | **all 25 settled** | **22/25** | **0.88** | 156 / 237 (400) | 100002: no relay planned, the in-air handover fallback dropped the charger. 100008: the relay's last leg re-planned with the held charger dipping below the table on the carry (collision). 100013: seated 5 mm high after 2 probes on the untilted grasp, toppled at release |
| store_tools | v1 (a84cccd), stopped at the 30-min hold | 100000-01 | 0/2 | 0.25 (tape measure) | 264-277 (900) | The hammer held at its CoM station lifted with 11-13 mm drift, was judged loose and put down twice (now carried on for >= 2 cm tolerances). The wrench held at its end slid out. The pliers' slice grasp slid out. One episode spent 1269 s planning relays (now under 12k faces, a hold-quality cache, relay search capped at 150 s; 27 s locally) |

Store_tools v2 (000e5be: planning-time caps, loose grasps carried on, uneven carries 0.09 rad/step), 100000-09: **0/10,
mean 0.20** (two episodes 0.5, three 0.25, five 0). Steps 163-409 of 900; planning 56-93 s (one 423 s). The tape measure went in 3/5 (a 4th landed 20 mm high on another tool).
- Pliers: slice grasp across the outline slid out, 5/5.
- Hammer: CoM station, lifted 13-15 mm loose. The re-planned descent for that in-hand pose went below the table, so the
  place was refused (3).
- Wrench: lifted with 4-5 mm drift, then lost on the carry (3). In 100001 it was knocked out by the other hand, which
  was left open above the failed pliers pick, and flung off the table.
- Fixes queued as store_tools v3 (57530c6): `drop_ok` (let go over the slot when the descent is blocked; the ladder only
  needs the tool near its slot) and parking the idle arm out of the carry's reach including the held object.

Fill_pen v1 (000e5be: servo tail with probes, 4 mm seat tolerance raised to 2 cm), 100000-04 (stopped: superseded):
**0/5, mean 0.04** (0, 0, 0.1, 0.1, 0); skills2 had 0.08. Steps 670-975 of 1100; planning 90-307 s.
- The servo's first hover put the pen tip 3 mm above the bbox top. With the remaining xy error the tip came down on the
  holder's rim, and the shoulder joint then lagged its command by 0.28 rad, pressing the pen on the rim. The descent
  log reads "+114 mm after the descent" (the pen never moved), 3/5.
- 4-18 probes then dragged the pen and knocked the holder or the first pen over.
- Later pens: no grasp-place pair, or a relay chain that failed.
- Fix (e544a41, queued as fill_pen v2): pens hover 8 mm above the holder, a second corrected hover when the first is
  > 2 mm off in xy, no probing, and a servo descent that stops short (IK) is resumed. The plug keeps its 3 mm hover.

Organize v1 (000e5be), first chunk, 100000-02: 0/3; scores 0.5, 0.5, 0.25.
- The mouse goes in 3/3.
- Alarm: on the drawer 1/3. 1/3 grasp_slip (rose 23 mm). In 1/3 the transfer screen's 2500-IK budget ran out before
  its feasible pairs (budget raised to 6000 in later commits).
- Keyboard: pushed into the frame 1/3 ("within 10 mm after 4 strokes", the face-height push). 2/3 had no stroke plan
  on either axis (the keyboard's front edge 0.18-0.2 m from the arm bases).
- Figurine: no grasp-place pair next to the drawer, 3/3.

## 4. Failure taxonomy (what the traces with measured joints showed)

1. **The arm's practical envelope is smaller than the IK's.** Two cases. (a) Places 0.66-0.67 m from the arm's base: the measured TCP stopped about 30 mm above the commanded one, at the strip, and probing could not help (plug v1 7/10). (b) A place with the wrist on j6's ±3.14 stop (and j4 at 1.7) stalled the same way (plug v2 1/5). Filters: `PLACE_REACH` 0.60 m and `wrist_safe` (j4 ≤ 1.6, |j6| ≤ 3.0) on transfer and handover places. A far target then goes through a relay to the other arm.
2. **Sag under load.** At the pre-place the measured TCP sits 2-10 mm off the commanded one (more on long reaches), and the in-hand pose looks perfect. Re-measuring the object alone cannot see this error. The servo tail commands through C = T_cmd · T_meas⁻¹, which brought the blade tips to within 0.5 mm of where the successes seated (plug).
3. **The grasp turns about the closing axis.** A key held on its round bow, carried while the hand turned it 90° about that axis, swung 50° and fell (skills2 v2, 6/6). A key held on its 5 mm shaft slid out at the lift or on the carry (skills3 v1, 6/6). A key held at the shaft/bow junction touched only the pads' edges and stayed on the table (skills3 v2, 11/11). Contact depth along the approach and contact at the pad's edge now count in `lift_hold`.
4. **Unstable relay poses.** A charger stood on its two blades (13 × 8 mm footprint, centre of mass 35 mm up) passed the footprint test and toppled (plug v1 relays 2/2). `rests_stably` now also needs the footprint's half-width ≥ 0.2 × the CoM height.
5. **The finger model differs between generator and checker.** The box grasp generator estimated the fingertips 0.8-1 mm higher than the collision model, so every box grasp of a thin lying tool failed the 2 mm table margin and only whole-outline slices remained (the store_tools v2/v3 failures). Both now use the same finger points.
6. **Planning time.** Up to 21 min for one store_tools episode (relay legs, each screening ~250 grasps with 50 ray casts on 28k-face meshes). That breaks the 30-min GPU-hold rule. Fixed with 12k-face clustering, a hold-quality cache and a 150 s relay cap; chunks are now ≤ 5 layouts.

## 5. Interfaces (for integration / the agent layer)

- `Skills.plan_place(..., servo=False, servo_ref=None)`: the closed-loop tail. It is on in `insert_into`, `fill_container`, `stack_on_pegs` and `slots_and_press`; every other caller is unchanged.
- `Skills.no_refine` (set of labels): the wrist refine is not asked about them (kong tiles, per perception).
- `Skills._held_review`: with a refine hook, a re-orienting pick re-views the held object (refine kind `"held"`). More than 60° off the plan counts as loose and is redone.
- `library.stack_objects(objects, nest, ordered)` and `stacking.plan_stack(..., ordered)`: bottom-to-top order (agent layer).
- New skills: `stack_on_pegs(groups, base, pegs)` and `slots_and_press(objects, container, slots, press_tag)`. New planner rules: pegs, slots + lever, colour-ordered stack. `AssetInfo.rgb` and `world.colour_name`.
- `insert_into` skips the receptacle the object is in, and matches insert frames to named openings (`slot` / `opening` / `insert` functional points).
- No renames. `planner.solve` / `plan` and `SKILLS` keep their signatures; SKILLS gained two entries.
