# PEN round 2: preserve the successful baseline paths

Lead requested the original-five regression diagnosis, a conditional replacement,
and 20 paired development layouts. This continues
[round 1](2026-11-06-dojo-pen-ceiling.md) in PR #397; defaults remain off.

## Trace diagnosis before the replacement

The broad inset bundle scored 40/25/0/10/10 against 25/100/10/25/10.
The name “inset” here denotes the four-switch bundle, not a one-factor ablation.
Exact trace differences do not establish which of its four switches alone caused
all downstream changes.

- **100000, +15:** three pens logged inside instead of two; no lost baseline tier.
- **100001, -75:** first target3 (oil_pen/00000) changes from yaw 180, root
  (-1.1,+13.9) mm to yaw 345, root (-1.1,-11.1) mm, with 7 rather than 3 mm
  opening clearance. It settles fully into the cup: its leading end is 113.8 mm
  below the rim 15 steps after release, at 16.9 degrees tilt. The release-time
  height alone misleadingly suggests a rim hang. Next, target2 (oil_pen/00001)
  is refused at fit selection. On this exact post-release world, both retained
  inset attempts fail; the baseline's later centered fit with 2 mm occupied-pen
  padding finds a 1 mm-clearance fit (yaw 240, root +15.9,-7.1 mm). The inset
  branch had removed this relaxation. Target0 (pen/00001) then has no complete
  executable transfer despite 330 screened grasp/place pairs; target1 succeeds
  after its relay. Baseline eventually inserts all four, including retrying both
  slim pens. Only two are inserted by the bundle. The baseline cup is in the
  shared comfortable workspace and is never relocated.
- **100002, -10:** baseline moves the holder from (+.0624,-.1403) m to
  (-.1124,-.2674) m. The stay branch leaves it at its original pose. Its first
  target2 oil drop stops high (servo residual +21.4 mm, xy 7.9 mm after two
  corrections); the settled leading end is only 0.1 mm below the rim. Target3,
  the baseline's scored oil pen, has zero reachable grasp/place pairs. Baseline
  target3 uses q5 command 1.6675 versus measured 1.5797, with 15.6 mm TCP lag,
  and nevertheless settles inside at 113.2 mm depth. The safer limit and changed
  fit/holder pose do not preserve that successful drop. The remote pen/00004
  and slipping pen/00000 were already baseline failures.
- **100003, -15:** relocation logs failure (+47.1 mm height residual followed
  by a failed holder regrasp), but physically moves the cup from
  (+.2342,+.0043) m to (-.1281,-.2456) m. Both oils then land without TCP lag.
  Stay preserves the far original cup: target2 oil_pen/00000 relays inside,
  but target3 oil_pen/00001 has 6/72 reachable left grasps and zero drop pairs.
  Its fit also changes from 1 to 2 mm clearance. Both slim pens are rolled
  pen/00000 and slip in both arms. One scored oil is lost.
- **100004, tied 10:** the bundle still inserts target3 oil, while target0
  stops high (+37.7 mm descent residual) and never enters. It additionally tips
  the holder during left-arm parking near step 360. Baseline leaves it upright.
  A banked score tie therefore conceals a material physical regression.

All ten original trajectories were replayed physics-only with
`scripts.perception.m95_replay --every 1 --full`. Raw traces, replay snapshots,
and CPU diagnostics remain under `~/dojo-data/pen/r2` and `~/controlr-pen/pen-r2-*.py`.
All ten replays reproduced their recorded scores exactly; baseline 100001 also
reproduced full success (619 actions). Replay episode time totals 405.4 seconds,
excluding process startup. The same-world packing check uses the recorded object poses and production
`_fill_drop_target`, not a fitted outcome label.

## Frozen conditional profile

`fill_guard=geometry` (default `off`) applies the four-switch profile only when:
the holder is rigid; its x center lies between the two arm bases; its nearest
arm is farther than `comfy - FILL_FINGER_CLEAR` (0.40 m at defaults); its farthest
arm is beyond `comfy` (0.42 m); and at least one horizontal slender shaft is flat
(body ratio >=6, horizontal long axis, a body face within 10 degrees of world z).
Otherwise it calls the baseline wrapper. No layout IDs, asset IDs, scores,
checker calls, or replay measurements enter the decision. All four independent
switches must be off; the profile scopes and restores them even on exceptions.
The off path does not even evaluate the condition. G3 records the decision and
features. The criteria are development heuristics derived from these failures,
not a contact model or proof of stability.

Commit 0bb9348a introduced the gate. A preliminary comparison completed four A
and three B episodes before it was stopped by checking the exact G3 module and
output directory, then TERM/KILL on those own PIDs. Those incomplete jobs are
not the final comparison. The lateral-span refinement at **0fc83e8f** was frozen
before inspecting any scoring outcomes of the five added variant-2 draws.
A runtime review then caught an extra `sk.refresh()` in the gate: camera
providers may perform tracking or belief writes on refresh. Before any of the
five added layouts produced scores, the intermediate `final_A20` / `final_B20`
jobs were stopped by their checked module/output PIDs 252091 / 252092. Their
partial results are not the final panel. B's intermediate code stamp was stale
(0bb9348a); source hashes retained under the stream root identified its actual
0fc83e8f source. This metadata issue is corrected for the final launch.

**Final source freeze: 3fda84cd.** The condition consumes the world already at
skill entry, with no extra provider read. A dedicated test fails if the helper
calls refresh. Both final jobs use byte-identical source checked by SHA256 for
library, Skills, G3 and runtime config, and the deployment stamp was written
before either launch. Final outputs are `guard_A20` / `guard_B20`.

## Validation and resource record

Final cohort: original 100000–04, previous private variant-2 100030–39, and five
new private variant-2 100040–44. All 20 presettled records are fixed under
`~/dojo-data/pen/r2/simlab/layouts/fill_pen_holder`. The old fifteen are copied
byte-for-byte; new raw records and the presettle index stay in the private root.
Pass `SIMLAB_DEV_LAYOUT_SUFFIX=.settled` and the explicit layout directory to
both arms. Never use 100005–09, 100020–29, or >=100100 for tuning.

A: defaults. B: only `--skill-attr fill_guard=geometry`. Command for either arm:

```sh
SIMLAB_DEV_LAYOUT_SUFFIX=.settled scripts/simlab/rdpy.sh -u -m dojo.skills.g3 run \
  --task fill_pen_holder --seeds 100000-100004,100030-100044 \
  --layouts "$HOME/dojo-data/pen/r2/simlab/layouts/fill_pen_holder" \
  --trace --out "$HOME/dojo-data/pen/r2/guard_A20"
```

B uses `guard_B20` and the single switch above. Resource wrappers use light3/4,
check used VRAM <=24576 MiB before starting, and constrain each job to 10G RAM /
600% CPU with single-thread BLAS. Replays finished on light3 before the final launches;
B uses light4. At most two PEN jobs hold locks. No camera lock, system changes,
Claude route, or live LLM calls. Results and final verification follow in the
linked experiment report once both arms finish.
