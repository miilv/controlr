# TOOLS: matching-slot approach, clearance and held-pose experiments

**Question:** can slot motion changes recover the store-tools skill ceiling with perfect perception?
**Setup:** physics-only G3, compute6, no LLM. Baseline `ebcbca694616e452071793f46735325b891291a7`.
Fresh v15 records, 100000–04 variant 0 and 100030–39 variant 1; all 15 passed two-pass presettling.
Source under `~/controlr-tools`; data, actions, traces, layouts and logs under `~/dojo-data/tools`.
At most two light locks; admission VRAM <=24 GiB, disk >=10 GB; 8 GB RAM / 600% CPU per scope.
No compute3, compute4, sealed directories, measurement-only IDs or reserved IDs used.

## Baseline and classes

| Layout family | n | Success | Mean score (0–100) | Sample SD |
|---|---:|---:|---:|---:|
| v0, 100000–04 | 5 | 2 | 75.0 | 25.0 |
| v1, 100030–39 | 10 | 2 | 52.5 | 38.1 |
| All | 15 | 4 | 60.0 | 35.1 |

The variant-1 ceiling matches the historical ~55. The paired ten-layout set below is easier (77.5 in A);
do not compare its treatment mean with the full baseline or historical numbers.

Of 40 tool opportunities in that paired baseline, nine miss credit:

- Four carry losses: pliers 100001/02, wrench 100003, hammer 100030. Hover attachment shifts 104–300 mm.
- One tail/contact loss: wrench 100033 moves another 38 mm / 31 degrees during the closed tail after an 11 mm
  hover shift; final error 32 mm xy / 28 mm z / 69 degrees.
- Two wrench pick slips: 100001/100031, rise 42/48 mm instead of 80 mm.
- Two unsuccessful slot-transfer completions: tape measure 100002 (failed relay pick/completion), hammer 100033
  (134/148 reachable left grasps, no usable grasp-place pair in A, no in-air handover for a 228 mm object).

This is a credited-loss taxonomy. Raw failed attempts overlap it: e.g. 100003 hammer fails the orientation
verification yet still gets credit. Do not count every verification miss as a score loss.

A mesh/trace clearance audit of 100030 shows the hammer's lowest point reaching the table six carry steps in,
before reaching the toolbox. Later fixture contact accompanies the large shift. Measured TCP lag remains under
3 mm in the baseline hammer approach. The last-hover read is too late; the known joint-5 stop is not this case.

## Independent switches and paired outcomes

Every switch is new and default off; no preset/default was changed. Each trial varies one switch only.

| Lever | Physics code | Dev screen | A -> B | Disposition |
|---|---|---|---|---|
| `slot_approach=linear` | `83c2f587` (record stamp `0706dad9`, later docs/tests only) | 100030 v1 | 75 -> 50 | negative; pliers newly lost, hammer still lost |
| `slot_hover=low` | `4f6f2fef` | ten pairs below | 77.5 -> 67.5 | reject |
| `slot_place=servo` | `c0aa7a63` | 100033 v1 | 50 -> 50 | null score; wrench error worsens to 130 mm |
| `slot_carry_read=on` | `774b8030` (corrected v2) | 100030 v1, 100003 v0 | 75 -> 0; 75 -> 25 | negative; checked corrections infeasible |

The lower-hover trial uses 15 mm pre-height / 3 mm support clearance instead of 40/15 mm, retaining checked
approaches. It can change selected grasps: 100030 hammer changes to a wide slice and fails to lift.
The servo-tail trial uses the existing measured tail without insertion probes; it does not address earlier carry loss.

| Dev draw | A baseline | B lower hover | B − A |
|---|---:|---:|---:|
| 100000 | 100 | 100 | 0 |
| 100001 | 50 | 75 | +25 |
| 100002 | 50 | 25 | −25 |
| 100003 | 75 | 25 | −50 |
| 100004 | 100 | 100 | 0 |
| 100030 v1 | 75 | 50 | −25 |
| 100031 v1 | 75 | 50 | −25 |
| 100032 v1 | 100 | 100 | 0 |
| 100033 v1 | 50 | 75 | +25 |
| 100034 v1 | 100 | 75 | −25 |

**n=10 paired; success 4/10 -> 3/10.** Mean delta **−10.0 points**, sample SD of paired deltas **24.2**.
Score improvements/regressions/ties: **2/5/3**. Success flips: **0 wins / 1 loss**. v0 and paired v1 halves each lose
10 points. The 100030 pilot is included once, not rerun or doubled. No infrastructure failures or missing meshes
occurred in these rows. No camera run of this lever is justified.

## Carry-read follow-up

A bounded feedback trial addresses the earlier loss: at most four checked Cartesian hops, interpolated object
orientation, attachment updated between hops. G3 reads the true measured world; perception uses only accepted
current-view held reads under the existing uncertainty gate, with no observer/show moves. Missing first read
keeps legacy motion; a later unavailable read retains the last accepted attachment. Infeasible hops stop motion
without releasing; existing recovery handles the retained grasp. No servo/low-hover/linear-switch bundle is enabled.

The initial 3bb99584 implementation used the stored attachment for the interpolation start. Review added a
failing regression test and corrected it to the first accepted current pose. Both version-1 rows remain on disk
as bug-confounded observations. The corrected 774b8030 was then run on both same dev cases (no camera data).

Both corrected screens are negative: **75 -> 0** on 100030 and **75 -> 25** on 100003. Tape-measure correction refuses
at hop 2 (IK orientation residual about 2 degrees); wrench corrections hit a table-clearance refusal or an
unreachable final waypoint; hammer correction also fails IK. Refused partial paths are not executed, and the
existing recovery releases the retained grasp. No expanded carry-read trial is justified. Camera transfer is
unproven and the prospective camera gate is blocked.

**No lever raises the ceiling.** Retain all defaults off; do not dispatch a camera comparison from these results.
The experimental switches are retained on the unmerged branch for reproducibility/review, not adoption. The next
mechanical question is a grasp/route pair that stays supported and leaves correction reach, rather than another
carry speed or last-hover measurement change.

## Validation, cost and camera gate

Baseline full local suite: **1783 passed, 187 skipped**. Focused switch/legacy tests: **49 passed**. The added camera-path test exercises the actual uncertainty gate
and raises if the controller tries to read a true object pose. Final CI is linked on the PR. No live model calls, tokens or spend. Completed-episode wall
costs (excluding startup, presettle and load): baseline 4463.1 s; Cartesian 192.6 s; low-hover 1827.3 s; servo 154.0 s; carry-read v1 572.0 s and corrected v2 613.7 s. Total **31 completed episodes**, **7822.7 s** recorded episode wall
(~2.17 lane-hours), plus **178.9 s** recorded load time and uncounted Kit startup/presettle overhead.
Jobs yield between chunks; all artifacts remain on compute6. No camera frames exist, so there are no camera videos.

Prospective specifications: [Cartesian gate](2026-11-06-tools-slot-camera-ab.md) (G3 prerequisite failed),
[measured carry gate](2026-11-06-tools-carry-read-camera-ab.md) (requires ten-pair G3 support and lead release).
Both require a frozen cx-only camera stack, fresh salted dev layouts, >=10 pairs, the repository's score/flip rule,
guard cells and a full harmful-belief-write audit. No Claude-route calls or camera jobs are authorized here.

Detailed setup, traces, failed approaches and chronology: [journal](../journal/2026-11-06-dojo-tools-slot-placement.md).
