# COIN: isolate the unresolved disc opening verdict

## Evidence and scope

Read-only diagnosis used exactly 15 camera rows: compute4
`~/dojo-data/integ-c4/runs2/c4b_coin_{off,stereo}_100000` through `100004`,
and `~/dojo-data/integ-c1-mirror/runs2/c4b_depositcoin_rb_100000` through `100004`.
No measurement episode, sealed data, or id >=100100 was opened. No compute4 writes or GPU jobs.
Local scratch was 134 MB. No LLM calls.

Provenance: off/stereo head `ad5b50e`, skills `eff3994`, submit_v3,
command grip source; mirror head `ce4a4ce`, submit_v3, grip_check on.
SHA256 of each report's config, serialized with json.dumps(sort_keys=True):
`ec4376401f2f` (off/stereo) and `15f0a6a26886` (mirror), abbreviated.
These are historical diagnostic rows, not current-candidate measurements.
The repository has no tracked CANDIDATE.txt; the read-only copy at
`compute4:~/dojo-integ-c4-state/CANDIDATE.txt` was checked instead. It uses submit_v3,
DOJO_GRIP_SOURCE=v2, grip_check on, held_pose=request via preset, and does not enable
held_thin or slot_read. It additionally sets ported routes, classify p2, peg base
refinement and belt_wrist_veto, among other task-specific options. No claim that
these older rows reproduce the candidate's score of 20 is made.

## Classification

All 15 fail success; mean score 0.0667 (off .08, stereo .04, mirror .08).
There are five distinct layouts, not fifteen independent layouts.

- **Opening-negative handover abort: 6/15.** All three families at 100000/100003.
  Each report has two opening-only held_look negatives, confidence 1, measured and
  commanded opening 0, followed by a belief drop: **12 events** total.
  Coin/stand final perception xy errors are <=0.5 mm and yaw <=3.6 degrees.
  Five rows' dev GT probes show 149.1–149.7 mm coin rise, consistent with the
  previously replay-verified false drops. Stereo 100003 shows only 3.4 mm rise:
  its actual grasp outcome is unresolved, and it is NOT counted as a demonstrated
  physically held coin. The reported 373/413 mm handover error is after the belief
  drop, not a measured in-hand displacement.
- **Coin absent: 5/15.** All families at 100001; stereo 100002/100004.
  No closes, zero GT rise. 100001 stand xy error 18.5–20.6 mm.
  Stereo 100002 has a stand within 0.8 mm/1.7 degrees but no coin. Stereo 100004
  does not retain the stand either. These are perception/existence failures.
- **Bad pick geometry / no sustained lift: 4/15.** off and mirror at 100002/100004.
  Off 100002 final coin xy 1 mm but yaw 48.4 degrees; off 100004 xy 12.1 mm.
  Mirror 100002 initial coin xy 67.4 mm, final 5.2 mm, stand still 67.5 mm off;
  bank 49.1 mm/19.2 degrees off. Mirror 100004 coin 9.2 mm/26 degrees off.
  GT peak rise is 0–10.5 mm. Pick logs report success but later place misses are
  156–385 mm xy and 71 mm low: these do not establish insertion failures, since
  the coin was never demonstrably carried to the bank.
- **Isolated insertion or timing class: 0/15 demonstrated.** No time caps;
  the old scoring-id handover jam described in prior journals was not reopened
  or used to choose this change.

Dominant fixable classes are unresolved opening negatives and missing/incorrect
coin poses. This change addresses the first only. slot_read stays parked.

## Change

`RuntimeConfig.disc_hold_evidence="on"` enables the existing opening-resolution
unknown rule only for disc geometry (`precision.disc_axis`: thickness <5 mm,
other extents within 5%). Default off preserves today's behavior, including events.
This deliberately isolates the disc portion of the older, broader held_thin switch;
it is not a new perception algorithm or a re-test of the rejected slot_read bundle.
Keys and rectangular tools keep today's path. Disc classification uses asset
geometry, never task names or layout ids. Only an opening-only negative with a
measured opening below the sensor resolution is made unknown. Colour negatives
and missing measurements keep the baseline path. No new belief pose is written.
The real-slip risk is explicit: an unresolved empty disc grasp can be retained;
closed-loop camera scoring must decide whether that costs successes or time.

Unit tests cover both disc axis conventions, a rectangular thin tool, a thick
block, colour evidence, missing measurements, default off and invalid values.
Validation results and run paths are recorded below after execution.
