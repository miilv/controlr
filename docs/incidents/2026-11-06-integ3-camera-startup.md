# Compute3 camera lane: genuine Isaac startup failure

INTEG3 prepared the shared compute3 camera lane with the compute6 offline Kit
extension cache and 600-second early/hard watchdog recipe. Package parity
passed against compute4's 416a2a03 snapshot with zero differences. Driver
595.99.02 remained unchanged. This recipe did not establish working genuine
Isaac 5.1 rendering on this host.

The first START was 15:41:19Z for PEN A/100020. Isaac crashed during startup,
before producing an episode row. Four subsequent short attempts were pulled
by the existing lane's automatic no-row retry/dispatch logic before STOP took
effect: A/100020 retry, B/100020, B/100020 retry, B/100021. All five attempts
finished without an episode row. Dispatch drained at 15:44:06Z.

The current-attempt logs explicitly say `[crash] A crash has occurred` and
write new dump IDs; these are not merely old crash-upload messages. Fatal
stacks include libomni.usd and libcarb tasking. Startup also reports an errno=28
change-watch failure despite about 99 GB disk free; that may reflect a watcher
limit, not disk exhaustion. Root cause is not established by this attempt.
The [earlier genuine-render incident](2026-11-06-seg-isaac51-renderer.md)
already records a startup failure on the repaired driver with RTX scene-library
frames. The present failure must not be attributed solely to that driver
without a bounded dev diagnosis. INTEG3 missed this second incident at preflight.

The existing launcher returns rc=0 even after a native driver startup crash;
the lane's no-row check prevents counting it as a valid comparison, but its
one automatic retry then continues to other registered rows. This consumed
both permitted attempts for A and B on 100020 and the original B attempt on
100021. No treatment outcome, scoring frame or tuning diagnosis was inspected.
No valid pairs, adoption verdict, watchdog change or renderer substitution.

STOP remains set with 158 queue rows. All original 80-pair manifests and raw,
settled and index hashes remain intact; exhausted 100020 arms are absent from
the active queue and must not be silently refilled. A compatible renderer and
a prospective infrastructure-attempt amendment are needed before resuming.
The lane is not left running. No system/driver/service/VPN configuration or
other stream was changed. Owned side-service/power cleanup is recorded in
[the lane journal](../journal/2026-11-06-integ3-camera-lane.md).

Evidence is compute3 `~/dojo-data/integ3-1106/meta/blocked-snapshot.json`,
`~/dojo-integ-c3-state/lane2_heavy1.log`, and each retained attempt's
`integ_eval.log`. No model transcripts or calls.jsonl files were created.
