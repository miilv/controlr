# 2026-11-07 — IMIT2 camera A/B prepared and queued on compute6

The lead approved the three-arm design (A = frozen candidate, C = A + home wait + kept sides, B = C + the four IMIT2
switches; comparisons C vs A and B vs C). [Pre-registration](../experiments/2026-11-07-imit2-camera-prereg.md) on main
(`e649a2c1`, inputs frozen in `6d10fa0c`) before the queue write.

- **Prep (compute3, light locks, one job, 8 G / 600 % scopes, zero model calls):** salt frozen in git first
  (`4d7fc372`); 20 imitation ids 100000–09 / 100020–29 and the first five stack_blocks / classify_objects ids generated
  and settled on the first pass; demos 18/20 at speed 1.25, 100021 at 1.5, **100002 invalid at both → excluded**
  (n = 19). A first launch failed before generating anything (RoboDojo home not set in the child env; attempt kept).
  15:12Z–15:59Z.
- **Deployment:** merged main `89b194be` (contains #421) at `compute6:~/dojo-integ-c6-89b194be` by the MATCH2 recipe
  ([deploy.sh](../experiments/data/imit2-camera/deploy.sh)): package parity `diff={}`, candidate parity true, catalogue
  gate OK. Block copied through the agent box, 84 / 84 files SHA256-verified; the physics kit replaced by compute6's.
  All 87 rows construct; C − A and B − C differ only in the registered fields.
- **Queue:** 87 rows appended at the **end** of `~/dojo-integ-c6-state/queue2.txt` at 16:06:53Z under `wave-start.lock`
  then `queue2.txt.lock`, after the POUR re-test rows: positions **81–167**; queue SHA256 `8b644a61…` → `48e6835e…`;
  every row pins `AGENTIC_MODEL=claude/claude-sonnet-5-5`. Rough ETA: start ~18:45Z, finish ~02:15Z (8 Nov).
  [Receipt](../experiments/data/imit2-camera/dispatch.json).
- Writes on compute6: only `~/dojo-integ-c6-89b194be`, `~/dojo-data/imit2-cam` and the queue append. No process
  killed. Verdict (both comparisons, guards, write audit) after the rows drain.
