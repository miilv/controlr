# 2026-10-29 — deposit_coin: the camera failure classes, and the coin the agent named "coin" (stream C)

There are 15 camera rows on compute4, all 0 / 15: c4b_coin_off / stereo 100000-04 and c4b_depositcoinsg_rb 100005-09.
G3 is 20 / 25. Every row was checked against layout GT, the dev GT probe and the run logs. 0 LLM.

| class | rows | cause | lever |
|---|---|---|---|
| grasp_slip "the disc fell out on the carry" | 5 (off/stereo 100000, 100003; rb 100009) | held_look reads the 1.9 mm coin as "jaws closed on nothing", so the belief drops it | `held_thin` (#281) |
| coin0 not found | 5 (off/stereo 100001; stereo 100002 / 04; rb 100008) | its own pixels refute a thin slot child | `thin_slot_keep` (#283) |
| plan: the coin never lifted | 3 (off 100002, 100004; rb 100007) | the stand's yaw / offset (2), and a failed refine's `suggest_xy` re-hovered the pick 42-53 mm off a correct belief (rb 100007) | `slot_read` (#281; an ok read replaces that refine) |
| plan: no edge grasp ("no rim grasp of coin0 (axis 2)") | 1 (rb 100005) | the agentic read named it "coin", not coin0. The settle inside the agent's adapter looks children up by the task label, so it left the coin lying flat at z 0.765 | **`EstimatorConfig.slot_settle_merged`** (this change) |
| plan: place_miss after a handover, the coin at the right xy but 64 mm low | 1 (rb 100006) | skill side (the beliefs were within 1 mm / 0.4 deg) | — |

**`slot_settle_merged` (default off).** `agentic.record.settle_record_slots` runs `seed_refine.settle_slot_children` on
the merged record, where the labels are the task's, and writes the settled poses back.

**Offline**, on the agentic records of all 15 rows (18 records):
- 100005's coin goes from 89 deg to 14.5 deg normal error (standing in its stand; the remaining error is the stand's
  yaw, which `slot_read` measures at the hover);
- every other record is unchanged.

**The queued coin A/B.** Arm B (slot_read + held_thin + thin_slot_keep) targets 13 of the 15. This flag would add 1
more (`DOJO_PERCEPTION_CFG`).
