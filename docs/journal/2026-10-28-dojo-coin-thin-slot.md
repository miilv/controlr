# 2026-10-28 — deposit_coin (c): the coin "not found" at frame 0 (stream C)

**Diagnosis (5.1 camera row c4b_coin_off 100001, coin never found, nothing to plan).**
- B1 places the coin from the stand's slot, then checks the coin's own head-view pixels: its silhouette minus the stand's
  and every other object's.
- The stand was read 19 mm off on a tight fit (conf 0.63). The coin's ~2 mm edge-on pixels fell on the table, and the
  check reported "support slot empty".
- The thin-child exemption (`THIN_CHILD_M`: "placed from its base, unverified") only applies on a low-contrast or
  re-chosen base fit.
- Later records only carried the stand over ("support base not found" for the coin). The wrist search refines of a
  2 mm disc never fit.

**Change: `EstimatorConfig.thin_slot_keep` (default off).** A thin slot child is never refuted by its own few
pixels, whatever the base's fit. It is placed from its base and flagged unverified. The pick hover's `slot_read`
(PR #281) measures it there.

**Offline (B1 on the saved 5.1 frame 0 of c4b_coin_off 100000-04, fold 0 of bg_folds_51_ilia, main e9ca3e9;
`scripts/perception/coin_frame0_eval.py`):**

| layout | stand xy / yaw off | coin, flag off | coin, flag on |
|---|---|---|---|
| 100000 | 0.4 mm / 2.8° | found, 0.4 mm | same |
| 100001 | 19 mm / 1.5° | not found ("support slot empty") | unverified, 19 mm |
| 100002 | 119 mm / 51° | not found | unverified, 119 mm (the stand is wrong) |
| 100003 | 10 mm / 26° | not found | unverified, 10 mm / 26° |
| 100004 | 12 mm / 3.5° | unverified, 12 mm | same |

- With the flag the coin exists in all 5, wherever its stand was read.
- The slot read corrects stand errors up to ~30 mm at the hover: 24 of 25 applied reads right from 30 mm-off priors
  on 5.1. That covers 100001 / 03 / 04.
- 100002's 119 mm stand is a B1 stand miss. In the camera run the agentic read moved it to 0-1 mm.
- The camera A/B with `slot_read` + `held_thin` decides.
