# 2026-10-27: deposit_coin: the handover taker's slot target with the hand straight down (G3, 25 paired layouts)

**Question.** Five of deposit_coin's nine G3 misses are in-air handovers whose coin lands flat on the bank or on the
table. Is it the order in which the taker's slot targets are tried?

**Setup.**
- G3 mode A: physics-only twin, TRUE state; all 25 settled dev layouts (100000-100024); `SIMLAB_USD_SYNC_LINKS=1`;
  `--trace`; compute6 light slots 1-2 in parallel; no LLM calls.
- Before: main 48f5dde. After: 48f5dde + `hand_down_first`, plus a one-arm carry at 0.06 rad/step that changed no
  one-arm outcome and was dropped from the PR.

**Diagnosis** (traces of skills-5's coin_final, measured joints): the taker's hand at the slot hover leans 46-90 deg
from vertical in all five handover misses. Its finger roots then rest over the pig's back, and the `fingers_out`
clearance stops the descent 0-7.5 mm down: the coin is let go 0-1.5 mm into the slot.

**Results.**

| layout | before | after |
|---|---|---|
| 100001 / 09 / 13 / 16 / 22 (handover) | 0.2 / 0.2 / 0.2 / 0.2 / 0.2 | 1.0 x5 |
| 100003 (handover) | 1.0 | 0.2 (in the bank, 0.8 mm under the lowest success: the bbox bottom bound) |
| 100000 / 02 / 17 / 19 | 0.0 / 0.2 / 0.0 / 0.0 | the same |
| the other 15 | 1.0 | 1.0 |
| **total** | **16/25, mean 0.688** | **20/25, mean 0.816** |

**Conclusion.** Rescued 5, lost 1 (landing noise at the checker's bound); steps median 163 -> 168 of 300. The
handover now succeeds whenever its legs plan and hold.
- Left: pad slides at the extraction from the stand (2), a loose giver pick (1), an unreachable taker descent (1).
- In mode B the coin first has to be found (breadth2: perception, #78).
