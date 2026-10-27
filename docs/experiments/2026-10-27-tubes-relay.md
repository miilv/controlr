# 2026-10-27: insert_tubes: the as-is relay of a lying tube, places within reach, one-arm tubes first (G3, 25 paired)

**Question.** insert_tubes loses in G3 a third of its layouts at 0.4. Do these remove those losses?
- a relay for a tube lying on the table;
- deferring off-CoM plans behind it;
- a tighter place reach;
- one-arm tubes first;
- a depth check on "in".

**Setup.**
- G3 (physics-only twin, TRUE state), all 25 settled dev layouts, `SIMLAB_USD_SYNC_LINKS=1`, compute6 light slots 1-4,
  no LLM calls.
- Before: main 48f5dde. After: the PR head 2681f3d (on main 0127871, which has #93).
- G3 mode B on 100000-14: `--mode-b`, 2 mm belief offsets on the three tubes, `--belief-drop fixtures`; before = main
  0127871.

**Results.**

| | n | before | after | rescued / lost | steps median / max (500) |
|---|---|---|---|---|---|
| mode A | 25 | 16/25, 0.784 | **24/25, 0.976** | 8 / 0 | 336 / 430 -> 344 / 485 |
| mode A, relay + off-CoM deferral only (7effcc9) | 25 | 16/25 | 21/25, 0.904 | 7 / 2 (both a right-arm place 0.56+ m out) | 345 / 485 |
| mode B, fixtures belief | 15 | 6/15, 0.613 | **9/15, 0.760** | 3 / 0 | 305 / 473 -> 452 / 481 |

- Mode A: rescued 100005 / 06 / 08 / 09 / 10 / 11 / 19 / 22. Left: 100013, two relays in one episode, out of steps.
- Mode B: five of the six left ran out of steps in a relay (449-481 of 500).
- CoM-held places by per-layout hole distance (mode A, all runs): 143/144 in below 0.56 m; 0/4 at 0.562-0.567 m; 2/2
  at 0.596 m (the left arm). Off-CoM places: 2 in / 6 missed.

**Conclusion.** In G3 mode A insert_tubes goes from 0.78 to 0.98 (8 rescued, 0 lost of 25). In mode B the relay
helps (+3 of 15, none lost), but the 500-step budget caps it: a relay leg costs ~110 steps.
- Proxy B already spends ~440 steps on three direct tubes, so in proxy / rendered B mostly the step-free parts count:
  the one-arm-first order and the depth check.
- Next: a cheaper relay put-down; confirm in proxy B.

Data: compute6 `~/dojo-data/precision/g3/`:
- `tubes_g3main_48f5dde`, `tubes_t6a_2681f3d`, `tubes_t6b_2681f3d`, `tubes_relay2_7effcc9`;
- `tubes_mbfx_main_0127871`, `tubes_mbfx_t6_2681f3d`.
