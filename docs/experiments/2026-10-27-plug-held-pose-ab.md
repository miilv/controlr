# 2026-10-27: plug_in_charger: held_pose on vs off in the runtime (rendered 5.1, 10 paired layouts)

**Question.** G3 mode B with the held-pose protocol (an in-hand read at the pre-insert hover, an observer look, a
show pose) took plug_in_charger from 2/10 to 8/10 (skills-9). Does `held_pose: auto` do that in the runtime?

**Setup.**
- Genuine Isaac 5.1 rendered episodes (`integ_eval.sh --renderer isaac51`), compute6, preset `submit_v1`.
- Two arms: as is (`held_pose` off) vs `--config {"held_pose": "auto"}`. main 2a6e30d.
- Settled dev layouts 100000-100009, run in chunks of 5 with the arms interleaved.
- LLM calls: the agentic perceiver's, ~3 per episode (~60 in all).

**Results.**

| layout | off | auto | auto's held_pose reads |
|---|---|---|---|
| 100000 | 1 | 1 | 2 (1 ok at the hover, not adopted) |
| 100001 | 1 | 1 | 1 |
| 100002 | 0 plan (no relay plan) | 0 plan | 0 |
| 100003 | 0 (flip loop) | 0 (flip loop) | 2 |
| 100004 | 1 | 0 (flip loop) | 0 |
| 100005 | 0 perception_gross | 0 perception_gross | 1 |
| 100006 | 1 | 1 | 2 |
| 100007 | 1 | 0 belief_mismatch | 2 (fit_poor, out_of_range) |
| 100008 | 0 belief_mismatch | 0 belief_mismatch | 2 |
| 100009 | 1 | 1 | 2 |
| **total** | **6/10** | **4/10** | 14 reads: 1 ok, 0 adopted |

- No candidate observer look or show pose planned with our IK / collision model ("palm / link at table" around the
  low hover; "arms closer than 4 cm"), so the protocol read at the hover only. 13 of 14 hover reads were not ok:
  ambiguous 8, fit_poor 4, out_of_range 1.
- Auto's two losses are not held_pose's doing: 100004 is the flip loop, with no read at all; 100007 read nothing
  usable and ran the same probing tail as off.

**Conclusion.** No gain, n = 10, and both losses are noise from other causes. `held_pose` stays off (lead).
- The lever is a show pose that plans: our own generic candidates when none of perception's does (branch
  `feat/dojo-precision-showlook`).
- The flip loop is a runtime fix of its own (journal §5).

Data: compute6 `~/dojo-data/precision/runs/B_plug_hp{off,auto}_2a6e30d_1000{00,05}/`.
