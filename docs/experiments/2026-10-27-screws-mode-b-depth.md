# 2026-10-27: fasten_screws in mode B: a sweep model instead of the belief's nut depth (G3 mode B, 10 paired layouts)

**Question.** With #77's fixtures rule (a released nut stays on its bolt in the belief), why does G3 mode B score 0.39
with every nut placed and turned (mode A: 24/25), and does modelling the nut's depth fix it?

**Setup.**
- G3 mode B: physics-only twin, the TRUE state behind `--mode-b --belief-offset nut0:0.002,0 --belief-offset
  nut1:0,0.002 --belief-offset nut2:-0.002,0 --belief-drop fixtures`.
- Settled dev layouts 100000-100009, `SIMLAB_USD_SYNC_LINKS=1`, default `DOJO_CATALOG`, compute3 light lanes. No LLM
  calls.
- Before: Integration-3's run on 5ef91bb (main 48600bf + #77 + #79). After: deaeab7 = 5ef91bb + `screw_on`'s mode-B
  depth model (journal 2026-10-27-dojo-precision.md §2). Same layouts, offsets and hooks.

**Results.**

| layout | before score (steps) | after score (steps) |
|---|---|---|
| 100000 | 0.5 (998) | 1.0 (1245) |
| 100001 | 0.5 (1048) | 1.0 (1288) |
| 100002 | 0.5 (1062) | 1.0 (1294) |
| 100003 | 0.5 (977) | 1.0 (1209) |
| 100004 | 0.0 (1117) | 1.0 (1349) |
| 100005 | 0.5 (956) | 1.0 (1150) |
| 100006 | 0.5 (996) | 1.0 (1228) |
| 100007 | 0.2 (1082) | 1.0 (1314) |
| 100008 | 0.5 (1121) | 1.0 (1353) |
| 100009 | 0.2 (994) | 1.0 (1226) |
| **total** | **0/10, mean 0.39** | **10/10, mean 1.00**; steps median 1266, max 1353 of 1900 |

- Before: 3 sweeps per nut. The belief went down with the hand, 2.6 / 2.7 / 2.8 mm per sweep, against mode A's
  measured 0.7-0.9 then 2.2-2.5. The loop stopped at a believed 11.3 mm with the nut truly ~8.5 mm down, short of the
  checker's 9 mm. The success stage never latched (`stages_left` 1).
- After: 4 sweeps per nut, as mode A does.

**Conclusion.** Rescued 10, lost 0 of 10 paired layouts.
- This is G3 mode B, i.e. the belief emulated by `heldsim`. The runtime may still differ, where perception's post-place
  re-measure of a nut on its bolt can update the belief: the min(belief, model) rule keeps the deeper turn either way.
- Next: proxy / rendered B once #77 is on main.

Data:
- compute3 `~/dojo-data/precision/g3/screws_mbfx_model_deaeab7/`;
- compute3 `~/dojo-data/integ3/g3/screws_mb2_dfx_5ef91bb/`.
