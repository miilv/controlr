# 2026-10-28 — Stream P: general_pickup, our pick first, then the ported pick_up (`ported_pickup=library`, default v1)

**Why.** The ported pick_up (Mikhail's top-down mesh-cloud grasps) scored 8/10 in G3 and 4/8 on the camera arm. Its fails
are repeated failed grasps on irregular targets: the bicycle (100006), the origami crane (100008). Each one runs the
200-step budget out.

**Done** (branch `feat/dojo-port-pickup-library`):
- `dojo/skills/pickup.py` `lift_up(sk, obj, by, fallback)`:
  - our `Skills.pick` with a lift of by + 5 cm, one more try after a slip;
  - when ours does not lift it and ≥ 120 steps remain, robot-harness's ported pick_up on the same object (`"@<label>"`).
- The route grounds the object first by its public description.json (exact phrase), then by its category words.
- It is behind `RuntimeConfig.ported_pickup="library"` / G3 `--skill-attr ported_pickup=library`; the default "v1" is the
  ported pick_up alone. Approved by the lead as one generic skill for that instruction.

**G3 mode gt** (compute3, settled v0 100000-09, one job at a time):

| arm | success | notes |
|---|---|---|
| v1 (ported pick_up, 5bfe623) | 8/10 | 100006, 100008: step budget |
| ours alone (17eab65) | 6/10 | 100002 / 03 / 07: no feasible grasp after 3 steps (planning); wins 100006; 29-61 steps |
| **ours, then v1 (f5b7a96)** | **9/10** | the union; 100008 (origami crane) fails both |

Row: compute3 `~/dojo-data/g3c/general_pickup_f5b7a96/`.

**Spend:** 0 LLM; 20 short G3 jobs on compute3.
