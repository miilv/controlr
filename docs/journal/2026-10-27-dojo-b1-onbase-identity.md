# 2026-10-27 — B1: the laptop on its stand (on_base_refit) and the identity margin swap (both default off)

Follow-ups of the perceiver-models bake-off (report: `docs/experiments/2026-10-27-perceiver-models.md` §5, PR #195): no
model fixed store_laptop's flipped laptop or the headset identity, so both went to B1. Everything runs on CPU
silhouettes (`DOJO_GL=0`), with 0 LLM calls.

**Why B1 flips the laptop.** It does not prefer the 180° solution as such. The laptop sits on its stand (config:
`relative_plane laptop_stand/...`, xlim -0.05..0.05). That placement is neither the zero-offset stack, a support slot
nor a container, so B1 searched it as a table object.
- B1 fitted the screen as a laptop lying on the TABLE about 180 mm behind the stand, and the yaw fell out of that wrong fit.
- At the stand's height the silhouette alone picks the true yaw: on 100000 the loss is 0.45 at the true yaw vs 0.73 at
  the flip. No texture or height cue is needed.

**`EstimatorConfig.on_base_refit`.**
- What it does: once the base is chosen, a child that the config places on a static Geometry base (not a conveyor, not a
  container, not a support slot) with an offset range is refitted resting on the base's top. It starts from the base's
  middle with 8 yaw seeds, inside the base's footprint.
- When the refit wins: it replaces the table fit unless its loss is worse by more than 0.03 (`on_base_margin`), and a
  label the table search missed is looked for there (kept at a loss <= 0.5).
- Scope: of every task config, only store_laptop's laptop qualifies.
- Results on genuine-5.1 store_laptop frames (Integration's c4 runs; the scorer's GT is the settled record moved by the
  laptop's root-vs-mesh offset):

  | B1 path | fixed | unchanged | still not found | worse |
  |---|---|---|---|---|
  | frame 0 (10 frames) | 3: 183-200 mm / -176° -> 26-28 mm / -12..+26° | 3 | 4 | 0 |
  | scan-primary (7 frames) | 4: 43-193 mm / -170° -> 3-36 mm / 0..+12°, one of them found from not found | 1 (100004_v2) | 2 | 0 |

  On the D0 sets it changes nothing by construction: no task there qualifies.

**`EstimatorConfig.identity_margin_swap`** (polish: another asset whose combined cost beats the kept one by > 0.01
replaces it; off when the embedding decided).
- eval_g1 frame 0, off vs on, ids 100000-09 / 100020-49 only: d0 400, d0_51 400, d0_ext 643 and series 30 objects =
  1,473 objects. **0 changes** in failure class or identity.
- On the 17 laptop estimates it never fires. The negative headset margin (-0.04) came from the runtime's record and does
  not reproduce offline. There, headset 00002 vs 00004 has positive margins of 0.03-0.08, so a cost rule cannot separate
  them.
- **Verdict:** harmless and inert. The headset identity needs the Identity stream's embedding path (`embed_b1`), not a
  margin rule.

**Next:** a store_laptop G3 / rendered A/B with `on_base_refit`, once Integration has resolved the laptop's
root-vs-mesh offset (otherwise the twin puts the right estimate 80 mm off).
