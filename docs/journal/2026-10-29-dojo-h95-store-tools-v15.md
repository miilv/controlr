# 2026-10-29: store_tools_in_toolbox on v15 dev draws: the classes; regrasping a loose pick is negative (stream H)

Method: G3 mode gt (physics-only twin, GT world, 0 LLM, traced).
- Layouts: my own v15 dev draws, store_tools_in_toolbox 100030-49 variant 1 (simlab/v15, main 53f10437), presettled
  like Integration's (compute3:~/dojo-data/h95/v15dev). 20 of 20 settled.
- Today's defaults. No measurement ids were used. Time-boxed at ~2 h (the lead).

## The ceiling and the classes

- Base: n = 11 episodes (100030-33, 35-37, 40-43), mean 0.59, 2 successes (100032, 100042).
- Planning time binds the screen: plan_s is 97-986 s per episode, mostly the grasp-place pair / relay search, so a
  30-min G3 job finishes only 3-5 episodes (BACKLOG "planning time on irregular meshes").
- Tools placed, of 7 base episodes read in full: tape measure 6/7, hammer 4/7, wrench 3/7, pliers 3/7.

The 11 tools lost in those 7 episodes:

| class | n | what |
|---|---|---|
| moved in the hand during the carry / place | 7 | "in-hand shift" 45-394 mm, "re-plan failed", landed 20-390 mm off and 18-89 deg tilted. All four tools; most after a "loose" pick (pliers slice grasp 8-9 mm drift; hammer box grasp tilted 15 deg, 8.7 mm) |
| no grasp-place pair at the slot | 2 | hammer: the left arm reaches 134/148 grasps, 0 pair with an IK at its slot, and the right reaches none; pliers likewise; "no in-air handover for a 228 mm object" |
| pick slip | 1 | wrench box grasp 34 mm off its centre, rose 49 of 80 mm |
| landed out of tolerance, still counted | 1 | pliers 20.5 deg tilted: the verify says miss, the checker counted it |

## Tried: `slot_loose` "regrasp" (branch `feat/dojo-h95-slot-regrasp`, not merged)

- Inside `place_in_slots` only (`Skills._loose_regrasp`, scoped and reset), a loose pick is put down and its leg
  re-planned, as tighter places already do (`run_transfer`'s `LOOSE_OK_TOL` rule).
- v2 also stops offering the grasps that came out loose for that object.

| arm | 100030 | 100031 | 100032 |
|---|---|---|---|
| base | 0.75 | 0.75 | 1.0 |
| regrasp (same grasp again) | 0.75 | 0.50 | - |
| regrasp v2 (another grasp) | 0.75 | 0.50 | 0.75 |

- The same grasp comes back loose: 2 of 2 re-picks.
- Another grasp is loose too or slips: the pliers' next slice (z 0.016) slipped (rose 15 mm); the hammer's next grasp
  held at the pick but still moved 90 mm in the carry.
- The base had carried those loose pliers into their slot. This matches skills.py's own note ("a hammer held at its
  centre of mass with 12 mm drift was put down twice and never placed").
- **Parked, no PR.** What binds is the carry and its dynamics, not the pick verdict: the hammer moves even after an
  "ok" pick. The next lever is in the carry, and the 2 pairing failures are reach at the slot.

Spend: 0 LLM calls. G3 on compute3: 7 store_tools jobs (30 min each, 3-5 episodes) and 1 presettle; one at a time.
