# 2026-10-28: stream N, pour_by_language as a thin wrapper over the ported pour

pour_by_language (Open, 800 steps). The checker scores 20 / 50 / 100 for the first / two / all three pours. A pour
counts when at least 95 % of a bottle's liquid sits in its bowl and at most 10 % is left in the bottle. Scoring and
the order check both happen when both arms are at home (rising edge). At such a moment, a later bowl filled before an
earlier one fails the episode.

Built: route "pour_order" (`dojo/skills/pour_order.py`, behind `RuntimeConfig.new_routes`, default off; route-only,
`library.ROUTE_ONLY`).
- Pairs and their order come from the instruction's clauses: "from the <colour> bottle into the <colour> bowl".
- Each colour word is assigned to one bottle / bowl by the least total colour distance over the named colours, using
  perception's colour name or else the catalogue mean colour. The catalogue colours: bottles red 255/0/0, turquoise
  0/200/151, violet 123/97/255; bowls black 11, white 221, brown 63/48/38.
- Each pour is stream P's `ported(sk, "pour_bottle", ...)` (robot-harness's pour_bottle, D-C14 args). The pours run
  strictly in order (lead's ruling). One that did not run its whole procedure stops the sequence.
- The ported skill homes both arms after each pour. That is what banks 20 / 50: at a final home alone, a middle pour
  that failed under a later one that worked matches no score tier.

Not measured. P's physics-only G3 latches the fluid check rarely (pour_liquid 0/9, _random 1/6). One ported pour
takes ~300-390 steps (P), so 800 steps hold two pours, not three. The lead's call: measure once P's rendered camera arm
shows whether the fluid latches in the full sim. Then make it step-lean: the next arm pre-grasps during a pour, and
there are fewer trips home.
