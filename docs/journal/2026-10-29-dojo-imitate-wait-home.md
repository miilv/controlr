# 2026-10-29: imitate after detection: the copy starts while the demonstrator returns (`imitate_wait` "home"); re-entry keeps the halves (`imitate_keep_sides`) (stream W)

Context: after detection, what loses imitate's score on real runs? Dev camera rows only. 0 LLM calls.

**What the dev rows show (100002: 3 of 3 runs, the batch-1 / r2 / smask rows on two shas)**
- The watch sees all 5 departures (the last at step 585).
- `wait_demo_done` ("origin") returns "still after 79" at step 664, and our left arm moves at 664.
- RoboDojo's first query ends the episode at 669: our arms away from their origin while `support_arm0` is away from
  its own (pos 0.3 / rot 40). Score 0, before any pick.

**Why "still"**
- Dev 100002's demo tail: physics-only replay with the arms held, frames every 4 steps, Isaac 6.0 proxy RGB.
- The runtime's motion read (head frame vs 12 steps back, outside our arms) sees 1-22 changed px over steps 648-672.
  Meanwhile the demonstrator's joints move 1.78 -> 0.83 rad from home.
- The arm returns outside the head view, so 12 calm reads in a row trigger while it is mid-return. GT home is at
  ~708 (|dq| 0.003).
- 100004's tail has no such blind window: the read sees the return until 736.

**What the flag does: `RuntimeConfig.imitate_wait` "home" (default "still"; G3 skill attr too)**
- As "origin", then `wait_support_home` holds until `provider.support_at_home()` reads the support robot back at its
  home, every 2 steps, at most 400, while one copy still fits.
- At eval the read is `inverse_layout.support_home`: the head image inside the Franka's mask vs its stored appearance
  (`data/support/`, #344), at home within 15 mean |RGB|.
- In G3 the read is GT joints vs the episode start.
- Unknown proceeds as before: no reader, or no support robot at home at frame 0 (`support_home0`, event
  `support_home0`).
- Dev demo tails, proxy renders:

| layout | at home (diff) | returning (diff) | home read first true | GT \|dq\| then |
|---|---|---|---|---|
| 100002 | 9.8-13.5 | 16-77 | 708 | 0.003 |
| 100004 | 9.8-13.5 | 16-77 | 728 | 0.007 |

- The runtime renderer's frame 0 reads 0.9-2.4, so the 15 threshold has margin both ways.

**The other two dev rows (smask, B perception)**

| row | score | copy started | what happened |
|---|---|---|---|
| 100001 | 0.3 | step 838, after the demo | 2 of 5 copied, then a re-plan (a placed object read "fallen") |
| 100003 | 0 | step 808 | the first twin, our side's "aim3", had 0/296 reachable grasps; 3 tries, then the sequence stopped |

- **100001.** The re-entered `copy_demonstration` recomputed the demonstrator's half as the far half of the objects as
  they now stand. Our two objects in basket0 moved that line, our toy car counted as the demonstrator's, and the copy
  ended at "no free object like toy_car" (2/5). `RuntimeConfig.imitate_keep_sides` (default False; G3 skill attr)
  keeps the first entry's halves.
- **100003.** A frame-0 detection loss: t4 was not found, so the twin was a phantom. That is the detection lever's
  ground (#344), not this change.

**Classes on the dev rows, by cost**

| class | rows | cost | fix |
|---|---|---|---|
| copy started while the demonstrator returns | 100002, 3 of 3 runs | the episode ends at 0, before any pick | `imitate_wait` "home" |
| re-entry recomputes the halves | 100001 | the copy stops at 2/5 | `imitate_keep_sides` |
| phantom twin from a frame-0 miss | 100003 | 0 | detection (#344); not built here |

The camera check of both flags needs Integration's dev rows (100001-03, B perception + "home" + keep_sides).
