# 2026-10-26 — objects in or on a fixture's support slot (make_toast bread, play_Xylophone mallet)

**Question.** In integration-2's breadth run (mode B, proxy), two tasks scored 0/5 because one object was never found
in any record:
- **make_toast:** the four bread slices standing in the bread shelf;
- **play_Xylophone:** the mallet resting on its stand.

Why, and does one fix cover both?

**Cause** (`dojo/perception/inverse_layout.py`).
- **The config pins each object in a support slot of another object** (relative_plane "<base>/<support key>",
  pinned offset, no random turn):
  - the bread: "bread_shelf/bread_shelf/k", offset (-0.02, -0.007);
  - the mallet: "mallet_stand/mallet_stand/0", offset (0, 0).
- **B1 has no path for that.**
  - The slot branch takes only zero-offset labels.
  - The "contained" branch takes container categories only.
  - Table-resting proposals cannot fit a slice standing 4 cm up in a slot, or a mallet lying on a stand, so every
    record reads "no foreground explained".
- **The shelf itself was also missed in 3 of 10 dev layouts.** From the head view it is mostly hidden behind its four
  slices, so the shelf alone fits nothing.

**Fix.**
1. **Slot groups.** A label pinned in a support slot of another label, whose base asset has that support frame in the
   catalogue, is taken out of the blob proposals.
2. **Slot poses.** Each slice is placed from the chosen base: the base pose combined with the public
   `passive.support` frame and the pinned offset (`support_slot_pose`). This reproduces our layout generator exactly
   (0.0 mm, 0.0° on 10 layouts).
   - **Settled poses.** The generator's pose is the DROP pose. Physics lands every slice 16-17 mm lower, leaning
     4-6°, identically in every layout. `slot_settle.json` holds the settled relative poses (our physics-only Isaac
     5.1 settle of the 10 dev layouts per task). Spread: bread ≤ 3.6 mm; the mallet on its stand ≤ 1.8 mm in either
     stand orientation.
3. **Per-view check.** The pixels only the slice would explain must be foreground. An empty slot is reported as
   `_not_found` "support slot empty". Confidence is capped by the base's.
4. **Assembly proposals.** The base is also proposed as an ASSEMBLY: shelf plus the settled slices as one mesh, with
   the same frame, so a pose of the assembly is a pose of the shelf. It is refined as one shape, then reported as the
   shelf.
5. **Catalogue.** The bases' support frames had to be annotated (`catalog annotate`: `Geometry/bread_shelf/00000`,
   `Rigid/mallet_stand/00000`). Without them, nothing changes.

**Data.**
- **`d0_toast` / `d0_xylo`.** Our dev layouts 100000-100009 per task, frame 0 plus frame 1 (left wrist hovering 10 cm
  over the base). States are physics-only Isaac 5.1 (truth = the settled poses); RGB is the Isaac 6.0 proxy.
- **Breadth frames.** Integration's breadth episodes 100000-100004 per task, frame 0. Their truth is the generator pose
  only, so only "found" is counted.
- GPU light lane, no LLM calls.

## Results

**make_toast (bread in the shelf):**

| set | variant | shelf found | bread found | bread within 15 mm | bread xy median / p90 |
|---|---|---|---|---|---|
| d0_toast frame 0 | main | 7/10 | **0/40** | 0 | — |
| d0_toast frame 0 | slots (no assembly) | 7/10 | 28/40 | 25 | 3.3 / 15.5 mm |
| d0_toast frame 0 | **slots + assembly** | **10/10** | **40/40** | 25 | 8.3 / 41.9 mm |
| d0_toast frame 0 + wrist look | main | 7/10 | 0/40 | 0 | — |
| d0_toast frame 0 + wrist look | **slots + assembly** | **10/10** | **40/40** | **40** | **1.0 / 1.6 mm** |
| integ breadth frame 0 | main | 5/5 | 0/20 | — | — |
| integ breadth frame 0 | slots + assembly | 5/5 | **20/20** | — | — |

**play_Xylophone (mallet on its stand):**

| set | variant | stand found | mallet found | mallet within 15 mm | mallet xy median / p90 |
|---|---|---|---|---|---|
| d0_xylo frame 0 | main | 10/10 | **0/10** | 0 | — |
| d0_xylo frame 0 | **slots + assembly** | 10/10 | **7/10** | 5 | 1.3 / 24.3 mm |
| d0_xylo frame 0 + wrist look | main | 10/10 | 0/10 | 0 | — |
| d0_xylo frame 0 + wrist look | **slots + assembly** | 10/10 | **10/10** | **9** | **0.8 / 16.7 mm** |
| integ breadth frame 0 | main | 5/5 | 0/5 | — | — |
| integ breadth frame 0 | slots + assembly | 5/5 | **4/5** | — | — |

- **Frame-0 mallet misses.** Three are "support slot empty": the stand was fitted 20-75 mm off at frame 0, so the slot
  check saw no mallet. The wrist look fixes them.
- **One remaining miss.** The mallet was placed 148 mm off on the wrong side: the stand's yaw is ambiguous by 180°.
  Its confidence was 0.23, so a look would re-check it.
- **Frame 0 alone is too coarse in 4 of 10 layouts.** The shelf sits near the left home gripper, so it ends up
  17-45 mm off. The slices inherit that error; their confidence is below 0.7 in 9 of the 15 slices more than 15 mm off.
- **One wrist view into the shelf makes all 40 exact.** That is the runtime's look_plan look (confidence < 0.7) or the
  scan.
- **Regression.** D0 frame 0, stack / tower / push × 10, main vs this branch: identical (130 objects). The slot path
  only runs where a base's support frame is in the catalogue.
- An earlier variant re-fitted the assembly AFTER a shelf-only selection. It moved good shelves 10-20 mm, so it was
  dropped.

## Caveats / next

- **Bread off the shelf is not proposed by B1 in this task.** A slice dropped on the table mid-episode is invisible to
  the generic proposals. The runtime belief and the agentic perceiver cover it.
- **Other tasks with support slots are unchanged until their bases are annotated:** deposit_coin, fill_egg_holder,
  the conveyor tasks.
- **`slot_settle.json` covers bread / shelf and mallet / stand only.** Another slot task needs its own settle (one
  physics-only make_d0 run, then `slot_settle_build.py`).
