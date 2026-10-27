# 2026-10-27 — build_tower pieces never proposed: blobs merged with the pinned base boards and with each other

**Question.** Integration's build_tower scan-trigger run (proxy B, 25 paired layouts, docs/experiments/
2026-10-26-scan-trigger-tower.md) still lost 7 episodes to perception_gross. Which of them are B1 misses at frame 0,
why does B1 miss a piece in full view, and does a fix hold up on the G1 sets?

**Data.**
- **Trigger-run frame 0s.** The saved frame 0 of all 25 trigger-run episodes (`B_build_scantrigger_*`, 6.0 proxy,
  compute3).
- **Truth.** The settled dev layouts. Identical pieces are matched by position. Within tolerance = build_tower's
  23 mm / 5 deg.
- **G1 regression.** D0 on genuine Isaac 5.1 frames (`d0_51`, 5.1 calibration) and D0-ext (6.0 proxy), frame 0,
  main vs this branch, light slot 7 on compute5-8.
- No LLM calls.

## Attribution (trigger arm, 3ac3597)

| class | layouts | note |
|---|---|---|
| phantom stacked pillar (z +38 mm, scan-primary record) | 100011, 100014, 100022 | removed by #62 (on main) |
| missing piece, fully visible | 100000, 100010, 100022, 100024 | known 0.83-1.00, foreground 0.96-1.00, nothing covers it; 100024 fine on main |
| shape / place confusion | 100008 | board 00001 and a pillar ~110 mm off each |

## Cause

- **The pinned base boards are placed before the proposals, but their pixels stay in the head foreground.** These
  are block0 / block7, at config slots. A piece within a few cm of a board merges into the board's blob. The merged
  blob is more than 3x the piece's silhouette, and `_propose` skips blobs that size.
  - 100000: board blob 6.9k px + pillar 1.0k px = an 8.2k "unexplained" blob.
  - 100010: board + board 00001 + pillar.
- **Pieces lying together form one blob that no candidate can take whole.** 100022: three pieces within 10 cm,
  3.5k px.

## Fix

Two `EstimatorConfig` defaults, both on:
- `slot_blob_cut`: the pinned objects' silhouettes are cut out of the head foreground before blobbing and in the
  unexplained list.
  - The residual rounds keep the pinned objects' pixels. With them cut there too, organize_table 100000's figure in
    front of the pinned cushion lost its residual blob and became the wrong figurine (0.4 -> 40 mm).
- `split_merged`: a blob larger than 2.5x the largest pending candidate's silhouette also proposes object-sized
  pieces (`_split_blob`). The whole blob stays a proposal.

## Results

**Trigger-run frame 0s (6.0 proxy, 25 layouts, 200 pieces, tower tolerance 23 mm / 5 deg; the first version, before the residual-round change).**
- Within tolerance 166 -> 174, missing 8 -> 3.
- 8 pieces fixed (100000, 100010 x2, 100011, 100014, 100022 x3), 0 lost.
- 100011 / 100014's frame-0 pillar errors were the ones the scan later turned into stacked phantoms.

**G1 regression, main @6dd647c vs this branch, same box, frame 0.**

| set | task | n | within tol main -> fix | missed | gross |
|---|---|---|---|---|---|
| d0_51 (genuine 5.1) | build_tower | 200 | **164 -> 173** | **13 -> 4** | 19 -> 17 |
| d0_51 | stack_blocks | 75 | 60 -> 60 | 5 -> 5 | 4 -> 4 |
| d0_51 | push_T | 50 | 41 -> 41 | 0 -> 0 | 8 -> 8 |
| d0_51 | stack_bowls | 75 | 67 -> 67 | 0 -> 0 | 1 -> 1 |
| d0_51 | put_bottles_into_dustbin | 125 | 104 -> 104 | 3 -> 3 | 11 -> 11 |
| d0_51 | pack_objects_into_box_random | 125 | 89 -> 89 | 2 -> 2 | 28 -> 28 |
| d0_ext (6.0) | make_kong | 220 | 220 -> 220 | 0 -> 0 | 0 -> 0 |
| d0_ext | press_by_number | 50 | 50 -> 50 | 0 -> 0 | 0 -> 0 |
| d0_ext | cover_blocks | 60 | 60 -> 60 | 0 -> 0 | 0 -> 0 |
| d0_ext | organize_table | 90 | 69 -> 69 | 5 -> 5 | 13 -> 13 |
| d0_ext | store_tools_in_toolbox | 50 | 48 -> 48 | 1 -> 1 | 1 -> 1 |
| d0_ext | general_pickup | 10 | 7 -> 7 | 0 -> 0 | 3 -> 3 |
| d0_ext | fill_pen_holder | 50 | 49 -> 49 | 0 -> 0 | 1 -> 1 |
| d0_ext | swap_T | 20 | 20 -> 20 | 0 -> 0 | 0 -> 0 |
| d0_ext | align_blocks | 40 | 34 -> 34 | 0 -> 0 | 2 -> 2 |
| d0_ext | plug_in_charger | 20 | 15 -> 15 | 1 -> 1 | 4 -> 4 |
| d0_ext | insert_key | 20 | 19 -> 19 | 1 -> 1 | 0 -> 0 |
| d0_ext | stack_blocks_by_language | 30 | 28 -> 28 | 1 -> 1 | 0 -> 0 |

- **build_tower d0_51:** 11 pieces change class, all for the better: 9 missed -> found within tolerance, 2 gross ->
  ok. Every other set is identical object for object.
- **organize_table:** the first version lost organize_table 100000's garage figure (0.4 -> 40 mm), from the residual
  rounds. The final version keeps those rounds as on main; the organize, tower and slot-task rows above are its runs.
- **A variant that added the cut blobs** (keeping the uncut ones too) was worse on the tower: 170, with 6 new gross
  pillars from the uncut merged blobs.

**Not covered.** 100008's board / pillar trade (both ~110 mm off) and the 3 pieces still missed on d0_51 (100005,
100012, 100020) are different cases.

0 LLM calls; GPU light slot 7 on compute5-8.
