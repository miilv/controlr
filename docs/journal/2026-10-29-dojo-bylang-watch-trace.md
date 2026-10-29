# 2026-10-29: where B1 loses classify_objects_by_language's watches (stream S, diagnosis)

On p60, the watches (watch 4-7) are the biggest perception loss in classify_objects_by_language. Swapping the
segmenter (v2b, v3ep6, v2cn) changes nothing (journal 2026-10-29-dojo-seg-v2c). This entry traces where B1 drops them.

- **Frames: dev only.**
  - p60 100000-04 (6.0 proxy; the bake's `cfg_link_off.json`, v2b server).
  - Genuine-5.1 renders 100030-37 v1 (render_v3new; each carries 3 extra-pool objects, so their counts are not the
    task's).
  - No measurement id was opened.
- **Tool:** `scripts/perception/sam/b1_trace.py`. For one frame it records:
  - the head blobs and the segmenter masks;
  - the series count and the seg-propose trace;
  - per GT object, the blobs and masks over its pixels and the nearest prediction.
  - It refuses the measurement ids.
- **Outputs:** compute3:~/dojo-data/s-watch/{trace,probe}_{p60,r51}.jsonl. 0 LLM calls.
- **Lane:** classify × 2 went back to Misha's side today, so nothing below is adopted or A/B'd.

## The record on the 5 dev frames

| frame | objects (GT) | B1 members | watches right |
|---|---|---|---|
| 100000 | 9 (3 car, 3 watch, 3 pepper) | 6 | 0 / 3 |
| 100001 | 6 | 6 | 1 / 2 |
| 100002 | 6 | 6 | 0 / 2 |
| 100003 | 9 (3 chocolate, 3 pepper, 3 watch) | 6 | 0 / 3 |
| 100004 | 6 | 6 | 0 / 2 |

All 6-9 objects sit under one label group (`rigid0`, prefixes cat0-2).

## Classes

1. **The series count stays at the range minimum.**
   - `rigid0`'s range is [6, 9] (step 3). The "seg" count counts masks of the group's exact classes.
   - v2b has no class for car 10-15, pepper 0-2 or watch 4-7. Cars come out as tape_measure / book / toy_car, and
     peppers as tape_measure or nothing. So the count is 0-1 on 5 of 5 dev frames and 8 of 8 renders, and n = 6.
   - The two 9-object frames (100000, 100003) can place only 6. 5 of the 11 missed watches are in them.
2. **Basket-shadow phantoms take member slots.**
   - The head foreground includes the baskets' shadow on the table in front of the left and middle baskets. It is one
     blob of 2400-3400 px, merged with whatever lies there. Its unexplained part is ~1000-1500 px around
     (-0.26, -0.08) in every frame.
   - B1 fits 1-3 members on it at confidence 0.04-0.16: 8 of the 30 members on 100000-04. Their categories include
     ones the instruction does not name (chocolate_bar and wooden_toy on 100004).
   - The 5.1 renders show the same low-confidence members in front of the baskets, 1-6 per frame.
3. **Thin metal rings break into fragments.**
   - A watch's foreground comes in 2-3 pieces of 21-104 px each: 7 of the 11 misses. None is proposed while a phantom
     or a bigger blob is free.
   - A watch with one coherent blob of >= ~190 px is found: 100001's watch 4, and 4 of 4 such watches in the renders.
   - 2 misses are watches merged into the shadow blob (100000, 100003 watch 7). 2 are hardly visible (56 / 90 GT px,
     known 0.17-0.40, under a wrist gripper: 100002 and 100004 watch 4).
4. **The segmenter's watch masks are refused.**
   - v2b masks 3 of the 12 dev watches as watch/00000-01 (0.44-0.97). One of these is explained by its member.
   - It masks 1 more watch as garage/00014.
   - seg_propose "add" refuses the unexplained watch masks: "add: not the group's exact class" (the candidates are
     watch 4-7). The same happens to 4 more in the renders.

## A probe, default off: `seg_add_category` (PR, not adopted)

**The change.** With "add", an unexplained mask of the group's category may add or move a member, not only one of the
exact class. The fit uses that category's shapes only. With `seg_add_move_weak` (default on), a weak member, which here
is usually a shadow phantom, moves onto the watch mask.

**Results (dev only, no gate):**

| frames | watches right | all objects right | losses |
|---|---|---|---|
| p60 100000-04 | 1 -> 2 of 12 | 20 -> 21 of 36 | none; 100000's phantom at (-0.153, -0.053) moved onto watch 7's mask, IoU 0.63 |
| 5.1 renders 100030-37 | 5 -> 8 of 15 | 35 -> 37 of 81 | 1: 100034's pepper 2, a correct member at confidence 0.11, moved onto a watch mask |

It needs the frame-0 bake on all tasks and paired runs before anyone adopts it.

## What the probe does not touch (for the cell's owner)

- **Count.** A category-level seg count would not reach 7 with v2b, because the cars and peppers are not seen as
  their categories. The count needs:
  - B1's own count of object-sized unexplained blobs once the shadow is gone, or
  - a segmenter with by_language classes. v2cn has them, but detected these watches weakly.
- **Shadow.** A scene background that includes the baskets would remove the phantoms and the merged blob. The baskets
  sit at fixed poses in classify_objects and by_language. A shadow-tolerant foreground (darker, same chroma) would
  also work, but changes every task.
- **Fragments.** Merging nearby small foreground pieces (each < ~110 px, a few px apart) into one blob before the
  proposals would help. Thin rings, the watches here, split.
