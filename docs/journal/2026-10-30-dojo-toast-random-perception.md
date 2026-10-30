# 2026-10-30: make_toast_random frame 0: a twin vocabulary gap and a segmenter gap; `twin_union_pool` "weak" (stream W)

Context: make_toast_random scores 0/5 strict (G3 77). H's census of the 5 dev camera rows (100000-04, be7a798b) shows
every row ending perception_gross, with every bread slice at conf ~0.45. H owns the grasp side; this entry is the
perception side. Dev only, 0 LLM calls.

**What it is.** The runtime routes the `_random` scenes to the vocabulary "make_toast" (same instruction, source
"table"). That causes two gaps.

1. **The routed vocabulary lacks the scene's toaster.**
   - make_toast's toaster candidates are 00000 / 00001.
   - The `_random` layouts (same placements, plus 15 clutter objects) hold toaster 00004 (100000-03) or 00002 (100004),
     both in the catalogue.
   - B1 fits a wrong toaster model: 7-40 mm off at conf 0.2-0.4, or not found (100002).
   - The skill's slot poses come from that model.
2. **The twin segmenter (v2b) has no class for toaster 00002 / 00004.**
   - Its 511 classes include toaster 00000 / 00001 only.
   - On the head view it detects the white 00004 toaster in 1 of 4 frames (as 00000, score 0.17), and 00002 as 00000
     at 0.67.
   - The white toaster's head blob merges with the clutter: 16-41k px against its own 9-10k, next to a white kettle and
     a doll house.
   - So even with the right vocabulary (oracle `make_toast_random`), B1 misses it (100000) or places it 138 / 304 mm
     off (100002 / 03).
   - Bread and shelf fit within 1-21 mm at conf 0.42-0.49 (std scenes: 0.68-0.83). On 100004 the shelf and all four
     slices are missing in every arm.

**Frame-0 results (service estimate, rows' cfg, dev 100000-04):**

| vocabulary / pool | toaster right asset | toaster within 10 mm | std toaster |
|---|---|---|---|
| routed (today) | 0/5 | 3/5 (wrong models) | 5/5 |
| routed + `twin_union_pool` on (#325, A/B not adopted) | 2/5 | 3/5 | 4/5 (100003 → 00000) |
| routed + `twin_union_pool` "weak" (new) | 2/5 | 4/5 | 5/5 (never runs) |
| oracle `make_toast_random` | 3/5 | 2/5 | n/a |

**`twin_union_pool` "weak"** (a third value; False / True keep their meaning)
- The routed estimate runs first.
- Only when a label the twin pool would extend (here the toaster) is missing or below conf 0.5, the union estimate
  runs too.
- The union estimate is kept when it finds as many of those labels at a mean confidence no lower than the routed
  one's minus 0.02.
- `_perception.twin_union` records what ran and which was kept.
- The std scenes' toasters fit at 0.87-0.98, so they never pay the second estimate or the asset swap. That swap is
  the global pool's std loss.

**Left for others**
- Segmenter classes for toaster 00002 / 00004 (and other `_random`-only assets): S's lane.
- The bread conf ~0.45 next to clutter, and 100004's missing shelf: not diagnosed.
- On dev only 100001 has every object right after the change, so expect at most +1/5 until the segmenter learns the
  `_random` toasters.

Tests: `tests/dojo/test_twin_union_weak.py`.
