# 2026-10-30: sort_nesting_dolls camera census, and row_size_read (stream N)

**Census.** 10 dev camera rows (100000-04, std and `_random`, deploy 57dd7f36, the candidate config with
save_look_frames). The belief at the row decision (`dev_belief_probe`'s first snap) is compared with GT. The checker ranks
dolls by `category_idx % 5`, and within every texture set that is the height class: 130 / 110 / 90 / 70 / 50 mm (the
`_random` sets are 128 / 110 / 91 / 69 / 50). The sets differ in width.

| class | rows |
|---|---|
| size order wrong at the decision | 5 / 10 (`_random` 4 / 5, std 1 / 5) |
| a shared believed size class (two dolls read as one height) | 7 / 10 |
| order right, a doll not placed (ik_fail, 0 grasp-place pairs) | 2 (std 100001, 100004) |
| order right, all placed, checker 0 (cause not in the report) | 2 (std 100003, `_random` 100002) |
| success | 1 (std 100002, its order right by a lucky tie) |

- **std:** 4 of 5 rows had exactly one doll one class off, always onto another doll's class. Three of those four orders
  came out right only through the tie-break.
- **`_random`:** perception searched only the std pool, because the vocabulary is the std set (one instruction for
  both). The texture set was right 0 / 5 in every row, and the dolls read one class bigger (wider at the same height).
- **xy at the decision:** ≤ 4 mm for 44 of 50 dolls. Gross only in `_random` 100004 (33 / 38 / 50 mm), plus one 26 mm
  doll in std 100000.
- **Pick-hover identity reads** (`refine_identity`) changed 5 dolls' sizes mid-episode, 4 of them to the truth. All of
  them came after `row_by_size` had fixed the order, and it never re-plans it.
- **Stereo height read** (head + wrist plane sweep on the frame-0 / scan / look frames, top-down max height per doll):
  right within ~10 mm when it reads at all, but it misses most dolls in each frame and has 100 mm outliers in the scan
  frames. Dropped.

**`RuntimeConfig.row_size_read`** (default off; `Skills.row_size_read`). Before the order is decided, `rows.size_reads`
does a wrist look straight down at each doll whose believed height class another doll shares:
- The trigger is the instruction's strict order, which needs distinct sizes. No size is assigned from the config.
- Each look calls the refine hook of kind "size", which runs the pick hover's identity read over the label's config group.
  "look_union" also adds the std / `_random` twin config's candidates (one instruction, both pools).
- Order: nearest an arm first. The looks stop when no class is shared, after `MAX_SIZE_LOOKS` (4), or when the steps left
  fall under 80 per doll.
- Then both arms go home, and the order ranks by height (`rows.height`, in 8 mm classes, then volume). A volume misranks
  dolls believed from different sets: std 13 is 70 mm tall and 244 cm3, std 2 is 90 mm and 232 cm3.

Off, nothing changes: `row_slots(by="volume")`, no looks.

**Not yet measured.** G3 has no perception, so it can't test the reads. The A/B needs dev camera rows: off vs look vs
look_union, std and `_random`.
