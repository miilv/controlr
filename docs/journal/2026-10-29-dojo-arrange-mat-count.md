# 2026-10-29: arrange's mat row "one pitch off" is a count error (stream W)

**Report.** In 3 of 9 arrange camera runs on 3ad78f28 (largestnumber 100008 / 100009, random 100008), every mat
estimate showed ~85 mm of error, exactly the mat pitch. All ended `perception_gross` with score 0-0.15.

**Not a slide.** `runtime/eval.perception_eval` keeps only the GT Geometry labels that appear in the record. B1 placed 4
mats, so GT mat_4 is dropped. The min-sum assignment then ties: 4 × 85 mm = 3 × 0 + 340. It picks the label-wise
pairing, so every mat shows one pitch. In fact B1's 4 mats sit on 4 of the 5 real mats. The w51 bake gives the same 4 at
0-1 mm plus 1 not found. (Scorer caveat: a missed Geometry object shows up as offsets, not as a miss.)

**The count.** On every 5-digit w51 layout (17 / 17 across both variants), B1 placed 4 digits + 4 mats:
- v2b sees at most 4 of the small numerals;
- series "seg" therefore falls to the range minimum of 4;
- the linked mats (one per digit, `linked_count`) follow at 4;
- yet the fifth mat was detected (cube_cushion 0.97, logged "group full").

**`EstimatorConfig.series_link_raise`** (default off). A linked_count group's own exact-class head masks raise its
series' count within the series' public range [4, 5], and never lower it. The raised labels are only looked for:
- the occluded fallback never fills them;
- with `glyph_read`, a raised member counts only if the reader reads it as a digit (`_drop_unread_raised`);
- otherwise the label goes to `_not_found`, which gives "4 read + 1 not found", never a placeholder.

The relation used is the config's public linked_count, which was already used one-way. There is no task keying.

**A/B on w51** (the only frame set with arrange frames; the flag changes nothing else by construction). Both arms use
the current defaults. Glyph off: the bake has no glyph server.

| | digits found | mats found / ok10 | phantoms |
|---|---|---|---|
| arrange 5-digit (45 each) | 38 -> 43 | 38 / 38 -> 45 / 45 | 0 -> 0 |
| arrange_random 5-digit (45 each) | 36 -> 44 | 36 / 35 -> 45 / 43 | 2 -> 2 |
| 4-digit layouts (both) | unchanged | unchanged | unchanged |

On the 5-digit layouts, digits within 10 mm went 65 -> 75 of 90. 3 of the 13 new fills sat 126-686 mm from any digit:
fits on other foreground, which the phantom count does not show because they match the unfound GT digit. That is why
the glyph guard was added. It can only remove a fill, so with glyph on the worst case is "4 read + 1 not found". The
guard's own effect needs a glyph server: Integration's dev check (arrange ×2, 100000-04).
