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

## Follow-up: series_link_raise tipped vocab_explain to solve_equation (Integration slrchk, 8e7f6502)

**What happened.** In 5 of 10 dev runs (arrange ×2, 100000-04, slr on), `vocab_explain` chose solve_equation:
- arrange found 9 of 10 labels (the raised fifth digit not found) at mean confidence 0.83, score 0.75;
- solve_equation "found" every label at 0.37, score 0.37. Its config-pinned slots are always placed, at low confidence.
- The rule lets a set that finds a larger share of its labels replace the incumbent at any score. Both candidates came
  from the catalogue fallback (no table match), so the instruction never weighed in beyond the candidate list.

**The fix.**
- A raised label's `_not_found` entry carries `raised: true`. `_explain_vocabulary` doesn't count it: the label was
  only looked for, so not finding it is no evidence against the set.
- `RuntimeConfig.vocab_found_needs_score` (default off) adds a weighting guard: a set that finds a larger share wins
  only if its score is within `vocab_explain_margin` of the incumbent's. With it on, the 0.37-vs-0.75 swap can't happen
  for any reason.

**Tests.** Regression test with arrange (9 found + 1 raised) vs solve (all found at 0.37): arrange is chosen. Without
the raised flag, solve is chosen, as before; with the guard on, arrange is chosen.

## Follow-up: the agent filled a raised digit label (slrchk2 lnumr 100001)

**What happened.** B1 left the raised fifth digit "rigid0" `_not_found` (`raised: true`). The agentic perceiver then
matched that not-found label with an object at (-0.395, -0.205). It came in unread, as a placeholder number/00000, and
12 wrist looks never read it (a clutter spot in the random variant). The skill stopped at "digits unknown: ['rigid0']".
Without the raise, it would have placed the four read digits on five correct pads.

**Fix** (`agentic/record.merge_into_record`). A raised label of a digit category takes an agent object only with a
digit read. For a 3-D numeral, the digit comes from the glyph reader alone, so the agent never fills a raised numeral
label: the label stays `_not_found` ("4 read + 1 not found"). Not-found labels that weren't raised are filled as before.
