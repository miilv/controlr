# 2026-10-29 — stream N: classify camera rows planned nothing (one belief label for six items)

**Symptom.** Every classify_objects and classify_objects_by_language camera row on compute4 (runs2,
`c4b_classify{,lang}sg_port_100005-08`, 3ad78f28) ended the same way. Decision "finish": `no skill for: '<instruction>'`,
score 0. Perception was mostly right: 6-9 items of the right categories, plus basket0-2.

**Root cause.** B1 (`inverse_layout._series_labels_out`) numbers a single-prefix series (`rigid2` -> `cube_0..k`). It keeps a
series with per-category prefixes under its config label, because which category the generator numbered first is not
observable. So all of classify's items are `rigid0`. `estimate.objects_from_record` keys the belief by label, and only
the last `rigid0` survived. The ported classify route needs >= 2 non-basket rigid items, so it did not fire. At G3 (true
state) the items carry GT labels (`cat0_0`, ...), which is why the ceiling rows ran.

**Fix (default off).** `RuntimeConfig.series_labels`: "keep" (today) | "numbered". With "numbered", a label the record holds more
than once names each of its objects `<label>_<k>` (k by x then y, `estimate.series_names`). This applies to every
`objects_from_record` path the runtime takes: belief init, resync, the belt track, and the tracker's write-back. The
record itself is not relabelled. `_label_entry` maps `rigid0_k` to its config entry for the twin. No GT, no layout:
the numbering is a handle, and the checker uses its own labels.

**Where B1 repeats a label** (public `tasks.json`: groups with `label_prefix` and count > 1):

| task | label | prefixes | count | today |
|---|---|---|---|---|
| classify_objects | rigid0 | cat0, cat1, cat2 | 3-9 | repeated: belief keeps 1 (20 of 20 records on compute4) |
| classify_objects_by_language | rigid0 | cat0, cat1, cat2 | 6-9 | repeated: belief keeps 1 (20 of 20 records) |
| sweep_blocks_random | rigid2 | target_1, target_2, target_3 | 3-6 | repeated by the code; no camera row found on compute3/4 |
| sweep_blocks | rigid2 | cube | 3-5 | numbered `cube_k` by B1 |
| arrange_largest_number(_random) | rigid0 / geometry0 | digit / mat | 4-5 | numbered `digit_k` / `mat_k` |
| solve_equation | rigid0 | num | 10 | numbered `num_k` |

Explicit label lists (nesting dolls x2, stack_blocks x2, imitate_sorting_sequence) are distinct labels. A scan of all
578 camera reports on compute4 (38 tasks) found repeated labels only in the two classify tasks.

**Offline replay** of the 8 planning records (agentic2), with no simulator: "keep" routes 0/8 and "numbered" routes 8/8 to
`classify_into_bins_v3b`.

**Next to expect.** Perceived-category errors. Lang 100005 and 100006 perceive exactly the instruction's three
categories. Lang 100007 and 100008__retry see 4 and 5, so the ported `fixed_map` will raise. Free-sort 100007 sees 4
categories for 3 baskets.

The flag value is "numbered", not "unique": `test_legality_select_ties` (D-C17) counts every `"unique"` string constant
under `dojo/runtime` as a consumer of the select mode of that name.
