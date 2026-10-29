# 2026-10-29: measurement layouts for 14 tasks are regenerated at simlab v15 (lead decision)

## Finding
- Integration's layout audit (compute4 vs compute3) found the two boxes md5-identical, except pick_from_conveyor_by_image.
  - compute4 / compute3's shared copies were simlab v2: the basket sat inside the conveyor body, z 0.668 under the 0.765 belt.
  - Every camera row there ended "no skill" (basket not found).
  - Fixed: replaced by v13 (backups in `~/dojo-data/simlab/layouts_bak/`), and the old rows are renamed `*.stalelayout`.
- A fidelity table followed: 100005-09 + 100020-24 for all 54 tasks were regenerated at v15 (53dd9ed8; task configs + object metadata only) and compared with the stored records.
  - 40 tasks are identical.
  - 14 changed, built by older generators whose later fixes plausibly matter:
    - v1, no home-pose gripper keep-out: push_T, stack_blocks, stack_bowls, build_tower, put_bottles, pack_random.
    - v2, before v3's parent-frame offsets: fill_pen, insert_key, plug_in_charger, store_tools, swap_blocks.
    - v6: fill_egg_holder.
    - v11, before v14's random conveyor-child frames and the lying bottle: match_and_pick_from_conveyor.
    - v13: pick_from_conveyor_by_image.
  - make_kong differs in assets only.

## Decision
- Those 14 tasks get a v15 measurement set: same ids, `~/dojo-data/simlab/layouts_v15/<task>/` on both boxes, families tagged `v15`, and a `layout_gen` field per row.
- For those cells, v15 candidate rows replace old-generator rows once v15 has n ≥ 5. A/B pairs never mix generators.
- Old rows stay on disk and are marked "old-gen" in the 54-cell table.

## Why
- A measurement set built by a generator with a known fidelity bug mis-states the board.
- It also misdirects work: put_bottles, insert_key, fill_pen, fill_egg and match_and_pick are next in line for skill and perception work.
