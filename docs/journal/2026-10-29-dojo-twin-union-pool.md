# 2026-10-29: twin_union_pool, std / _random pairs share one candidate pool (stream W)

**Why.** A std task and its `_random` twin have the same instruction, so routing can't tell which variant a scene is.
Perception then searches only the routed set's candidate assets:
- `_random` scenes routed to the std set (pour_liquid, dolls, push_T, arrange; the `_random`-twin convention) are
  searched with std assets;
- pack's std scenes are searched with `_random` assets, because the table routes pack to `pack_objects_into_box_random`.

12 such pairs differ in their candidate assets: pack, hang_mugs, push_T, stack_blocks / bowls, dolls, toast, laptop,
sweep, fold, arrange and pour.

**`EstimatorConfig.twin_union_pool`** (default off). Each label's candidates and categories are the routed set's plus
its twin's (`union_twin_pool`, from the public configs). It is one fixed rule and never picks by variant.

**Offline frame 0** (routed set, pool off → on, 3 frames per task, pack std 6; ok10 = exact asset within 10 mm):

| frames | ok10 off → on | xy10 off → on |
|---|---|---|
| pour_liquid_random | 0 → 6 of 6 | 4 → 6 |
| sort_nesting_dolls_random | 0 → 6 of 15 | 6 → 7 |
| push_T_random | 2 → 4 of 6 | 4 → 4 |
| pack_objects_into_box (std) | 6 → 24 of 30 | 6 frames |
| pour_liquid (std) | 6 → 3 of 6 | 6 → 6 |
| sort_nesting_dolls (std) | 6 → 4 of 15 | 10 → 10 |
| arrange ×2, pack_random, push_T, stack_blocks, stack_bowls | unchanged | unchanged |

**The std losses are exact-asset identity between near-identical twins:** the random mug or bottle model, and the same
doll size in another texture. Found and xy are unchanged. A tie bias toward the routed set's assets (+0.02 on identity
cost, plus a routed-shape re-fit) changed nothing, so it was dropped. Whether the identity swap costs the skills
anything is for a rendered A/B:
- gains: pour_liquid_random, dolls_random, push_T_random, pack (std);
- guards: pour_liquid, dolls, push_T (std).
