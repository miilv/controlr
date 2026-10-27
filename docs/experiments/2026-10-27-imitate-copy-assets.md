# 2026-10-27 — imitate_sorting_sequence: the demonstrator's side takes our side's assets

Follows [imitate-frame0](2026-10-27-imitate-frame0.md) (same data, metrics and scene background; read that first).

**Question.** After the frame-0 fixes, the plan was right on 8 of 13 recorded demo layouts. Most misses were category
swaps among the demonstrator's small, far objects (action camera / garage figure / watch), while our near side read 64
of 65 right. The config says aim_i is t_i's asset (`same_as_label`). Does making the demonstrator's side a permutation
of our side's assets fix them?

**Change.** `inverse_layout._quota_copy_assets` (`EstimatorConfig.quota_copy_assets`, on): when the source list is
complete (one object per label, every category once), each copy-list object is re-fitted at its own blob as each source
asset and a Hungarian (each source asset once) picks them by total cost.

**Data.** Memory's 13 genuine-5.1 demo sequences; scene background from the 12 non-demo layouts; main's departure
read (#155). No LLM calls.

| arm | demo found / category / asset (of 65) | layouts all 10 right | order exact | twins right (of 65) | false departures | plan right |
|---|---|---|---|---|---|---|
| main (#152 + #155) | 63 / 57 / 51 | 8 | 10 / 13 | 56 | 0 | 8 / 13 |
| + copy assets | 63 / **63** / 56 | 10 | 11 / 13 | **62** | 0 | **10 / 13** |

- Our side unchanged (64 / 64 / 57).
- Left: 100014's far phone and 100017's near watch not found at frame 0; 100023's far watch hidden behind a figure.
- Cost: 25 re-fits per imitate frame; the frame-0 estimate took ~110 s instead of ~70 s (5 in parallel, loaded box).
- No other task has a copy list: G1 sets unaffected by construction.

**Limits.** n = 13; offline (frame 0 + recorded departures), not the picks; plan right is an upper bound on success.
