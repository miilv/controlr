# 2026-10-26 — phantom stacked pieces in the scan-primary estimate

**Question.** Integration-2's tower scan A/B (proxy, 25 layouts) lost 3 episodes to the same artefact:
- 100011, 100014, 100022;
- the scan-primary B1 estimate put a phantom pillar (key 00002) on top of a real one: z 0.822 vs 0.784, one pillar
  height, conf 0.57-0.62;
- the phantom took the label of a real pillar, which then counted as found.

Does requiring multi-view support for an elevated hypothesis remove the phantoms, and does it lose any real stacks?

**Rule** (`dojo/perception/inverse_layout.py`).
- **Support per view.** An elevated (stacked) hypothesis is judged per view on the pixels it ALONE would explain:
  its silhouette minus every other chosen object's, its support included, restricted to known pixels.
  - A view **supports** it at >= 50 % foreground there.
  - A view **refutes** it below 25 %.
  - A view with too few exclusive pixels is uninformative: it has < 30 px, or < 15 % of the silhouette.
- **Acceptance.** It needs >= 2 supporting views and more supporting views than refuting ones.
- **Config.** `EstimatorConfig.elevated_multiview`:
  - `"identical"` (default): only a piece on an IDENTICAL piece. This covers `_stack_proposals` and every level
    above the first of `_tower_proposals`.
  - `"all"`: every elevated hypothesis. `scan.estimate_with_scan` uses it, since an episode start has nothing
    stacked.
  - `"off"`.
- **Code paths covered.** The search's full fuse runs the same estimate.

**Data.**
- Integration's saved `look_frames` are the last look_plan look, not the scan frame, so their scan records cannot be
  replayed.
- The same scenes are in our D0 scan set: `d0s` frame_S1 = the dev layouts rendered at the frozen scan pose by the
  same Isaac 6.0 proxy, plus frame 0.
- Runs: `eval_g1` on a GPU light lane (EGL), 25 layouts per task. "main" = before, "new" = this branch.
- No LLM calls.

## Results

| set | variant | tower (200): ok / gross / missed | elevated objects | stack (75) ok | push (50) ok | objects changed vs main |
|---|---|---|---|---|---|---|
| scan-primary + frame 0, canonical bg (as the integ runs) | main | 189 / 1 / 8 | **1** (100022 block1, z +38 mm, 552 mm off) | 53 | 25 | — |
| same | new, identical | 189 / 0 / 9 | **0** | 53 | 25 | 1 (the phantom → missed) |
| same | new, all | 189 / 0 / 9 | **0** | 53 | 25 | 1 (the phantom → missed) |
| scan-primary + frame 0, scan-aware bg | main | 187 / 1 / 8 | **1** (100022) | 58 | 25 | — |
| same | new, all | 187 / 0 / 9 | **0** | 58 | 25 | 1 (the phantom → missed) |
| frame 0 (stack, tower) | main / identical / all | 165 / 20 / 10 (all three) | 0 / 0 / 0 | 54 | — | 0 |
| frame 0 (pack × 8; 190 s per frame) | main / identical / all | — | 0 / 0 / 0 | — | — | 0 |

- **Phantoms removed: 1 of 1 reproducible. Real stacks lost: 0.** Nothing is stacked at frame 0 in any layout, and
  no object other than the phantom moved in any compared run (1605 object rows).
- **The phantom becomes a "missed" object.** In the runtime, the scan's frame-0 fill and the search then look for it,
  instead of a real pillar's label being taken.
- **100011 and 100014 do not reproduce on D0** with the current code and either background, so the integ
  runs' scan frames are needed to replay them.

## Caveats / next

- **n is one reproducible phantom.** The rule is conservative by construction: no frame-0 or table-level object
  changed.
- **Integration-2 should save the scan frame** with `save_look_frames`. Today only frame 0 and the last look are
  kept, so scan-record artefacts cannot be replayed.
