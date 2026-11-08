# 2026-11-08: the f-catalogue/1 extraction (our half of the F split), built catalogue-wide

**Done:**
- **Code:**
  - `dojo/skills/fcatalogue.py`: the extraction library. It is class-free; see "Found along the way".
  - `scripts/dojo/f_catalogue_extract.py`: `build`, `check` and `compare`.
  - `scripts/dojo/fcat/check.py`: the round trip, the verb readers and CONV.
  - `scripts/dojo/fcat/functarget_ref.py`: the partner's prototype, byte for byte.
  - `dojo/legality/roots.py`: the denied path components for asset-reading tools.
- **Tests:** `tests/dojo/test_f_catalogue.py`, 25 tests, CPU only, on synthetic assets.
- **Answers:** to the spec's §6 questions, on main first (9c694e82,
  [2026-11-08-f-catalogue-answers](2026-11-08-f-catalogue-answers.md)).
- **Catalogue-wide run:** compute3, CPU, nice, 2 processes. The file is
  `~/dojo-data/fcat/v3/f-catalogue-0c49bf1d05d4-69261f09.json`.

  | item | value |
  |---|---|
  | sha256 | `8e38f2201d37fa06ce11c731d308a8c72299c226d8e4996ffa2e082aebdc7ae7` |
  | size | 962,383 bytes |
  | keys | 743 keys, 742 entries |
  | build / check / compare | 68.6 s / 3.7 s / 51.8 s |
  | LLM calls | 0 |

  Results: [experiment report](../experiments/2026-11-08-f-catalogue.md).

**Verdict against the stopping condition: (a).**
- **Round trip:** 358 frames and 245 tags (every frame tag of every asset), 0 failures, max error 4.4e-16. This holds
  with two prototype changes:
  - lever and hinge targets keep their frame orientation;
  - tags keep the file's order.

  Run verbatim, the prototype fails 5 tags: `laptop_hinge` x4 and toaster/00004's `toast_botton`.
- **CONV:** pinned by the semantic axis of 181 / 181 typed frames, plus 57 measured per-tag overrides.
- **Readers:** today's readers agree within 6e-16, `slot_targets` and `insert_into`'s composition included.
- **Derived cases:**

  | case | measured | reference |
  |---|---|---|
  | tray: resting surface below the rim | 22.73 mm | ORG2: 23 mm |
  | coin stand: lip at 5 mm, exact section | 3.02 mm | COIN2: 3.1 mm |
  | coin stand: lip at 5 mm, COIN2's ±1 mm slab | 3.16 mm | COIN2: 3.1 mm |
  | coin stand: slot width | 2.58 mm | COIN2: 2.6 mm |

**Decisions:**
- **`catalogue_hash`** = sha256 of `index.json` (Q1). The build refuses when `index.json`'s keys differ from the meshes
  it reads, and the CLI takes one catalogue root.
- **Frames are in the root frame.** `root_in_asset` is given per Articulation; it is not the identity on 4 laptops.
- **Methods, picked by measurement:**
  - the heightmap resting surface (slicing picks hidden or flipped faces; the hull misses the tray by its rim);
  - the 8 mm base band (ORG2's; `radius_by_band` carries 4 / 8 / 12 mm);
  - the exact-section lip;
  - the mesh opening fit for slot widths (frame corners kept beside it).
- **Q7: `refine_insertion` stays a runtime step.** It is a pair fit, and the bank's own floor is 100.8 mm below its
  frame.
- **Tags typed null for Codex:** `check`, `checkpoint`, `be_placed`, `coin_insert_center_same_side`,
  `coin_insert_center_diff_side`. That is 5 tag names, 15 asset tags.

**Found along the way:**
- **The partner's adapter needs eight §1 changes,** listed in the report and in BACKLOG P1:
  1. `tag_conv` (and slot / socket now have sign -1);
  2. lever / hinge orientation;
  3. tag order;
  4. `@top` / `@base` targets from `derived`;
  5. `root_in_asset`;
  6. `joints_by_tag.origin` (laptop_hinge's frame is 45 mm off its axis);
  7. hole radius from the mesh (15 mm raw vs 12.4 mm);
  8. accept extra keys.
- **The catalogue's frames are mating poses,** not "z out of the opening". Slot frames on the toolbox, the coin stand
  and toaster/00004 put the semantic axis in another column. That is why CONV needs per-tag overrides, and why a
  perception-side slot reader must get the type-row frame, not the raw one, for 7 slot frames.
- **Legality engine: a new class in `dojo/` is reached through its method names.** An untyped `x.pose()`, `x.frame()`
  or `x.read_bytes()` reaches it. The first draft's `Reader` / `RootMap` classes and the vendored prototype's
  `FuncTarget.frame` were pulled into the inference call graph that way, and their numpy calls became "unresolved"
  findings that fail the ratchet.
  - The fix: plain functions, and the harness and prototype under `scripts/`. DEVELOPMENT's pitfalls now note it.
  - The partner's `functarget.py` will meet the same thing in a profile with `report_unresolved`.
- **Many catalogue meshes have flipped normals,** so mesh slicing by face orientation is unreliable catalogue-wide.

**Not done / next step:**
- The founders decide the split at the 8 Nov planning.
- Codex rules on the null list, and the partner adapts their §1.
- The lead copies the v3 file to compute4 / compute6 `~/dojo-data/fcat/` and pins its sha256 in `hybrid_pins`.
- A G3 placement check on spent ids would settle the 62 band-sensitive bases.
