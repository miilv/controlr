# 2026-11-08: the f-catalogue/1 extraction, catalogue-wide (round trip, CONV, tag types, derived-geometry methods)

**Question.** Our half of the F split agreed on 7 Nov at 23:12Z, built against the partner's adapter spec
(`f-adapter-spec.md`, section 1):
- Can one catalogue-wide extraction emit `f-catalogue/1` whose frames round-trip through the partner's
  `from_asset` / `as_frames`, bit-exact within 1e-9 for every tag of every asset?
- Can CONV be pinned for every type by matching today's verb readers?
- Can every tag be typed by grammar and frame structure alone?
- Do the mesh-derived contact fields reproduce our measured cases: ORG2's tray, and COIN2's stand lip and slot?

Answers to the spec's §6 questions: [journal 2026-11-08-f-catalogue-answers](../journal/2026-11-08-f-catalogue-answers.md).

**Verdict: stopping condition (a) is met.** The patch-dependent parts are part (a) of the verdict and depend on two
changes to the partner's prototype (point 2 below).
1. **The file covers the whole catalogue and validates clean.** 743 keys are listed and 742 have entries. The one key
   skipped, `Clutter/vehicle/00000`, has no aligned bbox, and the file names it.
2. **The round trip passes for every frame tag: 358 frames, 245 tags, 0 failures, max error 4.4e-16.** This needs two
   changes to the partner's prototype, applied in our harness as patch p1:
   - lever and hinge targets keep their frame's orientation;
   - tags keep the file's order.

   The prototype run verbatim fails 5 tags. The four `laptop_hinge` frames lose 18.5 deg (matrix error 0.317), and
   toaster/00004's `toast_botton` loses 1.8e-7.
3. **CONV is pinned for every type.**
   - Every measured typed frame of an upright owner points within 25 deg of its semantic direction: 181 of 181, worst
     3.2 deg. The prototype's table manages 0/14 slots, 0/6 sockets and 72/121 supports.
   - A slot target's `frame()` under the pinned row is exactly the frame `coin_contact.slot_targets_from_pose` reads
     (0.0 deg).
   - The existing readers give identical answers on catalogue and round-tripped frames, within 6e-16.
4. **The derived methods reproduce the measured cases:**

   | case | our measurement | reference |
   |---|---|---|
   | tray: resting surface below the rim | 22.7 mm | ORG2: 23 mm |
   | coin stand: half-width at 5 mm depth | 3.02 mm (exact section); 3.16 mm with COIN2's ±1 mm slab | COIN2: 3.08–3.16 mm |
   | coin stand: bbox half-width | 5.55 mm | COIN2: 5.55 mm |
   | coin stand: slot width at mid-depth | 2.58 mm | COIN2: 2.6 mm |

**Where the numbers hold.** The data limits are under "Data limits"; they do not block (a).

## Setup
- **Code:** `dojo/skills/fcatalogue.py` (the extraction library), `scripts/dojo/f_catalogue_extract.py`
  (`build` / `check` / `compare`), `scripts/dojo/fcat/check.py` (round trip, readers, CONV) and
  `scripts/dojo/fcat/functarget_ref.py` (the partner's prototype `20261107T2309Z-misha-functarget.py`, byte for
  byte, sha256 `679865c8…`). Tests: `tests/dojo/test_f_catalogue.py` (25, CPU, synthetic assets).
- **Inputs:** compute3, read-only, CPU only, `nice -n 10`, 2 processes.
  - RoboDojo's `Assets/Object/RoboDojo`: `metadata.json` for every key; `object.usd` for the 11 Articulations,
    read through pxr from `~/dojo-perception/.venv`.
  - `~/dojo-perception-data/catalog`: `index.json` and 550 `geom.npz`.
  - Nothing else is opened. `Reader` allows only these file names under these two roots, and a test spies on
    `open` / `read_bytes` over a synthetic tree that has a task config, `tasks.json` and `info.json` planted next to
    the assets.
- **Build hash:** `catalogue_hash` = sha256 of `index.json` = `0c49bf1d05d435f2…`, the same file on compute3, 4 and 6
  (answers Q1). `listing_sha256` = `011199e2…`, over the 1305 files read with their sha256.
- **Output:** `~/dojo-data/fcat/v3/f-catalogue-0c49bf1d05d4-69261f09.json` on compute3 (not in git).
  - sha256 `8e38f2201d37fa06ce11c731d308a8c72299c226d8e4996ffa2e082aebdc7ae7`, 962,383 bytes.
  - The name ends in the code hash, over this script and the library.
  - Wall times, 2 processes: build 68.6 s, `check` 3.7 s, `compare` 51.8 s.
  - v0-v2 were earlier builds with the same frames; only rule wording, `radius_by_band` and the code hash differ.
- **Cost:** 0 LLM calls, no simulator, no layout ids.

## Coverage

| item | count |
|---|---|
| keys listed / entries | 743 / 742 |
| with passive functional / active functional / support frames | 36 / 62 / 93 |
| with `joints_by_tag` (all from the USD) | 11 (toaster 4, laptop 4, SpringButton 2, egg_holder 1) |
| `root_in_asset` not the identity | 4 (laptop/00000-03: +77.1 mm in y, -3.5 mm in z; same as `dojo/skills/data/articulation_roots.json`) |
| with a mesh / derived entries | 550 / 480 assets |
| `@top` (resting surface) / `@base` (lowest band) | 453 / 480 |
| derived opening fits: slot / hole / socket / support | 10 / 10 / 2 / 3 (plus 104 local support surfaces) |
| `tag_conv` overrides | 57 (support 50, slot 7) |

Typed tags (counts per asset tag):

| type | count |
|---|---|
| support | 568 (incl. 453 `@top`) |
| base | 486 (incl. 480 `@base`) |
| slot | 20 |
| handle | 18 |
| hole | 10 |
| strike_point | 9 |
| lever | 6 |
| hinge | 4 |
| peg | 4 |
| socket | 3 |
| hook | 2 |
| null | 118 |

The types keyhole and rim have no catalogue frames.

## Tag -> type map
**The rule.** It is the same for every key. It never consults task configs, and no category or key is a branch in
the code.
- **Grammar:** the tag's words are matched against a lexicon of general feature words (`fcatalogue.LEXICON`); the
  head word is the last feature word.
- **Structure:**
  - no `frame` → null;
  - a support entry is a circle, so `slot` there means hole;
  - a `bottom` that outlines a region with several frames is a support;
  - a lever or hinge needs a USD joint with limits.
- **Active frames with no feature word** take the passive type that the catalogue's tags give their owner's category:
  `hammer` → `hammer_slot` → slot.

**Typed null for a Codex ruling (5 tag names, 15 asset tags):**

| tag | assets | why |
|---|---|---|
| `checkpoint` | toaster/00002, car, electric_toothbrush, hammer (active) | names a point where a result is checked |
| `check` | pen, oil_pen | the same |
| `be_placed` | factory_bolt/00000-04 | names a goal role |
| `coin_insert_center_same_side`, `coin_insert_center_diff_side` | coin (active) | "same / different side" of an unstated reference reads as a goal convention; the word `insert` alone would type `slot` |

**Null with no task semantics:**
- `GetParentLink` (egg_holder): no feature word. Its USD joint is the lid's revolute joint, so it is a hinge candidate
  if a ruling allows joint-bearing tags without a word.
- piggy_bank `center`: no feature word.
- `bbox_0..7` (xylophone bar outlines).
- 12 garment particle tags: no frame.

**Kind-named tags typed by their feature word, the kind dropped (for transparency):**
- slot: `toast_slot1` / `toast_slot2` (toaster x4), `hammer_slot` / `pliers_slot` / `tape_measure_slot` /
  `wrench_slot` (toolbox), `key_slot`, `slot_opening`;
- hole: `test_tube_slot/0..9`;
- support (support groups named after the owner's category): `bread_shelf/k`, `vertical_coin_stand/0`,
  `mallet_stand/0`, `laptop_stand/0`, `egg_basket/0`, `cube_cushion/0`, `cell/k`, `block/k`, `mahjong_0`.

## CONV (per type, plus measured per-tag overrides)
**How it was pinned.** For every typed passive frame whose owner rests upright (+z up in an `active.place`
orientation), the frame column (±) nearest the type's semantic direction was measured:
- into an opening = down;
- up a peg, a support or a strike face;
- a base = down.

**The pinned table** (`fcatalogue.CONV`, written to `build.conv`):

| type | axis column | sign | across column | changed from the prototype |
|---|---|---|---|---|
| hole | 2 | -1 | 0 | no |
| slot | 2 | -1 | 0 | **yes** (prototype: +1) |
| socket | 2 | -1 | 0 | **yes** (prototype: +1) |
| keyhole | 2 | -1 | 0 | **yes** (prototype: +1); by analogy with the openings measured, no data |
| rim | 2 | -1 | 0 | **yes** (prototype: +1); by analogy, no data |
| peg | 2 | +1 | 0 | no |
| support | 2 | +1 | 0 | no |
| strike_point | 2 | +1 | 0 | no |
| base | 2 | -1 | 0 | no |
| hook, handle, lever, hinge | 2 | +1 | 0 | no; no semantic axis to measure (a lever / hinge uses its joint's axis) |

**Measured agreement with the semantic direction:**

| type | n | pinned row or override, within 25 deg | prototype row, within 25 deg | overrides |
|---|---|---|---|---|
| support | 121 | 121 (max 1.0 deg) | 72 | 50 |
| strike_point | 24 | 24 | 24 | 0 |
| slot | 14 | 14 (max 3.2 deg) | 0 | 7 |
| hole | 10 | 10 | 10 | 0 |
| socket | 6 | 6 | 0 | 0 |
| peg | 4 | 4 | 4 | 0 |
| base | 2 | 2 | 2 | 0 |

**Why overrides exist.** The catalogue's frames are mating poses: the pose the occupant's own frame takes. So some
assets' frames put the semantic axis in another column.
- Examples:
  - the toolbox tool slots: axis column 1;
  - toaster/00004's slots;
  - the coin stand: column 1 up;
  - the bread shelf: column 0;
  - flipped supports such as the mahjong tiles and the dustbin.
- These get `tag_conv[tag] = [a, s, c]`, measured, with the evidence in `tag_conv_why`.
- `across` (a slot's or a support's long direction) is taken from the mesh opening fit where it differs from the
  row's: the bank's slot runs along column 1.

**The readers.**
- Single-asset readers, at a seeded random owner pose, catalogue frames vs round-tripped frames:
  `library.functional_frames`, `precision.frame_points_w`, `precision.hole_frames`, `library.slot_frame`,
  `library._support_raw`. All agree within 5e-16 with p1.
- Pair readers, 62 occupants x 128 containers, also within 6e-16 with p1:
  - `library.slot_targets`: 25 target poses;
  - `insert_into`'s frame composition: 175 poses.
- Without p1:
  - `functional_frames` differs by 0.317 (laptop_hinge);
  - the pair readers differ in order, because `from_asset` sorts tags and a verb that takes the first matching
    frame would try another one first.
- `coin_contact.slot_targets_from_pose`, fed a slot target's type-row `frame()` built from (point, axis, across),
  gives exactly the poses from the frame its docstring defines (x along, y across, z out): 0 difference.

**A slot override's raw frame is not the reader's frame.** For the 7 slot frames with an override, `frame()` rebuilds
the raw catalogue frame (needed for the exact round trip), and that frame is not what the perception-side reader
expects. The adapter must hand such readers the type-row frame. The 7 are toaster/00004 x2, piggy_bank `slot_opening`
(across) and the 4 toolbox slots.

## Derived geometry: methods compared, picked by measurement
**Resting surface (`@top`, support).** Over the 480 upright meshes:

| method | refused | drawer (ORG2's tray, 23 mm) | note |
|---|---|---|---|
| **heightmap** (ORG2's top height map, 3 mm cells, largest 4-connected region flat within 4 mm) | 27 (flat region < 4 cm²) | **22.73 mm below the top** | visible-from-above by construction |
| slice (the horizontal band with the most up-facing triangle area) | 148 | 118.7 mm (wrong: an internal / bottom face) | flipped normals in many meshes: picks bottoms (monitor/00002: 229 mm below its base plane) |
| hull (the top facet of the convex hull, i.e. what a bbox / silhouette top gives) | 81 | 0.0 mm (misses the tray by 22.7 mm) | equals the rim |

- **Agreement:** where all three answer (301 assets), heightmap and slice differ by more than 4 mm on 62, and
  heightmap and hull on 88.
- **Picked:** heightmap.
- **The surface is below the visible top in many assets:** the heightmap surface lies more than 4 mm below the top on
  140 of 453 assets, and more than 10 mm on 113. These are trays, recesses and lipped plates, where the camera-visible
  top misleads.

**Base band (`@base`, base): lowest band of the mesh at 4 / 8 / 12 mm, 480 meshes, 0 refused.**
- **Synthetic pedestal** (a 15 mm pedestal 6 mm tall under a 25 mm body; the test): the 4 mm band reads the pedestal;
  8 and 12 mm reach the body.
- **Catalogue:**

  | comparison | median | p90 | assets over 5 mm |
  |---|---|---|---|
  | radius at 8 mm minus at 4 mm | 0.35 mm | 6.6 mm | 62 / 480 |
  | radius at 12 mm minus at 8 mm | 0.0 mm | 5.0 mm | 48 / 480 |

- **Picked: 8 mm,** ORG2's stand-placement band, the only band with placement evidence (G3: figures placed with base
  xy errors of 2.1-7.5 mm).
- **Every entry also carries `radius_by_band`** at 4 / 8 / 12 mm, so a consumer can see when a base is
  band-sensitive.
- **Limit:** there is no catalogue ground truth for a base, so the pick between 4 and 8 mm on those 62 assets is open.
  It would need a placement measurement (G3), not geometry.

**Lip** (the holder's outer half-width at depths 1-10 mm below the local top). On the coin stand at 5 mm:
- the exact horizontal section gives **3.02 mm**;
- COIN2's `wall_intercept` reading, the max over vertices in a ±1 mm slab, gives **3.16 mm**;
- COIN2's measured range is 3.08-3.16 mm, against a 5.55 mm bbox half-width.

The slab's max lies one slab deeper: +0.14 mm at the stand's 0.135 mm/mm taper. **Picked:** the exact section. The
params name the slab offset, so the wall probe's intercept can be re-based.

**Slot width / length / depth.** Two methods were compared: the catalogue's corner frames, and the mesh opening fit.
The mesh fit takes the inner-wall gap of the horizontal section at mid-depth, or at the deepest walled mouth depth
for openings into hollow bodies, plus the length along the slot and the depth to the first hit of the axis ray.

| opening | mesh: width x length, depth | corners | note |
|---|---|---|---|
| vertical_coin_stand (support) | **2.58 mm** x 24.4 mm, 15.9 mm | — | COIN2: 2.6 mm; it tapers from 3.0 mm at the mouth to 2.24 mm at the floor (`width_profile`) |
| piggy_bank `slot_opening` | 5.44 x 34.5 mm (at the 2 mm mouth), 102 mm to the inside floor | 2 frames at one point: no extent | runs along column 1 (measured `across`) |
| toaster/00000-02 slots | 25.9-39.6 mm wide | 25-45 x 125-150 mm | corners outline the floor footprint and sit 0.8-5.8 mm wider than the walls at mid-depth |
| toaster/00004 slots | unreliable (depth 2 mm) | 161 x 93 mm | its corner frames outline both slots (a data quirk `library.slot_openings` already notes) |
| test_tube_slot holes | 24.8 mm diameter (mouth), 54 mm deep | — | the raw support radius, 15 mm, is not the hole's: 12.4 mm |
| socket/0, socket/2 | 2.2 / 0.8 mm | — | blade slots; socket/1 refused (closed at the frame point) |
| refused | — | — | toolbox `pliers_slot` / `wrench_slot` (tool-shaped recesses, no walls within the window), toaster/00002 and 00004 `toast_slot2` |

**Picked:** the mesh fit for `extent` (the binding width), with the corner extent kept beside it in
`derived[tag].corners`.

**Q7 (`refine_insertion`'s 22.8 mm): it stays a runtime step.**
- The bank's own floor is 100.8 mm below `slot_opening` (a hollow body), so the 22.8 mm is not an intrinsic depth.
- It is `refine_insertion`'s pair fit: the moved object's footprint against the receptacle's depth map, plus a floor
  press whose docstring cites the evaluator.

## Data limits (numbers)
- **No mesh:** 192 Clutter assets have none in the perception catalogue. Their frames are in the file; no derived
  geometry exists for them.
- **No bbox:** 1 key (`Clutter/vehicle/00000`) has no aligned bbox in `metadata.json`; it is skipped and named.
- **Not upright:** 70 assets with a mesh have no upright place orientation. They get no `@top` / `@base`, and their
  frames have no measured CONV, so the type row applies.
- **`@top` refused:** 27 upright assets, flat region under 4 cm².
- **Opening fits refused or unreliable:** 4 of 14 passive slot tags are refused, and toaster/00004's slot fit is
  unreliable. Socket/1 is refused.
- **CONV unpinned by data:** keyhole and rim (no frames); hook, handle, lever and hinge (no semantic axis in v2).
- **Base band:** 62 of 480 bases are band-sensitive by more than 5 mm (8 vs 4 mm). There is no geometric ground truth
  to choose between them.

## What the partner's adapter must change in §1 (sent with the PR)
1. **`tag_conv`:** per (asset, tag) `[axis_col, sign, across_col]`, applied before `build.conv`. 57 tags need it.
   - The pinned `build.conv` changes slot and socket (and, by analogy, keyhole and rim) to sign -1: axis into the
     opening.
   - Perception-side readers of a slot need the type-row frame, not the raw frame of an overridden tag: 7 slots.
2. **Lever / hinge orientation:** keep the frame orientation (`as_frames` writes the identity). 5 tags fail without
   it.
3. **Tag order:** keep the file's tag order (`from_asset` sorts).
4. **Derived-only targets:** `@top` (support) and `@base` (base) are typed in `tag_types` and framed in
   `derived[tag].frame` (pose7), with no entry in `frames`.
5. **`root_in_asset`** (pose7, Articulations only; absent = the identity). Our belief poses are in the asset frame, so
   a laptop placed without it is 77 mm off.
6. **`joints_by_tag[tag].origin`:** a point on the joint axis, in the root frame. `laptop_hinge`'s frame point lies
   45.1 mm off its joint axis.
7. **Hole and socket radius:** use the derived mesh width (diameter at the mouth), not the raw support radius. The two
   differ: 15 mm raw against 12.4 mm in the mesh on the test-tube rack.
8. **Extra keys:** accept additional per-entry keys (`tag_rules`, `tag_conv_why`, `derived_refused`, `derived_note`,
   `width_profile`, `radius_by_band`, ...) and `build.sources.skipped` / `n_entries`. The spec refuses on missing or
   out-of-domain fields only.

## Reproduce
```bash
# compute3 (CPU), our checkout synced to ~/controlr-fcat; pxr from perception's venv
PY=~/dojo-perception/.venv/bin/python
A=~/dojo-robodojo/RoboDojo/Assets/Object/RoboDojo
C=~/dojo-perception-data/catalog
O=~/dojo-data/fcat/v3
nice -n 10 $PY scripts/dojo/f_catalogue_extract.py build --assets $A --catalog $C --out $O --workers 2
nice -n 10 $PY scripts/dojo/f_catalogue_extract.py check $O/f-catalogue-*.json --assets $A --out $O/check.json
nice -n 10 $PY scripts/dojo/f_catalogue_extract.py compare $O/f-catalogue-*.json --assets $A --catalog $C \
    --out $O/compare.json --workers 2
uv run pytest -q tests/dojo/test_f_catalogue.py        # CPU, synthetic
```
The build is deterministic: the test compares a 1-process and a 2-process build key for key. n = the whole catalogue,
one build. The method comparison is a single pass; no seeds are involved.
