# 2026-11-08: our answers to the partner's F adapter spec §6 (questions 1-7), for the catalogue extraction

These are our side's answers to section 6 of the partner's adapter spec (`f-adapter-spec.md`, branch night/F-ADAPTER
b49ea4d9, received on the bus at 0000Z). In the split agreed on 7 Nov at 23:12-23:17Z, we own the catalogue-wide
extraction (`f-catalogue/1`), the CONV pinning, the tag → type map and the catalogue-wide round trip. The extraction
itself follows in its own PR. Where an answer depends on the PR's measurements, this note says so; the PR's journal
entry carries those numbers.

The evidence comes from code at main 5f2874b4 and from read-only listings on compute3, compute4 and compute6 (8 Nov,
about 00:05Z). Those listings covered the perception catalogue's `index.json`, plus `metadata.json` from RoboDojo's
`Assets/Object/RoboDojo` on compute3. No task configs, checkers, layouts or logs were opened. No LLM calls were made.

## Q1. Build hash: is `catalogue_hash` the hash identity records carry today, or a new one?

**What carries a catalogue hash today.** Two records hash the same file, the perception catalogue's `index.json`:

| record | field | hash | code |
|---|---|---|---|
| every eval row | `catalog_md5` | md5 of `$DOJO_CATALOG/index.json` (else `~/dojo-perception-data/catalog`) | `dojo/runtime/eval.py:293-305`, written at `:335` / `:420` |
| every B1 perceive output | `data.catalog.sha256` (plus `data.id` = `b1d-` + sha256 over all B1 data roots) | sha256 of `index.json` under `render.catalog_root()` | `dojo/perception/j1_adapter.py:563-612` (`data_identity`) |

Neither is attached per object:
- A B1 layout record's `_perception` entry carries a pose, place tag, residual, confidence and per-view IoU
  (`dojo/perception/inverse_layout.py:24`), plus an `identity` mark (`dojo/perception/identity_marks.py`). It carries
  no hash.
- The partner's provenance-bearing identity record (GENERAL-LIB-INTERFACE item 4, Codex finding 4 in
  `2026-11-07-codex-reviews-partner-hybrid.md:119-146`) is not built yet. rd-gen's locate returns no `asset` today.

**Measured values (8 Nov, about 00:05Z).** The same file is on all three boxes:

| box | md5 | sha256 | keys |
|---|---|---|---|
| compute3 | `2cd7455cbdce2ab38e4e9d586e28db61` | `0c49bf1d05d435f24c37ab3981fb1a7bb4de9dc663b785958768f937c2489cb1` | 550 |
| compute4 | identical | identical | 550 |
| compute6 | identical | identical | 550 |

**Answer: `catalogue_hash` = sha256 of the perception catalogue's `index.json`.** It is the identical value B1
already emits as `data.catalog.sha256`, and it hashes the same bytes that `catalog_md5` hashes (md5 stays for old
rows). It is not a new hash per extraction.

Why this one:
- **It is the only hash an identity producer can know.** The adapter's gate asks: "was this identity chosen from the
  universe these frames describe?" The identity model knows the catalogue it searched, not our extraction run.
- **The extraction's inputs are pinned separately,** in two places:
  - `build.sources.listing_sha256` covers every file the extraction read, with each file's sha256;
  - the file's own sha256 goes in `hybrid_pins`.

  So two extraction runs over the same catalogue share `catalogue_hash` and differ in `listing_sha256` / file sha
  only if their inputs or the code differ.

**What the identity model must do.** Record `catalogue_hash` per identity record. It must be sha256 over the bytes of
`index.json` of the catalogue root it searched, computed the way `data_identity` does today.

**Two caveats.** The extraction checks both and refuses to build on either.

1. **`index.json` does not cover everything.** It holds only extents, mean colour, n_faces, place tags, z symmetry
   and names (`dojo/perception/catalog.py:597-612`). It does not cover `geom.npz`, `links.json` or the raw frames.
   - So a mesh re-convert that skips re-indexing leaves `catalogue_hash` unchanged.
   - The guard against that drift is `listing_sha256` plus the pinned file sha, not `catalogue_hash`.
   - The extraction records the `index.json` sha256 it saw. It refuses to build when `index.json`'s keys differ from
     the set of keys whose `geom.npz` it lists.
2. **Only the first catalogue root is hashed.** `DOJO_CATALOG` may list several roots (`dojo/skills/world.py:72-80`),
   but both `catalog_md5` (`eval.py:300`) and `render.catalog_root` (`dojo/perception/render.py:45-50`) hash only the
   first root that holds an `index.json`. Every box has one root today.
   - With overlays in further roots, the hash would not cover them.
   - The extraction refuses more than one catalogue root.

## Q2. Root frame: is the root the articulation-root prim, for frames and joints? Is `root_to_bbox` in that frame?

**What frame the raw data is in.** Raw `metadata.json` frames, the catalogue meshes (`geom.npz`) and `links.json` are
all in the asset's **default-prim frame** (`dojo/perception/catalog.py:9-11`, `dojo/skills/artroot.py:1-20`).
- **Non-Articulation types:** RoboDojo's official pose reads that same prim, so default prim = root.
- **Articulations:** RoboDojo poses the `PhysicsArticulationRootAPI` prim (`artroot.py:1-15`).

**Where the two frames differ.** This was measured from the USDs: `scripts/skills/articulation_roots.py` →
`dojo/skills/data/articulation_roots.json`, 11 Articulations checked. Only `Articulation/laptop/00000-00003` have a
non-identity root:
- prim `/root/E_body_1` at (0.031, -77.112, 3.522) mm, rotation identity;
- toaster, egg_holder and SpringButton agree with their records within 0.5 mm;
- toaster/00002 and 00004 have a non-Xformable root-API prim (a fixed joint), reported and not tabulated.

**Answer: yes, the root is the articulation-root prim, for frames, joints and `root_to_bbox` alike.**
- The file carries every position, axis, frame, joint anchor and `root_to_bbox` in the root frame:
  - for the 4 laptops, each value is mapped through `inv(T_root_in_asset)`;
  - for every other key, the raw values pass unchanged.
- `root_to_bbox` is the centre of the metadata's aligned bbox (`dojo/skills/world.py:189-203`), mapped the same way.
- Joints come from `links.json` (USD-derived), mapped the same way. The laptops' rotation is the identity, so their
  axes are unchanged and their origins shift.

**One addition to §1.** Our side's `ObjectState` and B1's records are in the **asset** frame (`artroot.py:16-19`:
`to_asset` / `to_root` convert where the two sides meet). A consumer that places targets with a belief pose from our
perception would be 77 mm off on a laptop.
- So the file also carries `root_in_asset` (pose7) per key.
- The adapter must either place targets with a root-frame (official-convention) pose, or convert with
  `root_in_asset`. This is listed for their §1.

## Q3. Tag types: which are null? Are `<kind>_slot` tags typed `slot` with the kind left out? Who rules on tags with task semantics?

**Inventory.** It was measured from raw `metadata.json`, every asset under `Assets/Object/RoboDojo` on compute3:

| measure | count |
|---|---|
| asset dirs with `metadata.json` | 743 (Rigid 461, Clutter 193, Geometry 66, Articulation 11, Garment 6, Fluid 4, Dynamic 2) |
| Geometry dir without `metadata.json` | 1 |
| assets with any passive functional frame | 36 |
| assets with any active functional frame | 62 |
| assets with any support entry | 93 |
| distinct tags: passive functional / active functional / support | 48 / 11 / 47 |

The perception catalogue covers 550 of these keys; the 193 Clutter keys have no mesh there. The file covers all 743,
so `n_assets` shows that nothing was filtered. Derived geometry is computed only where a mesh exists.

**The rule.** It is catalogue-wide and reads only the tag string and the frame structure. It never reads a task
config, and no category or key is a branch in the code.
- **Grammar:** the tag is split into words at `_`, `/`, `-` and digit runs, and the words are matched against a
  lexicon of general feature words (hole, slot, opening, socket, plug, keyhole, peg, pole, hook, handle, rim, hit,
  strike, lever, button, press, hinge, support, cradle, bottom, base, ...).
- **Structure:**
  - an entry with no `frame` (Garment `id`/`qpos` particle tags) is typed null;
  - a lever or hinge needs a joint (`parent_joint` resolved in the USD joint data), else it is typed null;
  - a support-section entry is a circle (centre + radius), so the word `slot` there types `hole`
    (`test_tube_slot/k`);
  - a support-section group with no feature word types `support` (`cell/k`, `basket/k`, `default/0`).
- **`<kind>_slot` tags** (`toast_slot1`, `hammer_slot`, `key_slot`, ...) are typed `slot`. The kind stays only in the
  tag name and is never part of the type.
- **An active frame with no feature word** (a tool's `center`) takes the type of the passive tags that name its
  owner's category. `hammer` → `hammer_slot` → `slot` is a catalogue-wide relation between asset files, not a task
  relation.

**Provisionally typed null and listed for Codex.** These carry checker or goal-role semantics:
- `check` (pen, oil_pen);
- `checkpoint` (toaster, car, electric_toothbrush, hammer);
- `be_placed` (factory_bolt);
- `coin_insert_center_same_side` / `_diff_side` (coin). The "same / different side" qualifier reads as a goal
  convention; the word `insert` alone would type `slot`.

**Typed null with no task semantics.** These are simply not targets:
- `GetParentLink` (egg_holder; no feature word);
- `bbox_0..7` (xylophone bar outlines; no feature word);
- the 12 garment particle tags (no frame).

**The final map.** The extraction PR emits the full map and a test pins it. The PR's journal entry lists the final
null set, which supersedes the provisional list here.

**Who rules.** Codex rules on each listed tag. The founders sign the ruling. A ruling changes the map in code, which
changes the extraction's sha256 and the file's sha256, and the pin is updated deliberately. Until a ruling, a listed
tag stays null: no target, and the verb falls back as in spec §3.

## Q4. Derived geometry: which methods and parameters? Are they recomputed per build, with hashes?

**Starting methods.** These are our existing, general implementations. The PR compares each against alternatives on
known shapes and on the catalogue, and picks by measured accuracy and refusals.

| field | today's method (code) | today's parameters | alternatives measured in the PR |
|---|---|---|---|
| support `surface` | top height map of the mesh, max z of dense samples per cell, dilated one cell (`dojo/skills/precision.py:91-133`), used by ORG2's `place_surface=mesh` (`dojo/skills/library.py:2220-2256`) | cell 3 mm, flat within 4 mm under the footprint | mesh slicing; convex hull; voxel height map |
| base band | lowest band of the mesh (`dojo/perception/base_refit.py:35-46` `base_band`; ORG2 stand placement `dojo/skills/torso.py:58` `base_radius`) | 8 mm (torso) / 12 mm (base_refit) | band 4 / 8 / 12 mm |
| `extent.lip` | holder half-width across the slot at a depth, from the mesh in a ±1 mm slab (`dojo/skills/coin_contact.py:123-141` `wall_intercept`) | at the squeeze depth (5 mm below the top) | a depth grid |
| slot width / length / depth | catalogue frame corners (`dojo/skills/library.py:3424-3435` `slot_frame`) or the mesh opening fit (`dojo/skills/insertion.py` `openings`) | | corners vs mesh fit |

**The cases the methods must reproduce:**
- ORG2: Geometry/drawer's resting surface 23 mm below its rim;
- COIN2: the vertical coin stand's half-width 3.1 mm at 5 mm depth, vs its 5.55 mm bbox half-width;
- COIN2: the stand's 2.6 mm slot.

**Recomputed per build, with hashes: yes.**
- Every `derived` entry carries its `method` and `params`.
- The build records the extraction script's sha256.
- Each key's `source.files` lists the sha256 of every file read for it (`metadata.json`, `geom.npz`, `links.json`).
- A changed mesh therefore changes the listing, and the file's sha256 with it.

## Q5. Sigma: can our perception give a pose sigma per object (xy, z, yaw)?

**Not per object today.**
- B1's records carry a pose, residual, confidence and per-view IoU, with no covariance
  (`dojo/perception/inverse_layout.py:24`).
- The only calibrated per-axis sigma we have is for a **held** object: `dojo/perception/held.py:5, 49-57, 697`. It is
  calibrated per geometry class, with gates of a 1 mm leading-point sigma and a 2 deg axis sigma.

**What we can give soon: per-class error tables from recorded runs.**
- These are 1-sigma xy / z / yaw by object class, scored dev-only against GT on spent ids. This is exactly
  strawman v2's 4.1.
- The numbers already measured:

| case | measure | value | n |
|---|---|---|---|
| ORG2, base xy (identity + pose fit) | median / p90 | 3.2 mm / 16.5 mm | 33 runs |
| ORG2, support poses more than 20 mm off | count / p90 | 5 / 33, p90 98 mm | 33 |
| ORG2, round-based figures, yaw off by more than 45 deg | count | 8 / 33 | 33 |
| COIN2, plane yaw at the pick | median / p75 / max | 3.9 deg / 9.9 deg / 17 deg | 32 samples |
| COIN2, offset along the normal | median | 0.8 mm | 32 samples |

- A fit-residual-based per-object sigma, the B1 cost's curvature as in `held._sigma`, is the next step on our side.
  It is not in this extraction.

**Until then,** catalogue targets on a tight tolerance measure first, as spec §2 says.

## Q6. Size and delivery: how large is the file? One file per build? Where does it live?

**Size.** The estimate is about 1 MB of JSON for 743 keys:
- the raw frames of the 183 assets that have any are 179 KB as JSON;
- the bbox, root data and source hashes add about 0.4 KB per key;
- derived entries are a few hundred bytes per tag.

The PR reports the measured size and the wall time.

**One file per build: yes.** It is named `f-catalogue-<catalogue_hash[:12]>-<script_sha256[:8]>.json`.

**Where it lives.**
- Built on compute3, CPU only, nice, under `~/dojo-data/fcat/`.
- The lead copies it to `~/dojo-data/fcat/` on compute4, and on compute6 if the hybrid runs there. Our agents use
  compute4 read-only.
- Its sha256 goes in `hybrid_pins`. The loader takes the path from the pinned configuration (spec §4).
- The file is not committed to git: it is derived from the RoboDojo assets and rebuilt per build.
- The PR commits a small synthetic fixture in the same schema for CI.

## Q7. Overlap: is COIN2's `refine_insertion` 22.8 mm correction part of the extraction (`derived`) or a runtime step?

**A runtime step.** Only the receptacle's intrinsic opening geometry goes in `derived`.

Why the refined pose is not intrinsic:
- **It is a pair fit.** `refine_insertion` (`dojo/skills/insertion.py:247-258`) searches xy shifts and yaws of the
  moved object's leading end, from the moved object's mesh at its current pose, against the receptacle's depth map.
  The result is a property of (moved object, receptacle, approach pose), not of the receptacle.
- **Its floor press cites the checker.** The `press_floor` lowering, which is most of the 22.8 mm, is motivated in
  its docstring by the checker ("the checker wants them >= 15 mm below the strip's top", `insertion.py:256-257`).
  Under the hard rule, that reasoning cannot enter an intrinsic catalogue.

What goes in `derived`: the receptacle's own opening at each frame, from its own mesh only:
- `extent.depth`, the floor below the frame;
- `extent.width` and `length`, the mesh opening fit vs the frame corners.

The PR measures how much of the 22.8 mm is the bank slot's own floor depth below `slot_opening`.
