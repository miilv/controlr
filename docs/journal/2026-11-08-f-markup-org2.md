# 2026-11-08 — ORG2 markup of the F strawman: support / stand / base / handle / place-site targets

Our input on the partner's [F strawman](2026-11-07-partner-f-strawman.md) for the joint planning, from what ORG2
built and measured ([G3 report](../experiments/2026-11-07-org2-g3.md), [camera prereg](../experiments/2026-11-07-org2-camera-prereg.md)).
Docs only; nothing here is built.

## 1. Does `FuncTarget` carry what stand placement, support-preserving release and push goals need?

What our code reads today (all from the support's and the object's catalogue geometry placed by their believed pose):

| ORG2 use | reads | in FuncTarget? |
|---|---|---|
| place onto a support's top (`_on_targets`, `place_surface=mesh`) | the support's **usable resting surface**: a height map of its mesh top, flat within 4 mm under the object's footprint; release 3 mm above it | **no**: there is no `support` type, and `point` / `extent` cannot say "the tray is 23 mm under the rim, open at the front" |
| stand placement (`stand_offset=base`) | the object's **base**: the lowest 8 mm band of its mesh, its radius about the origin; positions within 0.5 x that radius of the stand's centre | **no**: there is no `base` type for the moving object |
| place-site clearance (`torso.stand_points`, `clear_vias`) | other objects' current footprints, signed clearance, fixtures within 20 cm | **should not be** (scene state, changes every step) |
| push goal (`push_goal=clear`, `cover_box`) | the support's face extent and axes, the pushed object's footprint extents, pending place sites | partly: `extent {width, length}` + `across` cover the face; cover vs inside comes from relative size and is computed |

Proposed additions, small and intrinsic:
- **`support`** (owner = the fixture): `point` = centre of the usable resting surface, `axis` = its up normal,
  `across` = its long direction, `extent {width, length}` of the flat region (not the bbox), plus optional
  `surface: {z_min, z_max, flat_tol}` and `rim_height` (a recess or lip above the resting surface). Tray, stand plate,
  mat and frame are all `support` targets.
- **`base`** (owner = the object to be placed): `point` = the centre of its lowest band, `axis` = down, `extent
  {radius}`. Today it is derived from the mesh. Its offset from the root is the same catalogue-intrinsic offset the
  adapter needs anyway.
- Place-site clearance and push goals stay **verb-side computations** over targets plus the belief. They are logged
  with the targets they used: the chosen site, the clearance in mm and the neighbours considered. Storing them as
  targets would freeze scene state into an intrinsic object.

**Partner's ORG2 question (base origin).** Our place targets are poses of the object's **root frame**, which is what
the official pose reads. The bbox centre (`asset.center`, from the catalogue's aligned bbox in the root frame) is only
used to put the footprint over the target xy. For the figure assets the two coincide, and the base disc is centred on
the root to within 1-2 mm. Articulations are posed on their articulation-root prim (`dojo/skills/artroot.py`). The
root-to-bbox offset therefore comes from the catalogue's metadata, never from a layout, and `from_asset` should carry
it.

## 2. What the base-refit null implies for perception-sourced base / support targets

The base-footprint refit (lowest band slid in xy against the fg mask, image-quality gate) made **16 writes on 16
recorded camera runs: 14 harmful (+2.6 to +29.5 mm), 1 helpful, 1 neutral**. The gate held back 19 more ("base not
seen"). At head-camera range (~3 mm/px) the contact band sits in shadow, which the segmenter marks unknown, or under the
object's body, so the search follows the silhouette. Meanwhile the whole-silhouette identity + pose fit had a base xy
error of median 3.2 mm, p90 16.5 mm (33 runs). Its two failure classes are identity / pose failures, not contact reads:
one wide figure 14-26 mm off in 5/5 arms, and 2 gross misreads.

Implications:
- **Contact geometry (base, resting surface, support height) should come from the catalogue, placed by an accepted
  identity's pose**, not from a direct perception read of the contact. That also holds where the visible top misleads:
  the tray's resting surface is 23 mm below the rim the camera sees.
- For base / support types, the strawman's rule (c) "disagree -> the verb takes the perceived one" should be
  **reversed**: keep the catalogue target and flag the identity / pose for a re-look. Our data says the perceived
  contact read is the biased one.
- **A verb refuses a perception-only base / support target** when any of these holds:
  1. it has no accepted identity of its owner;
  2. its 1-sigma point error exceeds the verb's own budget: 0.5 x base radius for a stand place (6-35 mm for our
     figures); the 3 mm release gap for a support height, in which case it falls back to a guarded descent rather than
     an open-loop release;
  3. its evidence fails a visibility gate (contact band seen in fewer than N usable points per view, or a score at
     the current pose below the gate);
  4. the owner's own pose sigma alone already exceeds the budget (support poses were > 20 mm off in 5/33 runs, p90
     98 mm).

## 3. Appendage-aware grasp: a FuncTarget type or skill-internal?

**Skill-internal.** The "graspable side across the appendage line" (`grasp_torso`) is computed per call:
- from the mesh at the current pose: section width at a height, closing 90 or +-45 deg off the long horizontal axis;
- against the gripper geometry: finger corridors, the palm box;
- against the arm's base bearing and the neighbours.

None of that is intrinsic to the object: it changes with the arm, the yaw and the scene. A `handle` target fits
objects with a designed grasp feature (a mug's handle, a tool's grip).

What the skill does need from perception is a **yaw sigma**: the appendage line is only as good as the believed yaw.
Perception's yaw was off by more than 45 deg in 8/33 runs on round-based figures. A skill that keys the closing
direction on that line must widen its candidate set, or take a wrist look, when the yaw sigma is large. Better to
expose `sigma` on the owner's pose than to invent a grasp target.

## 4. Proposed test plan for support / stand targets

**4.0 CPU (synthetic + catalogue):**
- Adapter round trip for `support` / `base` catalogue-wide: `as_frames(from_asset(o))` against today's mesh-derived
  support height maps and base bands. This pins the point, axis and extent conventions.
- Resting-surface extraction on known shapes (recess / tray, flat stand plate, mat, frame): usable region vs bbox top.
  The tray case must yield the lower surface.
- Stand-site and cover-goal cases with neighbours, kept from `tests/dojo/test_org2.py` (stand points, signed
  clearance, vias, cover region, path clearance), fed targets instead of assets.

**4.1 Recorded frames, no simulator:**
- On the listed recorded runs of our slices (the 33 `c4b_organize_*` frame-0s, plus any other cell with a support),
  score the base / support targets three ways:
  1. catalogue via the accepted identity pose;
  2. perception-only;
  3. the 3-sigma comparison.
- GT is dev-only and scorer-side. Report per support kind: point, z and extent error; refusal rate under section 2's
  rule.
- The decision we want from it: whether any perception-only support / base target meets the 0.5 x r_base and 3 mm
  budgets.

**4.2 G3, spent dev ids:**
- (a) Parity: the ORG2 bundle fed by the adapter's targets must reproduce its G3 numbers on the same records.
  Evidence: dev alarm upright 9/9; panel B alarm 13/18 and figures placed 3/18.
- (b) Dose-response: inject the 4.1 error distributions as belief offsets (`--belief-offset`) on stand xy (0 / 5 /
  10 / 20 mm) and support z (0 / 3 / 6 / 10 mm). This finds where placement and release break, and sets each verb's
  refusal budget from measurement rather than by assumption.

**4.3 Camera** only after 4.2. The ORG2 camera A/B already queued on compute6 (positions 82-245) is the baseline that
targets must not lose against.
