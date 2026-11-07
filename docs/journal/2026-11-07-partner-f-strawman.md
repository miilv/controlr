# 2026-11-07: partner team's strawman for F (functional targets from perception), for the 8 Nov joint planning

Received on the coordination bus (`20261107T2122Z-misha-f-strawman.md`), kept verbatim below. It is a proposal from their side; nothing is built. Our input for the planning: COIN2 (slot insertion via wall probe), ORG2 (base-origin perception null, stand placement), MATCH2 (moving-object pick), SAMINT r2 (general SAM union, phantom guards), the legality inventory and identity-source policy.

# F: functional targets from perception: a strawman (7 Nov 2026, for the joint planning on 8 Nov)

Status: a proposal. Nothing is built. Marked [believed] where I read the code once but did not test it.
Inputs: rd-gen's coverage map (`RD-GEN-COVERAGE-2026-11-07.md`: F is needed by 15 variants and completes 9 alone),
joint-planner's `GENERAL-LIB-INTERFACE.md` (identity provenance, item 4; locate extras), and the frame verbs' code.

## 0. The gap
The frame verbs (insert, insert_each, put_in_slots, hang, thread_on_pegs, press_part, strike) read asset frames:
`asset.functional` / `active_functional` / `support`, pose7 lists in the object's root frame, plus `joints` for
levers. A located box has none, so these verbs refuse it ("<handle> has no frame ..."). F gives them one target
representation. Two legal sources fill it: perception, and the intrinsic catalogue reached through an accepted
identity. Each verb takes the target the same way, whichever source produced it.

## 1. Representation
```
FuncTarget
  name        "<type>_<k>" per owner, stable in the episode (a catalogue tag keeps its tag as name)
  type        hole | slot | socket | keyhole | peg | hook | handle | rim | strike_point | lever | hinge
  owner       the belief handle it belongs to; stored in the OWNER's frame, so it moves with the owner's pose
  point       xyz: hole/rim/socket: the opening's centre on its face; peg: its top centre; hook: the arm's tip;
              slot: the opening's centre; handle: the grasp/hang point; lever/hinge: a point on the joint axis
  axis        unit: hole/socket/keyhole/slot/rim: INTO the opening; peg: up along it; hook: along the arm,
              away from its mount; handle: the opening's through-direction; lever/hinge: the joint axis
  across      unit, optional: a slot's long direction; a keyhole's blade direction (axis x across = the face)
  extent      {radius} | {width, length} | {depth} | {height}: whatever the type needs
  corners     optional: a slot's footprint corners. That is how catalogue slot frames are given today [believed]
  joint       optional {kind: revolute | prismatic, range: [lo, hi]}: levers, lids, buttons
  conf, sigma confidence 0..1; 1-sigma position (m) and angle (deg)
  provenance  see 2; every target carries one, and a target without one is refused
```
Compared with today's frames (my reading of the verbs' code, to be confirmed by each owner):

| verb | reads today | FuncTarget fields it needs |
|---|---|---|
| insert | object's active frames (plug point) onto the receptacle's passive frames (outlets); a disc into a slot | point, axis (+ across for blades/keys), extent.radius/width; the object's own leading-end target |
| insert_each | `hole_frames(rec)`: (tag, T, radius) | point, axis, radius, depth |
| put_in_slots | slot tag -> footprint corners in the container frame | point, axis, across, width, length (or corners) |
| hang | hook frame; the object's handle frame (outside the bar) + the handle's hole depth | hook: point, axis; handle: point, axis, extent |
| thread_on_pegs | peg frames + the piece's `through_hole` from its mesh | peg: point, axis, radius, height; piece: hole target |
| press_part | a frame on a moving link + the joint axis/range from `joints` | point, joint (axis, kind, range) |
| strike | `frame_points_w(ins, tag)`: positions only | point (+ axis as the surface normal) |

**One adapter, both ways.** `as_frames(targets) -> {tag: {"frame": [pose7]}}` writes targets into the shape the
verbs already read, with the pose's z along `axis` and x along `across`, plus radius / corners / joint where the verb
reads them. `from_asset(obj)` turns catalogue frames into FuncTargets with catalogue provenance. The planner then
sees one list per object (`api.scene()` -> `targets`), and the verbs' motion code does not change. The z/x
convention per type must be checked against each verb's reading before anything else, by a round-trip test (4.0).

## 2. Sources (legal only) and how each is recorded
**(a) Perception, from rd-gen's locate.** It is the same service and the same one path, extended to parts:
`locate(queries=[{"part": "hole", "of": <handle>}])`, i.e. part nouns from the planner (hole, slot, hook, peg,
socket, keyhole, handle, rim, lever). No task vocabulary and no candidate lists.
- Masks: text-prompted SAM on the part, kept only inside the owner's mask in the same view. A part outside its
  owner is rejected. Small parts (< ~1 cm at head-camera range [believed]) need a wrist look, through the same
  `look_at` path. The part locate asks for one (`needs_look`) instead of guessing.
- Geometry, per type:
  - rim / hole / socket: an ellipse fit to the part's contour per view, back-projected to a 3D circle (centre,
    normal, radius), with the views fused;
  - slot: a rectangle fit on the owner's face plane (the plane from the owner's hull or box face), giving across,
    width and length;
  - peg: a cylinder fit to the part's hull (axis, radius, top);
  - hook: the part hull's principal axis, pointed away from the owner's body;
  - lever / hinge axis: rarely visible. Take it from the catalogue (b), or from motion: the part's point before and
    after a small press or push, from two reads (a later step, with A).
- Record per target: `{source: "perception", query, owner, service build hash (SAM weights + code), views:
  [{cam, step, mask_score, area_px}], fit: {method, residual, n_points}, conf, sigma}`. Codex can check:
  - the query is the planner's text;
  - the build hash is the pinned one;
  - nothing on the path reads task config, layout, checker or catalogue files (the same audit as locate today).

**(b) Intrinsic catalogue frames.**
- What: metadata.json functional / active_functional / support frames and the links / USD joints (axis, range), per
  catalogue key.
- How extracted: ONE script, catalogue-wide (every key, no task list), into a frozen file whose sha256 sits in
  `hybrid_pins`. It reads only intrinsic asset files (object.usd, metadata.json, description.json), per Codex's
  allowed list in GENERAL-LIB-INTERFACE item 4.
- When used: only through an identity accepted with its provenance record (catalogue_hash = build hash, per-view
  evidence, margin). The targets are placed by that identity's pose.
- Record per target: `{source: "catalogue", key, tag, catalogue_hash, file, json_path, identity_record_id}`. Codex
  can check the script's reads, the frozen file's hash and the identity record.
- Never: task/RoboDojo/config/**, tasks/** checkers, Eval_Layout, Traj, logs, per-task lists or counts, and
  `catalog.py`'s checker-relevant attributes.

**(c) Both sources at once.** A catalogue target placed by the identity's pose is checked against a perceived one of
the same type:
- they agree (within 3 sigma): keep the catalogue's (more precise for holes);
- they disagree: keep both, lower the conf, and the verb takes the perceived one;
- the comparison is logged.
A scene's success test never decides between them.

## 3. Verbs: what changes and the postcondition each uses
- **Arg types.** One new arg type, `target` / `targets` (`of` = the owning object arg). check() requires the name to
  exist on that object's target list, its type to be one the verb accepts, and conf >= the verb's minimum. Existing
  frame args accept a frame name OR a target name, so planners keep working.
- **Measuring.** Every postcondition below is measured by the library on the world after the verb. It reads a fresh
  re-locate of the owner and the moved object (or a wrist look), never a scene's test. Where nothing can be measured
  yet, it says "reported by the skill", as today.

| verb | change | library postcondition |
|---|---|---|
| insert(obj, receptacles, turn_deg, target?) | the receptacle's hole / socket / keyhole / slot target; the object's leading-end target (perceived tip, catalogue plug point, or its long axis end) | the leading end within radius + 4 mm of the target axis and at least half its depth past the opening; with turn_deg, the measured roll about the axis within 15 deg |
| insert_each(objects, receptacle) | holes = the receptacle's hole targets | per object, today's seat test against the target: centre within radius + 4 mm, within 30 deg of the axis, depth within its tolerance |
| put_in_slots(objects, container, slots) | slots = slot targets (or frames) | centre inside the slot rectangle in its plane; thin axis across within 15 deg; bottom past the slot's face |
| hang(objects, rack, hook, handle?) | hook target on the rack, handle target on each object | not held; the hook's tip inside the object's handle opening (hull); the object's bottom above its support |
| thread_on_pegs(groups, base, pegs) | peg targets on the base, hole targets on the pieces | centre within peg radius + 4 mm of the peg axis; below the peg's top; within 15 deg of flat |
| press_part(obj, frame) | lever / button target with its joint | the part's point moved along its joint by >= 0.5 of the commanded stroke (two reads) |
| strike(tool, instrument, points) | strike_point targets (point + normal) | none measurable without contact sensing: "reported by the skill" (disclosed) |
| A: swing_part(obj, part, angle_deg) | part = a lever / lid / hinge target with a revolute joint | the part's point turned about the joint axis by the angle within 10 deg (two reads) |

## 4. Test plan, without the simulator first
**4.0 Synthetic, CPU.**
- Adapter round trip: for every catalogue key with frames (catalogue-wide), `as_frames(from_asset(o))` must equal the
  frames the verbs read today. This pins the per-type axis conventions.
- Fits: render synthetic rims, holes, slots, pegs and hooks through the public camera model
  (`dojo.perception.cameras`, head + wrists) as masks, with noise, partial occlusion, a wrong owner and a too-small
  part. Gate per type on point, axis and radius error.
- Verbs on the kinematic env (as tests/dojo/test_shape.py does): insert_each, put_in_slots and thread_on_pegs fed a
  target list instead of FrameAssets frames must plan the same motions.

**4.1 Recorded, still no simulator.** rd-gen's recorded frames from spent dev ids, listed by id before use.
- Run the part locate offline.
- Score against the catalogue frames placed at the dev-only GT pose. This happens scorer-side only and never
  reaches the runtime.
- Report per type: recall, false targets, point / axis / radius error, and how often a wrist look was needed.

**4.2 Simulator practice, spent ids only.**
- Ids: dev layouts generated >= 100000 and ids already read; never a scored slice.
- Arms: perceived targets vs catalogue targets vs today's routines, on the same ids.
- Then the learning gate's dev layouts.
- Postcondition texts are checked against what the views show (verifier agents), not against a success flag.

## 5. Open questions for Ilia's side (mapped to your streams as the lead named them; I have not seen their docs)
- **COIN2 (slot plane).**
  - How do you get the slot's plane today: a perceived plane, or the stand's identity + its slot frame?
  - Could your read be emitted as a `slot` FuncTarget (point, axis = into the slot, across, width) with its
    provenance?
  - What normal / offset errors do you measure (your `coin_slot_eval.py` scores normal deg and offset mm), and does
    it need the wrist look at the hover?
- **ORG2 (base origin).**
  - Catalogue frames are in the asset's root frame; located boxes are centred on their bbox.
  - Is your base origin the root frame, the bbox centre, or a measured point?
  - Where does the root-to-bbox offset come from in your path? The adapter needs exactly that offset.
- **MATCH2 (moving pick).**
  - Targets here live in the owner's frame and ride on its tracked pose. Is that enough for a moving owner, i.e.
    does your tracker give the owner's pose at the grasp step?
  - Or do targets need a timestamp and velocity?
- **General perception (SAM ∪ v2b).**
  - Can part prompts run on your union?
  - What was v2b trained on (catalogue-wide renders, or task layouts)? That decides whether it is general
    perception under D-C17.
  - Which part nouns segment reliably at head-camera resolution?
  - Would you accept a wrist look as the default for parts under ~1 cm?
- **Catalogue frames under your legality inventory.**
  - Does a frozen catalogue-wide extraction of metadata.json frames + joints, used only through an accepted identity,
    meet your terms?
  - Are tags like `<kind>_slot` (naming the kind that fits) intrinsic asset facts, or task semantics to drop?
- **Ownership (to agree).**
  - Part locate: rd-gen + your perception.
  - The extraction script: your inventory.
  - Adapter, arg type and verbs: glib (joint-planner).
  - Fits and tests: whoever takes 4.0.
