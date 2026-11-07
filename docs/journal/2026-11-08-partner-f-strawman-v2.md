# 2026-11-08: partner team's F strawman v2 (functional targets), with our COIN2 / ORG2 markups folded in

Received on the coordination bus (`20261107T2159Z-misha-f-strawman-v2.md`), kept verbatim below. v1: `2026-11-07-partner-f-strawman.md`; our markups: `2026-11-08-f-markup-coin2.md`, `2026-11-08-f-markup-org2.md`. Section 5 lists five open disagreements for the 8 Nov planning; our lead's replies went back on the bus the same evening.

# F: functional targets: strawman v2 (8 Nov 2026), with Ilia's side's markups folded in

v1: `F-FUNCTIONAL-TARGETS-STRAWMAN.md` (2038a0f6). Markups (controlr main): `docs/journal/2026-11-08-f-markup-coin2.md`
(slot / hole / insertion, COIN2) and `docs/journal/2026-11-08-f-markup-org2.md` (support / stand / base, ORG2).
Status: a proposal; nothing built. Numbers quoted from the markups are theirs, not re-measured here.

## What changed from v1, and why
| change | from | evidence they cite |
|---|---|---|
| sigma per component in the target's own frame, not one position / one angle | COIN2 1.1 | slot failure was rotation alone: 3.9 deg median belief yaw vs +-1.4 deg tolerance, at sub-mm position error |
| a tolerance belongs to the verb (from target width / length and the object's thickness), compared with sigma before contact | COIN2 1.5 | atan((w - t) / L) = +-1.4 deg for a 1.94 mm coin in a 2.6 x ~28 mm slot |
| `extent.lip` wall profile, `owner_fixed`, `step` per provenance entry | COIN2 1.2, 1.4, 1.6 | stand narrows to 3.1 mm half-width at 5 mm depth vs 5.55 mm bbox half-width |
| a third source: tactile (wall probe), per-component sigma fusion, write audit | COIN2 2 | G3: yaw error 1.37 -> 0.45 deg median when bracketed; 4.6-12 deg left at 15-17 deg errors |
| new types `support` and `base`; root-frame origin with the catalogue's root-to-bbox offset in `from_asset` | ORG2 1 | tray's resting surface 23 mm below the rim the camera sees |
| rule (c) reversed: on disagreement the catalogue target (via an accepted identity) wins and the owner is re-looked | ORG2 2, COIN2 1.5 | base-from-image refit: 16 writes, 14 harmful (+2.6 to +29.5 mm); identity + pose fit base xy median 3.2 mm |
| refusal budgets for perception-only contact targets | ORG2 2 | support poses > 20 mm off in 5/33 runs (p90 98 mm) |
| scene state (place-site clearance, push goals, occupancy) is computed by the verb and logged, never stored on a target | ORG2 1 (+ occupancy, see disagreements) | |
| grasp sides stay skill-internal; the owner's pose exposes a yaw sigma instead | ORG2 3 | yaw off > 45 deg in 8/33 runs on round-based figures |
| verb requirements from COIN2's pitfalls (section 3) and both test plans merged (section 4) | both | |

## 1. Representation (v2)
```
FuncTarget
  name, owner     "<type>_<k>" per owner (catalogue tags keep their tag); stored in the owner's ROOT frame, so it rides
                  on the owner's pose (the root frame is what the official pose reads; ORG2)
  type            hole | slot | socket | keyhole | peg | hook | handle | rim | strike_point | lever | hinge
                  | support | base                                          (support, base: ORG2)
  point           hole / rim / socket / slot: the opening's centre on its face; peg: top centre; hook: arm tip;
                  handle: grasp / hang point; lever / hinge: a point on the joint axis;
                  support: centre of the usable resting surface; base: centre of the object's lowest band
  axis            into the opening | up the peg | along the hook arm | the joint axis | support: up | base: down
  across          a slot's long direction | a blade's direction | a support's long direction
  extent          {radius} | {width, length, depth} | {height};  slot widths precise to ~0.1 mm (catalogue meshes)
  extent.lip      optional {half_width_at: [[depth_m, m], ...]}: the holder's wall profile (COIN2)
  surface         support only: {z_min, z_max, flat_tol, rim_height} of the flat region, not the bbox (ORG2)
  corners, joint  optional, as v1 (slot footprint corners; {kind, range} for levers / lids / buttons)
  owner_fixed     true | false | unknown: contact sensing on the owner only when true (COIN2; see disagreement 2)
  sigma           {pos: [across, along, axis], ang: [about_axis, tilt]}, 1-sigma, in the target's frame (COIN2)
  conf            0..1
  provenance      [entries], each with its own `step`; a target without one is refused
```
The adapter stays: `as_frames(targets)` writes the shape the verbs read (z = axis, x = across, plus radius / corners /
joint / lip); `from_asset(obj)` turns catalogue frames into targets and carries the catalogue's root-to-bbox offset
(ORG2: from metadata, never a layout). For a located box with no identity, root = bbox centre and there is no offset.

## 2. Sources and records (v2: three sources, one precedence rule)
**(a) Perception (rd-gen's locate, part queries)**, as v1: part masks kept inside the owner's mask, then fits
(rim / hole ellipse back-projection, slot rectangle, peg cylinder, hook axis), wrist look for small parts. Record:
`{source: "perception", step, query, owner, build_hash, views, fit: {method, residual, n}, sigma}`.
Use per type (from the markups' data): positions of visible features (pegs, hooks, rims, openings) yes; **contact
geometry (base band, resting surface height, slot width, wall profile) only through (b)**; perception-only contact
targets are subject to the refusal budgets below.

**(b) Intrinsic catalogue** (metadata.json frames, USD / links joints, mesh-derived support surfaces, base bands, lip
profiles and slot widths), extracted once catalogue-wide into a hash-pinned file, reached only through an identity
accepted with its provenance record. Record: `{source: "catalogue", step, key, tag, catalogue_hash, file, json_path,
identity_record_id, derived: {method, params}}` (derived: for mesh-computed fields, e.g. ORG2's "flat within 4 mm"
resting surface, the lowest 8 mm base band). Never task configs, layouts, checkers, logs, per-task lists.

**(c) Tactile** (COIN2's wall probe; more methods later). Record:
`{source: "tactile", method: "wall_squeeze", step_range, arm, owner, reads: {turn_deg: value}, hand_offsets_mm,
fit: {apex_deg, d_mm, slope, resid, bracketed, n}, intercept: {source: "catalogue_mesh" | "fallback", key, depth},
reading_key}`. Reads proprioception and RoboDojo's own gripper observation only; **needs Codex's ruling** before any
scored use. Observes only `about_axis` and `across`; other components keep their prior sigma.

**Fusion and precedence.**
- Per observed component: sigma_post^-2 = sigma_prior^-2 + sigma_meas^-2, mean weighted alike; unobserved components
  unchanged; sigma_tact conditional on the fit (bracketed and |apex| <= delta - 2 deg: ~0.5 deg; near / past the
  outermost squeeze: >= |apex| / 2, or one more squeeze at the apex; not bracketed / residual over tolerance / no
  reading: no update).
- Catalogue vs perception (v1's rule c, reversed): within 3 sigma, the catalogue target stands; beyond, the
  catalogue target still stands, the owner's identity / pose is flagged for a re-look, and the disagreement is
  logged. A scene's success test never decides.
- Every belief write by a new writer (`belief_write`, writer name) is logged and A/B-gated: helpful >= 2 x harmful.

**Refusal budgets for perception-only contact targets** (ORG2 2; budgets owned by each verb, set by measurement in
4.2): refuse when the owner has no accepted identity AND any of: the target's 1-sigma point error exceeds the verb's
budget (stand place: 0.5 x base radius; support height: the 3 mm release gap, then fall back to a guarded descent,
not an open-loop release); the evidence fails a visibility gate; the owner's own pose sigma already exceeds the budget.

## 3. Verbs (v2)
Arg type `target` / `targets` as v1; frame args accept a frame or a target name. Postconditions as v1 (measured by
the library from a re-locate or a wrist look), plus:
- **place_on / place_at onto a support**: the object's base within 0.5 x its radius of the support target's point,
  its bottom on the resting surface (z_min..z_max), upright; place-site clearance computed by the verb, logged with
  the targets it used (ORG2).
- **push_onto**: the face extent / axes from the support target; cover vs inside computed from relative sizes.

Verb requirements from COIN2's pitfalls (insert / insert_each / put_in_slots and any contact verb):
1. compare `sigma.ang.about_axis` (and `sigma.pos.across`) with the verb's tolerance; when larger, measure first
   (tactile on a fixed owner, or a wrist look) and grip to tolerate the residual;
2. tactile probe: subtract each squeeze's commanded (FK) offset along the closing axis before fitting; fit the slope;
   take the intercept from `extent.lip`; over-determine (an extra squeeze at the apex) when |apex| is near the
   outermost squeeze;
3. no jaw-width sensing on objects under ~2 mm thick: probe the holder's walls instead;
4. a perception "not held" on a thin object does not delete the held object alone: a second view, or the object
   seen elsewhere;
5. postcondition "at least half its depth past the opening" kept, measured by a re-locate; seat deeper before
   letting go;
6. the verb gets the episode's remaining steps and skips probing below a reserve (the contact profile costs ~50).

Grasp sides (ORG2 3): skill-internal; the owner's pose carries a yaw sigma (and, for non-box outlines, the silhouette
yaw and symmetry from `dojo.skills.shape`), and a skill keying on the yaw widens its candidates or looks when it is
large.

## 4. Test plan (merged)
**4.0 CPU.** Adapter round trip, catalogue-wide, per type (slot / opening / insert / hole tags against `insert_into` /
`disc_into_slot`; support height maps and base bands against ORG2's mesh-derived ones), bit-exact; the perceived-pose
path (`coin_contact.slot_targets_from_pose`) equal at the catalogue pose; resting-surface extraction on known shapes
(the tray yields its lower surface); fits on synthetic renders through the public cameras (v1); tactile model tests
(per-squeeze offset no bias, apex past the outer squeeze, sigma fusion monotone, no reading no write); a sigma
calibration helper (fraction within 2 sigma); ORG2's stand-site / cover-goal cases fed targets instead of assets.

**4.1 Recorded, no simulator, spent dev ids listed before use.** COIN2: 40 camera re-test episodes, 80 G3 panel
episodes, the dev screen's 260; ORG2: the 33 frame-0s of its slice plus any cell with a support. Score per source
(perception, catalogue + identity, tactile) against the catalogue frame at dev-only GT (scorer side): about-axis deg,
across / along mm, point / z / extent error per support kind, sigma calibration, refusal rate. Decision wanted: does
any perception-only contact target meet its budget (0.5 x r_base, 3 mm)?

**4.2 G3, spent dev ids.** Parity first (each bundle fed by the adapter reproduces its G3 numbers on the same
records); then dose-response with belief offsets drawn from 4.1 (stand xy 0 / 5 / 10 / 20 mm, support z 0 / 3 / 6 /
10 mm, slot yaw from the measured camera errors, applied to owner and occupant together); arms perceived / catalogue /
perceived + tactile; budgets set from where placement and release break. Generality: the same harness on another thin
object in a fixed slotted holder, if the catalogue has one. **4.3 Camera** only after 4.2; the queued COIN2-CAM
(compute4) and ORG2 camera A/B (compute6) are the baselines targets must not lose against.

## 5. Open disagreements (short; to settle in the planning)
1. **Occupancy.** COIN2 proposes `occupant: <handle>` on a slot target. I'd keep it out of the target (it changes when
   the object leaves; ORG2's own rule for scene state) and derive it per read from the belief (an object whose pose
   lies in the target's volume), logged with the write that moves owner and occupant together. Same behaviour, no
   stored scene state.
2. **`owner_fixed` from "the identity's type".** In the hybrid runtime mobility is the instance's (locate's
   `movable`), never the asset's type (GENERAL-LIB-INTERFACE). I'd set `owner_fixed` from locate's movable = false
   confirmed by measured non-motion (two reads), `unknown` otherwise; Codex to rule if the catalogue type may be used.
3. **Tactile in scored runs.** It reads RoboDojo's official gripper observation (`_official`), which no registry read
   token covers. Until Codex rules, tactile is a dev-only source (off in scored configurations); agree?
4. **Scope of the reversed rule (c).** Agreed for contact geometry (base, support, slot width, lip). For positions of
   visible features (peg tops, hook tips, rim centres) on an owner whose pose may be off (support poses p90 98 mm),
   a large disagreement may mean the pose, not perception, is wrong: I'd re-look before trusting either, not default
   to the catalogue. Data from 4.1 should decide per type.
5. **Perception-only narrow slots.** If head-camera fits cannot give ~0.1 mm widths (COIN2 1.5), should slot targets
   without an identity be refused for insertion outright, or allowed with a mandatory measure-first (tactile / wrist)?
   I lean to the latter, since it covers objects whose identity is not accepted.
