# 2026-11-08 — markup of the partner's F strawman: slot / hole / insertion (COIN2)

On [the F strawman](2026-11-07-partner-f-strawman.md), for the joint planning. Evidence: COIN2
([journal](2026-11-07-dojo-coin2.md), [G3 report](../experiments/2026-11-07-coin2-contact.md),
[camera prereg](../experiments/2026-11-07-coin2-camera-prereg.md)). No LLM calls, no sim for this note.

**How we get a slot today** (answer to their §5 COIN2):
- **Destination slot (the bank's opening):** the receptacle's believed pose × its catalogue functional frame
  (`slot|opening|insert` tags), then a mesh fine fit (`insertion.refine_insertion`: the nominal frames sit 0.5–1 mm off
  and at the surface; the fit lowered them 22.8 mm).
- **Source slot (the stand the coin stands in):** the stand's believed yaw sets the coin's plane (slot settle), and
  `disc_align=walls` measures it by touch.
- **Errors measured:**
  - camera belief at the pick, 32 samples: plane yaw 3.9° median (p75 9.9°, max 17°); offset along the normal 0.8 mm
    median;
  - at release in G3 under those errors: slot-plane error 1.4° and 1.15 mm across the slot (median).
  - No wrist look is needed for the bank. The hover refine of the coin failed ("perception not ok") in the camera
    rows.

## 1. Does `FuncTarget` carry what the slot-insertion profile needs?

Mostly: `point`, `axis` (into), `across` (long direction), `extent.width/length/depth`. Missing or under-specified:

1. **Anisotropic sigma.** One position sigma and one angle sigma cannot express what matters. A slot needs:
   - position across it (thin direction), along it, and along `axis`;
   - rotation about `axis` (the plane's yaw for a vertical slot), and tilt.
   COIN2's failure was the rotation alone (3.9° vs a ±1.4° tolerance) at sub-mm position error. Proposal:
   `sigma = {pos: [across_thin, along, axis], ang: [about_axis, tilt]}` (1-sigma, the target's own frame).
2. **The wall profile around the opening.** The wall probe needs the holder's outer half-width at the squeeze depth.
   For the coin stand that is 3.08–3.16 mm 5 mm below the top, against a 5.55 mm bbox half-width: the stand narrows at
   the top, so extents are wrong here. Proposal: `extent.lip = {half_width_at: [[depth_m, m], ...]}`, from the
   owner's catalogue mesh (intrinsic) or a perceived hull. Absent = the verb may not use the wall sensor's intercept
   (yaw only).
3. **Occupancy.** A source slot holds the object we pick (`occupant: <handle>`). The probe rewrites owner and occupant
   together, and the verb must know the occupant's thin axis lies across the slot.
4. **Fixed owner.** Tactile probing and wall contact are only legal on an owner that cannot move (here a fixed
   Geometry object). Proposal: `owner_fixed: true | false | unknown`, from the identity's type or a measured
   non-motion; the verbs refuse contact sensing when it is not `true`.
5. **Tolerance belongs to the verb, not the target.** The angular tolerance comes from the target's
   width / length and the object's thickness: atan((w − t) / L) ≈ ±1.4° for a 1.94 mm coin in a 2.6 mm,
   ~28 mm-long slot. The verb compares it with `sigma.ang.about_axis` and measures before contact when sigma is
   larger. This only works if `width` is precise to ~0.1 mm, which favours catalogue meshes over head-camera
   rectangle fits for slots this narrow.
6. **Measurement time** (`step`) per provenance entry: the source holder is static, but the destination is re-read
   after a handover or a bump.

## 2. The wall probe as a tactile provenance, and its sigma update

Record per target:
```
{source: "tactile", method: "wall_squeeze", arm, owner, step_range,
 reads: {turn_deg: official_jaw_value}, hand_offsets_mm: {turn_deg: commanded TCP offset along the closing axis},
 fit: {apex_deg, d_mm, slope, resid, bracketed, n}, intercept: {source: "catalogue_mesh" | "fallback", key, depth},
 reading_key: "<arm>_ee_joint_state_official"}
```
- **Legality:** it reads only proprioception, RoboDojo's own gripper observation (strict v2 `_official`, as
  `grip_check` does). It needs Codex's ruling, since the registry has no read token for it.
- **What it observes:** rotation about the slot's normal (yaw) and position across the slot (d). Nothing along the
  slot, nothing in z: those keep their prior sigma.
- **Sigma update** (independent-measurement fusion, per observed component only):
  σ_post⁻² = σ_prior⁻² + σ_tact⁻², mean weighted the same way. σ_tact is conditional on the fit:
  - bracketed, apex well inside the outermost squeeze (|apex| ≤ Δ − 2°), residual small: σ_yaw ≈ 0.5° (G3 panel: |err|
    1.37° → 0.45° median at those errors);
  - apex near or past the outermost squeeze: σ_yaw ≥ |apex| / 2 (G3: 15–17° errors left 4.6–12°). Or better, one
    more squeeze centred on the apex before writing;
  - not bracketed, residual above tolerance, or no reading: no update (today: no write).
  - Offset: σ_d ≈ 0.3 mm in the twin, but only after the per-squeeze correction below.
- **Write audit:** a write is logged (`belief_write`, writer `wall_probe`). It is a new writer in any A/B, so it is
  gated by helpful ≥ 2 × harmful.

## 3. Pitfalls a general implementation must handle (from COIN2)

| pitfall | what we measured | what the general code must do |
|---|---|---|
| belief yaw error > slot tolerance | camera plane yaw 3.9° median vs ±1.4°; 14 / 30 camera picks lost in the lift (the coin binds in the slot); G3 mode B: 88 → 19 at +10°, 6 at −12° | compare σ_about_axis with the tolerance (§1.5); when larger, measure first (tactile or a wrist look), and grip in a way that tolerates the residual (`disc_snap`: close as the fingertips clear the walls, 12 mm of the object held instead of 9) |
| the probe's bias at large errors | 3 squeezes / 3 unknowns: no consistency check; apex −7.4..−9.6 for a true −10, and 4–12° left at 15–17° | add a squeeze at the fitted apex when \|apex\| is near Δ (over-determined fit); inflate σ otherwise |
| per-squeeze hand offset | the first squeeze comes off a descent line whose IK leaves ~0.9 mm (kinematic test). The offset term is 0.0164 per mm, ~8.6° of yaw per mm | subtract each squeeze's commanded (FK) offset along the closing axis before fitting; record `hand_offsets_mm` (BACKLOG; the queued camera A/B still runs the uncorrected probe) |
| sensor model varies by arm / layout | V slope 0.0016–0.0021 per °; intercept from mesh 0.0362 vs 0.0347 measured | fit the slope; take the intercept from `extent.lip` |
| thin objects give no jaw-width signal | the official value read 0.0 on every coin close (40 / 40), whatever the yaw | tactile sensing on the holder's walls, never on a sub-2 mm object |
| held_look's false "dropped" | camera: 7 / 17 held coins read "dropped", 0 / 39 empty hands read "held" | a perception "not held" on a thin object must not delete the held target alone: require a second view, or the object seen elsewhere |
| the hold's depth, not its existence | coins high in the slot 9 / 40 (pushed up the pads at contact) | keep the strawman's "at least half its depth past the opening" postcondition, measured by a re-locate; seat the object deeper first (`disc_seat`) |
| step budget | the contact profile costs ~50 steps (B median 214 / 300; 1 budget end in 60) | the verb gets the episode's remaining steps and refuses or skips probing below a reserve |

## 4. Proposed test plan for slot / hole targets

**4.0 CPU.**
- Adapter round trip, slot types: for every catalogue key with `slot|opening|insert|hole` frames, the targets
  `as_frames(from_asset(o))` give to `insert_into` / `disc_into_slot` must equal today's (bit-exact).
- The perceived-pose path (`coin_contact.slot_targets_from_pose`) given the catalogue pose must equal it too.
- Tactile model: extend `tests/dojo/test_coin_contact_rules.py` (kinematic wall probe with synthetic official values):
  - a per-squeeze hand offset (must not bias the apex after the fix);
  - apex past the outer squeeze (must add a squeeze or inflate σ);
  - σ fusion monotone (σ_post ≤ σ_prior, unobserved components unchanged);
  - no reading → no write.
- Sigma calibration helper: given (estimate, σ, truth) rows, the fraction within 2σ.

**4.1 Recorded, no sim.** On spent dev ids only, listed before use:
- **Inputs:**
  - our 40 camera re-test episodes, with physics replays and layout GT for scoring only;
  - the 80 G3 panel episodes;
  - the dev screen's 260.
- **Score:**
  - perceived source / destination slot targets against the catalogue frame at GT: about-axis °, across mm,
    along mm;
  - σ calibration per source: perception, catalogue + identity, tactile.
- **Report alongside:** the held_look table, and the stage table from `scripts/skills/coin2_stages.py` (fingertip
  clearance at the grip, hold-loss stage, release slot-plane error / across offset, end state).

**4.2 Sim (G3 physics, spent ids).**
- **Harness:** mode B with per-layout belief errors drawn from the measured camera errors
  (`--belief-offset-file`, `scripts/skills/coin2_panel_offsets.py`), applied to the slot's owner and occupant
  together.
- **Arms:** perceived target, catalogue target, perceived + tactile fusion, on the same ids.
- **Gates:**
  - the strawman's postconditions, measured by a re-locate;
  - the stage table above;
  - σ calibration on the tactile updates.
- **Generality:** the same harness on another thin object in a fixed slotted holder (a card or phone in a vertical
  stand, if the catalogue has one). The contact sequence covers discs only today, which is exactly what this would
  expose.
- **Camera:** the queued COIN2-CAM A/B (70 rows on compute4) is the first such test.
