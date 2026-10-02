# Gap 6: Is ≤2 mm contact work feasible on a YAM-class, position-controlled, $3k arm, and with how much data?

*Written 2026-10-02. Everything below was read in primary sources: repos at the commits given, vendor pages, arXiv HTML, and the public Robocurve Tier 1 artifacts. **[derived]** = my computation. **UNVERIFIED** = could not be checked. "Clearance" is ambiguous in the literature, so I always say whether it is radial or diametral. Derivation script: `gaps/gap-6-sag.py`.*

## Bottom line

1. **The YAM is a low-gain joint-impedance arm, not a stiff position servo.** Every joint runs Damiao MIT mode, τ = Kp(q* − q) + Kd(q̇* − q̇) + τ_ff, with host-side MuJoCo gravity compensation at 250 Hz and no integrator. Defaults: Kp = 80/80/80/10/10/10 N·m/rad, Kd = 5/5/5/1.5/1.5/1.5.
   - Steady-state error = unmodelled torque / Kp.
   - Grasp-point stiffness is about **0.34 N/mm** in the softest direction **[derived]**: compliant insertion is natural, absolute accuracy is poor.
2. **Accuracy is cm-level open-loop and 1.5–7 mm with correction. No repeatability figure is published anywhere.**
   - Robocurve's measured rig facts: "free-space poses sag 1–3 cm below the commanded height".
   - AGP needed an outer integral loop on top of PD to reach **1.6–3 mm** (left arm) and **4–7 mm** (right arm) Cartesian settle.
   - Factory encoder zeros were off by up to **5.47°**. Calibration took the left arm from 8.32 mm to 1.36 mm RMS.
   - I2RT publishes no repeatability number for any YAM tier.
3. **Impedance is exposed, but Kd is capped.**
   - `MotorChainRobot.command_joint_state()` accepts per-command `kp`/`kd`. The MIT wire format limits Kp to [0, 500] and Kd to [0, 5], clamping silently. Stock J1–J3 already sit at Kd = 5, so a stiffer shoulder becomes under-damped.
   - Pure torque control = MIT with Kp = Kd = 0 plus τ_ff.
   - Effort is current-estimated, not a torque sensor. With 0.3 N·m Coulomb friction on J1–J3, the useful contact signal is ≥1 N, not the 14 mN·m quantum.
4. **Thermal limits bind in ordinary holds.**
   - Holding a top-down pose loads the shoulder (J2, a DM4340 rated 9 N·m) with **9.5–13.7 N·m beyond about 0.3 m radius [derived]**.
   - Tier 1 had **14/360 `overheat` terminations**: Opus 5 10/120, Opus 5.5 2/120, Astra 2/120.
5. **≤2 mm-clearance insertion on a YAM has already been done, without learning.** AGP (arXiv 2609.12541): an LLM agent writing scripted programs, wrist D405 depth, a 0.03 m/s cap and integral settle correction.
   - Four 3D-printed AutoMate pairs, radial clearance 0.73–2.83 mm: **8/10** (GPT-6 Astra high). Two-pair task: Opus 5 **5/5**.
   - ENPIRE reached 50 consecutive pin insertions on YAM with RL from one wrist camera and ≤0.3 mm steps, but allowed ≤8 retries.
   - **For M3:** the 2/20 Robocurve puzzle baseline is a strawman. The bar is a scripted wrist-camera servo with compliance.
6. **Data.** BC alone at 50–200 demos is mixed: ACT battery slot 96% (50 demos); ACT velcro 20% (100); ManiSkill peg (3 mm clearance, 100 demos) state DP 38%, state ACT 14%, every RGB policy 0%. **"100–200 demos per skill" is plausible only with wrist-camera relative alignment plus compliance, probably plus online correction (DAgger or RL).**

## 1. YAM characterization from primary sources

| Item | Value | Source |
|---|---|---|
| Price, payload, reach | $2,999 standard, $3,499 Pro, $4,299 Ultra; 2 kg nominal payload; 750 mm "max workspace range"; arm 5.02 kg with gripper; 95 mm gripper throw | i2rt.com product page; doc.i2rt.com/products/yam |
| Motors | J1–J3 DM4340 (40:1, rated 9 N·m, peak 27 N·m); J4–J6 DM4310 (10:1, rated 3 N·m, peak 7 N·m); gripper DM4310 | `i2rt/robots/config/yam_v1.yml`; Damiao spec pages |
| Encoders and wire resolution | Two 14-bit single-turn magnetic encoders per motor. CAN carries position as 16-bit over ±12.5 rad, i.e. 0.38 mrad (0.23 mm at 0.6 m) **[derived]**. Torque is 12-bit over ±28 N·m (DM4340, 13.7 mN·m) or ±10 N·m (DM4310, 4.9 mN·m) | `i2rt/motor_drivers/utils.py` `MotorConstants`; Damiao pages |
| Backlash | **Not published** by I2RT or Damiao (UNVERIFIED). The Pro tier claims "tighter build tolerances for improved repeatability" with no number | doc.i2rt.com/products/yam-pro |
| Control | MIT mode only (`set_control` encodes MIT or VEL). Host loop `CONTROL_FREQ = 250` Hz. KP 0–500 and KD 0–5 are encoded in 12 bits, and `float_to_uint` clips out-of-range values. Gravity comp uses `MuJoCoKDL.compute_inverse_dynamics` × per-joint `gravity_comp_factor` [1.0, 1.1, 1.1, 1.2, 1.0, 1.0] (hand-tuned model mismatch) | `dm_driver.py:33, 258–275`; `utils.py:39–42`; `motor_chain_robot.py:354–396, 507–523` |
| Variable impedance | `command_joint_state({"pos","vel","kp","kd"})` sets gains per command. `get_yam_robot(ee_mass=…, ee_inertia=…)` adds payload to gravity comp | `motor_chain_robot.py:578`; doc.i2rt.com API table |
| CAN budget | 7 motors × 2 frames × 130 bit × 250 Hz = 455 kbit/s against the driver's own 909 kbit/s ceiling (1 Mbit/s ÷ 1.1). Roughly 500 Hz is the ceiling per bus **[derived from the `dm_driver.py` bandwidth check]** | `dm_driver.py:455–463` |
| Gripper force | `GripperForceLimiter`: "clogged" when mean effort > 0.5 N·m and speed < 0.3, then a position offset = torque/Kp; `limit_gripper_force=50.0` N ≙ 0.73 N·m (0.096 m / 6.57 rad). Force-*limited* position control, not force control. AGP: contact ≥ 0.2 N·m for 0.1 s, squeeze 0.04 stroke, max 1.2 N·m | `get_robot.py:318`; `linear_4310.yml`; AGP `left_arm.yaml:131–140` |
| Effort as a sensor | Tier 1 rig facts: "Gripper effort spikes to ≈ 0.7–1.2 on every full close, empty or not… Effort cannot tell an empty close from a grasp." For the arm: "on a light touch the shoulder stops tracking while its effort stays negative… effort flipping positive means the arm is pressing" | Tier 1 system prompt, trial `omen-2/rig-3/adhoc_9d99ae23` |
| Thermal | Driver MOSFET trip 120 °C; coil limit user-set, "recommended not to exceed 100 °C" (one DM4340: `OT_Value` 100 RAM / 115 flash); errors 0xB/0xC. `inspect-robots-yam`: `motor_temp_limit` (off by default, wizard suggests 70 °C) on max(T_MOS, T_rotor), two reads, park, `"overheat"`. Metal: 70 °C | Damiao pages; `dm_motor_registers.md`; `embodiment.py:311, 1171–1173, 1694–1750` |
| Timeout | Factory default 400 ms; the motor drops to damping on timeout | i2rt README |

**Sim twins assume perfect gravity compensation** (Manda: "gravity compensated with Ai2's PD gains", `docs/bimanual_yam.md`), and I2RT's MJCF sets `actuatorfrcrange="-10 10"` on every joint. Sag and heat must be added explicitly.

## 2. Accuracy, sag and settle: measured vs derived

**Measured:**

| Source | Measurement |
|---|---|
| `inspect-robots-yam` `hold_check.py:92–97` | "settle is mode-independent (~0.012–0.015 rad on YAM joint 3)". Whether this is 0- or 1-indexed is UNVERIFIED: ×0.25 m gives 3–4 mm if it is the wrist, ×0.5 m gives 6–7.5 mm if it is the elbow |
| Tier 1 rig facts | "Free-space poses sag 1–3 cm below the commanded height, more at long reach and with a load"; "An arm left uncommanded drifts slowly downward over a minute"; "Pressing 1–2 cm 'below' the table is safe" (i.e. compliant) |
| AGP `left_arm.yaml:117` / `right_arm.yaml:136` | With an integral settle bias (`cartesian_settle_gain_s_inv: 4.0`, `max_bias_deg: 2.0`), the "left arm settles at 1.6–3 mm" and the "right arm settles 4–7 mm from Cartesian targets". A tighter gate "ends a third of the moves in SETTLE_MISS" |
| AGP i2rt patch (`yam_v1.yml` `motor_offsets_deg_by_channel`) | Right follower J4 zero is off by 5.47° (held-out error 4.68 mm / 0.55°). An 11-pose fit took the left arm from 8.32 mm/1.156° to 1.36 mm/0.257° RMS |
| AGP RIG_NOTES (towel) | "SETTLE_MISS … 3–10 mm"; extended pose: "a requested 30 mm lift achieved only about 10 mm" |
| Metal arm (same Damiao MIT protocol) | "shoulder_lift needed 5° [lead] to hold at long reach"; configured Kd 11/6.5 encoded as 5; fix = integral trim (≤0.75°/s, \|lead\| ≤ 2°) |
| YAM-UMI (community) | Tracker vs FK in teleop 7.7 mm median, mostly a 4.8 mm hand-eye residual (not repeatability) |

**Derived from I2RT's own MJCF** (MuJoCo 3.14; ~2,100 top-down grasp poses at z 0–0.20 m; sag = J·(Δτ/Kp)):

| Radius | J2 static τ_g, median (max) | J3 τ_g | Sag from 10% gravity-model error | Sag from an unmodelled 0.5 kg / 1 kg payload | J2 τ with 0.5 kg |
|---|---|---|---|---|---|
| 0.15–0.30 m | 6.5 (9.2) N·m | 1.6 | 1.7 mm | 3.2 / 6.3 mm | 7.4 N·m |
| 0.30–0.40 m | 9.5 (11.4) | 3.1 | 4.4 mm | 8.0 / 16.0 mm | 11.1 |
| 0.40–0.50 m | 12.1 (13.8) | 4.5 | 7.7 mm | 14.2 / 28.4 mm | 14.2 |
| 0.50–0.62 m | 13.7 (14.3) | 5.4 | 10.3 mm | 19.5 / 38.9 mm | 16.2 |

Translational stiffness at the grasp point (r 0.25–0.5 m, top-down) **[derived]**:

| Gains | Softest direction (p10–p90) | Stiffest direction |
|---|---|---|
| Default | 0.34 (0.24–0.47) N/mm | 2.6 N/mm |
| Pro (J4 Kp = 40) | 0.34 N/mm | 3.2 N/mm |
| Kp 200/200/200/40/40/40 | 1.0 N/mm | 7.2 N/mm |

At the 0.3 N·m friction level, the effort-derived contact threshold is on the order of 1 N **[derived]**.

**Meaning.** The derived numbers agree with the measured 1–3 cm. **The YAM's error is model error over low stiffness, not encoder noise.** Integral correction + `ee_mass` + zero calibration should reach roughly 1.5–3 mm; sub-millimetre work must close on the wrist camera. Robocurve's "−0.2 to −0.35 rad pitch sag" is mostly harness compounding of a ~0.013 rad mechanical settle: measured-state seeding, the oscillation hold (>2 reversals in 6 ticks freezes 10 ticks, `kinematics.py:226–230`), the 0.2 rad/tick clamp and the 0.35 rad resync (`config.py:169–175`).

## 3. Thermal: why trips happen

- **Tier 1 run JSONs (360 trials).** The 14 overheats were spread over 5 of 6 rigs: rig-1 5, rig-3 5, rig-4 2, rig-5 1, rig-6 1.
  - Opus 5 ran back-to-back. Trial `adhoc_9d99ae23` tripped after only **109 s / 8 calls**, starting about 40 s after the previous 434 s trial on rig-3 ended.
  - So heat carries over between trials; trial length alone does not explain the trips.
- **Effort trace of that trial (34 observations):**
  - right-arm J2 averaged 8.8 N·m and peaked at **18.7 N·m**;
  - the *idle* left arm parked at "home" held J3 at 7–10 N·m (mean 7.5).
  - Both are at or above the DM4340's 9 N·m rated torque.
- **Implications:** visual-servo dwell at long reach is the worst case. Park the idle arm folded, not at a hover "home"; keep fixtures inside r ≈ 0.35 m; log T_rotor/T_MOS and cool down between trials. Opus 5's 10/120 vs 2/120 is consistent with longer holds per trial (causal link UNVERIFIED).

## 4. Published ≤3 mm insertion with small policies and/or low-cost arms

| System | Arm | Task (clearance) | Policy / data | Sensing, compliance | Result (n) |
|---|---|---|---|---|---|
| ACT, 2304.13705 | ViperX 300 ("accuracy of 5–8mm"), Dynamixel PID | Slot Battery (no mm); Thread Velcro (3 × 25 mm loop, 2 mm tie); sim Bimanual Insertion (~5 mm) | ACT ~80M params, chunk 100, 50 Hz; 50 demos (Velcro 100) | RGB + joints, no F/T | Battery insert **96%** (25 evals); Velcro insert 20%; sim insert 32% scripted / 20% human (3 seeds × 50) |
| Diffusion Policy, 2303.04137 | sim only for insertion (real tasks on UR5/Franka) | Robomimic Square, ToolHang ("HiPrec", no mm) | DP; 200 ph demos | image | Square-ph 0.98 max / 0.92 avg; ToolHang 0.95 / 0.73 (3 seeds × 50 inits, last 10 ckpts) |
| ManiSkill3 via FAEA, 2601.20334 | sim | PegInsertionSide (3 mm, half-depth) | Opus 4.5 agent, 0 demos; DP/ACT/BC on 100 demos | state vs RGB | FAEA **0%** (5 seeds); DP-state 38%, ACT-state 14%, every RGB policy 0% |
| RL Token, 2604.23073 | in-house bimanual (hardware unstated), 50 Hz | M3 screw ("sub-millimeter alignment"), Ethernet, charger | VLA + RL head (MLP 2×256 / 3×512); 1–10 h demos + 400–1000 RL episodes (15 min–5 h) | 2 wrist + 1 base RGB | Screw critical phase 20% → **65%** (50 episodes); full task +40 pp |
| ENPIRE, 2606.19980 | 8 × bimanual YAM; PD + gravity comp; policy 30 Hz, joints 100 Hz | Pin into hole ("tight 4mm clearance" vs "4mm-diameter holes"; inconsistent) | RLPD-style RL with BC regularization; frozen ResNet-10; **right wrist image only** + z/rot6d/grip; Δ-EEF ≤ 0.3/0.3/0.6 mm per step (`pin_insertion_delta_eef.yaml`); demo count not stated | torque estimates in reward; torque-limited gripper | 50 consecutive successes **with ≤ 8 retries**; >1.5 h with 1 agent vs ~40 min with 8 |
| AGP, 2609.12541 | YAM; Mink IK; 0.03 m/s; integral settle | AutoMate printed pairs, radial 0.73 / 0.78 / 1.19 / 2.83 mm | LLM writes programs; one human demo video; no learning | wrist D405 RGB-D + 1080p overhead; default PD | Four-pair **8/10** (Astra high, mean 37.2 min). Two-pair (hex/post 1.19 mm + sleeve 2.83 mm radial, IDs inferred): Opus 5 5/5 (22.2 min), Fable 5.1 3/5, Astra 5/5 at each effort. Single-insert 8/8 (`RELEASED_EXPERIMENTS.md` §5) |
| Robocurve, Inspect Robots | YAM | Puzzle disc into groove (clearance not stated) | LLM direct EEF at 10 Hz, 224 px | 3 × 224 px RGB | Astra 2/20, Fable 5.1 2/20, Fable 5 0/20; reached the groove 8/20 and 9/20 |
| GPT-Policy, 2609.19138 | ARX X5 | Plug reinsertion (not stated) | Astra with video + action context, 1 demo | top + 2 wrist | 2/3 (0/3 without context) |
| RoboICL, 2609.34261 | Franka FR3 (not low-cost) | Peg-in-hole, 1 mm nominal | Astra, 3 shots | wrist D405 + external | 2/5; 3/5 "stall during insertion as successive RGB observations change little after contact" |
| piper-astra-jev | AgileX PiPER ($1,999) | SlideOver, ~2 mm per side | Scripted; SAM 3 + CAD-silhouette fit aligned to ≤ 0.8 mm; descend 5 mm/s, stop on lag > 3 mm | fixed D435i only | 1/1 (n = 1) |
| DexAgent | YamBox + Sharpa hands | Battery insertion | π0.5 fine-tuned on 500 sim episodes | 3 views | 6/10 |

**Patterns:** (1) every cheap-arm success closes the last millimetres with **relative** visual alignment (wrist camera or CAD fit), slow descents and passive compliance, never F/T; (2) RGB BC at 100 demos fails on ManiSkill peg, and RL on top of BC closes the gap to ~100% (ENPIRE, RLT); (3) LLM-only closed loops stall at contact (RoboICL, FAEA, Robocurve), while LLM-*authored* scripts with depth and compliance do not (AGP).

## 5. Implications for M3 / R4

- **Feasibility: yes for ≥1 mm radial clearance with chamfers.** It is unproven for 0.5 mm diametral clearance, and no source shows unchamfered, ≤0.25 mm-radial insertion on a YAM.
- **Re-baseline M3.** Replace "≥10/20 vs 2/20" with "beats a scripted wrist-servo + compliance baseline on the same rig". AGP-style scripts already reach 5/5–8/10.
- **Design order:** (1) calibrate encoder zeros; (2) integral settle + `ee_mass`; (3) contact detection from tracking lag + effort; (4) relative wrist-camera servo; (5) only then a learned head for residual alignment and seating. Feed the head Cartesian lag (commanded − FK(measured)), J2/J3 effort deltas and gripper effort.
- **Data plan:** 50–100 teleop demos of the *final 3 cm only*, starting from the scripted servo's hand-off pose; then on-robot correction (DAgger, or ENPIRE-style RLPD with BC regularization) with ≤0.3 mm/step caps; budget 1–5 h robot time per skill (RLT: 15 min–5 h).
- **Thermal policy:** r ≤ 0.35 m fixtures, a duty-cycle rule from §6 step 6, cool-downs between trials.

## 6. One-day bench protocol (per arm)

Tools: 0.01 mm dial indicator; plate with 3 conical seats; 0.25/0.5/1.0 kg weights; 0.1 N force gauge; AprilTag board; wrist D405; Ø10.0 mm pegs with Ø12.0 (2 mm diametral) and Ø11.0 mm holes, chamfered 0.5 × 45° and plain. Log q, q̇, effort, T_MOS, T_rotor and commands at 250 Hz.

| # | Time | Procedure | Report / decision |
|---|---|---|---|
| 0 | 0:30 | `dm_motor_registers.py read-all` per motor (OT_Value, KT_Value, PMAX/VMAX/TMAX, CTRL_MODE, Gr). Check achieved loop Hz. Run `inspect-robots-yam-holdcheck <ch> --zero-gravity {false,true}` (60 s). Construct with `temp_record_flag=True` | Register table; loop rate ≥ 245 Hz; settle/trend in rad per joint |
| 1 | 1:30 | **Repeatability:** 50 unidirectional returns to one pose (r 0.35 m, z 0.05 m), indicator on a ground stylus. Then ISO 9283-style 5 poses × 30 cycles, with bidirectional approaches for reversal (backlash) error. Run at default gains, then with an AGP-style integral settle | RP = mean + 3σ (mm); reversal error per axis; encoder-reported vs indicator |
| 2 | 1:30 | **Static sag map:** 3 radii (0.25/0.40/0.55) × 3 heights (0.02/0.10/0.20) × payload 0/0.25/0.5/1.0 kg. Command, wait 3 s, measure true TCP (indicator or tag), commanded pose, and FK(q_measured). Fit `gravity_comp_factor`/`ee_mass` and repeat | Sag (mm) vs §2 prediction; residual after fit (target ≤ 3 mm open-loop, ≤ 1 mm with integral) |
| 3 | 1:00 | **Kinematic zero calibration:** 11 board touches with AGP's `calib/fit_joint_offsets.py`; leave-one-out validation | RMS mm/° (AGP left arm: 1.36 mm / 0.257°) |
| 4 | 1:00 | **Contact detection:** 30 descents each at 5/10/20 mm/s onto the force gauge at 2 radii, plus 30 free-space descents. Detector = baseline-subtracted J2/J3 effort OR Cartesian lag > θ for 3 ticks | Force at detection (N) and latency (ms) at 95% detection with ≤ 1/30 false triggers |
| 5 | 1:00 | **Impedance:** push at the grasp point with 1/2/5 N in x/y/z under Kp ∈ {default, 2×, 4×} on J1–J3 (Kd = 5), and J4–J6 ∈ {10, 40, 80}. Step response. Then a host-side Cartesian impedance (MIT Kp = Kd = 0, τ = Jᵀ(KΔx − Dẋ) + g at 250 Hz) | Measured N/mm vs 0.34/2.6 N/mm predicted; maximum stable K (overshoot < 20%, no limit cycle) |
| 6 | 0:45 | **Thermal:** hold top-down at r 0.50 m with 0.5 kg until T reaches 65 °C, then park folded and log the cool-down | Time to 65/70 °C; cooling time constant; duty-cycle rule |
| 7 | 2:00 | **Scripted baseline insertion, 20 trials:** Ø10 into Ø12 chamfered (1 mm radial); hole randomized ±3 cm; peg grasped from a fixture. Steps: wrist-camera circle/tag fit; servo at ≤ 10 Hz to ≤ 0.5 mm image error at 3 cm standoff; descend at 5 mm/s; stop on the step-4 detector; ±1 mm spiral (0.3 mm pitch); release. Success = ≥ 10 mm deep and standing after withdrawal. If time allows, 10 trials on Ø11 unchamfered | Success/20 with Wilson CI; time per insertion; peak effort; failure classes. **This is the bar the learned head must beat** |

## Sources

- I2RT driver, MIT licence, `i2rt-robotics/i2rt` @ `120c3c8` (v1.3.6, 2026-09-17): `README.md`, `i2rt/robots/config/{yam_v1,yam_pro_v1,linear_4310}.yml`, `i2rt/motor_drivers/{dm_driver,utils}.py`, `i2rt/robots/{motor_chain_robot,get_robot,utils}.py`, `i2rt/motor_config_tool/dm_motor_registers.md`, `i2rt/robot_models/arm/yam/v1/yam.xml`, `i2rt/robot_models/gripper/linear_4310/linear_4310.xml`. https://github.com/i2rt-robotics/i2rt
- https://doc.i2rt.com/products/yam ; /yam-standard ; /yam-pro ; /yam-ultra ; /yam-box ; /yam-cell ; https://i2rt.com/products/yam-6-dof-arm
- Damiao: https://damiao.enactic.ai/en/products/hardware/dm-j4310-2ec-v1.1/ ; https://damiao.enactic.ai/en/products/hardware/dm-j4340-2ec-v1.0/
- `robocurve/inspect-robots-yam` @ `178c930`: `kinematics.py`, `config.py`, `hold_check.py`, `embodiment.py`, `docs/bimanual_yam.md` (Manda fork: `repos/manda-RoboLab-Verified`)
- Robocurve Tier 1: https://robocurve.org/opus-5-5-robodojo-rc-tier-1/ (`data/runs/*.json`, 360 trials); transcript https://robodojo-tier1-artifacts.pages.dev/log/omen-2/rig-3/adhoc_9d99ae23/ ; note `sources/robocurve-gpt6-astra.md`
- Agent as Policy: https://arxiv.org/abs/2609.12541 (Tables 1, 2, 5); repo `agent-as-policy` @ `c6875d0`: `hardware-bridge/config/{left,right}_arm.yaml`, `hardware-bridge/src/agp_yam_bridge/motion.py`, `third_party/i2rt/{UPSTREAM.md,tracked_diff.patch}`, `agp/README_interface_real.md`, `agp/paper_runs/RELEASED_EXPERIMENTS.md`, `agp/paper_runs/*/results.csv`, `calib/fit_joint_offsets.py`
- ENPIRE: https://arxiv.org/abs/2606.19980 ; `NVlabs/ENPIRE` @ `99ee90a`: `enpire/policy/pld/runtime/configs/experiment/pin_insertion_delta_eef.yaml`, `enpire/env/docs/REAL_WORLD_WORKFLOWS.md`
- ACT/ALOHA: https://arxiv.org/abs/2304.13705 ; Diffusion Policy: https://arxiv.org/abs/2303.04137 ; RL Token: https://arxiv.org/abs/2604.23073 ; FAEA: https://arxiv.org/html/2601.20334v2 ; ManiSkill2: https://arxiv.org/abs/2302.04659 ; RoboICL: https://arxiv.org/abs/2609.34261 ; GPT-Policy: https://arxiv.org/abs/2609.19138 (note `sources/gpt-policy-in-context.md`)
- Notes: `sources/piper-astra-jev.md`, `sources/metal-arm-harness.md`, `sources/dexagent.md`, `sources/manda-robotics.md`
- YAM-UMI: https://github.com/YosubShin/yam-umi
- Unreachable or absent: no I2RT repeatability or backlash spec exists on doc.i2rt.com or i2rt.com. The AGP and ENPIRE papers give no pin/hole chamfer. ENPIRE gives no demo count.
