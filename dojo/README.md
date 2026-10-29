# dojo: the RoboDojo v2 stack

Design: [../docs/design/robodojo-v2.md](../docs/design/robodojo-v2.md). Evidence: [../research/robodojo-teardown/](../research/robodojo-teardown/README.md).

| Module | Owner stream | What |
|---|---|---|
| `dojo/simlab/` | simlab | Dev-layout generator, headless RoboDojo runner, GT dumps (dev only), `Twin` API, determinism tests |
| `dojo/perception/` | perception | Asset catalogue (meshes, stable poses), inverse layout: RGB + proprio → layout record (placements relative to another label mapped through that base's pose and support frame; riders of a Dynamic base span its length), accuracy eval; scene backgrounds for fixtures that replace the table (`belt_bg`: the conveyor belt, `segment.with_scene`, keyed by the fixture's asset) or a scene with a scripted actor at home (`scripts/perception/scene_bg_frame0.py`: imitate's demonstrator arm + baskets, with `cam_head_noactor` for `watch.read_sources`' departure reads); the yaw of a round-outline object from a cut in its top (`top_feature`: insert_key's keyhole; B1's head yaw, `refine()`'s `yaw_source` / `yaw_read`); the in-hand pose of a held object (`held.held_pose`: own wrist + head + other wrist, render-and-compare, gripper-constrained, calibrated sigma; service op `held_pose`) and where to look at it (`show_pose`: observer looks, canonical show poses; service op `show_pose`), DEV data and eval (`held_data`, `held_eval`, `scripts/perception/held_*.sh`); identity by learned image embeddings (`embed_id`: a head crop against catalogue renders of every candidate, DINOv2 / SigLIP2 / CLIP-L; the belt read's per-blob pick and B1's identity polish behind `EstimatorConfig.embed_model`, default off; bake-off tools `scripts/perception/identity_bake*.py`, `belt_id_ab.py`, `identity_photo.py`); height above the table from the calibrated views (`stereo`: a plane sweep of the head against the wrists, left-right checked; `EstimatorConfig.stereo_check` compares every object's predicted surface with it, `stereo_reject="slot_bases"` drops a mismatched thin-child slot base; default off; dev eval `stereo_eval`, `scripts/perception/stereo_*.py`); learned multi-view depth through our own cameras (`mv3d`: VGGT-1B's per-view depth put through our K and FK poses, table-fitted, fused into the head; `stereo.check_heights` takes any height map; `EstimatorConfig.height_source="mv3d"` feeds the height check from it, `mv3d_propose` fits missing labels / height suspects at raised regions no object explains; both default off, a GPU server `python -m dojo.perception.mv3d serve` at `DOJO_MV3D_ADDR`; dev eval `mv3d_eval` (infer / eval / check / propose), `scripts/perception/mv/`); per-object doubt about B1's record without an LLM (`doubt`: the multi-view DINOv2 feature distance of each object at its pose, `featmetric`, plus the learned height check; `EstimatorConfig.b1_doubt` -> `_perception.b1_doubt`; consumers behind flags: the agentic gate's risk floor `AGENTIC_DOUBT_RISK`, the look macro's `RuntimeConfig.look_doubt`; all default off; `doubt.doubted(record)` for other streams); numbered series (one config label the generator expands into `<prefix>_<k>` objects: `EstimatorConfig.series_count`, scored by `eval_g1.series_labels`; `series_link_raise`, default off: a linked_count group's own masks, arrange's mats, raise its series' count, and a raised label is only looked for: never the occluded fallback, and with `glyph_read` only a member read as a digit); the grasp-time perception metric (`grasp_metric`: every grasp / place of an episode, the belief at that step vs GT
for the target, the place reference, the neighbours within R of the approach corridor and phantoms; the runtime's own
belief rebuilt from an integ report; GT from physics-only replays, `scripts/perception/m95_replay.py` /
`m95_score.py`; the frame-0 perfect-selector ceiling over every source, `m95_ceiling.py`); SAM mask proposals behind flags, default off (`sam_masks`: SAM 3.1 text / clicks + SAM 2.1 clicks from one GPU server, `sam_fg` / `sam_propose` / `series_count=sam`; bake-off scripts `scripts/perception/sam/`; `LocalSam`: the same models in process for offline evals); the grasp-time wrist check (`wrist_check`: at a pick / place hover, the target, a place's reference and every believed object within 10 cm of the approach, in the hover's wrist frame: existence from the hover refine + a SAM 2.1-S click, the head view's static-background foreground for what the wrist does not see, neighbour / reference fits, relocations by the refine's radius search, DINOv2 identity scored MUSE-style (CLS + GeM patch tokens) with a margin; service op `wrist_check`, `RuntimeConfig.wrist_check` off / log / apply / drop, default off; E1 offline on stream C's replayed sequences with stream M's metric: `scripts/perception/wrist/e1.py`, `e1_report.py`); the pose of an object right after its release (`release_read`: 6-DoF render-and-compare around the released pose in the releasing wrist, the head and the other wrist, `held_pose`'s machinery without the gripper; service op `release_read`, `RuntimeConfig.post_place_read` "6dof" when the post-place refine fails, default off; "6dof_search" adds `release_search` when the read finds nothing at the release: the pose the object had when the jaws closed (a missed grasp), then a head-view search of unexplained foreground near the release, service op `release_search`; E2 `scripts/perception/wrist/e2_release.py` + `e2_report.py`, the ceiling on M's replays `e2_oracle_m95.py`); A1, region-free re-detection at decision points (`redetect`: the twin segmenter on all cameras, content-hash cached, two score thresholds; z-buffered prediction; category-gated Hungarian association on mask IoU; "missing" promotions; arm-link contact sweep; region-free fit-in-mask over every resting pose; one joint assignment of masks over the active objects; footprint-overlap guard; corridor additions; lift identity; refuses GT / layout id / task in its input; fails without GL; service op `redetect`; the official offline scorer is stream C's harness calling `tick.run`, `scripts/perception/a1/x2.py` is a dev tool); background (continuous) perception over the frames an episode produces anyway (`bgperc`: per-object evidence since the robot last touched it, wrist refines of uncertain / required objects, optional head re-reads, held checks during a carry and contact suspects, moves need two agreeing views; service op `bg_step`, runtime `RuntimeConfig.bg_perception`, default off); per-step frame sequences of recorded episodes for that research (`seq_capture`: replay a run's actions in mode gt and keep every camera + state + dev GT; `seq_data`; offline replay and scoring with the grasp-time metric `scripts/perception/bgperc/`); an object pinned in a support slot read with its base at the pick hover (`slot_read`: base + child union silhouette against the wrist frame's own foreground with the base's table-coloured insert filled, a grid over the base's yaw period + polish, accepted on the hull of the nearby foreground and a yaw margin, refused on a border cut; service op `slot_read`, `RuntimeConfig.slot_read`, default off; offline `scripts/perception/coin_slot_eval.py`); a held_look "not held" from the jaw opening alone on an object thinner than the opening resolves (`j1_adapter.opening_resolution_gap`, 10.8 mm at a full close) is unknown with `RuntimeConfig.held_thin`, default off; a twin-trained instance segmenter whose class is the catalogue asset (`seg_twin`: labels, metrics, its GPU server; `EstimatorConfig.seg_propose`, default off: B1 fits inside its head-view masks, adds / moves / re-fits objects; training data from `dojo.simlab.seg_render`, tools `scripts/perception/seg/`); make_kong tile glyphs read by a twin-trained crop classifier (`glyph_read`, its GPU server; `EstimatorConfig.glyph_read`, default off: a read above the per-view margin sets a slot tile's asset, anything else stays "unknown"; training crops / classifier `scripts/perception/seg/kong_crops.py`, `kong_cls.py`; the same reader for any detector mask, `classify_mask`, with a digit / operator model for solve_equation / arrange_largest_number: number tiles by `category_idx % 10`, the five operator discs; no read on masks cut by the frame or bordered by an arm; a slot's read persists across the run's later records, `read_persist`: a later record without a read keeps it, an agreeing read refreshes it, a contradicting one makes it unknown, slot labels only since free labels are reassigned per estimate); digits shown on objects (`digits`: a number card's digit only from a glyph read, B1's identity confidence separate from its pose, `EstimatorConfig.glyph_min_margin`; the one rule that arbitrates B1's read against the agentic perceiver's explicit `value`; record / J1 / `ObjectState` digit fields, `digit_source` "unread" = unknown, and the planner reports the press count unknown) |
| `dojo/skills/` | skills, skills2, skills3 | WorldState, own-IK motion layer, grasps (top / side / rim / wall / box / slice; hold quality from the pad contact patch and the CoM lever, CoM stations), skills incl. re-orienting transfers (`plan_transfer`), relay chains and in-air handovers (`library.place_oriented`), a handover with the object kept as it rests for insertions no grasp from rest reaches, then a turn in the receptacle (`regrasp.py`: lengthwise giver pinch, the taker's grasp chosen at the target, a key turned through its lock's narrow zone; mode B: the wrist look, an upright turn about the hinge-free axis, a vertical-lag descent and a yaw sweep), topple, mesh peg-in-hole fit (`insertion.py`, + a sub-mm local fit of asset insert frames), a closed-loop insertion tail (`Skills._servo_tail`: sag-corrected, depth check, probing), mode B's held-pose protocol at the pre-insert hover (`Skills._held_protocol`: `provider.held_pose` read, observer look / show pose from `provider.show_pose` planned with our own IK, the servo on the adopted transform, a pressed spiral, thread-first seats, a guarded fallback no deeper than `blind_up`), a refined pick that keeps its paired grasp (`execute_pick`), place reach / wrist-limit filters, a generic skill library behind one planner entry point (`planner.solve`: instruction + WorldState -> skill), observation-driven skills that wait on watch reads (`interactive.py`: a scripted opponent's game, copying a demonstration, picks off a moving belt; `watch.py`: `provider.watch_read` / `actors_moving`, GT in G3), holders with holes and a hinged lid (`holders.py`: eggs carried lying as picked and let go off a hole's centre so they tip in, pushed off a basket wall first when no grasp fits; `--belief-offset LABEL:dx,dy,dyaw` in G3), objects dropped leading end down into a holder's opening and tools carried level into their slots (`library.fill_container` / `place_in_slots`), the centre of mass PhysX simulates per asset (`data/physics_com.json`, measured in the twin by `scripts/skills/physics_com.py`), an Articulation's RoboDojo pose (its articulation-root prim, the laptop's base link 77 mm off) <-> its asset (mesh) frame (`artroot.py`, `data/articulation_roots.json` from the USDs by `scripts/skills/articulation_roots.py`), Mikhail's measured skills, ported (`ported/`: robot-harness's skill files copied unchanged under `ported/skills/` with `SOURCE.txt`, run by `ported/harness_robot.HarnessRobot`, his robot-server verbs over our Skills and belief; ground truth only in G3 mode gt with a plain `GTProvider`, elsewhere the env's GT accessors raise `TruthDenied` and objects carry no labels; routes `planner.ported_plan` after ours, behind `RuntimeConfig.ported_routes` / G3 `--skill-attr ported_routes=on`, default off; `ported_swap` reach: swap_poses' carry height from both arms' reach, default fixed; `ported_drag` turn: planar_drag_p1, the turn direction by feasibility near 180 deg, default v1; every ported run is a `ported` event in the runtime report; `ported_pour` full: pour_p1, the whole roll profile, default v1; `ported_pickup` library: "pick up the X by N cm" with our `Skills.pick` first, then the ported pick_up on the same object, `pickup.py`, default v1), plan + verify (`TwinBackend`), G3 harness (`g3.py`, `scripts/skills/g3*.sh`, `deploy_skills3.sh`; mode-B hooks `--mode-b --belief-offset`, `--belief-drop runtime|fixtures` (a released object rests where the runtime belief's drop rule puts it unless the post-place re-measure sees it), `--look-read top_feature|axis` (a wrist look reads the receptacle as perception's keyhole read: residual 0.4 mm / 0.8 deg, `yaw_source` in `sk.last_refine`), `--held-pose noise|gt` / `--held-belief`: the runtime belief's in-hand transform and `held_pose` / `show_pose` answered from GT with perception-held's calibrated noise, `heldsim.py`, `data/held_noise.json`), kinematic dry runs (`scripts/skills/dryrun.py`); planner routes that are off by default (`routes.py`, `RuntimeConfig.new_routes`, G3 `--routes`): number / operator tiles onto a row of pads (`tiles.py`: an equation's missing tile, digits into the largest number), identical blocks slid into one aligned row without a lift (`align.py`) and objects stood in a row by size (`rows.py`), and bottles poured into bowls in the instruction's order through stream P's ported pour (`pour_order.py`, route-only) |
| `dojo/runtime/` | integration | The one proxy_eval policy (`RuntimePolicy`): perceive (+ look macro) → WorldState → supervisor (instruction-only) → skills with twin verify (bundle; the twin is the one for the instruction-derived scene, `dojo.bridge.twin.BundleTwins` in the dev drivers) → resync → finish; the belt tracker (`track_moving`, `RuntimeProvider._track`: B1 re-reads, a line fit per rider, newcomers under fresh labels, identity by the reads' vote, never-confirmed start readings dropped; a belt read the service marks `blind`, i.e. no scene background for its belt in the calibration set, is skipped and logged once as `belt_blind`; between riders the belt keeps its last measured velocity); firewall (episode info = instruction + step limit only); ONE frozen perception calibration set per service for every episode (`DOJO_CALIB_SET`, `dojo.perception.calib_set`: built from the calibration-only ids 100040-49 (D-C10b) with a manifest; the drivers refuse to evaluate them; `box_steps.sh calib_sets` installs it on every box); eval driver `scripts/dojo/integ_eval.sh` / `launch.sh` (modes gt / perception; every episode row has the dev-only `dev_gt_probe`: each label's max z rise over its reset pose and the GT object nearest the jaws at each close, never visible to the policy;
every report also has the dev-only `dev_belief_probe`: the provider's belief at every gripper change, the grasp-time
metric's belief source, `dojo.perception.grasp_metric`); the empty-jaw check (`RuntimeConfig.grip_check`, default off: RoboDojo's official gripper value, strict server v2's
`ee_joint_state_official` / dev `DOJO_GRIP_SOURCE=v2`, against the command after every settled close; an empty close
takes the object out of the jaws in the belief, a held one skips the vision held_look; `scripts/perception/h95_grip_offline.py`
re-judges saved runs on M's replays); consumer-tolerance events (`RuntimeConfig.consume_log`, on: logging only, the actions are identical either way; every
contact-sensitive skill action, i.e. a pick, place / insert / hang / thread, press, logs `{"kind": "consume", "step",
"skill", "label", "key", "pose7_used", "need": {"xy_mm", "z_mm", "yaw_deg"}, "purpose", "arm"}` from `Skills._consume`:
the belief pose it acts on, world / asset frame, and the precision it needs from the skill's own tolerances or geometry;
M's metric scores the belief error against `need`); A1's tick (`tick.py`: `run()`, the one entry point online and offline; triggers after an empty close / a release / a lift / a contact and at pick / place hovers (`triggers_from_events`), seeds per trigger kind, `decide()`'s write rules; a far hover relocation re-hovers; `RuntimeProvider.a1_tick`, semantic object versions + `belief_write` events, `fresh()`; `RuntimeConfig.a1` off / log / apply, default off); named, versioned config presets (`presets.py`: `--preset submit_v1` on eval / integ_eval / the bridge driver, `--config` over it, the name in every report) |
| `dojo/agent/` | agent | The hybrid LLM agent: manual (`prompts/manual_vN.md`; v6, the default: generic ending / banking rules with no checker numbers, judging phrases or per-task recipes, skill docs, `robo` API), scene with neutral ids (`scene.py`, never checker labels; v4, the default: every named frame a skill argument can name, numbered series with their positions), bounded persistent cell session, `robo` API over Skills / library / perception, cached append-only transcript loop, `AgentPolicy` on RuntimePolicy's `program=` seam (`--policy dojo.agent.policy:AgentPolicy`, agent config in `--config '{"extra":{"agent":{...}}}'`); Phase 1 grounding / routing benchmark (`dataset.py` [`--scene-version 4 --planner-meshes`], `gold.py`: one rule per config from the checker, `grounding.py`), e2e summary (`e2e_report.py`); a replay probe (`replay_probe.py`: a logged episode rebuilt up to one message, the next cell sampled per prompt variant, graded against a stub `robo`); the J2 skill registry, ilia side (`skills/registry.yaml`, schema `j2-registry/1` = robot-harness's `skills/registry.yaml` / `docs/REGISTRY.md`; `registry.py` loads either file unchanged, derives `contract_only` from `obs_needs.reads`, strict runs keep contract-only entries, `status: candidate` entries (unpromoted versions) reach no prompt unless `select(candidates=True)`; `AgentConfig.registry` builds the manual's skill list from it) |
| `dojo/bridge/` | bridge | Our full system behind robot-harness's strict server, i.e. behind the OFFICIAL RoboDojo eval client: robot_client_v2 client, strict reply firewall (`contract.py`: whitelist to the contract observation, banned keys fail closed, strict tier required), `RobotServerBackend` (observe + the skills Env protocol; per-step joint actions through `joint_steps`, robot-harness #8), perception routes (local service / the server's `perceive`), twin service for the instruction's scene, episode driver (`python -m dojo.bridge.driver`), fake strict server; [../docs/BRIDGE-RUNBOOK.md](../docs/BRIDGE-RUNBOOK.md) |
| `baselines/pyrua/` | pyrua | PyRUA-Lean reproduction on our infra (the floor) |

Hard rule (the only one): at eval time the policy uses only the official observation contract. Never read
`Assets/Eval_Layout/**` or eval-env state for development. Dev work uses our own generated layouts, where ground
truth is ours. A task config's identity relations between labels (`same_index_as_label`: which unread objects match)
are never policy input; its candidate pools are. An identity no image read stays unknown:
`_perception.identity` "unknown" -> `ObjectState.identity_known` False (`dojo/perception/identity_marks.py`,
`tests/dojo/test_legality_kong.py`).

Shared machine conventions:
- compute2 (4090 24 GB): RoboDojo install at `/root/controlr-robodojo` (read-only, shared); data at
  `/root/dojo-data/<stream>/`; code deploys at `/root/controlr-<stream>`.
- compute3 (5090 32 GB, shared with others): `~/dojo-*`.
- Before starting Isaac or a large model, check free VRAM with `nvidia-smi`; never touch other people's processes;
  stop what you started.
- GPU locks: ONE heavy GPU job per machine holds `~/dojo-gpu.lock` (Isaac with rendering, a bundle, a model;
  ≤ 30 min per hold, 90 s yield). Up to four physics-only light jobs hold one of `~/dojo-gpu-light{,2,3,4}.lock`
  each (g3's light lane: `DOJO_GPU_LOCKS="$DOJO_GPU_LIGHT_LOCKS"`).
- Scopes: big jobs run in `systemd-run --user --scope -q -p MemoryMax=... -p CPUQuota=...`: bundle 24G / 1400%,
  proxy 16G / 1200%, batch 20G / 1200%, g3 10G / 600%, light 8G / 600% (`$DOJO_SCOPE_*` in `~/dojo-env.sh`); start
  only with ≥ 25 GB MemAvailable.
- New GPU boxes: `scripts/bootstrap_gpu_box.sh HOST` (one) or `scripts/bootstrap_fleet.sh HOST...` (several,
  peer-to-peer) sets all of this up: compute3's layout, the locks, `~/dojo-env.sh`, and `~/dojo-box.json` (role
  `full` = 5.1 renders, official dev_eval + bridge; `physics` = 5.1 physics-only + Isaac 6.0 render for proxy_eval;
  capabilities from the smokes). Pick boxes by `~/dojo-box.json`; see
  [../docs/RUNBOOK-new-gpu-box.md](../docs/RUNBOOK-new-gpu-box.md).

## simlab quick reference

Run everything through `scripts/simlab/rdpy.sh` (RoboDojo's conda python + the team GPU lock). Use
`physics_only=True` / `--physics-only` on compute3, where Isaac Sim 5.1's RTX does not start.

**Asset firewall.** `Assets/Eval_Layout/` and every `Assets/Traj/` file except the generic make_kong /
play_tic_tac_toe demos are eval data.
- imitate_sorting_sequence's `Traj/<bench>/<task>/<eval_seed>/<seed>.pkl` files are paired with the eval
  layouts, and their `target_label` is the answer.
- Every `RoboDojoRunner` process (and `dev_eval`) installs `paths.install_asset_firewall()`. That is a
  permanent audit hook: open, list, glob or copy of those paths raises `PermissionError`, whoever calls.
- The imitate demo is routed to our own `~/dojo-data/simlab/traj/<task>/<seed>.pkl` (`demo_gen`), or a
  no-demo stub.
- `tests/dojo/test_simlab_firewall.py` fails on any dojo/ code that builds such a path.

| What | Entry point |
|---|---|
| layout schema, perturbation | `dojo/simlab/layout.py` |
| new dev layouts (seeds >= 100000) | `-m dojo.simlab.gen_layouts --task T --n 25` |
| pre-settled records `<id>.settled.json` (prefer these) | `scripts/simlab/presettle_all.sh [task ...]` (`-m dojo.simlab.presettle --task T`); dropped ids whose objects never moved in pass 1 (insert_key's frictionless key): `--keep-raw-unmoved` keeps their generated pose, never rewrites a settled record, marks `_simlab.presettle` |
| load layouts, official obs vs dev-only GT | `dojo.simlab.runner.RoboDojoRunner` |
| twin: load / snapshot / restore / step / verify / load_from_estimate | `dojo.simlab.twin.Twin` |
| scripted motion, replayable action lists | `dojo/simlab/motion.py` |
| D0 dataset | `scripts/simlab/d0_all.sh` (compute2), `scripts/simlab/d0_proxy.sh` (compute3: Isaac 6.0 renders) |
| D0-style frames 0-1 from pre-settled records, more tasks | `scripts/simlab/d0_ext.sh N task ...` (-> `~/dojo-data/d0_ext`) |
| labelled frames for the segmenter (new variant draws of 100030-49, Isaac instance masks, arm poses) | `-m dojo.simlab.seg_render --task T` (`scripts/perception/seg/render_queue.sh`; `--physics-only` + `render_d0_proxy.py --any-dirs --instance` for 6.0) |
| support-arm demos for imitate_sorting_sequence dev layouts | `-m dojo.simlab.demo_gen [--ids ...]` (-> `~/dojo-data/simlab/traj/`) |
| generate + pre-settle many tasks in order | `scripts/simlab/coverage.sh N task ...` |
| G2 determinism / sensitivity | `scripts/simlab/g2_all.sh OUT [--physics-only]` |
| twin throughput | `scripts/simlab/twin_bench.sh` |
| RoboDojo's eval client on dev layouts (official, needs 5.1 rendering) | `scripts/simlab/dev_eval.sh --task T --policy P` |
| PROXY dev-eval on compute3 (5.1 physics + 6.0 render on demand) | `scripts/simlab/proxy_eval.sh --task T --layouts ... --policy module:Class [--bundle]` |
| twin as a service for a policy (inside `--bundle`) | `dojo.simlab.twin_service.TwinClient` |
| minimal RoboDojo on another GPU | `scripts/simlab/install_min.sh` |

### Proxy dev-eval with your own policy (compute3)

```python
# mypkg/mypolicy.py, importable from the repo root (scripts add it to PYTHONPATH)
class MyPolicy:
    def reset(self, info):        # {"task", "layout_id", "instruction", "step_lim", ...}
        ...
    def next(self, obs):          # obs: images {cam_head, cam_left_wrist, cam_right_wrist: 480x640x3 uint8},
        ...                       #      state (official joints / flange poses / grippers), instruction,
                                  #      steps_used, step_lim
        return {"actions": [{"left_arm_joint_state": [6], "left_ee_joint_state": [g],
                             "right_arm_joint_state": [6], "right_ee_joint_state": [g]}, ...]}
        # or {"done": True}; one observation is rendered per next() call (~0.12 s)
```

```bash
ssh compute3 'cd ~/dojo-simlab && scripts/simlab/proxy_eval.sh --task stack_blocks \
    --layouts $(ls ~/dojo-data/simlab/layouts/stack_blocks/1000*.json | head -10) --policy mypkg.mypolicy:MyPolicy'
# -> ~/dojo-data/simlab/proxy_eval/<stamp>/_result.json (RoboDojo-style, marked "proxy")
```

Every episode goes through RoboDojo's own reset, `take_action` step accounting, `step_lim` and reward
latches; only the frames come from Isaac 6.0 (RoboDojo's stage, cameras and render settings). Episodes
are serial.

`--bundle [--twin-envs N]` adds a twin service (physics-only `Twin`, N <= 16, default 8) for the same task (a
`RuntimePolicy` whose instruction names another scene gets a twin for that scene from the driver's `BundleTwins`;
the policy never asks a twin which task it runs);
the whole bundle (episode env + render service + twin) is ONE GPU job: one lock, one scope (24 GB / 14 cores).
Inside the policy:

```python
from dojo.simlab.twin_service import TwinClient
tc = TwinClient()                                  # env vars set by the bundle
tc.load_from_estimate(layout_estimate, robot={"left_q": q_l, "right_q": q_r, "left_grip": 1.0, "right_grip": 1.0})
outs = tc.verify({"layout": layout_estimate, "robot": {...}}, [{"actions": seg, "meta": {...}}, ...],
                 perturb={"pos_mm": 2.0, "yaw_deg": 1.0}, k=4)    # outcome fields: see Twin.verify
```

`dojo.simlab.proxy_eval:VerifyReplayPolicy` (spec `verifyreplay:TRAJ.json[:k]`) is a dev demo of the
bundle. `info["layout_file"]` exists only in dev runs; a real policy passes its perception estimate.
Layout lists: prefer `<id>.settled.json` (objects at the poses RoboDojo's settle gives them).

Findings and numbers: [docs/journal/2026-10-24-dojo-simlab.md](../docs/journal/2026-10-24-dojo-simlab.md).
