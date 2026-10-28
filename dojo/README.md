# dojo: the RoboDojo v2 stack

Design: [../docs/design/robodojo-v2.md](../docs/design/robodojo-v2.md). Evidence: [../research/robodojo-teardown/](../research/robodojo-teardown/README.md).

| Module | Owner stream | What |
|---|---|---|
| `dojo/simlab/` | simlab | Dev-layout generator, headless RoboDojo runner, GT dumps (dev only), `Twin` API, determinism tests |
| `dojo/perception/` | perception | Asset catalogue (meshes, stable poses), inverse layout: RGB + proprio → layout record (placements relative to another label mapped through that base's pose and support frame; riders of a Dynamic base span its length), accuracy eval; scene backgrounds for fixtures that replace the table (`belt_bg`: the conveyor belt, `segment.with_scene`, keyed by the fixture's asset) or a scene with a scripted actor at home (`scripts/perception/scene_bg_frame0.py`: imitate's demonstrator arm + baskets, with `cam_head_noactor` for `watch.read_sources`' departure reads); the yaw of a round-outline object from a cut in its top (`top_feature`: insert_key's keyhole; B1's head yaw, `refine()`'s `yaw_source` / `yaw_read`); the in-hand pose of a held object (`held.held_pose`: own wrist + head + other wrist, render-and-compare, gripper-constrained, calibrated sigma; service op `held_pose`) and where to look at it (`show_pose`: observer looks, canonical show poses; service op `show_pose`), DEV data and eval (`held_data`, `held_eval`, `scripts/perception/held_*.sh`); identity by learned image embeddings (`embed_id`: a head crop against catalogue renders of every candidate, DINOv2 / SigLIP2 / CLIP-L; the belt read's per-blob pick and B1's identity polish behind `EstimatorConfig.embed_model`, default off; bake-off tools `scripts/perception/identity_bake*.py`, `belt_id_ab.py`, `identity_photo.py`); height above the table from the calibrated views (`stereo`: a plane sweep of the head against the wrists, left-right checked; `EstimatorConfig.stereo_check` compares every object's predicted surface with it, `stereo_reject="slot_bases"` drops a mismatched thin-child slot base; default off; dev eval `stereo_eval`, `scripts/perception/stereo_*.py`); learned multi-view depth through our own cameras (`mv3d`: VGGT-1B's per-view depth put through our K and FK poses, table-fitted, fused into the head; `stereo.check_heights` takes any height map; `EstimatorConfig.height_source="mv3d"` feeds the height check from it, `mv3d_propose` fits missing labels / height suspects at raised regions no object explains; both default off, a GPU server `python -m dojo.perception.mv3d serve` at `DOJO_MV3D_ADDR`; dev eval `mv3d_eval` (infer / eval / check / propose), `scripts/perception/mv/`); per-object doubt about B1's record without an LLM (`doubt`: the multi-view DINOv2 feature distance of each object at its pose, `featmetric`, plus the learned height check; `EstimatorConfig.b1_doubt` -> `_perception.b1_doubt`; consumers behind flags: the agentic gate's risk floor `AGENTIC_DOUBT_RISK`, the look macro's `RuntimeConfig.look_doubt`; all default off; `doubt.doubted(record)` for other streams); numbered series (one config label the generator expands into `<prefix>_<k>` objects: `EstimatorConfig.series_count`, scored by `eval_g1.series_labels`); the grasp-time perception metric (`grasp_metric`: every grasp / place of an episode, the belief at that step vs GT
for the target, the place reference, the neighbours within R of the approach corridor and phantoms; the runtime's own
belief rebuilt from an integ report; GT from physics-only replays, `scripts/perception/m95_replay.py` /
`m95_score.py`; the frame-0 perfect-selector ceiling over every source, `m95_ceiling.py`); SAM mask proposals behind flags, default off (`sam_masks`: SAM 3.1 text / clicks + SAM 2.1 clicks from one GPU server, `sam_fg` / `sam_propose` / `series_count=sam`; bake-off scripts `scripts/perception/sam/`; `LocalSam`: the same models in process for offline evals); the grasp-time wrist check (`wrist_check`: at a pick / place hover, the target, a place's reference and every believed object within 10 cm of the approach, in the hover's wrist frame: existence from the hover refine + a SAM 2.1-S click, the head view's static-background foreground for what the wrist does not see, neighbour / reference fits, relocations by the refine's radius search, DINOv2 identity scored MUSE-style (CLS + GeM patch tokens) with a margin; service op `wrist_check`, `RuntimeConfig.wrist_check` off / log / apply / drop, default off; E1 offline on stream C's replayed sequences with stream M's metric: `scripts/perception/wrist/e1.py`, `e1_report.py`); the pose of an object right after its release (`release_read`: 6-DoF render-and-compare around the released pose in the releasing wrist, the head and the other wrist, `held_pose`'s machinery without the gripper; service op `release_read`, `RuntimeConfig.post_place_read` "6dof" when the post-place refine fails, default off; E2 `scripts/perception/wrist/e2_release.py`, the ceiling on M's replays `e2_oracle_m95.py`) |
| `dojo/skills/` | skills, skills2, skills3 | WorldState, own-IK motion layer, grasps (top / side / rim / wall / box / slice; hold quality from the pad contact patch and the CoM lever, CoM stations), skills incl. re-orienting transfers (`plan_transfer`), relay chains and in-air handovers (`library.place_oriented`), a handover with the object kept as it rests for insertions no grasp from rest reaches, then a turn in the receptacle (`regrasp.py`: lengthwise giver pinch, the taker's grasp chosen at the target, a key turned through its lock's narrow zone; mode B: the wrist look, an upright turn about the hinge-free axis, a vertical-lag descent and a yaw sweep), topple, mesh peg-in-hole fit (`insertion.py`, + a sub-mm local fit of asset insert frames), a closed-loop insertion tail (`Skills._servo_tail`: sag-corrected, depth check, probing), mode B's held-pose protocol at the pre-insert hover (`Skills._held_protocol`: `provider.held_pose` read, observer look / show pose from `provider.show_pose` planned with our own IK, the servo on the adopted transform, a pressed spiral, thread-first seats, a guarded fallback no deeper than `blind_up`), a refined pick that keeps its paired grasp (`execute_pick`), place reach / wrist-limit filters, a generic skill library behind one planner entry point (`planner.solve`: instruction + WorldState -> skill), observation-driven skills that wait on watch reads (`interactive.py`: a scripted opponent's game, copying a demonstration, picks off a moving belt; `watch.py`: `provider.watch_read` / `actors_moving`, GT in G3), holders with holes and a hinged lid (`holders.py`: eggs carried lying as picked and let go off a hole's centre so they tip in, pushed off a basket wall first when no grasp fits; `--belief-offset LABEL:dx,dy,dyaw` in G3), objects dropped leading end down into a holder's opening and tools carried level into their slots (`library.fill_container` / `place_in_slots`), the centre of mass PhysX simulates per asset (`data/physics_com.json`, measured in the twin by `scripts/skills/physics_com.py`), an Articulation's RoboDojo pose (its articulation-root prim, the laptop's base link 77 mm off) <-> its asset (mesh) frame (`artroot.py`, `data/articulation_roots.json` from the USDs by `scripts/skills/articulation_roots.py`), plan + verify (`TwinBackend`), G3 harness (`g3.py`, `scripts/skills/g3*.sh`, `deploy_skills3.sh`; mode-B hooks `--mode-b --belief-offset`, `--belief-drop runtime|fixtures` (a released object rests where the runtime belief's drop rule puts it unless the post-place re-measure sees it), `--look-read top_feature|axis` (a wrist look reads the receptacle as perception's keyhole read: residual 0.4 mm / 0.8 deg, `yaw_source` in `sk.last_refine`), `--held-pose noise|gt` / `--held-belief`: the runtime belief's in-hand transform and `held_pose` / `show_pose` answered from GT with perception-held's calibrated noise, `heldsim.py`, `data/held_noise.json`), kinematic dry runs (`scripts/skills/dryrun.py`) |
| `dojo/runtime/` | integration | The one proxy_eval policy (`RuntimePolicy`): perceive (+ look macro) → WorldState → supervisor (instruction-only) → skills with twin verify (bundle) → resync → finish; the belt tracker (`track_moving`, `RuntimeProvider._track`: B1 re-reads, a line fit per rider, newcomers under fresh labels, identity by the reads' vote, never-confirmed start readings dropped; a belt read the service marks `blind`, i.e. no scene background for its belt in the fold, is skipped and logged once as `belt_blind`; between riders the belt keeps its last measured velocity); firewall; eval driver `scripts/dojo/integ_eval.sh` / `launch.sh` (modes gt / perception; every episode row has the dev-only `dev_gt_probe`: each label's max z rise over its reset pose and the GT object nearest the jaws at each close, never visible to the policy;
every report also has the dev-only `dev_belief_probe`: the provider's belief at every gripper change, the grasp-time
metric's belief source, `dojo.perception.grasp_metric`); the empty-jaw check (`RuntimeConfig.grip_check`, default off: RoboDojo's official gripper value, strict server v2's
`ee_joint_state_official` / dev `DOJO_GRIP_SOURCE=v2`, against the command after every settled close; an empty close
takes the object out of the jaws in the belief, a held one skips the vision held_look; `scripts/perception/h95_grip_offline.py`
re-judges saved runs on M's replays); named, versioned config presets (`presets.py`: `--preset submit_v1` on eval / integ_eval / the bridge driver, `--config` over it, the name in every report) |
| `dojo/agent/` | agent | The hybrid LLM agent: manual (checker conventions by instruction pattern, skill docs, `robo` API; `prompts/manual_vN.md`), scene with neutral ids (`scene.py`, never checker labels; v4, the default: every named frame a skill argument can name, numbered series with their positions), bounded persistent cell session, `robo` API over Skills / library / perception, cached append-only transcript loop, `AgentPolicy` on RuntimePolicy's `program=` seam (`--policy dojo.agent.policy:AgentPolicy`, agent config in `--config '{"extra":{"agent":{...}}}'`); Phase 1 grounding / routing benchmark (`dataset.py` [`--scene-version 4 --planner-meshes`], `gold.py`: one rule per config from the checker, `grounding.py`), e2e summary (`e2e_report.py`); a replay probe (`replay_probe.py`: a logged episode rebuilt up to one message, the next cell sampled per prompt variant, graded against a stub `robo`); the J2 skill registry, ilia side (`skills/registry.yaml`, schema `j2-registry/1` = robot-harness's `skills/registry.yaml` / `docs/REGISTRY.md`; `registry.py` loads either file unchanged, derives `contract_only` from `obs_needs.reads`, strict runs keep contract-only entries; `AgentConfig.registry` builds the manual's skill list from it) |
| `dojo/bridge/` | bridge | Our full system behind robot-harness's strict server, i.e. behind the OFFICIAL RoboDojo eval client: robot_client_v2 client, strict reply firewall (`contract.py`: whitelist to the contract observation, banned keys fail closed, strict tier required), `RobotServerBackend` (observe + the skills Env protocol; per-step joint actions through `joint_steps`, robot-harness #8), perception routes (local service / the server's `perceive`), twin service for the instruction's scene, episode driver (`python -m dojo.bridge.driver`), fake strict server; [../docs/BRIDGE-RUNBOOK.md](../docs/BRIDGE-RUNBOOK.md) |
| `baselines/pyrua/` | pyrua | PyRUA-Lean reproduction on our infra (the floor) |

Hard rule (the only one): at eval time the policy uses only the official observation contract. Never read
`Assets/Eval_Layout/**` or eval-env state for development. Dev work uses our own generated layouts, where ground
truth is ours.

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

`--bundle [--twin-envs N]` adds a twin service (physics-only `Twin`, N <= 16, default 8) for the same task;
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
