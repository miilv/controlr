# 2026-11-07: partner team's Codex legality audit of our sealed D-C19 system (controlr 127b65c4, CANDIDATE 5509d7be)

Received from the partner team on the coordination bus (`20261107T1723Z-misha-codex-audit-yours.md`), kept verbatim below for the record. Context: [general-system hard rule](2026-11-07-general-system-rule.md); BACKLOG item "task-keyed routing vs the RoboDojo single-system rule".

---- Codex report (verbatim) ----
# Read-only legality audit of `127b65c4`

Scope: commit `127b65c4` only, inspected with `git show`/`git grep`. I ignored all working-tree and later-commit changes. No network, tests, or writes were used.

One limitation: the supplied `--config` JSON is truncated after `"belt_wr...`. I audited every visible override and the `submit_v3` preset, but cannot certify fields missing from the truncated tail.

## Executive verdict

**Verdict: the sealed system at `127b65c4` is not defensibly a “single fixed system” under the quoted rule. It has multiple legality-blocking mechanisms.**

The fixed executable, checkpoints, calibration file, segmentation model, glyph readers, and global configuration are not enough. At inference time the system:

1. Converts the instruction into a benchmark task/vocabulary name.
2. Loads that task’s candidate assets, object counts, label schema, placement ranges, select modes, scene backgrounds, and sometimes fixed slots from `tasks.json`.
3. Selects named task-specialized skills and ported policies.
4. Selects policy variants that were explicitly tuned as the best implementation for particular benchmark task families.
5. Changes preprocessing and observation acquisition by recognized task semantics, including skipping scan/look/search for certain benchmark tasks.
6. Contains exact checker predicates, thresholds, staged scoring logic, and hidden task conventions in policy-side code.

Ordinary instruction understanding—e.g. parsing “stack the red block on the blue block” into a generic `pick` and `place` program—is permitted. This system goes materially further: the recognized instruction is effectively a key into benchmark-specific perception schemas and benchmark-specialized controllers. That is functionally a task router even though the launcher’s explicit task-name field is usually avoided.

The strongest blocking findings are:

- `instruction → vocabulary/task name → tasks.json task schema`;
- task-config object counts, candidate sets, placement distributions, fixed slots, and label prefixes used during scored inference;
- `ported_plan` selecting separately developed benchmark policies and task-specific tuned variants;
- checker-derived task rules and thresholds embedded in execution;
- task-dependent perception and pre-action procedures.

---

# 1. Instruction/task-dependent dispatch

## 1.1 Main planner selects distinct skills by instruction patterns

**Location:** `dojo/skills/planner.py:349-369`

> `p = _plan(instruction, world)`  
> `routes = ... world.notes ... ["routes"]`  
> `r = _route(text, world, routes)`

**Severity:** **Risky**, becoming **blocks legality** for the benchmark-specific routes described below.

A planner choosing generic capabilities from language is explicitly allowed. The concern is that many returned capabilities are complete benchmark solutions rather than general-purpose actions or reusable subgoals.

**Fix:** make the planner emit a small task-independent action graph—e.g. `observe`, `pick`, `place`, `push`, `insert`, `press`, `pour`, `wait`, `verify`—with arguments derived only from language and current observations. Remove named benchmark composites from the planner registry.

---

## 1.2 Built-in planner is a large instruction-template router

Representative branches in `dojo/skills/planner.py`:

| Lines | Quote / selected path | Classification |
|---|---|---|
| 378-394 | `if re.search(r"tic.?tac.?toe|\bplay\b", text)` → `play_board_game` | **Risky**: game semantics may justify a planner, but this is a dedicated benchmark controller. |
| 397-405 | conveyor + image → `pick_into_held` | **Risky** |
| 408-412 | conveyor + first/remember/match → `match_and_pick` | **Risky** |
| 416-427 | exact demonstration/order templates → `copy_demonstration` | **Risky** |
| 431-441 | lid/close → `fill_holes` plus lid action | **Fine/Risky boundary**: reasonable decomposition, but task-specific role assumptions are hard-coded. |
| 444-450 | `fold` → `fold_garment` | **Fine** as semantic skill selection, assuming the fold policy is generic. |
| 454-464 | `sweep` plus dustpan/broom recognition → `sweep_into` | **Fine/Risky** |
| 467-472 | `kong` → `declare_kong` | **Risky**: benchmark/game-specific solver. |
| 475-484 | strike/hit/play → `strike_in_order` | **Fine** |
| 487-502 | peg task → `stack_on_pegs`, pairing largest group with tallest peg | **Risky** if that pairing is not stated. |
| 505-522 | insert N items, then press lever/button | **Risky**: combines a benchmark recipe and inferred completion action. |
| 525-541 | hang → `hang_on_hooks` | **Fine** |
| 543-559 | ordered colour stack | **Fine** |
| 561-564 | multi-clause arrangements | **Fine** |
| 567-576 | matching slots → `place_in_slots` | **Fine/Risky** |
| 579-599 | screws/nuts paired by colour or proximity → `screw_on` | **Risky** where the pairing is not explicit. |
| 601-625 | insert/plug → `insert_each` or `insert_into`; `turn = -60` when “turn” appears | **Risky**: fixed turn direction/magnitude is not language-grounded. |
| 627-649 | number cards → `press_sequence` | **Risky**, discussed below. |
| 651-665 | swap → `swap_on_mats` | **Fine**, provided “press after each” comes from instruction. |
| 667-703 | tower/fill/put/stack | Mostly **Fine** generic instruction understanding. |

The use of observed geometry and language is legitimate. The problem is branches that encode benchmark-specific roles or procedures not entailed by the instruction.

---

## 1.3 `new_routes` enables named benchmark routes and asks them before generic planning

**Locations:**

- `dojo/skills/routes.py:1-30`
- `dojo/skills/planner.py:362-369`
- `dojo/runtime/policy.py:828-829`

> `NEW_ROUTES = ("equation", "largest_number", "align", "row_by_size", "pour_order", "pack", "store_laptop")`

> `provider.notes["routes"] = list(cfg.new_routes)`

The sealed config enables:

> `"new_routes":["equation","largest_number","align","row_by_size","pour_order"]`

**Severity:** **Blocks legality**.

These are explicitly named benchmark routes and are prioritized before the ordinary planner. Enabling all five globally does not make them general: the instruction determines which separately implemented benchmark policy runs.

**Fix:** replace these route names with generic capability registration. For example, equation solving should be a generic symbolic relation solver over perceived glyphs; alignment should use a generic spatial-constraint solver; row-by-size should be a generic sort-and-place operator.

---

## 1.4 Ported router chooses separately developed policies by instruction

**Location:** `dojo/skills/planner.py:271-346`

Examples:

> `cover` + `uncover` → `"cover_uncover_v2"`

> `swap` + `positions|poses|places` → `"swap_poses"`

> `pour` + `balls` → `"pour"` or `"pour_p1"`

> liquid/water/wine → `"pour_bottle"` or `"pour_bottle_lean"`

> sorting into baskets/bins → `"classify_into_bins_p1"`, `p2`, or `v3b`

> push/slide/drag + pad → `"planar_drag"` or `"planar_drag_p1"`

**Severity:** **Blocks legality**.

This is the clearest policy router. `dojo/skills/ported/__init__.py:1-7` says the code is a collection of measured RoboDojo skills copied from another harness, and that the planner chooses among them from the instruction.

The sealed configuration further chooses task-family-specific variants:

- `ported_swap="reach"`
- `ported_drag="turn"`
- `ported_pour="full"`
- `ported_pickup="library"`
- `ported_pour_aim="b10"`
- `ported_classify="p2"`

These options are global fields, but each only affects the recognized task family. In operation this is equivalent to a table of “use variant X on task family Y.”

**Fix:** expose generic primitives such as `pick`, `carry`, `release`, `push_contact`, `tilt_container`, and `sort_by_relation`. Use one shared search/optimization procedure to compose them. Remove `cover_uncover_v2`, `classify_into_bins_p2`, `pour_p1`, etc. as independently routed policies.

---

## 1.5 Ported policy table documents task-specific best variants

**Location:** `dojo/skills/ported/__init__.py:21-39`

> `TIER_A = { "cover_uncover_v2": ..., "swap_poses": ..., "pour": ..., "pour_bottle": ..., "planar_drag_p1": ..., "classify_into_bins_p2": ... }`

Comments describe variants such as:

> `p1 + a refused carry retried from higher up, then put back`

**Severity:** **Blocks legality**.

This is not merely an agent decomposing an instruction. It is a registry of benchmark policies and improved task-specific variants.

**Fix:** one shared controller and recovery policy should apply based on observed failure modes—e.g. every failed carry may retry at greater clearance—not only the classify task.

---

## 1.6 Perception explicitly maps instruction templates to task names

**Location:** `dojo/perception/j1_adapter.py:49-80`

> `TASK_PATTERNS = (`  
> `("press_by_number", ("red buttons", "number cards")),`  
> `("push_T", ("t-shaped block", "t-shaped pad")),`  
> `...`

> `def infer_task(instruction)`  
> `return task`

> `if task: return task, "caller"`

**Severity:** **Blocks legality**.

This is a literal instruction-to-task router. It also accepts a caller-provided task name.

**Fix:** remove task names from the perception API. Perception should take an open vocabulary of nouns/attributes/relations extracted from the instruction and return detections without selecting a task schema.

---

## 1.7 Runtime vocabulary table recognizes exact benchmark templates

**Location:** `dojo/runtime/supervisor.py:32-70`

Examples:

> `Vocabulary("press_by_number", (... "red buttons" ... "number cards" ... "blue button"))`

> `Vocabulary("pack_objects_into_box_random", (...))`

> `Vocabulary("make_kong", (...))`

> `Vocabulary("imitate_sorting_sequence", (...))`

> `Vocabulary("pour_liquid_into_cup", (exact public instruction template,))`

**Severity:** **Blocks legality**.

The names are task/config keys, and several patterns are effectively template fingerprints. This is not merely semantic parsing; it selects a benchmark-specific perception database.

**Fix:** replace `VOCABULARIES` with a task-independent noun/attribute extractor and a single asset index. Query candidates by semantic similarity or measured visual evidence without a task-key lookup.

---

## 1.8 Runtime turns vocabulary into the active “scene” and propagates it everywhere

**Location:** `dojo/runtime/policy.py:803-875`

> `vocab = perception_vocabulary(instruction)`

> `scene = vocab.name if vocab else ""`

> `_Checker(self.perception(), scene, ...)`

> `RuntimeProvider(..., scene, instruction, ...)`

> `self.perception().estimate([obs0], scene, self.fold)`

**Severity:** **Blocks legality**.

Avoiding the official task-name field is formal rather than substantive. The inferred vocabulary name becomes the task identifier used by perception, checking, watching, search, and agentic perception.

**Fix:** eliminate `scene`/task-key arguments from production perception and provider APIs.

---

# 2. Benchmark-task logic and unstated rules

## 2.1 `press_by_number`

**Location:** `dojo/skills/planner.py:627-649`

> `counted = max(kinds.values(), key=len)  # "the two red buttons": the repeated kind`

> `card = min(cards, key=distance-to-button)`

> `if confirm and "confirm" in text: seq.append((confirm[0].label, 1))`

**Severity:** **Risky**.

Positive finding: the planner only inserts the blue/confirm press if the instruction contains the word `confirm`. I did **not** find an unconditional execution-side “press blue after every red” rule in this planner path.

However, checker knowledge is explicitly provided to the perceiver:

**Location:** `dojo/perception/agentic/vendor/PERCEIVER-PROTOCOL.md:42,95`

> `checker rules visible in the public task code (press_by_number's blue presses)`

> `counts:[n0,n1],"order":"rule"`

**Severity:** **Blocks legality** if this perceiver participates in the sealed run.

It makes hidden checker procedure available to a policy component. Even if the main planner happens to require `confirm` text, the system’s perception/planning assistant is instructed that the blue-press rule may be taken from checker code.

**Fix:** remove checker-rule access from the perceiver. Every press must be licensed by parsed instruction semantics. If the instruction omits blue presses, the system must not infer them from the task checker.

---

## 2.2 Checker arithmetic duplicated exactly

**Location:** `dojo/skills/tiles.py:120-159`

> `"""The checker's arithmetic ... minus never below 0, division exact."""`

> division is accepted only when `a % b == 0`

**Severity:** **Risky / blocks legality**, depending on benchmark disclosure.

Ordinary arithmetic is fine, but choosing the benchmark checker’s special arithmetic domain is checker imitation, especially where the instruction does not define truncation, nonnegative subtraction, or exact division.

**Fix:** use standard symbolic arithmetic with ambiguity handling, or derive the convention from visible examples/instruction.

---

## 2.3 Checker thresholds embedded in tile placement

**Location:** `dojo/skills/tiles.py:32-55`

Examples:

> `ON_PAD = 0.035`

> `YAWS_DEG ... # checker: within 45`

> `REACH_SHIFTS ... # checker allows 18 / 20 mm`

**Severity:** **Risky**.

Motion tolerances are normal engineering constants, but comments and values show direct optimization against checker margins.

**Fix:** define tolerances from measured pad geometry and a task-independent safety margin, not checker acceptance radii.

---

## 2.4 Alignment implements the checker metric and ignores an instructed object

**Locations:** `dojo/skills/align.py:3-5,30,73`

> `blocks' centres lie within 8 mm of their best-fit line`

> `The checker never looks at the set square the instruction mentions`

> `SLIDE_LIFT = 0.003 # checker fails a rise above 7 mm`

**Severity:** **Blocks legality**.

Explicitly ignoring an instruction-mentioned object because the checker does not score it is benchmark exploitation, not general-purpose instruction following.

**Fix:** represent the set square as a geometric constraint/tool/reference and satisfy the natural-language task, independently of whether the checker examines it.

---

## 2.5 Row-by-size checker geometry

**Location:** `dojo/skills/rows.py:1-5,26`

> `public checker ... upright (15 deg)`

> `MIN_STEP = 0.042 # checker wants > 30 mm`

**Severity:** **Risky**.

The semantic operation is general, but exact arrangement geometry is checker-derived.

**Fix:** infer row spacing from object footprints plus collision clearance and uprightness from gravity/object axes.

---

## 2.6 Pack, store, sweep, garment, holders and insertion are checker-shaped policies

Policy-side modules openly define goals from checker code:

- `dojo/skills/pack.py:1-7`: target result defined as the public checker’s root-in-box and orientation conditions.
- `dojo/skills/store.py:5-7,75-110`: staged checker credits and an explicit `checker_ok`.
- `dojo/skills/sweep.py:8-16,37-45`: dustpan start tolerance, left/right final relation, and collection circle taken from checker behavior.
- `dojo/skills/garment.py:10,35-40`: fold windows and axes described as what the checker measures.
- `dojo/skills/holders.py:541,853,967`: exact containment/height/stage behavior optimized for the checker.
- `dojo/skills/insertion.py:257`: insertion depth chosen from checker threshold.
- `dojo/skills/artgeom.py:1-49`: helper functions explicitly reproduce “the checker’s functional points.”
- `dojo/skills/regrasp.py:15,85,927`: terminal poses and even when the checker passes are used to shape execution.
- `dojo/skills/precision.py:1168`: policy adjusted based on a discrepancy between its geometric test and checker result.
- `dojo/skills/kinematics.py:15`: gripper-open criterion described from scorer banking.

**Severity:** mostly **Risky**, with **Blocks legality** where the policy uses hidden thresholds, staged credit, or checker-only conditions not implied by the instruction.

**Fix:** define generic physical predicates:

- containment from measured meshes;
- support and stability from contact/geometry;
- alignment from instruction-specified relation;
- closed/open from measured articulation state;
- insertion from contact/depth/fixture geometry;
- sweeping from observed object transfer into a receptacle.

Do not import or replicate checker predicates.

---

## 2.7 Interactive tasks encode benchmark actor/checker timing

**Locations:**

- `dojo/skills/library.py:3759-3766`
- `dojo/runtime/policy.py:976-988`

> `tasks' checkers fail the episode the moment our arms are away from home ...`

> `No arm moves before it: no scan, looks or search`

**Severity:** **Blocks legality**.

The system recognizes demonstration/Kong/conveyor tasks and changes its pre-action procedure because of checker and actor timing. This violates “exact same inference procedure across all tasks” in a substantive sense.

**Fix:** use a task-independent actor-motion detector. Begin with the same observation routine for every episode; cancel or defer motions only when current visual motion/safety evidence requires it.

---

# 3. Runtime values from task configs, checker code, asset tables, or task counts

## 3.1 `tasks.json` is generated directly from benchmark task configs

**Location:** `dojo/perception/tasks.py:1-11,34-76`

> `Task -> candidate objects, read from RoboDojo's public task configs`

It imports:

- candidate assets and indices;
- label lists;
- object type;
- placement settings;
- `select_mode`;
- prohibited areas;
- whether the task is a `_random` variant.

**Severity:** **Blocks legality**.

This is the central leakage path. Public availability does not make it part of ordinary inference. It gives the policy task-conditioned object inventories and generator structure.

**Fix:** remove `tasks.json` from scored inference. Use one global asset catalogue and open-world detection.

---

## 3.2 Perception uses task-config candidates, counts and select modes

**Location:** `dojo/perception/inverse_layout.py:3-20`

> `Candidates from the task config`

> `respecting the task's counts and select mode`

**Severity:** **Blocks legality**.

Perception is not merely identifying visible objects. It solves a task-conditioned inverse-layout problem constrained by the benchmark generator.

**Fix:** generate proposals and identities solely from images, geometry, and a task-independent global catalogue.

---

## 3.3 Placement distributions and fixed slots come from task configs

**Locations:**

- `dojo/perception/inverse_layout.py:756-759`
- `dojo/perception/inverse_layout.py:1109-1125`
- `dojo/perception/inverse_layout.py:4469-4473`

> `Labels the config pins to a table slot`

> `Where the generator may put a label ... x/y ranges ... yaw range`

> `slots = fixed_slot_labels(task)`

**Severity:** **Blocks legality**.

These priors can reveal where an object should be even when image evidence is weak or absent.

The sealed override `pad_row_extend=true` further activates task-config pad-row logic at lines 4471-4473.

**Fix:** detect pads/slots visually and estimate their poses. Placement priors should be generic workspace bounds, identical across tasks.

---

## 3.4 Task-config quotas and identity relations

**Location:** `dojo/perception/inverse_layout.py:2549-2581`

> `per-category quota (public task config)`

> `select_category_nums`, `select_instance_nums`, `same_as_label`

**Severity:** **Blocks legality**.

These encode generator cardinality and relations not necessarily observable.

**Fix:** count and classify from observations. Unseen objects should remain unknown, not be inferred from generator constraints.

---

## 3.5 Series counts are clamped to task-config ranges

**Location:** `dojo/perception/inverse_layout.py:3369-3404`

> `clamped to the config's range`

> `lo, hi, step = instance_range(task, g)`

The sealed configuration sets `series_count="seg"`, but detected counts are still rounded and clamped to task-config bounds.

**Severity:** **Blocks legality**.

Using a segmenter does not cure the leakage when its result is corrected using task-specific minimum, maximum and multiplicity.

**Fix:** use raw instance detections with uncertainty and generic NMS/tracking; do not clamp them to task ranges.

---

## 3.6 Generator/checker label prefixes are reconstructed

**Location:** `dojo/perception/inverse_layout.py:5307-5320`

> `under the generator's labels ... as ... the checker's prefix lookups name them`

**Severity:** **Blocks legality**.

This is explicit checker-facing relabeling.

The sealed `series_labels="numbered"` causes runtime to expose numbered series labels (`dojo/runtime/policy.py:840`), increasing the relevance.

**Fix:** assign stable observation-derived UUIDs. Planner logic should use properties and tracks, never checker label conventions.

---

## 3.7 Task-conditioned select-tie policy

**Location:** `dojo/perception/select_ties.py:1-58`

> `TIE_OFF is keyed on the perception vocabulary`

> `if ... task in TIE_OFF: return "allow_duplicate"`

**Severity:** **Blocks legality**.

The identity algorithm changes for `make_kong`, `stack_blocks_by_language`, and `fasten_screws` based on an instruction-derived task name and a manual checker review.

**Fix:** use one identity-assignment rule for all tasks, based only on visual evidence and uncertainty.

---

## 3.8 Scene backgrounds selected from the inferred task’s asset set

**Location:** `dojo/perception/j1_adapter.py:83-100`

> `keys = scene_keys(task)`

> overlay with a scene background when one of the set’s fixtures has one

**Severity:** **Blocks legality**.

The calibration file may be globally frozen, but the active background changes according to the inferred task/vocabulary.

**Fix:** one generic background/foreground model, or estimate static background online from observations without a task key.

---

## 3.9 Asset descriptions and colour maps

**Locations:**

- `dojo/skills/ported/__init__.py:42-55`
- `dojo/skills/ported/skills/colour_order.py:5-20`

> instruction target grounded through asset `description.json`

> `CATALOGUE_COLOURS` copied from public task code/configs

**Severity:** asset descriptions alone are **Fine/Risky**; the hand-maintained benchmark colour map is **Risky**.

A global semantic asset catalogue can be a legitimate fixed-system component. A small mapping copied from benchmark instruction-generation code is closer to answer-key grounding.

**Fix:** infer colour from calibrated pixels or globally applicable asset metadata, not a task-specific hand table.

---

## 3.10 Step limit and task counts influence decisions

Examples:

- `dojo/runtime/supervisor.py:145-154`: retry/finish based on `world.step_lim`.
- `dojo/runtime/policy.py:1763-1767`: scan permitted based on fraction of official step limit.
- Many skills size loops from the inferred object count.

**Severity:** **Fine** for the officially observed `step_lim`; **blocks/risky** for object counts derived from task configs.

The firewall explicitly allows `step_lim`, so using it as a general resource budget is ordinary inference. It must not select a task policy.

---

# 4. Different tasks run different code paths or perception modes

## 4.1 Different skill modules and ported policies

**Locations:**

- `dojo/skills/library.py:3745-3826`
- `dojo/skills/planner.py:349-720`
- `dojo/skills/ported/__init__.py:67-76`

Different instructions dispatch to different complete controllers, including route-only policies not exposed as generic library skills.

**Severity:** **Blocks legality** for task-specialized composites; **Fine** for generic skill composition.

---

## 4.2 Different perception candidate sets

**Locations:**

- `dojo/runtime/supervisor.py:32-86`
- `dojo/runtime/policy.py:807-824`
- `dojo/perception/inverse_layout.py:4434-4473`

Every inferred task can receive different:

- label groups;
- asset candidates;
- instance ranges;
- fixed slots;
- placement fields;
- select modes;
- background assets.

**Severity:** **Blocks legality**.

---

## 4.3 Task-dependent scan/look/search suppression

**Location:** `dojo/runtime/policy.py:976-1034`

For `WAITS_ON_ACTOR` skills:

> `skipped: scan, look, search`

Other tasks may run scan, agentic perception, look fusion and search.

**Severity:** **Blocks legality**, because the branch is driven by task-recognizing planner rules and documented checker timing.

A uniform procedure may still condition actions on observed scene state, but this branch predicts the task before observing the actor behavior.

---

## 4.4 Task-dependent tracking

**Location:** `dojo/runtime/policy.py:1091-1094`

> `track_moving == "auto" and "conveyor" in cats`

**Severity:** **Fine/Risky**.

Conditioning tracking on an observed dynamic/conveyor object is general perception logic. It becomes risky because the conveyor category itself comes from a task-conditioned candidate set.

**Fix:** enable tracking based on measured optical/scene motion, not a task-derived category.

---

## 4.5 Held-pose perception only where a skill requests it

**Location:** `dojo/runtime/policy.py:1072-1079`

> `held_pose_on_request = ...`

> `today just insert_key's`

**Severity:** **Risky**.

The sealed `submit_v3` sets `held_pose="request"`, so different skills invoke different perception procedures. This can be legitimate active perception, but the current request points are benchmark-specialized.

**Fix:** one generic uncertainty policy should request in-hand pose whenever pose uncertainty threatens any downstream constraint.

---

## 4.6 Agentic perception receives instruction and wanted labels

**Locations:**

- `dojo/perception/agentic/j1.py:260-266`
- `dojo/perception/agentic/prompts/runner_robodojo_v1.md:14,38-39`
- corresponding v2 lines 18,42-43

> `Instruction: ...`

> `Wanted labels: ...`

**Severity:** **Fine** for instruction-conditioned open-vocabulary perception, **blocks legality** when wanted labels come from task-config/checker label schemas.

**Fix:** provide semantic queries such as “red button” or “container,” not generator labels or task-config object inventories.

---

## 4.7 Per-task identity mode

**Location:** `dojo/perception/select_ties.py:50-58`

Different inferred tasks run different assignment modes.

**Severity:** **Blocks legality**.

---

## 4.8 Globally fixed components

These aspects are comparatively sound:

- `submit_v3` is a versioned fixed preset: `dojo/runtime/presets.py:31-48`.
- `DOJO_COLOR_CALIB` points to one sealed calibration file.
- `--seg v2b`, `--glyph digit_v2`, and `--glyph-kong kong_v1b` appear globally pinned.
- `PERCEPTION_ENV["submit_v3"]` fixes `AGENTIC_SNAP=1`.
- The perception service documents one calibration set per service: `dojo/runtime/perception_service.py:100-108,400-430`.

**Severity:** **Fine**, but insufficient to overcome the task-conditioned logic above.

---

# Five highest-value changes

1. **Delete task-keyed perception from scored inference.**  
   Remove `VOCABULARIES`, `TASK_PATTERNS`, `task_for`, and task-name/`scene` parameters. Replace `tasks.json` candidate sets with a single global asset catalogue and observation-driven open-world detection.

2. **Remove all task-config generator priors.**  
   Do not use fixed slots, placement ranges, counts, linked counts, select modes, same-label relations, label prefixes, or task-specific backgrounds. `series_count="seg"` must return visual counts without clamping to config ranges.

3. **Collapse the benchmark-skill router into generic primitives.**  
   Remove `ported_plan`, `new_routes`, and named benchmark variants. Keep a common library of `pick/place/push/insert/press/pour/wait/observe/verify`; use one planner/search procedure to compose them.

4. **Purge checker-derived rules and thresholds from production policy.**  
   Remove `checker_ok`, exact checker arithmetic, stage-credit assumptions, checker-only orientation/containment thresholds, and the perceiver permission to use checker rules. Derive goals from instruction plus physical geometry.

5. **Make active perception and recovery policies uniform and evidence-triggered.**  
   Do not skip scan/look/search because a regex predicts an actor task. Trigger waiting from observed motion; trigger held-pose reads from uncertainty; trigger retry/clearance increases from measured failures. Apply these rules identically to every semantic task.

## Bottom line

The system is mechanically fixed but **behaviorally task-routed**. The principal issue is not that it understands instructions or invokes skills; the rules expressly allow that. The issue is that instruction recognition selects benchmark-specific object schemas, generator priors, perception behavior, checker-shaped predicates, and independently tuned task policies. At `127b65c4`, that crosses the quoted prohibition on a router switching policies by task and prevents a clean single-fixed-system claim.

Codex session ID: 01a1b6ed-20c8-7482-b720-d3222e098f16
Resume in Codex: codex resume 01a1b6ed-20c8-7482-b720-d3222e098f16
