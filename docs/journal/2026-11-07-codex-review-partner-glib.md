# 2026-11-07: partner team's Codex review of their general skill library (glib, night/JOINT-PLANNER)

Received from the partner team on the coordination bus (`20261107T1746Z-misha-glib-review-full.md`), kept verbatim below for the record. File:line references point to their tree (a controlr clone, head a4370c24/0a4fc760), not ours. Items on instruction leaks at internal boundaries, wrapper-only legality tests and near one-to-one benchmark verbs very likely apply to our skills too; see the LEGAL stream and [general-system rule](2026-11-07-general-system-rule.md).

## Review result

The HEAD implementation is not yet defensible as a rigid, task-independent library.

The largest legality problem is that there are effectively two registries:

- `glib.py` is intended to be the new general library.
- `model_planner.py` still consumes `model_registry.py`, whose default prompt exposes benchmark-specific policies and instruction-reading routes.

Even within `glib.py`, several verbs are benchmark solutions under generic names, and the legality tests inspect only their wrapper functions—not their transitive call graphs.

I did not run tests or modify files.

## Findings

### 1. The active model planner exposes explicit benchmark policies by default

- File: `dojo/skills/model_planner.py:59-60`
- Quote: `return SYSTEM % REG.describe(sealed_only=sealed_only, hide=hide)`
- File: `dojo/skills/model_registry.py:401-408`
- Quote: `def describe(sealed_only: bool = False, hide=()): ... if (sealed_only and not e.get("sealed", True)) or name in set(hide): continue`
- Severity: **blocks legality**
- Finding: `describe()` does not filter `scope="task"` or `reads_text=True`. `ModelPlanner` defaults to `sealed_only=False, hide=()`, so its fixed system exposes policies such as:
  - `tower_from_boards` / `build_tower` (`model_registry.py:54-59`)
  - `declare_kong` (`:83-90`)
  - `fold_garment` (`:155-161`)
  - `play_board_game` (`:170-178`)
  - `copy_demonstration` (`:179-188`)
  - numerous other task scripts
- Fix: Make the planner consume `glib.describe()` and `glib.check()/run()` exclusively. If the old registry remains, make `scope == "general" and not reads_text` an unconditional eligibility condition, not a caller option.

### 2. Default planner routes explicitly read instruction text inside the selected policy

- File: `dojo/skills/model_registry.py:311-326`
- Quote: `reads_text=True`
- File: `dojo/skills/model_registry.py:567-579`
- Quote: ported pour/classification routes are returned without explicit object handles.
- Severity: **blocks legality**
- Finding: `pour_bottle_into_cup`, `sort_into_baskets`, and `cover_and_uncover` admit that their implementations read the instruction. `to_runtime()` also maps several ported choices to empty argument dictionaries, forcing internal grounding. This is precisely a router selecting a task-specialized policy from task words.
- Fix: Remove all `reads_text` entries from the planner-visible registry. Require explicit object, target, grouping, and order arguments for every ported call, as the new `glib` API attempts to do.

### 3. `hide_instruction` is not honored by `cover_with` or `put_back`

- File: `dojo/skills/glib.py:381-386`
- Quote: `hide = s.impl == "dojo.skills.ported:ported"`
- File: `dojo/skills/general.py:149-151`
- Quote: `return ported(sk, "carry_keep_pose", ...)`
- File: `dojo/skills/general.py:154-169`
- Quote: both `cover_with` and `put_back` call `_carry`.
- File: `dojo/skills/ported/harness_robot.py:833-838`
- Quote: `text = "" if getattr(self.sk, "hide_instruction", False) else str(world.instruction or "")`
- Severity: **blocks legality**
- Finding: hiding is based only on the top-level registry implementation string. These two top-level implementations are `dojo.skills.general:*`, but call the ported adapter transitively, so `HarnessRobot.cmd_obs()` receives the real instruction. The 4a90a10e fix covers only direct `glib -> ported` calls.
- Fix: Hide instruction at the ported boundary whenever explicit handles are supplied, ideally inside `ported()`/`HarnessRobot`, rather than relying on callers. At minimum `_carry()` must set and restore `sk.hide_instruction` with `try/finally`.

### 4. The AST legality tests do not inspect the call graph

- File: `tests/dojo/test_glib.py:72-87`
- Quote: `fn = G.impl(name); src = inspect.getsource(fn)`
- Severity: **blocks legality**
- Finding: only the immediate registered function is inspected. For example:
  - `general.cover_with -> _carry -> ported -> HarnessRobot.cmd_obs`
  - `general.put_in_slots -> library.slots_and_press`
  - `general.place_tile -> tiles._place_tile/_verify_placed`
  - most library verbs call many helpers in large task-derived modules.
  
  A clean three-line wrapper around an arbitrary task policy passes.
- Fix: Build a closed, explicit allowlist of callable implementation roots and recursively scan repository-local callees, or enforce architectural separation: general skills may import only modules certified task-free. Add dynamic tests with an instruction sentinel that fails on any read.

### 5. The AST tests are trivially bypassable

- File: `tests/dojo/test_glib.py:63-69`
- Quote: docstrings are excluded; only direct attributes named `instruction` or `task` are rejected.
- File: `tests/dojo/test_glib.py:76-80`
- Quote: only strings shorter than 60 characters are checked, and with `TASK_RE.fullmatch`.
- Severity: **blocks legality**
- Finding: the test misses:
  - task names in docstrings;
  - task names embedded in longer strings;
  - `getattr(world, "instruction")`;
  - `world.__dict__["instruction"]`;
  - aliases and helper calls;
  - imported constants/config tables;
  - data-file reads;
  - indirect task/config access.
  
  The repository already contains long module docstrings naming exact tasks and checker predicates, which this design intentionally ignores.
- Fix: Scan all string literals with `search`, not `fullmatch`, without length/docstring exemptions. Reject reflective reads and known task/config symbols. More importantly, add transitive and runtime-sentinel checks; AST substring checks cannot establish noninterference.

### 6. `align_row` is effectively the `align_blocks` benchmark solution

- File: `dojo/skills/glib.py:123-125`
- Quote: `Slides identical blocks into one straight row ...`
- File: `dojo/skills/align.py:3-5`
- Quote: `RoboDojo align_blocks, public checker` and `The checker never looks at the set square`
- Severity: **blocks legality**
- Finding: this implementation explicitly optimizes the benchmark checker while discarding an instruction-named tool because the checker ignores it. That is not merely a generic row primitive; it embeds knowledge of what the benchmark rewards.
- Fix: Replace with a parameterized geometric primitive such as `align_centres(objects, line/frame, spacing, heading)` whose target geometry is supplied by the planner. Remove checker-specific verification and set-square omission from the implementation.

### 7. `row_by_size`, `pack_facing`, and `sweep_into` are near one-to-one benchmark policies

- Files/quotes:
  - `dojo/skills/rows.py:1-13`: names `sort_nesting_dolls_by_size` and reproduces its full success conditions.
  - `dojo/skills/pack.py:1-18`: names `pack_objects_into_box` and reproduces its checker.
  - `dojo/skills/sweep.py:8-17`: encodes the complete dustpan/broom procedure and checker-sensitive final arrangement.
- Registry: `dojo/skills/glib.py:126-130, 188-204`
- Severity: **blocks legality** for a strict board reading; at minimum **risky**
- Finding: each verb is sufficiently specific to be the benchmark task solution: object roles, sequence, orientation convention, final cleanup, and checker verification are all internal. The planner supplies little more than the task’s entities.
- Fix: decompose into lower-level verbs whose arguments fully specify target poses/frames, order, tool contact direction, and end-state requirements. The planner, using the same inference procedure, should compose those verbs.

### 8. `place_flat` embeds task-specific tile semantics

- File: `dojo/skills/glib.py:87-90`
- Quote: `number, letter or symbol tile ... upright and readable from the robot's side, squared to the table`
- File: `dojo/skills/general.py:89-110`
- Quote: delegates to `tiles._place_tile` and `_verify_placed`.
- File: `dojo/skills/tiles.py:36-38`
- Quote: headings and reach tolerances are justified against the checker.
- Severity: **risky**
- Finding: “readable from the robot’s side” and “squared to the table” are not entailed by generic placement on a pad. They encode the equation/digit-task presentation convention.
- Fix: expose orientation/read direction as explicit arguments, or replace this with a generic `place_at_frame(obj, target, orientation)`.

### 9. The documentation missed checker-derived constants in `carry_keep_pose`

- File: `dojo/skills/ported/skills/carry_keep_pose/skill_robodojo.py:6-10`
- Quote: `tol=0.015` and `root z within 12 mm ... axis within 10 deg`
- File: same, `:86-91`
- Quote: `xy_tol=tol, z_tol=0.012 ... tilt <= 10.0`
- Severity: **risky**
- Finding: these are verification thresholds in a task-derived carry implementation, but they are absent from the seven-row disclosure table. The file says it is extracted from `cover_uncover_v2`; the thresholds define success rather than merely safe motion.
- Fix: add all three to the disclosure, trace their provenance, and replace them with geometry/action-quality tolerances independent of any checker—or make the required tolerances explicit skill arguments.

### 10. The documentation understates checker-derived constants in `insert_each`

- File: `dojo/skills/precision.py:1161-1174`
- Quote: `r + 0.004`, upright within `30` degrees, and depth accepted within `EACH_PLACE_KW["seat_tol"]`; comment: `11.9 mm short counted as in, the checker did not`
- Severity: **risky**
- Finding: the disclosure table does not mention `insert_each`, although its postcondition was tightened specifically from checker disagreement. This affects whether the planner is told a skill succeeded.
- Fix: disclose and rejustify these values from hole/object geometry and insertion mechanics. Separate physical seating verification from benchmark success emulation.

### 11. `pack_facing` has undisclosed checker-shaped acceptance values

- File: `dojo/skills/pack.py:5-8`
- Quote: checker permits front-axis errors of `45 deg`/`60 deg`.
- File: `dojo/skills/pack.py:191-195`
- Quote: `tol_deg: float = 40.0`
- File: `dojo/skills/pack.py:228-230`
- Quote: placement uses `tol_tilt=45.0`, `tol_xy=0.06`, `tol_z=0.25`.
- Severity: **risky**
- Finding: the documentation’s `pack_facing` row only says “goal stated as the checker’s box test”; it does not enumerate these numeric thresholds. At least the 40° facing acceptance is directly checker-shaped.
- Fix: enumerate every checker-influenced value, then replace outcome verification with a library-defined physical contract or explicit caller-supplied tolerances.

### 12. Minimum cardinality preconditions are documented but not checked

- File: `dojo/skills/glib.py:95-100, 123-130`
- Quote: `two or more`, `three or more`.
- File: `dojo/skills/glib.py:284-289`
- Quote: `objs` validates only a non-empty list.
- Severity: **bug**
- Finding: `stack`, `align_row`, and `row_by_size` accept one object despite their contracts. `stack_objects` subsequently dereferences `objects[0]`, so empty is caught, but semantically invalid one-element plans pass.
- Fix: add `min_len`/`max_len` to `Arg`, serialize it in `as_json()`, and enforce it generically.

### 13. `groups` and `pairs` permit duplicate or self-referential objects

- File: `dojo/skills/glib.py:290-297`
- Quote: group/pair structure is checked, but uniqueness is not.
- Severity: **bug**
- Finding:
  - one object can appear in multiple peg groups;
  - a screw pair can be `[x, x]`;
  - repeated pairs are allowed;
  - unlike `objs`, duplicates are not rejected.
  
  These can generate destructive or nonsensical motions.
- Fix: validate every leaf as a string, reject duplicates across groups unless explicitly allowed, reject identical elements within pairs, and add role/movability constraints.

### 14. `screw_on` does not validate which member is movable

- File: `dojo/skills/glib.py:179-181`
- Quote: `pairs: "[part, partner] pairs"` with no movability metadata.
- Severity: **bug**
- Finding: the contract says “movable threaded part onto its fixed partner,” but `check()` accepts any two known objects in either order.
- Fix: introduce a structured pair schema with per-position constraints, or a dedicated `movable_fixed_pairs` type.

### 15. Optional default frames are never checked against the objects

- File: `dojo/skills/glib.py:166-173`
- Quote: `handle_frame ... required=False, default="handle"`
- File: `dojo/skills/glib.py:274-278`
- Quote: omitted optional arguments are skipped during validation.
- Severity: **bug**
- Finding: `hang(..., handle_frame omitted)` passes even when one or more objects lack a `handle` frame. The default is inserted only after validation at `glib.py:379`, so the precondition checker claims the call is legal and motion then fails.
- Fix: materialize defaults before validation, then validate the resulting arguments—including default frames—for every owner.

### 16. Floating-point validation accepts NaN

- File: `dojo/skills/glib.py:308-321`
- Quote: `isinstance(v, (int, float))` followed by range comparisons.
- Severity: **bug**
- Finding: `float("nan")` passes type validation and all `<`/`>` range comparisons. `model_registry.py:442-443` correctly includes `v == v`, but `glib.py` does not.
- Fix: use `math.isfinite(float(v))` for all numeric arguments.

### 17. `look_at` reports success even when every look fails or every object disappears

- File: `dojo/skills/general.py:127-146`
- Quote: exceptions become `seen[l] = "no look (...)"`, yet the function always returns `{"ok": True, ...}`.
- Severity: **bug**
- Finding: the planner receives `ok: look_at` even if nothing was observed. Unknown/gone objects are likewise compatible with success.
- Fix: return failure when no requested object was successfully re-read; include per-object success in structured details and distinguish partial success.

### 18. `Outcome` overstates its semantic guarantee

- File: `dojo/skills/glib.py:16-18`
- Quote: `ok: <name>: <what holds now>`
- File: `dojo/skills/glib.py:360-365`
- Quote: success text is derived only from implementation `ok`/`verified`; `_what()` merely echoes arguments.
- Severity: **bug**
- Finding: the outcome does not necessarily establish “what holds now.” Many implementations return procedural success, and `_what()` does not re-check the advertised postcondition. For example, `look_at` is unconditionally successful; other library calls may report `ok` without a fresh geometric verification.
- Fix: weaken the wording to “skill reported success,” or add explicit postcondition verifiers per verb and use only those to generate `ok: ... holds now`.

### 19. The free-target check uses a fixed centre-radius heuristic

- File: `dojo/skills/glib.py:253-258`
- Quote: `radius: float = 0.035`
- File: `dojo/skills/glib.py:336-340`
- Severity: **risky**
- Finding: target occupancy ignores support geometry, object footprint, height, and whether the nearby object is actually on the target. It can reject a legal call or accept an occupied large support. `general.py` duplicates the same 35 mm rule at lines 18-25.
- Fix: compute overlap/on-support from oriented footprints and vertical relation. Keep one shared implementation rather than two constants.

### 20. Ported-handle coverage test does not prove complete grounding isolation

- File: `tests/dojo/test_glib.py:82-86`
- Quote: it counts top-level values starting with `@` and compares only against `Arg.type == "obj"`.
- Severity: **risky**
- Finding: this does not verify:
  - handles nested in `objs`, groups, or pairs;
  - all semantically relevant objects;
  - that the vendored implementation does not consult instruction text despite handles;
  - transitive ported calls such as `_carry`.
- Fix: instrument `HarnessRobot.cmd_obs()` with an instruction sentinel and execute every ported path. Validate the complete transformed argument tree against the source arguments.

## Five most important fixes

1. Replace `model_planner -> model_registry` with the new `glib` interface, and make task-scoped or instruction-reading entries impossible to expose.
2. Move instruction hiding into the ported adapter itself; fix the `cover_with`/`put_back` transitive leak.
3. Replace the current AST tests with transitive import/call-graph auditing plus runtime instruction-sentinel tests.
4. Remove or decompose checker-shaped benchmark policies—especially `align_row`, `row_by_size`, `pack_facing`, and `sweep_into`—into parameter-complete geometric verbs.
5. Strengthen `glib.check()` with cardinalities, finite-number checks, nested uniqueness/role constraints, and validation of inserted defaults.

Codex session ID: 01a1b709-9a2a-7db2-938b-12eceda9c50f
Resume in Codex: codex resume 01a1b709-9a2a-7db2-938b-12eceda9c50f
