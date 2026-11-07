# 2026-11-07: partner team's Codex reviews: hybrid agent + runtime hook, and glib re-check

Received from the partner team on the coordination bus (`20261107T1813Z-misha-codex-reviews-full.md`), kept verbatim below. File:line refers to the partner's trees. Context: [general-system rule](2026-11-07-general-system-rule.md), [glib review](2026-11-07-codex-review-partner-glib.md).

**Two notes from the re-check (part 2 below), relevant to your inventory:**
1. The audit did not index the dynamically loaded vendored skills (ported/skills/** loaded through HarnessRobot.cmd_skill). One of them, pour_p1/skill_robodojo.py, implements a task checker as its success predicate and is reachable from glib pour_out. Any skill loaded by path needs the same scan. joint-planner is extending the tool to follow spec_from_file_location imports and the cmd_* dispatch.
2. Pin carrier exemptions to single lines, not whole functions, and run the sentinel through the production provider (RuntimeProvider), not only GTProvider.

=================== PART 1: Codex review of the hybrid agent + hook (verbatim) ===================

# Read-only review

## Verdict

The nominal path is general-purpose: I found no current prompt branch, task-name lookup, per-task procedure table, task-specific perception mode, or checker threshold selected from the instruction. The model chooses among a fixed set of general verbs, perception queries, and Python primitives, which is consistent with the board’s rule.

It is not yet safe or legally closed, however. The main blockers are:

1. Model-written Python runs in-process and can escape the AST restrictions.
2. The runtime interface permits an arbitrary locator to inspect the instruction, and does not enforce the frozen agent/locator/perception configuration.
3. Asset identity is trusted without provenance or validation.
4. The learning lint and promotion gate do not yet enforce the stated legality contract.

The requested snapshots were moving during review. I re-read the changed files after each material update.

- Initial required command, `git log -1 --format='%h %cs' -- AGENTS.md docs/`: `0ff5639f 2026-11-07`.
- Initial relevant status included modified `gen_loop.py`, `gen_prompt.py`, `tests/test_rdgen_loop.py`, and new `RD-GEN-DESIGN-2026-11-07.md`.
- Final RD-GEN snapshot reviewed: `b46e8d436882`; clean. Final docs revision: `46d55649 2026-11-07`; final `git status --short -- AGENTS.md docs/ tools/ skills/` was empty.
- Requested JOINT snapshot: `0a4fc760`; final concurrent snapshot re-reviewed: `2579500bf36e`; clean.
- At `0a4fc760`, `HybridApi.prim` was absent, so the two systems were not composable. JOINT commit `2579500b` added it during this review.

I did not modify anything.

## Legality findings

### 1. The Python sandbox is trivially escapable

**Severity: blocks legality**

`tools/perceiver/robodojo/j1/gen_session.py:18`

> `ALLOWED_IMPORTS = {..., "typing"}`

`tools/perceiver/robodojo/j1/gen_session.py:109`

> `exec(compile(tree, ...), self.g)`

The AST rejects `_` attributes and selected builtins, but an allowed module exposes the real process:

```python
import typing
b = typing.sys.modules["builtins"]
drv = b.getattr(robot, "_drv")
```

I ran a pure-Python proof. It recovered a private `_secret` attribute despite the AST rule:

> `reflection_escape: None None reachable-private-state`

From there a cell can recover `open`, `__import__`, `os`, `sys.modules`, environment variables, sockets, arbitrary loaded modules, and filesystem access. It can import routing/task modules even though the initial import closure is clean.

In JOINT, reflection reaches `robot._api`, `api.ctx`, `api.policy`, the belief, bridge, catalogue, raw runtime configuration, and the underlying environment wrappers. The nominal `TruthGuard` and `hide_instruction=True` are therefore not security boundaries. In J1 it can reach the raw server client and bypass the forbidden-command guard.

The policy-mode firewall means `dev_gt` should be absent during a normal mode-B episode, but unrestricted filesystem/process access can still read task configs, layouts, results, or simulator-side files if they exist on the host.

**Fix:** execute each cell in a separate OS process with a capability-only RPC connection to a small robot broker. Give it no repository mount, task files, environment, credentials, network, or simulator object. Apply a hard process timeout and memory/CPU limits. Treat the AST pass as usability lint only.

The new exact-import-closure and instruction-sentinel tests in `tests/test_rdgen_legality.py` are useful for the honest path, but they do not test adversarial cell execution.

### 2. The locator receives the instruction even though RD-GEN does not need it

**Severity: blocks legality**

`JOINT/dojo/runtime/hybrid.py:128-133`

> `self.locate_fn(self.observe(), queries, self.known())`

`JOINT/dojo/runtime/firewall.py:34`

> `OBS_KEYS = {"images", "state", "instruction", "steps_used", "step_lim"}`

The current `rdgen_hybrid.locate` ignores `obs["instruction"]`, so I found no actual task routing in that implementation. But the interface passes the instruction to any configured locator. A replacement locator can select task-specific candidates, thresholds, catalogue subsets, or code based on it.

The existing test explicitly permits this:

`JOINT/tests/dojo/test_hybrid.py:112-114`

> `assert set(obs) <= {"images", "state", "instruction", ...}`

**Fix:** construct a dedicated locator observation containing only RGB, public camera calibration/state, and step counters. Do not pass the instruction or full belief; pass only the queries and a minimal prior-object geometry structure. Add a test that the locator cannot receive the instruction even if it asks.

### 3. Arbitrary callback loading and hybrid flags do not enforce “one fixed system”

**Severity: blocks legality**

`JOINT/dojo/runtime/hybrid.py:43-48`

> `return getattr(importlib.import_module(mod), fn)`

`JOINT/dojo/runtime/policy.py:573-580`

> `hybrid_agent: str = ""`  
> `hybrid_locate: str = ""`  
> `return cls(**json.loads(os.environ.get("DOJO_RUNTIME", "{}") ...))`

Nothing in the runtime prevents the launcher from choosing different agent or locator modules by task. Similarly, hybrid mode skips several legacy paths, but it does not reject incompatible settings: `refine`, `a1`, `held_look`, `grip_check`, and other hooks can still be enabled at `policy.py:864-887` and `942-943`.

That is not evidence that current authors are routing by task; it is an unenforced path that makes a scored build difficult to defend.

**Fix:** add a scored-hybrid validator that:

- Allows exactly one agent and locator entry point.
- Pins their hashes, model checkpoint, prompt, library, catalogue, SAM code/weights, camera model, and runtime options.
- Rejects alternate perception hooks and verifier/twin/task-vocabulary settings.
- Emits one system fingerprint and asserts it is identical before every episode.

The separate Sonnet and Haiku J1 wrappers are fine for declared practice comparisons, but a scored system must freeze one.

### 4. The current asset identity contract trusts an unaudited string

**Severity: blocks legality**

`JOINT/dojo/runtime/hybrid.py:64-76`

> `t, c, i = str(d["asset"]).split("/")`  
> `info = assets.get(t, c, int(i), mesh=True)`

The hook accepts any locator-provided `"asset"` and immediately exposes its mesh, bbox, place/support/functional frames to the planner. It records neither how that identity was obtained nor the candidate universe.

Current RD-GEN does **not** use this capability:

`tools/perceiver/robodojo/gen/rdgen_hybrid.py:79-82`

> `d = {"noun": ..., "pos": ..., "extents": ..., "conf": ...}`

It returns no `asset`, so the current composed system gets box objects and frame-dependent verbs refuse. Thus the risky catalogue path is dormant rather than validated.

**Fix:** accept asset identity only as a structured, provenance-bearing result:

- Catalogue build hash and identity-model hash.
- Fixed candidate-universe hash.
- Per-view evidence, candidate scores, winning margin, and confidence.
- Explicit indication of whether the universe was full-catalogue or a generic visual/category subset.
- No task, layout, count, slot, or checker-derived candidates.

Reject an asset identity lacking that record. Do not silently turn it into a box.

### 5. The learning lint does not enforce its stated rules

**Severity: blocks legality for promotion**

`tools/perceiver/robodojo/gen/gen_learn.py:77`

> `def lint(code, task_names=()):`

`tools/perceiver/robodojo/gen/gen_learn.py:140-147`

> `"every number ... bound once as NAME ...; no task names or instruction words anywhere"`

`tools/perceiver/robodojo/gen/gen_learn.py:215-228`

> `task_names=()`  
> `"lint": lint(src, task_names)`

The prose promises checks that the implementation does not perform:

- Task names are checked only if the caller supplies them; there is no production caller yet.
- Instruction words are not checked.
- Scalar task constants are not rejected or classified.
- The named-constant/physical-reason rule is prompt text only.
- Task-shaped function names and paraphrased task procedures can pass.
- There is no implemented held-out promotion runner in this scope; the “gate” remains design text.

I verified that this passes with the default lint:

```python
def press_by_number(robot, arm):
    """Press exactly three times at benchmark-specific depth."""
    for i in range(3):
        robot.move_by(arm, dz=-0.027, contact=True)
```

The result was `[]`. Supplying `["press_by_number"]` then catches only the exact name.

**Fix:** make the task-name set mandatory and internal; reject missing configuration. Add checks for instruction substrings, task-shaped names/docs, unexplained numeric constants, handles, counts, coordinates, reflective calls, and non-whitelisted call graphs. Require human/legal review plus a separately implemented held-out gate before a candidate becomes callable. Promotion should produce a new frozen library version, never edit the active run.

## Safety and correctness findings

### 6. Stop and timeout exceptions are catchable by model code

**Severity: bug**

`tools/perceiver/robodojo/j1/gen_session.py:23-32`

> `class Stop(BaseException)`  
> `class CellTimeout(BaseException)`

`BaseException` remains in the safe builtins. A cell can write:

```python
try:
    robot.done("done")
except BaseException:
    pass
```

My proof returned:

> `stop_swallowed: None None True`

The same applies to the alarm exception. A bare `except:` also catches both.

**Fix:** external process termination. AST-rejecting `except BaseException` and bare `except` is useful defense in depth, but cannot make in-process execution safe.

### 7. JOINT cells have no wall-clock timeout

**Severity: bug**

`tools/perceiver/robodojo/j1/gen_session.py:99-102`

> `except (ValueError, AttributeError):  # not the main thread: no wall limit`

`JOINT/dojo/runtime/bridge.py:61`

> `threading.Thread(... name="dojo-runtime-program")`

The final hybrid runs inside the bridge worker thread. Python signals cannot install there, so every hybrid cell has no wall limit. An infinite loop blocks the episode and may survive normal cancellation.

**Fix:** the cell worker must be a killable subprocess. Do not rely on `SIGALRM` or cooperative exceptions.

### 8. Terminal skill outcomes are converted into ordinary failures

**Severity: bug**

`JOINT/dojo/skills/glib.py:389-390`

> `except (BudgetExceeded, EpisodeEnded) as e:`  
> `return Outcome(False, f"fail: ... budget ...")`

`JOINT/dojo/runtime/hybrid.py:177-192` treats this as a normal `Outcome` and returns it to the model.

Consequently, an episode success/end latch or exhausted budget can be reported as a failed skill after which the model continues. The primitive path is better: `HarnessRobot` puts `ended` in its reply and `gen_tools._call` converts that to `Stop`.

**Fix:** never flatten `EpisodeEnded`. Propagate it to `RuntimePolicy._episode`, which already handles it. Propagate `BudgetExceeded` similarly, or convert both to an explicit terminal result that the agent loop must stop on.

### 9. Steps-left becomes stale after a general skill

**Severity: bug**

`tools/perceiver/robodojo/j1/gen_loop.py:214`

> `sl = self.robot.steps_left()`

`gen_tools.Robot.steps_left()` reads its last primitive reply. A JOINT JSON skill runs through `api.run`, not `PrimDrv`, so it does not update `robot._last`. After a large glib skill, the footer and budget check can still show the pre-skill budget.

**Fix:** override `HybridRobot.steps_left()` to call `api.steps_left()` directly, and use it for every post-action row. Add a test where a skill consumes steps and the next model message sees the reduced value.

### 10. Handle association is not one-to-one and ignores height

**Severity: bug**

`JOINT/dojo/runtime/hybrid.py:143-145`

> `norm(...center_w[:2] - ...pos[:2])`  
> `lab = near[0][1] ...`

Two detections can choose the same existing handle because matched handles are not reserved during the loop. Stacked same-noun objects at the same XY are especially vulnerable. The second detection overwrites the first. Locator-supplied handles can also collide with an unrelated belief object and overwrite it.

Unseen objects never expire, so stale phantoms accumulate.

**Fix:** validate explicit handle uniqueness; associate an entire detection set to existing objects with one-to-one matching in 3D plus size/identity gates; reserve matches; maintain visibility-aware miss counters and tombstones.

### 11. Catalogue identity overrides explicit mobility

**Severity: bug**

`JOINT/dojo/runtime/hybrid.py:70-76`

> `typ = "Rigid" if d.get("movable", True) else "Geometry"`  
> `typ, cat, idx = t, c, int(i)`

When an asset is supplied, the explicit `movable` observation is discarded. A fixed fixture identified as a `Rigid/...` asset becomes movable; a movable instance of a `Geometry/...` asset becomes fixed.

The asset string is also not canonicalized or checked against a manifest before being used as a path key.

**Fix:** keep instance mobility separate from asset type. Validate the key against a frozen catalogue membership map and reject `..`, unexpected types/categories, noncanonical indices, and paths escaping the catalogue root.

### 12. `LAST` is mutable global cross-episode state

**Severity: bug**

`tools/perceiver/robodojo/gen/rdgen_hybrid.py:32`

> `LAST: dict = {}`

`rdgen_hybrid.py:81,123`

> `LAST[_key(q, o["xyz"])] = d`  
> `LAST.get(_key(text, f.get("pos") or []), {})`

Measurements are joined back to handles through a module-global dictionary keyed by noun and millimetre-rounded position. It is never cleared. Concurrent episodes, repeated scenes, or two close instances can cross-contaminate extras such as height, yaw, and score.

**Fix:** carry extras through `HybridApi.locate` and the belief return directly. At minimum, store them in an episode-owned object keyed by an explicit detection ID.

### 13. The local USD guard is soft, not a hard dollar cap

**Severity: risky**

`tools/perceiver/robodojo/j1/gen_loop.py:177-183`

> `r = self.call(...)`  
> `self.usd += cost`

`gen_loop.py:230-231`

> `if self.usd >= self.usd_cap:`

The guard is checked after a complete billed call, so it can overshoot. The new Sonnet value is `$1.10`, but `MAX_TOKENS=16000` and a growing prompt still make this a post-hoc guard. `MAX_CALLS` counts loop replies; transport attempts are only reported separately.

Step budgets and the 30-motion-per-cell cap are nominally enforced, but the sandbox escape bypasses the robot wrapper.

**Fix:** enforce dollars/tokens at the relay, with a hard per-episode ledger. Before each call reserve the maximum possible cost from the remaining allowance and lower `max_tokens` accordingly.

### 14. The current `api.prim` test is not hermetic and checks the wrong error field

**Severity: bug**

`JOINT/tests/dojo/test_hybrid.py:168-170`

> `assert rep["error"] is None`  
> `o = seen["obs"]`

On final JOINT `2579500b`, this test failed locally: 6 passed, 1 failed. The agent entered, but `api.prim("obs")` raised because `X5A.urdf` was not found. Runtime caught it under `error_handled`, while `rep["error"]` remained `None`, so the asserted error check passed and the later lookup failed misleadingly.

**Fix:** inject a public camera/X5 model fixture or set a pinned `DOJO_X5_DIR` in the test. Assert both `error` and `error_handled` are absent. Include the URDF and camera-model hashes in the scored manifest.

## Prompt and perception audit

I found no task names, per-task instructions, task-config values, or explicit checker tolerances in the current prompt. The task-name scan is weak against paraphrases, but manual inspection found generic verb procedures rather than benchmark-task procedures.

`tools/perceiver/robodojo/j1/gen_prompt.py:92`

> `Pressing a button or key ... 1-2 cm below ... Nothing reports a registered press ... Count presses exactly.`

The public PyRUA note says, at lines 86-94, the same generic 1–2 cm procedure, followed by:

> `measured on swap_blocks, every command from 1 to 2.8 cm deep ...`

That task-specific measured sentence was **not copied**. The task name, 2.8 cm result, and measured bottom-out depth are absent. The generic 1–2 cm press procedure was copied/paraphrased and should remain disclosed as generic operating knowledge.

The former exact final-scene centimetre/angle and home/open checker thresholds were also removed. Current line 103 says only:

> `Finish tidy, the same way in every episode ... robot.home(open=True).`

The multi-view SAM locator receives text queries, RGB, calibration, and generic parameters. It receives no task identifier or asset catalogue. Choosing query words, views, `low`, or thresholds inside the one fixed algorithm is ordinary active perception, not a task-policy router.

Its evidence remains limited: the geometry tests use perfect synthetic masks. There is no committed real-SAM accuracy/identity gate in this scope. The loader and weights come from an external sealed checkout and must be vendored or hash-pinned.

## Asset-catalogue identity

The architecture is defensible as general object knowledge **if** the same catalogue, identity procedure, candidate policy, and thresholds are used for every task. “Same catalogue” alone is insufficient: task-conditioned candidate pruning can encode the answer while still technically consulting one catalogue.

Legitimate source classes:

- Robot and camera geometry:
  - `env_cfg/arx_x5.yml`
  - `env_cfg/camera/camera_config.yml`
  - `env_cfg/camera/template.py`
  - `env_cfg/robot/dual_x5.yml`
  - `env_cfg/scene/default.yml`
  - `Assets/Robots/x5/robot_config.yml`
  - `Assets/Robots/x5/X5A.urdf`
- Intrinsic asset data:
  - `Assets/Object/RoboDojo/<Type>/<category>/<id>/object.usd[z]`
  - `metadata.json`
  - textures, materials, and `description.json`
- Catalogue-wide deterministic derivatives of those sources: meshes, bboxes, joints, stable poses, visual embeddings, and intrinsic affordance frames, with source hashes.

Illegitimate identity sources:

- `task/RoboDojo/config/**`
- `task/RoboDojo/tasks/**` checker/task implementations
- `Assets/Eval_Layout/**`
- `Assets/Traj/**`
- Results, episode logs, GT records, or scored-layout observations
- Per-task vocabularies, candidate lists, object counts, `select_mode`, slot maps, correspondence tables, or task-specific identity thresholds.

`JOINT/dojo/skills/world.py:189-203` reads `active.place`, `passive.support`, and functional frames from raw asset metadata. Those can be defended as intrinsic affordances if extracted catalogue-wide and independent of task/checker files.

However, `JOINT/dojo/perception/catalog.py:695-704` calls them:

> `Checker-relevant attributes`

and `catalog.py:614-616` contains a category-specific `number_card idx -> digit` rule. That particular rule does not currently flow through `HybridApi.object_from_located`, which loads raw metadata and geometry, but it should not be part of the clean identity catalogue unless the digit is mechanically derived from the asset’s texture/OCR or authoritative intrinsic asset metadata.

## Home/open convention

An unconditional “open both grippers and return both arms home” finalizer is legal under the board’s fixed-system rule. It does not choose code by task. It is best described as a disclosed benchmark-wide protocol prior, not as universal robot safety knowledge: opening a gripper is unsafe/wrong for a lift-or-hold goal.

There is also an inconsistency:

- `gen_prompt.py:103` and `j1_driver_v9gen.py:110-115` always open/home after motion.
- `JOINT/dojo/runtime/hybrid.py:230-235` skips finishing whenever any object is held.

The JOINT exception is not inherently a task leak; it is state/goal-conditioned behavior in one general algorithm. But “any object held” is too broad: an accidental attachment during a non-hold task also suppresses home.

Use one explicit final-state contract produced by the planner, such as `finish_mode = tidy | preserve_hold`, justified from the natural-language goal and executed by the same finalizer on every task. Record the decision in the transcript.

## Truncation and normal terminal behavior

The cut-output behavior is correct: `gen_loop.py:195-198` executes nothing when the reply is truncated. Cell stdout/value and transcript rows are clipped, and old conversation turns/images are trimmed; persistent Python state remains. That creates possible model confusion but not an execution escape.

The normal primitive path detects `ended`; the general-skill path currently does not, as described above. Runtime-level `EpisodeEnded` handling is otherwise present in `policy.py:1169-1189`.

## Tests run

- Sandbox proof: private-state access and `Stop` swallowing both reproduced.
- RD-GEN: 12 targeted pure-Python tests passed:
  - geometry suite;
  - learning harvest/lint/prompt/record-schema tests.
- JOINT at the requested-era hook: 6 hybrid tests passed before the new primitive test appeared.
- Final JOINT `2579500b`: 6 passed, 1 failed because the new primitive test could not find `X5A.urdf`; the underlying exception was stored in `error_handled`.
- The complete RD-GEN pytest files could not run in this read-only sandbox because their fixtures create temporary PNGs/directories and pytest capture requires a writable temp directory.

## Five most important fixes

1. Replace the in-process Python session with a killable, OS-isolated capability worker.
2. Close the scored configuration: sanitized locator input, exact agent/locator/model hashes, and rejection of all alternate perception/task hooks.
3. Define and enforce provenance for asset identity, using a fixed catalogue universe and no task/config/checker sources.
4. Fix hybrid runtime semantics: propagate terminal outcomes, report live steps, use one-to-one handles, and remove global `LAST`.
5. Make the learning gate real: mandatory legality lint, semantic/numeric review, held-out promotion, and a newly frozen library version only after passing.

Codex session ID: 01a1b71b-1cca-77c2-b7e3-b4cd8d91841a
Resume in Codex: codex resume 01a1b71b-1cca-77c2-b7e3-b4cd8d91841a


=================== PART 2: Codex re-check of glib fixes (verbatim) ===================

# Read-only re-check report

Scope: `0a4fc760..5a5a3837456fbde3554769b32366d09302d6dde8`.

The branch advanced during review:

- `07d8f1a4`: fixes 1–3 plus first audit.
- `2579500b`: hybrid integration.
- `f8b11760`: substantial fixes for 6–19.
- `5a5a3837`: notes only.

Current worktree has uncommitted replacements for `tests/dojo/test_glib_audit.py` and a new `dojo/legality/`; those are not part of `HEAD` and are excluded.

## Executive verdict

The active planner and instruction-hiding boundary are fixed. The committed audit is not a defensible proof of legality: it excludes dynamically loaded vendored code, and an active `pour_out → pour_p1` path contains and executes benchmark/checker-shaped logic that the audit cannot see.

| Original finding | Verdict |
|---|---|
| 1 | fixed |
| 2 | fixed |
| 3 | fixed |
| 4 | partly fixed |
| 5 | partly fixed |
| 20 | partly fixed |

Therefore the claim “findings 1–5 and 20 are fixed” is false as a group. Under a strict legality reading, findings 4/5 remain blocking.

Of findings 6–19, findings 8, 10, 14, and 19 retain material residuals. The others are fixed for the active GLib path.

## Tests

I ran, read-only:

```text
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 \
pytest -q -s -p no:cacheprovider \
  tests/dojo/test_glib_audit.py \
  tests/dojo/test_glib.py \
  tests/dojo/test_model_planner.py
```

Result on the clean committed f8 code:

```text
112 passed in 9.87s
```

The initial pytest attempt failed because no writable temporary directory was available; `-s` avoided capture/tempfile creation. No GPU or network was used.

## Claimed fixes

### One registry: confirmed for the active planner

- `dojo/skills/model_planner.py:23`
- Quote: `from dojo.skills import glib`
- `dojo/skills/model_planner.py:66-68`
- Quote: `return SYSTEM % glib.describe()`
- `dojo/skills/model_planner.py:131`
- Quote: `bad = glib.check(...)`
- `dojo/skills/model_planner.py:169`
- Quote: `def __init__(self, chat=None, model: str = MODEL, objects_root=None, max_calls: int = MAX_CALLS, log=None)`
- `dojo/skills/model_planner.py:292`
- Quote: `p = Plan("glib", {"steps": steps}, ...)`

There is no `hide`, `sealed_only`, or `opts` constructor option. Runtime dispatch sends `"glib"` to GLib’s `run_steps`.

The superseded registry is sealed at `describe()`/`check()`:

- `dojo/skills/model_registry.py:403-406`
- Quote: `return e.get("scope") == "general" and not e.get("reads_text") and not e.get("replaced")`
- `dojo/skills/model_registry.py:477-483`
- Quote: `if e is None or name in set(hide) or not eligible(name): return f"unknown skill {name!r}"`

Caveat: `model_registry.to_runtime()` itself does not enforce `eligible()`. A caller deliberately bypassing `check()` can still use task entries, but no active planner does so.

There is also a historical-script regression:

- `scripts/skills/model_loop_eval.py:318`
- Quote: `ModelPlanner(..., hide=REG.REPLACED)`

That now raises `TypeError`. The file declares itself historical, so this is a bug/nit rather than a legality blocker. Fix: remove the stale executable path or port it to GLib.

### Instruction hidden at the ported boundary: confirmed

- `dojo/skills/ported/__init__.py:67-75`
- Quote: `def has_handle(v) ...` recursively walks dicts/lists/tuples.
- `dojo/skills/ported/__init__.py:85-87`
- Quote: `hide_instruction=has_handle(args)`
- `dojo/skills/ported/harness_robot.py:836-843`
- Quote: `"instruction": "" if self.hide_instruction else str(world.instruction or "")`
- `dojo/skills/general.py:273-276`
- Quote: `{"obj": "@" + obj, ...}`

Thus `cover_with` and `put_back` reach `_carry`, which passes an explicit handle; the boundary hides the instruction. Direct GLib ported mappings likewise construct `@label` handles.

I found no current active GLib path that consumes instruction content after this boundary.

## Breaking the committed audit

### Concrete uncaught checker path

This is the strongest blocker.

1. GLib exposes `pour_out`:

   - `dojo/skills/glib.py:207-210`
   - Quote: `call=_ported("pour_p1", {"container": "container", "target": "target"})`

2. The adapter dynamically imports a file excluded by the static index:

   - `dojo/skills/ported/harness_robot.py:959-979`
   - Quote: `spec_from_file_location(...)`, `spec.loader.exec_module(mod)`, `rep = mod.run(...)`

3. That active file explicitly names the benchmark and its verifier:

   - `dojo/skills/ported/skills/pour_p1/skill_robodojo.py:9`
   - Quote: `pour_balls_into_vase ... score 0`
   - `:26-27`
   - Quote: `Verified = every ball inside the target's world AABB footprint ... tilted <= 15 deg, both arms home.`

4. It implements and uses that predicate:

   - `:182-184`
   - Quote: `lo[0] <= bpos[0] <= hi[0] ... bpos[2] > lo[2]`
   - `:424`
   - Quote: `sum(_in_target(...) for b in balls)`
   - `:436`
   - Quote: `fin["in_target"] == len(balls) and fin["cup_tilt"] <= 15.0 and ...`

This is not a dead docstring: the checker-shaped predicate determines `verified`, which GLib uses as the outcome. The committed static audit never parses this subtree, while the sentinel checks only instruction content.

- Severity: **blocks legality**
- Fix: replace this result with a generic physical pouring contract/postcondition owned by GLib, or audit the entire dynamically loaded subtree and eliminate task/checker-derived acceptance logic.

### Runtime sentinel claim is overstated

At `07d8f1a4` there were 29 verbs. Current HEAD has 27 after four removals and two additions.

- `tests/dojo/test_glib_audit.py:337-363`
- Quote: `@pytest.mark.parametrize("name", sorted(G.REGISTRY))`
- But:
  - `args = CALLS.get(name)`
  - `if args is None: ... assert not o.ok; return`
  - Comment: `refused by check, no code runs`

The committed `CALLS` covers 22/27 verbs. These five implementations are not entered:

- `hang`
- `press_part`
- `put_in_slots`
- `strike`
- `thread_on_pegs`

All six current vendored paths are exercised and required to produce empty instruction replies, which is useful. The negative control at `tests/dojo/test_glib_audit.py:366-370` is also valid.

- Severity: **blocks legality** as a proof gap
- Fix: construct valid fixture geometry and arguments for every verb; assert that each implementation root and intended branch was entered.

### Static call-graph blind spots

The graph is a meaningful improvement, but not closed:

- `tests/dojo/test_glib_audit.py:37-43` indexes only `dojo/skills/*.py`, two adapter files, and two runtime files. It omits `ported/skills/**`, most runtime code, perception, and nested packages.
- Local relative imports in `Index.callees()` ignore `ImportFrom.level` at `:139-145`.
- Module-level assignments, dictionaries, callbacks, constants, and module docstrings are not indexed; only functions, class methods, and assignment lambdas are recorded at `:86-101`.
- `x.m()` over-approximates only methods from indexed classes. Instances returned by unindexed factories remain invisible.
- `HarnessRobot.call`’s string-built dispatch is allowlisted at `:61-64`, but the `cmd_*` targets are not linked.
- `cmd_skill` then performs a second dynamic import into the excluded vendored subtree.
- Callback invocations from `RuntimeProvider.refresh` are outside the proof. Production refresh can invoke injected `checker`, `track_reader`, `watcher`, and `a1_client` objects; the sentinel uses `GTProvider`, not `RuntimeProvider`.
- Generic data reads are not prohibited. Current examples—`bowl_nest.json`, `physics_com.json`, `articulation_roots.json`—appear legitimate asset data, but the audit would accept a task/config file under an innocuous name.

Over-approximating same-named methods can inflate `len(seen) > 150` while still missing the actual receiver’s unindexed implementation.

### Pinned exceptions

`CARRIERS` says “two” but lists four.

- `GTProvider.refresh`: current use is plumbing, but the entire function is exempt once one exact line remains. A second instruction read would pass.
- `RuntimeProvider.refresh`: its entry is `None`, so nothing is pinned. This is too broad, especially because the dynamic sentinel never uses that provider.
- `RunnerEnv.instruction`: this is an accessor that actively converts the instruction to `str`, not merely a carrier. It is defensible only with a typed proof that GLib cannot call it; the current graph does not provide that.
- `HarnessRobot.cmd_obs`: the ternary is reasonable and current vendored runtime paths are observed, but broad whole-function exemption would permit added reads.

`glib.impl`’s reflective exception is reasonable because roots are derived from the fixed registry and missing roots fail. `HarnessRobot.call`’s exception is not sufficiently closed because the command targets and nested skill imports are not statically audited.

### Scanner bypasses

The original easy bypasses are mostly addressed for reached functions, but remaining holes include:

- task names only from fixture keys;
- concatenated strings/f-strings assembled from clean fragments;
- module-global strings/docstrings/constants;
- aliased or renamed task/config symbols;
- unindexed dynamic modules;
- carrier-wide exemptions;
- arbitrary data filenames.

The poison `str` also cannot prove every possible content access; it overrides a finite method list. Static rejection is therefore still required for every reachable source path.

## Original findings 1–20

### 1. Active planner exposed benchmark policies — fixed

- File: `dojo/skills/model_planner.py:23,66-68,131,292`
- Quote: `from dojo.skills import glib`; `SYSTEM % glib.describe()`; `glib.check(...)`; `Plan("glib", ...)`
- Severity: **blocks legality**
- Fix/status: implemented. Keep `model_registry` out of active planner/runtime imports; preferably make its unchecked `to_runtime()` private or enforce `eligible()` there too.

### 2. Planner routes read instruction text internally — fixed for the active planner

- File: `dojo/skills/glib.py:73-79`
- Quote: `_ported(...)` maps every named object to `"@" + a[k_src]`.
- File: `dojo/skills/ported/__init__.py:78-87`
- Quote: `hide_instruction=has_handle(args)`
- Severity: **blocks legality**
- Fix/status: implemented for active GLib routes. Remove or hard-seal old unchecked registry conversion paths to prevent future accidental reuse.

### 3. `cover_with`/`put_back` leaked the instruction — fixed

- File: `dojo/skills/general.py:273-294`
- Quote: `_carry(... {"obj": "@" + obj, ...})`; both verbs call `_carry`.
- File: `dojo/skills/ported/harness_robot.py:836-843`
- Quote: `"instruction": "" if self.hide_instruction ...`
- Severity: **blocks legality**
- Fix/status: implemented at the correct boundary.

### 4. Tests did not inspect the transitive call graph — partly fixed

- File: `tests/dojo/test_glib_audit.py:171-185`
- Quote: roots are traversed through `idx.callees(k)`.
- Counterexample: `dojo/skills/ported/harness_robot.py:959-979` dynamically imports excluded skill code.
- Severity: **blocks legality**
- Fix: index nested packages and dynamically loaded skill modules; resolve relative imports; model module-global callbacks/dataflow; fail closed on unresolved dispatch.

### 5. AST audit was trivially bypassable — partly fixed

- File: `tests/dojo/test_glib_audit.py:205-244`
- Quote: it now scans strings, direct/reflected instruction reads, `vars`, `__dict__`, selected symbols, and selected paths.
- Counterexample: active `pour_p1` checker logic is outside the index.
- Severity: **blocks legality**
- Fix: scan all reachable modules and module-level code; replace short symbol/path lists with fail-closed import/data-source policies; narrow carrier exceptions to exact AST dataflow.

### 6. `align_row` was the benchmark solution — fixed

- File: `dojo/skills/glib.py:22-26`
- Quote: `align_in_row ...` is listed among verbs no longer in the library.
- File: `tests/dojo/test_glib.py:21-25`
- Quote: `"align_in_row"` is in `KEPT_OUT`.
- Severity: **blocks legality**
- Fix/status: removed. Planner now has `slide_to` with explicit pose arguments.

### 7. `row_by_size`, `pack_facing`, `sweep_into` were benchmark policies — fixed

- File: `dojo/skills/glib.py:22-26`
- Quote: `row_by_size, pack_box, sweep_into` are explicitly excluded.
- File: `docs/notes/GENERAL-LIB-INTERFACE.md:135-139`
- Quote: they were removed and replaced by parameter-complete primitives.
- Severity: **blocks legality**
- Fix/status: removed from the active registry.

### 8. `place_flat` embedded tile-task presentation semantics — partly fixed

- File: `dojo/skills/glib.py:95-99`
- Quote: caller must provide `yaw_deg`.
- Residual: `dojo/skills/general.py:147-168`
- Quote: it still imports `ATTEMPTS`, `PLACE_KW`, `_verify_placed`, and other values from `dojo.skills.tiles`.
- `dojo/skills/tiles.py:220-223`
- Quote: `the checker's x -> x, y -> y` and a checker-derived face-down failure.
- Severity: **risky**
- Fix: move face-up/yaw construction and verification into a generic geometry helper with no tile-task/checker provenance; audit the imported module globals.

### 9. `carry_keep_pose` tolerances were undisclosed/checker-derived — fixed for GLib

- File: `dojo/skills/general.py:261-276`
- Quote: `carry_tols` derives XY/Z tolerance from object extents and passes them explicitly.
- File: `docs/notes/GENERAL-LIB-INTERFACE.md:157-160`
- Quote: the carry tolerances and derivation are disclosed.
- Severity: **risky**
- Fix/status: implemented. The vendored defaults remain, but active GLib calls override them.

### 10. `insert_each` checker-shaped acceptance was undisclosed — partly fixed

- File: `docs/notes/GENERAL-LIB-INTERFACE.md:159-161`
- Quote: centre `+4 mm`, upright `30 deg`, depth `8 mm` are now disclosed.
- Residual: `dojo/skills/precision.py:1161-1174`
- Quote: `r + 0.004`, `math.radians(30)`, and `EACH_PLACE_KW["seat_tol"]`; comment says `the checker did not`.
- Severity: **risky**
- Fix: define and justify an independent physical seating contract, remove checker-disagreement provenance from active acceptance logic, and separate physical verification from benchmark scoring.

### 11. `pack_facing` checker-shaped tolerances — fixed by removal

- File: `dojo/skills/glib.py:22-26`
- Quote: `pack_box` is not in the library.
- File: `docs/notes/GENERAL-LIB-INTERFACE.md:162`
- Quote: `40 / 45 deg, 0.06 / 0.25 m acceptance | gone with the verb`
- Severity: **risky**
- Fix/status: removed from active GLib.

### 12. Minimum cardinality was not enforced — fixed

- File: `dojo/skills/glib.py:49-50`
- Quote: `min_len`, `max_len`
- File: `dojo/skills/glib.py:112-113`
- Quote: `min_len=2`
- File: `dojo/skills/glib.py:363-365`
- Quote: list length is checked.
- Severity: **bug**
- Fix/status: implemented; removed three-object policies no longer require cardinality rules.

### 13. Groups/pairs allowed duplicates and self-pairs — fixed

- File: `dojo/skills/glib.py:321-338`
- Quote: `names an object twice`, `pairs an object with itself`, `names an object in two pairs`
- Severity: **bug**
- Fix/status: implemented.

### 14. `screw_on` did not validate movable/fixed roles — partly fixed

- File: `dojo/skills/glib.py:192-195`
- Quote: `roles=("movable", "any")`
- File: `dojo/skills/glib.py:338`
- Quote: movable leaves are selected by pair position.
- Severity: **bug**
- Finding: the first member is now guaranteed movable, but the second is explicitly `"any"`, not fixed. The original fixed-partner guarantee was weakened rather than enforced.
- Fix: use `roles=("movable", "fixed")` and validate the second member as a fixture/non-rigid body, or explicitly document and safely support a movable partner.

### 15. Optional default frames were not validated — fixed

- File: `dojo/skills/glib.py:272-300`
- Quote: `materialise()` inserts defaults before validation.
- File: `dojo/skills/glib.py:345-350`
- Quote: materialized frame names are checked against every owner.
- Severity: **bug**
- Fix/status: implemented.

### 16. Floating-point validation accepted NaN — fixed

- File: `dojo/skills/glib.py:282-283`
- Quote: `math.isfinite(float(v))`
- File: `dojo/skills/glib.py:352-356`
- Quote: numeric arguments must be finite before range checks.
- Severity: **bug**
- Fix/status: implemented.

### 17. `look_at` reported success when all looks failed — fixed

- File: `dojo/skills/general.py:256-258`
- Quote: `return {"ok": bool(read), ...}`
- File: `dojo/skills/glib.py:212-214`
- Quote: `Fails when no object could be read again.`
- Severity: **bug**
- Fix/status: implemented.

### 18. `Outcome` overstated postcondition certainty — fixed

- File: `dojo/skills/glib.py:16-18`
- Quote: distinguishes `verified` from `reported by the skill`.
- File: `dojo/skills/glib.py:485-489`
- Quote: a failed library postcondition turns success into failure; unverified outcomes say `reported by the skill`.
- Severity: **bug**
- Fix/status: implemented. Not every verb has a library postcondition, but the text no longer claims otherwise.

### 19. Occupancy used a fixed centre-radius heuristic — partly fixed

- File: `dojo/skills/general.py:22-39`
- Quote: centre must be inside the target’s oriented footprint and bottom within `REST_BAND`.
- File: `dojo/skills/glib.py:266-269`
- Quote: GLib delegates to the same shared predicate.
- Severity: **risky**
- Finding: the fixed radius and duplication are gone, and vertical relation was added. However, it still tests only the occupant’s centre, not oriented footprint overlap; a large object can overlap a support while its centre lies outside.
- Fix: perform 2-D oriented-footprint polygon overlap plus vertical/support relation, using one shared predicate.

### 20. Ported-handle test did not prove complete grounding isolation — partly fixed

- File: `dojo/skills/ported/__init__.py:67-87`
- Quote: recursive `has_handle()` and boundary `hide_instruction`.
- File: `tests/dojo/test_glib_audit.py:337-370`
- Quote: current vendored paths are spied and a no-handle negative control raises.
- Severity: **risky**, elevated to **blocks legality** as an audit claim
- Finding: current six GLib vendored paths do receive empty instruction text, and I found no current instruction leak. But `has_handle` means “any handle,” not “all semantic grounding is explicit”; dynamically loaded source/config/checker logic is excluded; and five current verbs never enter their implementation in the sentinel test.
- Fix: validate complete semantic handle coverage per mapping, audit the dynamically loaded subtree, and execute every implementation with valid fixtures.

## Final verdict on 1–5 and 20

- **1: fixed**
- **2: fixed**
- **3: fixed**
- **4: partly fixed**
- **5: partly fixed**
- **20: partly fixed**

The branch is improved substantially, but the committed legality audit still misses a concrete active checker-shaped path. I would not sign off the strict legality claim at `5a5a3837`.

Codex session ID: 01a1b72c-2acf-74a2-b9ed-250f1340ac43
Resume in Codex: codex resume 01a1b72c-2acf-74a2-b9ed-250f1340ac43
