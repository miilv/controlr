# Review: contracts, tests, UX (v0, 2026-10-02)

Lens: whether the architecture contract holds together, dead or duplicated code, error handling, test
quality, CLI ergonomics for experiments and sweeps, config coverage of the experiment axes, run-log
completeness for the future action head, and the quality of `system_v0.md` as the model reads it.

How findings were checked: by reading the code, running `uv run pytest -q` (246 passed, 12 skipped;
isaac and live tests skip correctly without `CONTROLR_ISAAC=1` / `CONTROLR_LIVE=1`), small local scripts,
a mock + `--fake-llm` run with `action.pos_unit=cm` (written to /tmp), and the existing live run dirs
`runs/20261002T131109Z_sim_waffle` and `runs/20261002T130156Z_sim_waffle`. No live LLM calls.

Severity: **blocker** = an experiment axis silently does nothing, or results are wrong without any
warning. **major** = misleads the model or the experimenter, or makes a planned use impossible.
**minor** = cleanup.

---

## Blockers

### 1. `prompt.fewshot` is a dead config field — the "few-shot demos" axis is a silent no-op (blocker)
- **Where:** `controlr/config.py:121`; `controlr/prompts/builder.py:18,113`.
- **Problem:** `PromptConfig.fewshot` ("path to a recorded run / demo file injected into the cached
  prefix") is never read. The only mentions in the code are a docstring and the `cache_warning` text,
  and that warning actively tells the user to set it.
- **Evidence:** `grep -rn fewshot controlr` finds only `builder.py:18` (docstring) and `:113` (the
  warning string). A sweep `grid: {prompt.fewshot: [null, demos/a.md]}` produces two identical arms
  with different labels.
- **Fix:**
  - Implement it: load a markdown file, or a run dir's `messages.jsonl` rendered as text, and append it
    as `{fewshot}` to the system template, so it lands in the cached prefix.
  - Until then, `_build`/`load_config` should raise when `fewshot` is not None, and `cache_warning` should
    stop recommending it.
  - Add a test that the demo text appears in the system prompt.

---

## Majors

### 2. The STATE example and the `ee_abs` grammar examples still use x = +300 mm; the workspace is at negative x (major)
- **Where:**
  - `controlr/prompts/builder.py:223` (`_example_state(... pos=(0.30, -0.05, 0.15))`), used for
    `{state_example}` at `builder.py:440`.
  - `controlr/protocol/grammar.py:135-137` (`ee_abs` examples `[0.30, -0.05, 0.15]` etc.).
  - `tests/test_prompts.py:30,48`.
- **Problem:** Integration fix 7 moved the worked example and Appendix B to the workspace centre, but two
  examples were missed:
  - Section 4 of the live manual still shows `STATE: tcp x=300 y=-50 z=150 mm`.
  - With `action.mode=ee_abs`, the grammar examples are `MOVE ee_abs 300 -50 150`, which is outside the
    workspace `x -550..-150`.
  - In the same prompt, Appendix B says `x=-350`. The model gets contradictory signs for the axis it
    already confuses (depth/sign errors are the main failure in the integration notes).
- **Why the tests miss it:** the test fixture `SPEC` has `workspace_lo=(0.10, ...)` (positive x, not the
  real rig), and `test_default_manual_contents_and_size` asserts the needle `"STATE: tcp x=300 y=-50 z=150 mm"`.
  The test enshrines the bug.
- **Evidence:** `runs/20261002T131109Z_sim_waffle/system_prompt.md`, section 4. A local render with
  `mode=ee_abs` printed `MOVE ee_abs 300 -50 150 / MOVE ee_abs 300 -50 80 GRIP open`.
- **Fix:**
  - Build `_example_state` and the `ee_abs`/`ee_delta` example values from `_example_xy(spec)` and
    `_table_top(spec)`. Pass `spec` into `_example_values`.
  - Change the test fixture to `ur3_cb3_spec()`.
  - Add a test: every example position in the manual (STATE example, grammar examples, worked example,
    Appendix B) parses and lies inside `[workspace_lo, workspace_hi]`.

### 3. The camera-axis sentence cannot tell +y from +z: "+x points right, +y points up, +z points up in the image" (major)
- **Where:** `controlr/observation/renderers.py:150-175` (`describe_axes`, 8-way direction words only);
  `builder.py:158-171`.
- **Problem:** On this rig the camera looks down about 25° from vertical, so +y (away from the camera) and
  +z (up) both project "up" in the image. They differ only in magnitude, and magnitude is dropped. The one
  calibrated, generated sentence meant to remove axis guesswork is ambiguous on exactly the axis
  (depth, y) that the integration notes call the dominant failure (80–100 mm depth misjudgements).
- **Evidence:** the line in the live `system_prompt.md`: "`scene`: +x points right, +y points up, +z
  points up in the image."
- **Fix:**
  - Report the projected length per 100 mm at the workspace centre, e.g.
    "+y: up the image, ~95 px per 100 mm; +z: up, ~30 px per 100 mm (448-px image)".
  - Add one sentence: "a pure +z move and a pure +y move both shift the gripper up the image; use the
    STATE line or the drop line to tell them apart."
  - Test it with a tilted camera, not only with `top_down_cam()`.

### 4. The table clearance protects only the TCP, but on this rig the open lower fingertip sits ~14 mm below it; the manual and Appendix B teach descending to the floor (major)
- **Where:**
  - `controlr/robot/safety.py:76-78` (`z_floor = table_z + table_clearance_m`, applied to the TCP).
  - `system_v0.md` §8 ("keeps the TCP ... at least 5 mm above the table").
  - `builder.py:254-256,275,280` (Appendix B: a 60 mm descent at 45 mm height, clamped to the floor).
- **Problem:** With rotation=none, the waffle start orientation tilts the jaw line to
  `(-0.27, +0.91, -0.33)`. With the gripper open (88 mm), one fingertip is 0.326 × 44 mm ≈ 14 mm lower
  than the TCP. At the envelope floor (TCP z = −10 + 5 = −5 mm), that fingertip is ~9 mm inside the mat.
  The result is a contact-force STOP, or the physics blow-ups the notes report.
  - The manual presents the clearance as a guarantee: "at least 5 mm above the table".
  - Appendix B (the only demonstration the model gets) shows a deliberate 60 mm blind descent into the
    clamp. That contradicts §8 ("plan around it") and §10 ("10 mm or less within a few centimetres of
    contact").
- **Evidence:** `rotvec_to_matrix(state0.tcp_rotvec)` from `runs/20261002T131109Z_sim_waffle/setup.json`
  gives R[2,0] = −0.326, so the fingertip z offsets are ±14.4 mm.
- **Fix:**
  - In `SafetyEnvelope`, check the lowest point of the gripper footprint against the floor
    (TCP ± half-opening × jaw axis, plus pad half-thickness), not just the TCP.
  - State the offset in `_tool_doc`: "with the gripper open, the lower fingertip is N mm below the TCP".
  - Rewrite Appendix B so it descends in steps that stop above the floor, and shows the clamp as a
    mistake rather than a technique.

### 5. Prose that starts with a keyword is parsed as a command; prose starting with "Status" stops the stream before the real MOVE (major)
- **Where:** `controlr/protocol/grammar.py:285-288` (`_STATUS_LINE_RE`, case-insensitive, start of
  line), `:401-417` (any line whose first token is MOVE/GRIP/HOLD/STATUS in any case is a command).
- **Problem:** Haiku writes 1–3 sentences of prose before the MOVE line (integration notes).
  - "Move the gripper left toward the packet." → `PARSE ERROR: mode 'the' is not available` plus a
    grammar reminder, every time.
  - "Hold on, …" and "Grip looks fine." → the same spurious errors.
  - "Status: ok so far. Next I'll descend." makes `is_complete` true, so the client closes the stream
    there. The real `MOVE` never arrives, the reply is parsed as `STATUS OK` with no action, and the turn
    is wasted.
- **Evidence:** a local script printed:
  - `parse_reply("Status: ok so far. Next I'll descend.\nMOVE ee_delta 0 0 -10\nSTATUS OK")` → actions `[]`,
    status OK, error "ignored 1 line(s) after STATUS".
  - `is_complete("Status: ok so far. ...")` → `True`.
- **Fix:**
  - Require uppercase keywords, or require a well-formed line: `STATUS` followed by a status word, then
    end of line or a short note with no sentence punctuation; `MOVE` followed by the mode or a number.
  - Lines that start with a keyword in mixed case and fail to parse should be treated as prose, not
    errors.
  - Add regression tests built from real Haiku replies in `runs/*/turns.jsonl`.

### 6. Unit leak: safety, goal and backend messages are always mm/deg, whatever `action.pos_unit`/`ang_unit` say (major)
- **Where:** `controlr/robot/safety.py:47-52` (`_mm`, `_deg` hard-coded); goal messages in
  `robot/mock.py` and `robot/isaac/*` (e.g. "TCP is 155 mm from the target"). The contract in
  `types.SafetyEvent.message` says "already in LLM units where relevant".
- **Problem:** The manual says "Never mix in other units". With `pos_unit=cm` the feedback mixes:
  ```
  EXEC: MOVE ee_delta 20.0 0.0 0.0 -> achieved dx=10.00 dy=0.00 dz=0.00 cm
  CLAMP: translation 200 mm exceeds the per-line limit 100 mm -> scaled to 100 mm
  GOAL: not reached (progress 0%) - TCP is 155 mm from the target
  ```
  (verified with `controlr run -c configs/mock.yaml --fake-llm --set action.pos_unit=cm`). This confounds
  the action-parameterisation axis.
- **Fix:** Give `SafetyEnvelope` the `ActionConfig`, or a formatter callable, and format with
  `feedback._p` / `_ang`. Backends should put numbers in `GoalReport.metrics` / event fields and let
  `feedback.py` format them. Add a test that no `mm`/`deg` appears in the feedback when the units are
  cm/rad.

### 7. `observation.state_text=false` leaves a manual that contradicts itself, and the ablation is not clean (major)
- **Where:** `builder.py:204-207` vs `system_v0.md:9,20,46,56-58,130,135`, the feedback block at
  `system_v0.md` §6 (`STATE: <measured state>` is listed unconditionally), and `builder.py:383`
  (worked example "STATE says …").
- **Problem:** With state_text off, the manual says "No STATE line is given in this configuration" and
  then, in the next paragraph, "The STATE line reports … e.g. STATE: tcp x=300 …". It goes on to tell the
  model to check the STATE line every turn, and lists STATE in the feedback format.
  - Proprioception also still reaches the model: EXEC "achieved dx=…", CLAMP/WARN text with absolute
    targets, and the `ee_marker` label "TCP z=…".
  - So the "state text" axis measures a self-contradictory prompt, not the absence of state.
- **Evidence:** a local render with `state_text=False` printed STATE references on manual lines
  9, 19, 48, 52, 61-63, 113, 165, 170, 188, 191.
- **Fix:**
  - Make the STATE paragraphs, the §6 line and the worked example conditional on `state_text`, and
    reword §1/§3/§10.
  - Document which other channels carry state.
  - Optionally add `observation.state_text: none|full|no_holding`, because `holding` is privileged sim
    information.

### 8. Enum-like config fields are not validated; typos and unimplemented values silently fall back (major)
- **Where:** `config.py:101` (`gripper`: any value other than `"binary"` acts as width mode),
  `config.py:102` (`format: tool` is never read), `config.py:129` (`goal_feedback`: an unknown value
  silently means "never" in `feedback._show_goal` and `builder._values`).
- **Problem:** The config module's own rule is "typos in experiment configs must not be silently ignored".
  Unknown keys raise, but unknown values do not.
- **Evidence:** `Config()` with `action.format="tool"`, `episode.goal_feedback="alwyas"`,
  `action.gripper="widht"` builds a manual with no error. It has no GOAL line and uses the width grammar.
- **Fix:** Add a `validate(cfg)` called from `load_config` with allowed sets for:
  - `mode`, `rotation`, `pos_unit`, `ang_unit`, `gripper`, `format` (only `text` until tool calls
    exist), `goal_feedback`, `cache`, `cache_ttl`, `backend`, `renderers`;
  - plus range checks (`max_chunk>=1`, `size>0`).

  Add one test per field.

### 9. The "multi-camera: tile vs separate" axis cannot be run on any backend (major)
- **Where:** `robot/isaac/client.py:156`, `robot/mock.py:160`, `robot/replay.py:149` — each returns
  exactly one camera; the Isaac server has one `Camera` (`server.py:126`).
- **Problem:** `observation.cameras` and `observation.tile` exist, but with one camera, `tile` only ever
  combines the frame with its own diff/heatmap. Ilia listed this axis. The integration notes name a
  second or top view as the next fix for depth errors.
- **Fix:**
  - Add a second calibrated camera to the Isaac server: a top-down or side view, with `CameraInfo` from
    its pose. Add a `cameras` list param to the mock.
  - Until then, `load_config` should fail loudly when more than one camera is requested from a
    single-camera backend (today it fails late with a `KeyError` after reset). README/config comments
    should state the limitation.

### 10. The run log is not yet sufficient for training an action head (major)
- **Where:** `loop.py:126-130` (`_action_rec` drops `q_target`), `:275-277` (`setup.json`), `:360-367`
  (terminal turn), `:383` (`rec["images"]`), `runlog.py`.
- **Problem:**
  - (a) Only the resized (448 px), q90 JPEG with overlays is saved. With `grid`/`ee_marker` on, no clean
    frame exists, and native-resolution frames are never stored.
  - (b) Camera calibration (`CameraInfo.K`, `T_cam_base`) and the `RobotSpec` (workspace, tcp_offset,
    table_z) are not logged anywhere, so overlays and projections cannot be reproduced offline.
  - (c) Per turn, the log lacks `state_before`, the resolved joint target `q_target` from the envelope,
    and the executed SI joint trajectory. `rec["images"]` holds the images after the action (the input to
    the *next* turn). Turn 0's images appear in no turn record (only in `messages.jsonl`). No observation
    is captured after the terminal turn, so a success has no final frame.
  - (d) `LLMResult.reasoning_text` is dropped. For thinking routes, this is the only record of why the
    model acted.
- **Fix:**
  - Optional `log.save_raw_frames` (PNG/npz at native resolution, per turn).
  - Write `cameras.json` and `spec.json` at setup.
  - Per-turn fields `obs_images` (input) and `next_obs_images`, `state_before`, `q_target`, `reasoning`.
  - One final observe + log on termination.
  - A test asserting that every (obs_t, action_t, obs_t+1) triple can be reconstructed from
    `turns.jsonl` alone.

### 11. `--fake-llm` runs are indistinguishable from live runs in `summary.json`, `config.yaml` and `report` (major)
- **Where:** `cli.py:534`, `loop.py:416-417`.
- **Problem:** A dry run writes the real `llm.model` id. FakeLLM's synthetic token and cache numbers
  appear as `totals` / `cache_read_share`, in a dir named `<stamp>_<name>` like a live run.
  - The only marker is `usage.raw.fake` inside `turns.jsonl`, and with early stop on, usage is `None`, so
    even that is absent.
  - `controlr report runs/*` therefore mixes dry runs into model comparisons. The local `runs/` already
    contains `*_fake_*` dirs, distinguishable only by hand-chosen names.
- **Evidence:** in the fake run in /tmp, `summary.json` = `{'model': 'claude/claude-sonnet-5', ...}`, and
  the string "fake" occurs nowhere in summary, config or setup.
- **Fix:**
  - Record `llm_backend: fake|live` (and the fake script path) in setup and summary.
  - Prefix the run dir name with `fake_`.
  - Make `report` show the column and skip fake runs unless `--include-fake` is given.

### 12. Sweeps cannot be smoke-tested offline or resumed, always exit 0, and re-run the planner at every point (major)
- **Where:** `controlr/bench/sweep.py:103-138`, `cli.py:163-173`.
- **Problem:**
  - There is no `--fake-llm` on `sweep`, so a 48-point Isaac sweep cannot be checked end to end without
    spending.
  - There is no `--resume`: a crash at point 37 restarts from 0, and Isaac with an Opus planner costs
    ~80 s of planner plus minutes of turns per point.
  - `cmd_sweep` returns 0 even if every point errored.
  - The planner (77–90 s, nondeterministic) is re-run at every point. With the planner on, each
    control-side ablation (renderers, size, chunking) is confounded by a different plan per point.
- **Fix:**
  - Add `--fake-llm` (shared with `run`).
  - Add `--resume <sweep dir>`, which skips indices already present in `aggregate.csv` with a terminal
    outcome.
  - Exit non-zero if any point has outcome `error`/`llm_error`.
  - Add `planner.plan_file` or a plan cache keyed by (task, seed, planner model, manual hash), so control
    axes can be compared on identical plans.

---

## Minors

13. **`success` and `outcome` can disagree** (`loop.py:41-50,414`).
    - `success` is the final goal check, whatever the model claimed. So `outcome=max_turns, success=True`
      and `outcome=fail, success=True` are both possible. The header comment still says `safety_stop` is
      "a STOP-level safety event", which predates `max_stops`.
    - There is no `episode.end_on_goal`: a reached goal without DONE burns the remaining turns.
    - Fix: document both fields in ARCHITECTURE, rename to `goal_success`, add `end_on_goal`.

14. **Dead or duplicated code.**
    - `builder.build_planner_messages` / `_image_url` (`builder.py:507-530`) is unused by the loop, which
      builds via `Transcript`. It ignores `state0`, so its output differs from what is actually sent
      (no tool doc).
    - `rotvec_to_matrix` / `rpy_to_matrix` / `matrix_to_rpy` exist twice (`protocol/feedback.py:63-95` and
      `robot/kinematics.py:48-95`). I checked numerically that they agree to 1e-16, but they will drift.
    - `make_robot`'s getattr fallback for `IsaacRobot` is defensive dead code.
    - `cli.report` imports `write_csv` from `bench.cache_probe`.
    - Fix: delete or reuse; keep one rotation module.

15. **Config defaults drift.**
    - `LLMConfig.usage_grace_s=0.0` (`config.py:43`) but `base.yaml:19` sets 0.3.
    - `base.yaml` claims to hold the shared defaults but omits `episode.max_stops`.
    - The manual's joint-table "home" column shows `START_Q` (pan 14°). sim_waffle actually resets to
      `task.params.start_q` (pan 10°), because `IsaacRobot` builds the spec from `params.home_q or START_Q`.
    - Fix: keep a single source for defaults, and set `spec.home_q` from the reset pose.

16. **The grammar examples tell a different story from §7** (`grammar.py:259-274`).
    - "cube" (this scene has a waffle packet), and `HOLD / STATUS DONE cube is inside the box` straight
      after `GRIP close`, with no release and lift. §7 says to release and lift before DONE.
    - In Appendix B, `GRIP open` is appended to a descent while the gripper is already open.
    - Fix: use neutral object words and end the example sequence with open, lift, DONE.

17. **The tool doc says "The fingers, the gripper body and the wrist lie BEHIND the TCP"** (`builder.py:413-417`).
    - The fingertips straddle the TCP along the jaw line. Only the finger bases, housing and wrist are
      behind it.
    - Fix: "the gripper housing and wrist lie behind the TCP along …; the fingertips are ±w/2 along the
      jaw line" (see 4).

18. **Turn logs are noisy** (`loop.py:114-123,416`).
    - `turns.jsonl` stores every response header per turn, including CSP and allow-headers (~2 KB of
      noise per turn).
    - `summary.json` embeds the whole planner record, including the full manual text.
    - Fix: keep an allow-list (request ids, `x-omniroute-*`, rate-limit headers) and put only a pointer to
      `planner.json` in the summary.

19. **Notes cut by early stop stay in the transcript** (e.g. "STATUS OK moving above est",
    "descending to check al" in the live waffle run).
    - The model re-reads its own mid-word truncations every turn.
    - Fix: during `usage_grace_s`, freeze the text at the first newline after STATUS, or at end of
      stream, instead of at the status word. Alternatively, drop the note from the stored reply
      (deterministic, so the cache is unaffected).

20. **The parser rejects some harmless formatting** (`grammar.py:312`): a trailing unit as a separate token
    (`MOVE ee_delta 0 0 -10 mm` → arity error) and a Unicode minus (`−20`). Fix: drop a final token equal
    to the configured unit, and map U+2212 to `-`.

21. **Sweep labels and CLI overrides interfere** (`sweep.py:52-56,122`).
    - Labels use only the last key segment, so `llm.model` and `planner.model` both become `model=`.
    - CLI `--set` is applied after the grid overrides, so `--set` on a grid key silently collapses the
      grid to one value.
    - Fix: use the full key in labels, and reject `--set` keys that are also grid keys.

22. **Error ordering and environment** (`loop.py:231-235`, `config.py` `load_dotenv(".env")`).
    - The robot is built (Isaac launch, ~15 s) before the API-key check.
    - `.env` is read relative to the CWD, so running from another dir silently loses it. An unexpanded
      `${OMNIROUTE_BASE_URL}` then reaches httpx as a TransportError and is retried with backoff.
    - Fix: validate the key and URL before `make_robot`; resolve `.env` relative to the repo/config.

23. **`report` cannot separate experiment arms** (`cli.py:603-625`): it has no name, seed, task, backend
    or fake column.

24. **`types.Action` docstring contradicts itself** (`types.py`): "rotvec-style rx ry rz" and then
    "extrinsic roll/pitch/yaw". The protocol uses RPY. Fix the docstring (this is a contract file).

25. **The loop-level prefix-stability test is nearly vacuous** (`tests/test_loop.py:73-80`).
    - It compares only `messages[:1]` (the system message) after a string-replace that leaves marker
      remnants.
    - I checked with a script that the full prefix, with markers removed, is byte-identical across 5
      turns with grid, ee_marker and diff on, so the property holds today. The test would not catch a
      regression in `_user_parts` ordering or a mutation of old turns.
    - Fix: strip `cache_control` structurally and assert `b[:n] == a` for every call pair, with overlays
      and diff enabled.

26. **The CLI is silent during the 77–90 s planner call**, and there is no way to preview the rendered
    system prompt without a backend reset (`cli.py` `_progress`).
    - Fix: print "planner … (model)" before and after the call. Add `controlr prompt -c cfg`, which
      renders with the mock or a recorded `setup.json` (state0 + cameras).

27. **ARCHITECTURE drift.** The module table omits `controlr/robot/spec.py`. The `ObservationRenderer` /
    `format_feedback` signatures in the contract differ from the code (the extra params are documented
    only in the docstrings). `turns.jsonl` fields such as `headers`, `images` and `feedback` are not in
    the contract.
