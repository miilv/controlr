# controlr — architecture (v0)

A robotics harness in the spirit of coding-agent harnesses: a **turn loop** where a
vision LLM (via one OpenAI-compatible endpoint — omniroute) reads camera frames, the
robot's operating manual and the task, and replies with **low-level numeric actions**
plus a **status**. The harness executes them through a safety envelope, waits for the
robot to settle, and appends the new observation + feedback as the next user turn.
The transcript is **append-only** and **cached** (system prompt, all past images, all
past actions); only the newest turn is uncached.

Non-goals (by design): no high-level skills ("pick the box"), no VLA. The model is the
controller. Alternatives are config switches, not code forks.

## Turn loop

```
reset(task) -> obs0
[planner]  one separate call (default claude/claude-opus-5-5-xhigh; effort via id suffix; own
           timeout 600 s, 1 retry): planner system prompt + task + obs0 -> plan text (shares no
           cache entry with the control transcript; only its text enters turn 0 when
           planner.include_in_context). planner.plan_file pins a plan instead (sweeps).
turn 0     user: RUN <run id> (llm.request_nonce) + task instruction (+ plan) + observation(obs0)
loop:      assistant reply (streamed; early-stop once grammar complete)
           -> parse (MOVE*/STATUS) -> SafetyEnvelope.filter -> robot.execute (blocks until settled)
           -> goal check -> end? -> observe -> user: feedback + observation
end:       DONE (goal verified unless trust_done) | FAIL | max_turns | episode.max_stops STOP events
           (default 3; a STOP of kind "unstable" ends at once) | parse-error streak
           | goal reached (only with episode.end_on_goal); then one final observe (logged)
```

The system prompt is built after `reset`: `build_system_prompt(cfg, spec, cameras=obs.cameras,
state0=ref)` where `ref = robot.reference_state() or obs.state` (the COMMANDED reset pose, free of
settle jitter) — the cameras give per-image axis directions and pixel lengths, `ref` gives the fixed
tool orientation (rotation=none), the jaw line and the fingertip drop. The same `ref` is the
safety envelope's reference orientation (`SafetyEnvelope.reset(ref)`). Still task-free, so it is
identical across episodes of one config. `prompt.fewshot` (a .md file or a run dir) is appended as
"Appendix C" (cached prefix). The run id goes into turn 0 (not the system prompt) because omniroute replays cached
*responses* for byte-identical requests.

The robot never moves while the model thinks. Latency is a measured quantity
(~1 s/turn is the target, not a constraint).

## Modules and ownership (each file has one owner; interfaces below are the contract)

| Path | Responsibility |
|---|---|
| `controlr/types.py` | shared dataclasses (frozen contract — do not change without updating all users) |
| `controlr/config.py` | experiment config (YAML + `--set` overrides); every experiment axis is a field |
| `controlr/llm/client.py` | streaming OpenAI-compatible chat client with timings + normalised usage + early stop |
| `controlr/llm/caching.py` | cache-marker placement per model route (applied at serialisation, never stored) |
| `controlr/llm/transcript.py` | append-only transcript; images encoded once and stored as bytes |
| `controlr/protocol/grammar.py` | reply grammar: render the spec for the prompt, parse replies, completeness check |
| `controlr/protocol/feedback.py` | feedback text for the next user turn (exec receipt, clamps, limits, events, goal, state) |
| `controlr/prompts/*.md`, `controlr/prompts/builder.py` | system prompt (robot operating manual) + planner prompt templates |
| `controlr/observation/renderers.py` | observation -> list of image parts + text (resize, overlays, diff, heatmap, tile) |
| `controlr/robot/base.py` | Robot ABC (contract) |
| `controlr/robot/spec.py` | the one `RobotSpec` constructor of the UR3 CB3 rig (shared by mock + Isaac) |
| `controlr/robot/kinematics.py` | UR3 CB3 FK/IK (DH, controller base frame), backend-independent; the ONE rotation-helper implementation |
| `controlr/robot/safety.py` | SafetyEnvelope: filter/clamp actions, near-limit warnings |
| `controlr/robot/mock.py` | kinematic mock robot (no physics; synthetic rendering) for tests |
| `controlr/robot/replay.py` | replays recorded frames regardless of actions (latency/caching benchmarks) |
| `controlr/robot/isaac/` | Isaac Sim 6.0 backend: PHANTOM's calibrated UR3 CB3 + Robotiq + D435 scene (server inside Isaac's python, numpy-only RPC client in controlr; `motion.py` = trajectory timing shared by both) |
| `controlr/loop.py` | episode runner (planner + turn loop), end conditions |
| `controlr/runlog.py` | run directory writer |
| `controlr/bench/` | cache probe, latency matrix, sweep runner |
| `controlr/cli.py` | `controlr run | prompt | bench-cache | bench-latency | sweep | report | models` |

## Contracts

### LLM client (`controlr/llm/client.py`)
```python
@dataclass
class Usage:  # normalised across routes
    prompt_tokens: int; completion_tokens: int
    cache_read_tokens: int; cache_write_tokens: int; reasoning_tokens: int
    raw: dict
@dataclass
class Timings:  # seconds, perf_counter relative to request start
    ttft: float | None; t_complete: float | None; t_end: float
    ttft_any: float | None = None    # first content OR thinking byte (ttft includes thinking!)
    t_wall: float | None = None      # incl. retries/backoff
    t_headers: float | None = None   # response headers (upload + router queue)
@dataclass
class LLMResult:
    text: str; usage: Usage | None; timings: Timings
    stopped_early: bool; finish_reason: str | None
    error: str | None; http_status: int | None; attempts: int
    reasoning_text: str = ""; headers: dict = {}; request_bytes: int = 0
    truncated: bool = False          # early stop cut the STATUS note (loop stores it stripped)
class LLMClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: float, max_retries: int, *,
                 usage_grace_s: float = 0.0, headers: dict | None = None, ...): ...
    def complete(self, model: str, messages: list[dict], *, max_tokens: int,
                 temperature: float | None = None, extra_body: dict | None = None,
                 stop_when: Callable[[str], bool] | None = None,
                 timeout_s: float | None = None, max_retries: int | None = None) -> LLMResult: ...
```
* POST `{base_url}/chat/completions`, `stream: true`, `stream_options: {include_usage: true}`, httpx.
* Usage fields seen through omniroute: `prompt_tokens`, `completion_tokens`,
  `prompt_tokens_details.cached_tokens`, `cache_read_input_tokens`,
  `cache_creation_input_tokens`, `completion_tokens_details.reasoning_tokens` / `reasoning_tokens`.
* `stop_when(text_so_far)` true -> `stopped_early=True`; with `usage_grace_s` > 0 (default
  0.3 s) the client keeps reading for the usage chunk and lets the text grow only to the end of
  the STATUS line (so the stored note does not depend on chunking); otherwise / if the grace
  ends first, `truncated=True` and the loop stores the reply with the partial note stripped
  (`grammar.strip_partial_note`). `completion_tokens` then includes discarded tokens.
* Retries: connection errors, 429, 5xx with exponential backoff; 4xx other -> no retry.

### Caching (`controlr/llm/caching.py`)
```python
def cache_style_for(model: str, configured: str) -> str          # "anthropic" | "none"
def apply_cache_markers(messages: list[dict], style: str, ttl: str = "5m") -> list[dict]  # returns a deep copy
```
* `anthropic` style (all `claude/`, `cc/`, `no-think/claude*` routes):
  `cache_control: {"type":"ephemeral"}` (+`"ttl":"1h"` when ttl=1h) on (1) the last content
  part of the system message, (2) the last content part of the **last** user message, and
  (3) the last content part of the **second-to-last** user message. Max 4 markers total.
* Markers are added on a copy at request time; the stored transcript never contains markers,
  so earlier messages stay byte-identical apart from the moving marker (unit-tested).
* **What sets the cache boundary through omniroute is NOT our markers** (review 2026-10-02, all
  live runs): every Claude route is served by omniroute's `cc` provider, which places its own
  breakpoints. The cached prefix ends right after the newest *assistant* reply (never marked by
  us): write_N = uncached_{N-1} + reply_{N-1}; the newest user turn (~300-430 tokens) is the
  uncached part; unmarked planner calls also cache their system prompt. `llm.cache` and
  `llm.cache_ttl` are therefore probably no-ops on these routes — unverified until the ~6-call
  probe (`bench-cache` with `--set llm.cache=none`, and with `cache_ttl=1h` + 6 min pause) is run
  with approved spend; do not sweep these axes on `cc` routes before that.
* Requests can land on different upstream accounts with separate caches (run 105745: turn 4 read
  0 and rewrote 5250 tokens while turn 3's entry was alive). The run log therefore keeps a per-turn
  cache trace and `summary.json` lists `cache_regressions` (turns that read less than the previous
  call read + wrote, minus 64 tokens); the CLI warns. `llm.extra_headers` can carry a router
  lease header (`X-OmniRoute-Lease-Owner` is advertised; effect unverified).
* Anthropic's 20-block lookback: the renderer warns (`setup.json` `lookback_warning`) when a turn
  can add > 18 content blocks (many cameras x derived images without `tile`).
* OpenAI / Gemini / others: automatic prefix caching -> `none` (no markers; prefix stability is
  still what matters).

### Transcript (`controlr/llm/transcript.py`)
```python
@dataclass(frozen=True)
class ImagePart: jpeg: bytes; sha: str; width: int; height: int; label: str  # data URL computed once
class Transcript:
    def __init__(self, system_text: str): ...
    def add_user(self, parts: list[str | ImagePart]) -> None
    def add_assistant(self, text: str) -> None
    def to_messages(self) -> list[dict]          # OpenAI chat format, images as data URLs, NO markers
    def to_log_records(self) -> list[dict]       # images replaced by {"image_sha": ...}
    def n_images(self) -> int
def encode_image(rgb: np.ndarray, quality: int, label: str) -> ImagePart  # JPEG once, deterministic
```
* Append-only: no method mutates or removes earlier messages. Assistant text is stored exactly
  as received (including early-stopped truncation).
* `to_messages()` must be byte-stable for the prefix: calling it after adding turn N+1 yields
  messages[:k] identical (json.dumps) to the previous call's messages[:k] (unit-tested).

### Reply grammar (`controlr/protocol/grammar.py`)
Text format (`action.format=text`, the only implemented format; other values are rejected by
`config.validate`):
```
MOVE <mode> v1 v2 ... [GRIP open|close|<mm>]     # 0..max_chunk lines; mode must equal the configured mode
GRIP open|close|<mm>                             # gripper-only line (counts as one action)
HOLD                                             # explicit no-op
STATUS OK|DONE|FAIL|STUCK|LIMIT [short note]     # exactly one, last line; ends the reply
```
* Units in the reply are the configured LLM units (default mm, deg). Values per mode:
  `ee_delta`/`ee_abs`: `x y z` (rotation=none), `x y z yaw` (rotation=yaw; for ee_abs an
  absolute heading — the tool keeps the reference tilt, SI `Action.values` has 4 entries),
  `x y z roll pitch yaw` (rotation=full; extrinsic about base x/y/z);
  `joint_delta`/`joint_abs`: one value per joint, in joint order.
* Tolerant parsing: ignore markdown fences, leading/trailing prose, a trailing unit word, Unicode
  minus; collect errors (wrong mode, wrong arity, non-numeric, > max_chunk lines, missing STATUS)
  for feedback. A line is a command when its keyword is UPPERCASE; a mixed-case keyword line
  ("Move the gripper left", "Status: ok so far.") counts only if it parses cleanly (STATUS: only
  as the last line), otherwise it is prose — no error. `is_complete` fires on uppercase STATUS.
```python
def grammar_spec(cfg: ActionConfig, spec: RobotSpec) -> str          # text block for the system prompt
def parse_reply(text: str, cfg: ActionConfig, spec: RobotSpec) -> ParsedReply   # SI units out
def is_complete(text: str) -> bool                                   # a STATUS line has been emitted
```

### Feedback (`controlr/protocol/feedback.py`)
```python
def format_feedback(turn: int, parsed: ParsedReply | None, report: ExecReport | None,
                    goal: GoalReport | None, obs: Observation, cfg: Config,
                    spec: RobotSpec | None = None) -> str
def to_llm_units(text: str, a: ActionConfig) -> str   # "<n> mm"/"<n> deg" -> configured units
```
Safety and backend messages (`SafetyEvent.message`, `GoalReport.message`) are written in
canonical mm / deg; `format_feedback` converts them, so `pos_unit=cm` never mixes units.
Compact, line-oriented, LLM units, e.g.
```
TURN 7
EXEC: MOVE ee_delta 20 0 -10 -> achieved dx=19.6 dy=0.2 dz=-9.8 mm (0.41 s)
CLAMP: z target 3 mm -> 20 mm (table clearance)
WARN: wrist_2 at 171 deg, soft limit 175 deg
EVENT: contact finger-object
GOAL: not reached (shown only per episode.goal_feedback)
STATE: tcp x=312 y=-45 z=88 mm yaw=12 deg | grip 42 mm open | holding: no
```
Parse errors are reported as `PARSE ERROR: ...` + one-line reminder of the grammar. With
`observation.state_text=false` there is no STATE line and the manual never mentions one; the
EXEC line (achieved motion), CLAMP texts and the `ee_marker` label still carry some state.

### Observation renderers (`controlr/observation/renderers.py`)
```python
@dataclass
class RenderedObservation: images: list[ImagePart]; text: str
class ObservationRenderer:
    def __init__(self, cfg: ObservationConfig, action_cfg: ActionConfig | None = None,
                 spec: RobotSpec | None = None, encoder=None): ...
    def render(self, obs: Observation, prev: Observation | None, turn: int) -> RenderedObservation
def describe_axes(info, origin, long_edge=None, per="100 mm") -> str  # direction + px per 100 mm
```
Renderers (applied per camera, in configured order): `raw`; `grid` (base-frame XY grid on the
table plane projected via CameraInfo, labelled in LLM units); `axes` (base x/y/z arrows at the
TCP); `ee_marker` (projected TCP + gripper footprint); `diff` (extra image: |I_t - I_{t-1}|
amplified); `heatmap` (extra image: colour heat of change magnitude over the current frame);
`tile` (all images of the turn in one grid image, labelled). Resize to `size` long edge with
LANCZOS, never upscale. Deterministic output for identical input (cache stability).

### Robot backends
`controlr/robot/base.py::Robot` (reset/observe/state/execute/check_goal/close; optional
`reference_state()` — the commanded reset state, default None).
Contract additions in `types.py` (backwards compatible, defaults None): `RobotSpec.finger_pad`
(pad half length / half width / thickness, m), `Action.q_path` (envelope waypoints); `Action.values`
of ee_abs + rotation=yaw has 4 entries (x, y, z, yaw).
Factory: `controlr.robot.make_robot(cfg: Config) -> Robot`.
* `kinematics.py`: `fk(q) -> (pos, rotvec)` of the TCP in the UR controller base frame
  (DH from PHANTOM `phantom/sim/kinematics.py`: a=[0,-0.24365,-0.21325,0,0,0],
  d=[0.1519,0,0,0.11235,0.08535,0.0819], alpha=[pi/2,0,0,pi/2,-pi/2,0]; TCP offset +z 0.18 m
  for Robotiq 2F-85 fingertip centre — configurable), `ik(pos, rotvec, q_seed) -> q | None`
  (damped least squares, joint-limit aware, nearest to seed).
* `safety.py`: `SafetyEnvelope(spec, cfg).reset(state0)`, `.filter(actions, state) -> (actions_out,
  events)`: per-line step limits (TCP travel also in joint modes), workspace box, table clearance
  of the LOWEST FINGERTIP (`RobotSpec.finger_pad`; a tilted open gripper reaches 15-45 mm below
  its TCP), joint soft limits, near-limit warnings, clamp vs reject. ee moves: IK every
  `safety.path_step_m` (5 mm) along the straight TCP line -> `Action.q_path` waypoints; an
  unreachable or near-singular stretch (joint jump > 0.02 rad/mm + 0.05) SHORTENS the move
  (CLAMP "moved N % of the way"; < 10 % -> skipped, `ik_fail`). rotation=none keeps the
  reference orientation from `reset` (a contact tilt is undone; WARN > 3 deg). Joint moves are
  checked at samples along the joint path. Backend-independent.
* Isaac backend (primary sim): reuses PHANTOM's Isaac Sim 6.0 reconstruction of the real rig
  (`~/phantom-icra-2027/phantom` on compute3: `phantom/sim/scene.py`, `camera.py`, `kinematics.py`,
  native gripper, calibrated D435 640x480, waffle packet + box; carton/egg task scenes). A server
  process runs inside Isaac's python (`isaac-sim-6.0/python.sh`, headless) and exposes
  reset/observe/state/execute/check_goal over `multiprocessing.connection` with numpy-only payloads
  (same pattern as PHANTOM `phantom/sim/remote_policy.py`). `controlr.robot.isaac.IsaacRobot` is the
  client. Physics is paused between turns; `execute` passes each action's waypoints in ONE
  min-jerk motion timed so PEAK speeds respect the TCP/joint limits (`motion.py`), samples
  contacts every step during motion, stops on table/box and robot-object force (baselines per
  contact pair), freezes the target at any stop (also during settling), then settles and the D435
  view is rendered. Reach targets are sampled feasible (IK, fingertip floor, tool body clear of the
  box). One camera (`scene`); the mock adds an optional synthetic `top` camera. Tasks: the PHANTOM waffle pick-to-box first, then simple reach/push variants.
  The mock robot covers GPU-free unit tests.

### Run log (`controlr/runlog.py`)
`runs/<UTC timestamp>_[fake_]<name>/`: `config.yaml`, `system_prompt.md`, `plan.md`,
`images/<sha>.jpg` (exact bytes sent), `messages.jsonl` (transcript with image refs),
`turns.jsonl`, `summary.json`, `setup.json` (reset time, instruction, cache style, `llm_backend`
live|fake, state0 + reference state, turn-0 image shas, lookback warning), `cameras.json` (K,
T_cam_base), `spec.json` (RobotSpec), `planner.json` (planner request/usage/timings/reasoning),
`raw/<turn>_<cam>.png` (native frames without overlays, `log.save_raw_frames`).

`turns.jsonl`, one record per turn — enough to rebuild (obs_t, action_t, obs_t+1):
`obs_images` (shas the model saw), `state_before`, `reply` (stored, note-stripped if truncated;
`reply_streamed` when it differs), `reasoning`/`reasoning_chars`, `actions` / `executed` (SI values,
`q_target`, `q_path`), `events`, `goal`, `next_obs_images` + `state` (after the action; also after
the terminal turn), `feedback` (absent on the terminal turn), `llm` (all Timings fields), `usage`,
`request_bytes`, allow-listed `headers` (x-omniroute-*, request ids, rate limits), `timings`.

`summary.json`: `outcome`, `success` (goal check at the end), `success_verified` (outcome ==
success: DONE claimed AND verified — aggregate this), `llm_backend`, token totals,
`cache_read_share` (episode) + `cache_read_share_turns`, `cache_regressions`, latency percentiles
with `n`/`missing` counts (`llm_ttft_s` = first content incl. thinking, `llm_ttft_any_s`,
`llm_headers_s`, `llm_complete_s`, `llm_end_s`, `llm_wall_s`, `exec_s`, `cycle_s`), and a pointer
to `planner.json`. `controlr report` skips fake runs unless `--include-fake`.
Later: export to LeRobotDataset for the action head.

### Config validation
`load_config` / `run_episode` call `config.validate`: enum-like values (`action.mode/rotation/
pos_unit/ang_unit/gripper/format`, `episode.goal_feedback`, `llm.cache/cache_ttl`,
`robot.backend`, renderer names) and basic ranges are checked — a typo raises instead of silently
meaning something else. `.env` is read from the CWD, then from the repo root.

### Sweeps
`controlr sweep FILE [--fake-llm] [--resume SWEEP_DIR] [--set ...]`: labels use full keys; `--set`
on a grid key is rejected; finished points are skipped on resume; exit code 1 if any point ended
in error/llm_error/interrupted. Pin `planner.plan_file` for control-side ablations.

## Deployment
Code runs on **compute3** (`physicalai`, RTX 5090, Ubuntu 24.04, py3.12, Isaac Sim 6.0) under
`~/controlr` with a uv venv; the Isaac server runs under Isaac's own python. `scripts/deploy.sh` rsyncs the repo (no
`.env` in git — copied separately).
