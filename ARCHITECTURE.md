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
turn 0     user: RUN <run id> (llm.request_nonce) + TASK: instruction (+ plan) + observation(obs0)
loop:      assistant reply (streamed; early-stop once grammar complete)
           -> parse (MOVE*/STATUS) -> SafetyEnvelope.set_obstacles(robot.obstacles()) (current
           pose) -> SafetyEnvelope.filter -> robot.execute (blocks until settled)
           -> goal check -> end? -> observe -> user: feedback + observation
           (llm.overlap_tail, default on: the call runs on a worker thread; the loop parses and
           executes the reply as it stands at the STATUS word while the client reads the rest of
           the STATUS note and the usage chunk; the transcript gets the call's final reply, so it
           is identical to the non-overlapped path. Turn record: timings.exec_overlapped,
           timings.llm_tail = call time left after execution; overlap_mismatch if the final
           reply's actions ever differed from the executed ones)
end:       DONE (goal verified unless trust_done) | FAIL | max_turns | episode.max_stops STOP events
           (default 3; a STOP of kind "unstable" ends at once) | parse-error streak
           | goal reached (only with episode.end_on_goal) | the backend ended the episode
           (`Robot.episode_over()`, outcome env_end: RoboDojo's step limit / success latch);
           then one final observe (logged)
```

The system prompt is built after `reset`: `build_system_prompt(cfg, spec, cameras=obs.cameras,
state0=ref, obstacles=robot.obstacles())` where `ref = robot.reference_state() or obs.state` (the COMMANDED reset pose, free of
settle jitter) — the cameras give per-image axis directions and pixel lengths, `ref` gives the fixed
tool orientation (rotation=none), the jaw line and the fingertip drop. The same `ref` is the
safety envelope's reference orientation (`SafetyEnvelope.reset(ref)`). `obstacles` (optional
`Robot.obstacles()`: Isaac's blue box at its reset pose) let section 8 say what the envelope knows
about the box (`safety.box_collision`). Still task-free, so it is identical across episodes of one
config. The manual describes exactly the feedback the config gives (`feedback.level`,
`observation.state_text`, `observation.tactile`): `system_v0.md` has placeholders for every
feedback-dependent sentence, and with the legacy settings (`feedback.level=full`,
`state_text=true`, `tactile=true`, `box_collision=off`) it renders byte-identical to before
(`test_prompts::test_legacy_settings_reproduce_the_old_manual_byte_for_byte`). `prompt.fewshot` (a .md file or a run dir) is appended as
"Appendix C" (cached prefix). The run id goes into turn 0 (not the system prompt) because omniroute replays cached
*responses* for byte-identical requests.

The robot never moves while the model thinks. Latency is a measured quantity
(~1 s/turn is the target, not a constraint). With `llm.backend=decisions` the "assistant reply"
is rendered from a decision model's typed answers (see Decision head): ~0.3 s per call, many
small steps.

## Modules and ownership (each file has one owner; interfaces below are the contract)

| Path | Responsibility |
|---|---|
| `controlr/types.py` | shared dataclasses (frozen contract — do not change without updating all users) |
| `controlr/config.py` | experiment config (YAML + `--set` overrides); every experiment axis is a field |
| `controlr/llm/client.py` | streaming OpenAI-compatible chat client with timings + normalised usage + early stop |
| `controlr/llm/decisions.py` | decision head (`llm.backend=decisions`): `DecisionsClient` — the `LLMClient` signature over OpenRouter's Decisions API; transcript → state + typed questions → answers rendered as grammar text |
| `controlr/llm/caching.py` | cache-marker placement per model route (applied at serialisation, never stored) |
| `controlr/llm/codex_auth.py` | controlr's OWN ChatGPT login for the Codex backend (device code, rotating refresh; token file outside the repo, one holder) — used by `scripts/codex_direct_stand.py`, not by the loop yet |
| `controlr/llm/transcript.py` | append-only transcript; images encoded once and stored as bytes |
| `controlr/protocol/grammar.py` | reply grammar: render the spec for the prompt, parse replies, completeness check |
| `controlr/protocol/feedback.py` | feedback text for the next user turn (`short`: TASK / STATE / WARN / STOP; `full`: exec receipt, clamps, limits, events, goal, state) |
| `controlr/prompts/*.md`, `controlr/prompts/builder.py` | system prompt (robot operating manual) + planner prompt templates |
| `controlr/observation/renderers.py` | observation -> list of image parts + text (resize, overlays, diff, heatmap, tile) |
| `controlr/robot/base.py` | Robot ABC (contract) |
| `controlr/robot/spec.py` | the `RobotSpec` constructors: the UR3 CB3 rig (shared by mock + Isaac) and RoboDojo's dual ARX X5 |
| `controlr/robot/kinematics.py` | UR3 CB3 FK/IK (DH, controller base frame), backend-independent; the ONE rotation-helper implementation |
| `controlr/robot/safety.py` | SafetyEnvelope: filter/clamp actions, near-limit warnings, predictive wrist/housing-vs-box check; `CartesianEnvelope` for backends that plan the joints (`make_envelope`) |
| `controlr/robot/obstacles.py` | known obstacles (`BoxObstacle`: an open-top box, walls + floor) and the robot body centre line checked against them; numpy only |
| `controlr/robot/mock.py` | kinematic mock robot (no physics; synthetic rendering) for tests |
| `controlr/robot/replay.py` | replays recorded frames regardless of actions (latency/caching benchmarks) |
| `controlr/robot/isaac/` | Isaac Sim 6.0 backend: PHANTOM's calibrated UR3 CB3 + Robotiq + D435 scene (server inside Isaac's python, numpy-only RPC client in controlr; `motion.py` = trajectory timing shared by both) |
| `controlr/robot/robodojo/` | RoboDojo backend ([docs/ROBODOJO.md](docs/ROBODOJO.md)): `serve.py` (`controlr robodojo-serve`), `client.py` (`RoboDojoRobot`), `protocol.py`, `shim/` (runs inside RoboDojo's eval client as XPolicyLab policy `controlr`) |
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
* Idle connections are kept 120 s (`KEEPALIVE_S`; httpx's 5 s default re-handshakes between slow turns).

### Decision head (`controlr/llm/decisions.py`, `llm.backend=decisions`)
A decision model (`openai/gpt-6-luna-decisions`) writes no text: it reads a `state` and returns
probabilities for named, typed questions — `score` (ordered levels), `choice`, `noul` (yes/no).
`POST {decisions.base_url}/decisions` (OpenRouter `/api/alpha/decisions`, key
`decisions.api_key_env`; omniroute has no such route), one non-streamed request per turn.
```python
class DecisionsClient:   # same complete() signature and LLMResult as LLMClient
    def __init__(self, base_url: str, api_key: str, cfg: Config, timeout_s: float, max_retries: int, *,
                 headers=None, transport=None, sleep=time.sleep, rng=None): ...
def build_state(messages, dcfg) -> list | dict          # transcript -> state
def build_questions(dcfg, acfg, tpl) -> dict            # prompts/<decisions.questions>.yaml
def render_reply(answers, dcfg, acfg) -> (str, dict)    # answers -> grammar text + log record
```
* Questions per turn (fixed key order): `dx dy dz` (+ `dyaw` with rotation=yaw) as `score` over
  `decisions.levels` / `yaw_levels` (LLM units, contain 0), `grip` choice keep|open|close,
  `status` choice CONTINUE|DONE|FAIL. Supported: `ee_delta`, rotation none|yaw, binary gripper,
  single arm (`config.validate`).
* Reduction: `expected` = probability-weighted level (an unsure model takes a smaller step),
  `argmax` = likeliest level; `|step| < deadband` → 0; a missing answer → 0. Rendering: DONE/FAIL →
  `STATUS <s>` alone (no motion on a terminal turn); open/close → `GRIP <g>` alone (in place); all
  zero → `HOLD`; else `MOVE ee_delta dx dy dz [dyaw]` (1 decimal). The rendered text is the stored
  reply, so the parser, envelope, feedback and run log are the chat path's.
* `state` is rebuilt every turn (the API keeps no history): `manual` (system text, with
  `include_manual`), `task` (turn 0's text: task, plan, first STATE), `turn`, `recent_steps` (last
  `history` (action, feedback) pairs), and the newest frame(s) — as content parts after a JSON
  text part (`image_mode=parts`) or a data-URL list field (`field`). No prompt caching; the
  transcript stays append-only for the log.
* `head: split` (`decisions_v2+`): per axis a direction `choice` neg|zero|pos (`dir_<a>`) and an
  unsigned distance `score` over `decisions.magnitudes` (`far_<a>`); step = (P(pos) − P(neg)) ×
  expected distance (`argmax`: likeliest sign × likeliest distance). A signed ordinal scale is read
  as "how far", not "which way" (journal 2026-10-07-decision-head-tuning). No yaw yet.
  `DecisionsClient.set_scene(cameras, origin, task)` (called by the loop after reset) computes per
  axis `axis_hints` (how it points in each image) and `axis_views`: the image where the axis reads
  unambiguously (its projection ≥ 35° from every other axis that projects ≥ 30 % as long; longest
  wins). `decisions_v3+` ask the direction there as an image relation ("to the left of",
  "nearer the top edge of the image than"); z without such an image as height above the table.
  With the `fovea` renderer the relation is asked in `<camera> fovea` (relative to the magenta
  cross) plus the same question on the full frame (`wide_<a>`): the crop's "level" mass defers to
  it — a target outside the crop reads as "level" there. `{target}` (`decisions_v5`,
  `decisions.target`, `{task}` → the instruction) names what is located; concrete names ("the red
  ball") are what makes it work. Image parts are preceded by "image `<name>`:" when
  `renderers.image_labels` matches the turn's images.
* A refused question (`502 ... refused to answer question "<q>"`; the same payload is refused again)
  is dropped and the request repeated without backoff (≤ 3 drops; its axis holds);
  `decisions.refused` in the turn record.
* `LLMResult.decisions` (turn record `decisions`): reduced steps, grip, status, per-question
  probabilities / score / confidence, response id. `Usage.prompt_tokens` = `input_tokens`;
  `raw.cost` is OpenRouter's cost. Timings: one response time (`ttft = t_complete = t_end`).
* The planner always uses the chat client (`run_episode(planner_llm=...)`; default: `llm` for
  chat control or a fake, else `make_llm_client(cfg)`); `make_control_client(cfg)` picks the
  control client.

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
                    spec: RobotSpec | None = None, task: str | None = None) -> str
def visible_events(events: list[SafetyEvent], cfg: Config) -> list[SafetyEvent]
def to_llm_units(text: str, a: ActionConfig) -> str   # "<n> mm"/"<n> deg" -> configured units
```
Safety and backend messages (`SafetyEvent.message`, `.brief`, `GoalReport.message`) are written in
canonical mm / deg; `format_feedback` converts them, so `pos_unit=cm` never mixes units.

`feedback.level=short` (the default) — only four line types, in this order:
```
TASK: <instruction>                 (turn 0 head; every turn with feedback.repeat_task)
STATE: tcp x=312 y=-45 z=88 mm yaw=12 deg | grip 42 mm open        (no `holding`)
WARN: move shortened: table clearance                               (one short line per issue)
STOP: the gripper pushed against the blue box
```
WARN = every clamp / skipped move (`SafetyEvent.brief`, a few words without measurements), every
unusable reply line (`reply not understood: ... - nothing executed`), near-limit and box-proximity
warnings and a DONE whose check failed (`task not complete yet`); contacts below the stop force,
settle notices and fingertip events are not shown. With `feedback.orientation=direction` STATE describes the tool orientation as where it points and how its jaw
line lies (`tool points 30 deg below horizontal, toward +y [+0.00 +0.87 -0.50] | jaw line along x`)
instead of roll/pitch/yaw (`system_v2` explains it and the rotation signs). STOP = a stopped motion (`.brief`). No TURN /
EXEC / EVENT / GOAL / PARSE ERROR lines.

`feedback.level=full` — the legacy receipt (events of kind `tactile` only
with `observation.tactile`; `holding` in STATE only with `full` + `tactile`).
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
`fovea` (extra image `<camera> fovea`: a square crop of the NATIVE frame, `fovea_px` wide, around
the projected TCP — shifted inside the frame at the edges — scaled to `size`, the one upscale;
magenta cross at the TCP; `image_labels(cfg)` = the labels of a turn's images when they do not
depend on the previous frame);
`tile` (all images of the turn in one grid image, labelled). Resize to `size` long edge with
LANCZOS, never upscale. Deterministic output for identical input (cache stability).

### Robot backends
`controlr/robot/base.py::Robot` (reset/observe/state/execute/check_goal/close; optional
`reference_state()` — the commanded reset state, default None; optional `scene_record()` — the
sampled task scene for `setup.json`, default None; optional `obstacles()` — known obstacles at their
current pose, default []).
Contract additions in `types.py` (backwards compatible, defaults None / ""): `SafetyEvent.brief`
(the event in a few words, no measurements: the short feedback), `ExecReport.backend` (backend
diagnostics for the run log only), `RobotSpec.finger_pad`
(pad half length / half width / thickness, m), `Action.q_path` (envelope waypoints); `Action.values`
of ee_abs + rotation=yaw has 4 entries (x, y, z, yaw).
Factory: `controlr.robot.make_robot(cfg: Config) -> Robot` (not for `robodojo`: its episodes are
started by RoboDojo, see below). Optional `episode_over() -> str | None`: the backend ended the
episode itself (the loop ends with outcome `env_end`).

Multi-arm robots and backend kinematics (contract additions, defaults keep single-arm code
unchanged): `RobotSpec.arms` (e.g. `("L", "R")`; all arms share the spec — frame, workspace,
gripper), `RobotState.arms` (one state per arm; the top-level fields mirror the first arm),
`Action.arm` / `Action.step` (the parts of one reply line, one per arm, move together),
`RobotSpec.kinematics` (`ur3` = UR3 kinematics in the harness; `backend` = the backend plans the
joints: `make_envelope` returns a `CartesianEnvelope` — same step limits, workspace, table
clearance and rotation conventions, no IK — and approved ee actions carry `Action.tcp_target`),
`Observation.notes` (backend feedback lines after STATE, e.g. RoboDojo's `STEPS:`). With arms the
grammar is `MOVE L <mode> v.. [GRIP g] R <mode> v.. [GRIP g]` / `GRIP L g R g`, STATE is one line
per arm, and the manual is `system_v1` (`build_system_prompt` uses `_values_arms`).
* RoboDojo backend (`robot/robodojo/`, [docs/ROBODOJO.md](docs/ROBODOJO.md)): RoboDojo's eval
  client owns the episode and calls the shim (`XPolicyLab/policy/controlr/deploy.py`) per episode;
  the shim connects to `controlr robodojo-serve` (localhost, HMAC), announces the episode and
  serves observe / execute / check_goal / done. `RoboDojoRobot` converts flange (link6) poses to
  the grasp-point TCP (150.1 mm along flange +x, tool z on that axis), 0..1 grippers to widths and
  `L`/`R` to `left`/`right`; the shim plans each arm target with RoboDojo's cuRobo, resamples to
  ≤ 0.05 rad per joint per env step, moves grippers after the arm, sends both arms every step and
  marks an episode controlr ended early as failed.
* `kinematics.py`: `fk(q) -> (pos, rotvec)` of the TCP in the UR controller base frame
  (DH from PHANTOM `phantom/sim/kinematics.py`: a=[0,-0.24365,-0.21325,0,0,0],
  d=[0.1519,0,0,0.11235,0.08535,0.0819], alpha=[pi/2,0,0,pi/2,-pi/2,0]; TCP offset +z 0.18 m
  for Robotiq 2F-85 fingertip centre — configurable), `ik(pos, rotvec, q_seed) -> q | None`
  (damped least squares, joint-limit aware, nearest to seed).
* `obstacles.py`: `BoxObstacle` (centre at the floor underside, yaw, outer size, wall and floor
  thickness, `movable`) — walls and floor only, the opening is free (the gripper places INTO the
  box); `box_from_bin_info(bin_info)`; `body_points(kin, q)` = the wrist / gripper-housing centre
  line TCP → flange → wrist 3 → wrist 2 → wrist 1 without the first 60 mm (the fingers);
  `body_clearance(kin, q, obstacles)`.
* `safety.py`: `SafetyEnvelope(spec, cfg).reset(state0)`, `.set_obstacles(obstacles)` (the loop
  passes `Robot.obstacles()` at their CURRENT pose before every filter), `.filter(actions, state) ->
  (actions_out, events)`: per-line step limits (TCP travel also in joint modes), workspace box, table clearance
  of the LOWEST FINGERTIP (`RobotSpec.finger_pad`; a tilted open gripper reaches 15-45 mm below
  its TCP), joint soft limits, near-limit warnings, clamp vs reject. ee moves: IK every
  `safety.path_step_m` (5 mm) along the straight TCP line -> `Action.q_path` waypoints; an
  unreachable or near-singular stretch (joint jump > 0.02 rad/mm + 0.05) SHORTENS the move
  (CLAMP "moved N % of the way", with the turned angle when the move turns, and the kinematic
  reason: elbow straight = edge of reach, wrist_2 near 0/180 = wrist singularity; < 10 % ->
  skipped, `ik_fail`). rotation=none keeps the reference orientation from `reset` (a contact tilt
  is undone; WARN > 3 deg). rotation=yaw (`SafetyEnvelope(..., rotation=cfg.action.rotation)`,
  set by the loop and the backends' fallback envelopes) holds only the reference roll/pitch: the
  target heading is the current yaw + dyaw (ee_delta) or the commanded yaw (ee_abs), dyaw clamped
  to `max_step_rad`; a contact tilt is undone, a turn is kept. Joint moves are
  checked at samples along the joint path. Known obstacles (`safety.box_collision`): along the
  move's waypoints the body centre line must keep `box_clearance_m` (50 mm) from the box walls and
  floor; a waypoint closer than that AND closer than anything before on this move is a violation
  (moving away or along is always allowed): `block` shortens the move there (kind `box`, CLAMP;
  skipped below 10 %, rejected with `clamp: false`), `warn` executes it and adds a WARN (kind
  `box_warn`), `off` does not check. Every clamp event carries a `brief` for the short feedback.
  Backend-independent.
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
  view is rendered. The blue box `/World/Bin` is made ONE dynamic rigid body (its five PHANTOM
  colliders, mass `task.params.box_mass_kg` 0.4 kg, PHANTOM's friction; `box_dynamic: false` =
  kinematic, i.e. the old immovable box); `reset` restores its pose; the server's `state()` carries
  the box's CURRENT geometry (`tasks.bin_info_at`), which the goal check, the envelope
  (`IsaacRobot.obstacles()`) and `tasks.body_box_clearance` use. Force stop rules
  (`robot/isaac/contacts.py`, CPU-tested): arm/gripper vs table at `contact_force_stop_n`, vs the box
  at `box_force_stop_n`, vs the packet at `object_force_stop_n` only while NOT held; while held, the
  packet's own contact view against table/mat/box at `held_object_force_stop_n` ("the held packet
  pushed against the box wall") — the grip's pad forces never stop the arm. `execute` returns a
  `profile` (targets / simulate / contacts / bookkeeping / settle seconds) and the client puts it,
  the contact peaks (incl. pad forces), the gripper's own object detection and the box shift into
  `ExecReport.backend` (run log only). Reach and push targets are TEXT relative to visible objects
  (no markers), drawn per seed; the instruction comes from the server's episode. Cameras: the
  calibrated D435 (`scene`) and, with `robot.params.extra_cameras` (server `--extra-cameras`),
  virtual `top` (looking down: right = +x, up = +y) and `side` (horizontal along +x: right = −y,
  up = +z) views aimed between the mat and the box (`robot/isaac/cameras.py`, numpy; PHANTOM's
  intrinsics helpers; a running server without them is refused). The mock has the same synthetic
  `top` / `side` (`robot.params.cameras`). Tasks: the PHANTOM waffle pick-to-box first, then simple reach/push variants.
  The mock robot covers GPU-free unit tests.

### Run log (`controlr/runlog.py`)
`runs/<UTC timestamp>_[fake_]<name>/`: `config.yaml`, `system_prompt.md`, `plan.md`,
`images/<sha>.jpg` (exact bytes sent), `messages.jsonl` (transcript with image refs),
`turns.jsonl`, `summary.json`, `setup.json` (reset time, instruction, cache style, `llm_backend`
live|fake, state0 + reference state, turn-0 image shas, lookback warning, `scene` =
`Robot.scene_record()`: the sampled task scene — Isaac: packet pose / yaw / yaw offset from nominal /
tilt / settle drift, box pose, start joints / TCP / tool yaw / start yaw offset, marker, zone; mm, deg), `cameras.json` (K,
T_cam_base), `spec.json` (RobotSpec), `planner.json` (planner request/usage/timings/reasoning),
`raw/<turn>_<cam>.png` (native frames without overlays, `log.save_raw_frames`).

`turns.jsonl`, one record per turn — enough to rebuild (obs_t, action_t, obs_t+1) (+ `backend`:
`ExecReport.backend` diagnostics never shown to the model — Isaac: server profile, wall / sim seconds,
contact peaks incl. tactile pad forces, holding by pads and by gripper stall, box pose / shift, stop):
`obs_images` (shas the model saw), `state_before`, `reply` (stored, note-stripped if truncated;
`reply_streamed` when it differs), `reasoning`/`reasoning_chars`, `actions` / `executed` (SI values,
`q_target`, `q_path`), `events`, `goal`, `next_obs_images` + `state` (after the action; also after
the terminal turn), `feedback` (absent on the terminal turn), `llm` (all Timings fields), `usage`,
`request_bytes`, allow-listed `headers` (x-omniroute-*, request ids, rate limits), `timings`.

`setup.json` also records `obstacles` (the backend's known obstacles at reset; `controlr prompt
--setup` uses them). `summary.json`: `sim_time` (sum of backend sim seconds, server and execution
wall seconds, `sim_per_wall`), `outcome`, `success` (goal check at the end), `success_verified` (outcome ==
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
