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
[planner]  one separate call (default claude-opus-5-5, reasoning_effort=xhigh): system prompt
           + task + obs0 -> plan text (not cached, not part of the control transcript
           except as text in turn 0 when planner.include_in_context)
turn 0     user: task instruction (+ plan) + observation(obs0)
loop:      assistant reply (streamed; early-stop once grammar complete)
           -> parse (MOVE*/STATUS) -> SafetyEnvelope.filter -> robot.execute (blocks until settled)
           -> goal check -> end? -> observe -> user: feedback + observation
end:       DONE (goal verified unless trust_done) | FAIL | max_turns | STOP event | parse-error streak
```

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
| `controlr/robot/kinematics.py` | UR3 CB3 FK/IK (DH, controller base frame), backend-independent |
| `controlr/robot/safety.py` | SafetyEnvelope: filter/clamp actions, near-limit warnings |
| `controlr/robot/mock.py` | kinematic mock robot (no physics; synthetic rendering) for tests |
| `controlr/robot/replay.py` | replays recorded frames regardless of actions (latency/caching benchmarks) |
| `controlr/robot/mujoco_sim/` | MuJoCo UR3 + Robotiq 2F-85 scene, cameras, tasks (reach / push / pick_place) |
| `controlr/loop.py` | episode runner (planner + turn loop), end conditions |
| `controlr/runlog.py` | run directory writer |
| `controlr/bench/` | cache probe, latency matrix, sweep runner |
| `controlr/cli.py` | `controlr run | bench-cache | bench-latency | sweep | report | models` |

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
@dataclass
class LLMResult:
    text: str; usage: Usage | None; timings: Timings
    stopped_early: bool; finish_reason: str | None
    error: str | None; http_status: int | None; attempts: int
class LLMClient:
    def __init__(self, base_url: str, api_key: str, timeout_s: float, max_retries: int): ...
    def complete(self, model: str, messages: list[dict], *, max_tokens: int,
                 temperature: float | None = None, extra_body: dict | None = None,
                 stop_when: Callable[[str], bool] | None = None) -> LLMResult: ...
```
* POST `{base_url}/chat/completions`, `stream: true`, `stream_options: {include_usage: true}`, httpx.
* Usage fields seen through omniroute: `prompt_tokens`, `completion_tokens`,
  `prompt_tokens_details.cached_tokens`, `cache_read_input_tokens`,
  `cache_creation_input_tokens`, `completion_tokens_details.reasoning_tokens` / `reasoning_tokens`.
* `stop_when(text_so_far)` true -> stop reading, close the stream, `stopped_early=True`
  (usage may then be missing -> `usage=None`; that is acceptable and logged).
* Retries: connection errors, 429, 5xx with exponential backoff; 4xx other -> no retry.

### Caching (`controlr/llm/caching.py`)
```python
def cache_style_for(model: str, configured: str) -> str          # "anthropic" | "none"
def apply_cache_markers(messages: list[dict], style: str, ttl: str = "5m") -> list[dict]  # returns a deep copy
```
* `anthropic` style (all `claude/`, `cc/`, `no-think/claude*` routes, verified through omniroute):
  `cache_control: {"type":"ephemeral"}` (+`"ttl":"1h"` when ttl=1h) on (1) the last content
  part of the system message, (2) the last content part of the **last** user message, and
  (3) the last content part of the **second-to-last** user message (keeps a read point inside the
  20-block lookback even if a turn adds many blocks). Max 4 markers total.
* Markers are added on a copy at request time; the stored transcript never contains markers,
  so earlier messages stay byte-identical apart from the moving marker (verified: cache reads
  cover everything but the newest turn).
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
Text format (default; `action.format=text`):
```
MOVE <mode> v1 v2 ... [GRIP open|close|<mm>]     # 0..max_chunk lines; mode must equal the configured mode
GRIP open|close|<mm>                             # gripper-only line (counts as one action)
HOLD                                             # explicit no-op
STATUS OK|DONE|FAIL|STUCK|LIMIT [short note]     # exactly one, last line; ends the reply
```
* Units in the reply are the configured LLM units (default mm, deg). Values per mode:
  `ee_delta`/`ee_abs`: `x y z` (rotation=none), `x y z yaw` (rotation=yaw),
  `x y z roll pitch yaw` (rotation=full; extrinsic about base x/y/z);
  `joint_delta`/`joint_abs`: one value per joint, in joint order.
* Tolerant parsing: ignore markdown fences, leading/trailing prose, case; collect errors
  (wrong mode, wrong arity, non-numeric, > max_chunk lines, missing STATUS) for feedback.
```python
def grammar_spec(cfg: ActionConfig, spec: RobotSpec) -> str          # text block for the system prompt
def parse_reply(text: str, cfg: ActionConfig, spec: RobotSpec) -> ParsedReply   # SI units out
def is_complete(text: str) -> bool                                   # a STATUS line has been emitted
```

### Feedback (`controlr/protocol/feedback.py`)
```python
def format_feedback(turn: int, parsed: ParsedReply | None, report: ExecReport | None,
                    goal: GoalReport | None, obs: Observation, cfg: Config) -> str
```
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
Parse errors are reported as `PARSE ERROR: ...` + one-line reminder of the grammar.

### Observation renderers (`controlr/observation/renderers.py`)
```python
@dataclass
class RenderedObservation: images: list[ImagePart]; text: str
class ObservationRenderer:
    def __init__(self, cfg: ObservationConfig): ...
    def render(self, obs: Observation, prev: Observation | None, turn: int) -> RenderedObservation
```
Renderers (applied per camera, in configured order): `raw`; `grid` (base-frame XY grid on the
table plane projected via CameraInfo, labelled in LLM units); `axes` (base x/y/z arrows at the
TCP); `ee_marker` (projected TCP + gripper footprint); `diff` (extra image: |I_t - I_{t-1}|
amplified); `heatmap` (extra image: colour heat of change magnitude over the current frame);
`tile` (all images of the turn in one grid image, labelled). Resize to `size` long edge with
LANCZOS, never upscale. Deterministic output for identical input (cache stability).

### Robot backends
`controlr/robot/base.py::Robot` (reset/observe/state/execute/check_goal/close).
Factory: `controlr.robot.make_robot(cfg: Config) -> Robot`.
* `kinematics.py`: `fk(q) -> (pos, rotvec)` of the TCP in the UR controller base frame
  (DH from PHANTOM `phantom/sim/kinematics.py`: a=[0,-0.24365,-0.21325,0,0,0],
  d=[0.1519,0,0,0.11235,0.08535,0.0819], alpha=[pi/2,0,0,pi/2,-pi/2,0]; TCP offset +z 0.18 m
  for Robotiq 2F-85 fingertip centre — configurable), `ik(pos, rotvec, q_seed) -> q | None`
  (damped least squares, joint-limit aware, nearest to seed).
* `safety.py`: `SafetyEnvelope(spec, cfg).filter(actions, state) -> (actions_out, events)`:
  per-line step limits, workspace box, table clearance, joint soft limits (via IK for ee modes),
  near-limit warnings, clamp vs reject. Backend-independent.
* MuJoCo backend: UR3 CB3 from PHANTOM's URDF (`assets/sim/ur3`, BSD) + Menagerie
  `robotiq_2f85` (Apache-2.0), table, task objects, cameras `scene` (fixed, D435-like, 640x480)
  and `wrist`. Execution: Cartesian straight-line interpolation with IK at waypoints at
  `safety.max_tcp_speed_m_s`, position-controlled joints, then settle; gripper actuation; contact
  monitoring -> events. Headless rendering via EGL (`MUJOCO_GL=egl`) on compute2.
* Tasks: `reach` (TCP within tol of a marker), `push` (cube into a zone), `pick_place`
  (cube into a bowl). Randomised object poses by seed; success + progress metrics.

### Run log (`controlr/runlog.py`)
`runs/<UTC timestamp>_<name>/`: `config.yaml`, `system_prompt.md`, `plan.md`,
`images/<sha>.jpg` (exact bytes sent), `messages.jsonl` (transcript with image refs),
`turns.jsonl` (per turn: timings, usage, reply text, parsed actions, exec report, events,
goal, state), `summary.json` (outcome, turns, token totals, cache-read share, latency p50/p90).
Later: export to LeRobotDataset for the action head.

## Deployment
Code runs on **compute2** (`isr-lab-4`, RTX 4090, Ubuntu 22.04, py3.10/3.11) under
`/root/controlr` with a uv venv; `MUJOCO_GL=egl`. `scripts/deploy.sh` rsyncs the repo (no
`.env` in git — copied separately). compute3 (Isaac Sim 6.0, PHANTOM scene) is for the
later Isaac backend once its NVIDIA driver works.
