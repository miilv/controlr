# 2026-10-25: dojo agent layer: grounding benchmark and the hybrid agent loop

Stream: agent layer (branch `feat/dojo-agent`, code `dojo/agent/`, tests `tests/dojo/test_agent_*.py`).
Strategy (lead): one fixed hybrid system. An LLM agent reads the instruction and the scene. It either calls ONE
generic skill with arguments, or writes Python cells over a `robo` API (primitives + skills + perception + the
verify that lives inside the skills). The model follows PyRUA-Lean's design (persistent cells, typed results,
benchmark-aware manual; Apache-2.0, adapted, not copied).

## 1. What was built

- **Manual** (`manual.py`, `prompts/manual_vN.md`). Sections:
  - world frame;
  - how episodes are judged (home, bank with open grippers, exact counts, push-not-lift, scripted arms, memory);
  - v2 only: instruction patterns and how they are judged (lead's P4: phrased by instruction pattern, no task
    names);
  - skill docs: what each skill does, which roles its arguments are, and what it does NOT do;
  - the `robo` API;
  - working rules.

  Skill docs follow the deployed signatures: v2 documents `stack_objects(ordered=...)` only if the library has it.
- **Scene** (`scene.py`). WorldState becomes text with neutral ids `<category>_<k>` (left to right). Checker labels
  ("target", "cat0_1", "red") never reach the model; `SceneIds` maps ids back to labels.
  - Per object: kind, asset index, every caption name (or a short description), catalogue mean colour, digit,
    centre, yaw / tilt and size.
  - The catalogue colour wins over the caption: `Rigid/cube/00004` is captioned "metallic gray" but renders red.
- **Phase 1 benchmark** (`dataset.py`, `gold.py`, `grounding.py`): see the
  [experiment report](../experiments/2026-10-25-agent-grounding.md).
  - Instructions come from each config's own `gen_instruction` (exec'd against a label-query stub) on our dev
    layouts. For the 23 configs without simlab layouts, fresh `gen_layouts` records were generated
    (`~/dojo-data/agent/layouts`).
- **Phase 2 agent** (`session.py`, `robo.py`, `loop.py`, `policy.py`):
  - **Session.** A persistent Python namespace with notebook semantics (the trailing expression is echoed) and a
    private `print`. It is restricted: an import allowlist, no `open` / `exec` / dunder access.
  - **Bounded cells.** `CellCheckpoint` and `EpisodeFinished` are `BaseException`s, so the model's `except
    Exception` cannot swallow them. `Robo` raises the checkpoint at the next primitive once a cell has spent
    `max_cell_steps` env steps (250) or made `max_cell_contacts` contacts (4).
  - **`robo` API.** Typed `Result(ok, status, note, steps, arm)` over `Skills`: pick, place, place_on, drop_into,
    push, press, handover, move_to, grip, wait, bank, home and skill(name, **args); plus perception (scene, obj,
    state, show, refine, search) and `finish()`.
    - Every action converts labels into ids in its notes.
    - A divergence (resync) re-plans with a fresh Skills.
    - IK failures return the reachable alternatives as (change, xyz).
  - **Loop.** controlr's append-only `Transcript`: images are JPEG-encoded once, and cache markers apply on Claude
    routes only.
    - Turn 0 is the nonce + instruction + scene + head frame (640 px).
    - Each later turn is the cell's receipts / stdout / value / error / checkpoint + STATE, plus up to 3 images
      (the head frame after each contact, and `robo.show`).
    - The loop ends on `finish()`, the episode ending, `max_calls` (30) or 2 replies without code.
  - **`AgentPolicy`** = `RuntimePolicy(program=...)` on integration's seam (8588701). Setup (perception, looks,
    search, provider, verifier) and the tail (finish unless ok, still steps, error handling) are integration's.
    - The agent's config comes from `RuntimeConfig.extra.agent`, so queue jobs can carry it.
    - LLM usage goes into `report["agent"]`, which is live, because the main thread can close the report while
      the worker is parked.
    - Logs (messages with image shas, calls, cells) go to `$OUT/agent/<stamp>_epNNN/`.
  - **Dry runs on the kinematic env with real models** (3 episodes, 6 calls):
    - Sonnet wrote a 4-cell pick-and-lift and recovered from IK failures through `path='auto'`;
    - Astra-high made one `robo.skill('stack_objects')` call and `finish()` (1 call, 185 steps).

## 2. Coordination

- Integration added the program seam, `--policy`, `"agent"` in episode rows, `--repo DIR` jobs in the queue, and
  the cover_blocks / general_pickup / align_blocks vocabularies (8588701, 8c2340c).
- skills3 added `stack_objects(ordered=True)` and an `insert_into` that skips the receptacle the object is in
  (63577bd). Not merged into integ yet, so manual v2 does not document `ordered` until it is.
- The lead's decisions:
  - P4: the manual may state checker conventions phrased by instruction pattern;
  - Phase 2 runs with verify off, identical for all three models;
  - stack_bowls replaces press_by_number in mode B (no SpringButton / number_card in perception's catalogue);
  - press_by_number additionally runs in mode A;
  - lock order: interleaved 1:1 with integration's jobs.

## 3. Landmines

- `claude/claude-sonnet-5-5-high` returns 400 through omniroute (not in the live catalog). Sonnet's effort is set
  only through `extra_body.reasoning_effort`.
- The router's usage carries no cost field, so cost is reported as tokens only (`cx/` is subscription).
- Asset captions can be wrong (red cube captioned gray) or missing (hoe: only the description "yellow ice axe").
- `gen_instruction` descriptions pick a random asset description. "Pick up the yellow ice axe" names a `hoe`, so
  the scene must show descriptions, not only categories.

## 4. Phase 2 results

Report: [agent-e2e](../experiments/2026-10-25-agent-e2e.md).

- **Mode B, verify off, 6 tasks × 2 layouts:** Sonnet 5.5 5/12, Astra-high 5/12, Sol 6.1 4/12. Every model made
  the same skill-or-code choice in every episode; the failures came from perception and skills.
- **press_by_number in mode A** with the v2 convention: 6/6.
- **cx/ GPT-6 empty replies:** an undeclared `python` tool call was dropped by the router. Fixed with a declared
  tool and `LLMResult.tool_calls`.
- **Perception before / after de21b05:** the general_pickup estimate went from 501 s to seconds.
- **Batching:** one bundle job per task ran every model arm (`AgentConfig.models`); the lead allowed up to 50 min.
- **Spend:** Phase 1 1,140 calls; Phase 2 ~430.
