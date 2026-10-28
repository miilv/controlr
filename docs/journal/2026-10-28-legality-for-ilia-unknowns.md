# 2026-10-28: legality, "For Ilia" items: held / pose unknowns stay unknown

Source: robot-harness `docs/notes/G1F-VERIFICATION-2026-10-28.md` (For Ilia 1-13) and
`docs/notes/P1S-VERIFICATION-2026-10-27.md` (For Ilia 1-3), read-only. Their refs are to the imported copy of our stack
(controlr db48652); every ref was re-located on main 1e0f7b0 before the change. This PR is the class-A half: an
unknown that became a confident value. The budget / episode-state half is the sibling PR
(journal 2026-10-28-legality-for-ilia-budget).

Tests: `tests/dojo/test_legal_for_ilia_unknowns.py` (real functions, fakes only: FakeChat, a two-box finger model,
the vendored `j1_schema`; no Isaac, GPU, LLM or catalogue), plus one in `tests/dojo/test_runtime.py`. The two
`measure_opening` / B1 `held_look` tests need cv2 + scipy (skipped in CI); they pass locally with
`uv run --with opencv-python-headless --with scipy`.

## Fixed

| item | what was wrong | fix | test |
|---|---|---|---|
| P1S-2 (I1-3) | the agent's held_look without a yes / no returned `held False, confidence 0.0`, top level and per arm; a B1 exception in mode `b1` returned `{"_error"}` | `held: None, confidence 0.0, error: "no answer"` at the top level and in the arm's entry; a B1 error is `held: None, error: "B1 error: ..."` (no `_error` key) | `test_agent_without_a_yes_no_is_unknown_and_the_schema_refuses_it`, `test_b1_error_in_mode_b1_is_unknown_not_an_error_key` |
| G1F-4 (G2-3) | a uniform or tied-score wrist frame: `measure_opening` took the first opening, `held False` at 1.0 | `opening_fit_rejected`: best score < 0.05, or an opening >= 0.2 away within 0.02 of the best, is no measurement (`rejected`); the colour test decides, a not-held verdict then capped at 0.3 (`opening_rejected` in the arm's entry). A wrist frame with Lab-L std < 2 is `held: None` ("uniform wrist frame") | `test_tied_or_flat_silhouette_scores_are_rejected`, `test_measure_opening_rejects_a_uniform_frame_and_keeps_a_visible_jaw`, `test_held_look_on_a_uniform_wrist_frame_is_unknown` |
| G1F-11 (G2-10) | the other arm's answer used for the asked arm: B1's top level aggregated both arms, the agentic B1-first gate read it when the asked arm had no entry, and the runtime service fell back to it | `held_summary`: with `arm`, the top level is that arm's verdict, a missing one an explicit unknown entry; the agentic gate reads only the asked arm (else escalates in `b1_then_agent`, unknown in `b1`); `_Perception.held_look` returns the requested arm's verdict or `held: None` + `error` | `test_b1_summary_never_answers_for_the_other_arm`, `test_agentic_gate_does_not_take_the_other_arms_b1_answer[True/False]`, `test_runtime_held_look_answers_for_the_requested_arm_or_unknown` |
| G1F-1 (G1-1, adapter half) | `merge_prior` gave a prior without confidence 0.5 and carried a static one at 0.45; a prior without quat crashed the revert (`p["quat"]`) or was carried quat-less | `prior_unusable`: a prior entry needs its own confidence in 0..1 and a full pose; otherwise it never reverts, carries or re-identifies: a current reading at its place stands as read, else it is listed in `last_seen` only (no invented confidence, `prior_unusable` says why). `_carry_static` the same | `test_merge_prior_never_invents_a_confidence_or_a_pose` |
| P1S-3 (I1-5) | `bridge/perception.record_from_perceived`: a missing quat became `[1, 0, 0, 0]` with no marker | only `ROTATION_FREE_CATEGORIES` (`ball`) take identity; any other quat-less (or non-rotation) object is refused as a planning row: `_orientation_unknown`, `_unusable` (with its position) and `_not_found` with its asset as the candidate. Rows carry `_perception.orientation_known` / `orientation` | `test_an_object_without_a_quat_is_not_a_planning_row` |

Consumers of the held unknown, traced:
- robot-harness's J1 server: `validate_held` refuses `held: None` ("held must be a bool"), i.e. its B1-failure path
  (operator, or `B1-error` under D-C7's `J1_HELD_FALLBACK=no`), never the B1 "no".
- runtime: `RuntimeProvider._held_check` (belief.py) has an explicit unknown branch: the grasp stays, no second look,
  event `unknown` + `kept`, stat `held_look_unknown` (`test_held_look_unknown_keeps_the_grasp_and_is_logged`). The old
  `False / 0.0` was also kept (confidence < 0.5), so the runtime's actions do not change; only the event does.
  `policy._held_look` already returned `held: None` on an exception; `held_confirm` (mode agent) treats None as "not
  sure" (grasp kept), as before.
- offline evals (`val_replay51`, `heldsyn_eval`, bgperc `replay_eval`) read the per-arm entry; `held: None` there was
  already possible ("fingertip region not visible").
- the orientation refusal: the runtime treats a `_not_found` label as missing (no plan on it, `verify_unavailable`,
  the search may re-measure it with its asset). Nothing in the runtime can hold "position known, orientation unknown"
  today, so the refusal covers every plan on that object, not only orientation-dependent ones.

## Deferred

- G1F-3 (G2-2 digit on ambiguous evidence), G1F-5 (G2-4 keep copies drop the digit): the card-digit stream.
- G1F-6 (G2-5 cross-category XY matching in `add_unmentioned` / `seed_refine`): a matching-rule change that needs
  rendered validation; `add_unmentioned`'s copy path is also the card-digit stream's.
- G1F-7 (G2-6 tracker XY-only jitter): changes which tracked poses survive; needs rendered validation.
- G1F-8 (G2-7 `merge_prior` XY-only pairing): the revert rule was tuned on rendered J1 cases (push L3, the re-read base
  under a hovering arm); a 3-D / gripper-proximity rule needs those replays. This PR only stops unscored priors.
- G1F-9 (G2-8 refine rebuilds the wrist camera from FK): `refine()`'s camera seam and the edge-fallback quality touch
  every refine caller; needs rendered validation.
- G1F-10 (G2-9 tracker lost on a worker restart): needs the tracker state outside the worker; the sibling PR flags it
  (`tracker=lost` in the gated reply's notes).
- G1F-2, G1F-12, G1F-13, P1S-1: the sibling PR.

## Behaviour changes, and can rendered numbers change?

- Agent held_look without an answer: `held` None instead of False. Strict server: the B1-failure path (it already
  treated confidence <= 0 as no answer, P1S); runtime: same action (grasp kept). Rendered numbers: no.
- B1 held_look with `arm`: the top level is the asked arm's (before: any arm held, min confidence). The strict server
  and the runtime read the arm's entry; the agentic gate now escalates (b1_then_agent) when only the other arm
  answered. Rendered numbers: only in episodes where the asked arm had no B1 verdict (no wrist frame / calibration /
  joints, an open command): one more agent call, or unknown instead of the other arm's answer.
- Rejected silhouette fits / uniform frames: yes, rendered held_look verdicts can change, but only on frames whose
  fit was flat or tied (the colour test then decides, a negative capped at 0.3 so `b1_then_agent` asks the agent and
  the runtime never drops on it), and on uniform frames (unknown). The logged `opening.score` of past runs is the best
  score only, so the incidence cannot be counted offline.
- `merge_prior`: only priors without their own confidence or full pose change (our B1 always writes both; the agentic
  perceiver's quat-less agent points are the realistic case, where the revert crashed before). Rendered numbers:
  only on the J1 route with such priors.
- `record_from_perceived`: only the bridge's `--perception server` route (the official path runs `--perception
  local`), and only for objects the server returned without a quat. Rendered numbers on the official path: no.
