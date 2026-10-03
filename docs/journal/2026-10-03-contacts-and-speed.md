# Contacts, short feedback and Isaac speed (2026-10-02/03)

Branch `feat/contacts-and-speed` (base `chore/project-structure`). Report with measurements and
the validation round: [experiments/2026-10-02-contacts-and-speed.md](../experiments/2026-10-02-contacts-and-speed.md).

## Default changes (rule 1: every one is a config field; the old behaviour is one setting away)

| field | old | new | why |
|---|---|---|---|
| `task.params.box_dynamic` (Isaac waffle/reach/push) | immovable box (PHANTOM static colliders) | `true`: the box is ONE dynamic rigid body, `box_mass_kg` 0.4 | the position-controlled arm against an immovable wall gave 102 kN / 136 kN contacts and PhysX divergence; a light tote slides |
| `safety.box_collision` | (no check) | `warn` | predictive wrist/housing-vs-box check; `block` shortens moves, `off` = old |
| `safety.box_force_stop_n` | (= `contact_force_stop_n`, 80 N) | 30 N | a light touch of the box stops the arm and is reported |
| `safety.box_push_stop_m` | — | 0.005 | the 0.4 kg box slides at 2–4 N, below any force limit: touching it while it moved > 5 mm stops too (measured: a 160 mm push never crossed 30 N) |
| held-packet stop | robot-vs-packet force (pads), 40 N | packet-vs-environment force (`held_object_force_stop_n`, null = `object_force_stop_n` 40 N) | the grip's own pad forces reached 85–89 N in free air (false STOPs); not configurable back — the old rule was a bug |
| `observation.tactile` | (tactile always) | `false` | Ilia: no fingertip forces / "touched the packet" / pad-based `holding` in the feedback |
| `feedback.level` | (full receipt) | `short`: TASK / STATE (no `holding`) / WARN / STOP only | Ilia: "keep feedback super short. Only TASK, STATE (without holding), WARN, STOP" |
| `observation.state_text` | `true` | `true` (unchanged; it was briefly `false` on the branch for an intermediate design Ilia replaced) | |
| Isaac speed (`robot.params` defaults in `client.py`) | USD write-back every step, PHANTOM's per-body contact views read every 1 ms step, `World.step` + `apply_action` wrappers | USD write-back only before renders, one contact matrix, no per-body views, PhysX stepped and targets set directly; physics unchanged (1 ms, 64 / 8 iterations) | 0.18x -> ~0.7x real time. A first default (32 iterations, 5 ms ticks, ~0.94x) blew up 2 of 4 live grasps (replay-confirmed: the 5 ms ticks) and was replaced before the final round |
| reach / push tasks | red ball on a pole / green square | text targets relative to visible objects, drawn per seed | Ilia: "the text instruction must be enough" |
| control model | Sonnet 5.5 / Opus 5.5 / Haiku in benches | Sonnet 5.5 only in `configs/bench/latency.yaml`, `configs/sweeps/example.yaml` | Ilia: "don't test haiku, sonnet-5-5 only for now" |

## system_v0.md

The template gained placeholders for every sentence that depends on the feedback (`loop_item1`,
`evidence`, `feedback_doc`, `told`, `objects_rule`, `nomove_rule`; `state_check` now includes "the
new image and"). With the legacy settings (`feedback.level=full`, `state_text=true`,
`tactile=true`, `box_collision=off`) every manual renders byte-identical to commit 8a90afa
(`test_prompts::test_legacy_settings_reproduce_the_old_manual_byte_for_byte` pins the sha256 of six
manuals; `controlr prompt --setup` on the rotation round's run reproduces its `system_prompt.md`).
The default manual therefore differs from the rotation round's: it names only the four short line
types, has no `holding`, and section 8 says the envelope does not keep the arm out of the box but
contact is felt (a light touch stops the arm; the box slides).

## Notes

- The intermediate `none / minimal / full` feedback design was replaced the same day by Ilia's
  `short` spec; only `short` and `full` exist.
- `holding` is no longer shown by default. A Robotiq-style object detection (`holding_grip`:
  the closed fingers held > 0.03 closure short of the command) is computed and logged in
  `turns.jsonl` `backend` but never shown.
- Containment in a dynamic box: resting partly on the 3 mm mat the box tips 0.65°, and a yaw-only
  interior test put the packet standing on the floor 0.6 mm "below" it — the scripted expert
  "failed". Fixed by testing in the box's own frame (`tasks.to_box_interior`).
- `pkill -f` over ssh killed the ssh session itself (its command line contains the pattern);
  processes are now killed by PID (DEVELOPMENT pitfalls).
