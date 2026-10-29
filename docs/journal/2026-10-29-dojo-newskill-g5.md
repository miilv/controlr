# 2026-10-29 — stream N: Misha's G5 verification items 3 / 4 / 9 / 13 and ports P2-3 / P2-6

Sources: robot-harness docs/notes/G5-VERIFICATION-2026-10-29.md ("For Ilia") and REVIEW-ILIA-PORTS-2026-10-29.md.

| item | where | change | candidate behaviour |
|---|---|---|---|
| G5 #3 late agent search | `runtime/policy._late_search`, `agent/policy._AgentCtx._search`, `belief.merge_found` | the search returns None when it found nothing; only new finds are added to the live belief (held, released, poses, checkpoint kept) instead of `init_from_record` + `make_skills` | agent program only (not the rule supervisor) |
| G5 #4 unmeasured release | `belief` (`_unmeasured`, `measured_since_release`), `skills.verify_place` | `RuntimeConfig.place_verdict` "measured": a release no measurement saw since is "unverified", and only measured labels skip the head check | behind the flag, default "belief" = today |
| G5 #9 look fusion | `runtime/policy._look` | the fused estimate wrapped as `_scan` is: an error keeps the frame-0 record, so the buffered home runs with the first skill | failure path only |
| G5 #13 agentic budget | `RuntimeConfig.agentic_count_held` | False: the agent's held looks no longer spend `agentic_max` | new field, default True = today (presets frozen) |
| ports P2-3 | `ported()` | a ball pour verified with no ball in the belief is not verified ("no ball evidence") | report / verdict only (ok unchanged) |
| ports P2-6 | `harness_robot` | the official gripper key is read only behind `provider.grip_check` (D-C11); ported runs record `grip_source` | unchanged for the candidate (grip_check on) |

Tests: `tests/dojo/test_g5_runtime.py`; in `test_port_adapter.py`, `test_a_ball_pour_without_ball_evidence_is_not_verified`
and `test_the_official_gripper_key_is_not_read_without_the_grip_check`.
