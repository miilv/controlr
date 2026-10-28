# 2026-10-28: SEC fixes on our side: redacted provider errors, one key source per process, credential-free children

Stream SEC of the Legality work. Input: robot-harness `docs/notes/SEC-VERIFICATION-2026-10-28.md` ("Ilia-side changes
proposed" 1-4; verdict table findings 1, 5, 6). robot-harness fixed these at its own boundary (`tools/perceiver/keyguard.py`,
capped wrappers); this PR fixes the source in our stack, so our own uncapped runs get the same protection. Branch
`fix/legal-sec-redaction`. 0 LLM calls, no GPU, no ssh. Fake keys only (`sk-ant-FAKE-0000-...`, an opaque
`FAKEopaque...`).

**Done:**
- `controlr/keyguard.py` (new, stdlib only, so the agentic perceiver's urllib client can import it without httpx):
  `redact` / `clip` (redact the whole text, then cut) / `safe_label`, `register` (exact values), `router_credentials`
  (one source), `child_env` (credential-free child environment). Patterns and `safe_label` follow robot-harness's
  keyguard (attributed in the module docstring), plus a `Basic` scheme and JSON-escaped quotes.
- Finding 1 (error bodies, exception text):
  - `controlr/llm/client.py`: `_error_message` redacts before its 400-character cut. Stream-error chunks and
    transport-exception texts are redacted and bounded. The client registers its own key, so a bare echo that matches
    no pattern is caught too.
  - `controlr/llm/decisions.py`: the same for its error, transport and malformed-body paths.
  - `dojo/perception/agentic/llm.py` `UrllibChat`: the error body (read up to 64 KiB; a token cut by that read cap is
    dropped) is redacted whole, then cut to 300 characters. Before, the raw first 300 bytes were kept. The exception
    path is redacted too.
  - `dojo/perception/agentic/runner.py`: one redaction step before anything is written. The error gets every pattern,
    in the transcript and in `stats["errors"]`. Every transcript line drops registered keys: reply text, tool output,
    the parsed answer.
- Finding 5 (response fields in provenance):
  - `LLMResult.headers` keeps only `KEEP_HEADER_PREFIXES`: the set `loop._diag_headers` already logged, so run logs do
    not change. Values are redacted, and a name with a credential word (`x-omniroute-api-key`) is dropped.
  - `finish_reason` is a safe label in `LLMClient`, `UrllibChat` and the runner's transcript.
  - The decision record's served `id` / `model` / `provider` are safe labels.
- Items 3 and 4 (one dotenv source):
  - `dojo/agent/llmio.py` no longer falls back to `~/skoltech/controlr/.env` / `~/controlr/.env`.
  - `agentic/llm.py` no longer merges `AGENTIC_ENV_FILE` with `OMNIROUTE_*` from the environment.
  - Rule, in both: the named file gives both values, else the environment gives both. A named file plus `OMNIROUTE_*`
    in the environment is refused, and so is a named file that lacks a value (no fallback).
  - `bridge_episode.sh` now names `DOJO_AGENT_DOTENV` (default `$HOME/controlr/.env`, the driver only).
    `official_eval.sh` / `integ_eval.sh` already named theirs.
  - All three exit 2 at the start, before any lock wait, when `OMNIROUTE_*` is exported as well. `isr_env.sh` names
    both files and is unchanged.
- Finding 6 (child environments): `keyguard.child_env` is now used by:
  - `j1_adapter.install_env()` (B1 worker; `model_calls=True` only for the agentic worker in `agentic/j1.py`);
  - `dojo/bridge/twin.py`;
  - the agent tools (`agentic/tools_rd.py`, `agentic/libero.py`);
  - the eval gate (`dojo/runtime/eval.py`).

  In `integ_eval.sh` the render and twin services start under `env -u AGENTIC_ENV_FILE`.
- Tests: `tests/test_sec_redaction.py` (32: patterns, the padded cut, LLMClient / DecisionsClient over
  `httpx.MockTransport`: 400 echo, padded, raw opaque echo, 5xx gave-up, stream error, transport error, header
  allowlist, safe labels, one source, child_env) and `tests/dojo/test_sec_env.py` (14). The second file covers:
  - a 127.0.0.1 server echoing `Authorization` in a 400, through UrllibChat and OmniChat into `run_agent`'s transcript
    and stats;
  - llmio under a fake HOME holding both old fallback files: none is read;
  - the refused merge;
  - real child processes (B1 worker, agentic worker, twin, LIBERO tool) that write only their variable NAMES.

  On main's sources (with only the new module added), 23 of the 46 tests fail. The 23 that pass are the new module's 21
  unit tests, one guard, and one case main already handled (an httpx JSON echo whose `message` field held no key). On
  the branch all 46 pass, and so does the full suite (1220 passed, 102 skipped, `test_llm_transcript` included).
- Docs: ENVIRONMENT.md (rows for `AGENTIC_ENV_FILE` / `DOJO_AGENT_DOTENV`, a "credentials in processes" section),
  ARCHITECTURE.md (keyguard module, LLM client redaction contract), DEVELOPMENT.md pitfall, BRIDGE-RUNBOOK.md (the
  manual AgentPolicy driver commands name the key file), BACKLOG (follow-ups below).

**Decisions:**
- **Denylist for child environments, not an allowlist.** The children read PATH, HOME, CUDA / EGL / DISPLAY,
  `PYTHON*`, `RD_*` / `DOJO_*` / `SIMLAB_*` / `J1_*`, and whatever RoboDojo's and Isaac's launchers read. An allowlist
  here would silently break the twin or a renderer the next time one of them reads a new variable. A credential, by
  contrast, has a recognisable name.
  - Dropped: the router names and the two file variables, `OPENROUTER_*` / `ANTHROPIC_*` / `OPENAI_*`,
    `CONTROLR_CODEX_AUTH`, any name with a KEY / TOKEN / SECRET / PASSWORD / AUTH / COOKIE word (whole words: `TOKENS`
    stays), and any variable whose value holds a registered key.
  - Kept: our localhost HMAC authkeys (`SIMLAB_TWIN_AUTHKEY`, `SIMLAB_RENDER_AUTHKEY`, `DOJO_PERCEPTION_AUTHKEY`,
    `HARNESS_SHIM_AUTHKEY`).
  - `SSH_AUTH_SOCK` is dropped too. No child of ours uses ssh.
- **Refuse rather than let the file win.** If the file silently overrode an exported `OMNIROUTE_*`, someone who
  exported a different key on purpose would use the wrong one. The scripts check at the start, so a scored block
  fails before it takes a lock. A perception service never quietly runs B1-only because its agent refused the key.
- **`install_env(model_calls=False)` by default.** A new caller that forgets the flag gets the safe environment. The
  agentic worker opts in (a one-line change in `agentic/j1.py`).
- **Headers: the allowlist is the set the run logs already kept.** `loop.py` was not touched (touching it needs Isaac
  tests), so `run.jsonl` is unchanged.
- **Reply text is not pattern-redacted.** Only exact registered keys are removed from it, and only in the transcript
  copy. A pattern pass could mangle a legitimate reply, and the messages sent back to the model are untouched (cache
  invariants: no serialisation change).

**Behaviour changes for runs:**
- `dojo.agent.llmio` needs a named file or exported `OMNIROUTE_*`. Affected:
  - `python -m dojo.agent.grounding` / `dojo.agent.replay_probe` on the dev box: set
    `DOJO_AGENT_DOTENV=~/skoltech/controlr/.env`;
  - a hand-started `dojo.bridge.driver --policy dojo.agent.policy:AgentPolicy` without `isr_env.sh`: set
    `DOJO_AGENT_DOTENV` (BRIDGE-RUNBOOK updated);
  - `bridge_episode.sh` does it itself.
- `integ_eval.sh`, `official_eval.sh` and `bridge_episode.sh` exit 2 when `OMNIROUTE_*` is exported. A Python
  process that has both sources raises `RuntimeError` when it builds its client.
- Dotenv lines of the form `export OMNIROUTE_...=` are now read. Before, neither reader took them.
- Error strings: UrllibChat collapses whitespace in the body; stream errors are capped at 400 characters.

**Found along the way:** the controlr side has the same environment-inheritance pattern. `robot/isaac/client.py`
starts the Isaac server with `dict(os.environ)` while `remote_run.sh` sources `.env`, so the Isaac server holds
`OMNIROUTE_*`. Deferred to BACKLOG: `robot/` changes need the Isaac tests on compute3, and this stream has no ssh.

**Not done / next:**
- The controlr robot servers above.
- `catalog_check`, `held_eval`'s pool, and git / nvidia-smi calls. Their parents hold no credentials, so converting
  them buys little.
- Scoping the `AGENTIC_ENV_FILE` / `DOJO_AGENT_DOTENV` exports in the run scripts to their single consumer. These are
  paths, not keys.
- robot-harness must carry `controlr/keyguard.py` when it re-imports our stack (BACKLOG).
- The shared Unix account on isr-lab stays the real limit (robot-harness open item 1). Nothing in-process fixes it.
