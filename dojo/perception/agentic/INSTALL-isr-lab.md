# Agentic perceiver on isr-lab: pinned install (D-C6)

Install dir: `~/data2/ilia/agentic/` = `controlr-agentic/` (git archive of `controlr/`, `dojo/`, `pyproject.toml` at the
commit in `controlr-agentic/COMMIT`) + this README + `run/` (logs). No venv of its own: it reuses the J1 install's venv
(`~/data2/ilia/j1/controlr-j1/.venv`, py3.11; it has no httpx, so the runner's stdlib HTTP client is used) and the J1
install's catalogue / backgrounds / colour map for the B1 hint. Verify the pins: `cd controlr-agentic/dojo/perception/agentic/vendor
&& md5sum -c MANIFEST.md5` (the protocol and the tools, byte-identical to robot-harness at the commit in `SOURCE.txt`).

## J1 (RoboDojo strict server) environment
```bash
A=/home/anywherevla/data2/ilia/agentic
export J1_PERCEIVER=dojo.perception.agentic.j1:perceive      # held_look comes from the same module
export J1_PERCEIVER_PATH=$A/controlr-agentic
export J1_PERCEIVER_COMMIT=$(cat $A/controlr-agentic/COMMIT)
export J1_TIMEOUT_S=300                                      # one perceive call = 3-6 LLM calls, 30-110 s measured
export J1_HELD_TIMEOUT_S=120
export J1_ILIA_PYTHON=/home/anywherevla/data2/ilia/j1/controlr-j1/.venv/bin/python
export J1_ILIA_HOME=/home/anywherevla/data2/ilia/j1          # catalog/, bg/, color_calib.json for the B1 hint
export J1_GL=egl                                             # the B1 hint renders with GL (mesa = CPU)
export AGENTIC_MODEL=claude/claude-sonnet-5-5                # pinned per measurement arm (e.g. claude/claude-opus-5-5)
export AGENTIC_MAX_TURNS=8
export AGENTIC_B1=1                                          # B1 output as the hint, and the fallback if the agent fails
export AGENTIC_HELD=b1_then_agent                            # B1 held_look; the agent only when B1 conf < 0.5 or dark_risk
export AGENTIC_GATE=always                                   # measurement: the agent on every call; budget = first call + <= 1 more
export AGENTIC_REFINE=1                                      # B1's render-and-compare from the agent's points (needs GL like B1)
export AGENTIC_KEEP_UNMENTIONED=1                            # B1 objects the agent did not mention stay at half confidence
export AGENTIC_LOG_DIR=$A/run/<run-name>                     # per-call workspace, transcript.jsonl, calls.jsonl
export AGENTIC_WORKER_LOG=$A/run/agentic_worker.log
export AGENTIC_ENV_FILE=<file with OMNIROUTE_BASE_URL and OMNIROUTE_API_KEY>   # NOT installed: the key is the operator's
```
Every perceive reply's `notes` carries the model, `system_md5`, `protocol_md5`, turns, calls and the result source
(`agent` | `b1_fallback`). `calls.jsonl` has the tokens and per-call seconds.

## LIBERO (picks.json per capture)
```bash
cd $A/controlr-agentic && LIBERO_PY=$B/.venv-liberopro/bin/python LIBERO_HARNESS=$B/harness AGENTIC_ENV_FILE=... \
  AGENTIC_MODEL=claude/claude-sonnet-5-5 AGENTIC_LOG_DIR=$A/run/<run-name> \
  /home/anywherevla/data2/ilia/j1/controlr-j1/.venv/bin/python -m dojo.perception.agentic.libero \
  --cap <capture dir D> --instruction "<instruction>" --cell object_task --out <D>/picks.json --reply <rgbd_reply_01.json>
```
Exit 0 = picks.json written; 2 = no answer. The template's `locate` / `run` / `finish` then proceed unchanged.
