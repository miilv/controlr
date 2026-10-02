# controlr

A robotics harness built like a coding-agent harness: camera frames + a robot "operating manual"
system prompt + a task → a vision LLM over an OpenAI-compatible API → **low-level numeric
actions** and a status. Turn by turn, append-only, fully prompt-cached. An experimentation
platform: can a large API model be a robot's controller without a VLA — and with which prompt,
frame representation and action format?

Today: UR3 CB3 + Robotiq in Isaac Sim 6.0 (PHANTOM's calibrated scene), tasks waffle
pick-and-place and reach; models through omniroute. Results — [docs/experiments/](docs/experiments/README.md).

```bash
uv sync --extra dev && cp .env.example .env      # router key from the owner
uv run pytest -q
uv run controlr run -c configs/mock.yaml --fake-llm   # the full loop, no network, no GPU
```

| Document | Purpose |
|---|---|
| [CLAUDE.md](CLAUDE.md) | rules and map for agents (`AGENTS.md` is the same file) |
| [HUMAN.md](HUMAN.md) | how to give the agent tasks |
| [ARCHITECTURE.md](ARCHITECTURE.md) | design and module contracts |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | development playbook |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | experiment protocol and config axes |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) · [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | operations, machines, variables |
| [docs/BACKLOG.md](docs/BACKLOG.md) | task queue |
| [research/](research/README.md) | literature review and teardown of other harnesses |
