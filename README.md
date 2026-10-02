# controlr

Робо-харнесс в духе coding-агентов: кадры камеры + системный промпт «мануал робота» + задача →
vision-LLM по OpenAI-совместимому API → **низкоуровневые числовые действия** и статус. Пошагово,
append-only, с полным prompt-кэшем. Экспериментальная платформа: может ли большая API-модель
быть контроллером робота без VLA — и при каком промпте, представлении кадра и формате действий.

*A robotics harness built like a coding-agent harness: frames + a robot operating manual + a task
→ vision LLM → low-level numeric actions, turn by turn, fully prompt-cached.*

Сейчас: UR3 CB3 + Robotiq в Isaac Sim 6.0 (откалиброванная сцена PHANTOM), задачи
waffle pick-and-place и reach; модели — через omniroute. Результаты — [docs/experiments/](docs/experiments/README.md).

```bash
uv sync --extra dev && cp .env.example .env      # ключ роутера — у владельца
uv run pytest -q
uv run controlr run -c configs/mock.yaml --fake-llm   # весь цикл без сети и GPU
```

| Документ | Зачем |
|---|---|
| [CLAUDE.md](CLAUDE.md) | правила и карта для агентов (`AGENTS.md` — то же) |
| [HUMAN.md](HUMAN.md) | как ставить задачи агенту |
| [ARCHITECTURE.md](ARCHITECTURE.md) | устройство и контракты модулей |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | плейбук разработки |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | протокол экспериментов и оси конфига |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) · [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | эксплуатация, машины, переменные |
| [docs/BACKLOG.md](docs/BACKLOG.md) | очередь задач |
| [research/](research/README.md) | литобзор и разбор чужих харнессов |
