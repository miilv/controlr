# controlr — робо-харнесс

Харнесс в духе coding-агентов, только для робота: кадры камеры + системный промпт
(«мануал робота») + задача → vision-LLM по API → **низкоуровневые числовые действия**
(`MOVE ee_delta dx dy dz`, `GRIP`, `STATUS`) → safety-оболочка → робот → новый кадр +
фидбек. Пошаговый цикл (робот стоит, пока модель думает), транскрипт append-only и
целиком закэширован. Цель — экспериментальная платформа: проверить, что большие
API-модели с правильным промптом и архитектурой справляются как контроллер без VLA.

Сейчас: UR3 CB3 + Robotiq в Isaac Sim 6.0 (откалиброванная сцена PHANTOM) на `compute3`;
модели — через один OpenAI-совместимый роутер (omniroute). Дальше — реальный UR3.

## Компоненты

| Код | Что |
|---|---|
| `controlr/llm/` | стриминговый клиент (тайминги, usage, early stop), кэш-маркеры, append-only транскрипт, `FakeLLM` |
| `controlr/protocol/` | грамматика ответа (MOVE/GRIP/HOLD/STATUS) и текст фидбека |
| `controlr/prompts/` | системный промпт = мануал робота (`system_v0.md`), промпт планировщика |
| `controlr/observation/` | рендереры кадров: resize, grid, ee_marker, axes, diff, heatmap, tile |
| `controlr/robot/` | `Robot` ABC, `spec`, кинематика UR3, `SafetyEnvelope`, mock, replay, `isaac/` (сервер в питоне Isaac + клиент) |
| `controlr/loop.py`, `runlog.py`, `cli.py`, `bench/` | эпизод (планировщик + цикл), логи прогона, CLI, бенчмарки кэша/латентности, свипы |
| `configs/` | эксперименты в YAML (`extends:`), каждая ось эксперимента — поле конфига |
| `scripts/` | деплой и запуск на compute3, Isaac-сервер, видео прогона |
| `research/` | литобзор и разбор чужих харнессов (англ.) |
| `runs/` | результаты прогонов — **не в git** |

## Команды

```bash
uv sync --extra dev                         # окружение
uv run pytest -q                            # юнит-тесты (live/isaac сами скипаются) — то же гоняет CI
uv run controlr prompt -c configs/sim_waffle.yaml     # как модель увидит мануал (без робота и LLM)
uv run controlr run -c configs/mock.yaml --fake-llm   # сухой прогон всего пайплайна, без сети

# compute3 (Isaac): деплой, затем запуск — Isaac-сервер поднимется и остановится сам
scripts/deploy.sh
scripts/remote_run.sh run -c configs/sim_waffle_yaw.yaml --seeds 0,1,2,3
ssh compute3 'cd ~/controlr && CONTROLR_ISAAC=1 .venv/bin/python -m pytest -q tests/'   # ~20 мин

# анализ
uv run controlr report runs/<run_dir> [--csv out.csv]
uv run --no-project --with pillow python scripts/turn_video.py runs/<run_dir> docs/video/<name>.mp4 "<title>"
uv run controlr bench-cache --model claude/claude-sonnet-5-5 --turns 20   # ⚠️ живые вызовы
```

## Правила (обязательно)

**Шаг 0 любой задачи по коду: прочитай [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) и [ARCHITECTURE.md](ARCHITECTURE.md) ДО первого изменения.** Задача-эксперимент — ещё и [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

1. **Модель — это контроллер.** Никаких высокоуровневых скиллов («возьми коробку») и VLA в
   контуре действий. Альтернативы — переключатели конфига, а не форки кода. Каждая ось
   эксперимента — поле в `controlr/config.py`; дефолты молча не меняем (ломает сравнимость
   прогонов) — смена дефолта = запись в journal.
2. **Цикл:** ветка (`git worktree add ../controlr-<задача> -b feat/<задача>` для параллельных
   задач) → тесты локально → если трогал `robot/`, `loop.py`, safety или Isaac — `scripts/deploy.sh`
   и Isaac-тесты на compute3 → **PR открывай сам** → зелёный CI → merge. Прямой push в `main` —
   только docs (journal, experiments, incidents).
3. **Живые вызовы LLM = деньги.** Только в задачах с явным бюджетом (число вызовов). Юнит-тесты —
   только `FakeLLM` / `httpx.MockTransport`; живые тесты помечены `@pytest.mark.live` и идут лишь
   с `CONTROLR_LIVE=1`. Ключ — только `OMNIROUTE_*` из `.env`, **никогда** креды сессии агента
   (`ANTHROPIC_AUTH_TOKEN` и т.п.). Фактический расход (вызовы, токены) — в отчёт.
4. **compute3 — общая машина** (на ней живёт и чужая работа). Пишем только в `~/controlr*`; GPU
   наш, CPU/RAM — умеренно; чужие процессы не трогаем; Isaac-сервер гасим по завершении.
   `apt`/драйверы/ребут — только по явной просьбе ([RUNBOOK](docs/RUNBOOK.md)). PHANTOM
   (`~/phantom-icra-2027` на compute3, `~/skoltech/research` локально) **не модифицируем** —
   только импорт или копия с атрибуцией.
5. **Секреты:** `.env` живёт только локально и в `~/controlr/.env` на compute3 — не в
   dev-копиях, не в git, не в `runs/`, не в логах. Новая переменная → `.env.example` +
   [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
6. **Инварианты кэша:** транскрипт append-only, история не редактируется; картинка кодируется
   один раз и хранится байтами; текст фидбека детерминирован (фиксированные десятичные);
   reasoning effort внутри эпизода не меняется. Любое изменение сериализации → тест стабильности
   префикса + проверка `cache_read` на живом прогоне.
7. **Безопасность — в коде, не в промпте.** Каждое действие идёт через `SafetyEnvelope`; клэмпы
   и остановки возвращаются модели фидбеком. Реальный UR3 — только через драйверы PHANTOM +
   `SafetyMonitor` и только с явного разрешения на сессию (человек у e-stop).
8. **Эксперимент воспроизводим или не считается:** конфиг в git, seeds × повторы, при сравнении
   контрол-моделей — один и тот же план (`planner.plan_file`), n и разброс в отчёте; «2/4» — шум,
   не вывод. Отчёт — `docs/experiments/YYYY-MM-DD-<slug>.md` + строка в индексе; видео успехов и
   типичного провала — `scripts/turn_video.py`.
9. **Агенты-исследователи без побочных эффектов:** не качать модели/видео/аудио, не ставить
   torch/CUDA, клоны — shallow в `/tmp` и удалять; держать диск в узде (см.
   [incidents](docs/incidents/)).
10. **Хвосты в том же PR:** закрыл пункт [BACKLOG](docs/BACKLOG.md) — вычеркни, нашёл мину —
    добавь; изменил поведение — обнови [ARCHITECTURE](ARCHITECTURE.md)/[DEVELOPMENT](docs/DEVELOPMENT.md);
    значимая работа — `docs/journal/YYYY-MM-DD-<slug>.md`; инцидент — `docs/incidents/`.
    Стабильные доки — без дат и статусов, всё сиюминутное — в journal/experiments.
11. **Git-гигиена:** первым делом `git status`; `git add` точечно (никаких `-A`/`.`);
    `runs/`, `.env`, `research/repos/` не коммитим; видео — только короткие и нужные отчёту.

## Куда смотреть (по ситуации)

| Когда | Файл |
|---|---|
| Любая задача по коду | [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — плейбук: окружение, цикл, тесты, типовые изменения, подводные камни |
| Как устроена система, контракты модулей | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Поставить / прочитать эксперимент | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), отчёты — [docs/experiments/](docs/experiments/) |
| Что-то сломалось (compute3, Isaac, роутер, кэш) | [docs/RUNBOOK.md](docs/RUNBOOK.md) + [docs/incidents/](docs/incidents/) |
| Выбрать задачу / известные мины | [docs/BACKLOG.md](docs/BACKLOG.md) |
| Env-переменные, машины, пути | [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) |
| Контекст прошлых работ | [docs/journal/](docs/journal/), ревью кода — [docs/reviews/](docs/reviews/) |
| Почему так, что пробовали другие | [research/README.md](research/README.md) |
| Как ставить задачи агенту (для человека) | [HUMAN.md](HUMAN.md) |
