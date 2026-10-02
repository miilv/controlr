# Как разрабатывать controlr (плейбук для людей и агентов)

Правила — в [CLAUDE.md](../CLAUDE.md), устройство — в [ARCHITECTURE.md](../ARCHITECTURE.md),
постановка экспериментов — в [EXPERIMENTS.md](EXPERIMENTS.md). Здесь — как делать изменения.

## 0. Бутстрап

```bash
git clone git@github.com:miilv/controlr.git && cd controlr
uv sync --extra dev                 # python >= 3.10; зависимости: httpx, numpy, pillow, pyyaml
cp .env.example .env                # OMNIROUTE_BASE_URL / OMNIROUTE_API_KEY (ключ у владельца)
uv run pytest -q                    # должно быть зелёным без сети и без GPU
uv run controlr run -c configs/mock.yaml --fake-llm   # весь цикл на mock-роботе, 0 вызовов
```

Isaac и GPU — только на `compute3` (`ssh compute3`, см. [ENVIRONMENT.md](ENVIRONMENT.md)).
Локально Isaac не нужен: mock-робот и replay-робот покрывают всё, кроме физики.

## 1. Цикл задачи

```bash
git status                                            # чужое не трогаем
git worktree add ../controlr-<задача> -b feat/<задача>   # или просто ветка, если задача одна
# ... изменения + тесты к ним ...
uv run pytest -q                                      # юнит
uv run controlr prompt -c configs/sim_waffle.yaml     # если трогал промпт/грамматику/фидбек: перечитай как модель
scripts/deploy.sh                                     # если трогал robot/, loop.py, safety, isaac/
ssh compute3 'cd ~/controlr && CONTROLR_ISAAC=1 .venv/bin/python -m pytest -q tests/'
gh pr create --fill                                   # PR сам; CI = юнит-тесты + py_compile + secret scan
gh pr merge --merge --delete-branch                   # после зелёного CI
git worktree remove ../controlr-<задача>
```

Живые прогоны (деньги) — только в задаче с бюджетом, по протоколу [EXPERIMENTS.md](EXPERIMENTS.md).

## 2. Тесты: три яруса

| Ярус | Что | Как запускать | Где |
|---|---|---|---|
| юнит | всё без сети и без Isaac: грамматика, фидбек, кэш-маркеры, транскрипт, кинематика, safety, loop на `FakeLLM`+`MockRobot`, клиент на `httpx.MockTransport` | `uv run pytest -q` | локально + CI |
| isaac | сервер Isaac, трекинг движений, согласованность кинематики с симом, скриптовый эксперт | `CONTROLR_ISAAC=1 pytest -m isaac` (полный прогон ~20 мин) | только compute3 |
| live | реальные вызовы через роутер | `CONTROLR_LIVE=1 pytest -m live` | по бюджету, руками |

Правило: новая логика → юнит-тест. Изменил исполнение движений/контакты/сцену → isaac-тест.
Изменил сериализацию сообщений → тест стабильности префикса (`tests/test_llm_transcript.py`)
и один живой прогон с проверкой `cache_read` (см. RUNBOOK §кэш).

## 3. Карта модулей: что менять и как проверять

| Хочу | Трогаю | Проверяю |
|---|---|---|
| новый формат/режим действий | `protocol/grammar.py` (+ `config.ActionConfig`), `robot/safety.py` | `test_grammar` (примеры из спеки парсятся под всеми конфигами), `test_safety`; `controlr prompt` |
| текст фидбека / STATE | `protocol/feedback.py` | `test_feedback` (детерминизм!), `controlr prompt` |
| мануал робота / планировщик | `prompts/system_v0.md`, `prompts/planner_v0.md`, `prompts/builder.py` | `test_prompts`; новый вариант промпта = новый файл `system_vN.md` + `prompt.system: system_vN`, старый не правим (сравнимость) |
| наблюдение (оверлеи, diff, тайлинг) | `observation/renderers.py` | `test_renderers` (детерминизм, проекция известной точки) |
| новая задача в Isaac | `robot/isaac/tasks.py` (+ сервер, если нужны объекты) | `test_isaac_tasks` (без GPU) + скриптовый эксперт в isaac-тестах; кадры глазами (`docs/img/`) |
| новый бэкенд (реальный UR3 и т.п.) | `robot/<backend>.py`, фабрика `robot/__init__.py` | контракт `robot/base.py`; mock-тесты как образец |
| кэш / роутер / тайминги | `llm/` | `test_llm_*`; живой `bench-cache` (бюджет) |
| ось эксперимента | поле в `config.py` + `configs/base.yaml` | `test_cli`/`test_loop`; дефолт не меняет поведение |

## 4. Подводные камни (на которые уже наступили)

- **omniroute реплеит ответы** на байт-в-байт одинаковые запросы (~0.4 с, без usage). Поэтому
  `llm.request_nonce: true` кладёт run id в первый ход. В бенчмарках — тоже.
- **Effort через роутер — суффиксом id** (`claude/claude-opus-5-5-xhigh`), `extra_body.reasoning_effort`
  практически игнорируется.
- **Opus 5.5 не умеет выключать thinking**, Sonnet 5.5 думает и съедает `max_tokens` → пустой
  ответ; `llm.max_tokens` считает и thinking-токены. Для быстрых ходов — `no-think/...` маршруты.
- **Границу кэша на `claude/`/`cc/` выставляет сам роутер**, наши маркеры там, похоже, ни на что
  не влияют (не проверено пробой — см. BACKLOG). Haiku 4.5 не кэширует префикс < 4096 токенов.
- **Isaac идёт ~0.24× реального времени**: 100 мм движения ≈ 4 с — это часто больше, чем LLM.
- **Физика PhysX взрывается** при пережатом объекте; такие эпизоды заканчиваются `unstable`, а
  сервер переживает NaN (починено) — но смотри логи сервера при странностях.
- **Промпт и грамматика должны совпадать с конфигом**: примеры в мануале генерируются из
  `grammar_spec`, не пиши примеры руками.
- **Одна камера D435 под углом** плохо даёт глубину: ошибки 80–200 мм по глубине — главная причина
  провалов на сегодня (гипотеза — в BACKLOG).
- **`uv run` в корне репо создаёт `.venv`**; на compute3 venv в `~/controlr/.venv` собирает `deploy.sh`.

## 5. Что где живёт

| Что | Где | Правило |
|---|---|---|
| правила для агента | `CLAUDE.md` (`AGENTS.md` — симлинк) | коротко, без дат |
| архитектура и контракты | `ARCHITECTURE.md` | обновляется в PR, меняющем поведение |
| плейбук / рантайм-операции / окружение | `docs/DEVELOPMENT.md`, `RUNBOOK.md`, `ENVIRONMENT.md` | стабильные, без дат |
| протокол экспериментов | `docs/EXPERIMENTS.md` | стабильный |
| отчёты экспериментов | `docs/experiments/YYYY-MM-DD-<slug>.md` + индекс в README | датированы, задним числом не правим |
| журнал работ | `docs/journal/YYYY-MM-DD-<slug>.md` | датирован |
| ревью кода и fixlog | `docs/reviews/YYYY-MM-DD-<slug>.md` | датированы |
| инциденты | `docs/incidents/YYYY-MM-DD-<slug>.md` | постмортем в течение суток |
| очередь задач | `docs/BACKLOG.md` | вычёркивать в PR задачи |
| литература | `research/` | англ.; `research/repos/` — клоны, не в git |
| картинки / видео для доков | `docs/img/`, `docs/video/` | только то, на что ссылается отчёт |
| прогоны | `runs/` | не в git; отчёт ссылается на имя каталога |
