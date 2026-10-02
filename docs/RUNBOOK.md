# Runbook: когда что-то сломалось

Машины и пути — [ENVIRONMENT.md](ENVIRONMENT.md). Прошлые случаи — [incidents/](incidents/).
compute3 — общая машина: по умолчанию только чтение, своё — только в `~/controlr*`.

## 1. compute3: нет GPU (`nvidia-smi` не видит драйвер)

Типичная причина — авто-обновление ядра без модуля NVIDIA под новое ядро
([инцидент 2026-10-02](incidents/2026-10-02-compute3-nvidia-driver.md)).

```bash
ssh compute3 'uname -r; lsmod | grep -c nvidia; dpkg -l | grep linux-modules-nvidia | awk "{print \$2}"'
# модуля под текущее ядро нет -> (только по явной просьбе владельца; sudo без пароля есть)
ssh compute3 'sudo apt-get install -y linux-modules-nvidia-595-open-$(uname -r) && sudo modprobe nvidia nvidia_uvm && nvidia-smi'
```
Ребут не нужен, если `modprobe` прошёл. Ребут убивает чужие сессии/джобы — только с согласия.

## 2. Isaac-сервер

- Обычно им управляет `scripts/remote_run.sh`: поднимает, если порт `7801` свободен, ждёт
  готовности (до 600 с), гасит после прогона. Лог: `~/controlr/runs/isaac_server.log`.
- Тесты поднимают свой сервер на порту `7821` (`CONTROLR_ISAAC_TEST_PORT`), не на боевом.
- Ручной запуск (из `~/controlr`): `setsid nohup bash scripts/isaac_server.sh --port 7801 > runs/isaac_server.log 2>&1 &`
- Кто держит порт / как погасить свой сервер:
  ```bash
  ssh compute3 'ss -ltnp "sport = :7801"; pgrep -af "controlr/robot/isaac/server.py"'
  ssh compute3 'pkill -f "controlr/robot/isaac/server.py --phantom"'   # только наш процесс
  ```
- «Isaac server died» сразу → смотри хвост лога: чаще всего GPU (п.1) или `PHANTOM_ROOT`/`ISAAC_SIM_ROOT`.
- Эпизод закончился `unstable` → PhysX разошёлся (пережатый объект/удар). Это исход эпизода,
  не баг харнесса; повторяется на одном seed — смотри контакты в `turns.jsonl` и BACKLOG.

## 3. Роутер (omniroute) и модели

| Симптом | Причина / действие |
|---|---|
| ответ за ~0.4 с без usage | реплей кэшированного **ответа** роутером на одинаковый запрос; проверь `llm.request_nonce: true` |
| `400 unsupported_image_block` | маршрут без картинок (`dva/*`); бери `claude/`, `cc/`, `no-think/`, `cx/` |
| пустой ответ, `finish_reason` = length | thinking съел `llm.max_tokens`; подними лимит или `no-think/` маршрут |
| модели нет в `/models`, но нужна | пробуй прямой вызов — роутер принимает часть неперечисленных id (так было с `claude-sonnet-5-5`) |
| 429 / 5xx | клиент сам ретраит с backoff; при подписочных маршрутах реальный предел — лимиты подписки |

Список моделей: `uv run controlr models --filter claude`.

## 4. Кэш не работает (растёт стоимость, `cache_read` ≈ 0)

1. `summary.json` → `cache_regressions` (ходы, где кэш просел) и доля чтения по ходам в `turns.jsonl`.
2. Частые причины: изменился ранний байт транскрипта (недетерминированный текст/картинка),
   поменяли effort посреди эпизода, ход добавил > 20 блоков (lookback), префикс короче минимума
   модели (Haiku 4.5 — 4096 токенов), запросы ушли на разные апстрим-аккаунты роутера.
3. Изолированная проверка: `uv run controlr bench-cache --model <id> --turns 20` (⚠️ ~20 вызовов).

## 5. Диск

Локально прогоны (`runs/`) и клоны (`research/repos/`) — основные потребители; на compute3 —
`~/controlr/runs`. Перед тяжёлой работой: `df -h /`. Кэши `~/.cache/uv`, `~/.cache/huggingface`
общие — чистить только своё и только по договорённости
([инцидент 2026-10-02](incidents/2026-10-02-research-run-side-effects.md)).

## 6. Реальный UR3 (когда появится бэкенд)

Только с явного разрешения на сессию и человеком у e-stop. Драйверы и `SafetyMonitor` — из PHANTOM.
Порядок аварии: e-stop → `robot.hold()` → разбор логов → инцидент.
