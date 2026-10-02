# Окружение: переменные, машины, пути

Новая переменная → `.env.example` (без значения) + строка здесь, в том же PR.

## Переменные окружения

| Переменная | Где нужна | Что |
|---|---|---|
| `OMNIROUTE_BASE_URL` | везде, где живые вызовы | OpenAI-совместимый роутер (`.../v1`) |
| `OMNIROUTE_API_KEY` | там же | ключ роутера; только `.env` локально и `~/controlr/.env` на compute3 |
| `CONTROLR_ISAAC_AUTHKEY` | compute3 | общий секрет клиент↔Isaac-сервер; не задан → клиент сам генерирует на запуск (фолбэк `controlr-isaac-dev` — только для дев-бокса) |
| `CONTROLR_ISAAC_READY_FILE` | compute3 | файл-флаг готовности сервера (выставляет `remote_run.sh`) |
| `ISAAC_SIM_ROOT` | compute3 | дефолт `/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0` |
| `PHANTOM_ROOT` | compute3 | дефолт `/home/physicalai/phantom-icra-2027/phantom` |
| `CONTROLR_LIVE=1` | тесты | включить живые тесты (`@pytest.mark.live`) — тратит деньги |
| `CONTROLR_ISAAC=1` | тесты на compute3 | включить Isaac-тесты (`@pytest.mark.isaac`) |
| `CONTROLR_ISAAC_TEST_PORT` | тесты | порт тестового сервера (дефолт 7821, не боевой 7801) |
| `CONTROLR_ISAAC_IMG_DIR` | тесты | куда сохранить кадры из isaac-тестов для просмотра |

## Машины

| Хост (`~/.ssh/config`) | Что это | Для чего нам | Правила |
|---|---|---|---|
| локально | дев-бокс без GPU | код, юнит-тесты, mock/replay, отчёты, видео | — |
| `compute3` (`physicalai`) | Ubuntu 24.04, py3.12, RTX 5090 32 GB, Isaac Sim 6.0, PHANTOM | Isaac-бэкенд, все прогоны | общая машина; только `~/controlr*`; GPU наш |
| `compute2` (`isr-lab-4`, root) | Ubuntu 22.04, RTX 4090, Isaac Sim 5.1 в docker | запасной хост; кандидат под RoboDojo (Isaac 5.1) | общая лаб-машина; только `/root/controlr*` |
| `nuc` | NUC у реального стенда (PHANTOM деплой) | будущий бэкенд реального UR3 | не трогать без задачи |

## Пути

| Что | Где |
|---|---|
| деплой | `compute3:~/controlr` (`scripts/deploy.sh`, venv `.venv`, `.env`) |
| прогоны | `runs/` локально и `compute3:~/controlr/runs` (`remote_run.sh` синкает назад) |
| Isaac Sim | `compute3:/home/physicalai/AAAI_MultiAgenticSIM/isaac-sim-6.0` |
| PHANTOM | `compute3:~/phantom-icra-2027/phantom`, локально `~/skoltech/research` (не модифицируем) |
| реальный UR3 | IP и железо — `configs/hardware.yaml` в PHANTOM |
