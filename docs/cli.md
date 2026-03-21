# CLI и утилиты

## Полная установка окружения

Скрипт:

- `scripts/setup_full_env.sh`

Запуск:

```bash
scripts/setup_full_env.sh python3.13
```

Назначение:

- создать `.venv`;
- поставить полный runtime stack;
- обойти проблемные места в установке `LightAutoML`;
- поставить CPU-only `torch`.

## Прямой запуск LightAutoML

Скрипт:

- `scripts/run_lightautoml.py`

Пример:

```bash
python scripts/run_lightautoml.py \
  --data datasets/dataset__1.xlsx \
  --target "Электропотребление"
```

Что делает:

- читает файл в `DataFrame`;
- применяет `TabularPreprocessor`;
- обучает `LightAutoML`;
- считает holdout metrics;
- может сохранить модель и predictions.

Полезные аргументы:

- `--task`
- `--timeout`
- `--cpu-limit`
- `--cv`
- `--save-model`
- `--predictions-out`

## Smoke-check полного API-контура

Скрипт:

- `scripts/check_full_flow.py`

Пример:

```bash
python scripts/check_full_flow.py \
  --data datasets/dataset__1.xlsx \
  --target "Электропотребление" \
  --project-id smoke-demo \
  --sample-size 2
```

Что проверяет:

- `/health`
- `/training/run/file`
- `/predictions/run`
- `/explanations/run`
- `/decision/run`

Это минимальный сквозной smoke test для уже поднятого API.

## Генерация HTML-репорта

Скрипт:

- `scripts/render_visual_report.py`

Пример:

```bash
python scripts/render_visual_report.py \
  --project-id demo \
  --sample-size 40
```

Что делает:

- загружает артефакт champion-модели из реестра;
- берёт связанный датасет;
- строит sample предсказаний и объяснений;
- собирает DSS-рекомендации;
- генерирует HTML с графиками.

Результат:

- `storage/artifacts/<project-id>/visual-report.html`

## Тесты

Базовый прогон:

```bash
pytest -q
```

На этой итерации тесты покрывают:

- healthcheck;
- поток обучения, предсказания и рекомендаций;
- загрузку `xlsx`;
- preprocessing;
- применение preprocessing в пути предсказания.
