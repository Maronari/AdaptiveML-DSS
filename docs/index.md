# AdaptiveML DSS Docs

Это единая документация по текущему состоянию проекта `AdaptiveML DSS`.

Здесь собраны:

- установка и запуск;
- архитектура backend и ML-контура;
- API endpoints;
- CLI-утилиты;
- визуализация результатов;
- сводка по реализованной функциональности.

## Что уже реализовано

Проект уже собран в рабочий end-to-end поток:

`dataset -> training -> registry -> prediction -> explanation -> decision -> visualization`

На практике это означает, что система уже умеет:

- принимать `CSV`, `XLSX`, `XLS`;
- обучать модели через `LightAutoML`;
- использовать `sklearn` fallback;
- делать предсказания на новых данных;
- строить локальные объяснения через `SHAP`;
- генерировать рекомендации через `Experta`;
- хранить версии датасетов и моделей;
- запускаться через API и standalone CLI;
- собирать HTML-отчёт с графиками.

## Карта документации

- [Setup](setup.md) — установка, зависимости и нюансы окружения.
- [Architecture](architecture.md) — как устроен backend, ML, explainability и DSS.
- [API](api.md) — реализованные endpoints и их поток выполнения.
- [CLI & Tooling](cli.md) — утилиты для прямого запуска, smoke-check и генерации отчёта.
- [Visualization](visualization.md) — как сейчас устроен визуальный слой и отчёты.
- [Implementation Overview](implementation-overview.md) — полная техническая сводка по реализованной итерации.

## Быстрый старт

```bash
scripts/setup_full_env.sh python3.13
source .venv/bin/activate
uvicorn backend.main:app --reload
```

## Сборка документации

Установить зависимости:

```bash
python -m pip install -r requirements-docs.txt
```

Запустить локальный docs-site:

```bash
mkdocs serve
```

Собрать статическую версию:

```bash
mkdocs build
```
