# Документация AdaptiveML DSS

Это единая документация по текущему состоянию проекта `AdaptiveML DSS`.

Здесь собраны:

- установка и запуск;
- архитектура backend и ML-контура;
- API-эндпоинты;
- CLI-утилиты;
- визуализация результатов;
- сводка по реализованной функциональности.

## Что уже реализовано

Проект уже собран в рабочий сквозной поток:

`dataset -> training -> registry -> prediction -> explanation -> decision -> visualization`

На практике это означает, что система уже умеет:

- принимать `CSV`, `XLSX`, `XLS`;
- обучать модели через `LightAutoML`;
- использовать `sklearn` fallback;
- делать предсказания на новых данных;
- строить локальные объяснения через `SHAP`;
- генерировать рекомендации через `Experta`;
- хранить версии датасетов и моделей;
- показывать проекты, обучение, историю моделей и график в UI на `/app/`;
- хранить holdout-предсказания и training diagnostics рядом с моделью;
- запускаться через API и standalone CLI;
- собирать HTML-отчёт с графиками.

## Карта документации

- [Установка](setup.md) — установка, зависимости и нюансы окружения.
- [Архитектура](architecture.md) — как устроен backend, ML, explainability и DSS.
- [Схема БД](database.md) — описание компактной PostgreSQL-схемы и роли `JSONB`.
- [Общая ER-диаграмма](database-single-diagram.md) — одна диаграмма со всеми основными сущностями.
- [API](api.md) — реализованные эндпоинты и их поток выполнения.
- [CLI и утилиты](cli.md) — утилиты для прямого запуска, smoke-check и генерации отчёта.
- [Визуализация](visualization.md) — как сейчас устроен визуальный слой и отчёты.
- [Обзор реализации](implementation-overview.md) — полная техническая сводка по реализованной итерации.

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

Запустить локальный сайт документации:

```bash
mkdocs serve
```

Или поднять documentation-site в Docker:

```bash
docker compose up -d docs
```

Собрать статическую версию:

```bash
mkdocs build
```
