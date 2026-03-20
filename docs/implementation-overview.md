# Implementation Overview

Этот документ фиксирует, что именно было реализовано в текущей итерации проекта, как это устроено и как этим пользоваться.

## Что уже работает

Проект собран в рабочий end-to-end контур:

`dataset -> training -> model registry -> prediction -> explanation -> decision -> visualization`

На практике это означает, что система уже умеет:

- принимать табличные датасеты в `CSV`, `XLSX`, `XLS`;
- валидировать данные и определять тип ML-задачи;
- обучать модель через `LightAutoML`;
- использовать `sklearn` fallback, если реальный `LightAutoML` недоступен;
- делать предсказания на новых данных;
- строить локальные объяснения через `SHAP`;
- формировать рекомендации через `Experta`;
- хранить версии датасетов и моделей в файловом registry;
- запускаться через API, CLI и smoke-скрипты;
- генерировать HTML-отчёт с визуализацией.

## Архитектура

### Backend

Основной backend построен на `FastAPI`.

Ключевые файлы:

- `backend/main.py` — точка входа приложения;
- `backend/api/routes.py` — HTTP endpoints;
- `backend/services/*` — прикладная логика;
- `backend/models/domain.py` — доменные сущности для registry;
- `backend/utils/*` — совместимость и служебные функции.

### Service Layer

Система разнесена по отдельным сервисам:

- `DatasetService` — чтение `csv/xlsx`, валидация, schema checks;
- `TrainingService` — обучение и регистрация модели;
- `PredictionService` — загрузка champion-модели и inference;
- `ExplanationService` — SHAP/explainability;
- `DecisionService` — сбор фактов и генерация рекомендаций;
- `RegistryService` — файловый dataset/model registry.

### ML Layer

ML-часть реализована через:

- `automl/lightautoml_backend/adapter.py`
- `automl/training/preprocessing.py`
- `automl/evaluation/metrics.py`

Адаптер работает так:

1. принимает `pandas.DataFrame`;
2. прогоняет его через `TabularPreprocessor`;
3. определяет тип задачи;
4. обучает `LightAutoML`;
5. сохраняет bundle модели для повторного использования;
6. при ошибке real-path переключается на `sklearn` fallback.

### Explainability Layer

Объяснения строятся в:

- `explainability/shap_service/service.py`
- `explainability/fact_builder/builder.py`

`SHAP` используется для локальных факторов влияния. Если explainability path не срабатывает, система может вернуться к proxy-feature-impact.

### DSS Layer

Правила и рекомендации реализованы через:

- `dss/experta_engine/engine.py`
- `dss/rules/catalog.py`
- `dss/scenarios/defaults.py`
- `dss/recommendations/formatter.py`

На вход DSS получает:

- prediction;
- confidence;
- top factors из explainability.

На выходе формируется:

- risk level;
- summary;
- actions;
- rationale.

### Storage / Registry

Пока используется файловая реализация registry:

- `storage/datasets/` — сохранённые версии датасетов;
- `storage/artifacts/` — артефакты моделей и отчётов;
- `storage/registry/datasets.json` — metadata датасетов;
- `storage/registry/models.json` — metadata моделей.

Registry умеет:

- сохранять новую версию датасета;
- регистрировать новую модель;
- выбирать `champion` модель проекта;
- загружать serialised bundle через `joblib`.

## Поддержка входных данных

### Форматы

Сейчас поддерживаются:

- `CSV`
- `XLSX`
- `XLS`

Чтение реализовано в `backend/services/dataset_service.py`.

### Контракт для обучения

ML-модуль получает данные не как файл и не как JSON, а как `pandas.DataFrame`.

Требования:

- таблица не пустая;
- есть `target` колонка;
- есть хотя бы один feature;
- в `target` нет пропусков;
- в `target` минимум два различных значения.

### Контракт для предсказания

Для inference ожидается `DataFrame` без `target`, но с совместимым набором исходных признаков.

Перед предсказанием применяется тот же препроцессор, который был сохранён в model bundle во время обучения.

## Предобработка данных

Предобработка вынесена в отдельный модуль:

- `automl/training/preprocessing.py`

В нём находится `TabularPreprocessor`, который:

- определяет datetime-like колонки;
- определяет time-only колонки;
- преобразует дату/время в модельные признаки.

Текущие преобразования:

- `Дата` -> `Дата__ts`, `Дата__hour`, `Дата__dayofweek`, `Дата__month`
- `sunrise` / `sunset` -> `sunrise__minutes`, `sunset__minutes`

Сырые колонки после преобразования удаляются из модельного `DataFrame`.

Этот препроцессор используется единообразно:

- в обучении;
- в prediction-path;
- в explanation-path;
- в standalone CLI для `LightAutoML`.

## Совместимость окружения

### Python

Для полного стека зафиксирован `Python 3.13.x`.

Причины:

- `LightAutoML` и часть зависимостей конфликтуют с более новыми интерпретаторами;
- часть старых пинов требует обхода в bootstrap-скрипте.

Используемые файлы:

- `.python-version`
- `scripts/setup_full_env.sh`
- `requirements.txt`
- `requirements-core.txt`

### Важные обходы

В проект добавлены compatibility shims:

- `backend/utils/compat.py`

Они нужны для:

- `LightAutoML` на `NumPy 2.x`;
- `Experta` на `Python 3.13+`.

Также bootstrap-установка делает:

- CPU-only установку `torch`;
- установку `LightAutoML` отдельным шагом;
- более безопасную установку зависимостей с PyPI.

## API

Реализованы следующие endpoints:

- `GET /health`
- `POST /datasets/validate`
- `POST /datasets/validate/file`
- `POST /training/run`
- `POST /training/run/file`
- `POST /predictions/run`
- `POST /predictions/run/file`
- `POST /explanations/run`
- `POST /decision/run`

### Training Flow

`/training/run` и `/training/run/file` делают следующее:

1. читают данные;
2. валидируют dataset;
3. создают dataset version;
4. запускают `TabularAutoMLAdapter`;
5. получают metrics и bundle;
6. регистрируют model version;
7. помечают модель как `candidate` или `champion`.

### Prediction Flow

`/predictions/run`:

1. загружает champion bundle;
2. применяет сохранённый preprocessing;
3. выравнивает схему данных;
4. выполняет inference;
5. возвращает prediction и, где возможно, confidence.

### Explanation Flow

`/explanations/run`:

1. использует тот же prediction-path;
2. передаёт aligned features в `SHAP`;
3. возвращает top factors по строкам.

### Decision Flow

`/decision/run`:

1. берёт prediction;
2. берёт explanation;
3. собирает факты;
4. передаёт их в `Experta`;
5. возвращает recommendation payload.

## CLI и утилиты

### Полная установка окружения

```bash
scripts/setup_full_env.sh python3.13
```

### Прямой запуск LightAutoML

```bash
python scripts/run_lightautoml.py \
  --data datasets/dataset__1.xlsx \
  --target "Электропотребление"
```

Скрипт:

- читает `csv/xlsx/xls`;
- применяет `TabularPreprocessor`;
- обучает `LightAutoML`;
- считает holdout metrics;
- может сохранить модель и predictions;
- выводит JSON summary.

### Smoke-check полного API-контура

```bash
python scripts/check_full_flow.py \
  --data datasets/dataset__1.xlsx \
  --target "Электропотребление" \
  --project-id smoke-demo \
  --sample-size 2
```

Скрипт проверяет:

- `health`
- `training`
- `prediction`
- `explanation`
- `decision`

### Генерация визуального отчёта

```bash
python scripts/render_visual_report.py \
  --project-id demo \
  --sample-size 40
```

Результат сохраняется в:

- `storage/artifacts/<project-id>/visual-report.html`

Отчёт включает:

- метрики модели;
- распределение target;
- feature importance;
- actual vs predicted;
- top explanation factors;
- DSS recommendations.

## Визуализация

На текущем этапе полноценного frontend UI ещё нет. Вместо него добавлен локальный HTML-report generator.

Статус:

- `frontend/README.md` пока остаётся заглушкой;
- визуализация доступна как статический HTML из backend-данных и registry.

Это позволяет уже сейчас:

- показать результат обучения;
- визуально проверить важности признаков;
- увидеть объяснения и рекомендации без отдельного клиента.

## Тестирование

Текущий тестовый контур:

- `tests/test_api.py`
- `tests/test_preprocessing.py`

Покрытие на этой итерации включает:

- healthcheck;
- end-to-end training/prediction/decision flow;
- training from `xlsx` upload;
- корректность `TabularPreprocessor`;
- применение preprocessing в inference-path.

## Текущие ограничения

1. `LightAutoML` всё ещё пишет warning про NLP extras (`gensim`, `transformers`).
2. SHAP для больших sample size может быть дорогим по времени.
3. Registry пока файловый, не PostgreSQL/S3.
4. Frontend пока не реализован.
5. Monitoring, retraining jobs и MLflow/Evidently пока существуют как направление архитектуры, а не как полный production-flow.

## Что логично делать дальше

Самые сильные следующие шаги:

1. сделать frontend MVP;
2. стабилизировать explainability/logging;
3. добавить retraining и monitoring;
4. вынести policy сравнения `candidate`/`champion`;
5. перейти от file registry к production storage.
