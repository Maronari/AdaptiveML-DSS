# Архитектура

## Общая схема

Система состоит из нескольких слоёв:

- `FastAPI` backend;
- service layer;
- ML-адаптер на `LightAutoML`;
- explainability слой на `SHAP`;
- DSS слой на `Experta`;
- файловый registry для датасетов и моделей.

Поток данных выглядит так:

`upload/json -> DataFrame -> preprocessing -> model -> explanation -> facts -> recommendation`

## Бэкенд

Точка входа:

- `backend/main.py`

Маршруты:

- `backend/api/routes.py`

Основная идея backend:

- роуты принимают запрос;
- передают выполнение в сервисы;
- сервисы работают с registry, ML и DSS слоями;
- данные между слоями передаются как `pandas.DataFrame` и `dict` payload.

## Сервисный слой

### DatasetService

Файл:

- `backend/services/dataset_service.py`

Задачи:

- чтение `csv/xlsx/xls`;
- базовая валидация датасета;
- schema checks;
- определение типа задачи.

### TrainingService

Файл:

- `backend/services/training_service.py`

Задачи:

- создать dataset version;
- передать `DataFrame` в `TabularAutoMLAdapter`;
- зарегистрировать новую model version;
- выбрать `candidate`/`champion`.

### PredictionService

Файл:

- `backend/services/prediction_service.py`

Задачи:

- загрузить champion bundle;
- применить сохранённый preprocessing;
- выполнить inference;
- привести output к нормальному API payload.

### ExplanationService

Файл:

- `backend/services/explanation_service.py`

Задачи:

- переиспользовать prediction flow;
- запустить SHAP;
- собрать `top_factors`.

### DecisionService

Файл:

- `backend/services/decision_service.py`

Задачи:

- собрать prediction и explanation;
- превратить explanation в facts;
- прогнать facts через rule engine;
- вернуть recommendation.

## ML-слой

### Основной адаптер

Файл:

- `automl/lightautoml_backend/adapter.py`

Контракт:

- вход: `pandas.DataFrame` + имя target-колонки;
- выход: `TrainingResult` с metrics, feature names и serializable bundle.

Внутри:

1. применяется `TabularPreprocessor`;
2. определяется `task_type`;
3. данные делятся на train/test;
4. запускается `LightAutoML`;
5. считается качество;
6. формируется bundle.

Если real `LightAutoML` падает, адаптер переключается на `sklearn` fallback.

### Предобработка

Файл:

- `automl/training/preprocessing.py`

`TabularPreprocessor` сейчас:

- определяет datetime-like колонки;
- определяет time-only колонки;
- генерирует числовые признаки из даты и времени;
- удаляет сырые исходные колонки из модельной таблицы.

Это один и тот же объект, который используется:

- на обучении;
- на prediction;
- на explanation;
- в CLI `run_lightautoml.py`.

## Слой объяснений

Файлы:

- `explainability/shap_service/service.py`
- `explainability/fact_builder/builder.py`

Explainability path устроен так:

1. берётся training sample из bundle;
2. собирается SHAP explainer;
3. вычисляются локальные contributions;
4. contributions превращаются в `top_factors`;
5. `top_factors` дальше идут в DSS.

## DSS-слой

Файлы:

- `dss/experta_engine/engine.py`
- `dss/rules/catalog.py`
- `dss/scenarios/defaults.py`
- `dss/recommendations/formatter.py`

Входные факты:

- risk level;
- prediction;
- confidence;
- сильные положительные факторы;
- средние положительные факторы;
- отрицательные факторы.

Выход:

- `summary`
- `actions`
- `rationale`

## Хранилище / реестр

Файлы:

- `backend/services/registry_service.py`
- `storage/registry/filesystem.py`

Структура:

- `storage/datasets/` — версии датасетов;
- `storage/artifacts/` — модели и визуальные отчёты;
- `storage/registry/datasets.json` — metadata датасетов;
- `storage/registry/models.json` — metadata моделей.

Пока registry файловый. Это удобно для локальной разработки, но не является production storage.
