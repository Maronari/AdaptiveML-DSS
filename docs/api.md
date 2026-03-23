# API

## Реализованные эндпоинты

### Системные

- `GET /health`

### Проекты и модели

- `GET /projects`
- `POST /projects`
- `DELETE /projects/{project_id}`
- `GET /projects/{project_id}/models`
- `POST /projects/{project_id}/models/{version_id}/activate`

### Валидация датасета

- `POST /datasets/validate`
- `POST /datasets/validate/file`

### Обучение

- `POST /training/run`
- `POST /training/run/file`

### Предсказание

- `POST /predictions/run`
- `POST /predictions/run/file`
- `POST /predictions/compare`
- `POST /predictions/compare/file`
- `POST /predictions/compare/file/schema`
- `GET /predictions/compare/latest`

### Модели и прогноз

- `GET /models/latest`
- `GET /models/{version_id}`
- `GET /models/{version_id}/forecast`
- `POST /forecast/run`

### Объяснение

- `POST /explanations/run`

### Поддержка решений

- `POST /decision/run`

## Форматы входа

### JSON

Основной паттерн:

```json
{
  "project_id": "demo",
  "target": "target",
  "records": [
    {"feature_a": 1, "feature_b": 2, "target": 0},
    {"feature_a": 3, "feature_b": 4, "target": 1}
  ]
}
```

### Загрузка файла

Поддерживаются:

- `csv`
- `xlsx`
- `xls`

Эндпоинты загрузки используют `multipart/form-data`.

## Поток обучения

### JSON

`POST /training/run`

Что происходит:

1. JSON превращается в `DataFrame`;
2. датасет валидируется;
3. создаётся dataset version;
4. запускается ML adapter;
5. артефакт модели сериализуется в реестр;
6. в metadata модели сохраняются holdout-предсказания и training artifacts;
7. API возвращает метаданные модели и метрики.

### Загрузка файла

`POST /training/run/file`

Форма:

- `file`
- `project_id`
- `target`

## Поток предсказания

`POST /predictions/run`

Вход:

```json
{
  "project_id": "demo",
  "records": [
    {
      "Дата": "2024-05-01 12:00:00",
      "Рабочий день": 1,
      "sunrise": "05:12",
      "sunset": "20:11"
    }
  ]
}
```

На стороне backend:

1. загружается артефакт champion-модели;
2. применяется сохранённый preprocessing;
3. проверяется соответствие feature schema;
4. модель выполняет предсказание;
5. ответ нормализуется в формат API.

## История моделей проекта

`GET /projects/{project_id}/models`

Возвращает облегчённую историю версий модели по проекту:

- `status`, `is_latest`, `is_champion`
- `metric_value` и `primary_metric`
- связанный `dataset_version_id`
- `holdout_rows`
- `has_training_artifacts`

`POST /projects/{project_id}/models/{version_id}/activate`

Переключает выбранную версию в `champion`, а предыдущую активную версию переводит в `archived`.

## Сводка по модели

`GET /models/latest` и `GET /models/{version_id}` возвращают не только metadata модели, но и:

- `holdout_predictions`
- `training_artifacts`
- summary по forecasting bundle

## Поток объяснения

`POST /explanations/run`

Логика:

1. выполняется prediction-path;
2. aligned features передаются в SHAP;
3. по каждой строке собираются `top_factors`.

## Поток поддержки решений

`POST /decision/run`

Логика:

1. prediction;
2. explanation;
3. facts;
4. rule engine;
5. recommendation.

## Проверка здоровья

Простой endpoint:

```json
{"status": "ok"}
```

В `docker-compose.yml` этот endpoint используется как `healthcheck` для контейнера `api`.

## Ошибки

Валидационные ошибки на прикладном уровне преобразуются в HTTP `400`.

Примеры:

- пустой dataset;
- отсутствует target;
- mismatch features на prediction;
- отсутствует champion-модель для проекта.
