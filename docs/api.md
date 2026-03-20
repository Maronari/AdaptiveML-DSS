# API

## Реализованные endpoints

### System

- `GET /health`

### Dataset Validation

- `POST /datasets/validate`
- `POST /datasets/validate/file`

### Training

- `POST /training/run`
- `POST /training/run/file`

### Prediction

- `POST /predictions/run`
- `POST /predictions/run/file`

### Explainability

- `POST /explanations/run`

### Decision Support

- `POST /decision/run`

## Форматы входа

### Inline JSON

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

### File Upload

Поддерживаются:

- `csv`
- `xlsx`
- `xls`

Upload-endpoints используют `multipart/form-data`.

## Training Flow

### Inline

`POST /training/run`

Что происходит:

1. JSON превращается в `DataFrame`;
2. датасет валидируется;
3. создаётся dataset version;
4. запускается ML adapter;
5. bundle модели сериализуется в registry;
6. API возвращает model metadata и metrics.

### File Upload

`POST /training/run/file`

Форма:

- `file`
- `project_id`
- `target`

## Prediction Flow

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

1. загружается champion bundle;
2. применяется сохранённый preprocessing;
3. проверяется соответствие feature schema;
4. модель делает inference;
5. ответ нормализуется в API payload.

## Explanation Flow

`POST /explanations/run`

Логика:

1. выполняется prediction-path;
2. aligned features передаются в SHAP;
3. по каждой строке собираются `top_factors`.

## Decision Flow

`POST /decision/run`

Логика:

1. prediction;
2. explanation;
3. facts;
4. rule engine;
5. recommendation.

## Healthcheck

Простой endpoint:

```json
{"status": "ok"}
```

## Ошибки

Валидационные ошибки на прикладном уровне преобразуются в HTTP `400`.

Примеры:

- пустой dataset;
- отсутствует target;
- mismatch features на prediction;
- отсутствует champion-модель для проекта.
