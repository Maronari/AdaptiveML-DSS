# API

## Реализованные эндпоинты

### Системные

- `GET /health`

### Валидация датасета

- `POST /datasets/validate`
- `POST /datasets/validate/file`

### Обучение

- `POST /training/run`
- `POST /training/run/file`

### Предсказание

- `POST /predictions/run`
- `POST /predictions/run/file`

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
6. API возвращает метаданные модели и метрики.

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

## Ошибки

Валидационные ошибки на прикладном уровне преобразуются в HTTP `400`.

Примеры:

- пустой dataset;
- отсутствует target;
- mismatch features на prediction;
- отсутствует champion-модель для проекта.
