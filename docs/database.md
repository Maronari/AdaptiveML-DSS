# Схема базы данных

На этой странице описана компактная PostgreSQL-схема проекта из `storage/postgresql/schema.sql`.

Та же структура оформлена как initial migration Alembic:

- `alembic/versions/20260321_0001_compact_core_schema.py`

Подход к схеме намеренно гибридный:

- ключевые сущности жизненного цикла остаются реляционными;
- изменчивые структуры вынесены в `JSONB`;
- тяжелые файлы и bundle моделей хранятся вне PostgreSQL и в БД представлены ссылками.

В итоге вместо полной сильно нормализованной схемы используется компактное ядро из 10 таблиц:

1. `users`
2. `user_sessions`
3. `projects`
4. `project_memberships`
5. `dataset_versions`
6. `training_runs`
7. `model_versions`
8. `model_deployments`
9. `inference_runs`
10. `forecast_runs`

## Принципы проектирования

- Реляционные таблицы используются для владения, истории, связей и отката.
- `JSONB` используется там, где структура естественно меняется от запуска к запуску.
- История деплоя хранится отдельными записями, а не флагами внутри модели.
- Исходные датасеты, выгрузки, отчеты и serialized models лучше держать в файловом хранилище или object storage.

## Где используется `JSONB`

Основные поля с `JSONB`:

- `dataset_versions.schema_jsonb`
- `dataset_versions.stats_jsonb`
- `training_runs.requested_options_jsonb`
- `training_runs.effective_options_jsonb`
- `training_runs.evaluation_jsonb`
- `model_versions.metrics_jsonb`
- `model_versions.feature_names_jsonb`
- `model_versions.forecast_profile_jsonb`
- `inference_runs.predictions_jsonb`
- `inference_runs.comparison_metrics_jsonb`
- `inference_runs.explanations_jsonb`
- `inference_runs.recommendations_jsonb`
- `forecast_runs.recent_history_jsonb`
- `forecast_runs.forecast_jsonb`
- `forecast_runs.metrics_jsonb`

## Доступ и владение

```mermaid
erDiagram
    USERS ||--o{ USER_SESSIONS : opens
    USERS ||--o{ PROJECTS : owns
    PROJECTS ||--o{ PROJECT_MEMBERSHIPS : has
    USERS ||--o{ PROJECT_MEMBERSHIPS : joins

    USERS {
        uuid user_id PK
        citext username UK
        citext email UK
        text password_hash
        text display_name
        text user_status
    }

    USER_SESSIONS {
        uuid session_id PK
        uuid user_id FK
        text refresh_token_hash UK
        timestamptz expires_at
        timestamptz revoked_at
    }

    PROJECTS {
        uuid project_id PK
        citext project_code UK
        uuid owner_user_id FK
        text name
        text project_status
    }

    PROJECT_MEMBERSHIPS {
        uuid membership_id PK
        uuid project_id FK
        uuid user_id FK
        text role
        uuid granted_by_user_id FK
        timestamptz revoked_at
    }
```

## Реестр датасетов и обучение

```mermaid
erDiagram
    PROJECTS ||--o{ DATASET_VERSIONS : owns
    USERS ||--o{ DATASET_VERSIONS : uploads
    PROJECTS ||--o{ TRAINING_RUNS : contains
    USERS ||--o{ TRAINING_RUNS : requests
    DATASET_VERSIONS ||--o{ TRAINING_RUNS : feeds
    TRAINING_RUNS ||--|| MODEL_VERSIONS : produces
    PROJECTS ||--o{ MODEL_VERSIONS : owns
    DATASET_VERSIONS ||--o{ MODEL_VERSIONS : trains_on

    DATASET_VERSIONS {
        uuid dataset_version_id PK
        uuid project_id FK
        text source_name
        text source_kind
        text target_name
        text artifact_uri
        jsonb schema_jsonb
        jsonb stats_jsonb
        bigint rows_count
    }

    TRAINING_RUNS {
        uuid training_run_id PK
        uuid project_id FK
        uuid dataset_version_id FK
        uuid requested_by_user_id FK
        uuid base_model_version_id
        text run_kind
        text history_scope
        jsonb requested_options_jsonb
        jsonb effective_options_jsonb
        jsonb evaluation_jsonb
        text run_status
    }

    MODEL_VERSIONS {
        uuid model_version_id PK
        uuid project_id FK
        uuid training_run_id FK
        uuid dataset_version_id FK
        text target_name
        text artifact_uri
        text task_type
        text primary_metric
        jsonb metrics_jsonb
        jsonb feature_names_jsonb
        jsonb forecast_profile_jsonb
    }
```

## Деплой, откат и аудит

```mermaid
erDiagram
    PROJECTS ||--o{ MODEL_DEPLOYMENTS : tracks
    MODEL_VERSIONS ||--o{ MODEL_DEPLOYMENTS : deploys
    USERS ||--o{ MODEL_DEPLOYMENTS : changes
    MODEL_DEPLOYMENTS o|--o{ MODEL_DEPLOYMENTS : rollback_of

    PROJECTS ||--o{ INFERENCE_RUNS : logs
    MODEL_VERSIONS ||--o{ INFERENCE_RUNS : serves
    USERS ||--o{ INFERENCE_RUNS : requests

    PROJECTS ||--o{ FORECAST_RUNS : logs
    MODEL_VERSIONS ||--o{ FORECAST_RUNS : forecasts_with
    USERS ||--o{ FORECAST_RUNS : requests

    MODEL_DEPLOYMENTS {
        uuid deployment_id PK
        uuid project_id FK
        uuid model_version_id FK
        text deployment_role
        uuid deployed_by_user_id FK
        uuid rollback_of_deployment_id FK
        timestamptz deployed_at
        timestamptz undeployed_at
    }

    INFERENCE_RUNS {
        uuid inference_run_id PK
        uuid project_id FK
        uuid model_version_id FK
        uuid requested_by_user_id FK
        text request_mode
        bigint row_count
        jsonb predictions_jsonb
        jsonb comparison_metrics_jsonb
        jsonb explanations_jsonb
        jsonb recommendations_jsonb
    }

    FORECAST_RUNS {
        uuid forecast_run_id PK
        uuid project_id FK
        uuid model_version_id FK
        uuid requested_by_user_id FK
        integer horizon_minutes
        integer steps
        jsonb recent_history_jsonb
        jsonb forecast_jsonb
        jsonb metrics_jsonb
    }
```

## Практические замечания

- `projects` и `project_memberships` закрывают и владение проектом, и совместную работу без отдельного справочника ролей.
- `dataset_versions` заменяет источник, snapshot и таблицы с колонками одной записью датасета плюс `schema_jsonb` и `stats_jsonb`.
- `training_runs` хранит настройки запуска, режим переобучения, окно истории и результат сравнения с предыдущей моделью.
- `model_versions` содержит по одной записи на обученную модель, а метрики и профиль прогноза хранит в `JSONB`.
- `model_deployments` является источником истины для текущей `champion`-модели и текущей `serving`-модели.
- Откат оформляется как новая запись деплоя, указывающая на старую `model_version`.
- `inference_runs` и `forecast_runs` дают аудит без разрастания схемы в десятки таблиц построчных результатов.

## Почему схема стала меньше

В полной нормализованной версии пришлось бы выделять отдельные таблицы для:

- email-адресов
- внешних идентификаторов
- каталогов ролей
- источников датасетов
- snapshot-версий датасетов
- колонок датасета
- каталогов target
- каталогов алгоритмов
- каталогов метрик
- slot-таблиц для deploy
- prediction rows
- explanation rows
- decision rows
- точки прогноза

Текущий вариант оставляет реляционными только те части, которые реально важно фильтровать и связывать запросами, а остальное переносит в `JSONB`. Для текущей стадии проекта это более практичный баланс.

## Миграции и запуск

Локальный запуск PostgreSQL:

```bash
docker compose up -d postgres
```

Применение миграций:

```bash
python -m alembic upgrade head
```

Если нужен единый вид всех сущностей, откройте страницу `Общая ER-диаграмма` в навигации документации.
