# Общая ER-диаграмма

На этой странице собрана одна общая Mermaid ER-диаграмма для компактной PostgreSQL-схемы проекта.

Она соответствует:

- `storage/postgresql/schema.sql`
- `alembic/versions/20260321_0001_compact_core_schema.py`

```mermaid
erDiagram
    USERS ||--o{ USER_SESSIONS : opens
    USERS ||--o{ PROJECTS : owns
    USERS ||--o{ PROJECT_MEMBERSHIPS : joins
    USERS ||--o{ DATASET_VERSIONS : uploads
    USERS ||--o{ TRAINING_RUNS : requests
    USERS ||--o{ MODEL_DEPLOYMENTS : changes
    USERS ||--o{ INFERENCE_RUNS : requests
    USERS ||--o{ FORECAST_RUNS : requests

    PROJECTS ||--o{ PROJECT_MEMBERSHIPS : has
    PROJECTS ||--o{ DATASET_VERSIONS : owns
    PROJECTS ||--o{ TRAINING_RUNS : contains
    PROJECTS ||--o{ MODEL_VERSIONS : owns
    PROJECTS ||--o{ MODEL_DEPLOYMENTS : tracks
    PROJECTS ||--o{ INFERENCE_RUNS : logs
    PROJECTS ||--o{ FORECAST_RUNS : logs

    DATASET_VERSIONS ||--o{ TRAINING_RUNS : feeds
    DATASET_VERSIONS ||--o{ MODEL_VERSIONS : trains_on
    TRAINING_RUNS ||--|| MODEL_VERSIONS : produces
    MODEL_VERSIONS ||--o{ MODEL_DEPLOYMENTS : deploys
    MODEL_DEPLOYMENTS o|--o{ MODEL_DEPLOYMENTS : rollback_of
    MODEL_VERSIONS ||--o{ INFERENCE_RUNS : serves
    MODEL_VERSIONS ||--o{ FORECAST_RUNS : forecasts_with

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

    DATASET_VERSIONS {
        uuid dataset_version_id PK
        uuid project_id FK
        text source_name
        text source_kind
        text target_name
        text dataset_format
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

## Как читать диаграмму

- `training_runs` хранит и первичное обучение, и контролируемое переобучение.
- `model_deployments` фиксирует историю назначений `champion` и `serving`, поэтому откат выражается отдельной записью.
- Детальные результаты предсказаний, объяснений и рекомендаций хранятся внутри `JSONB`, чтобы схема оставалась компактной.
- Для подробных пояснений по каждой группе таблиц используйте страницу `Схема БД`.
