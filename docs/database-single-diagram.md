# Общая ER-диаграмма

На этой странице собрана одна общая Mermaid ER-диаграмма для упрощенной PostgreSQL-схемы проекта.

Она соответствует:

- `storage/postgresql/schema.sql`
- всем миграциям Alembic до `head`

```mermaid
erDiagram
    USERS ||--o{ PROJECTS : владеет
    PROJECTS ||--o{ DATASET_VERSIONS : хранит
    PROJECTS ||--o{ TRAINING_RUNS : содержит
    PROJECTS ||--o{ MODEL_VERSIONS : версионирует
    PROJECTS ||--o{ MODEL_DEPLOYMENTS : активирует

    DATASET_VERSIONS ||--o{ TRAINING_RUNS : подает_на_вход
    DATASET_VERSIONS ||--o{ MODEL_VERSIONS : источник_для
    TRAINING_RUNS ||--|| MODEL_VERSIONS : порождает
    MODEL_VERSIONS ||--o{ MODEL_DEPLOYMENTS : разворачивается_как
    MODEL_DEPLOYMENTS o|--o{ MODEL_DEPLOYMENTS : откат_от

    USERS {
        uuid user_id PK
        citext username UK
        citext email UK
        text password_hash
        text display_name
        timestamptz created_at
        timestamptz updated_at
    }

    PROJECTS {
        uuid project_id PK
        citext project_code UK
        uuid owner_user_id FK
        text name
        text description
        timestamptz created_at
    }

    DATASET_VERSIONS {
        uuid dataset_version_id PK
        uuid project_id FK
        text source_name
        text source_kind
        text target_name
        text dataset_format
        text artifact_uri
        text content_hash
        jsonb schema_jsonb
        jsonb stats_jsonb
        bigint rows_count
        timestamptz created_at
    }

    TRAINING_RUNS {
        uuid training_run_id PK
        uuid project_id FK
        uuid dataset_version_id FK
        uuid base_model_version_id
        text run_kind
        text history_scope
        jsonb requested_options_jsonb
        jsonb effective_options_jsonb
        jsonb evaluation_jsonb
        text run_status
        text error_message
        timestamptz started_at
        timestamptz finished_at
    }

    MODEL_VERSIONS {
        uuid model_version_id PK
        uuid project_id FK
        uuid training_run_id FK
        uuid dataset_version_id FK
        int version_number
        text target_name
        text artifact_uri
        text task_type
        text primary_metric
        jsonb metrics_jsonb
        jsonb feature_names_jsonb
        jsonb forecast_profile_jsonb
        timestamptz created_at
    }

    MODEL_DEPLOYMENTS {
        uuid deployment_id PK
        uuid project_id FK
        uuid model_version_id FK
        text deployment_reason
        uuid rollback_of_deployment_id FK
        timestamptz deployed_at
        timestamptz undeployed_at
    }
```

## Как читать диаграмму

- `dataset_versions` хранит версии входных данных проекта.
- `training_runs` фиксирует как первичное обучение, так и контролируемое переобучение на новых данных.
- `model_versions` хранит полную историю обученных моделей внутри проекта.
- `version_number` позволяет пользователю выбирать конкретную версию модели для отката.
- `model_deployments` хранит историю активации модели, поэтому откат выражается новой записью, а не перезаписью старой.
