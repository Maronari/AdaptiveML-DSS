# Схема базы данных

На этой странице описана упрощенная PostgreSQL-схема проекта из `storage/postgresql/schema.sql`.

Текущее состояние схемы формируется миграциями Alembic:

- `alembic/versions/20260321_0001_compact_core_schema.py`
- `alembic/versions/20260322_0002_simplify_mvp_schema.py`

Схема сознательно упрощена под MVP:

- убраны поля жизненного цикла пользователей и проектов;
- убраны пользовательские сессии;
- убрана таблица участников проекта;
- убран слой аудита для предсказаний и прогноза;
- сохранены версии датасетов, версии моделей и история активации модели.

В итоге ядро состоит из 6 таблиц:

1. `users`
2. `projects`
3. `dataset_versions`
4. `training_runs`
5. `model_versions`
6. `model_deployments`

## Принципы проектирования

- Реляционные таблицы оставлены только там, где действительно важны связи, фильтрация и откат.
- `JSONB` используется для метаданных датасета, настроек обучения, метрик и профиля прогноза.
- Версионность моделей выражена явно через `model_versions.version_number`.
- Активная модель проекта хранится через историю `model_deployments`, а не флагом внутри модели.
- Тяжелые файлы и артефакты моделей остаются вне PostgreSQL и в БД представлены ссылками.

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

## Пользователи и проекты

```mermaid
erDiagram
    USERS ||--o{ PROJECTS : владеет

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
```

## Датасеты, обучение и версии моделей

```mermaid
erDiagram
    PROJECTS ||--o{ DATASET_VERSIONS : хранит
    PROJECTS ||--o{ TRAINING_RUNS : содержит
    PROJECTS ||--o{ MODEL_VERSIONS : хранит
    DATASET_VERSIONS ||--o{ TRAINING_RUNS : подает_на_вход
    DATASET_VERSIONS ||--o{ MODEL_VERSIONS : обучает_на
    TRAINING_RUNS ||--|| MODEL_VERSIONS : порождает

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
```

## Активация модели и откат

```mermaid
erDiagram
    PROJECTS ||--o{ MODEL_DEPLOYMENTS : отслеживает
    MODEL_VERSIONS ||--o{ MODEL_DEPLOYMENTS : активирует
    MODEL_DEPLOYMENTS o|--o{ MODEL_DEPLOYMENTS : откат_от

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

## Практические замечания

- `projects.owner_user_id` достаточно для MVP, если каждый проект принадлежит одному пользователю.
- `dataset_versions` хранит версии загруженных и сгенерированных датасетов без отдельного каталога колонок.
- `training_runs` фиксирует режим `initial` или `retrain`, выбранное окно истории и результат сравнения со старой моделью.
- `model_versions` сохраняет полноценную историю обученных моделей, а `version_number` делает эту историю человекочитаемой.
- `model_deployments` является источником истины для текущей активной модели проекта.
- Откат оформляется как новая запись в `model_deployments`, указывающая на более раннюю версию модели.

## Почему схема стала меньше

Из MVP убраны сущности, которые усложняли схему без прямой пользы на текущем этапе:

- `user_sessions`
- `project_memberships`
- `inference_runs`
- `forecast_runs`
- поля жизненного цикла `user_status`, `project_status`, `archived_at`
- вторичные ссылки на пользователя в версиях датасетов, запусках обучения и деплое

Если эти сценарии понадобятся позже, их можно вернуть отдельными миграциями, не ломая ядро реестра.

## Миграции и запуск

Локальный запуск PostgreSQL:

```bash
docker compose up -d postgres
```

Применение миграций:

```bash
python -m alembic upgrade head
```

Если база уже была создана по старой версии схемы, новая миграция упростит ее до текущего MVP-состояния.

Если нужен единый вид всех сущностей, откройте страницу `Общая ER-диаграмма` в навигации документации.
