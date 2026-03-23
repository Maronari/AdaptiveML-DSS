-- AdaptiveML DSS
-- MVP PostgreSQL schema:
-- a compact relational core for users, projects, dataset versions,
-- training runs, model versions and model deployment history.
--
-- Heavy objects such as uploaded datasets, trained model bundles and reports
-- should stay in object storage or filesystem storage. PostgreSQL keeps only
-- queryable metadata and links to those artifacts.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS citext;

CREATE SCHEMA IF NOT EXISTS adaptive_ml_dss;
SET search_path TO adaptive_ml_dss, public;

-- ---------------------------------------------------------------------------
-- Users
-- ---------------------------------------------------------------------------

CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username CITEXT NOT NULL UNIQUE,
    email CITEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- Projects
-- ---------------------------------------------------------------------------

CREATE TABLE projects (
    project_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_code CITEXT NOT NULL UNIQUE,
    owner_user_id UUID NOT NULL REFERENCES users(user_id),
    name TEXT NOT NULL,
    description TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ix_projects_owner_user_id
    ON projects (owner_user_id);

-- ---------------------------------------------------------------------------
-- Dataset registry
-- ---------------------------------------------------------------------------

CREATE TABLE dataset_versions (
    dataset_version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    source_name TEXT NOT NULL,
    source_kind TEXT NOT NULL
        CHECK (source_kind IN ('upload', 'inline', 'api', 'import', 'generated', 'retraining')),
    target_name TEXT NOT NULL,
    dataset_format TEXT NULL
        CHECK (dataset_format IS NULL OR dataset_format IN ('csv', 'xlsx', 'xls', 'json', 'parquet', 'other')),
    artifact_uri TEXT NOT NULL,
    content_hash TEXT NULL,
    schema_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(schema_jsonb) = 'object'),
    stats_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(stats_jsonb) = 'object'),
    rows_count BIGINT NOT NULL CHECK (rows_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ix_dataset_versions_project_id
    ON dataset_versions (project_id);

-- ---------------------------------------------------------------------------
-- Training and model registry
-- ---------------------------------------------------------------------------

CREATE TABLE training_runs (
    training_run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    dataset_version_id UUID NOT NULL REFERENCES dataset_versions(dataset_version_id) ON DELETE RESTRICT,
    base_model_version_id UUID NULL,
    run_kind TEXT NOT NULL
        CHECK (run_kind IN ('initial', 'retrain')),
    history_scope TEXT NOT NULL
        CHECK (history_scope IN ('all_history', 'last_30_days', 'last_60_days', 'last_90_days')),
    requested_options_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(requested_options_jsonb) = 'object'),
    effective_options_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(effective_options_jsonb) = 'object'),
    evaluation_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(evaluation_jsonb) = 'object'),
    run_status TEXT NOT NULL
        CHECK (run_status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    error_message TEXT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ NULL,
    CHECK (finished_at IS NULL OR finished_at >= started_at)
);

CREATE INDEX ix_training_runs_project_id
    ON training_runs (project_id);

CREATE INDEX ix_training_runs_dataset_version_id
    ON training_runs (dataset_version_id);

CREATE TABLE model_versions (
    model_version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    training_run_id UUID NOT NULL UNIQUE REFERENCES training_runs(training_run_id) ON DELETE CASCADE,
    dataset_version_id UUID NOT NULL REFERENCES dataset_versions(dataset_version_id) ON DELETE RESTRICT,
    version_number INTEGER NOT NULL CHECK (version_number >= 1),
    target_name TEXT NOT NULL,
    artifact_uri TEXT NOT NULL,
    task_type TEXT NOT NULL
        CHECK (task_type IN ('binary', 'multiclass', 'regression')),
    primary_metric TEXT NOT NULL,
    metrics_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(metrics_jsonb) = 'object'),
    feature_names_jsonb JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(feature_names_jsonb) = 'array'),
    forecast_profile_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(forecast_profile_jsonb) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (project_id, version_number)
);

CREATE INDEX ix_model_versions_project_id
    ON model_versions (project_id);

CREATE INDEX ix_model_versions_dataset_version_id
    ON model_versions (dataset_version_id);

-- ---------------------------------------------------------------------------
-- Deployments and rollback
-- One active deployment per project.
-- Rollback is recorded as a new deployment pointing to an older model version.
-- ---------------------------------------------------------------------------

CREATE TABLE model_deployments (
    deployment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    model_version_id UUID NOT NULL REFERENCES model_versions(model_version_id) ON DELETE RESTRICT,
    deployment_reason TEXT NULL,
    rollback_of_deployment_id UUID NULL REFERENCES model_deployments(deployment_id),
    deployed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    undeployed_at TIMESTAMPTZ NULL,
    CHECK (undeployed_at IS NULL OR undeployed_at >= deployed_at)
);

CREATE UNIQUE INDEX ux_model_deployments_active
    ON model_deployments (project_id)
    WHERE undeployed_at IS NULL;

CREATE INDEX ix_model_deployments_model_version_id
    ON model_deployments (model_version_id);

COMMIT;
