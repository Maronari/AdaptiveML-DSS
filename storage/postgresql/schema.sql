-- AdaptiveML DSS
-- Compact PostgreSQL schema:
-- relational core for auth, projects, dataset/model registry and deployments
-- plus JSONB payloads for flexible metadata, metrics and audit details.
--
-- Heavy objects such as uploaded datasets, trained model bundles, prediction
-- exports and reports should stay in object storage or filesystem storage.
-- This schema stores references, lifecycle state and queryable metadata.

BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS citext;

CREATE SCHEMA IF NOT EXISTS adaptive_ml_dss;
SET search_path TO adaptive_ml_dss, public;

-- ---------------------------------------------------------------------------
-- Users and access
-- ---------------------------------------------------------------------------

CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username CITEXT NOT NULL UNIQUE,
    email CITEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL,
    user_status TEXT NOT NULL DEFAULT 'active'
        CHECK (user_status IN ('active', 'disabled', 'locked')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    refresh_token_hash TEXT NOT NULL UNIQUE,
    user_agent TEXT NULL,
    ip_address INET NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (expires_at > created_at),
    CHECK (revoked_at IS NULL OR revoked_at >= created_at)
);

CREATE INDEX ix_user_sessions_user_id
    ON user_sessions (user_id);

CREATE TABLE projects (
    project_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_code CITEXT NOT NULL UNIQUE,
    owner_user_id UUID NOT NULL REFERENCES users(user_id),
    name TEXT NOT NULL,
    description TEXT NULL,
    project_status TEXT NOT NULL DEFAULT 'active'
        CHECK (project_status IN ('active', 'archived')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    archived_at TIMESTAMPTZ NULL,
    CHECK (archived_at IS NULL OR archived_at >= created_at)
);

CREATE INDEX ix_projects_owner_user_id
    ON projects (owner_user_id);

CREATE TABLE project_memberships (
    membership_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    role TEXT NOT NULL
        CHECK (role IN ('owner', 'admin', 'editor', 'viewer')),
    granted_by_user_id UUID NULL REFERENCES users(user_id),
    granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ NULL,
    CHECK (revoked_at IS NULL OR revoked_at >= granted_at)
);

CREATE UNIQUE INDEX ux_project_memberships_active
    ON project_memberships (project_id, user_id)
    WHERE revoked_at IS NULL;

CREATE INDEX ix_project_memberships_user_id
    ON project_memberships (user_id);

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
    created_by_user_id UUID NULL REFERENCES users(user_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ix_dataset_versions_project_id
    ON dataset_versions (project_id);

CREATE INDEX ix_dataset_versions_created_by_user_id
    ON dataset_versions (created_by_user_id);

-- ---------------------------------------------------------------------------
-- Training and model registry
-- ---------------------------------------------------------------------------

CREATE TABLE training_runs (
    training_run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    dataset_version_id UUID NOT NULL REFERENCES dataset_versions(dataset_version_id) ON DELETE RESTRICT,
    requested_by_user_id UUID NULL REFERENCES users(user_id),
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

CREATE INDEX ix_training_runs_requested_by_user_id
    ON training_runs (requested_by_user_id);

CREATE TABLE model_versions (
    model_version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    training_run_id UUID NOT NULL UNIQUE REFERENCES training_runs(training_run_id) ON DELETE CASCADE,
    dataset_version_id UUID NOT NULL REFERENCES dataset_versions(dataset_version_id) ON DELETE RESTRICT,
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
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ix_model_versions_project_id
    ON model_versions (project_id);

CREATE INDEX ix_model_versions_dataset_version_id
    ON model_versions (dataset_version_id);

-- ---------------------------------------------------------------------------
-- Deployments and rollback
-- One active champion and one active serving deployment per project.
-- Rollback is recorded as a new deployment pointing to an older model version.
-- ---------------------------------------------------------------------------

CREATE TABLE model_deployments (
    deployment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    model_version_id UUID NOT NULL REFERENCES model_versions(model_version_id) ON DELETE RESTRICT,
    deployment_role TEXT NOT NULL
        CHECK (deployment_role IN ('champion', 'serving')),
    deployed_by_user_id UUID NULL REFERENCES users(user_id),
    deployment_reason TEXT NULL,
    rollback_of_deployment_id UUID NULL REFERENCES model_deployments(deployment_id),
    deployed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    undeployed_at TIMESTAMPTZ NULL,
    CHECK (undeployed_at IS NULL OR undeployed_at >= deployed_at)
);

CREATE UNIQUE INDEX ux_model_deployments_active
    ON model_deployments (project_id, deployment_role)
    WHERE undeployed_at IS NULL;

CREATE INDEX ix_model_deployments_model_version_id
    ON model_deployments (model_version_id);

-- ---------------------------------------------------------------------------
-- Prediction / explanation / decision audit
-- Variable payloads are intentionally stored in JSONB to avoid over-normalizing
-- row-level prediction, explanation and recommendation details.
-- ---------------------------------------------------------------------------

CREATE TABLE inference_runs (
    inference_run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    model_version_id UUID NOT NULL REFERENCES model_versions(model_version_id) ON DELETE RESTRICT,
    requested_by_user_id UUID NULL REFERENCES users(user_id),
    request_mode TEXT NOT NULL
        CHECK (request_mode IN ('inline', 'file', 'batch', 'compare')),
    input_artifact_uri TEXT NULL,
    row_count BIGINT NOT NULL CHECK (row_count >= 0),
    predictions_jsonb JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(predictions_jsonb) = 'array'),
    comparison_metrics_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(comparison_metrics_jsonb) = 'object'),
    explanations_jsonb JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(explanations_jsonb) = 'array'),
    recommendations_jsonb JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(recommendations_jsonb) = 'array'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ix_inference_runs_project_id
    ON inference_runs (project_id);

CREATE INDEX ix_inference_runs_model_version_id
    ON inference_runs (model_version_id);

CREATE TABLE forecast_runs (
    forecast_run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    model_version_id UUID NOT NULL REFERENCES model_versions(model_version_id) ON DELETE RESTRICT,
    requested_by_user_id UUID NULL REFERENCES users(user_id),
    horizon_minutes INTEGER NOT NULL CHECK (horizon_minutes >= 1),
    steps INTEGER NOT NULL CHECK (steps >= 1),
    recent_history_jsonb JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(recent_history_jsonb) = 'array'),
    forecast_jsonb JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(forecast_jsonb) = 'array'),
    metrics_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(metrics_jsonb) = 'object'),
    warning_text TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ix_forecast_runs_project_id
    ON forecast_runs (project_id);

CREATE INDEX ix_forecast_runs_model_version_id
    ON forecast_runs (model_version_id);

COMMIT;
