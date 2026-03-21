"""create compact core schema

Revision ID: 20260321_0001
Revises:
Create Date: 2026-03-21 21:20:00

"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260321_0001"
down_revision = None
branch_labels = None
depends_on = None


SCHEMA = "adaptive_ml_dss"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS citext")
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

    op.create_table(
        "users",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("username", postgresql.CITEXT(), nullable=False),
        sa.Column("email", postgresql.CITEXT(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("user_status", sa.Text(), nullable=False, server_default=sa.text("'active'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint("user_status IN ('active', 'disabled', 'locked')", name="ck_users_user_status"),
        sa.PrimaryKeyConstraint("user_id"),
        sa.UniqueConstraint("username", name="uq_users_username"),
        sa.UniqueConstraint("email", name="uq_users_email"),
        schema=SCHEMA,
    )

    op.create_table(
        "user_sessions",
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("refresh_token_hash", sa.Text(), nullable=False),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("ip_address", postgresql.INET(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint("expires_at > created_at", name="ck_user_sessions_expires_after_create"),
        sa.CheckConstraint("revoked_at IS NULL OR revoked_at >= created_at", name="ck_user_sessions_revoked_after_create"),
        sa.ForeignKeyConstraint(["user_id"], [f"{SCHEMA}.users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("session_id"),
        sa.UniqueConstraint("refresh_token_hash", name="uq_user_sessions_refresh_token_hash"),
        schema=SCHEMA,
    )
    op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"], unique=False, schema=SCHEMA)

    op.create_table(
        "projects",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("project_code", postgresql.CITEXT(), nullable=False),
        sa.Column("owner_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("project_status", sa.Text(), nullable=False, server_default=sa.text("'active'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("project_status IN ('active', 'archived')", name="ck_projects_project_status"),
        sa.CheckConstraint("archived_at IS NULL OR archived_at >= created_at", name="ck_projects_archived_after_create"),
        sa.ForeignKeyConstraint(["owner_user_id"], [f"{SCHEMA}.users.user_id"]),
        sa.PrimaryKeyConstraint("project_id"),
        sa.UniqueConstraint("project_code", name="uq_projects_project_code"),
        schema=SCHEMA,
    )
    op.create_index("ix_projects_owner_user_id", "projects", ["owner_user_id"], unique=False, schema=SCHEMA)

    op.create_table(
        "project_memberships",
        sa.Column("membership_id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("granted_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("granted_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("role IN ('owner', 'admin', 'editor', 'viewer')", name="ck_project_memberships_role"),
        sa.CheckConstraint("revoked_at IS NULL OR revoked_at >= granted_at", name="ck_project_memberships_revoked_after_granted"),
        sa.ForeignKeyConstraint(["project_id"], [f"{SCHEMA}.projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], [f"{SCHEMA}.users.user_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["granted_by_user_id"], [f"{SCHEMA}.users.user_id"]),
        sa.PrimaryKeyConstraint("membership_id"),
        schema=SCHEMA,
    )
    op.create_index(
        "ux_project_memberships_active",
        "project_memberships",
        ["project_id", "user_id"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("revoked_at IS NULL"),
    )
    op.create_index("ix_project_memberships_user_id", "project_memberships", ["user_id"], unique=False, schema=SCHEMA)

    op.create_table(
        "dataset_versions",
        sa.Column("dataset_version_id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("source_kind", sa.Text(), nullable=False),
        sa.Column("target_name", sa.Text(), nullable=False),
        sa.Column("dataset_format", sa.Text(), nullable=True),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=True),
        sa.Column("schema_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("stats_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("rows_count", sa.BigInteger(), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint("source_kind IN ('upload', 'inline', 'api', 'import', 'generated', 'retraining')", name="ck_dataset_versions_source_kind"),
        sa.CheckConstraint("dataset_format IS NULL OR dataset_format IN ('csv', 'xlsx', 'xls', 'json', 'parquet', 'other')", name="ck_dataset_versions_dataset_format"),
        sa.CheckConstraint("jsonb_typeof(schema_jsonb) = 'object'", name="ck_dataset_versions_schema_jsonb"),
        sa.CheckConstraint("jsonb_typeof(stats_jsonb) = 'object'", name="ck_dataset_versions_stats_jsonb"),
        sa.CheckConstraint("rows_count >= 0", name="ck_dataset_versions_rows_count"),
        sa.ForeignKeyConstraint(["project_id"], [f"{SCHEMA}.projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by_user_id"], [f"{SCHEMA}.users.user_id"]),
        sa.PrimaryKeyConstraint("dataset_version_id"),
        schema=SCHEMA,
    )
    op.create_index("ix_dataset_versions_project_id", "dataset_versions", ["project_id"], unique=False, schema=SCHEMA)
    op.create_index("ix_dataset_versions_created_by_user_id", "dataset_versions", ["created_by_user_id"], unique=False, schema=SCHEMA)

    op.create_table(
        "training_runs",
        sa.Column("training_run_id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("requested_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("base_model_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("run_kind", sa.Text(), nullable=False),
        sa.Column("history_scope", sa.Text(), nullable=False),
        sa.Column("requested_options_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("effective_options_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("evaluation_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("run_status", sa.Text(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("run_kind IN ('initial', 'retrain')", name="ck_training_runs_run_kind"),
        sa.CheckConstraint("history_scope IN ('all_history', 'last_30_days', 'last_60_days', 'last_90_days')", name="ck_training_runs_history_scope"),
        sa.CheckConstraint("jsonb_typeof(requested_options_jsonb) = 'object'", name="ck_training_runs_requested_options_jsonb"),
        sa.CheckConstraint("jsonb_typeof(effective_options_jsonb) = 'object'", name="ck_training_runs_effective_options_jsonb"),
        sa.CheckConstraint("jsonb_typeof(evaluation_jsonb) = 'object'", name="ck_training_runs_evaluation_jsonb"),
        sa.CheckConstraint("run_status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')", name="ck_training_runs_run_status"),
        sa.CheckConstraint("finished_at IS NULL OR finished_at >= started_at", name="ck_training_runs_finished_after_started"),
        sa.ForeignKeyConstraint(["project_id"], [f"{SCHEMA}.projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["dataset_version_id"], [f"{SCHEMA}.dataset_versions.dataset_version_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["requested_by_user_id"], [f"{SCHEMA}.users.user_id"]),
        sa.PrimaryKeyConstraint("training_run_id"),
        schema=SCHEMA,
    )
    op.create_index("ix_training_runs_project_id", "training_runs", ["project_id"], unique=False, schema=SCHEMA)
    op.create_index("ix_training_runs_dataset_version_id", "training_runs", ["dataset_version_id"], unique=False, schema=SCHEMA)
    op.create_index("ix_training_runs_requested_by_user_id", "training_runs", ["requested_by_user_id"], unique=False, schema=SCHEMA)

    op.create_table(
        "model_versions",
        sa.Column("model_version_id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("training_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_name", sa.Text(), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("task_type", sa.Text(), nullable=False),
        sa.Column("primary_metric", sa.Text(), nullable=False),
        sa.Column("metrics_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("feature_names_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("forecast_profile_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint("task_type IN ('binary', 'multiclass', 'regression')", name="ck_model_versions_task_type"),
        sa.CheckConstraint("jsonb_typeof(metrics_jsonb) = 'object'", name="ck_model_versions_metrics_jsonb"),
        sa.CheckConstraint("jsonb_typeof(feature_names_jsonb) = 'array'", name="ck_model_versions_feature_names_jsonb"),
        sa.CheckConstraint("jsonb_typeof(forecast_profile_jsonb) = 'object'", name="ck_model_versions_forecast_profile_jsonb"),
        sa.ForeignKeyConstraint(["project_id"], [f"{SCHEMA}.projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["training_run_id"], [f"{SCHEMA}.training_runs.training_run_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["dataset_version_id"], [f"{SCHEMA}.dataset_versions.dataset_version_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("model_version_id"),
        sa.UniqueConstraint("training_run_id", name="uq_model_versions_training_run_id"),
        schema=SCHEMA,
    )
    op.create_index("ix_model_versions_project_id", "model_versions", ["project_id"], unique=False, schema=SCHEMA)
    op.create_index("ix_model_versions_dataset_version_id", "model_versions", ["dataset_version_id"], unique=False, schema=SCHEMA)

    op.create_table(
        "model_deployments",
        sa.Column("deployment_id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("deployment_role", sa.Text(), nullable=False),
        sa.Column("deployed_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("deployment_reason", sa.Text(), nullable=True),
        sa.Column("rollback_of_deployment_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("deployed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("undeployed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("deployment_role IN ('champion', 'serving')", name="ck_model_deployments_role"),
        sa.CheckConstraint("undeployed_at IS NULL OR undeployed_at >= deployed_at", name="ck_model_deployments_undeployed_after_deployed"),
        sa.ForeignKeyConstraint(["project_id"], [f"{SCHEMA}.projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_version_id"], [f"{SCHEMA}.model_versions.model_version_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["deployed_by_user_id"], [f"{SCHEMA}.users.user_id"]),
        sa.ForeignKeyConstraint(["rollback_of_deployment_id"], [f"{SCHEMA}.model_deployments.deployment_id"]),
        sa.PrimaryKeyConstraint("deployment_id"),
        schema=SCHEMA,
    )
    op.create_index(
        "ux_model_deployments_active",
        "model_deployments",
        ["project_id", "deployment_role"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("undeployed_at IS NULL"),
    )
    op.create_index("ix_model_deployments_model_version_id", "model_deployments", ["model_version_id"], unique=False, schema=SCHEMA)

    op.create_table(
        "inference_runs",
        sa.Column("inference_run_id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("requested_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("request_mode", sa.Text(), nullable=False),
        sa.Column("input_artifact_uri", sa.Text(), nullable=True),
        sa.Column("row_count", sa.BigInteger(), nullable=False),
        sa.Column("predictions_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("comparison_metrics_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("explanations_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("recommendations_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint("request_mode IN ('inline', 'file', 'batch', 'compare')", name="ck_inference_runs_request_mode"),
        sa.CheckConstraint("row_count >= 0", name="ck_inference_runs_row_count"),
        sa.CheckConstraint("jsonb_typeof(predictions_jsonb) = 'array'", name="ck_inference_runs_predictions_jsonb"),
        sa.CheckConstraint("jsonb_typeof(comparison_metrics_jsonb) = 'object'", name="ck_inference_runs_comparison_metrics_jsonb"),
        sa.CheckConstraint("jsonb_typeof(explanations_jsonb) = 'array'", name="ck_inference_runs_explanations_jsonb"),
        sa.CheckConstraint("jsonb_typeof(recommendations_jsonb) = 'array'", name="ck_inference_runs_recommendations_jsonb"),
        sa.ForeignKeyConstraint(["project_id"], [f"{SCHEMA}.projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_version_id"], [f"{SCHEMA}.model_versions.model_version_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["requested_by_user_id"], [f"{SCHEMA}.users.user_id"]),
        sa.PrimaryKeyConstraint("inference_run_id"),
        schema=SCHEMA,
    )
    op.create_index("ix_inference_runs_project_id", "inference_runs", ["project_id"], unique=False, schema=SCHEMA)
    op.create_index("ix_inference_runs_model_version_id", "inference_runs", ["model_version_id"], unique=False, schema=SCHEMA)

    op.create_table(
        "forecast_runs",
        sa.Column("forecast_run_id", postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("requested_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("horizon_minutes", sa.Integer(), nullable=False),
        sa.Column("steps", sa.Integer(), nullable=False),
        sa.Column("recent_history_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("forecast_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("metrics_jsonb", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("warning_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint("horizon_minutes >= 1", name="ck_forecast_runs_horizon_minutes"),
        sa.CheckConstraint("steps >= 1", name="ck_forecast_runs_steps"),
        sa.CheckConstraint("jsonb_typeof(recent_history_jsonb) = 'array'", name="ck_forecast_runs_recent_history_jsonb"),
        sa.CheckConstraint("jsonb_typeof(forecast_jsonb) = 'array'", name="ck_forecast_runs_forecast_jsonb"),
        sa.CheckConstraint("jsonb_typeof(metrics_jsonb) = 'object'", name="ck_forecast_runs_metrics_jsonb"),
        sa.ForeignKeyConstraint(["project_id"], [f"{SCHEMA}.projects.project_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_version_id"], [f"{SCHEMA}.model_versions.model_version_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["requested_by_user_id"], [f"{SCHEMA}.users.user_id"]),
        sa.PrimaryKeyConstraint("forecast_run_id"),
        schema=SCHEMA,
    )
    op.create_index("ix_forecast_runs_project_id", "forecast_runs", ["project_id"], unique=False, schema=SCHEMA)
    op.create_index("ix_forecast_runs_model_version_id", "forecast_runs", ["model_version_id"], unique=False, schema=SCHEMA)


def downgrade() -> None:
    op.drop_index("ix_forecast_runs_model_version_id", table_name="forecast_runs", schema=SCHEMA)
    op.drop_index("ix_forecast_runs_project_id", table_name="forecast_runs", schema=SCHEMA)
    op.drop_table("forecast_runs", schema=SCHEMA)

    op.drop_index("ix_inference_runs_model_version_id", table_name="inference_runs", schema=SCHEMA)
    op.drop_index("ix_inference_runs_project_id", table_name="inference_runs", schema=SCHEMA)
    op.drop_table("inference_runs", schema=SCHEMA)

    op.drop_index("ix_model_deployments_model_version_id", table_name="model_deployments", schema=SCHEMA)
    op.drop_index("ux_model_deployments_active", table_name="model_deployments", schema=SCHEMA)
    op.drop_table("model_deployments", schema=SCHEMA)

    op.drop_index("ix_model_versions_dataset_version_id", table_name="model_versions", schema=SCHEMA)
    op.drop_index("ix_model_versions_project_id", table_name="model_versions", schema=SCHEMA)
    op.drop_table("model_versions", schema=SCHEMA)

    op.drop_index("ix_training_runs_requested_by_user_id", table_name="training_runs", schema=SCHEMA)
    op.drop_index("ix_training_runs_dataset_version_id", table_name="training_runs", schema=SCHEMA)
    op.drop_index("ix_training_runs_project_id", table_name="training_runs", schema=SCHEMA)
    op.drop_table("training_runs", schema=SCHEMA)

    op.drop_index("ix_dataset_versions_created_by_user_id", table_name="dataset_versions", schema=SCHEMA)
    op.drop_index("ix_dataset_versions_project_id", table_name="dataset_versions", schema=SCHEMA)
    op.drop_table("dataset_versions", schema=SCHEMA)

    op.drop_index("ix_project_memberships_user_id", table_name="project_memberships", schema=SCHEMA)
    op.drop_index("ux_project_memberships_active", table_name="project_memberships", schema=SCHEMA)
    op.drop_table("project_memberships", schema=SCHEMA)

    op.drop_index("ix_projects_owner_user_id", table_name="projects", schema=SCHEMA)
    op.drop_table("projects", schema=SCHEMA)

    op.drop_index("ix_user_sessions_user_id", table_name="user_sessions", schema=SCHEMA)
    op.drop_table("user_sessions", schema=SCHEMA)

    op.drop_table("users", schema=SCHEMA)
    op.execute(f"DROP SCHEMA IF EXISTS {SCHEMA}")
