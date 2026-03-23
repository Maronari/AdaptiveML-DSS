"""simplify core schema for MVP

Revision ID: 20260322_0002
Revises: 20260321_0001
Create Date: 2026-03-22 13:10:00

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260322_0002"
down_revision = "20260321_0001"
branch_labels = None
depends_on = None


SCHEMA = "adaptive_ml_dss"


def upgrade() -> None:
    op.drop_index("ix_forecast_runs_model_version_id", table_name="forecast_runs", schema=SCHEMA)
    op.drop_index("ix_forecast_runs_project_id", table_name="forecast_runs", schema=SCHEMA)
    op.drop_table("forecast_runs", schema=SCHEMA)

    op.drop_index("ix_inference_runs_model_version_id", table_name="inference_runs", schema=SCHEMA)
    op.drop_index("ix_inference_runs_project_id", table_name="inference_runs", schema=SCHEMA)
    op.drop_table("inference_runs", schema=SCHEMA)

    op.drop_index("ix_project_memberships_user_id", table_name="project_memberships", schema=SCHEMA)
    op.drop_index("ux_project_memberships_active", table_name="project_memberships", schema=SCHEMA)
    op.drop_table("project_memberships", schema=SCHEMA)

    op.drop_index("ix_user_sessions_user_id", table_name="user_sessions", schema=SCHEMA)
    op.drop_table("user_sessions", schema=SCHEMA)

    op.drop_index("ix_dataset_versions_created_by_user_id", table_name="dataset_versions", schema=SCHEMA)
    op.drop_index("ix_training_runs_requested_by_user_id", table_name="training_runs", schema=SCHEMA)
    op.drop_index("ux_model_deployments_active", table_name="model_deployments", schema=SCHEMA)

    op.execute(f"ALTER TABLE {SCHEMA}.dataset_versions DROP COLUMN created_by_user_id")
    op.execute(f"ALTER TABLE {SCHEMA}.training_runs DROP COLUMN requested_by_user_id")
    op.execute(f"ALTER TABLE {SCHEMA}.users DROP COLUMN user_status")
    op.execute(f"ALTER TABLE {SCHEMA}.projects DROP COLUMN project_status")
    op.execute(f"ALTER TABLE {SCHEMA}.projects DROP COLUMN archived_at")
    op.execute(f"ALTER TABLE {SCHEMA}.model_deployments DROP COLUMN deployed_by_user_id")
    op.execute(f"ALTER TABLE {SCHEMA}.model_deployments DROP COLUMN deployment_role")

    op.add_column(
        "model_versions",
        sa.Column("version_number", sa.Integer(), nullable=True),
        schema=SCHEMA,
    )
    op.execute(
        f"""
        WITH ranked AS (
            SELECT
                model_version_id,
                ROW_NUMBER() OVER (
                    PARTITION BY project_id
                    ORDER BY created_at, model_version_id
                ) AS version_number
            FROM {SCHEMA}.model_versions
        )
        UPDATE {SCHEMA}.model_versions AS mv
        SET version_number = ranked.version_number
        FROM ranked
        WHERE mv.model_version_id = ranked.model_version_id
        """
    )
    op.alter_column("model_versions", "version_number", nullable=False, schema=SCHEMA)
    op.create_check_constraint(
        "ck_model_versions_version_number",
        "model_versions",
        "version_number >= 1",
        schema=SCHEMA,
    )
    op.create_unique_constraint(
        "uq_model_versions_project_version_number",
        "model_versions",
        ["project_id", "version_number"],
        schema=SCHEMA,
    )

    op.create_index(
        "ux_model_deployments_active",
        "model_deployments",
        ["project_id"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("undeployed_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ux_model_deployments_active", table_name="model_deployments", schema=SCHEMA)

    op.drop_constraint(
        "uq_model_versions_project_version_number",
        "model_versions",
        type_="unique",
        schema=SCHEMA,
    )
    op.drop_constraint(
        "ck_model_versions_version_number",
        "model_versions",
        type_="check",
        schema=SCHEMA,
    )
    op.drop_column("model_versions", "version_number", schema=SCHEMA)

    op.add_column(
        "model_deployments",
        sa.Column("deployment_role", sa.Text(), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        "model_deployments",
        sa.Column("deployed_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        schema=SCHEMA,
    )
    op.execute(f"UPDATE {SCHEMA}.model_deployments SET deployment_role = 'serving'")
    op.alter_column("model_deployments", "deployment_role", nullable=False, schema=SCHEMA)
    op.create_check_constraint(
        "ck_model_deployments_role",
        "model_deployments",
        "deployment_role IN ('champion', 'serving')",
        schema=SCHEMA,
    )
    op.create_foreign_key(
        None,
        "model_deployments",
        "users",
        ["deployed_by_user_id"],
        ["user_id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
    )
    op.create_index(
        "ux_model_deployments_active",
        "model_deployments",
        ["project_id", "deployment_role"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("undeployed_at IS NULL"),
    )

    op.add_column(
        "projects",
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        schema=SCHEMA,
    )
    op.add_column(
        "projects",
        sa.Column("project_status", sa.Text(), nullable=False, server_default=sa.text("'active'")),
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_projects_project_status",
        "projects",
        "project_status IN ('active', 'archived')",
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_projects_archived_after_create",
        "projects",
        "archived_at IS NULL OR archived_at >= created_at",
        schema=SCHEMA,
    )

    op.add_column(
        "users",
        sa.Column("user_status", sa.Text(), nullable=False, server_default=sa.text("'active'")),
        schema=SCHEMA,
    )
    op.create_check_constraint(
        "ck_users_user_status",
        "users",
        "user_status IN ('active', 'disabled', 'locked')",
        schema=SCHEMA,
    )

    op.add_column(
        "training_runs",
        sa.Column("requested_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        schema=SCHEMA,
    )
    op.create_foreign_key(
        None,
        "training_runs",
        "users",
        ["requested_by_user_id"],
        ["user_id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
    )
    op.create_index(
        "ix_training_runs_requested_by_user_id",
        "training_runs",
        ["requested_by_user_id"],
        unique=False,
        schema=SCHEMA,
    )

    op.add_column(
        "dataset_versions",
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        schema=SCHEMA,
    )
    op.create_foreign_key(
        None,
        "dataset_versions",
        "users",
        ["created_by_user_id"],
        ["user_id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
    )
    op.create_index(
        "ix_dataset_versions_created_by_user_id",
        "dataset_versions",
        ["created_by_user_id"],
        unique=False,
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
    op.execute(
        f"""
        INSERT INTO {SCHEMA}.project_memberships (
            project_id,
            user_id,
            role,
            granted_by_user_id,
            granted_at
        )
        SELECT
            project_id,
            owner_user_id,
            'owner',
            owner_user_id,
            created_at
        FROM {SCHEMA}.projects
        """
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
