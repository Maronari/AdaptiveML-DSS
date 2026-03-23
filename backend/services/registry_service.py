from __future__ import annotations

from datetime import UTC, datetime
import re
import shutil
from typing import Any
from uuid import uuid4

import pandas as pd

from backend.models.domain import DatasetVersion, ModelVersion, ProjectRecord
from backend.services.object_storage_service import ObjectStorageService
from backend.services.settings import get_settings
from storage.registry.filesystem import FilesystemRegistry
from storage.registry.sqlite import SQLiteRegistry


class RegistryService:
    def __init__(self) -> None:
        """Manage dataset and model versions in the SQLite-backed registry."""
        self.settings = get_settings()
        self.registry = SQLiteRegistry(self.settings.registry_db_path)
        self.legacy_registry = FilesystemRegistry(self.settings.registry_dir)
        self.object_storage = ObjectStorageService()
        self._migrate_legacy_registry()

    def create_dataset_version(
        self,
        project_id: str,
        source_name: str,
        target: str,
        frame: pd.DataFrame,
    ) -> DatasetVersion:
        """Persist a new dataset snapshot and register its metadata."""
        self._ensure_project_record(project_id)
        version_id = self._new_version_id("dataset")
        storage_ref = self.object_storage.save_dataset_frame(
            project_id=project_id,
            version_id=version_id,
            frame=frame,
        )

        record = DatasetVersion(
            version_id=version_id,
            project_id=project_id,
            source_name=source_name,
            target=target,
            path=storage_ref.path,
            schema={column: str(dtype) for column, dtype in frame.dtypes.items()},
            rows=int(len(frame)),
            created_at=self._now(),
            storage_backend=storage_ref.storage_backend,
            bucket=storage_ref.bucket,
            object_key=storage_ref.object_key,
        )
        datasets = self.registry.read("datasets")
        datasets.append(record.to_dict())
        self.registry.write("datasets", datasets)
        return record

    def load_dataset_version_frame(self, version_id: str) -> pd.DataFrame:
        """Load the persisted frame for a dataset version id."""
        dataset = self.get_dataset_version(version_id)
        return self.object_storage.load_dataset_frame(dataset)

    def get_dataset_version(self, version_id: str) -> dict[str, Any]:
        """Return one dataset version record by id."""
        datasets = self.registry.read("datasets")
        for dataset in datasets:
            if dataset["version_id"] == version_id:
                return dataset
        raise ValueError(f"Dataset version '{version_id}' was not found.")

    def register_model_version(
        self,
        project_id: str,
        dataset_version_id: str,
        target: str,
        task_type: str,
        feature_names: list[str],
        metrics: dict[str, float],
        bundle: dict[str, Any],
        promotion_mode: str = "auto",
    ) -> ModelVersion:
        """Persist a trained model bundle and update champion status."""
        if promotion_mode not in {"auto", "candidate"}:
            raise ValueError("promotion_mode must be 'auto' or 'candidate'.")

        self._ensure_project_record(project_id)
        models = self.registry.read("models")
        version_id = self._new_version_id("model")
        storage_ref = self.object_storage.save_model_bundle(
            project_id=project_id,
            version_id=version_id,
            bundle=bundle,
        )

        primary_metric = self._primary_metric(task_type)
        status = "candidate"
        if promotion_mode == "auto":
            champion = self._find_champion(models, project_id)
            if champion is None or self._is_better(metrics, champion["metrics"], primary_metric):
                status = "champion"
                for model in models:
                    if model["project_id"] == project_id and model["status"] == "champion":
                        model["status"] = "archived"

        record = ModelVersion(
            version_id=version_id,
            project_id=project_id,
            dataset_version_id=dataset_version_id,
            artifact_path=storage_ref.path,
            task_type=task_type,
            target=target,
            metrics=metrics,
            primary_metric=primary_metric,
            status=status,
            feature_names=feature_names,
            created_at=self._now(),
            storage_backend=storage_ref.storage_backend,
            bucket=storage_ref.bucket,
            object_key=storage_ref.object_key,
        )
        models.append(record.to_dict())
        self.registry.write("models", models)
        return record

    def get_champion_bundle(self, project_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load the current champion model metadata and serialized bundle."""
        models = self.registry.read("models")
        champion = self._find_champion(models, project_id)
        if champion is None:
            raise ValueError(f"No champion model found for project '{project_id}'.")

        bundle = self.object_storage.load_model_bundle(champion)
        return champion, bundle

    def get_latest_model_bundle(self, project_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load the latest model metadata and serialized bundle for a project."""
        latest = self.get_latest_model_version(project_id)
        bundle = self.object_storage.load_model_bundle(latest)
        return latest, bundle

    def get_model_version(self, version_id: str) -> dict[str, Any]:
        """Return one model version record by id."""
        models = self.registry.read("models")
        for model in models:
            if model["version_id"] == version_id:
                return model
        raise ValueError(f"Model version '{version_id}' was not found.")

    def get_model_bundle(self, version_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load a model version and its serialized bundle."""
        model = self.get_model_version(version_id)
        bundle = self.object_storage.load_model_bundle(model)
        return model, bundle

    def get_latest_model_version(self, project_id: str) -> dict[str, Any]:
        """Return the newest model version record for a project."""
        models = self.registry.read("models")
        for model in reversed(models):
            if model["project_id"] == project_id:
                return model
        raise ValueError(f"No model versions found for project '{project_id}'.")

    def save_latest_comparison(self, comparison: dict[str, Any]) -> None:
        """Persist the latest compact comparison payload for the graph page."""
        self.registry.write("latest_comparison", [comparison])

    def get_latest_comparison(self) -> dict[str, Any]:
        """Return the most recent compact comparison payload."""
        comparisons = self.registry.read("latest_comparison")
        if not comparisons:
            raise ValueError("No comparison data found yet.")
        return comparisons[-1]

    def activate_model_version(self, project_id: str, version_id: str) -> ModelVersion:
        """Mark the selected version as champion and archive the previous champion."""
        models = self.registry.read("models")
        selected: dict[str, Any] | None = None

        for model in models:
            if model["project_id"] != project_id:
                continue
            if model["status"] == "champion":
                model["status"] = "archived"
            if model["version_id"] == version_id:
                selected = model

        if selected is None:
            raise ValueError(f"Model version '{version_id}' was not found for project '{project_id}'.")

        selected["status"] = "champion"
        self.registry.write("models", models)
        return ModelVersion(**selected)

    def create_project(self, name: str) -> dict[str, Any]:
        """Create a new project entry that can be selected before any training run."""
        normalized_name = " ".join(name.split())
        if not normalized_name:
            raise ValueError("Project name must not be empty.")

        projects = self.registry.read("projects")
        existing_project_ids = {project["project_id"] for project in projects}
        existing_project_ids.update(dataset["project_id"] for dataset in self.registry.read("datasets"))
        existing_project_ids.update(model["project_id"] for model in self.registry.read("models"))

        project_id = self._build_unique_project_id(normalized_name, existing_project_ids)
        record = ProjectRecord(
            project_id=project_id,
            name=normalized_name,
            created_at=self._now(),
        )
        projects.append(record.to_dict())
        self.registry.write("projects", projects)
        return record.to_dict()

    def list_projects(self) -> list[dict[str, Any]]:
        """Return project summaries aggregated from explicit records and registry activity."""
        project_records = self.registry.read("projects")
        datasets = self.registry.read("datasets")
        models = self.registry.read("models")

        projects_by_id: dict[str, dict[str, Any]] = {}

        for record in project_records:
            projects_by_id[record["project_id"]] = dict(record)

        for dataset in datasets:
            project = projects_by_id.setdefault(
                dataset["project_id"],
                {
                    "project_id": dataset["project_id"],
                    "name": dataset["project_id"],
                    "created_at": dataset["created_at"],
                },
            )
            project["created_at"] = min(project["created_at"], dataset["created_at"])

        for model in models:
            project = projects_by_id.setdefault(
                model["project_id"],
                {
                    "project_id": model["project_id"],
                    "name": model["project_id"],
                    "created_at": model["created_at"],
                },
            )
            project["created_at"] = min(project["created_at"], model["created_at"])

        summaries: list[dict[str, Any]] = []
        for project_id, record in projects_by_id.items():
            project_datasets = [dataset for dataset in datasets if dataset["project_id"] == project_id]
            project_models = [model for model in models if model["project_id"] == project_id]
            latest_dataset = self._latest_by_created_at(project_datasets)
            latest_model = self._latest_by_created_at(project_models)
            champion_model = next(
                (model for model in reversed(project_models) if model["status"] == "champion"),
                None,
            )
            latest_activity_at = max(
                [record["created_at"]]
                + [dataset["created_at"] for dataset in project_datasets]
                + [model["created_at"] for model in project_models]
            )

            summaries.append(
                {
                    "project_id": project_id,
                    "name": record.get("name") or project_id,
                    "created_at": record["created_at"],
                    "latest_activity_at": latest_activity_at,
                    "last_dataset_at": latest_dataset["created_at"] if latest_dataset else None,
                    "last_trained_at": latest_model["created_at"] if latest_model else None,
                    "dataset_versions": len(project_datasets),
                    "model_versions": len(project_models),
                    "target": (
                        latest_model["target"]
                        if latest_model
                        else latest_dataset["target"] if latest_dataset else None
                    ),
                    "task_type": latest_model["task_type"] if latest_model else None,
                    "latest_model_version_id": latest_model["version_id"] if latest_model else None,
                    "champion_model_version_id": champion_model["version_id"] if champion_model else None,
                    "has_models": bool(project_models),
                    "has_champion_model": champion_model is not None,
                    "status": champion_model["status"] if champion_model else "empty",
                }
            )

        return sorted(summaries, key=lambda item: item["latest_activity_at"], reverse=True)

    def delete_project(self, project_id: str) -> dict[str, Any]:
        """Delete one project together with all related registry records and stored files."""
        projects = self.registry.read("projects")
        datasets = self.registry.read("datasets")
        models = self.registry.read("models")
        latest_comparison = self.registry.read("latest_comparison")

        project_exists = any(project["project_id"] == project_id for project in projects)
        dataset_records = [dataset for dataset in datasets if dataset["project_id"] == project_id]
        model_records = [model for model in models if model["project_id"] == project_id]

        if not project_exists and not dataset_records and not model_records:
            raise ValueError(f"Project '{project_id}' was not found.")

        self.registry.write(
            "projects",
            [project for project in projects if project["project_id"] != project_id],
        )
        self.registry.write(
            "datasets",
            [dataset for dataset in datasets if dataset["project_id"] != project_id],
        )
        self.registry.write(
            "models",
            [model for model in models if model["project_id"] != project_id],
        )
        self.registry.write(
            "latest_comparison",
            [item for item in latest_comparison if item.get("project_id") != project_id],
        )

        for dataset in dataset_records:
            self.object_storage.delete_dataset(dataset)
        for model in model_records:
            self.object_storage.delete_model(model)

        shutil.rmtree(self.settings.datasets_dir / project_id, ignore_errors=True)
        shutil.rmtree(self.settings.artifacts_dir / project_id, ignore_errors=True)

        return {
            "project_id": project_id,
            "deleted": True,
            "dataset_versions_removed": len(dataset_records),
            "model_versions_removed": len(model_records),
        }

    @staticmethod
    def _primary_metric(task_type: str) -> str:
        """Choose the metric that decides champion promotion."""
        return "rmse" if task_type == "regression" else "f1_weighted"

    @staticmethod
    def _is_better(
        candidate_metrics: dict[str, float],
        champion_metrics: dict[str, float],
        primary_metric: str,
    ) -> bool:
        """Compare candidate and champion using the chosen primary metric."""
        candidate = float(candidate_metrics[primary_metric])
        champion = float(champion_metrics[primary_metric])

        if primary_metric == "rmse":
            return candidate < champion
        return candidate > champion

    @staticmethod
    def _find_champion(models: list[dict[str, Any]], project_id: str) -> dict[str, Any] | None:
        """Return the active champion model record for a project."""
        for model in models:
            if model["project_id"] == project_id and model["status"] == "champion":
                return model
        return None

    @staticmethod
    def _new_version_id(prefix: str) -> str:
        """Create a short version identifier with a stable prefix."""
        return f"{prefix}-{uuid4().hex[:12]}"

    @staticmethod
    def _now() -> str:
        """Return the current UTC timestamp in ISO format."""
        return datetime.now(UTC).isoformat()

    def _migrate_legacy_registry(self) -> None:
        """Import old JSON registry collections into SQLite on first access."""
        for name in ("projects", "datasets", "models", "latest_comparison"):
            if self.registry.read(name):
                continue
            legacy_payload = self.legacy_registry.read(name)
            if legacy_payload:
                self.registry.write(name, legacy_payload)

    def _ensure_project_record(self, project_id: str, name: str | None = None) -> dict[str, Any]:
        """Ensure an explicit project record exists for the selected project id."""
        projects = self.registry.read("projects")
        for project in projects:
            if project["project_id"] == project_id:
                if name and project.get("name") != name:
                    project["name"] = name
                    self.registry.write("projects", projects)
                return project

        record = ProjectRecord(
            project_id=project_id,
            name=name or project_id,
            created_at=self._now(),
        )
        projects.append(record.to_dict())
        self.registry.write("projects", projects)
        return record.to_dict()

    @staticmethod
    def _latest_by_created_at(records: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Return the most recent record according to its ISO timestamp."""
        if not records:
            return None
        return max(records, key=lambda item: item["created_at"])

    @classmethod
    def _build_unique_project_id(cls, name: str, existing_ids: set[str]) -> str:
        """Create a URL-safe project id and avoid collisions with existing project ids."""
        base = cls._slugify_project_name(name)
        if base not in existing_ids:
            return base

        suffix = 2
        while f"{base}-{suffix}" in existing_ids:
            suffix += 1
        return f"{base}-{suffix}"

    @staticmethod
    def _slugify_project_name(name: str) -> str:
        """Convert a display name into a lowercase project id."""
        slug = name.casefold()
        slug = slug.replace("ё", "е")
        slug = re.sub(r"[^a-z0-9а-я]+", "-", slug)
        slug = slug.strip("-")
        return slug or "project"
