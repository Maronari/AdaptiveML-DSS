from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd

from backend.models.domain import DatasetVersion, ModelVersion
from backend.services.settings import get_settings
from storage.registry.filesystem import FilesystemRegistry


class RegistryService:
    def __init__(self) -> None:
        """Manage dataset and model versions in the filesystem registry."""
        self.settings = get_settings()
        self.registry = FilesystemRegistry(self.settings.registry_dir)

    def create_dataset_version(
        self,
        project_id: str,
        source_name: str,
        target: str,
        frame: pd.DataFrame,
    ) -> DatasetVersion:
        """Persist a new dataset snapshot and register its metadata."""
        version_id = self._new_version_id("dataset")
        project_dir = self.settings.datasets_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = project_dir / f"{version_id}.csv"
        frame.to_csv(dataset_path, index=False)

        record = DatasetVersion(
            version_id=version_id,
            project_id=project_id,
            source_name=source_name,
            target=target,
            path=str(dataset_path),
            schema={column: str(dtype) for column, dtype in frame.dtypes.items()},
            rows=int(len(frame)),
            created_at=self._now(),
        )
        datasets = self.registry.read("datasets")
        datasets.append(record.to_dict())
        self.registry.write("datasets", datasets)
        return record

    def load_dataset_version_frame(self, version_id: str) -> pd.DataFrame:
        """Load the persisted frame for a dataset version id."""
        dataset = self.get_dataset_version(version_id)
        return pd.read_csv(dataset["path"])

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

        models = self.registry.read("models")
        version_id = self._new_version_id("model")
        artifact_dir = self.settings.artifacts_dir / project_id / version_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = artifact_dir / "model.joblib"
        joblib.dump(bundle, artifact_path)

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
            artifact_path=str(artifact_path),
            task_type=task_type,
            target=target,
            metrics=metrics,
            primary_metric=primary_metric,
            status=status,
            feature_names=feature_names,
            created_at=self._now(),
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

        bundle = joblib.load(champion["artifact_path"])
        return champion, bundle

    def get_model_version(self, version_id: str) -> dict[str, Any]:
        """Return one model version record by id."""
        models = self.registry.read("models")
        for model in models:
            if model["version_id"] == version_id:
                return model
        raise ValueError(f"Model version '{version_id}' was not found.")

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
