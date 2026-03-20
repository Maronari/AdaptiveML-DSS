from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd

from backend.models.domain import DatasetVersion, ModelVersion
from backend.services.settings import get_settings
from storage.registry.filesystem import FilesystemRegistry


class RegistryService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.registry = FilesystemRegistry(self.settings.registry_dir)

    def create_dataset_version(
        self,
        project_id: str,
        source_name: str,
        target: str,
        frame: pd.DataFrame,
    ) -> DatasetVersion:
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

    def register_model_version(
        self,
        project_id: str,
        dataset_version_id: str,
        target: str,
        task_type: str,
        feature_names: list[str],
        metrics: dict[str, float],
        bundle: dict[str, Any],
    ) -> ModelVersion:
        models = self.registry.read("models")
        version_id = self._new_version_id("model")
        artifact_dir = self.settings.artifacts_dir / project_id / version_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = artifact_dir / "model.joblib"
        joblib.dump(bundle, artifact_path)

        primary_metric = self._primary_metric(task_type)
        status = "candidate"
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
        models = self.registry.read("models")
        champion = self._find_champion(models, project_id)
        if champion is None:
            raise ValueError(f"No champion model found for project '{project_id}'.")

        bundle = joblib.load(champion["artifact_path"])
        return champion, bundle

    @staticmethod
    def _primary_metric(task_type: str) -> str:
        return "rmse" if task_type == "regression" else "f1_weighted"

    @staticmethod
    def _is_better(
        candidate_metrics: dict[str, float],
        champion_metrics: dict[str, float],
        primary_metric: str,
    ) -> bool:
        candidate = float(candidate_metrics[primary_metric])
        champion = float(champion_metrics[primary_metric])

        if primary_metric == "rmse":
            return candidate < champion
        return candidate > champion

    @staticmethod
    def _find_champion(models: list[dict[str, Any]], project_id: str) -> dict[str, Any] | None:
        for model in models:
            if model["project_id"] == project_id and model["status"] == "champion":
                return model
        return None

    @staticmethod
    def _new_version_id(prefix: str) -> str:
        return f"{prefix}-{uuid4().hex[:12]}"

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
