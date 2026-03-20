from __future__ import annotations

from typing import Any

from fastapi import UploadFile

from automl.lightautoml_backend.adapter import TabularAutoMLAdapter
from backend.services.dataset_service import DatasetService
from backend.services.registry_service import RegistryService
from backend.utils.io import dataframe_from_records


class TrainingService:
    def __init__(self) -> None:
        """Wire dataset validation, registry access and the ML adapter."""
        self.dataset_service = DatasetService()
        self.registry_service = RegistryService()
        self.adapter = TabularAutoMLAdapter()

    def train(
        self,
        project_id: str,
        target: str,
        records: list[dict[str, Any]],
        source_name: str = "inline",
    ) -> dict[str, Any]:
        """Train a model from inline records and register the result."""
        frame = dataframe_from_records(records)
        self.dataset_service._validate_training_frame(frame, target)

        dataset_version = self.registry_service.create_dataset_version(
            project_id=project_id,
            source_name=source_name,
            target=target,
            frame=frame,
        )
        training_result = self.adapter.train(frame=frame, target=target)
        model_version = self.registry_service.register_model_version(
            project_id=project_id,
            dataset_version_id=dataset_version.version_id,
            target=target,
            task_type=training_result.task_type,
            feature_names=training_result.feature_names,
            metrics=training_result.metrics,
            bundle=training_result.bundle,
        )

        return {
            "project_id": project_id,
            "dataset_version": dataset_version.to_dict(),
            "model_version": model_version.to_dict(),
            "metrics": training_result.metrics,
            "task_type": training_result.task_type,
            "backend": training_result.backend_name,
        }

    async def train_from_upload(
        self,
        project_id: str,
        target: str,
        upload: UploadFile,
    ) -> dict[str, Any]:
        """Train a model from an uploaded CSV or Excel file."""
        frame = await self.dataset_service.read_uploaded_tabular(upload)
        records = frame.to_dict(orient="records")
        source_name = upload.filename or "uploaded.csv"
        return self.train(
            project_id=project_id,
            target=target,
            records=records,
            source_name=source_name,
        )
