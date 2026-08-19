from __future__ import annotations

from typing import Any

from fastapi import UploadFile
from fastapi.concurrency import run_in_threadpool

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
        training_options: dict[str, Any] | None = None,
        name: str | None = None,
        unit: str | None = None,
    ) -> dict[str, Any]:
        """Train a model from inline records and register the result."""
        frame = dataframe_from_records(records)
        return self._train_frame(
            project_id=project_id,
            target=target,
            frame=frame,
            source_name=source_name,
            training_options=training_options,
            name=name,
            unit=unit,
        )

    async def train_from_upload(
        self,
        project_id: str,
        target: str,
        upload: UploadFile,
        training_options: dict[str, Any] | None = None,
        name: str | None = None,
        unit: str | None = None,
    ) -> dict[str, Any]:
        """Train a model from an uploaded CSV or Excel file."""
        frame = await self.dataset_service.read_uploaded_tabular(upload)
        records = frame.to_dict(orient="records")
        source_name = upload.filename or "uploaded.csv"
        return await run_in_threadpool(
            self.train,
            project_id=project_id,
            target=target,
            records=records,
            source_name=source_name,
            training_options=training_options,
            name=name,
            unit=unit,
        )

    def train_from_dataset_version(
        self,
        project_id: str,
        dataset_version_id: str,
        training_options: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Train a model from a dataset version that was uploaded earlier."""
        dataset_version = self.registry_service.get_dataset_version(dataset_version_id)
        if dataset_version["project_id"] != project_id:
            raise ValueError(
                f"Dataset version '{dataset_version_id}' does not belong to project '{project_id}'."
            )

        frame = self.registry_service.load_dataset_version_frame(dataset_version_id)
        return self._train_frame(
            project_id=project_id,
            target=dataset_version["target"],
            frame=frame,
            source_name=dataset_version["source_name"],
            training_options=training_options,
            dataset_version=dataset_version,
            name=name,
        )

    def _train_frame(
        self,
        project_id: str,
        target: str,
        frame,
        source_name: str,
        training_options: dict[str, Any] | None = None,
        dataset_version: dict[str, Any] | None = None,
        name: str | None = None,
        unit: str | None = None,
    ) -> dict[str, Any]:
        """Train on a prepared frame, optionally reusing an existing dataset version."""
        self.dataset_service._validate_training_frame(frame, target)

        if dataset_version is None:
            created_dataset_version = self.registry_service.create_dataset_version(
                project_id=project_id,
                source_name=source_name,
                target=target,
                frame=frame,
                unit=unit,
            )
            dataset_payload = created_dataset_version.to_dict()
            dataset_version_id = created_dataset_version.version_id
        else:
            dataset_payload = dataset_version
            dataset_version_id = dataset_version["version_id"]

        training_result = self.adapter.train(
            frame=frame,
            target=target,
            training_options=training_options,
        )
        model_version = self.registry_service.register_model_version(
            project_id=project_id,
            dataset_version_id=dataset_version_id,
            target=target,
            task_type=training_result.task_type,
            feature_names=training_result.feature_names,
            metrics=training_result.metrics,
            bundle=training_result.bundle,
            holdout_predictions=training_result.holdout_predictions,
            training_artifacts=training_result.training_artifacts,
            name=name,
        )
        forecasting_bundle = training_result.bundle.get("forecasting")

        return {
            "project_id": project_id,
            "dataset_version": dataset_payload,
            "model_version": model_version.to_dict(),
            "metrics": training_result.metrics,
            "task_type": training_result.task_type,
            "backend": training_result.backend_name,
            "training_options": training_result.training_options,
            "forecasting": {
                "available": forecasting_bundle is not None,
                "default_horizon_minutes": (
                    forecasting_bundle["default_horizon_minutes"] if forecasting_bundle is not None else None
                ),
                "base_frequency_minutes": (
                    forecasting_bundle["base_frequency_minutes"] if forecasting_bundle is not None else None
                ),
                "forecast_metrics": forecasting_bundle["metrics"] if forecasting_bundle is not None else None,
            },
            "warnings": training_result.warnings,
        }
