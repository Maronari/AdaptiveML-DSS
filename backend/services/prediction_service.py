from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import UploadFile

from automl.evaluation.metrics import evaluate_predictions
from automl.training.forecasting import run_forecast_from_bundle
from backend.services.dataset_service import DatasetService
from backend.services.registry_service import RegistryService
from backend.utils.io import dataframe_from_records, dataframe_to_records, pythonize


class PredictionService:
    def __init__(self) -> None:
        """Provide champion-model inference for API and internal callers."""
        self.dataset_service = DatasetService()
        self.registry_service = RegistryService()

    def predict(self, project_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Run prediction for inline records using the current champion model."""
        champion, bundle = self.registry_service.get_champion_bundle(project_id)
        frame = dataframe_from_records(records)
        predictions = self.predict_with_bundle(bundle=bundle, frame=frame)
        return {
            "project_id": project_id,
            "model_version": champion["version_id"],
            "task_type": champion["task_type"],
            "predictions": predictions,
        }

    async def predict_from_upload(self, project_id: str, upload: UploadFile) -> dict[str, Any]:
        """Run prediction for an uploaded CSV or Excel file."""
        frame = await self.dataset_service.read_uploaded_tabular(upload)
        return self.predict(project_id=project_id, records=frame.to_dict(orient="records"))

    def compare_with_actual(
        self,
        project_id: str,
        target: str,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return actual/predicted pairs for a labeled dataset."""
        champion, bundle = self.registry_service.get_champion_bundle(project_id)
        if target != champion["target"]:
            raise ValueError(
                f"Target column '{target}' does not match champion target '{champion['target']}'."
            )
        frame = dataframe_from_records(records)
        if target not in frame.columns:
            raise ValueError(f"Target column '{target}' is missing.")

        actual = frame[target].copy()
        inference_frame = frame.drop(columns=[target])
        predictions = self.predict_with_bundle(bundle=bundle, frame=inference_frame)
        predicted_values = [item["prediction"] for item in predictions]
        metrics = evaluate_predictions(
            task_type=champion["task_type"],
            y_true=actual,
            y_pred=predicted_values,
        )
        raw_rows = dataframe_to_records(frame)
        saved_at = datetime.now(UTC).isoformat()

        comparison_items = []
        for index, prediction in enumerate(predictions):
            comparison_items.append(
                {
                    "row_index": index,
                    "actual": pythonize(actual.iloc[index]),
                    "prediction": prediction["prediction"],
                    "confidence": prediction["confidence"],
                    "record": raw_rows[index],
                }
            )

        response = {
            "project_id": project_id,
            "model_version": champion["version_id"],
            "task_type": champion["task_type"],
            "target": target,
            "rows": int(len(frame)),
            "columns": frame.columns.tolist(),
            "metrics": metrics,
            "items": comparison_items,
            "saved_at": saved_at,
        }
        self.registry_service.save_latest_comparison(self._build_compact_comparison(response))
        return response

    async def compare_with_actual_from_upload(
        self,
        project_id: str,
        target: str,
        upload: UploadFile,
    ) -> dict[str, Any]:
        """Return actual/predicted pairs for an uploaded labeled file."""
        frame = await self.dataset_service.read_uploaded_tabular(upload)
        return self.compare_with_actual(
            project_id=project_id,
            target=target,
            records=frame.to_dict(orient="records"),
        )

    async def inspect_comparison_upload(
        self,
        project_id: str,
        target: str,
        upload: UploadFile,
    ) -> dict[str, Any]:
        """Validate whether an uploaded labeled file matches the champion schema."""
        champion, bundle = self.registry_service.get_champion_bundle(project_id)
        frame = await self.dataset_service.read_uploaded_tabular(upload)
        uploaded_columns = frame.columns.tolist()
        target_present = target in frame.columns
        inference_frame = frame.drop(columns=[target]) if target_present else frame
        preprocessor = bundle.get("preprocessor")
        prepared = self._prepare_frame(bundle, inference_frame)
        expected_features = bundle["feature_names"]
        prepared_columns = prepared.columns.tolist()
        missing_features = [
            feature for feature in expected_features if feature not in prepared_columns
        ]
        missing_inputs = self._resolve_missing_inputs(
            inference_frame=inference_frame,
            preprocessor=preprocessor,
            missing_features=missing_features,
        )
        expected_input_columns = set(expected_features)
        if preprocessor is not None:
            expected_input_columns.update(getattr(preprocessor, "datetime_columns", []))
            expected_input_columns.update(getattr(preprocessor, "time_columns", []))

        extra_features = [
            column
            for column in uploaded_columns
            if column not in expected_input_columns and column != target
        ]
        target_matches_champion = target == champion["target"]

        return {
            "project_id": project_id,
            "model_version": champion["version_id"],
            "task_type": champion["task_type"],
            "target": target,
            "expected_target": champion["target"],
            "target_matches_champion": target_matches_champion,
            "uploaded_columns": uploaded_columns,
            "prepared_columns": prepared_columns,
            "expected_features": expected_features,
            "missing_features": missing_features,
            "missing_inputs": missing_inputs,
            "extra_features": extra_features,
            "ready": target_present and target_matches_champion and not missing_features,
        }

    def get_latest_comparison(self) -> dict[str, Any]:
        """Return the latest compact comparison payload for graph rendering."""
        return self.registry_service.get_latest_comparison()

    def get_latest_model_summary(self, project_id: str) -> dict[str, Any]:
        """Return metadata for the newest model version of a project."""
        model, bundle = self.registry_service.get_latest_model_bundle(project_id)
        return self._build_model_summary(model, bundle)

    def get_model_summary(self, version_id: str) -> dict[str, Any]:
        """Return metadata for one model version."""
        model, bundle = self.registry_service.get_model_bundle(version_id)
        return self._build_model_summary(model, bundle)

    def forecast(
        self,
        project_id: str,
        horizon_minutes: int = 30,
        steps: int = 1,
    ) -> dict[str, Any]:
        """Forecast future target values from the champion model's stored history."""
        champion, bundle = self.registry_service.get_champion_bundle(project_id)
        if champion["task_type"] != "regression":
            raise ValueError("Forecasting is available only for regression models.")

        forecast_payload = run_forecast_from_bundle(
            bundle=bundle,
            steps=steps,
            horizon_minutes=horizon_minutes,
        )
        return {
            "project_id": project_id,
            "model_version": champion["version_id"],
            "task_type": champion["task_type"],
            **forecast_payload,
        }

    def forecast_model_version(
        self,
        version_id: str,
        horizon_minutes: int | None = None,
        steps: int | None = None,
    ) -> dict[str, Any]:
        """Forecast future target values for a specific stored model version."""
        model, bundle = self.registry_service.get_model_bundle(version_id)
        if model["task_type"] != "regression":
            raise ValueError("Forecasting is available only for regression models.")

        forecasting = bundle.get("forecasting")
        if not forecasting or not forecasting.get("enabled"):
            raise ValueError("Forecasting is unavailable for this model.")

        resolved_horizon = int(horizon_minutes or forecasting.get("default_horizon_minutes") or 30)
        resolved_steps = int(steps or 12)
        forecast_payload = run_forecast_from_bundle(
            bundle=bundle,
            steps=resolved_steps,
            horizon_minutes=resolved_horizon,
        )
        return {
            "project_id": model["project_id"],
            "model_version": model["version_id"],
            "task_type": model["task_type"],
            **forecast_payload,
        }

    def predict_with_bundle(self, bundle: dict[str, Any], frame) -> list[dict[str, Any]]:
        """Predict from a preloaded model bundle and a raw DataFrame."""
        prepared = self._prepare_frame(bundle, frame)
        aligned = self.dataset_service.validate_prediction_frame(prepared, bundle["feature_names"])
        raw_predictions, confidences = self._predict_outputs(bundle, aligned)

        output = []
        for index, prediction in enumerate(raw_predictions):
            output.append(
                {
                    "row_index": index,
                    "prediction": pythonize(prediction),
                    "confidence": pythonize(confidences[index]) if confidences is not None else None,
                }
            )
        return output

    def _predict_outputs(self, bundle: dict[str, Any], frame):
        """Dispatch inference to LightAutoML or sklearn-compatible models."""
        model = bundle["pipeline"]
        task_type = bundle["task_type"]
        backend_name = bundle.get("backend_name", "sklearn-fallback")

        if backend_name == "lightautoml":
            data = model.predict(frame).data
            return self._decode_lightautoml_output(
                task_type=task_type,
                raw_output=data,
                class_mapping=bundle.get("class_mapping"),
            )

        raw_predictions = model.predict(frame)
        confidences = self._predict_confidence(model, frame)
        return raw_predictions, confidences

    @staticmethod
    def _predict_confidence(model, frame):
        """Return max class probability when the model exposes predict_proba."""
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(frame)
            return probabilities.max(axis=1)
        return None

    @staticmethod
    def _decode_lightautoml_output(task_type: str, raw_output, class_mapping: dict[str, int] | None):
        """Convert raw LightAutoML output into labels and confidences."""
        import numpy as np

        data = np.asarray(raw_output)
        if task_type == "regression":
            return data.reshape(-1), None

        reverse_mapping = None
        if class_mapping:
            reverse_mapping = {int(index): label for label, index in class_mapping.items()}

        if task_type == "binary":
            probabilities = data.reshape(-1)
            positive_class = reverse_mapping.get(1, 1) if reverse_mapping else 1
            negative_class = reverse_mapping.get(0, 0) if reverse_mapping else 0
            predictions = np.where(probabilities >= 0.5, positive_class, negative_class)
            confidences = np.maximum(probabilities, 1.0 - probabilities)
            return predictions, confidences

        indices = data.argmax(axis=1)
        if reverse_mapping:
            predictions = np.array([reverse_mapping.get(int(index), int(index)) for index in indices])
        else:
            predictions = indices
        confidences = data.max(axis=1)
        return predictions, confidences

    @staticmethod
    def _prepare_frame(bundle: dict[str, Any], frame):
        """Apply the stored preprocessor to a raw inference frame."""
        preprocessor = bundle.get("preprocessor")
        return preprocessor.transform(frame) if preprocessor is not None else frame

    @staticmethod
    def _resolve_missing_inputs(inference_frame, preprocessor, missing_features: list[str]) -> list[str]:
        """Map missing engineered model features back to user-provided source columns."""
        if not missing_features:
            return []

        source_columns = inference_frame.columns
        derived_feature_sources: dict[str, str] = {}

        if preprocessor is not None:
            for column in getattr(preprocessor, "datetime_columns", []):
                derived_feature_sources[f"{column}__ts"] = column
                derived_feature_sources[f"{column}__hour"] = column
                derived_feature_sources[f"{column}__dayofweek"] = column
                derived_feature_sources[f"{column}__month"] = column
            for column in getattr(preprocessor, "time_columns", []):
                derived_feature_sources[f"{column}__minutes"] = column

        missing_inputs: list[str] = []
        for feature in missing_features:
            source = derived_feature_sources.get(feature, feature)
            candidate = source if source not in source_columns else feature
            if candidate not in missing_inputs:
                missing_inputs.append(candidate)
        return missing_inputs

    @staticmethod
    def _build_compact_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
        """Reduce the full comparison payload to chart-ready data only."""
        return {
            "project_id": comparison["project_id"],
            "model_version": comparison["model_version"],
            "task_type": comparison["task_type"],
            "target": comparison["target"],
            "rows": comparison["rows"],
            "metrics": comparison.get("metrics") or {},
            "saved_at": comparison.get("saved_at"),
            "items": [
                {
                    "actual": item.get("actual"),
                    "prediction": item.get("prediction"),
                }
                for item in comparison.get("items", [])
            ],
        }

    @staticmethod
    def _build_model_summary(model: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
        """Normalize model metadata for the frontend."""
        forecasting = bundle.get("forecasting") or {}
        return {
            "version_id": model["version_id"],
            "project_id": model["project_id"],
            "dataset_version_id": model["dataset_version_id"],
            "task_type": model["task_type"],
            "target": model["target"],
            "primary_metric": model["primary_metric"],
            "metrics": model["metrics"],
            "status": model["status"],
            "feature_names": model["feature_names"],
            "created_at": model["created_at"],
            "forecasting": {
                "available": bool(forecasting.get("enabled")),
                "default_horizon_minutes": forecasting.get("default_horizon_minutes"),
                "base_frequency_minutes": forecasting.get("base_frequency_minutes"),
                "recent_history_rows": len(forecasting.get("recent_history", [])),
                "forecast_model": forecasting.get("forecasting_model"),
            },
        }
