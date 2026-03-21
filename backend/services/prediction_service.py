from __future__ import annotations

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

        return {
            "project_id": project_id,
            "model_version": champion["version_id"],
            "task_type": champion["task_type"],
            "target": target,
            "rows": int(len(frame)),
            "columns": frame.columns.tolist(),
            "metrics": metrics,
            "items": comparison_items,
        }

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

    def predict_with_bundle(self, bundle: dict[str, Any], frame) -> list[dict[str, Any]]:
        """Predict from a preloaded model bundle and a raw DataFrame."""
        preprocessor = bundle.get("preprocessor")
        prepared = preprocessor.transform(frame) if preprocessor is not None else frame
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
