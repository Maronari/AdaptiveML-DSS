from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from fastapi import UploadFile

from automl.evaluation.metrics import evaluate_predictions, primary_metric_name
from automl.lightautoml_backend.adapter import TabularAutoMLAdapter
from automl.training.forecasting import build_timestamp_series, infer_timestamp_descriptor
from backend.services.dataset_service import DatasetService
from backend.services.prediction_service import PredictionService
from backend.services.registry_service import RegistryService
from backend.utils.io import dataframe_from_records


RETRAINING_WINDOWS = {
    "all_history": None,
    "last_30_days": 30,
    "last_60_days": 60,
    "last_90_days": 90,
}


@dataclass(slots=True)
class RetrainingOptions:
    history_scope: str = "all_history"
    minimum_relative_improvement: float = 0.03
    evaluation_fraction: float = 0.2
    auto_activate: bool = True

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None = None) -> "RetrainingOptions":
        payload = payload or {}
        history_scope = str(payload.get("history_scope", "all_history")).strip().lower() or "all_history"
        minimum_relative_improvement = float(payload.get("minimum_relative_improvement", 0.03))
        evaluation_fraction = float(payload.get("evaluation_fraction", 0.2))
        auto_activate = bool(payload.get("auto_activate", True))

        if history_scope not in RETRAINING_WINDOWS:
            raise ValueError(
                "Unsupported history_scope. Choose one of: "
                + ", ".join(RETRAINING_WINDOWS.keys())
                + "."
            )
        if minimum_relative_improvement < 0:
            raise ValueError("minimum_relative_improvement must be greater than or equal to 0.")
        if not 0.0 < evaluation_fraction < 0.5:
            raise ValueError("evaluation_fraction must be greater than 0 and less than 0.5.")

        return cls(
            history_scope=history_scope,
            minimum_relative_improvement=minimum_relative_improvement,
            evaluation_fraction=evaluation_fraction,
            auto_activate=auto_activate,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_scope": self.history_scope,
            "minimum_relative_improvement": self.minimum_relative_improvement,
            "evaluation_fraction": self.evaluation_fraction,
            "auto_activate": self.auto_activate,
        }


class RetrainingService:
    def __init__(self) -> None:
        self.dataset_service = DatasetService()
        self.registry_service = RegistryService()
        self.adapter = TabularAutoMLAdapter()
        self.prediction_service = PredictionService()

    def retrain(
        self,
        project_id: str,
        target: str,
        records: list[dict[str, Any]],
        source_name: str = "inline",
        training_options: dict[str, Any] | None = None,
        retraining_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Train a challenger model on historical+new data and compare it with the current champion."""
        current_model, current_bundle = self.registry_service.get_champion_bundle(project_id)
        if current_model["target"] != target:
            raise ValueError(
                f"Target '{target}' does not match the current champion target '{current_model['target']}'."
            )

        new_frame = dataframe_from_records(records)
        self.dataset_service._validate_training_frame(new_frame, target)
        if len(new_frame) < 6:
            raise ValueError("Retraining batch must contain at least 6 rows.")

        historical_frame = self.registry_service.load_dataset_version_frame(current_model["dataset_version_id"])
        retraining = RetrainingOptions.from_payload(retraining_options)

        candidate_frame, evaluation_frame, selection_summary = self._build_candidate_and_evaluation_frames(
            historical_frame=historical_frame,
            new_frame=new_frame,
            target=target,
            options=retraining,
        )

        training_result = self.adapter.train(
            frame=candidate_frame,
            target=target,
            training_options=training_options,
        )
        if training_result.task_type != current_model["task_type"]:
            raise ValueError(
                "Retraining changed the inferred task type. "
                f"Current champion is '{current_model['task_type']}', candidate is '{training_result.task_type}'."
            )

        dataset_version = self.registry_service.create_dataset_version(
            project_id=project_id,
            source_name=f"retraining:{retraining.history_scope}:{source_name}",
            target=target,
            frame=candidate_frame,
        )
        candidate_model = self.registry_service.register_model_version(
            project_id=project_id,
            dataset_version_id=dataset_version.version_id,
            target=target,
            task_type=training_result.task_type,
            feature_names=training_result.feature_names,
            metrics=training_result.metrics,
            bundle=training_result.bundle,
            promotion_mode="candidate",
        )

        current_eval_metrics = self._evaluate_bundle(
            bundle=current_bundle,
            task_type=current_model["task_type"],
            frame=evaluation_frame,
            target=target,
        )
        candidate_eval_metrics = self._evaluate_bundle(
            bundle=training_result.bundle,
            task_type=training_result.task_type,
            frame=evaluation_frame,
            target=target,
        )
        improvement = self._calculate_improvement(
            task_type=training_result.task_type,
            current_metrics=current_eval_metrics,
            candidate_metrics=candidate_eval_metrics,
            minimum_relative_improvement=retraining.minimum_relative_improvement,
        )

        activated = False
        activation_reason = "Candidate did not beat the current champion on the evaluation slice."
        candidate_record = candidate_model.to_dict()
        if improvement["meets_threshold"] and retraining.auto_activate:
            activated_model = self.registry_service.activate_model_version(
                project_id=project_id,
                version_id=candidate_model.version_id,
            )
            candidate_record = activated_model.to_dict()
            activated = True
            activation_reason = (
                f"Candidate beat the current champion by {improvement['relative_gain_percent']:.2f}% "
                f"on {improvement['primary_metric']}."
            )
        elif improvement["is_better"]:
            activation_reason = (
                f"Candidate improved {improvement['primary_metric']}, but the gain "
                f"({improvement['relative_gain_percent']:.2f}%) did not reach the "
                f"required threshold ({improvement['minimum_relative_improvement_percent']:.2f}%)."
            )

        return {
            "project_id": project_id,
            "target": target,
            "current_model_version": current_model["version_id"],
            "candidate_model_version": candidate_record,
            "candidate_backend": training_result.backend_name,
            "task_type": training_result.task_type,
            "training_options": training_result.training_options,
            "retraining_options": retraining.to_dict(),
            "dataset_version": dataset_version.to_dict(),
            "selection_summary": selection_summary,
            "evaluation": {
                "rows": int(len(evaluation_frame)),
                "current_model_metrics": current_eval_metrics,
                "candidate_model_metrics": candidate_eval_metrics,
                "profit": improvement,
            },
            "candidate_training_metrics": training_result.metrics,
            "forecasting": {
                "available": training_result.bundle.get("forecasting") is not None,
            },
            "activated": activated,
            "activation_reason": activation_reason,
            "warnings": training_result.warnings,
        }

    async def retrain_from_upload(
        self,
        project_id: str,
        target: str,
        upload: UploadFile,
        training_options: dict[str, Any] | None = None,
        retraining_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrain from an uploaded labeled file."""
        frame = await self.dataset_service.read_uploaded_tabular(upload)
        records = frame.to_dict(orient="records")
        source_name = upload.filename or "uploaded.csv"
        return self.retrain(
            project_id=project_id,
            target=target,
            records=records,
            source_name=source_name,
            training_options=training_options,
            retraining_options=retraining_options,
        )

    def _build_candidate_and_evaluation_frames(
        self,
        historical_frame: pd.DataFrame,
        new_frame: pd.DataFrame,
        target: str,
        options: RetrainingOptions,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Return candidate training data and an honest evaluation slice from the new batch."""
        ordered_new_frame, new_timestamps = self._sort_temporal_frame_if_possible(new_frame, target)
        evaluation_rows = max(2, int(round(len(ordered_new_frame) * options.evaluation_fraction)))
        if len(ordered_new_frame) - evaluation_rows < 1:
            raise ValueError("Retraining batch is too small after reserving the evaluation slice.")

        new_training_frame = ordered_new_frame.iloc[:-evaluation_rows].copy()
        evaluation_frame = ordered_new_frame.iloc[-evaluation_rows:].copy()
        history_frame = self._select_history_frame(
            historical_frame=historical_frame,
            target=target,
            history_scope=options.history_scope,
        )

        candidate_frame = pd.concat([history_frame, new_training_frame], ignore_index=True)
        candidate_frame = candidate_frame.drop_duplicates(keep="last").reset_index(drop=True)
        self.dataset_service._validate_training_frame(candidate_frame, target)

        summary = {
            "history_scope": options.history_scope,
            "historical_rows_loaded": int(len(historical_frame)),
            "historical_rows_selected": int(len(history_frame)),
            "new_rows_received": int(len(new_frame)),
            "new_rows_used_for_training": int(len(new_training_frame)),
            "new_rows_reserved_for_evaluation": int(len(evaluation_frame)),
            "candidate_training_rows": int(len(candidate_frame)),
        }
        if new_timestamps is not None and not new_timestamps.empty:
            summary["evaluation_start"] = evaluation_frame["__retraining_timestamp__"].iloc[0].isoformat()
            summary["evaluation_end"] = evaluation_frame["__retraining_timestamp__"].iloc[-1].isoformat()
            summary["history_window_days"] = RETRAINING_WINDOWS[options.history_scope]

        candidate_frame = candidate_frame.drop(columns=["__retraining_timestamp__"], errors="ignore")
        evaluation_frame = evaluation_frame.drop(columns=["__retraining_timestamp__"], errors="ignore")
        return candidate_frame, evaluation_frame, summary

    def _select_history_frame(
        self,
        historical_frame: pd.DataFrame,
        target: str,
        history_scope: str,
    ) -> pd.DataFrame:
        """Choose the historical subset according to the retraining window policy."""
        if history_scope == "all_history":
            return historical_frame.copy()

        window_days = RETRAINING_WINDOWS[history_scope]
        assert window_days is not None

        ordered_history, history_timestamps = self._sort_temporal_frame_if_possible(historical_frame, target)
        if history_timestamps is None or history_timestamps.empty:
            raise ValueError(f"History scope '{history_scope}' requires a temporal dataset with a detectable timestamp.")

        anchor_timestamp = history_timestamps.iloc[-1]
        lower_bound = anchor_timestamp - pd.Timedelta(days=window_days)
        selected = ordered_history.loc[ordered_history["__retraining_timestamp__"] >= lower_bound].copy()
        if selected.empty:
            raise ValueError(
                f"No historical rows were found inside the selected {window_days}-day retraining window."
            )
        return selected

    @staticmethod
    def _sort_temporal_frame_if_possible(
        frame: pd.DataFrame,
        target: str,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Sort the frame by timestamp when a temporal descriptor is available."""
        descriptor = infer_timestamp_descriptor(frame=frame, target=target)
        if descriptor is None:
            return frame.reset_index(drop=True).copy(), None

        timestamps = build_timestamp_series(frame=frame, descriptor=descriptor)
        ordered = frame.copy()
        ordered["__retraining_timestamp__"] = timestamps
        ordered = ordered.dropna(subset=["__retraining_timestamp__"]).sort_values("__retraining_timestamp__")
        ordered = ordered.reset_index(drop=True)
        if ordered.empty:
            raise ValueError("Temporal retraining could not parse usable timestamps from the dataset.")
        return ordered, ordered["__retraining_timestamp__"]

    def _evaluate_bundle(
        self,
        bundle: dict[str, Any],
        task_type: str,
        frame: pd.DataFrame,
        target: str,
    ) -> dict[str, float]:
        """Evaluate a bundle on a labeled holdout frame."""
        actual = frame[target]
        inference_frame = frame.drop(columns=[target])
        predictions = self.prediction_service.predict_with_bundle(bundle=bundle, frame=inference_frame)
        predicted_values = [item["prediction"] for item in predictions]
        return evaluate_predictions(task_type=task_type, y_true=actual, y_pred=predicted_values)

    @staticmethod
    def _calculate_improvement(
        task_type: str,
        current_metrics: dict[str, float],
        candidate_metrics: dict[str, float],
        minimum_relative_improvement: float,
    ) -> dict[str, Any]:
        """Compute absolute and relative gain for the primary metric."""
        metric_name = primary_metric_name(task_type)
        current_value = float(current_metrics[metric_name])
        candidate_value = float(candidate_metrics[metric_name])
        lower_is_better = metric_name == "rmse"

        if lower_is_better:
            absolute_gain = current_value - candidate_value
        else:
            absolute_gain = candidate_value - current_value

        relative_gain = 0.0
        if current_value != 0:
            relative_gain = absolute_gain / abs(current_value)
        elif absolute_gain > 0:
            relative_gain = 1.0

        return {
            "primary_metric": metric_name,
            "current_value": round(current_value, 6),
            "candidate_value": round(candidate_value, 6),
            "absolute_gain": round(absolute_gain, 6),
            "relative_gain": round(relative_gain, 6),
            "relative_gain_percent": round(relative_gain * 100.0, 4),
            "minimum_relative_improvement": round(minimum_relative_improvement, 6),
            "minimum_relative_improvement_percent": round(minimum_relative_improvement * 100.0, 4),
            "lower_is_better": lower_is_better,
            "is_better": absolute_gain > 0,
            "meets_threshold": absolute_gain > 0 and relative_gain >= minimum_relative_improvement,
        }
