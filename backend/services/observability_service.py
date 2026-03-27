from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow

from backend.services.settings import get_settings


class ObservabilityService:
    """Persist MLflow runs and lightweight artifact bundles for background jobs."""

    def __init__(self) -> None:
        self.settings = get_settings()
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_dir.resolve().as_uri())

    def log_training_run(
        self,
        *,
        project_id: str,
        job_id: str,
        run_kind: str,
        payload: dict[str, Any],
        artifact_dir: Path,
    ) -> dict[str, Any]:
        """Log one training/retraining payload to MLflow."""
        experiment_id = mlflow.set_experiment(f"AdaptiveML DSS / {project_id}").experiment_id
        run_name = f"{run_kind}:{job_id}"

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "project_id": project_id,
                    "job_id": job_id,
                    "run_kind": run_kind,
                    "task_type": str(payload.get("task_type") or ""),
                    "model_version": str(
                        payload.get("model_version", {}).get("version_id")
                        or payload.get("candidate_model_version", {}).get("version_id")
                        or ""
                    ),
                }
            )
            self._log_nested_params("payload", payload)
            self._log_numeric_metrics(self._extract_metrics(payload))

            artifact_dir.mkdir(parents=True, exist_ok=True)
            summary_path = artifact_dir / "summary.json"
            summary_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(summary_path), artifact_path="summary")
            return {
                "tracking_uri": mlflow.get_tracking_uri(),
                "experiment_id": experiment_id,
                "run_id": run.info.run_id,
                "run_name": run_name,
            }

    def log_monitoring_run(
        self,
        *,
        project_id: str,
        job_id: str,
        payload: dict[str, Any],
        artifact_dir: Path,
    ) -> dict[str, Any]:
        """Log one monitoring payload and its report artifacts to MLflow."""
        experiment_id = mlflow.set_experiment(f"AdaptiveML DSS / {project_id}").experiment_id
        run_name = f"monitoring:{job_id}"

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "project_id": project_id,
                    "job_id": job_id,
                    "run_kind": "monitoring",
                    "reference_dataset_version": str(payload.get("reference_dataset_version_id") or ""),
                    "current_dataset_version": str(payload.get("current_dataset_version_id") or ""),
                }
            )
            self._log_nested_params("monitoring", payload)
            metrics = payload.get("metrics") or {}
            self._log_numeric_metrics(metrics)

            if artifact_dir.exists():
                mlflow.log_artifacts(str(artifact_dir), artifact_path="monitoring")

            return {
                "tracking_uri": mlflow.get_tracking_uri(),
                "experiment_id": experiment_id,
                "run_id": run.info.run_id,
                "run_name": run_name,
            }

    @staticmethod
    def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
        """Flatten the main metric sections used by training and retraining payloads."""
        metrics: dict[str, float] = {}
        for prefix, section in (
            ("training", payload.get("metrics") or {}),
            ("candidate_training", payload.get("candidate_training_metrics") or {}),
            ("current_eval", payload.get("evaluation", {}).get("current_model_metrics") or {}),
            ("candidate_eval", payload.get("evaluation", {}).get("candidate_model_metrics") or {}),
        ):
            for key, value in section.items():
                if isinstance(value, (int, float)):
                    metrics[f"{prefix}.{key}"] = float(value)

        profit = payload.get("evaluation", {}).get("profit") or {}
        for key, value in profit.items():
            if isinstance(value, (int, float)):
                metrics[f"profit.{key}"] = float(value)
        return metrics

    @staticmethod
    def _log_numeric_metrics(metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))

    def _log_nested_params(self, prefix: str, payload: dict[str, Any]) -> None:
        """Log scalar params recursively while keeping MLflow key length reasonable."""
        for key, value in self._flatten(prefix, payload).items():
            if value is None:
                continue
            mlflow.log_param(key[:240], str(value)[:500])

    def _flatten(self, prefix: str, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            output: dict[str, Any] = {}
            for key, nested in value.items():
                nested_prefix = f"{prefix}.{key}"
                output.update(self._flatten(nested_prefix, nested))
            return output
        if isinstance(value, list):
            if all(not isinstance(item, (dict, list)) for item in value):
                return {prefix: ",".join(str(item) for item in value)}
            return {prefix: json.dumps(value, ensure_ascii=True)}
        return {prefix: value}
