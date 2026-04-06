from __future__ import annotations

from typing import Any
from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift

from backend.services.observability_service import ObservabilityService
from backend.services.registry_service import RegistryService
from backend.services.settings import get_settings


class MonitoringService:
    """Generate real Evidently drift reports for project datasets and log them to MLflow."""

    def __init__(self) -> None:
        self.registry_service = RegistryService()
        self.settings = get_settings()
        self.observability_service = ObservabilityService()

    def monitor_project(self, project_id: str, job_id: str) -> dict[str, Any]:
        """Compare the champion reference dataset with the latest dataset and build drift artifacts."""
        champion, _bundle = self.registry_service.get_champion_bundle(project_id)
        reference_dataset = self.registry_service.get_dataset_version(champion["dataset_version_id"])
        current_dataset = self.registry_service.get_latest_dataset_version(project_id)
        reference_frame = self.registry_service.load_dataset_version_frame(reference_dataset["version_id"])
        current_frame = self.registry_service.load_dataset_version_frame(current_dataset["version_id"])

        common_columns = [
            column for column in reference_frame.columns
            if column in current_frame.columns
        ]
        if not common_columns:
            raise ValueError("Monitoring requires at least one common column between reference and current datasets.")

        reference_prepared = reference_frame[common_columns].copy()
        current_prepared = current_frame[common_columns].copy()
        metrics = [DriftedColumnsCount(columns=common_columns)]
        selected_columns = common_columns[: min(8, len(common_columns))]
        metrics.extend(ValueDrift(column=column) for column in selected_columns)

        report = Report(metrics=metrics)
        snapshot = report.run(
            current_data=current_prepared,
            reference_data=reference_prepared,
        )

        artifact_dir = self.settings.monitoring_dir / project_id / job_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        html_path = artifact_dir / "drift_report.html"
        json_path = artifact_dir / "drift_report.json"
        snapshot.save_html(str(html_path))
        snapshot.save_json(str(json_path))

        report_dict = snapshot.dict()
        drift_metrics = self._extract_drift_metrics(report_dict)
        result = {
            "project_id": project_id,
            "job_id": job_id,
            "model_version": champion["version_id"],
            "reference_dataset_version_id": reference_dataset["version_id"],
            "current_dataset_version_id": current_dataset["version_id"],
            "reference_rows": int(len(reference_prepared)),
            "current_rows": int(len(current_prepared)),
            "columns_analyzed": common_columns,
            "metrics": drift_metrics,
            "artifacts": {
                "html_report": str(html_path),
                "json_report": str(json_path),
            },
        }

        result["mlflow"] = self.observability_service.log_monitoring_run(
            project_id=project_id,
            job_id=job_id,
            payload=result,
            artifact_dir=artifact_dir,
        )
        return result

    @staticmethod
    def _extract_drift_metrics(report_dict: dict[str, Any]) -> dict[str, Any]:
        """Summarize the most useful drift values from an Evidently snapshot."""
        metrics = report_dict.get("metrics") or []
        summary: dict[str, Any] = {
            "drifted_columns_count": 0.0,
            "drifted_columns_share": 0.0,
            "value_drift": {},
        }
        for item in metrics:
            metric_name = str(item.get("metric_name") or "")
            value = item.get("value")
            if metric_name.startswith("DriftedColumnsCount"):
                if isinstance(value, dict):
                    summary["drifted_columns_count"] = float(value.get("count", 0.0))
                    summary["drifted_columns_share"] = float(value.get("share", 0.0))
            elif metric_name.startswith("ValueDrift("):
                config = item.get("config") or {}
                column = config.get("column") or metric_name
                try:
                    summary["value_drift"][column] = float(value)
                except (TypeError, ValueError):
                    continue
        return summary
