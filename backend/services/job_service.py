from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.services.monitoring_service import MonitoringService
from backend.services.observability_service import ObservabilityService
from backend.services.registry_service import RegistryService
from backend.services.retraining_service import RetrainingService
from backend.services.settings import get_settings
from backend.services.training_service import TrainingService


JOB_STATUS_VALUES = ("queued", "running", "failed", "done")


class JobService:
    """Manage async background jobs and dispatch them to the right worker handler."""

    def __init__(self) -> None:
        self.registry_service = RegistryService()
        self.training_service = TrainingService()
        self.retraining_service = RetrainingService()
        self.monitoring_service = MonitoringService()
        self.observability_service = ObservabilityService()
        self.settings = get_settings()

    def enqueue_training_dataset(
        self,
        *,
        project_id: str,
        dataset_version_id: str,
        training_options: dict[str, Any],
    ) -> dict[str, Any]:
        job = self.registry_service.create_background_job(
            job_type="training_dataset",
            project_id=project_id,
            payload={
                "project_id": project_id,
                "dataset_version_id": dataset_version_id,
                "training_options": dict(training_options),
            },
        )
        return job.to_dict()

    def enqueue_retraining_dataset(
        self,
        *,
        project_id: str,
        dataset_version_id: str,
        training_options: dict[str, Any],
        retraining_options: dict[str, Any],
    ) -> dict[str, Any]:
        job = self.registry_service.create_background_job(
            job_type="retraining_dataset",
            project_id=project_id,
            payload={
                "project_id": project_id,
                "dataset_version_id": dataset_version_id,
                "training_options": dict(training_options),
                "retraining_options": dict(retraining_options),
            },
        )
        return job.to_dict()

    def enqueue_monitoring_project(self, *, project_id: str) -> dict[str, Any]:
        job = self.registry_service.create_background_job(
            job_type="monitoring_project",
            project_id=project_id,
            payload={"project_id": project_id},
        )
        return job.to_dict()

    def list_jobs(
        self,
        *,
        project_id: str | None = None,
        job_type: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        return {
            "items": self.registry_service.list_background_jobs(
                project_id=project_id,
                job_type=job_type,
                limit=limit,
            )
        }

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self.registry_service.get_background_job(job_id)

    def claim_next_job(self, *, job_types: list[str], worker_name: str) -> dict[str, Any] | None:
        return self.registry_service.claim_next_background_job(job_types=job_types, worker_name=worker_name)

    def append_log(self, job_id: str, source: str, message: str) -> dict[str, Any]:
        return self.registry_service.append_background_job_log(job_id=job_id, source=source, message=message)

    def process_job(self, job_id: str, worker_name: str) -> dict[str, Any]:
        """Execute one claimed job and finalize its status."""
        job = self.registry_service.get_background_job(job_id)
        if job["status"] != "running":
            raise ValueError(f"Background job '{job_id}' must be running before processing.")

        job_type = str(job["job_type"])
        payload = dict(job.get("payload") or {})
        artifact_dir = self.settings.jobs_dir / job_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        try:
            if job_type == "training_dataset":
                result = self._handle_training_dataset(payload, job_id=job_id, artifact_dir=artifact_dir)
            elif job_type == "retraining_dataset":
                result = self._handle_retraining_dataset(payload, job_id=job_id, artifact_dir=artifact_dir)
            elif job_type == "monitoring_project":
                result = self._handle_monitoring_project(payload, job_id=job_id)
            else:
                raise ValueError(f"Unsupported job type '{job_type}'.")

            return self.registry_service.finish_background_job(job_id=job_id, status="done", result=result)
        except Exception as exc:
            self.registry_service.append_background_job_log(job_id, worker_name, str(exc))
            return self.registry_service.finish_background_job(
                job_id=job_id,
                status="failed",
                error=str(exc),
            )

    def _handle_training_dataset(self, payload: dict[str, Any], *, job_id: str, artifact_dir: Path) -> dict[str, Any]:
        project_id = str(payload["project_id"])
        dataset_version_id = str(payload["dataset_version_id"])
        self.append_log(job_id, "training-worker", f"Запускаю обучение по датасету {dataset_version_id}.")
        result = self.training_service.train_from_dataset_version(
            project_id=project_id,
            dataset_version_id=dataset_version_id,
            training_options=payload.get("training_options") or {},
        )
        self.append_log(job_id, "training-worker", f"Обучение завершено: модель {result['model_version']['version_id']}.")
        result["mlflow"] = self.observability_service.log_training_run(
            project_id=project_id,
            job_id=job_id,
            run_kind="training",
            payload=result,
            artifact_dir=artifact_dir,
        )
        return result

    def _handle_retraining_dataset(self, payload: dict[str, Any], *, job_id: str, artifact_dir: Path) -> dict[str, Any]:
        project_id = str(payload["project_id"])
        dataset_version_id = str(payload["dataset_version_id"])
        self.append_log(job_id, "retraining-worker", f"Запускаю переобучение по датасету {dataset_version_id}.")
        result = self.retraining_service.retrain_from_dataset_version(
            project_id=project_id,
            dataset_version_id=dataset_version_id,
            training_options=payload.get("training_options") or {},
            retraining_options=payload.get("retraining_options") or {},
        )
        candidate_model = result["candidate_model_version"]["version_id"]
        self.append_log(job_id, "retraining-worker", f"Переобучение завершено: кандидат {candidate_model}.")
        result["mlflow"] = self.observability_service.log_training_run(
            project_id=project_id,
            job_id=job_id,
            run_kind="retraining",
            payload=result,
            artifact_dir=artifact_dir,
        )
        return result

    def _handle_monitoring_project(self, payload: dict[str, Any], *, job_id: str) -> dict[str, Any]:
        project_id = str(payload["project_id"])
        self.append_log(job_id, "monitoring-worker", f"Запускаю мониторинг drift для проекта {project_id}.")
        result = self.monitoring_service.monitor_project(project_id=project_id, job_id=job_id)
        self.append_log(job_id, "monitoring-worker", "Отчёт drift сформирован и залогирован в MLflow.")
        return result
