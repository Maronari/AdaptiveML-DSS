import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.schemas.api import (
    DatasetUnitUpdateRequest,
    DatasetValidationRequest,
    DecisionForecastRequest,
    DecisionRequest,
    ForecastRequest,
    ProjectCreateRequest,
    PredictionComparisonRequest,
    PredictionRequest,
    RenameRequest,
    RetrainRequest,
    TrainRequest,
)
from backend.services.dataset_service import DatasetService
from backend.services.dss_config_service import DssConfigService
from backend.services.decision_service import DecisionService
from backend.services.explanation_service import ExplanationService
from backend.services.job_service import JobService
from backend.services.prediction_service import PredictionService
from backend.services.registry_service import RegistryService
from backend.services.retraining_service import RetrainingService
from backend.services.training_service import TrainingService


router = APIRouter()
logger = logging.getLogger(__name__)
dataset_service = DatasetService()
training_service = TrainingService()
prediction_service = PredictionService()
explanation_service = ExplanationService()
decision_service = DecisionService()
retraining_service = RetrainingService()
registry_service = RegistryService()
job_service = JobService()
dss_config_service = DssConfigService()


def _bad_request(exc: Exception) -> HTTPException:
    """Convert domain validation errors into HTTP 400 responses."""
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/health", tags=["system"])
def healthcheck() -> dict[str, str]:
    """Return a minimal liveness response for health checks."""
    return {"status": "ok"}


@router.get("/projects", tags=["projects"])
def list_projects() -> dict[str, Any]:
    """Return existing projects enriched with training activity metadata."""
    return {"items": registry_service.list_projects()}


@router.get("/projects/{project_id}/models", tags=["projects"])
def list_project_models(project_id: str) -> dict[str, Any]:
    """Return stored model versions for one project."""
    try:
        return registry_service.list_model_versions(project_id)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/projects/{project_id}/models/{version_id}/activate", tags=["projects"])
def activate_project_model(project_id: str, version_id: str) -> dict[str, Any]:
    """Switch the active champion model for a project."""
    try:
        model = registry_service.activate_model_version(project_id=project_id, version_id=version_id)
        return model.to_dict()
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/projects", tags=["projects"], status_code=status.HTTP_201_CREATED)
def create_project(payload: ProjectCreateRequest) -> dict[str, Any]:
    """Create a project entry before any dataset upload or training run."""
    try:
        project = registry_service.create_project(payload.name)
        return project
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.delete("/projects/{project_id}", tags=["projects"])
def delete_project(project_id: str) -> dict[str, Any]:
    """Delete a project together with its datasets, models and stored artifacts."""
    try:
        return registry_service.delete_project(project_id)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/datasets/validate", tags=["datasets"])
def validate_dataset(payload: DatasetValidationRequest) -> dict[str, Any]:
    """Validate an inline dataset payload."""
    try:
        return dataset_service.validate_inline_dataset(
            project_id=payload.project_id,
            target=payload.target,
            records=payload.records,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/datasets/validate/file", tags=["datasets"])
async def validate_dataset_file(
    file: UploadFile = File(...),
    target: str = Form(...),
    project_id: str = Form("default"),
) -> dict[str, Any]:
    """Validate an uploaded dataset file."""
    try:
        return await dataset_service.validate_uploaded_dataset(
            project_id=project_id,
            target=target,
            upload=file,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/datasets/inspect/file", tags=["datasets"])
async def inspect_dataset_file(
    file: UploadFile = File(...),
    project_id: str = Form("default"),
) -> dict[str, Any]:
    """Inspect an uploaded dataset before the user selects the target."""
    try:
        return await dataset_service.inspect_uploaded_dataset(
            project_id=project_id,
            upload=file,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/datasets/inspect/files", tags=["datasets"])
async def inspect_dataset_files(
    files: list[UploadFile] = File(...),
    project_id: str = Form("default"),
) -> dict[str, Any]:
    """Inspect several uploaded files before the user selects the target."""
    try:
        return await dataset_service.inspect_uploaded_datasets(
            project_id=project_id,
            uploads=files,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/datasets/register/file", tags=["datasets"], status_code=status.HTTP_201_CREATED)
async def register_dataset_file(
    file: UploadFile = File(...),
    target: str = Form(...),
    project_id: str = Form("default"),
    name: str | None = Form(None),
    unit: str | None = Form(None),
) -> dict[str, Any]:
    """Persist an uploaded dataset version after target confirmation."""
    try:
        return await dataset_service.register_uploaded_dataset(
            project_id=project_id,
            target=target,
            upload=file,
            name=name,
            unit=unit,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/datasets/register/files", tags=["datasets"], status_code=status.HTTP_201_CREATED)
async def register_dataset_files(
    files: list[UploadFile] = File(...),
    target: str = Form(...),
    project_id: str = Form("default"),
    name: str | None = Form(None),
    unit: str | None = Form(None),
) -> dict[str, Any]:
    """Persist a dataset version assembled from several uploaded files."""
    try:
        return await dataset_service.register_uploaded_datasets(
            project_id=project_id,
            target=target,
            uploads=files,
            name=name,
            unit=unit,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.patch("/datasets/{version_id}", tags=["datasets"])
def rename_dataset(version_id: str, payload: RenameRequest) -> dict[str, Any]:
    """Rename an existing dataset version."""
    try:
        return registry_service.rename_dataset_version(version_id, payload.name)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.patch("/datasets/{version_id}/unit", tags=["datasets"])
def update_dataset_unit(version_id: str, payload: DatasetUnitUpdateRequest) -> dict[str, Any]:
    """Set or correct the measurement unit of an already-registered dataset version."""
    try:
        return registry_service.update_dataset_unit(version_id, payload.unit)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.get("/datasets/{version_id}", tags=["datasets"])
def get_registered_dataset(version_id: str) -> dict[str, Any]:
    """Return a saved dataset summary for the training page."""
    try:
        return dataset_service.get_registered_dataset_summary(version_id)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.get("/projects/{project_id}/datasets/latest", tags=["datasets"])
def get_latest_project_dataset(project_id: str) -> dict[str, Any]:
    """Return the newest saved dataset summary for one project."""
    try:
        return dataset_service.get_latest_project_dataset_summary(project_id)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/training/run", tags=["training"])
def run_training(payload: TrainRequest) -> dict[str, Any]:
    """Train a model from inline records."""
    try:
        logger.info(
            "Training request started: project_id=%s target=%s rows=%s source=%s",
            payload.project_id,
            payload.target,
            len(payload.records),
            payload.source_name,
        )
        return training_service.train(
            project_id=payload.project_id,
            target=payload.target,
            records=payload.records,
            source_name=payload.source_name,
            training_options=payload.training_options.model_dump(),
            name=payload.name,
            unit=payload.unit,
        )
    except ValueError as exc:
        logger.warning("Training request failed: project_id=%s error=%s", payload.project_id, exc)
        raise _bad_request(exc) from exc


@router.post("/training/run/file", tags=["training"])
async def run_training_file(
    file: UploadFile = File(...),
    target: str = Form(...),
    project_id: str = Form("default"),
    task_type: str = Form("auto"),
    backend: str = Form("auto"),
    preset: str = Form("tabular"),
    algos: str = Form("lgb,linear_l2"),
    timeout_seconds: int = Form(30),
    cpu_limit: int = Form(1),
    test_size: float = Form(0.2),
    cv_folds: int = Form(0),
    enable_forecast: bool = Form(True),
    name: str | None = Form(None),
) -> dict[str, Any]:
    """Train a model from an uploaded dataset file."""
    try:
        logger.info(
            "Training upload started: project_id=%s target=%s filename=%s",
            project_id,
            target,
            file.filename,
        )
        response = await training_service.train_from_upload(
            project_id=project_id,
            target=target,
            upload=file,
            training_options={
                "task_type": task_type,
                "backend": backend,
                "preset": preset,
                "algos": algos,
                "timeout_seconds": timeout_seconds,
                "cpu_limit": cpu_limit,
                "test_size": test_size,
                "cv_folds": cv_folds,
                "enable_forecast": enable_forecast,
            },
            name=name,
        )
        logger.info(
            "Training upload completed: project_id=%s model_version=%s backend=%s",
            project_id,
            response["model_version"]["version_id"],
            response["backend"],
        )
        return response
    except ValueError as exc:
        logger.warning("Training upload failed: project_id=%s error=%s", project_id, exc)
        raise _bad_request(exc) from exc


@router.post("/training/run/dataset", tags=["training"])
def run_training_dataset(
    dataset_version_id: str = Form(...),
    project_id: str = Form("default"),
    task_type: str = Form("auto"),
    backend: str = Form("auto"),
    preset: str = Form("tabular"),
    algos: str = Form("lgb,linear_l2"),
    timeout_seconds: int = Form(30),
    cpu_limit: int = Form(1),
    test_size: float = Form(0.2),
    cv_folds: int = Form(0),
    enable_forecast: bool = Form(True),
    name: str | None = Form(None),
) -> dict[str, Any]:
    """Train a model from a previously saved dataset version."""
    try:
        return training_service.train_from_dataset_version(
            project_id=project_id,
            dataset_version_id=dataset_version_id,
            training_options={
                "task_type": task_type,
                "backend": backend,
                "preset": preset,
                "algos": algos,
                "timeout_seconds": timeout_seconds,
                "cpu_limit": cpu_limit,
                "test_size": test_size,
                "cv_folds": cv_folds,
                "enable_forecast": enable_forecast,
            },
            name=name,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/jobs/training/dataset", tags=["jobs"])
def enqueue_training_dataset_job(
    dataset_version_id: str = Form(...),
    project_id: str = Form("default"),
    task_type: str = Form("auto"),
    backend: str = Form("auto"),
    preset: str = Form("tabular"),
    algos: str = Form("lgb,linear_l2"),
    timeout_seconds: int = Form(30),
    cpu_limit: int = Form(1),
    test_size: float = Form(0.2),
    cv_folds: int = Form(0),
    enable_forecast: bool = Form(True),
    name: str | None = Form(None),
) -> dict[str, Any]:
    """Queue a background training job for one stored dataset version."""
    try:
        return job_service.enqueue_training_dataset(
            project_id=project_id,
            dataset_version_id=dataset_version_id,
            training_options={
                "task_type": task_type,
                "backend": backend,
                "preset": preset,
                "algos": algos,
                "timeout_seconds": timeout_seconds,
                "cpu_limit": cpu_limit,
                "test_size": test_size,
                "cv_folds": cv_folds,
                "enable_forecast": enable_forecast,
            },
            name=name,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/retraining/run", tags=["training"])
def run_retraining(payload: RetrainRequest) -> dict[str, Any]:
    """Retrain a challenger model from inline labeled records."""
    try:
        return retraining_service.retrain(
            project_id=payload.project_id,
            target=payload.target,
            records=payload.records,
            source_name=payload.source_name,
            training_options=payload.training_options.model_dump(),
            retraining_options=payload.retraining_options.model_dump(),
            name=payload.name,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/retraining/inspect/file", tags=["training"])
async def inspect_retraining_file(
    file: UploadFile = File(...),
    project_id: str = Form("default"),
) -> dict[str, Any]:
    """Inspect one uploaded dataset against the champion model before retraining."""
    try:
        return await retraining_service.inspect_retraining_upload(
            project_id=project_id,
            upload=file,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/retraining/inspect/files", tags=["training"])
async def inspect_retraining_files(
    files: list[UploadFile] = File(...),
    project_id: str = Form("default"),
) -> dict[str, Any]:
    """Inspect several uploaded datasets against the champion model before retraining."""
    try:
        return await retraining_service.inspect_retraining_uploads(
            project_id=project_id,
            uploads=files,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/retraining/run/file", tags=["training"])
async def run_retraining_file(
    file: UploadFile = File(...),
    target: str = Form(...),
    project_id: str = Form("default"),
    task_type: str = Form("auto"),
    backend: str = Form("auto"),
    preset: str = Form("tabular"),
    algos: str = Form("lgb,linear_l2"),
    timeout_seconds: int = Form(30),
    cpu_limit: int = Form(1),
    test_size: float = Form(0.2),
    cv_folds: int = Form(0),
    enable_forecast: bool = Form(True),
    history_scope: str = Form("all_history"),
    minimum_relative_improvement: float = Form(0.03),
    evaluation_fraction: float = Form(0.2),
    auto_activate: bool = Form(True),
    name: str | None = Form(None),
) -> dict[str, Any]:
    """Retrain a challenger model from an uploaded labeled dataset file."""
    try:
        return await retraining_service.retrain_from_upload(
            project_id=project_id,
            target=target,
            upload=file,
            training_options={
                "task_type": task_type,
                "backend": backend,
                "preset": preset,
                "algos": algos,
                "timeout_seconds": timeout_seconds,
                "cpu_limit": cpu_limit,
                "test_size": test_size,
                "cv_folds": cv_folds,
                "enable_forecast": enable_forecast,
            },
            retraining_options={
                "history_scope": history_scope,
                "minimum_relative_improvement": minimum_relative_improvement,
                "evaluation_fraction": evaluation_fraction,
                "auto_activate": auto_activate,
            },
            name=name,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/retraining/run/dataset", tags=["training"])
def run_retraining_dataset(
    dataset_version_id: str = Form(...),
    project_id: str = Form("default"),
    task_type: str = Form("auto"),
    backend: str = Form("auto"),
    preset: str = Form("tabular"),
    algos: str = Form("lgb,linear_l2"),
    timeout_seconds: int = Form(30),
    cpu_limit: int = Form(1),
    test_size: float = Form(0.2),
    cv_folds: int = Form(0),
    enable_forecast: bool = Form(True),
    history_scope: str = Form("all_history"),
    minimum_relative_improvement: float = Form(0.03),
    evaluation_fraction: float = Form(0.2),
    auto_activate: bool = Form(True),
    name: str | None = Form(None),
) -> dict[str, Any]:
    """Retrain a challenger model from a previously saved dataset version."""
    try:
        return retraining_service.retrain_from_dataset_version(
            project_id=project_id,
            dataset_version_id=dataset_version_id,
            training_options={
                "task_type": task_type,
                "backend": backend,
                "preset": preset,
                "algos": algos,
                "timeout_seconds": timeout_seconds,
                "cpu_limit": cpu_limit,
                "test_size": test_size,
                "cv_folds": cv_folds,
                "enable_forecast": enable_forecast,
            },
            retraining_options={
                "history_scope": history_scope,
                "minimum_relative_improvement": minimum_relative_improvement,
                "evaluation_fraction": evaluation_fraction,
                "auto_activate": auto_activate,
            },
            name=name,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/jobs/retraining/dataset", tags=["jobs"])
def enqueue_retraining_dataset_job(
    dataset_version_id: str = Form(...),
    project_id: str = Form("default"),
    task_type: str = Form("auto"),
    backend: str = Form("auto"),
    preset: str = Form("tabular"),
    algos: str = Form("lgb,linear_l2"),
    timeout_seconds: int = Form(30),
    cpu_limit: int = Form(1),
    test_size: float = Form(0.2),
    cv_folds: int = Form(0),
    enable_forecast: bool = Form(True),
    history_scope: str = Form("all_history"),
    minimum_relative_improvement: float = Form(0.03),
    evaluation_fraction: float = Form(0.2),
    auto_activate: bool = Form(True),
    name: str | None = Form(None),
) -> dict[str, Any]:
    """Queue a background retraining job for one stored dataset version."""
    try:
        return job_service.enqueue_retraining_dataset(
            project_id=project_id,
            dataset_version_id=dataset_version_id,
            training_options={
                "task_type": task_type,
                "backend": backend,
                "preset": preset,
                "algos": algos,
                "timeout_seconds": timeout_seconds,
                "cpu_limit": cpu_limit,
                "test_size": test_size,
                "cv_folds": cv_folds,
                "enable_forecast": enable_forecast,
            },
            retraining_options={
                "history_scope": history_scope,
                "minimum_relative_improvement": minimum_relative_improvement,
                "evaluation_fraction": evaluation_fraction,
                "auto_activate": auto_activate,
            },
            name=name,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/jobs/monitoring/project", tags=["jobs"])
def enqueue_monitoring_project_job(project_id: str = Form("default")) -> dict[str, Any]:
    """Queue a background monitoring job for the selected project."""
    try:
        return job_service.enqueue_monitoring_project(project_id=project_id)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.get("/jobs", tags=["jobs"])
def list_jobs(
    project_id: str | None = None,
    job_type: str | None = None,
    limit: int | None = 20,
) -> dict[str, Any]:
    """List recent background jobs, optionally filtered by project or type."""
    try:
        return job_service.list_jobs(project_id=project_id, job_type=job_type, limit=limit)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.get("/jobs/{job_id}", tags=["jobs"])
def get_job(job_id: str) -> dict[str, Any]:
    """Return one background job with logs and result payload."""
    try:
        return job_service.get_job(job_id)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/predictions/run", tags=["prediction"])
def run_prediction(payload: PredictionRequest) -> dict[str, Any]:
    """Run prediction for inline records."""
    try:
        return prediction_service.predict(
            project_id=payload.project_id,
            records=payload.records,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/predictions/compare", tags=["prediction"])
def compare_prediction(payload: PredictionComparisonRequest) -> dict[str, Any]:
    """Compare actual target values with champion-model predictions."""
    try:
        return prediction_service.compare_with_actual(
            project_id=payload.project_id,
            target=payload.target,
            records=payload.records,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/predictions/run/file", tags=["prediction"])
async def run_prediction_file(
    file: UploadFile = File(...),
    project_id: str = Form("default"),
) -> dict[str, Any]:
    """Run prediction for an uploaded dataset file."""
    try:
        return await prediction_service.predict_from_upload(
            project_id=project_id,
            upload=file,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/predictions/compare/file", tags=["prediction"])
async def compare_prediction_file(
    file: UploadFile = File(...),
    target: str = Form(...),
    project_id: str = Form("default"),
) -> dict[str, Any]:
    """Compare actual target values with champion-model predictions for an uploaded file."""
    try:
        return await prediction_service.compare_with_actual_from_upload(
            project_id=project_id,
            target=target,
            upload=file,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/predictions/compare/file/schema", tags=["prediction"])
async def inspect_compare_prediction_file_schema(
    file: UploadFile = File(...),
    target: str = Form(...),
    project_id: str = Form("default"),
) -> dict[str, Any]:
    """Inspect whether an uploaded labeled file matches the champion model schema."""
    try:
        return await prediction_service.inspect_comparison_upload(
            project_id=project_id,
            target=target,
            upload=file,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.get("/predictions/compare/latest", tags=["prediction"])
def get_latest_prediction_comparison() -> dict[str, Any]:
    """Return the latest saved compact comparison payload."""
    try:
        return prediction_service.get_latest_comparison()
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.get("/models/latest", tags=["models"])
def get_latest_model(project_id: str = "default") -> dict[str, Any]:
    """Return metadata for the newest stored model of the selected project."""
    try:
        return prediction_service.get_latest_model_summary(project_id=project_id)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.get("/models/{version_id}", tags=["models"])
def get_model(version_id: str) -> dict[str, Any]:
    """Return metadata for one stored model version."""
    try:
        return prediction_service.get_model_summary(version_id=version_id)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.patch("/models/{version_id}", tags=["models"])
def rename_model(version_id: str, payload: RenameRequest) -> dict[str, Any]:
    """Rename an existing model version."""
    try:
        return registry_service.rename_model_version(version_id, payload.name)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.get("/models/{version_id}/forecast", tags=["models"])
def get_model_forecast(
    version_id: str,
    horizon_minutes: int | None = None,
    steps: int | None = None,
) -> dict[str, Any]:
    """Return forecast data for a specific stored model version."""
    try:
        return prediction_service.forecast_model_version(
            version_id=version_id,
            horizon_minutes=horizon_minutes,
            steps=steps,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/forecast/run", tags=["forecast"])
def run_forecast(payload: ForecastRequest) -> dict[str, Any]:
    """Forecast future target values from the current champion model."""
    try:
        return prediction_service.forecast(
            project_id=payload.project_id,
            horizon_minutes=payload.horizon_minutes,
            steps=payload.steps,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.get("/forecast/{run_id}/comparison", tags=["forecast"])
def get_forecast_comparison(run_id: str) -> dict[str, Any]:
    """Compare a saved forecast run against actual data registered afterwards."""
    try:
        return prediction_service.get_forecast_comparison(run_id=run_id)
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/explanations/run", tags=["explanation"])
def run_explanation(payload: PredictionRequest) -> dict[str, Any]:
    """Build explanations for inline inference records."""
    try:
        return explanation_service.explain(
            project_id=payload.project_id,
            records=payload.records,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/decision/run", tags=["decision"])
def run_decision(payload: DecisionRequest) -> dict[str, Any]:
    """Run the full DSS path for inline inference records."""
    try:
        return decision_service.evaluate(
            project_id=payload.project_id,
            records=payload.records,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/decision/forecast", tags=["decision"])
def run_forecast_decision(payload: DecisionForecastRequest) -> dict[str, Any]:
    """Run DSS recommendations on future forecast points."""
    try:
        if payload.version_id:
            return decision_service.evaluate_forecast_model_version(
                version_id=payload.version_id,
                horizon_minutes=payload.horizon_minutes,
                steps=payload.steps,
                point_count=payload.point_count,
            )

        return decision_service.evaluate_forecast(
            project_id=payload.project_id,
            horizon_minutes=payload.horizon_minutes,
            steps=payload.steps,
            point_count=payload.point_count,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.get("/dss/rulesets", tags=["decision"])
def get_dss_rule_sets() -> dict[str, Any]:
    """Return the current DSS rule-engine configuration."""
    try:
        return dss_config_service.get_rule_engine_config()
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.put("/dss/rulesets", tags=["decision"])
def save_dss_rule_sets(payload: dict[str, Any]) -> dict[str, Any]:
    """Persist the DSS rule-engine configuration override."""
    try:
        return dss_config_service.save_rule_engine_config(payload)
    except ValueError as exc:
        raise _bad_request(exc) from exc
