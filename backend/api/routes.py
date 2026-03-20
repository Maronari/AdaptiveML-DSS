from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from backend.schemas.api import (
    DatasetValidationRequest,
    DecisionRequest,
    PredictionRequest,
    TrainRequest,
)
from backend.services.dataset_service import DatasetService
from backend.services.decision_service import DecisionService
from backend.services.explanation_service import ExplanationService
from backend.services.prediction_service import PredictionService
from backend.services.training_service import TrainingService


router = APIRouter()
dataset_service = DatasetService()
training_service = TrainingService()
prediction_service = PredictionService()
explanation_service = ExplanationService()
decision_service = DecisionService()


def _bad_request(exc: Exception) -> HTTPException:
    """Convert domain validation errors into HTTP 400 responses."""
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.get("/health", tags=["system"])
def healthcheck() -> dict[str, str]:
    """Return a minimal liveness response for health checks."""
    return {"status": "ok"}


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


@router.post("/training/run", tags=["training"])
def run_training(payload: TrainRequest) -> dict[str, Any]:
    """Train a model from inline records."""
    try:
        return training_service.train(
            project_id=payload.project_id,
            target=payload.target,
            records=payload.records,
            source_name=payload.source_name,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc


@router.post("/training/run/file", tags=["training"])
async def run_training_file(
    file: UploadFile = File(...),
    target: str = Form(...),
    project_id: str = Form("default"),
) -> dict[str, Any]:
    """Train a model from an uploaded dataset file."""
    try:
        return await training_service.train_from_upload(
            project_id=project_id,
            target=target,
            upload=file,
        )
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
