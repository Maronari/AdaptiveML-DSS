from typing import Any

from pydantic import BaseModel, Field


class TrainingOptionsRequest(BaseModel):
    task_type: str = Field(default="auto")
    backend: str = Field(default="auto")
    preset: str = Field(default="tabular")
    algos: list[str] = Field(default_factory=lambda: ["lgb", "linear_l2"])
    timeout_seconds: int = Field(default=30, ge=5)
    cpu_limit: int = Field(default=1, ge=1)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    cv_folds: int = Field(default=0, ge=0)
    enable_forecast: bool = Field(default=True)


class RetrainingOptionsRequest(BaseModel):
    history_scope: str = Field(default="all_history")
    minimum_relative_improvement: float = Field(default=0.03, ge=0.0)
    evaluation_fraction: float = Field(default=0.2, gt=0.0, lt=0.5)
    auto_activate: bool = Field(default=True)


class DatasetValidationRequest(BaseModel):
    project_id: str = Field(default="default")
    target: str
    records: list[dict[str, Any]]


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class TrainRequest(BaseModel):
    project_id: str = Field(default="default")
    target: str
    source_name: str = Field(default="inline")
    records: list[dict[str, Any]]
    training_options: TrainingOptionsRequest = Field(default_factory=TrainingOptionsRequest)
    name: str | None = Field(default=None, max_length=120)


class RetrainRequest(BaseModel):
    project_id: str = Field(default="default")
    target: str
    source_name: str = Field(default="inline")
    records: list[dict[str, Any]]
    training_options: TrainingOptionsRequest = Field(default_factory=TrainingOptionsRequest)
    retraining_options: RetrainingOptionsRequest = Field(default_factory=RetrainingOptionsRequest)
    name: str | None = Field(default=None, max_length=120)


class RenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class PredictionRequest(BaseModel):
    project_id: str = Field(default="default")
    records: list[dict[str, Any]]


class PredictionComparisonRequest(PredictionRequest):
    target: str


class ForecastRequest(BaseModel):
    project_id: str = Field(default="default")
    horizon_minutes: int = Field(default=30, ge=1)
    steps: int = Field(default=1, ge=1)


class DecisionRequest(PredictionRequest):
    pass


class DecisionForecastRequest(BaseModel):
    project_id: str = Field(default="default")
    version_id: str | None = None
    horizon_minutes: int = Field(default=30, ge=1)
    steps: int = Field(default=1, ge=1)
    point_count: int | None = Field(default=None, ge=1)
