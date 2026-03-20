from typing import Any

from pydantic import BaseModel, Field


class DatasetValidationRequest(BaseModel):
    project_id: str = Field(default="default")
    target: str
    records: list[dict[str, Any]]


class TrainRequest(BaseModel):
    project_id: str = Field(default="default")
    target: str
    source_name: str = Field(default="inline")
    records: list[dict[str, Any]]


class PredictionRequest(BaseModel):
    project_id: str = Field(default="default")
    records: list[dict[str, Any]]


class DecisionRequest(PredictionRequest):
    pass
