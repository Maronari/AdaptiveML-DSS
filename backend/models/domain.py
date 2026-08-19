from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DatasetVersion:
    """Metadata describing one persisted dataset snapshot."""
    version_id: str
    project_id: str
    source_name: str
    target: str
    path: str
    schema: dict[str, str]
    rows: int
    created_at: str
    name: str | None = None
    storage_backend: str = "filesystem"
    bucket: str | None = None
    object_key: str | None = None
    unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataclass into a plain dictionary."""
        return asdict(self)


@dataclass(slots=True)
class ProjectRecord:
    """Metadata describing a user-visible project entry."""
    project_id: str
    name: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataclass into a plain dictionary."""
        return asdict(self)


@dataclass(slots=True)
class ModelVersion:
    """Metadata describing one trained model artifact."""
    version_id: str
    project_id: str
    dataset_version_id: str
    artifact_path: str
    task_type: str
    target: str
    metrics: dict[str, float]
    primary_metric: str
    status: str
    feature_names: list[str]
    created_at: str
    name: str | None = None
    storage_backend: str = "filesystem"
    bucket: str | None = None
    object_key: str | None = None
    holdout_predictions: list[dict[str, Any]] = field(default_factory=list)
    training_artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataclass into a plain dictionary."""
        return asdict(self)


@dataclass(slots=True)
class BackgroundJob:
    """Metadata describing one asynchronous background job."""

    job_id: str
    job_type: str
    project_id: str
    status: str
    payload: dict[str, Any]
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    worker_name: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataclass into a plain dictionary."""
        return asdict(self)
