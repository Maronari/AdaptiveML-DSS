from dataclasses import asdict, dataclass
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dataclass into a plain dictionary."""
        return asdict(self)
