from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import joblib
import pandas as pd

from backend.services.settings import get_settings


@dataclass(slots=True)
class StoredObjectRef:
    path: str
    storage_backend: str
    bucket: str | None = None
    object_key: str | None = None


class ObjectStorageService:
    """Store dataset snapshots and model bundles on filesystem or MinIO."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.backend = self.settings.object_storage_backend.casefold()
        self._client = None
        self._ensured_buckets: set[str] = set()

    def save_dataset_frame(self, project_id: str, version_id: str, frame: pd.DataFrame) -> StoredObjectRef:
        """Persist one dataset snapshot and return its storage reference."""
        if self.backend == "minio":
            object_key = f"{project_id}/{version_id}.csv"
            payload = frame.to_csv(index=False).encode("utf-8")
            self._put_object(
                bucket=self.settings.object_storage_datasets_bucket,
                object_key=object_key,
                payload=payload,
                content_type="text/csv",
            )
            return StoredObjectRef(
                path=f"s3://{self.settings.object_storage_datasets_bucket}/{object_key}",
                storage_backend="minio",
                bucket=self.settings.object_storage_datasets_bucket,
                object_key=object_key,
            )

        project_dir = self.settings.datasets_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = project_dir / f"{version_id}.csv"
        frame.to_csv(dataset_path, index=False)
        return StoredObjectRef(
            path=str(dataset_path),
            storage_backend="filesystem",
        )

    def load_dataset_frame(self, dataset_record: dict[str, Any]) -> pd.DataFrame:
        """Load a dataset snapshot using either MinIO or a legacy filesystem path."""
        if dataset_record.get("storage_backend") == "minio":
            payload = self._get_object(
                bucket=self._require_record_field(dataset_record, "bucket"),
                object_key=self._require_record_field(dataset_record, "object_key"),
            )
            return pd.read_csv(BytesIO(payload))

        return pd.read_csv(dataset_record["path"])

    def save_model_bundle(self, project_id: str, version_id: str, bundle: dict[str, Any]) -> StoredObjectRef:
        """Persist a model bundle and return its storage reference."""
        if self.backend == "minio":
            object_key = f"{project_id}/{version_id}/model.joblib"
            buffer = BytesIO()
            joblib.dump(bundle, buffer)
            self._put_object(
                bucket=self.settings.object_storage_artifacts_bucket,
                object_key=object_key,
                payload=buffer.getvalue(),
                content_type="application/octet-stream",
            )
            return StoredObjectRef(
                path=f"s3://{self.settings.object_storage_artifacts_bucket}/{object_key}",
                storage_backend="minio",
                bucket=self.settings.object_storage_artifacts_bucket,
                object_key=object_key,
            )

        artifact_dir = self.settings.artifacts_dir / project_id / version_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "model.joblib"
        joblib.dump(bundle, artifact_path)
        return StoredObjectRef(
            path=str(artifact_path),
            storage_backend="filesystem",
        )

    def load_model_bundle(self, model_record: dict[str, Any]) -> dict[str, Any]:
        """Load a trained model bundle from object storage or a legacy local path."""
        if model_record.get("storage_backend") == "minio":
            payload = self._get_object(
                bucket=self._require_record_field(model_record, "bucket"),
                object_key=self._require_record_field(model_record, "object_key"),
            )
            return joblib.load(BytesIO(payload))

        return joblib.load(model_record["artifact_path"])

    def delete_dataset(self, dataset_record: dict[str, Any]) -> None:
        """Delete one persisted dataset snapshot if it still exists."""
        self._delete_record_object(dataset_record, path_field="path")

    def delete_model(self, model_record: dict[str, Any]) -> None:
        """Delete one persisted model bundle if it still exists."""
        self._delete_record_object(model_record, path_field="artifact_path")

    def _delete_record_object(self, record: dict[str, Any], path_field: str) -> None:
        if record.get("storage_backend") == "minio":
            bucket = record.get("bucket")
            object_key = record.get("object_key")
            if bucket and object_key:
                self._delete_object(bucket=bucket, object_key=object_key)
            return

        path = record.get(path_field)
        if path:
            Path(path).unlink(missing_ok=True)

    def _put_object(self, bucket: str, object_key: str, payload: bytes, content_type: str) -> None:
        self._ensure_bucket(bucket)
        self.client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=payload,
            ContentType=content_type,
        )

    def _get_object(self, bucket: str, object_key: str) -> bytes:
        response = self.client.get_object(Bucket=bucket, Key=object_key)
        body = response["Body"]
        return body.read() if hasattr(body, "read") else body

    def _delete_object(self, bucket: str, object_key: str) -> None:
        self.client.delete_object(Bucket=bucket, Key=object_key)

    def _ensure_bucket(self, bucket: str) -> None:
        if bucket in self._ensured_buckets:
            return
        try:
            self.client.head_bucket(Bucket=bucket)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code not in {"404", "NoSuchBucket"}:
                raise
            self.client.create_bucket(Bucket=bucket)
        except AttributeError:
            try:
                self.client.create_bucket(Bucket=bucket)
            except Exception:
                pass
        self._ensured_buckets.add(bucket)

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=self._build_endpoint_url(),
                aws_access_key_id=self.settings.object_storage_access_key,
                aws_secret_access_key=self.settings.object_storage_secret_key,
                region_name=self.settings.object_storage_region,
                config=Config(s3={"addressing_style": "path"}),
            )
        return self._client

    def _build_endpoint_url(self) -> str | None:
        endpoint = (self.settings.object_storage_endpoint or "").strip()
        if not endpoint:
            return None
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        scheme = "https" if self.settings.object_storage_secure else "http"
        return f"{scheme}://{endpoint}"

    @staticmethod
    def _require_record_field(record: dict[str, Any], field_name: str) -> str:
        value = record.get(field_name)
        if not value:
            raise ValueError(f"Stored object metadata is missing required field '{field_name}'.")
        return str(value)
