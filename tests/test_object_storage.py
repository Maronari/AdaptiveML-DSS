from io import BytesIO

import pandas as pd
import pytest

from backend.services.object_storage_service import ObjectStorageService
from backend.services.registry_service import RegistryService
from backend.services.settings import get_settings


class _FakeS3Client:
    def __init__(self) -> None:
        self.buckets: dict[str, dict[str, bytes]] = {}

    def head_bucket(self, Bucket: str) -> None:
        if Bucket not in self.buckets:
            from botocore.exceptions import ClientError

            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}},
                "HeadBucket",
            )

    def create_bucket(self, Bucket: str) -> None:
        self.buckets.setdefault(Bucket, {})

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        self.buckets.setdefault(Bucket, {})[Key] = Body

    def get_object(self, Bucket: str, Key: str) -> dict[str, BytesIO]:
        return {"Body": BytesIO(self.buckets[Bucket][Key])}

    def delete_object(self, Bucket: str, Key: str) -> None:
        self.buckets.setdefault(Bucket, {}).pop(Key, None)


@pytest.fixture()
def minio_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTIVEML_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("ADAPTIVEML_OBJECT_STORAGE_BACKEND", "minio")
    monkeypatch.setenv("ADAPTIVEML_OBJECT_STORAGE_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("ADAPTIVEML_OBJECT_STORAGE_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("ADAPTIVEML_OBJECT_STORAGE_SECRET_KEY", "minioadmin")
    monkeypatch.setenv("ADAPTIVEML_OBJECT_STORAGE_DATASETS_BUCKET", "test-datasets")
    monkeypatch.setenv("ADAPTIVEML_OBJECT_STORAGE_ARTIFACTS_BUCKET", "test-artifacts")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_object_storage_service_roundtrip_with_minio_backend(minio_env):
    service = ObjectStorageService()
    service._client = _FakeS3Client()
    frame = pd.DataFrame(
        [
            {"feature": 1, "target": 10.5},
            {"feature": 2, "target": 12.0},
        ]
    )

    dataset_ref = service.save_dataset_frame(project_id="demo", version_id="dataset-1", frame=frame)
    reloaded_frame = service.load_dataset_frame(
        {
            "storage_backend": dataset_ref.storage_backend,
            "bucket": dataset_ref.bucket,
            "object_key": dataset_ref.object_key,
            "path": dataset_ref.path,
        }
    )
    pd.testing.assert_frame_equal(reloaded_frame, frame)

    bundle = {"pipeline": {"name": "demo-model"}, "feature_names": ["feature"], "task_type": "regression"}
    model_ref = service.save_model_bundle(project_id="demo", version_id="model-1", bundle=bundle)
    reloaded_bundle = service.load_model_bundle(
        {
            "storage_backend": model_ref.storage_backend,
            "bucket": model_ref.bucket,
            "object_key": model_ref.object_key,
            "artifact_path": model_ref.path,
        }
    )
    assert reloaded_bundle["pipeline"]["name"] == "demo-model"

    service.delete_dataset(
        {
            "storage_backend": dataset_ref.storage_backend,
            "bucket": dataset_ref.bucket,
            "object_key": dataset_ref.object_key,
            "path": dataset_ref.path,
        }
    )
    service.delete_model(
        {
            "storage_backend": model_ref.storage_backend,
            "bucket": model_ref.bucket,
            "object_key": model_ref.object_key,
            "artifact_path": model_ref.path,
        }
    )
    assert service._client.buckets["test-datasets"] == {}
    assert service._client.buckets["test-artifacts"] == {}


def test_registry_service_stores_minio_references_and_loads_them(minio_env):
    service = RegistryService()
    service.object_storage._client = _FakeS3Client()
    frame = pd.DataFrame(
        [
            {"feature": 1, "target": 0},
            {"feature": 2, "target": 1},
        ]
    )

    dataset_version = service.create_dataset_version(
        project_id="demo",
        source_name="inline",
        target="target",
        frame=frame,
    )
    assert dataset_version.storage_backend == "minio"
    assert dataset_version.path.startswith("s3://test-datasets/demo/dataset-")
    assert dataset_version.object_key.startswith("demo/dataset-")

    loaded_frame = service.load_dataset_version_frame(dataset_version.version_id)
    pd.testing.assert_frame_equal(loaded_frame, frame)

    model_version = service.register_model_version(
        project_id="demo",
        dataset_version_id=dataset_version.version_id,
        target="target",
        task_type="regression",
        feature_names=["feature"],
        metrics={"rmse": 0.1},
        bundle={"pipeline": {"name": "demo-model"}, "feature_names": ["feature"], "task_type": "regression"},
    )
    assert model_version.storage_backend == "minio"
    assert model_version.artifact_path.startswith("s3://test-artifacts/demo/model-")
    assert model_version.object_key.startswith("demo/model-")

    model_record, bundle = service.get_model_bundle(model_version.version_id)
    assert model_record["version_id"] == model_version.version_id
    assert bundle["pipeline"]["name"] == "demo-model"

    delete_result = service.delete_project("demo")
    assert delete_result["deleted"] is True
    assert service.object_storage._client.buckets["test-datasets"] == {}
    assert service.object_storage._client.buckets["test-artifacts"] == {}
