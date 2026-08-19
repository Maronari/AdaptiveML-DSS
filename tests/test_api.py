import importlib
import math
import sys
from io import BytesIO
from pathlib import Path

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from automl.lightautoml_backend import adapter as lama_adapter
from automl.lightautoml_backend.adapter import LIGHTAUTOML_AVAILABLE


TRAINING_DATASET = [
    {"temperature": 62, "vibration": 0.31, "pressure": 90, "target": 0},
    {"temperature": 71, "vibration": 0.44, "pressure": 95, "target": 0},
    {"temperature": 88, "vibration": 0.85, "pressure": 110, "target": 1},
    {"temperature": 92, "vibration": 0.91, "pressure": 118, "target": 1},
    {"temperature": 75, "vibration": 0.49, "pressure": 98, "target": 0},
    {"temperature": 96, "vibration": 0.95, "pressure": 122, "target": 1},
    {"temperature": 68, "vibration": 0.37, "pressure": 92, "target": 0},
    {"temperature": 90, "vibration": 0.88, "pressure": 115, "target": 1},
    {"temperature": 73, "vibration": 0.43, "pressure": 96, "target": 0},
    {"temperature": 94, "vibration": 0.93, "pressure": 120, "target": 1},
]


def build_forecasting_dataset(rows: int = 96) -> list[dict[str, float | str | int]]:
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=rows, freq="h")
    dataset = []
    for index, timestamp in enumerate(timestamps):
        signal = (
            180.0
            + index * 0.35
            + 22.0 * math.sin(2.0 * math.pi * timestamp.hour / 24.0)
            + (8.0 if timestamp.dayofweek < 5 else -4.0)
        )
        dataset.append(
            {
                "Дата": timestamp.isoformat(),
                "Рабочий день": int(timestamp.dayofweek < 5),
                "Электропотребление": round(signal, 3),
            }
        )
    return dataset


def build_retraining_dataset(rows: int = 180) -> list[dict[str, float | str | int]]:
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=rows, freq="D")
    dataset = []
    for index, timestamp in enumerate(timestamps):
        regime_shift = 60.0 if index >= 150 else 0.0
        weekly_pattern = 8.0 if timestamp.dayofweek < 5 else -4.0
        target = 120.0 + index * 2.4 + regime_shift + weekly_pattern
        dataset.append(
            {
                "timestamp": timestamp.isoformat(),
                "period_index": index,
                "daily_temperature": round(14.0 + math.sin(index / 6.0) * 4.5, 3),
                "target": round(target, 3),
            }
        )
    return dataset


def build_hourly_energy_dataset() -> list[dict[str, float | int]]:
    return [
        {"Час": 0, "Температура": 18.0, "Потребление": 110.0},
        {"Час": 4, "Температура": 17.5, "Потребление": 104.0},
        {"Час": 8, "Температура": 19.2, "Потребление": 126.0},
        {"Час": 12, "Температура": 23.1, "Потребление": 149.0},
        {"Час": 16, "Температура": 24.6, "Потребление": 158.0},
        {"Час": 20, "Температура": 21.4, "Потребление": 136.0},
        {"Час": 1, "Температура": 17.8, "Потребление": 108.0},
        {"Час": 5, "Температура": 17.1, "Потребление": 103.0},
        {"Час": 9, "Температура": 20.3, "Потребление": 129.0},
        {"Час": 13, "Температура": 23.7, "Потребление": 151.0},
    ]


def build_timestamped_energy_dataset() -> list[dict[str, float | int | str]]:
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=10, freq="4h")
    dataset = []
    for index, timestamp in enumerate(timestamps):
        dataset.append(
            {
                "Дата": timestamp.isoformat(),
                "Час": int(timestamp.hour),
                "Температура": round(17.5 + index * 0.8, 1),
                "Потребление": round(104.0 + index * 5.5, 1),
            }
        )
    return dataset


def build_weather_dataset() -> list[dict[str, float | int | str]]:
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=10, freq="4h")
    dataset = []
    for index, timestamp in enumerate(timestamps):
        dataset.append(
            {
                "dateByOurs": timestamp.isoformat(),
                "Температура наружная": round(15.0 + index * 0.7, 1),
                "Облачность": round((index % 5) * 0.2, 2),
            }
        )
    return dataset


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTIVEML_STORAGE_ROOT", str(tmp_path / "storage"))
    for module_name in [
        "backend.main",
        "backend.api.routes",
        "backend.services.settings",
        "backend.services.dss_config_service",
        "backend.services.registry_service",
        "backend.services.dataset_service",
        "backend.services.job_service",
        "backend.services.monitoring_service",
        "backend.services.observability_service",
        "backend.services.training_service",
        "backend.services.prediction_service",
        "backend.services.retraining_service",
        "backend.services.decision_service",
        "backend.services.explanation_service",
        "dss.rules.loader",
        "dss.experta_engine.engine",
    ]:
        sys.modules.pop(module_name, None)

    from backend.services.settings import get_settings

    get_settings.cache_clear()
    app_module = importlib.import_module("backend.main")

    return TestClient(app_module.app)


def test_healthcheck(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_favicon_returns_empty_response(client: TestClient):
    response = client.get("/favicon.ico")
    assert response.status_code == 204
    assert response.text == ""


def test_root_redirects_to_frontend_shell(client: TestClient):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in {302, 307}
    assert response.headers["location"] == "/app/"


def test_frontend_app_served(client: TestClient):
    response = client.get("/app/")
    assert response.status_code == 200
    assert "AutoML" in response.text
    assert 'id="container"' in response.text
    assert 'id="newProjectModal"' in response.text


def test_projects_endpoint_lists_created_and_inferred_projects(client: TestClient):
    create_response = client.post("/projects", json={"name": "Электроэнергия март"})
    assert create_response.status_code == 201
    created_project = create_response.json()
    assert created_project["project_id"] == "электроэнергия-март"

    train_response = client.post(
        "/training/run",
        json={
            "project_id": "projects-list-demo",
            "target": "target",
            "records": TRAINING_DATASET,
        },
    )
    assert train_response.status_code == 200

    list_response = client.get("/projects")
    assert list_response.status_code == 200
    items = list_response.json()["items"]
    project_ids = {item["project_id"] for item in items}
    assert "электроэнергия-март" in project_ids
    assert "projects-list-demo" in project_ids

    explicit_project = next(item for item in items if item["project_id"] == "электроэнергия-март")
    inferred_project = next(item for item in items if item["project_id"] == "projects-list-demo")
    assert explicit_project["has_models"] is False
    assert inferred_project["has_models"] is True
    assert inferred_project["model_versions"] >= 1


def test_project_models_history_endpoint_lists_versions(client: TestClient):
    first_training = client.post(
        "/training/run",
        json={
            "project_id": "models-history-demo",
            "target": "target",
            "records": TRAINING_DATASET,
        },
    )
    assert first_training.status_code == 200

    second_training = client.post(
        "/training/run",
        json={
            "project_id": "models-history-demo",
            "target": "target",
            "records": TRAINING_DATASET,
        },
    )
    assert second_training.status_code == 200

    history_response = client.get("/projects/models-history-demo/models")
    assert history_response.status_code == 200
    body = history_response.json()
    assert body["project_id"] == "models-history-demo"
    assert body["latest_model_version_id"] == second_training.json()["model_version"]["version_id"]
    assert len(body["items"]) == 2
    assert body["items"][0]["version_id"] == second_training.json()["model_version"]["version_id"]
    assert body["items"][0]["dataset_source_name"] == "inline"
    assert body["items"][0]["metric_value"] is not None
    assert "holdout_predictions" not in body["items"][0]
    assert "training_artifacts" not in body["items"][0]
    assert body["items"][0]["holdout_rows"] >= 1
    assert body["items"][0]["has_training_artifacts"] is True


def test_project_models_history_endpoint_can_activate_specific_version(client: TestClient):
    first_training = client.post(
        "/training/run",
        json={
            "project_id": "models-activate-demo",
            "target": "target",
            "records": TRAINING_DATASET,
        },
    )
    assert first_training.status_code == 200
    second_training = client.post(
        "/training/run",
        json={
            "project_id": "models-activate-demo",
            "target": "target",
            "records": TRAINING_DATASET,
        },
    )
    assert second_training.status_code == 200
    history_before_activate = client.get("/projects/models-activate-demo/models")
    assert history_before_activate.status_code == 200
    before_body = history_before_activate.json()
    inactive_item = next(item for item in before_body["items"] if not item["is_champion"])
    previously_champion = before_body["champion_model_version_id"]
    target_version = inactive_item["version_id"]

    activate_response = client.post(f"/projects/models-activate-demo/models/{target_version}/activate")
    assert activate_response.status_code == 200
    assert activate_response.json()["version_id"] == target_version
    assert activate_response.json()["status"] == "champion"

    history_response = client.get("/projects/models-activate-demo/models")
    assert history_response.status_code == 200
    body = history_response.json()
    assert body["champion_model_version_id"] == target_version
    activated_item = next(item for item in body["items"] if item["version_id"] == target_version)
    previous_champion_item = next(item for item in body["items"] if item["version_id"] == previously_champion)
    assert activated_item["is_champion"] is True
    assert previous_champion_item["status"] == "archived"


def test_delete_project_removes_registry_records_and_storage(client: TestClient):
    train_response = client.post(
        "/training/run",
        json={
            "project_id": "delete-demo",
            "target": "target",
            "records": TRAINING_DATASET,
        },
    )
    assert train_response.status_code == 200
    training_payload = train_response.json()

    dataset_path = Path(training_payload["dataset_version"]["path"])
    artifact_path = Path(training_payload["model_version"]["artifact_path"])
    assert dataset_path.exists()
    assert artifact_path.exists()

    list_response = client.get("/projects")
    assert list_response.status_code == 200
    assert "delete-demo" in {item["project_id"] for item in list_response.json()["items"]}

    delete_response = client.delete("/projects/delete-demo")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True
    assert delete_response.json()["dataset_versions_removed"] == 1
    assert delete_response.json()["model_versions_removed"] == 1

    list_after_delete = client.get("/projects")
    assert list_after_delete.status_code == 200
    assert "delete-demo" not in {item["project_id"] for item in list_after_delete.json()["items"]}

    latest_model_response = client.get("/models/latest", params={"project_id": "delete-demo"})
    assert latest_model_response.status_code == 400
    assert not dataset_path.exists()
    assert not artifact_path.exists()


def test_training_prediction_and_decision_flow(client: TestClient):
    train_response = client.post(
        "/training/run",
        json={
            "project_id": "demo",
            "target": "target",
            "records": TRAINING_DATASET,
        },
    )
    assert train_response.status_code == 200
    training_payload = train_response.json()
    assert training_payload["model_version"]["status"] == "champion"

    inference_records = [
        {"temperature": 97, "vibration": 0.97, "pressure": 125},
        {"temperature": 66, "vibration": 0.32, "pressure": 91},
    ]
    predict_response = client.post(
        "/predictions/run",
        json={"project_id": "demo", "records": inference_records},
    )
    assert predict_response.status_code == 200
    prediction_payload = predict_response.json()
    assert len(prediction_payload["predictions"]) == 2

    decision_response = client.post(
        "/decision/run",
        json={"project_id": "demo", "records": inference_records},
    )
    assert decision_response.status_code == 200
    decision_payload = decision_response.json()
    assert len(decision_payload["recommendations"]) == 2
    assert "actions" in decision_payload["recommendations"][0]["recommendation"]


def test_training_from_xlsx_upload(client: TestClient):
    frame = pd.DataFrame(TRAINING_DATASET)
    payload = BytesIO()
    frame.to_excel(payload, index=False)
    payload.seek(0)

    response = client.post(
        "/training/run/file",
        files={
            "file": (
                "train.xlsx",
                payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"project_id": "xlsx-demo", "target": "target"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] in {"lightautoml", "sklearn-fallback"}


def test_dataset_inspect_file_suggests_target_for_semicolon_csv(client: TestClient):
    frame = pd.DataFrame(build_hourly_energy_dataset())
    target_column = frame.columns[-1]
    csv_payload = frame.to_csv(index=False, sep=";").encode("cp1251")

    response = client.post(
        "/datasets/inspect/file",
        files={
            "file": (
                "energy.csv",
                csv_payload,
                "text/csv",
            )
        },
        data={"project_id": "upload-inspect-demo"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == "upload-inspect-demo"
    assert body["rows"] == len(frame)
    assert body["recommended_target"] == "Потребление"
    assert body["target_candidates"]
    assert body["target_candidates"][0]["column"] == "Потребление"
    assert "Температура" in body["columns"]
    assert len(body["sample_rows"]) >= 1
    assert body["column_type_summary"]["total_columns"] == 3
    assert body["column_type_summary"]["items"][0]["kind"] == "numeric"
    assert body["temporal_context"]["available"] is False
    assert body["target_summaries"][target_column]["task_type"] == "regression"
    assert body["project_context"]["project_id"] == "upload-inspect-demo"
    assert body["project_context"]["is_new_project"] is True


def test_dataset_inspect_file_returns_temporal_and_project_context(client: TestClient):
    project_response = client.post("/projects", json={"name": "Upload Context Demo"})
    assert project_response.status_code == 201
    project_id = project_response.json()["project_id"]

    seed_frame = pd.DataFrame(build_timestamped_energy_dataset())
    date_column = seed_frame.columns[0]
    target_column = seed_frame.columns[-1]
    seed_payload = BytesIO()
    seed_frame.to_excel(seed_payload, index=False)
    seed_payload.seek(0)

    register_response = client.post(
        "/datasets/register/file",
        files={
            "file": (
                "seed.xlsx",
                seed_payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"project_id": project_id, "target": target_column},
    )
    assert register_response.status_code == 201
    dataset_version_id = register_response.json()["dataset_version"]["version_id"]

    train_response = client.post(
        "/training/run/dataset",
        data={
            "project_id": project_id,
            "dataset_version_id": dataset_version_id,
            "backend": "sklearn",
        },
    )
    assert train_response.status_code == 200

    inspect_frame = pd.DataFrame(build_timestamped_energy_dataset())
    inspect_payload = BytesIO()
    inspect_frame.to_excel(inspect_payload, index=False)
    inspect_payload.seek(0)

    inspect_response = client.post(
        "/datasets/inspect/file",
        files={
            "file": (
                "inspect.xlsx",
                inspect_payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"project_id": project_id},
    )
    assert inspect_response.status_code == 200
    body = inspect_response.json()
    assert body["temporal_context"]["available"] is True
    assert body["temporal_context"]["column"] == date_column
    assert body["temporal_context"]["frequency_minutes"] == 240.0
    assert body["target_summaries"][target_column]["numeric_stats"]["max"] > 0
    assert body["project_context"]["dataset_versions"] == 1
    assert body["project_context"]["model_versions"] == 1
    assert body["project_context"]["latest_dataset_source_name"] == "seed.xlsx"
    assert body["project_context"]["latest_model_version_id"] is not None


def test_dataset_inspect_and_register_multiple_files(client: TestClient):
    energy_frame = pd.DataFrame(build_timestamped_energy_dataset())
    weather_frame = pd.DataFrame(build_weather_dataset())

    energy_payload = BytesIO()
    energy_frame.to_excel(energy_payload, index=False)
    energy_payload.seek(0)

    weather_payload = BytesIO()
    weather_frame.to_excel(weather_payload, index=False)
    weather_payload.seek(0)

    inspect_response = client.post(
        "/datasets/inspect/files",
        files=[
            (
                "files",
                ("energy.xlsx", energy_payload.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
            (
                "files",
                ("weather.xlsx", weather_payload.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
        ],
        data={"project_id": "batch-upload-demo"},
    )
    assert inspect_response.status_code == 200
    inspect_body = inspect_response.json()
    assert inspect_body["project_id"] == "batch-upload-demo"
    assert inspect_body["rows"] == len(energy_frame)
    assert "timestamp" in inspect_body["columns"]
    assert "Температура наружная" in inspect_body["columns"]
    assert "Потребление" in inspect_body["columns"]
    assert "Потребление" in inspect_body["target_summaries"]

    register_response = client.post(
        "/datasets/register/files",
        files=[
            (
                "files",
                ("energy.xlsx", energy_payload.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
            (
                "files",
                ("weather.xlsx", weather_payload.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
        ],
        data={"project_id": "batch-upload-demo", "target": "Потребление"},
    )
    assert register_response.status_code == 201
    register_body = register_response.json()
    assert register_body["rows"] == len(energy_frame)
    assert register_body["target"] == "Потребление"
    assert register_body["dataset_version"]["project_id"] == "batch-upload-demo"
    assert register_body["source_files"] == ["energy.xlsx", "weather.xlsx"]


def test_dataset_register_and_train_from_saved_dataset(client: TestClient):
    frame = pd.DataFrame(TRAINING_DATASET)
    payload = BytesIO()
    frame.to_excel(payload, index=False)
    payload.seek(0)

    register_response = client.post(
        "/datasets/register/file",
        files={
            "file": (
                "train.xlsx",
                payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"project_id": "registered-dataset-demo", "target": "target"},
    )
    assert register_response.status_code == 201
    register_body = register_response.json()
    dataset_version_id = register_body["dataset_version"]["version_id"]

    dataset_response = client.get(f"/datasets/{dataset_version_id}")
    assert dataset_response.status_code == 200
    dataset_body = dataset_response.json()
    assert dataset_body["dataset_version"]["version_id"] == dataset_version_id
    assert dataset_body["target"] == "target"

    train_response = client.post(
        "/training/run/dataset",
        data={
            "project_id": "registered-dataset-demo",
            "dataset_version_id": dataset_version_id,
            "backend": "sklearn",
        },
    )
    assert train_response.status_code == 200
    train_body = train_response.json()
    assert train_body["dataset_version"]["version_id"] == dataset_version_id
    assert train_body["backend"] == "sklearn-fallback"
    assert train_body["model_version"]["status"] == "champion"


def test_latest_project_dataset_endpoint_returns_newest_saved_dataset(client: TestClient):
    first_frame = pd.DataFrame(TRAINING_DATASET[:4])
    second_frame = pd.DataFrame(TRAINING_DATASET)

    first_payload = BytesIO()
    first_frame.to_excel(first_payload, index=False)
    first_payload.seek(0)
    second_payload = BytesIO()
    second_frame.to_excel(second_payload, index=False)
    second_payload.seek(0)

    first_response = client.post(
        "/datasets/register/file",
        files={
            "file": (
                "first.xlsx",
                first_payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"project_id": "latest-dataset-demo", "target": "target"},
    )
    assert first_response.status_code == 201

    second_response = client.post(
        "/datasets/register/file",
        files={
            "file": (
                "second.xlsx",
                second_payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"project_id": "latest-dataset-demo", "target": "target"},
    )
    assert second_response.status_code == 201

    latest_response = client.get("/projects/latest-dataset-demo/datasets/latest")
    assert latest_response.status_code == 200
    latest_body = latest_response.json()
    assert latest_body["dataset_version"]["version_id"] == second_response.json()["dataset_version"]["version_id"]
    assert latest_body["rows"] == len(second_frame)
    assert latest_body["source_name"] == "second.xlsx"


def test_training_with_explicit_sklearn_options(client: TestClient):
    response = client.post(
        "/training/run",
        json={
            "project_id": "sklearn-demo",
            "target": "target",
            "records": TRAINING_DATASET,
            "training_options": {
                "task_type": "binary",
                "backend": "sklearn",
                "preset": "utilized",
                "algos": ["cb", "xgb"],
                "timeout_seconds": 45,
                "cpu_limit": 1,
                "test_size": 0.3,
                "cv_folds": 4,
                "enable_forecast": False,
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "sklearn-fallback"
    assert body["training_options"]["requested"]["backend"] == "sklearn"
    assert body["training_options"]["requested"]["preset"] == "utilized"
    assert body["training_options"]["effective"]["preset"] == "sklearn-random-forest"
    assert body["training_options"]["effective"]["algos"] == ["rf"]
    assert body["warnings"]


def test_training_auto_backend_falls_back_when_lightautoml_raises_value_error(client: TestClient, monkeypatch):
    def failing_lightautoml(*args, **kwargs):
        raise ValueError("synthetic LightAutoML failure")

    monkeypatch.setattr(lama_adapter, "LIGHTAUTOML_AVAILABLE", True)
    monkeypatch.setattr(lama_adapter.TabularAutoMLAdapter, "_train_with_lightautoml", failing_lightautoml)

    response = client.post(
        "/training/run",
        json={
            "project_id": "auto-fallback-demo",
            "target": "target",
            "records": TRAINING_DATASET,
            "training_options": {
                "backend": "auto",
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "sklearn-fallback"
    assert body["warnings"]
    assert "synthetic LightAutoML failure" in body["warnings"][0]


@pytest.mark.skipif(not LIGHTAUTOML_AVAILABLE, reason="LightAutoML is not available")
def test_training_with_utilized_lightautoml_preset(client: TestClient):
    response = client.post(
        "/training/run",
        json={
            "project_id": "utilized-demo",
            "target": "target",
            "records": TRAINING_DATASET,
            "training_options": {
                "task_type": "binary",
                "backend": "lightautoml",
                "preset": "utilized",
                "algos": ["lgb", "linear_l2"],
                "timeout_seconds": 10,
                "cpu_limit": 1,
                "test_size": 0.2,
                "cv_folds": 2,
                "enable_forecast": False,
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "lightautoml"
    assert body["training_options"]["effective"]["preset"] == "utilized"
    assert body["training_options"]["effective"]["algos"] == ["lgb", "linear_l2"]
    assert body["training_options"]["effective"]["cv_folds"] == 2


def test_prediction_compare_from_xlsx_upload(client: TestClient):
    train_response = client.post(
        "/training/run",
        json={
            "project_id": "compare-demo",
            "target": "target",
            "records": TRAINING_DATASET,
        },
    )
    assert train_response.status_code == 200

    frame = pd.DataFrame(TRAINING_DATASET)
    payload = BytesIO()
    frame.to_excel(payload, index=False)
    payload.seek(0)

    response = client.post(
        "/predictions/compare/file",
        files={
            "file": (
                "compare.xlsx",
                payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"project_id": "compare-demo", "target": "target"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == len(TRAINING_DATASET)
    assert body["target"] == "target"
    assert body["items"][0]["record"]["target"] in {0, 1}
    assert "prediction" in body["items"][0]
    assert body["metrics"]


def test_compare_schema_accepts_matching_upload(client: TestClient):
    dataset = build_hourly_energy_dataset()
    train_response = client.post(
        "/training/run",
        json={
            "project_id": "schema-demo",
            "target": "Потребление",
            "records": dataset,
            "training_options": {
                "task_type": "regression",
                "backend": "sklearn",
                "enable_forecast": False,
            },
        },
    )
    assert train_response.status_code == 200

    frame = pd.DataFrame(dataset)
    payload = BytesIO()
    frame.to_excel(payload, index=False)
    payload.seek(0)

    response = client.post(
        "/predictions/compare/file/schema",
        files={
            "file": (
                "matching.xlsx",
                payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"project_id": "schema-demo", "target": "Потребление"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["missing_features"] == []
    assert body["target_matches_champion"] is True
    assert "Час" in body["expected_features"]


def test_compare_schema_reports_missing_features(client: TestClient):
    dataset = build_hourly_energy_dataset()
    train_response = client.post(
        "/training/run",
        json={
            "project_id": "schema-missing-demo",
            "target": "Потребление",
            "records": dataset,
            "training_options": {
                "task_type": "regression",
                "backend": "sklearn",
                "enable_forecast": False,
            },
        },
    )
    assert train_response.status_code == 200

    frame = pd.DataFrame(dataset).drop(columns=["Час"])
    payload = BytesIO()
    frame.to_excel(payload, index=False)
    payload.seek(0)

    response = client.post(
        "/predictions/compare/file/schema",
        files={
            "file": (
                "missing_hour.xlsx",
                payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"project_id": "schema-missing-demo", "target": "Потребление"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    assert body["target_matches_champion"] is True
    assert body["missing_features"] == ["Час"]
    assert "Температура" in body["uploaded_columns"]


def test_compare_schema_reports_source_datetime_column_instead_of_engineered_features(client: TestClient):
    dataset = build_timestamped_energy_dataset()
    train_response = client.post(
        "/training/run",
        json={
            "project_id": "schema-datetime-demo",
            "target": "Потребление",
            "records": dataset,
            "training_options": {
                "task_type": "regression",
                "backend": "sklearn",
                "enable_forecast": False,
            },
        },
    )
    assert train_response.status_code == 200

    frame = pd.DataFrame(dataset).drop(columns=["Дата"])
    payload = BytesIO()
    frame.to_excel(payload, index=False)
    payload.seek(0)

    response = client.post(
        "/predictions/compare/file/schema",
        files={
            "file": (
                "missing_datetime.xlsx",
                payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"project_id": "schema-datetime-demo", "target": "Потребление"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    assert body["missing_inputs"] == ["Дата"]
    assert body["missing_features"] == [
        "Дата__ts",
        "Дата__hour",
        "Дата__dayofweek",
        "Дата__month",
    ]


def test_latest_comparison_returns_most_recent_saved_payload(client: TestClient):
    dataset = build_hourly_energy_dataset()
    train_response = client.post(
        "/training/run",
        json={
            "project_id": "latest-graph-demo",
            "target": "Потребление",
            "records": dataset,
            "training_options": {
                "task_type": "regression",
                "backend": "sklearn",
                "enable_forecast": False,
            },
        },
    )
    assert train_response.status_code == 200

    frame = pd.DataFrame(dataset)
    payload = BytesIO()
    frame.to_excel(payload, index=False)
    payload.seek(0)

    compare_response = client.post(
        "/predictions/compare/file",
        files={
            "file": (
                "latest.xlsx",
                payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"project_id": "latest-graph-demo", "target": "Потребление"},
    )
    assert compare_response.status_code == 200

    latest_response = client.get("/predictions/compare/latest")
    assert latest_response.status_code == 200
    body = latest_response.json()
    assert body["project_id"] == "latest-graph-demo"
    assert body["task_type"] == "regression"
    assert body["rows"] == len(dataset)
    assert body["saved_at"]
    assert len(body["items"]) == len(dataset)
    assert set(body["items"][0]) == {"actual", "prediction"}


def test_latest_model_and_forecast_endpoints(client: TestClient):
    forecasting_dataset = build_forecasting_dataset()

    train_response = client.post(
        "/training/run",
        json={
            "project_id": "model-api-demo",
            "target": "Электропотребление",
            "records": forecasting_dataset,
        },
    )
    assert train_response.status_code == 200
    trained_model_version = train_response.json()["model_version"]["version_id"]

    latest_model_response = client.get("/models/latest", params={"project_id": "model-api-demo"})
    assert latest_model_response.status_code == 200
    latest_model = latest_model_response.json()
    assert latest_model["version_id"] == trained_model_version
    assert latest_model["forecasting"]["available"] is True
    assert latest_model["project_id"] == "model-api-demo"
    assert {"r", "r2", "mse", "rmse", "aic", "bic"} <= set(latest_model["metrics"])
    assert latest_model["holdout_predictions"]
    assert latest_model["training_artifacts"]["holdout_rows"] == len(latest_model["holdout_predictions"])
    assert latest_model["training_artifacts"]["feature_importances"]
    assert latest_model["training_artifacts"]["training_options"]["effective"]["backend"]

    model_response = client.get(f"/models/{trained_model_version}")
    assert model_response.status_code == 200
    model_payload = model_response.json()
    assert model_payload["version_id"] == trained_model_version
    assert model_payload["target"] == "Электропотребление"
    assert model_payload["holdout_predictions"]
    assert model_payload["holdout_predictions"][0]["actual"] is not None
    assert "prediction" in model_payload["holdout_predictions"][0]
    assert model_payload["training_artifacts"]["forecasting"]["available"] is True
    assert model_payload["forecasting"]["metrics"]
    assert model_payload["forecasting"]["historical_fit_rows"] > 0

    forecast_response = client.get(
        f"/models/{trained_model_version}/forecast",
        params={"steps": 5},
    )
    assert forecast_response.status_code == 200
    forecast_payload = forecast_response.json()
    assert forecast_payload["model_version"] == trained_model_version
    assert forecast_payload["steps"] == 5
    assert len(forecast_payload["historical_fit"]) > 0
    assert set(forecast_payload["historical_fit"][0]) == {"timestamp", "target", "prediction"}
    assert len(forecast_payload["recent_history"]) > 0
    assert len(forecast_payload["forecast"]) == 5


def test_forecast_run_from_temporal_regression_dataset(client: TestClient):
    forecasting_dataset = build_forecasting_dataset()

    train_response = client.post(
        "/training/run",
        json={
            "project_id": "forecast-demo",
            "target": "Электропотребление",
            "records": forecasting_dataset,
        },
    )
    assert train_response.status_code == 200
    training_body = train_response.json()
    assert training_body["task_type"] == "regression"
    assert {"r", "r2", "mse", "rmse", "aic", "bic"} <= set(training_body["metrics"])
    assert training_body["forecasting"]["available"] is True
    assert {"r", "r2", "mse", "rmse", "aic", "bic"} <= set(training_body["forecasting"]["forecast_metrics"])

    forecast_response = client.post(
        "/forecast/run",
        json={
            "project_id": "forecast-demo",
            "horizon_minutes": 30,
            "steps": 3,
        },
    )
    assert forecast_response.status_code == 200
    body = forecast_response.json()
    assert body["target"] == "Электропотребление"
    assert body["steps"] == 3
    assert body["base_frequency_minutes"] == 60
    assert len(body["forecast"]) == 3
    assert len(body["recent_history"]) > 0
    assert body["warning"] is not None
    assert body["forecast"][0]["timestamp"] < body["forecast"][1]["timestamp"] < body["forecast"][2]["timestamp"]


def test_forecast_returns_extended_recent_history_window(client: TestClient):
    forecasting_dataset = build_forecasting_dataset(rows=120)

    train_response = client.post(
        "/training/run",
        json={
            "project_id": "forecast-history-demo",
            "target": "Электропотребление",
            "records": forecasting_dataset,
        },
    )
    assert train_response.status_code == 200
    trained_model_version = train_response.json()["model_version"]["version_id"]

    forecast_response = client.get(
        f"/models/{trained_model_version}/forecast",
        params={"steps": 3},
    )
    assert forecast_response.status_code == 200
    body = forecast_response.json()
    assert len(body["recent_history"]) == 120


def test_dataset_inspect_file_suggests_unit_from_unit_column(client: TestClient):
    frame = pd.DataFrame(build_hourly_energy_dataset())
    frame["Единица измерения"] = "кВт·ч"
    csv_payload = frame.to_csv(index=False).encode("utf-8")

    response = client.post(
        "/datasets/inspect/file",
        files={"file": ("energy-with-unit.csv", csv_payload, "text/csv")},
        data={"project_id": "unit-inspect-demo"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["suggested_unit"] == "кВт·ч"


def test_forecast_comparison_matches_postfactum_actuals_and_computes_errors(client: TestClient):
    forecasting_dataset = build_forecasting_dataset()

    train_response = client.post(
        "/training/run",
        json={
            "project_id": "forecast-comparison-demo",
            "target": "Электропотребление",
            "records": forecasting_dataset,
            "unit": "МВт·ч",
        },
    )
    assert train_response.status_code == 200
    train_body = train_response.json()
    assert train_body["dataset_version"]["unit"] == "МВт·ч"

    forecast_response = client.post(
        "/forecast/run",
        json={
            "project_id": "forecast-comparison-demo",
            "horizon_minutes": 60,
            "steps": 3,
        },
    )
    assert forecast_response.status_code == 200
    forecast_body = forecast_response.json()
    assert forecast_body["unit"] == "МВт·ч"
    run_id = forecast_body["run_id"]
    forecast_points = forecast_body["forecast"]
    assert len(forecast_points) == 3

    prediction_0 = float(forecast_points[0]["prediction"])
    prediction_1 = float(forecast_points[1]["prediction"])
    actual_0 = prediction_0 + 5.0
    actual_1 = prediction_1 - 3.25

    postfactum_frame = pd.DataFrame(
        [
            {"Дата": forecast_points[0]["timestamp"], "Электропотребление": actual_0},
            {"Дата": forecast_points[1]["timestamp"], "Электропотребление": actual_1},
        ]
    )
    postfactum_csv = postfactum_frame.to_csv(index=False).encode("utf-8")

    register_response = client.post(
        "/datasets/register/file",
        files={"file": ("postfactum.csv", postfactum_csv, "text/csv")},
        data={"project_id": "forecast-comparison-demo", "target": "Электропотребление"},
    )
    assert register_response.status_code == 201

    comparison_response = client.get(f"/forecast/{run_id}/comparison")
    assert comparison_response.status_code == 200
    comparison_body = comparison_response.json()
    assert comparison_body["run_id"] == run_id
    assert comparison_body["target"] == "Электропотребление"
    assert comparison_body["unit"] == "МВт·ч"

    points = comparison_body["points"]
    assert len(points) == 3

    assert points[0]["actual"] == pytest.approx(actual_0)
    assert points[0]["abs_error"] == pytest.approx(actual_0 - prediction_0, abs=1e-4)
    expected_mape_0 = abs(actual_0 - prediction_0) / abs(actual_0) * 100.0
    assert points[0]["mape_percent"] == pytest.approx(expected_mape_0, abs=1e-2)

    assert points[1]["actual"] == pytest.approx(actual_1)
    assert points[1]["abs_error"] == pytest.approx(actual_1 - prediction_1, abs=1e-4)

    assert points[2]["actual"] is None
    assert points[2]["abs_error"] is None
    assert points[2]["mape_percent"] is None

    aggregate = comparison_body["aggregate"]
    assert aggregate["matched_points"] == 2
    expected_mean_abs_error = (abs(actual_0 - prediction_0) + abs(actual_1 - prediction_1)) / 2
    assert aggregate["mean_abs_error"] == pytest.approx(expected_mean_abs_error, abs=1e-4)
    assert aggregate["mean_mape_percent"] is not None


def test_retraining_all_history_activates_better_candidate(client: TestClient):
    dataset = build_retraining_dataset()

    train_response = client.post(
        "/training/run",
        json={
            "project_id": "retrain-demo",
            "target": "target",
            "records": dataset[:150],
            "training_options": {
                "task_type": "regression",
                "backend": "sklearn",
                "enable_forecast": False,
            },
        },
    )
    assert train_response.status_code == 200
    initial_model_version = train_response.json()["model_version"]["version_id"]

    new_frame = pd.DataFrame(dataset[150:])
    payload = BytesIO()
    new_frame.to_excel(payload, index=False)
    payload.seek(0)

    retrain_response = client.post(
        "/retraining/run/file",
        files={
            "file": (
                "new_labeled.xlsx",
                payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={
            "project_id": "retrain-demo",
            "target": "target",
            "backend": "sklearn",
            "enable_forecast": "false",
            "history_scope": "all_history",
            "minimum_relative_improvement": "0.01",
        },
    )
    assert retrain_response.status_code == 200
    body = retrain_response.json()
    assert body["current_model_version"] == initial_model_version
    assert body["activated"] is True
    assert body["candidate_model_version"]["status"] == "champion"
    assert body["evaluation"]["profit"]["is_better"] is True
    assert body["evaluation"]["profit"]["meets_threshold"] is True
    assert body["selection_summary"]["historical_rows_selected"] == 150
    assert body["selection_summary"]["new_rows_reserved_for_evaluation"] == 6


def test_retraining_from_saved_dataset_version(client: TestClient):
    dataset = build_retraining_dataset()

    train_response = client.post(
        "/training/run",
        json={
            "project_id": "retrain-dataset-version-demo",
            "target": "target",
            "records": dataset[:150],
            "training_options": {
                "task_type": "regression",
                "backend": "sklearn",
                "enable_forecast": False,
            },
        },
    )
    assert train_response.status_code == 200

    new_frame = pd.DataFrame(dataset[150:])
    payload = BytesIO()
    new_frame.to_excel(payload, index=False)
    payload.seek(0)

    register_response = client.post(
        "/datasets/register/file",
        files={
            "file": (
                "retrain.xlsx",
                payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"project_id": "retrain-dataset-version-demo", "target": "target"},
    )
    assert register_response.status_code == 201
    dataset_version_id = register_response.json()["dataset_version"]["version_id"]

    retrain_response = client.post(
        "/retraining/run/dataset",
        data={
            "project_id": "retrain-dataset-version-demo",
            "dataset_version_id": dataset_version_id,
            "backend": "sklearn",
            "enable_forecast": "false",
            "history_scope": "all_history",
            "minimum_relative_improvement": "0.01",
        },
    )
    assert retrain_response.status_code == 200
    body = retrain_response.json()
    assert body["dataset_version"]["project_id"] == "retrain-dataset-version-demo"
    assert body["candidate_model_version"]["version_id"]


def test_retraining_recent_window_uses_last_30_days_history(client: TestClient):
    dataset = build_retraining_dataset()

    train_response = client.post(
        "/training/run",
        json={
            "project_id": "retrain-window-demo",
            "target": "target",
            "records": dataset[:150],
            "training_options": {
                "task_type": "regression",
                "backend": "sklearn",
                "enable_forecast": False,
            },
        },
    )
    assert train_response.status_code == 200

    response = client.post(
        "/retraining/run",
        json={
            "project_id": "retrain-window-demo",
            "target": "target",
            "records": dataset[150:],
            "training_options": {
                "task_type": "regression",
                "backend": "sklearn",
                "enable_forecast": False,
            },
            "retraining_options": {
                "history_scope": "last_30_days",
                "minimum_relative_improvement": 0.01,
                "auto_activate": False,
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["activated"] is False
    assert body["selection_summary"]["history_scope"] == "last_30_days"
    assert body["selection_summary"]["historical_rows_selected"] == 31
    assert body["selection_summary"]["new_rows_used_for_training"] == 24
    assert body["selection_summary"]["candidate_training_rows"] == 55


def test_retraining_inspect_file_returns_champion_compatibility(client: TestClient):
    dataset = build_retraining_dataset()

    train_response = client.post(
        "/training/run",
        json={
            "project_id": "retrain-inspect-demo",
            "target": "target",
            "records": dataset[:150],
            "training_options": {
                "task_type": "regression",
                "backend": "sklearn",
                "enable_forecast": False,
            },
        },
    )
    assert train_response.status_code == 200

    new_frame = pd.DataFrame(dataset[150:])
    payload = BytesIO()
    new_frame.to_excel(payload, index=False)
    payload.seek(0)

    response = client.post(
        "/retraining/inspect/file",
        files={
            "file": (
                "retrain_batch.xlsx",
                payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"project_id": "retrain-inspect-demo"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["expected_target"] == "target"
    assert body["recommended_target"] == "target"
    assert body["champion_model"]["target"] == "target"
    assert body["champion_model"]["version_id"]
    assert body["compatibility"]["ready"] is True
    assert body["compatibility"]["target_present"] is True
    assert body["compatibility"]["missing_inputs"] == []
    assert body["compatibility"]["validation_error"] is None
    assert body["project_context"]["model_versions"] == 1


def test_retraining_inspect_file_reports_missing_features(client: TestClient):
    dataset = build_hourly_energy_dataset()
    frame = pd.DataFrame(dataset)
    target_column = frame.columns[-1]
    missing_column = frame.columns[0]
    train_response = client.post(
        "/training/run",
        json={
            "project_id": "retrain-inspect-missing-demo",
            "target": target_column,
            "records": dataset,
            "training_options": {
                "task_type": "regression",
                "backend": "sklearn",
                "enable_forecast": False,
            },
        },
    )
    assert train_response.status_code == 200

    frame = frame.drop(columns=[missing_column])
    payload = BytesIO()
    frame.to_excel(payload, index=False)
    payload.seek(0)

    response = client.post(
        "/retraining/inspect/file",
        files={
            "file": (
                "missing_hour.xlsx",
                payload.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"project_id": "retrain-inspect-missing-demo"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["expected_target"] == target_column
    assert body["compatibility"]["ready"] is False
    assert body["compatibility"]["target_present"] is True
    assert body["compatibility"]["missing_features"] == [missing_column]
    assert body["compatibility"]["missing_inputs"] == [missing_column]
    assert frame.columns[0] in body["compatibility"]["uploaded_columns"]


def test_jobs_monitoring_endpoint_returns_queued_job_and_can_be_polled(client: TestClient):
    create_response = client.post("/projects", json={"name": "Jobs Demo"})
    assert create_response.status_code == 201
    project_id = create_response.json()["project_id"]

    enqueue_response = client.post("/jobs/monitoring/project", data={"project_id": project_id})
    assert enqueue_response.status_code == 200
    job = enqueue_response.json()
    assert job["status"] == "queued"
    assert job["job_type"] == "monitoring_project"
    assert job["project_id"] == project_id

    list_response = client.get("/jobs", params={"project_id": project_id})
    assert list_response.status_code == 200
    items = list_response.json()["items"]
    assert len(items) == 1
    assert items[0]["job_id"] == job["job_id"]
    assert items[0]["status"] == "queued"

    detail_response = client.get(f"/jobs/{job['job_id']}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["job_id"] == job["job_id"]
    assert detail["status"] == "queued"
    assert detail["result"] is None
    assert detail["logs"] == []


def test_dss_rulesets_endpoint_allows_read_and_write_config(client: TestClient):
    read_response = client.get("/dss/rulesets")
    assert read_response.status_code == 200
    original_config = read_response.json()
    assert "default_rule_set" in original_config
    assert "rule_sets" in original_config

    updated_config = {
        **original_config,
        "default_rule_set": "inline_default",
        "rule_sets": {
            **original_config["rule_sets"],
            "inline_default": {
                **original_config["rule_sets"]["inline_default"],
                "scenarios": {
                    **original_config["rule_sets"]["inline_default"]["scenarios"],
                    "observe": [
                        "Observe from test suite",
                        *original_config["rule_sets"]["inline_default"]["scenarios"]["observe"][1:],
                    ],
                },
            },
        },
    }

    save_response = client.put("/dss/rulesets", json=updated_config)
    assert save_response.status_code == 200
    saved_config = save_response.json()
    assert saved_config["rule_sets"]["inline_default"]["scenarios"]["observe"][0] == "Observe from test suite"

    reread_response = client.get("/dss/rulesets")
    assert reread_response.status_code == 200
    reread_config = reread_response.json()
    assert reread_config["rule_sets"]["inline_default"]["scenarios"]["observe"][0] == "Observe from test suite"
