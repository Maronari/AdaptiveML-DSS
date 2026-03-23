import importlib
import math
import sys
from io import BytesIO

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


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTIVEML_STORAGE_ROOT", str(tmp_path / "storage"))
    for module_name in [
        "backend.main",
        "backend.api.routes",
        "backend.services.settings",
        "backend.services.registry_service",
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


def test_frontend_app_served(client: TestClient):
    response = client.get("/app/")
    assert response.status_code == 200
    assert "AdaptiveML DSS" in response.text
    assert "prediction-chart" in response.text


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

    model_response = client.get(f"/models/{trained_model_version}")
    assert model_response.status_code == 200
    model_payload = model_response.json()
    assert model_payload["version_id"] == trained_model_version
    assert model_payload["target"] == "Электропотребление"

    forecast_response = client.get(
        f"/models/{trained_model_version}/forecast",
        params={"steps": 5},
    )
    assert forecast_response.status_code == 200
    forecast_payload = forecast_response.json()
    assert forecast_payload["model_version"] == trained_model_version
    assert forecast_payload["steps"] == 5
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
    assert training_body["forecasting"]["available"] is True

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
