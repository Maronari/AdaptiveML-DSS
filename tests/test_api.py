import importlib
import sys
from io import BytesIO

import pytest
import pandas as pd
from fastapi.testclient import TestClient


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
