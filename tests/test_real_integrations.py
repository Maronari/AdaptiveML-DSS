import importlib
import sys

import pandas as pd
import pytest

from automl.lightautoml_backend.adapter import LIGHTAUTOML_AVAILABLE
from dss.experta_engine.engine import EXPERTA_AVAILABLE
from explainability.shap_service.service import SHAP_AVAILABLE


def build_binary_training_records(rows: int = 48) -> list[dict[str, float | int]]:
    records = []
    for index in range(rows):
        temperature = 55 + index % 10
        vibration = 0.2 + (index % 6) * 0.12
        pressure = 80 + (index % 8) * 4
        score = temperature * 0.8 + vibration * 60 + pressure * 0.25
        target = int(score > 86)
        records.append(
            {
                "temperature": temperature,
                "vibration": round(vibration, 3),
                "pressure": pressure,
                "target": target,
            }
        )
    return records


@pytest.fixture()
def isolated_modules(tmp_path, monkeypatch):
    monkeypatch.setenv("ADAPTIVEML_STORAGE_ROOT", str(tmp_path / "storage"))

    for module_name in [
        "backend.services.settings",
        "backend.services.object_storage_service",
        "backend.services.registry_service",
        "backend.services.dataset_service",
        "backend.services.training_service",
        "backend.services.prediction_service",
        "backend.services.explanation_service",
        "backend.services.decision_service",
    ]:
        sys.modules.pop(module_name, None)

    settings_module = importlib.import_module("backend.services.settings")
    settings_module.get_settings.cache_clear()

    return {
        "adapter": importlib.import_module("automl.lightautoml_backend.adapter"),
        "training_service": importlib.import_module("backend.services.training_service"),
        "prediction_service": importlib.import_module("backend.services.prediction_service"),
        "explanation_service": importlib.import_module("backend.services.explanation_service"),
    }


@pytest.mark.skipif(not LIGHTAUTOML_AVAILABLE, reason="LightAutoML is not available")
def test_lightautoml_adapter_trains_real_model_and_predicts(isolated_modules):
    adapter_module = isolated_modules["adapter"]
    prediction_module = isolated_modules["prediction_service"]

    frame = pd.DataFrame(build_binary_training_records())
    adapter = adapter_module.TabularAutoMLAdapter()
    result = adapter.train(
        frame=frame,
        target="target",
        training_options={
            "task_type": "binary",
            "backend": "lightautoml",
            "preset": "tabular",
            "algos": ["linear_l2"],
            "timeout_seconds": 5,
            "cpu_limit": 1,
            "test_size": 0.2,
            "cv_folds": 2,
            "enable_forecast": False,
        },
    )

    predictor = prediction_module.PredictionService()
    predictions = predictor.predict_with_bundle(bundle=result.bundle, frame=frame.drop(columns=["target"]).head(5))

    assert result.backend_name == "lightautoml"
    assert result.bundle["backend_name"] == "lightautoml"
    assert result.training_options["effective"]["backend"] == "lightautoml"
    assert result.training_options["effective"]["algos"] == ["linear_l2"]
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert len(predictions) == 5
    assert {item["prediction"] for item in predictions} <= {0, 1}
    assert all(
        item["confidence"] is None or 0.0 <= float(item["confidence"]) <= 1.0
        for item in predictions
    )


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP is not available")
def test_explanation_service_uses_real_shap_kernel_path(isolated_modules):
    training_module = isolated_modules["training_service"]
    explanation_module = isolated_modules["explanation_service"]

    training_service = training_module.TrainingService()
    explanation_service = explanation_module.ExplanationService()
    training_service.train(
        project_id="shap-real-demo",
        target="target",
        records=build_binary_training_records(rows=24),
        training_options={
            "task_type": "binary",
            "backend": "sklearn",
            "enable_forecast": False,
        },
    )

    explanation = explanation_service.explain(
        "shap-real-demo",
        [
            {"temperature": 64, "vibration": 0.8, "pressure": 108},
            {"temperature": 56, "vibration": 0.22, "pressure": 84},
        ],
    )

    top_factors = [
        factor
        for item in explanation["items"]
        for factor in item["top_factors"]
    ]

    assert explanation["method"] == "shap-kernel"
    assert len(explanation["items"]) == 2
    assert top_factors
    assert all("shap_value" in factor for factor in top_factors)
    assert all(factor["impact_score"] >= 0 for factor in top_factors)


@pytest.mark.skipif(not EXPERTA_AVAILABLE, reason="Experta is not available")
def test_decision_engine_uses_experta_rule_to_escalate_medium_risk():
    engine_module = importlib.import_module("dss.experta_engine.engine")
    recommendation = engine_module.DecisionEngine().recommend(
        {
            "risk_level": "medium",
            "prediction": 1,
            "confidence": 0.6,
            "strong_positive_factors": ["temperature"],
            "medium_positive_factors": [],
            "negative_factors": [],
        }
    )

    assert recommendation["risk_level"] == "medium"
    assert recommendation["summary"] == "High-risk case requires immediate mitigation and manual review."
    assert recommendation["actions"]
    assert "Сильные положительные факторы" in recommendation["rationale"][0]
