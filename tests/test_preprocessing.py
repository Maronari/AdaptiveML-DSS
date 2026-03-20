import numpy as np
import pandas as pd

from automl.training.preprocessing import TabularPreprocessor
from backend.services.prediction_service import PredictionService


class DummyBinaryModel:
    def predict(self, frame: pd.DataFrame):
        scores = (frame["temperature"] + frame["Дата__hour"] + frame["sunrise__minutes"]) > 390
        return scores.astype(int).to_numpy()

    def predict_proba(self, frame: pd.DataFrame):
        labels = self.predict(frame)
        probabilities = np.where(labels == 1, 0.85, 0.15)
        return np.column_stack([1.0 - probabilities, probabilities])


def test_tabular_preprocessor_extracts_datetime_and_time_features():
    frame = pd.DataFrame(
        [
            {"Дата": "2026-03-20 08:30:00", "sunrise": "06:15", "temperature": 18.5, "target": 0},
            {"Дата": "2026-03-21 21:10:00", "sunrise": "06:12", "temperature": 22.0, "target": 1},
        ]
    )

    transformed = TabularPreprocessor().fit_transform(frame, target="target")

    assert "Дата" not in transformed.columns
    assert "sunrise" not in transformed.columns
    assert "Дата__ts" in transformed.columns
    assert "Дата__hour" in transformed.columns
    assert "Дата__dayofweek" in transformed.columns
    assert "Дата__month" in transformed.columns
    assert "sunrise__minutes" in transformed.columns
    assert transformed["sunrise__minutes"].tolist() == [375.0, 372.0]
    assert transformed["target"].tolist() == [0, 1]


def test_prediction_service_applies_tabular_preprocessor_before_alignment():
    training_frame = pd.DataFrame(
        [
            {"Дата": "2026-03-20 08:30:00", "sunrise": "06:15", "temperature": 18.5, "target": 0},
            {"Дата": "2026-03-21 21:10:00", "sunrise": "06:12", "temperature": 22.0, "target": 1},
        ]
    )
    preprocessor = TabularPreprocessor().fit(training_frame, target="target")
    feature_names = [
        "temperature",
        "Дата__ts",
        "Дата__hour",
        "Дата__dayofweek",
        "Дата__month",
        "sunrise__minutes",
    ]
    bundle = {
        "pipeline": DummyBinaryModel(),
        "task_type": "binary",
        "backend_name": "sklearn-fallback",
        "feature_names": feature_names,
        "preprocessor": preprocessor,
    }

    raw_inference = pd.DataFrame(
        [
            {"Дата": "2026-03-22 09:00:00", "sunrise": "06:10", "temperature": 25.0},
            {"Дата": "2026-03-22 02:00:00", "sunrise": "06:10", "temperature": 8.0},
        ]
    )

    predictions = PredictionService().predict_with_bundle(bundle=bundle, frame=raw_inference)

    assert len(predictions) == 2
    assert predictions[0]["prediction"] == 1
    assert predictions[1]["prediction"] == 0
    assert predictions[0]["confidence"] == 0.85
    assert predictions[1]["confidence"] == 0.85
