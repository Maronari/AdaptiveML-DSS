from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_predictions(task_type: str, y_true, y_pred) -> dict[str, float]:
    """Compute task-specific evaluation metrics with rounded values."""
    if task_type == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "r2": round(r2, 6),
        }

    accuracy = float(accuracy_score(y_true, y_pred))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))
    return {
        "accuracy": round(accuracy, 6),
        "f1_weighted": round(f1_weighted, 6),
    }


def primary_metric_name(task_type: str) -> str:
    """Return the metric used for model promotion decisions."""
    return "rmse" if task_type == "regression" else "f1_weighted"


def scoring_function(task_type: str) -> Callable:
    """Build a scorer compatible with permutation-based importance estimation."""
    def score(model, x_reference, y_reference) -> float:
        predictions = model.predict(x_reference)
        metrics = evaluate_predictions(task_type=task_type, y_true=y_reference, y_pred=predictions)
        return float(metrics[primary_metric_name(task_type)])

    return score
