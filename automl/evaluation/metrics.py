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


def _safe_round(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 6)


def _regression_information_criteria(
    residuals: np.ndarray,
    sample_size: int,
    parameter_count: int,
) -> dict[str, float]:
    """Estimate information criteria under a Gaussian residual assumption."""
    if sample_size <= 0:
        return {}

    safe_sample_size = max(int(sample_size), 1)
    safe_parameter_count = max(int(parameter_count), 1)
    rss = float(np.sum(np.square(residuals)))
    sigma2 = max(rss / safe_sample_size, np.finfo(float).eps)

    aic = safe_sample_size * float(np.log(sigma2)) + 2.0 * safe_parameter_count
    bic = safe_sample_size * float(np.log(sigma2)) + safe_parameter_count * float(np.log(safe_sample_size))
    metrics = {
        "aic": round(aic, 6),
        "bic": round(bic, 6),
    }

    correction_denominator = safe_sample_size - safe_parameter_count - 1
    if correction_denominator > 0:
        aicc = aic + (2.0 * safe_parameter_count * (safe_parameter_count + 1)) / correction_denominator
        metrics["aicc"] = round(aicc, 6)

    return metrics


def evaluate_predictions(task_type: str, y_true, y_pred, parameter_count: int | None = None) -> dict[str, float]:
    """Compute task-specific evaluation metrics with rounded values."""
    if task_type == "regression":
        y_true_array = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred_array = np.asarray(y_pred, dtype=float).reshape(-1)
        sample_size = int(min(len(y_true_array), len(y_pred_array)))
        if sample_size == 0:
            return {}

        y_true_array = y_true_array[:sample_size]
        y_pred_array = y_pred_array[:sample_size]
        residuals = y_true_array - y_pred_array
        mse = float(mean_squared_error(y_true_array, y_pred_array))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true_array, y_pred_array))
        r2 = float(r2_score(y_true_array, y_pred_array))

        correlation = None
        if sample_size >= 2 and np.std(y_true_array) > 0 and np.std(y_pred_array) > 0:
            correlation = float(np.corrcoef(y_true_array, y_pred_array)[0, 1])

        metrics = {
            "r": _safe_round(correlation),
            "mse": round(mse, 6),
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "r2": round(r2, 6),
        }
        metrics.update(
            _regression_information_criteria(
                residuals=residuals,
                sample_size=sample_size,
                parameter_count=parameter_count if parameter_count is not None else 1,
            )
        )
        return {key: value for key, value in metrics.items() if value is not None}

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
