from __future__ import annotations

from typing import Any


POSITIVE_LABELS = {"1", "true", "yes", "high", "risk", "failure", "positive", "anomaly"}


def build_facts(
    task_type: str,
    prediction: Any,
    confidence: float | None,
    top_factors: list[dict[str, Any]],
) -> dict[str, Any]:
    prediction_token = str(prediction).strip().lower()
    score = confidence if confidence is not None else 0.5

    if task_type != "regression" and prediction_token not in POSITIVE_LABELS:
        score = 1 - score

    if score >= 0.75:
        risk_level = "high"
    elif score >= 0.45:
        risk_level = "medium"
    else:
        risk_level = "low"

    strong_positive = [
        factor["feature"]
        for factor in top_factors
        if factor["direction"] == "positive" and factor["strength"] == "strong"
    ]
    medium_positive = [
        factor["feature"]
        for factor in top_factors
        if factor["direction"] == "positive" and factor["strength"] == "medium"
    ]
    negative_factors = [
        factor["feature"] for factor in top_factors if factor["direction"] == "negative"
    ]

    return {
        "risk_level": risk_level,
        "prediction": prediction,
        "confidence": confidence,
        "strong_positive_factors": strong_positive,
        "medium_positive_factors": medium_positive,
        "negative_factors": negative_factors,
    }
