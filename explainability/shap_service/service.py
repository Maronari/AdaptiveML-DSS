from __future__ import annotations

import numpy as np
import pandas as pd

from backend.utils.io import pythonize


try:
    import shap  # type: ignore  # noqa: F401

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def try_shap_explanations(bundle: dict, frame):
    """Build row-level SHAP explanations when the environment supports it."""
    if not SHAP_AVAILABLE:
        return None

    task_type = bundle.get("task_type")
    if task_type == "multiclass":
        return None

    training_sample = bundle.get("training_sample")
    if not training_sample:
        return None

    background = pd.DataFrame.from_records(training_sample)
    if background.empty:
        return None
    background = background[frame.columns].head(min(20, len(background)))

    predict_fn = _predict_fn(
        bundle=bundle,
        task_type=task_type,
        columns=frame.columns.tolist(),
    )
    try:
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(frame, nsamples=min(100, 2 * len(frame.columns) + 20))
    except Exception:
        return None

    values = np.asarray(shap_values)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    items = []
    for row_index, (_, row) in enumerate(frame.iterrows()):
        row_values = values[row_index]
        top_indices = np.argsort(np.abs(row_values))[::-1][:5]
        top_factors = []
        for feature_index in top_indices:
            feature = frame.columns[feature_index]
            impact = float(row_values[feature_index])
            if impact == 0:
                continue
            top_factors.append(
                {
                    "feature": feature,
                    "value": pythonize(row.iloc[feature_index]),
                    "baseline": pythonize(background[feature].iloc[0]),
                    "direction": "positive" if impact >= 0 else "negative",
                    "strength": _strength(abs(impact)),
                    "impact_score": round(abs(impact), 6),
                    "shap_value": round(impact, 6),
                }
            )
        items.append({"row_index": row_index, "top_factors": top_factors})

    return {"method": "shap-kernel", "items": items}


def _predict_fn(bundle: dict, task_type: str, columns: list[str]):
    """Create a SHAP-compatible prediction callable over model bundle outputs."""
    model = bundle["pipeline"]
    backend_name = bundle.get("backend_name", "sklearn-fallback")

    def predict(data):
        """Predict on a raw matrix provided by SHAP."""
        frame = pd.DataFrame(data, columns=columns)
        if backend_name == "lightautoml":
            output = np.asarray(model.predict(frame).data)
            return output.reshape(-1)

        if task_type == "regression":
            return np.asarray(model.predict(frame)).reshape(-1)

        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(frame))
            if probabilities.ndim == 2 and probabilities.shape[1] > 1:
                return probabilities[:, 1]
            return probabilities.reshape(-1)

        return np.asarray(model.predict(frame)).reshape(-1)

    return predict


def _strength(score: float) -> str:
    """Map a numeric contribution magnitude to a human-readable label."""
    if score >= 0.66:
        return "strong"
    if score >= 0.33:
        return "medium"
    return "weak"
