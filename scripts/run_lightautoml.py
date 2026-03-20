#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automl.evaluation.metrics import evaluate_predictions
from backend.services.dataset_service import DatasetService
from backend.utils.compat import patch_numpy_for_lightautoml


patch_numpy_for_lightautoml()

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

from automl.training.preprocessing import TabularPreprocessor


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for direct LightAutoML training."""
    parser = argparse.ArgumentParser(
        description="Run LightAutoML directly on a CSV or Excel dataset."
    )
    parser.add_argument("--data", required=True, help="Path to input CSV file.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument(
        "--task",
        choices=["auto", "binary", "multiclass", "regression"],
        default="auto",
        help="Task type. Defaults to auto inference.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Training timeout in seconds.",
    )
    parser.add_argument(
        "--cpu-limit",
        type=int,
        default=1,
        help="CPU limit passed to LightAutoML.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout size for metrics calculation.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        help="Override CV folds for LightAutoML. 0 means auto.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="LightAutoML verbosity.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=10,
        help="How many top feature importances to print.",
    )
    parser.add_argument(
        "--save-model",
        help="Optional path to save the trained LightAutoML object via joblib.",
    )
    parser.add_argument(
        "--predictions-out",
        help="Optional path to save holdout predictions CSV.",
    )
    return parser.parse_args()


def task_name_to_lama(task_name: str) -> Task:
    """Map the project task name to a LightAutoML Task."""
    if task_name == "regression":
        return Task("reg")
    return Task(task_name)


def auto_cv_folds(target: pd.Series, task_type: str) -> int:
    """Pick a safe CV value for small training datasets."""
    if task_type == "regression":
        return max(2, min(5, len(target)))

    class_counts = target.value_counts(dropna=True)
    if class_counts.empty:
        return 2
    return max(2, min(5, int(class_counts.min())))


def decode_predictions(task_type: str, raw_output, class_mapping: dict | None):
    """Convert raw LightAutoML predictions into labels and confidences."""
    import numpy as np

    data = np.asarray(raw_output)
    if task_type == "regression":
        return data.reshape(-1), None

    reverse_mapping = None
    if class_mapping:
        reverse_mapping = {int(index): label for label, index in class_mapping.items()}

    if task_type == "binary":
        probabilities = data.reshape(-1)
        positive_class = reverse_mapping.get(1, 1) if reverse_mapping else 1
        negative_class = reverse_mapping.get(0, 0) if reverse_mapping else 0
        predictions = np.where(probabilities >= 0.5, positive_class, negative_class)
        confidences = np.maximum(probabilities, 1.0 - probabilities)
        return predictions, confidences

    indices = data.argmax(axis=1)
    if reverse_mapping:
        predictions = [reverse_mapping.get(int(index), int(index)) for index in indices]
    else:
        predictions = indices.tolist()
    confidences = data.max(axis=1)
    return predictions, confidences


def feature_scores_payload(automl: TabularAutoML, top_n: int) -> list[dict]:
    """Extract the top feature importances for CLI output."""
    try:
        scores = automl.get_feature_scores()
    except Exception:
        return []

    if scores is None or scores.empty:
        return []

    return [
        {
            "feature": str(row["Feature"]),
            "importance": float(row["Importance"]),
        }
        for _, row in scores.head(top_n).iterrows()
    ]


def main() -> int:
    """Train LightAutoML on a local dataset and print a JSON summary."""
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")

    frame = DatasetService.read_tabular_file(data_path)
    DatasetService._validate_training_frame(frame, args.target)

    task_type = args.task
    if task_type == "auto":
        task_type = DatasetService.infer_task_type(frame[args.target])

    preprocessor = TabularPreprocessor()
    prepared = preprocessor.fit_transform(frame, target=args.target)
    feature_names = [column for column in prepared.columns if column != args.target]
    source_features = [column for column in frame.columns if column != args.target]
    x = prepared[feature_names].copy()
    y = prepared[args.target].copy()
    stratify = y if task_type in {"binary", "multiclass"} and y.nunique() > 1 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=stratify,
    )

    train_frame = x_train.copy()
    train_frame[args.target] = y_train.to_numpy()

    cv_folds = args.cv if args.cv > 0 else auto_cv_folds(y_train, task_type)
    automl = TabularAutoML(
        task=task_name_to_lama(task_type),
        timeout=args.timeout,
        cpu_limit=args.cpu_limit,
        general_params={"use_algos": [["lgb", "linear_l2"]]},
        reader_params={"advanced_roles": False, "cv": cv_folds},
    )
    automl.fit_predict(train_frame, roles={"target": args.target}, verbose=args.verbose)

    raw_predictions = automl.predict(x_test).data
    class_mapping = getattr(automl.reader, "class_mapping", None)
    predictions, confidences = decode_predictions(task_type, raw_predictions, class_mapping)
    metrics = evaluate_predictions(task_type=task_type, y_true=y_test, y_pred=predictions)

    if args.save_model:
        output_path = Path(args.save_model).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": automl,
                "preprocessor": preprocessor,
                "target": args.target,
                "task_type": task_type,
                "feature_names": feature_names,
                "class_mapping": class_mapping,
            },
            output_path,
        )

    if args.predictions_out:
        predictions_frame = x_test.copy()
        predictions_frame["target_true"] = y_test.to_numpy()
        predictions_frame["prediction"] = predictions
        if confidences is not None:
            predictions_frame["confidence"] = confidences
        output_path = Path(args.predictions_out).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_frame.to_csv(output_path, index=False)

    payload = {
        "data": str(data_path),
        "target": args.target,
        "task_type": task_type,
        "rows": int(len(frame)),
        "source_features": source_features,
        "model_features": feature_names,
        "cv_folds": cv_folds,
        "metrics": metrics,
        "class_mapping": class_mapping,
        "top_feature_importances": feature_scores_payload(automl, args.top_features),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
