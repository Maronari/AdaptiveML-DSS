from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from automl.evaluation.metrics import evaluate_predictions, primary_metric_name, scoring_function
from automl.training.preprocessing import TabularPreprocessor
from backend.services.dataset_service import DatasetService
from backend.services.settings import get_settings
from backend.utils.compat import patch_numpy_for_lightautoml


try:
    patch_numpy_for_lightautoml()
    import lightautoml  # type: ignore  # noqa: F401
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task

    LIGHTAUTOML_AVAILABLE = True
except Exception:
    LIGHTAUTOML_AVAILABLE = False


@dataclass(slots=True)
class TrainingResult:
    """Normalized training output returned by any backend path."""
    task_type: str
    feature_names: list[str]
    metrics: dict[str, float]
    bundle: dict[str, Any]
    backend_name: str


class TabularAutoMLAdapter:
    def __init__(self) -> None:
        """Create a training adapter with project-wide settings."""
        self.settings = get_settings()

    def train(self, frame: pd.DataFrame, target: str) -> TrainingResult:
        """Train with LightAutoML first and fallback to sklearn on failure."""
        if LIGHTAUTOML_AVAILABLE:
            try:
                return self._train_with_lightautoml(frame=frame, target=target)
            except Exception:
                pass
        return self._train_with_sklearn(frame=frame, target=target)

    def _train_with_lightautoml(self, frame: pd.DataFrame, target: str) -> TrainingResult:
        """Train a real LightAutoML model and build a reusable bundle."""
        preprocessor = TabularPreprocessor()
        prepared = preprocessor.fit_transform(frame, target=target)
        task_type = DatasetService.infer_task_type(frame[target])
        feature_names = [column for column in prepared.columns if column != target]

        x = prepared[feature_names].copy()
        y = prepared[target].copy()
        numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_features = [column for column in feature_names if column not in numeric_features]

        stratify = y if task_type in {"binary", "multiclass"} and y.nunique() > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=self.settings.random_state,
            stratify=stratify,
        )

        train_frame = x_train.copy()
        train_frame[target] = y_train.to_numpy()
        cv_folds = self._lightautoml_cv_folds(y_train, task_type)
        automl_task = Task("reg") if task_type == "regression" else Task(task_type)
        automl = TabularAutoML(
            task=automl_task,
            timeout=30,
            cpu_limit=1,
            general_params={"use_algos": [["lgb", "linear_l2"]]},
            reader_params={"advanced_roles": False, "cv": cv_folds},
        )
        automl.fit_predict(train_frame, roles={"target": target}, verbose=0)

        raw_predictions = automl.predict(x_test).data
        predictions, class_mapping = self._decode_lightautoml_predictions(
            automl=automl,
            task_type=task_type,
            raw_predictions=raw_predictions,
        )
        metrics = evaluate_predictions(
            task_type=task_type,
            y_true=y_test,
            y_pred=predictions,
        )

        bundle = {
            "pipeline": automl,
            "task_type": task_type,
            "target": target,
            "feature_names": feature_names,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_importances": self._lightautoml_feature_importances(automl, feature_names),
            "baseline_profile": self._baseline_profile(x_train, numeric_features, categorical_features),
            "training_sample": x_train.head(50).to_dict(orient="records"),
            "metrics": metrics,
            "primary_metric": primary_metric_name(task_type),
            "backend_name": "lightautoml",
            "class_mapping": class_mapping,
            "preprocessor": preprocessor,
        }

        return TrainingResult(
            task_type=task_type,
            feature_names=feature_names,
            metrics=metrics,
            bundle=bundle,
            backend_name="lightautoml",
        )

    def _train_with_sklearn(self, frame: pd.DataFrame, target: str) -> TrainingResult:
        """Train the sklearn fallback path with the same external contract."""
        tabular_preprocessor = TabularPreprocessor()
        prepared = tabular_preprocessor.fit_transform(frame, target=target)
        task_type = DatasetService.infer_task_type(frame[target])
        feature_names = [column for column in prepared.columns if column != target]

        x = prepared[feature_names].copy()
        y = prepared[target].copy()
        numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_features = [column for column in feature_names if column not in numeric_features]

        stratify = y if task_type in {"binary", "multiclass"} and y.nunique() > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=self.settings.random_state,
            stratify=stratify,
        )

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        feature_preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ]
        )

        if task_type == "regression":
            estimator = RandomForestRegressor(
                n_estimators=300,
                random_state=self.settings.random_state,
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=300,
                random_state=self.settings.random_state,
            )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", feature_preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)

        predictions = pipeline.predict(x_test)
        metrics = evaluate_predictions(
            task_type=task_type,
            y_true=y_test,
            y_pred=predictions,
        )
        feature_importances = self._estimate_feature_importance(
            model=pipeline,
            task_type=task_type,
            x_reference=x_test,
            y_reference=y_test,
            feature_names=feature_names,
        )

        bundle = {
            "pipeline": pipeline,
            "task_type": task_type,
            "target": target,
            "feature_names": feature_names,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_importances": feature_importances,
            "baseline_profile": self._baseline_profile(x_train, numeric_features, categorical_features),
            "training_sample": x_train.head(50).to_dict(orient="records"),
            "metrics": metrics,
            "primary_metric": primary_metric_name(task_type),
            "backend_name": "sklearn-fallback",
            "class_mapping": self._class_mapping(y),
            "preprocessor": tabular_preprocessor,
        }

        return TrainingResult(
            task_type=task_type,
            feature_names=feature_names,
            metrics=metrics,
            bundle=bundle,
            backend_name=bundle["backend_name"],
        )

    def _estimate_feature_importance(
        self,
        model,
        task_type: str,
        x_reference: pd.DataFrame,
        y_reference: pd.Series,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Estimate fallback feature importance via permutation scoring."""
        scorer = scoring_function(task_type)
        baseline_score = scorer(model, x_reference, y_reference)
        lower_is_better = primary_metric_name(task_type) == "rmse"
        rng = np.random.default_rng(self.settings.random_state)
        importances: dict[str, float] = {}

        for feature in feature_names:
            shuffled = x_reference.copy()
            shuffled[feature] = shuffled[feature].sample(
                frac=1.0,
                random_state=int(rng.integers(0, 1_000_000)),
            ).to_numpy()
            shuffled_score = scorer(model, shuffled, y_reference)
            change = shuffled_score - baseline_score if lower_is_better else baseline_score - shuffled_score
            importances[feature] = max(float(change), 0.0)

        total = sum(importances.values())
        if total == 0:
            uniform_weight = 1.0 / max(len(feature_names), 1)
            return {feature: uniform_weight for feature in feature_names}

        return {feature: round(value / total, 6) for feature, value in importances.items()}

    @staticmethod
    def _baseline_profile(
        frame: pd.DataFrame,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Build numeric means and categorical modes for factor descriptions."""
        numeric = {feature: float(frame[feature].mean()) for feature in numeric_features}
        categorical = {}
        for feature in categorical_features:
            mode = frame[feature].mode(dropna=True)
            categorical[feature] = str(mode.iloc[0]) if not mode.empty else None
        return {"numeric": numeric, "categorical": categorical}

    @staticmethod
    def _class_mapping(target: pd.Series) -> dict[str, int] | None:
        """Return a deterministic label-to-index mapping for classification."""
        if target.nunique(dropna=True) < 2:
            return None
        unique = sorted(target.dropna().unique().tolist(), key=lambda value: str(value))
        return {str(label): index for index, label in enumerate(unique)}

    @staticmethod
    def _decode_lightautoml_predictions(automl, task_type: str, raw_predictions):
        """Decode raw LightAutoML outputs into plain predictions and labels."""
        data = np.asarray(raw_predictions)
        if task_type == "regression":
            return data.reshape(-1), None

        class_mapping = getattr(automl.reader, "class_mapping", None)
        reverse_mapping = None
        if class_mapping:
            reverse_mapping = {int(index): label for label, index in class_mapping.items()}

        if task_type == "binary":
            positive_class = reverse_mapping.get(1, 1) if reverse_mapping else 1
            negative_class = reverse_mapping.get(0, 0) if reverse_mapping else 0
            labels = np.where(data.reshape(-1) >= 0.5, positive_class, negative_class)
            return labels, class_mapping

        indices = data.argmax(axis=1)
        if reverse_mapping:
            labels = np.array([reverse_mapping.get(int(index), int(index)) for index in indices])
        else:
            labels = indices
        return labels, class_mapping

    @staticmethod
    def _lightautoml_feature_importances(automl, feature_names: list[str]) -> dict[str, float]:
        """Extract normalized feature importances from LightAutoML when available."""
        try:
            scores = automl.get_feature_scores()
        except Exception:
            scores = None

        if scores is None or scores.empty:
            uniform = 1.0 / max(len(feature_names), 1)
            return {feature: uniform for feature in feature_names}

        values = {
            str(row["Feature"]): float(row["Importance"])
            for _, row in scores.iterrows()
            if str(row["Feature"]) in feature_names
        }
        total = sum(max(value, 0.0) for value in values.values())
        if total <= 0:
            uniform = 1.0 / max(len(feature_names), 1)
            return {feature: uniform for feature in feature_names}

        return {
            feature: round(max(values.get(feature, 0.0), 0.0) / total, 6)
            for feature in feature_names
        }

    @staticmethod
    def _lightautoml_cv_folds(target: pd.Series, task_type: str) -> int:
        """Choose a safe CV value for LightAutoML on small datasets."""
        if task_type == "regression":
            return max(2, min(5, len(target)))

        class_counts = target.value_counts(dropna=True)
        if class_counts.empty:
            return 2
        return max(2, min(5, int(class_counts.min())))
