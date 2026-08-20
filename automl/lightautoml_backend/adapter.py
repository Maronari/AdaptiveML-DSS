from __future__ import annotations

from dataclasses import dataclass, field
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
from automl.training.forecasting import build_forecasting_bundle
from automl.training.preprocessing import TabularPreprocessor
from backend.services.dataset_service import DatasetService
from backend.services.settings import get_settings
from backend.utils.compat import patch_numpy_for_lightautoml
from backend.utils.io import dataframe_to_records, pythonize


patch_numpy_for_lightautoml()

try:
    import lightautoml  # type: ignore  # noqa: F401
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task

    LIGHTAUTOML_AVAILABLE = True
except Exception:
    LIGHTAUTOML_AVAILABLE = False
    TabularAutoML = None
    Task = None

if LIGHTAUTOML_AVAILABLE:
    try:
        from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML

        TABULAR_UTILIZED_AVAILABLE = True
    except Exception:
        TABULAR_UTILIZED_AVAILABLE = False
        TabularUtilizedAutoML = None
else:
    TABULAR_UTILIZED_AVAILABLE = False
    TabularUtilizedAutoML = None


SUPPORTED_TASK_TYPES = {"auto", "binary", "multiclass", "regression"}
SUPPORTED_BACKENDS = {"auto", "lightautoml", "sklearn"}
SUPPORTED_PRESETS = {"tabular", "utilized"}
SUPPORTED_LIGHTAUTOML_ALGOS = {"lgb", "cb", "xgb", "rf", "linear_l2", "nn"}


def _default_lama_algos() -> list[str]:
    return ["lgb", "linear_l2"]


@dataclass(slots=True)
class TrainingOptions:
    task_type: str = "auto"
    backend: str = "auto"
    preset: str = "tabular"
    algos: list[str] = field(default_factory=_default_lama_algos)
    timeout_seconds: int = 30
    cpu_limit: int = 1
    test_size: float = 0.2
    cv_folds: int = 0
    enable_forecast: bool = True

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None = None) -> "TrainingOptions":
        payload = payload or {}
        task_type = str(payload.get("task_type", "auto")).strip().lower() or "auto"
        backend = str(payload.get("backend", "auto")).strip().lower() or "auto"
        preset = str(payload.get("preset", "tabular")).strip().lower() or "tabular"
        timeout_seconds = int(payload.get("timeout_seconds", 30))
        cpu_limit = int(payload.get("cpu_limit", 1))
        test_size = float(payload.get("test_size", 0.2))
        cv_folds = int(payload.get("cv_folds", 0))
        enable_forecast = bool(payload.get("enable_forecast", True))
        algos = cls._normalize_algos(payload.get("algos"))

        if task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"Unsupported task_type '{task_type}'. Choose one of: {', '.join(sorted(SUPPORTED_TASK_TYPES))}."
            )
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. Choose one of: {', '.join(sorted(SUPPORTED_BACKENDS))}."
            )
        if preset not in SUPPORTED_PRESETS:
            raise ValueError(
                f"Unsupported preset '{preset}'. Choose one of: {', '.join(sorted(SUPPORTED_PRESETS))}."
            )
        unknown_algos = [algo for algo in algos if algo not in SUPPORTED_LIGHTAUTOML_ALGOS]
        if unknown_algos:
            raise ValueError(
                f"Unsupported LightAutoML algorithms: {', '.join(unknown_algos)}. "
                f"Choose from: {', '.join(sorted(SUPPORTED_LIGHTAUTOML_ALGOS))}."
            )
        if timeout_seconds < 5:
            raise ValueError("timeout_seconds must be at least 5.")
        if cpu_limit < 1:
            raise ValueError("cpu_limit must be at least 1.")
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be greater than 0 and less than 1.")
        if cv_folds == 1 or cv_folds < 0:
            raise ValueError("cv_folds must be 0 for auto or at least 2.")

        return cls(
            task_type=task_type,
            backend=backend,
            preset=preset,
            algos=algos,
            timeout_seconds=timeout_seconds,
            cpu_limit=cpu_limit,
            test_size=test_size,
            cv_folds=cv_folds,
            enable_forecast=enable_forecast,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "backend": self.backend,
            "preset": self.preset,
            "algos": list(self.algos),
            "timeout_seconds": self.timeout_seconds,
            "cpu_limit": self.cpu_limit,
            "test_size": self.test_size,
            "cv_folds": self.cv_folds,
            "enable_forecast": self.enable_forecast,
        }

    @staticmethod
    def _normalize_algos(raw_algos: Any) -> list[str]:
        if raw_algos is None:
            return _default_lama_algos()
        if isinstance(raw_algos, str):
            values = raw_algos.split(",")
        else:
            values = list(raw_algos)

        normalized: list[str] = []
        for value in values:
            algo = str(value).strip().lower()
            if algo and algo not in normalized:
                normalized.append(algo)
        return normalized or _default_lama_algos()


@dataclass(slots=True)
class TrainingResult:
    """Normalized training output returned by any backend path."""

    task_type: str
    feature_names: list[str]
    metrics: dict[str, float]
    bundle: dict[str, Any]
    backend_name: str
    training_options: dict[str, Any]
    warnings: list[str]
    holdout_predictions: list[dict[str, Any]] = field(default_factory=list)
    training_artifacts: dict[str, Any] = field(default_factory=dict)


class TabularAutoMLAdapter:
    def __init__(self) -> None:
        """Create a training adapter with project-wide settings."""
        self.settings = get_settings()

    def train(
        self,
        frame: pd.DataFrame,
        target: str,
        training_options: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """Train with the selected backend and return a normalized bundle."""
        options = TrainingOptions.from_payload(training_options)
        task_type = self._resolve_task_type(frame[target], options.task_type)
        requested_options = options.to_dict()
        warnings: list[str] = []

        if options.backend == "sklearn":
            warnings.extend(self._sklearn_ignored_option_warnings(options))
            result = self._train_with_sklearn(
                frame=frame,
                target=target,
                task_type=task_type,
                options=options,
            )
            return self._finalize_training_result(result, requested_options, warnings)

        if options.backend == "lightautoml":
            if not LIGHTAUTOML_AVAILABLE:
                raise ValueError("LightAutoML backend is not available in this environment.")
            try:
                result = self._train_with_lightautoml(
                    frame=frame,
                    target=target,
                    task_type=task_type,
                    options=options,
                )
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f"LightAutoML training failed: {exc}") from exc
            return self._finalize_training_result(result, requested_options, warnings)

        if LIGHTAUTOML_AVAILABLE:
            try:
                result = self._train_with_lightautoml(
                    frame=frame,
                    target=target,
                    task_type=task_type,
                    options=options,
                )
                return self._finalize_training_result(result, requested_options, warnings)
            except Exception as exc:
                warnings.append(f"LightAutoML training failed and sklearn fallback was used: {exc}")

        warnings.extend(self._sklearn_ignored_option_warnings(options))
        result = self._train_with_sklearn(
            frame=frame,
            target=target,
            task_type=task_type,
            options=options,
        )
        return self._finalize_training_result(result, requested_options, warnings)

    def _train_with_lightautoml(
        self,
        frame: pd.DataFrame,
        target: str,
        task_type: str,
        options: TrainingOptions,
    ) -> TrainingResult:
        """Train a LightAutoML preset and build a reusable bundle."""
        if options.preset == "utilized" and not TABULAR_UTILIZED_AVAILABLE:
            raise ValueError("TabularUtilizedAutoML preset is not available in this environment.")

        preprocessor = TabularPreprocessor()
        prepared = preprocessor.fit_transform(frame, target=target)
        forecasting_bundle = self._build_forecasting_bundle(
            frame=frame,
            target=target,
            task_type=task_type,
            options=options,
        )
        feature_names = [column for column in prepared.columns if column != target]

        x = prepared[feature_names].copy()
        y = prepared[target].copy()
        numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_features = [column for column in feature_names if column not in numeric_features]

        stratify = y if task_type in {"binary", "multiclass"} and y.nunique() > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=options.test_size,
            random_state=self.settings.random_state,
            stratify=stratify,
        )

        train_frame = x_train.copy()
        train_frame[target] = y_train.to_numpy()
        cv_folds = options.cv_folds if options.cv_folds > 0 else self._lightautoml_cv_folds(y_train, task_type)
        automl_task = Task("reg") if task_type == "regression" else Task(task_type)
        automl_class = TabularUtilizedAutoML if options.preset == "utilized" else TabularAutoML
        automl = automl_class(
            task=automl_task,
            timeout=options.timeout_seconds,
            cpu_limit=options.cpu_limit,
            general_params={"use_algos": [options.algos]},
            reader_params={"advanced_roles": False, "cv": cv_folds},
        )
        automl.fit_predict(train_frame, roles={"target": target}, verbose=0)
        fitted_algos = self._extract_fitted_composition(automl)
        model_composition = (
            {"source": "fitted", "algos": fitted_algos}
            if fitted_algos
            else {"source": "requested", "algos": list(options.algos)}
        )
        model_leaderboard = self._extract_blend_leaderboard(automl)

        raw_predictions = automl.predict(x_test).data
        predictions, class_mapping = self._decode_lightautoml_predictions(
            automl=automl,
            task_type=task_type,
            raw_predictions=raw_predictions,
            fallback_class_mapping=self._class_mapping(frame[target]),
        )
        metrics = evaluate_predictions(
            task_type=task_type,
            y_true=y_test,
            y_pred=predictions,
            parameter_count=len(feature_names) + 1 if task_type == "regression" else None,
        )
        holdout_predictions = self._build_holdout_predictions(
            x_holdout=x_test,
            y_holdout=y_test,
            predictions=predictions,
            confidences=self._lightautoml_confidences(task_type=task_type, raw_predictions=raw_predictions),
            task_type=task_type,
        )

        effective_options = self._effective_options(
            options=options,
            task_type=task_type,
            backend="lightautoml",
            preset=options.preset,
            algos=options.algos,
            cv_folds=cv_folds,
            enable_forecast=forecasting_bundle is not None,
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
            "preset_name": options.preset,
            "class_mapping": class_mapping,
            "preprocessor": preprocessor,
            "forecasting": forecasting_bundle,
            "model_composition": model_composition,
            "model_leaderboard": model_leaderboard,
        }

        return TrainingResult(
            task_type=task_type,
            feature_names=feature_names,
            metrics=metrics,
            bundle=bundle,
            backend_name="lightautoml",
            training_options={"effective": effective_options},
            warnings=[],
            holdout_predictions=holdout_predictions,
        )

    def _train_with_sklearn(
        self,
        frame: pd.DataFrame,
        target: str,
        task_type: str,
        options: TrainingOptions,
    ) -> TrainingResult:
        """Train the sklearn fallback path with the same external contract."""
        tabular_preprocessor = TabularPreprocessor()
        prepared = tabular_preprocessor.fit_transform(frame, target=target)
        forecasting_bundle = self._build_forecasting_bundle(
            frame=frame,
            target=target,
            task_type=task_type,
            options=options,
        )
        feature_names = [column for column in prepared.columns if column != target]

        x = prepared[feature_names].copy()
        y = prepared[target].copy()
        numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_features = [column for column in feature_names if column not in numeric_features]

        stratify = y if task_type in {"binary", "multiclass"} and y.nunique() > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=options.test_size,
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

        estimator_kwargs = {
            "n_estimators": 300,
            "random_state": self.settings.random_state,
            "n_jobs": options.cpu_limit,
        }
        if task_type == "regression":
            estimator = RandomForestRegressor(**estimator_kwargs)
        else:
            estimator = RandomForestClassifier(**estimator_kwargs)

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
            parameter_count=len(feature_names) + 1 if task_type == "regression" else None,
        )
        holdout_predictions = self._build_holdout_predictions(
            x_holdout=x_test,
            y_holdout=y_test,
            predictions=predictions,
            confidences=self._predict_confidence(pipeline, x_test),
            task_type=task_type,
        )
        feature_importances = self._estimate_feature_importance(
            model=pipeline,
            task_type=task_type,
            x_reference=x_test,
            y_reference=y_test,
            feature_names=feature_names,
        )
        effective_options = self._effective_options(
            options=options,
            task_type=task_type,
            backend="sklearn",
            preset="sklearn-random-forest",
            algos=["rf"],
            cv_folds=None,
            enable_forecast=forecasting_bundle is not None,
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
            "preset_name": "sklearn-random-forest",
            "class_mapping": self._class_mapping(y),
            "preprocessor": tabular_preprocessor,
            "forecasting": forecasting_bundle,
            "model_composition": {"source": "unavailable", "algos": None},
            "model_leaderboard": None,
        }

        return TrainingResult(
            task_type=task_type,
            feature_names=feature_names,
            metrics=metrics,
            bundle=bundle,
            backend_name="sklearn-fallback",
            training_options={"effective": effective_options},
            warnings=[],
            holdout_predictions=holdout_predictions,
        )

    def _build_forecasting_bundle(
        self,
        frame: pd.DataFrame,
        target: str,
        task_type: str,
        options: TrainingOptions,
    ) -> dict[str, Any] | None:
        """Train an auxiliary forecasting head for temporal regression datasets."""
        if task_type != "regression" or not options.enable_forecast:
            return None
        return build_forecasting_bundle(
            frame=frame,
            target=target,
            random_state=self.settings.random_state,
        )

    @staticmethod
    def _resolve_task_type(target: pd.Series, requested_task_type: str) -> str:
        """Validate explicit task selection against the target column."""
        if requested_task_type == "auto":
            return DatasetService.infer_task_type(target)

        cleaned = target.dropna()
        unique_count = cleaned.nunique()
        if requested_task_type == "binary":
            if unique_count != 2:
                raise ValueError("Binary task requires exactly two distinct target values.")
            return "binary"

        if requested_task_type == "multiclass":
            if unique_count < 3:
                raise ValueError("Multiclass task requires at least three distinct target values.")
            return "multiclass"

        if not pd.api.types.is_numeric_dtype(cleaned):
            raise ValueError("Regression task requires a numeric target column.")
        return "regression"

    @staticmethod
    def _sklearn_ignored_option_warnings(options: TrainingOptions) -> list[str]:
        """Warn when user-selected LightAutoML options do not apply to sklearn."""
        ignored = []
        if options.preset != "tabular":
            ignored.append("preset")
        if options.algos != _default_lama_algos():
            ignored.append("algos")
        if options.cv_folds > 0:
            ignored.append("cv_folds")
        if options.timeout_seconds != 30:
            ignored.append("timeout_seconds")
        if not ignored:
            return []
        return [
            "The selected sklearn backend ignores these LightAutoML-only options: "
            + ", ".join(ignored)
            + "."
        ]

    @staticmethod
    def _effective_options(
        options: TrainingOptions,
        task_type: str,
        backend: str,
        preset: str,
        algos: list[str],
        cv_folds: int | None,
        enable_forecast: bool,
    ) -> dict[str, Any]:
        """Return the actual training options used by the fitted model."""
        return {
            "task_type": task_type,
            "backend": backend,
            "preset": preset,
            "algos": list(algos),
            "timeout_seconds": options.timeout_seconds if backend == "lightautoml" else None,
            "cpu_limit": options.cpu_limit,
            "test_size": options.test_size,
            "cv_folds": cv_folds,
            "enable_forecast": enable_forecast,
        }

    @staticmethod
    def _finalize_training_result(
        result: TrainingResult,
        requested_options: dict[str, Any],
        warnings: list[str],
    ) -> TrainingResult:
        """Attach requested/effective options and warnings to the model bundle."""
        result.training_options = {
            "requested": requested_options,
            "effective": result.training_options["effective"],
        }
        result.warnings.extend(warnings)
        result.bundle["training_options"] = result.training_options
        result.bundle["warnings"] = list(result.warnings)
        result.training_artifacts = TabularAutoMLAdapter._build_training_artifacts(
            bundle=result.bundle,
            backend_name=result.backend_name,
            training_options=result.training_options,
            warnings=result.warnings,
            holdout_predictions=result.holdout_predictions,
        )
        return result

    @staticmethod
    def _build_holdout_predictions(
        x_holdout: pd.DataFrame,
        y_holdout: pd.Series,
        predictions,
        confidences,
        task_type: str,
    ) -> list[dict[str, Any]]:
        """Serialize holdout rows together with actual and predicted outputs."""
        holdout_rows = dataframe_to_records(x_holdout.reset_index(drop=True))
        actual_values = [pythonize(value) for value in y_holdout.reset_index(drop=True).tolist()]
        predicted_values = [pythonize(value) for value in np.asarray(predictions).reshape(-1).tolist()]
        confidence_values = None
        if confidences is not None:
            confidence_values = [pythonize(value) for value in np.asarray(confidences).reshape(-1).tolist()]

        items: list[dict[str, Any]] = []
        for index, record in enumerate(holdout_rows):
            prediction = predicted_values[index]
            actual = actual_values[index]
            item = {
                "row_index": index,
                "record": record,
                "actual": actual,
                "prediction": prediction,
                "confidence": confidence_values[index] if confidence_values is not None else None,
            }
            if task_type == "regression":
                item["error"] = round(float(prediction) - float(actual), 6)
            else:
                item["matched"] = prediction == actual
            items.append(item)
        return items

    @staticmethod
    def _predict_confidence(model, frame):
        """Return max class probability when the estimator exposes predict_proba."""
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(frame)
            return probabilities.max(axis=1)
        return None

    @staticmethod
    def _build_training_artifacts(
        bundle: dict[str, Any],
        backend_name: str,
        training_options: dict[str, Any],
        warnings: list[str],
        holdout_predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Persist lightweight training diagnostics next to registry metadata."""
        forecasting = bundle.get("forecasting") or {}
        model_composition = bundle.get("model_composition") or {"source": "unavailable", "algos": None}
        return {
            "backend": backend_name,
            "preset": bundle.get("preset_name"),
            "numeric_features": list(bundle.get("numeric_features") or []),
            "categorical_features": list(bundle.get("categorical_features") or []),
            "feature_importances": dict(bundle.get("feature_importances") or {}),
            "baseline_profile": dict(bundle.get("baseline_profile") or {}),
            "training_sample": list(bundle.get("training_sample") or []),
            "class_mapping": dict(bundle.get("class_mapping") or {}),
            "training_options": training_options,
            "model_composition": {
                "source": model_composition.get("source", "unavailable"),
                "algos": model_composition.get("algos"),
            },
            "model_leaderboard": bundle.get("model_leaderboard"),
            "warnings": list(warnings),
            "holdout_rows": len(holdout_predictions),
            "forecasting": {
                "available": bool(forecasting.get("enabled")),
                "default_horizon_minutes": forecasting.get("default_horizon_minutes"),
                "base_frequency_minutes": forecasting.get("base_frequency_minutes"),
                "forecast_model": forecasting.get("forecasting_model"),
                "metrics": forecasting.get("metrics"),
            },
        }

    @staticmethod
    def _lightautoml_confidences(task_type: str, raw_predictions):
        """Normalize LightAutoML raw outputs into per-row confidence scores."""
        data = np.asarray(raw_predictions)
        if task_type == "regression":
            return None
        if task_type == "binary":
            probabilities = data.reshape(-1)
            return np.maximum(probabilities, 1.0 - probabilities)
        return data.max(axis=1)

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
    def _class_mapping(target: pd.Series) -> dict[Any, int] | None:
        """Return a deterministic label-to-index mapping for classification."""
        if target.nunique(dropna=True) < 2:
            return None
        unique = sorted(target.dropna().unique().tolist(), key=lambda value: str(value))
        return {label: index for index, label in enumerate(unique)}

    @staticmethod
    def _decode_lightautoml_predictions(
        automl,
        task_type: str,
        raw_predictions,
        fallback_class_mapping: dict[str, int] | None = None,
    ):
        """Decode raw LightAutoML outputs into plain predictions and labels."""
        data = np.asarray(raw_predictions)
        if task_type == "regression":
            return data.reshape(-1), None

        reader = getattr(automl, "reader", None)
        class_mapping = getattr(reader, "class_mapping", None) if reader is not None else None
        if class_mapping is None:
            class_mapping = fallback_class_mapping
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

    _ALGO_NAME_FRAGMENTS = (
        ("lightgbm", "lgb"),
        ("catboost", "cb"),
        ("xgb", "xgb"),
        ("rfsklearn", "rf"),
        ("randomforest", "rf"),
        ("linear", "linear_l2"),
        ("torch", "nn"),
        ("neural", "nn"),
        ("nn", "nn"),
    )

    @classmethod
    def _normalize_algo_name(cls, raw_name: str) -> str:
        """Map a raw LightAutoML model/algo class name to the app's short algo vocabulary."""
        lowered = raw_name.lower()
        for fragment, code in cls._ALGO_NAME_FRAGMENTS:
            if fragment in lowered:
                return code
        return raw_name

    @classmethod
    def _extract_fitted_composition(cls, automl) -> list[str] | None:
        """Introspect a fitted LightAutoML object for the algorithms it actually used.

        LAMA's ``TabularAutoML``/``TabularUtilizedAutoML`` may drop or down-weight
        algorithms that were requested via ``use_algos`` if they don't fit the time
        budget. This walks the fitted pipeline structure (``automl.levels`` -> list of
        ``MLPipeline`` -> ``.ml_algos``) to recover the algorithms genuinely present in
        the fitted ensemble. Entirely defensive: any failure (API differences across
        LightAutoML versions, unexpected structure, etc.) simply yields ``None`` so the
        caller can fall back to the requested algo list.
        """
        try:
            # TabularUtilizedAutoML wraps several inner TabularAutoML instances; try to
            # discover them, but always fall back to treating `automl` itself as the
            # (only) instance to introspect.
            inner_automls = []
            for attr_name in ("outer_pipes", "inner_automls", "automls"):
                candidates = getattr(automl, attr_name, None)
                if not candidates:
                    continue
                for candidate in candidates:
                    inner = getattr(candidate, "automl", None) or getattr(candidate, "pipeline", None) or candidate
                    if inner is not None and hasattr(inner, "levels"):
                        inner_automls.append(inner)
                if inner_automls:
                    break
            if not inner_automls:
                inner_automls = [automl]

            found: list[str] = []
            for inner in inner_automls:
                levels = getattr(inner, "levels", None) or []
                for level in levels:
                    for ml_pipeline in level or []:
                        ml_algos = getattr(ml_pipeline, "ml_algos", None) or []
                        for algo in ml_algos:
                            raw_name = getattr(algo, "name", None) or type(algo).__name__
                            normalized = cls._normalize_algo_name(str(raw_name))
                            if normalized not in found:
                                found.append(normalized)

            return found or None
        except Exception:
            return None

    @classmethod
    def _extract_blend_leaderboard(cls, automl) -> list[dict] | None:
        """Extract the final blend's winning algorithm and per-algorithm comparison.

        LAMA's own ``TabularAutoML.create_model_str_desc()`` builds a human-readable
        string from ``automl.collect_model_stats()`` (model name -> averaged fold count)
        paired with ``automl.blender.wts`` (the ``WeightedBlender`` weight for each model
        on the final level). This replicates that same pairing to get structured data
        instead of a formatted string, so the report can show which algorithm actually
        "won" (highest blend weight) and how the others compare. Entirely defensive:
        LightAutoML internals differ across versions/presets (e.g. ``TabularUtilizedAutoML``
        may not expose ``collect_model_stats``/``blender`` directly), so any failure
        simply yields ``None`` and the caller falls back to the plain composition list.
        """
        try:
            model_stats = sorted(automl.collect_model_stats().items())
            if not model_stats:
                return None

            last_level = model_stats[-1][0].split("_")[1]
            last_level_models = [stat for stat in model_stats if stat[0].startswith(f"Lvl_{last_level}")]

            weights = getattr(getattr(automl, "blender", None), "wts", None)
            if weights is None or len(weights) != len(last_level_models):
                weights = [None] * len(last_level_models)

            leaderboard = []
            for (model_name, folds), weight in zip(last_level_models, weights):
                raw_algo = model_name.rsplit("_", 1)[-1]
                leaderboard.append(
                    {
                        "model_name": model_name,
                        "algo": cls._normalize_algo_name(raw_algo),
                        "folds": int(folds),
                        "weight": round(float(weight), 6) if weight is not None else None,
                    }
                )

            leaderboard.sort(key=lambda entry: (entry["weight"] is None, -(entry["weight"] or 0.0)))
            return leaderboard or None
        except Exception:
            return None

    @staticmethod
    def _lightautoml_cv_folds(target: pd.Series, task_type: str) -> int:
        """Choose a safe CV value for LightAutoML on small datasets."""
        if task_type == "regression":
            return max(2, min(5, len(target)))

        class_counts = target.value_counts(dropna=True)
        if class_counts.empty:
            return 2
        return max(2, min(5, int(class_counts.min())))
