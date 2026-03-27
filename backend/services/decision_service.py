from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from backend.services.dataset_service import DatasetService
from backend.services.explanation_service import ExplanationService
from backend.services.prediction_service import PredictionService
from backend.services.registry_service import RegistryService
from backend.utils.io import dataframe_from_records
from dss.experta_engine.engine import DecisionEngine
from explainability.fact_builder.builder import build_facts


class DecisionService:
    def __init__(self) -> None:
        """Combine predictions and explanations into DSS recommendations."""
        self.dataset_service = DatasetService()
        self.prediction_service = PredictionService()
        self.explanation_service = ExplanationService()
        self.registry_service = RegistryService()
        self.engine = DecisionEngine()

    def evaluate(self, project_id: str, records: list[dict]) -> dict:
        """Run the full decision pipeline for inline records."""
        champion, bundle = self.registry_service.get_champion_bundle(project_id)
        frame = dataframe_from_records(records)
        predictions = self.prediction_service.predict_with_bundle(bundle=bundle, frame=frame)
        explanations = self.explanation_service.explain_with_bundle(
            bundle=bundle,
            frame=frame,
            predictions=predictions,
        )

        recommendations = []
        for explanation in explanations["items"]:
            facts = build_facts(
                task_type=champion["task_type"],
                prediction=explanation["prediction"],
                confidence=explanation["confidence"],
                top_factors=explanation["top_factors"],
            )
            recommendation = self.engine.recommend(facts, rule_set="inline_default")
            recommendations.append(
                {
                    "row_index": explanation["row_index"],
                    "prediction": explanation["prediction"],
                    "confidence": explanation["confidence"],
                    "facts": facts,
                    "recommendation": recommendation,
                    "top_factors": explanation["top_factors"],
                }
            )

        return {
            "project_id": project_id,
            "model_version": champion["version_id"],
            "task_type": champion["task_type"],
            "recommendations": recommendations,
        }

    def evaluate_forecast(
        self,
        project_id: str,
        horizon_minutes: int,
        steps: int,
        point_count: int | None = None,
    ) -> dict[str, Any]:
        """Run DSS on future forecast points instead of historical preview rows."""
        champion, _bundle = self.registry_service.get_champion_bundle(project_id)
        if champion["task_type"] != "regression":
            raise ValueError("Forecast-driven DSS is available only for regression models.")

        forecast_payload = self.prediction_service.forecast(
            project_id=project_id,
            horizon_minutes=horizon_minutes,
            steps=steps,
        )
        return self._build_forecast_recommendations(
            model_version=champion["version_id"],
            project_id=project_id,
            task_type=champion["task_type"],
            forecast_payload=forecast_payload,
            point_count=point_count,
        )

    def evaluate_forecast_model_version(
        self,
        version_id: str,
        horizon_minutes: int,
        steps: int,
        point_count: int | None = None,
    ) -> dict[str, Any]:
        """Run DSS on future forecast points for one specific stored model version."""
        model, _bundle = self.registry_service.get_model_bundle(version_id)
        if model["task_type"] != "regression":
            raise ValueError("Forecast-driven DSS is available only for regression models.")

        forecast_payload = self.prediction_service.forecast_model_version(
            version_id=version_id,
            horizon_minutes=horizon_minutes,
            steps=steps,
        )
        return self._build_forecast_recommendations(
            model_version=model["version_id"],
            project_id=model["project_id"],
            task_type=model["task_type"],
            forecast_payload=forecast_payload,
            point_count=point_count,
        )

    def _build_forecast_recommendations(
        self,
        model_version: str,
        project_id: str,
        task_type: str,
        forecast_payload: dict[str, Any],
        point_count: int | None = None,
    ) -> dict[str, Any]:
        """Convert one forecast payload into sampled DSS recommendations."""
        history = list(forecast_payload.get("recent_history") or [])
        forecast = self._sample_forecast_points(
            list(forecast_payload.get("forecast") or []),
            point_count=point_count,
        )
        if not history:
            raise ValueError("Stored recent history is unavailable for forecast-driven DSS.")
        if not forecast:
            raise ValueError("Forecast did not produce future points for DSS evaluation.")

        baseline = self._build_forecast_baseline(history)
        recommendations = []
        previous_prediction = None
        for point in forecast:
            top_factors = self._build_forecast_factors(
                prediction=float(point["prediction"]),
                timestamp=point["timestamp"],
                baseline=baseline,
                previous_prediction=previous_prediction,
            )
            facts = self._build_forecast_facts(
                prediction=float(point["prediction"]),
                top_factors=top_factors,
                baseline=baseline,
            )
            recommendation = self.engine.recommend(facts, rule_set="forecast_default")
            recommendations.append(
                {
                    "row_index": int(point["step"]) - 1,
                    "step": int(point["step"]),
                    "timestamp": point["timestamp"],
                    "prediction": round(float(point["prediction"]), 6),
                    "confidence": None,
                    "facts": facts,
                    "recommendation": recommendation,
                    "top_factors": top_factors,
                }
            )
            previous_prediction = float(point["prediction"])

        return {
            "project_id": project_id,
            "model_version": model_version,
            "task_type": task_type,
            "target": forecast_payload.get("target"),
            "requested_horizon_minutes": forecast_payload.get("requested_horizon_minutes"),
            "base_frequency_minutes": forecast_payload.get("base_frequency_minutes"),
            "warning": forecast_payload.get("warning"),
            "recent_history": history,
            "forecast": forecast,
            "baseline": {
                "recent_mean": round(float(baseline["mean"]), 6),
                "recent_std": round(float(baseline["std"]), 6),
                "recent_p75": round(float(baseline["p75"]), 6),
                "recent_p90": round(float(baseline["p90"]), 6),
                "last_actual": round(float(baseline["last_actual"]), 6),
            },
            "recommendations": recommendations,
        }

    @staticmethod
    def _sample_forecast_points(
        forecast_rows: list[dict[str, Any]],
        point_count: int | None = None,
    ) -> list[dict[str, Any]]:
        """Mirror the graph-page downsampling so DSS sees the same visible points."""
        sanitized_rows = []
        for index, item in enumerate(forecast_rows):
            timestamp = item.get("timestamp")
            prediction = item.get("prediction")
            try:
                numeric_prediction = float(prediction)
            except (TypeError, ValueError):
                continue
            if not timestamp or not np.isfinite(numeric_prediction):
                continue
            step = item.get("step")
            sanitized_rows.append(
                {
                    "step": int(step) if step is not None else index + 1,
                    "timestamp": timestamp,
                    "prediction": numeric_prediction,
                }
            )

        if not sanitized_rows:
            return []
        if point_count is None or point_count >= len(sanitized_rows):
            return sanitized_rows

        last_index = len(sanitized_rows) - 1
        selected_indexes: set[int] = set()
        for position in range(point_count):
            ratio = 1.0 if point_count == 1 else position / (point_count - 1)
            selected_indexes.add(int(round(ratio * last_index)))

        return [sanitized_rows[index] for index in sorted(selected_indexes)]

    @staticmethod
    def _build_forecast_baseline(history: list[dict[str, Any]]) -> dict[str, float]:
        """Summarize the recent actual series as a baseline for forecast DSS rules."""
        values = [float(item["target"]) for item in history if item.get("target") is not None]
        if not values:
            raise ValueError("Recent history does not contain target values.")

        values_array = np.asarray(values, dtype=float)
        mean = float(values_array.mean())
        std = float(values_array.std())
        if std < 1e-9:
            std = max(abs(mean) * 0.05, 1.0)

        return {
            "mean": mean,
            "std": std,
            "p75": float(np.quantile(values_array, 0.75)),
            "p90": float(np.quantile(values_array, 0.90)),
            "last_actual": float(values[-1]),
        }

    def _build_forecast_facts(
        self,
        prediction: float,
        top_factors: list[dict[str, Any]],
        baseline: dict[str, float],
    ) -> dict[str, Any]:
        """Convert one future prediction into DSS facts based on recent-history thresholds."""
        if prediction >= baseline["p90"] or prediction >= baseline["mean"] + baseline["std"]:
            risk_level = "high"
        elif prediction >= baseline["p75"] or prediction >= baseline["mean"] + baseline["std"] * 0.35:
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
            "prediction": round(float(prediction), 6),
            "confidence": None,
            "strong_positive_factors": strong_positive,
            "medium_positive_factors": medium_positive,
            "negative_factors": negative_factors,
        }

    def _build_forecast_factors(
        self,
        prediction: float,
        timestamp: str,
        baseline: dict[str, float],
        previous_prediction: float | None,
    ) -> list[dict[str, Any]]:
        """Create simple, readable drivers for one forecast point."""
        factors: list[dict[str, Any]] = []
        std = baseline["std"]
        comparison_value = baseline["last_actual"] if previous_prediction is None else previous_prediction

        factors.append(
            self._make_forecast_factor(
                feature="Отклонение от среднего уровня",
                value=prediction,
                baseline=baseline["mean"],
                delta=prediction - baseline["mean"],
                positive_threshold=0.35 * std,
                negative_threshold=-0.35 * std,
            )
        )
        factors.append(
            self._make_forecast_factor(
                feature="Отклонение от последнего факта",
                value=prediction,
                baseline=comparison_value,
                delta=prediction - comparison_value,
                positive_threshold=0.25 * std,
                negative_threshold=-0.25 * std,
            )
        )
        factors.append(
            self._make_forecast_factor(
                feature="Выход за P90 недавней истории",
                value=prediction,
                baseline=baseline["p90"],
                delta=prediction - baseline["p90"],
                positive_threshold=0.0,
                negative_threshold=-0.75 * std,
            )
        )

        parsed_timestamp = self._parse_timestamp(timestamp)
        if parsed_timestamp is not None:
            peak_delta = 1.0 if parsed_timestamp.hour in {7, 8, 9, 10, 17, 18, 19, 20, 21} else -1.0
            factors.append(
                self._make_forecast_factor(
                    feature="Попадание в пиковые часы",
                    value=parsed_timestamp.hour,
                    baseline="вне пика",
                    delta=peak_delta,
                    positive_threshold=1.0,
                    negative_threshold=-1.0,
                    scale=1.0,
                )
            )

        filtered = [factor for factor in factors if factor["impact_score"] > 0]
        filtered.sort(key=lambda item: item["impact_score"], reverse=True)
        return filtered[:5]

    @staticmethod
    def _make_forecast_factor(
        feature: str,
        value: Any,
        baseline: Any,
        delta: float,
        positive_threshold: float,
        negative_threshold: float,
        scale: float | None = None,
    ) -> dict[str, Any]:
        """Normalize one forecast heuristic into the factor payload used by the UI."""
        denominator = abs(scale) if scale is not None else max(abs(positive_threshold), abs(negative_threshold), 1.0)
        impact_score = 0.0
        direction = "neutral"
        if delta >= positive_threshold:
            direction = "positive"
            impact_score = abs(delta) / denominator if denominator else abs(delta)
        elif delta <= negative_threshold:
            direction = "negative"
            impact_score = abs(delta) / denominator if denominator else abs(delta)

        strength = "weak"
        if impact_score >= 0.66:
            strength = "strong"
        elif impact_score >= 0.33:
            strength = "medium"

        return {
            "feature": feature,
            "value": value,
            "baseline": baseline,
            "direction": direction,
            "strength": strength,
            "impact_score": round(float(min(impact_score, 9.999999)), 6),
        }

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        """Parse forecast timestamps stored in ISO format."""
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
