from __future__ import annotations

from typing import Any

from backend.services.dataset_service import DatasetService
from backend.services.prediction_service import PredictionService
from backend.services.registry_service import RegistryService
from backend.utils.io import dataframe_from_records, pythonize
from explainability.shap_service.service import try_shap_explanations


class ExplanationService:
    def __init__(self) -> None:
        self.dataset_service = DatasetService()
        self.prediction_service = PredictionService()
        self.registry_service = RegistryService()

    def explain(self, project_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        champion, bundle = self.registry_service.get_champion_bundle(project_id)
        frame = dataframe_from_records(records)
        predictions = self.prediction_service.predict_with_bundle(bundle=bundle, frame=frame)
        explanations = self.explain_with_bundle(bundle=bundle, frame=frame, predictions=predictions)
        return {
            "project_id": project_id,
            "model_version": champion["version_id"],
            "method": explanations["method"],
            "items": explanations["items"],
        }

    def explain_with_bundle(
        self,
        bundle: dict[str, Any],
        frame,
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        preprocessor = bundle.get("preprocessor")
        prepared = preprocessor.transform(frame) if preprocessor is not None else frame
        aligned = self.dataset_service.validate_prediction_frame(prepared, bundle["feature_names"])
        shap_payload = try_shap_explanations(bundle=bundle, frame=aligned)
        if shap_payload is not None:
            items = []
            shap_items = {item["row_index"]: item for item in shap_payload["items"]}
            for index, prediction in enumerate(predictions):
                items.append(
                    {
                        "row_index": index,
                        "prediction": prediction["prediction"],
                        "confidence": prediction["confidence"],
                        "top_factors": shap_items.get(index, {}).get("top_factors", []),
                    }
                )
            return {"method": shap_payload["method"], "items": items}

        importances = bundle["feature_importances"]
        baseline_profile = bundle["baseline_profile"]
        items = []
        for index, row in aligned.iterrows():
            top_factors = []
            for feature, importance in importances.items():
                value = row[feature]
                factor = self._describe_factor(
                    feature=feature,
                    value=value,
                    importance=float(importance),
                    baseline_profile=baseline_profile,
                )
                if factor["impact_score"] > 0:
                    top_factors.append(factor)

            top_factors.sort(key=lambda item: item["impact_score"], reverse=True)
            items.append(
                {
                    "row_index": int(index),
                    "prediction": predictions[index]["prediction"],
                    "confidence": predictions[index]["confidence"],
                    "top_factors": top_factors[:5],
                }
            )

        return {"method": "proxy-feature-impact", "items": items}

    @staticmethod
    def _describe_factor(
        feature: str,
        value: Any,
        importance: float,
        baseline_profile: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        numeric_baselines = baseline_profile["numeric"]
        categorical_baselines = baseline_profile["categorical"]

        if feature in numeric_baselines:
            baseline = float(numeric_baselines[feature])
            current = float(value)
            delta = current - baseline
            impact_score = abs(delta) * importance
            direction = "positive" if delta >= 0 else "negative"
        else:
            baseline = categorical_baselines.get(feature)
            current = str(value)
            impact_score = importance if current != baseline else importance * 0.2
            direction = "positive" if current != baseline else "neutral"

        strength = "weak"
        if impact_score >= 0.66:
            strength = "strong"
        elif impact_score >= 0.33:
            strength = "medium"

        return {
            "feature": feature,
            "value": pythonize(value),
            "baseline": pythonize(baseline),
            "direction": direction,
            "strength": strength,
            "impact_score": round(float(impact_score), 6),
        }
