from __future__ import annotations

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
            recommendation = self.engine.recommend(facts)
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
