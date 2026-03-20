from __future__ import annotations

from backend.utils.compat import patch_collections_for_experta
from dss.recommendations.formatter import format_recommendation
from dss.rules.catalog import DEFAULT_RULES
from dss.scenarios.defaults import DEFAULT_SCENARIOS


try:
    patch_collections_for_experta()
    from experta import DefFacts, Fact, KnowledgeEngine, P, Rule

    EXPERTA_AVAILABLE = True
except Exception:
    EXPERTA_AVAILABLE = False


if EXPERTA_AVAILABLE:
    class RecommendationFact(Fact):
        pass


    class AdaptiveDecisionKnowledgeEngine(KnowledgeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.selected_rule = DEFAULT_RULES[-1]

        @DefFacts()
        def bootstrap(self):
            yield Fact(boot=True)

        @Rule(RecommendationFact(risk_level="high"))
        def high_risk(self):
            self.selected_rule = DEFAULT_RULES[0]

        @Rule(RecommendationFact(risk_level="medium", strong_positive_count=P(lambda count: count > 0)))
        def escalated_medium_risk(self):
            self.selected_rule = DEFAULT_RULES[0]

        @Rule(RecommendationFact(risk_level="medium"))
        def medium_risk(self):
            if self.selected_rule["risk_level"] != "high":
                self.selected_rule = DEFAULT_RULES[1]

        @Rule(RecommendationFact(risk_level="low"))
        def low_risk(self):
            if self.selected_rule["risk_level"] not in {"high", "medium"}:
                self.selected_rule = DEFAULT_RULES[2]


class DecisionEngine:
    def __init__(self) -> None:
        self.engine = AdaptiveDecisionKnowledgeEngine() if EXPERTA_AVAILABLE else None

    def recommend(self, facts: dict) -> dict:
        if self.engine is not None:
            self.engine.reset()
            self.engine.selected_rule = DEFAULT_RULES[-1]
            self.engine.declare(
                RecommendationFact(
                    risk_level=facts["risk_level"],
                    strong_positive_count=len(facts["strong_positive_factors"]),
                    medium_positive_count=len(facts["medium_positive_factors"]),
                    negative_count=len(facts["negative_factors"]),
                )
            )
            self.engine.run()
            matched_rule = self.engine.selected_rule
        else:
            risk_level = facts["risk_level"]
            matched_rule = next(
                (
                    rule
                    for rule in DEFAULT_RULES
                    if rule["risk_level"] == risk_level
                ),
                DEFAULT_RULES[-1],
            )

        scenario = DEFAULT_SCENARIOS[matched_rule["scenario"]]
        return format_recommendation(rule=matched_rule, scenario=scenario, facts=facts)
