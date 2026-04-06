from __future__ import annotations

from typing import Any

from dss.recommendations.formatter import format_recommendation
from dss.rules.loader import get_rule_set_config

# Compatibility flag preserved for legacy integration tests and callers.
# The DSS engine is now config-driven and no longer uses the Experta runtime.
EXPERTA_AVAILABLE = False


class DecisionEngine:
    def __init__(self, default_rule_set: str | None = None) -> None:
        """Prepare a config-driven DSS selector that can switch rule sets at runtime."""
        self.default_rule_set = default_rule_set

    def recommend(self, facts: dict[str, Any], rule_set: str | None = None) -> dict[str, Any]:
        """Choose a scenario from the configured rule set and format the final payload."""
        resolved_rule_set, rule_set_config = get_rule_set_config(rule_set or self.default_rule_set)
        enriched_facts = self._enrich_facts(facts)
        matched_rule = self._match_rule(rule_set_config["rules"], enriched_facts)
        scenario = rule_set_config["scenarios"][matched_rule["scenario"]]
        return format_recommendation(
            rule=matched_rule,
            scenario=scenario,
            facts=facts,
            rule_set_name=resolved_rule_set,
        )

    @staticmethod
    def _enrich_facts(facts: dict[str, Any]) -> dict[str, Any]:
        """Add derived counters so the config layer can match numeric conditions."""
        strong_positive_factors = list(facts.get("strong_positive_factors") or [])
        medium_positive_factors = list(facts.get("medium_positive_factors") or [])
        negative_factors = list(facts.get("negative_factors") or [])

        enriched = dict(facts)
        enriched["strong_positive_count"] = len(strong_positive_factors)
        enriched["medium_positive_count"] = len(medium_positive_factors)
        enriched["negative_count"] = len(negative_factors)
        return enriched

    def _match_rule(self, rules: list[dict[str, Any]], facts: dict[str, Any]) -> dict[str, Any]:
        """Return the highest-priority rule whose conditions match the current facts."""
        ordered_rules = sorted(
            rules,
            key=lambda rule: (int(rule.get("priority", 0)), str(rule.get("rule_id", ""))),
            reverse=True,
        )
        for rule in ordered_rules:
            if self._matches_conditions(facts, rule.get("conditions", {})):
                return rule
        raise ValueError("No DSS rule matched the provided facts.")

    @staticmethod
    def _matches_conditions(facts: dict[str, Any], conditions: dict[str, Any]) -> bool:
        """Evaluate one config rule against the enriched fact payload."""
        for key, expected in conditions.items():
            if key.endswith("_min"):
                field_name = key[:-4]
                actual = facts.get(field_name)
                if actual is None or float(actual) < float(expected):
                    return False
                continue

            if key.endswith("_max"):
                field_name = key[:-4]
                actual = facts.get(field_name)
                if actual is None or float(actual) > float(expected):
                    return False
                continue

            if key.endswith("_in"):
                field_name = key[:-3]
                actual = facts.get(field_name)
                allowed_values = expected if isinstance(expected, list) else [expected]
                if actual not in allowed_values:
                    return False
                continue

            if key.endswith("_equals"):
                field_name = key[:-7]
                if facts.get(field_name) != expected:
                    return False
                continue

            if facts.get(key) != expected:
                return False

        return True
