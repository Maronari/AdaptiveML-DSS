from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.services.settings import get_settings


BUILTIN_RULE_SETS_PATH = Path(__file__).with_name("default_rule_sets.json")
OVERRIDE_RULE_SETS_FILENAME = "dss_rule_sets.json"


def load_rule_engine_config() -> dict[str, Any]:
    """Load DSS rule sets from storage override or fallback to the built-in config."""
    settings = get_settings()
    override_path = settings.registry_dir / OVERRIDE_RULE_SETS_FILENAME
    source_path = override_path if override_path.exists() else BUILTIN_RULE_SETS_PATH

    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("DSS rule engine config must be a JSON object.")

    rule_sets = payload.get("rule_sets")
    if not isinstance(rule_sets, dict) or not rule_sets:
        raise ValueError("DSS rule engine config must define non-empty 'rule_sets'.")

    default_rule_set = payload.get("default_rule_set")
    if not default_rule_set or default_rule_set not in rule_sets:
        raise ValueError("DSS rule engine config has an invalid 'default_rule_set'.")

    for rule_set_name, rule_set in rule_sets.items():
        _validate_rule_set(rule_set_name, rule_set)

    return payload


def get_rule_set_config(rule_set_name: str | None = None) -> tuple[str, dict[str, Any]]:
    """Return one configured rule set by name, or the default set when omitted."""
    payload = load_rule_engine_config()
    resolved_name = rule_set_name or payload["default_rule_set"]
    rule_sets = payload["rule_sets"]
    if resolved_name not in rule_sets:
        raise ValueError(f"Unknown DSS rule set '{resolved_name}'.")
    return resolved_name, rule_sets[resolved_name]


def _validate_rule_set(rule_set_name: str, rule_set: Any) -> None:
    """Ensure one rule set has the minimum structure required by DecisionEngine."""
    if not isinstance(rule_set, dict):
        raise ValueError(f"DSS rule set '{rule_set_name}' must be a JSON object.")

    scenarios = rule_set.get("scenarios")
    rules = rule_set.get("rules")
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError(f"DSS rule set '{rule_set_name}' must define non-empty 'scenarios'.")
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"DSS rule set '{rule_set_name}' must define non-empty 'rules'.")

    for scenario_name, actions in scenarios.items():
        if not isinstance(actions, list) or not all(isinstance(item, str) and item.strip() for item in actions):
            raise ValueError(
                f"DSS scenario '{scenario_name}' in rule set '{rule_set_name}' must be a non-empty string list."
            )

    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError(f"DSS rule in rule set '{rule_set_name}' must be a JSON object.")
        if not str(rule.get("rule_id") or "").strip():
            raise ValueError(f"DSS rule in rule set '{rule_set_name}' is missing 'rule_id'.")
        if not str(rule.get("scenario") or "").strip():
            raise ValueError(f"DSS rule '{rule.get('rule_id')}' is missing 'scenario'.")
        if rule["scenario"] not in scenarios:
            raise ValueError(
                f"DSS rule '{rule.get('rule_id')}' points to unknown scenario '{rule['scenario']}'."
            )
        if "summary" not in rule or not str(rule["summary"]).strip():
            raise ValueError(f"DSS rule '{rule.get('rule_id')}' is missing 'summary'.")
        conditions = rule.get("conditions", {})
        if not isinstance(conditions, dict):
            raise ValueError(f"DSS rule '{rule.get('rule_id')}' must define 'conditions' as an object.")
