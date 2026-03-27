from __future__ import annotations

from dss.rules.loader import get_rule_set_config


_rule_set_name, _rule_set = get_rule_set_config("inline_default")
DEFAULT_SCENARIOS = dict(_rule_set.get("scenarios") or {})
