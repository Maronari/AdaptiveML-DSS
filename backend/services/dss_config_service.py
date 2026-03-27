from __future__ import annotations

import json
from typing import Any

from backend.services.settings import get_settings
from dss.rules.loader import OVERRIDE_RULE_SETS_FILENAME, load_rule_engine_config


class DssConfigService:
    """Read and persist DSS rule-set configuration for the UI editor."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def get_rule_engine_config(self) -> dict[str, Any]:
        return load_rule_engine_config()

    def save_rule_engine_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("DSS rule config payload must be a JSON object.")

        target_path = self.settings.registry_dir / OVERRIDE_RULE_SETS_FILENAME
        target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return load_rule_engine_config()
