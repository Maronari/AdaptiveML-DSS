from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.utils.io import read_json, write_json


class FilesystemRegistry:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def read(self, name: str) -> list[dict[str, Any]]:
        return read_json(self.base_dir / f"{name}.json")

    def write(self, name: str, payload: list[dict[str, Any]]) -> None:
        write_json(self.base_dir / f"{name}.json", payload)
