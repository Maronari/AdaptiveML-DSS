from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


class SQLiteRegistry:
    """Persist registry collections inside a lightweight SQLite database."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def read(self, name: str) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                "SELECT payload_json FROM registry_entries WHERE name = ?",
                (name,),
            ).fetchone()
        if row is None:
            return []
        return json.loads(row[0])

    def write(self, name: str, payload: list[dict[str, Any]]) -> None:
        payload_json = json.dumps(payload, ensure_ascii=True, indent=2)
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """
                INSERT INTO registry_entries(name, payload_json)
                VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (name, payload_json),
            )
            connection.commit()

    def _ensure_schema(self) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS registry_entries (
                    name TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.commit()
