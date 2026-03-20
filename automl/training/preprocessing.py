from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any

import pandas as pd


@dataclass(slots=True)
class TabularPreprocessor:
    """Convert raw tabular columns into model-friendly engineered features."""
    datetime_columns: list[str] = field(default_factory=list)
    time_columns: list[str] = field(default_factory=list)

    def fit(self, frame: pd.DataFrame, target: str | None = None) -> "TabularPreprocessor":
        """Detect datetime-like and time-only columns on the training frame."""
        self.datetime_columns = []
        self.time_columns = []

        for column in frame.columns:
            if target is not None and column == target:
                continue

            series = frame[column]
            if self._is_time_only(series):
                self.time_columns.append(column)
                continue
            if self._is_datetime_like(series):
                self.datetime_columns.append(column)

        return self

    def fit_transform(self, frame: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
        """Fit the preprocessor and immediately transform the same frame."""
        self.fit(frame, target=target)
        return self.transform(frame, target=target)

    def transform(self, frame: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
        """Apply learned datetime and time projections to a new frame."""
        transformed = frame.copy()

        for column in list(self.datetime_columns):
            if column not in transformed.columns:
                continue
            parsed = pd.to_datetime(transformed[column], errors="coerce")
            transformed[f"{column}__ts"] = (parsed.astype("int64") // 10**9).where(parsed.notna(), 0)
            transformed[f"{column}__hour"] = parsed.dt.hour.fillna(0).astype(int)
            transformed[f"{column}__dayofweek"] = parsed.dt.dayofweek.fillna(0).astype(int)
            transformed[f"{column}__month"] = parsed.dt.month.fillna(0).astype(int)
            transformed = transformed.drop(columns=[column])

        for column in list(self.time_columns):
            if column not in transformed.columns:
                continue
            transformed[f"{column}__minutes"] = transformed[column].map(self._to_minutes).fillna(0.0)
            transformed = transformed.drop(columns=[column])

        return transformed

    @staticmethod
    def _is_time_only(series: pd.Series) -> bool:
        """Detect columns that look like time-of-day values."""
        non_null = series.dropna()
        if non_null.empty:
            return False

        sample = non_null.head(20)
        if sample.map(lambda value: isinstance(value, time)).all():
            return True

        if not pd.api.types.is_object_dtype(sample) and not pd.api.types.is_string_dtype(sample):
            return False

        as_text = sample.astype(str).str.strip()
        if as_text.empty:
            return False

        matches = as_text.str.match(r"^\d{1,2}:\d{2}(:\d{2})?$")
        return bool(matches.mean() >= 0.8)

    @staticmethod
    def _is_datetime_like(series: pd.Series) -> bool:
        """Detect columns that should be interpreted as datetimes."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        non_null = series.dropna()
        if non_null.empty:
            return False

        sample = non_null.head(100)
        if pd.api.types.is_numeric_dtype(sample):
            return False

        parsed = pd.to_datetime(sample, errors="coerce")
        success_rate = parsed.notna().mean()
        return bool(success_rate >= 0.8)

    @staticmethod
    def _to_minutes(value: Any) -> float | None:
        """Convert a time-like value into minutes from midnight."""
        if value is None or pd.isna(value):
            return None

        if isinstance(value, time):
            return float(value.hour * 60 + value.minute + value.second / 60.0)

        text = str(value).strip()
        parts = text.split(":")
        if len(parts) not in {2, 3}:
            return None

        try:
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2]) if len(parts) == 3 else 0
        except ValueError:
            return None

        return float(hour * 60 + minute + second / 60.0)
