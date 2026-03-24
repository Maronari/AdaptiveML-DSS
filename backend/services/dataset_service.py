from __future__ import annotations

import csv
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import UploadFile

from backend.services.registry_service import RegistryService
from backend.utils.io import dataframe_from_records, dataframe_to_records


TARGET_NAME_HINTS = (
    "target",
    "label",
    "class",
    "y",
    "result",
    "outcome",
    "response",
    "таргет",
    "цель",
    "целе",
    "класс",
    "метка",
)

IDENTIFIER_NAME_HINTS = (
    "id",
    "uuid",
    "guid",
    "код",
    "номер",
    "no",
)

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251", "latin-1")
CSV_DELIMITERS = (None, ";", ",", "\t", "|")


class DatasetService:
    def __init__(self) -> None:
        self.registry_service = RegistryService()

    def validate_inline_dataset(
        self,
        project_id: str,
        target: str,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate inline records and return a dataset summary."""
        frame = dataframe_from_records(records)
        return self._build_summary(project_id=project_id, target=target, frame=frame)

    async def validate_uploaded_dataset(
        self,
        project_id: str,
        target: str,
        upload: UploadFile,
    ) -> dict[str, Any]:
        """Validate an uploaded tabular file and return its summary."""
        frame = await self.read_uploaded_tabular(upload)
        return self._build_summary(project_id=project_id, target=target, frame=frame)

    async def inspect_uploaded_dataset(
        self,
        project_id: str,
        upload: UploadFile,
    ) -> dict[str, Any]:
        """Inspect an uploaded file before the target is chosen."""
        source_name = upload.filename or "uploaded.csv"
        frame = await self.read_uploaded_tabular(upload)
        return self._build_inspection_summary(project_id=project_id, source_name=source_name, frame=frame)

    async def register_uploaded_dataset(
        self,
        project_id: str,
        target: str,
        upload: UploadFile,
    ) -> dict[str, Any]:
        """Persist an uploaded dataset after the user confirms the target column."""
        source_name = upload.filename or "uploaded.csv"
        frame = await self.read_uploaded_tabular(upload)
        self._validate_training_frame(frame, target)
        dataset_version = self.registry_service.create_dataset_version(
            project_id=project_id,
            source_name=source_name,
            target=target,
            frame=frame,
        )
        summary = self._build_summary(project_id=project_id, target=target, frame=frame)
        summary["source_name"] = source_name
        summary["dataset_version"] = dataset_version.to_dict()
        return summary

    def get_registered_dataset_summary(self, version_id: str) -> dict[str, Any]:
        """Return a training-ready summary for one saved dataset version."""
        dataset = self.registry_service.get_dataset_version(version_id)
        frame = self.registry_service.load_dataset_version_frame(version_id)
        summary = self._build_summary(
            project_id=dataset["project_id"],
            target=dataset["target"],
            frame=frame,
        )
        summary["source_name"] = dataset["source_name"]
        summary["dataset_version"] = dataset
        return summary

    def get_latest_project_dataset_summary(self, project_id: str) -> dict[str, Any]:
        """Return the newest saved dataset summary for one project."""
        dataset = self.registry_service.get_latest_dataset_version(project_id)
        frame = self.registry_service.load_dataset_version_frame(dataset["version_id"])
        summary = self._build_summary(
            project_id=dataset["project_id"],
            target=dataset["target"],
            frame=frame,
        )
        summary["source_name"] = dataset["source_name"]
        summary["dataset_version"] = dataset
        return summary

    async def read_uploaded_tabular(self, upload: UploadFile) -> pd.DataFrame:
        """Read an uploaded CSV or Excel file into a DataFrame."""
        if not upload.filename:
            raise ValueError("Uploaded file must have a filename.")
        content = await upload.read()
        if not content:
            raise ValueError("Uploaded file is empty.")

        suffix = Path(upload.filename).suffix.lower()
        return self.read_tabular_bytes(content=content, suffix=suffix)

    @staticmethod
    def read_tabular_file(path: str | Path) -> pd.DataFrame:
        """Read a local CSV or Excel file by extension."""
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return DatasetService._normalize_frame(DatasetService._read_csv_bytes(file_path.read_bytes()))
        if suffix in {".xlsx", ".xls"}:
            return DatasetService._normalize_frame(pd.read_excel(file_path))
        raise ValueError("Only CSV and Excel files are supported.")

    @staticmethod
    def read_tabular_bytes(content: bytes, suffix: str) -> pd.DataFrame:
        """Read CSV or Excel payload from raw bytes."""
        if suffix == ".csv":
            return DatasetService._normalize_frame(DatasetService._read_csv_bytes(content))
        if suffix in {".xlsx", ".xls"}:
            return DatasetService._normalize_frame(pd.read_excel(BytesIO(content)))
        raise ValueError("Only CSV and Excel files are supported.")

    def _build_summary(
        self,
        project_id: str,
        target: str,
        frame: pd.DataFrame,
    ) -> dict[str, Any]:
        """Build a lightweight validation summary for UI/API responses."""
        self._validate_training_frame(frame, target)
        schema = {column: str(dtype) for column, dtype in frame.dtypes.items()}
        missing = {column: int(value) for column, value in frame.isna().sum().items()}
        duplicates = int(frame.duplicated().sum())

        return {
            "project_id": project_id,
            "target": target,
            "rows": int(len(frame)),
            "columns": frame.columns.tolist(),
            "schema": schema,
            "missing_values": missing,
            "duplicates": duplicates,
            "task_type": self.infer_task_type(frame[target]),
            "sample_rows": dataframe_to_records(frame.head(12)),
        }

    def _build_inspection_summary(
        self,
        project_id: str,
        source_name: str,
        frame: pd.DataFrame,
    ) -> dict[str, Any]:
        """Build a lightweight dataset preview before the target is known."""
        if frame.empty:
            raise ValueError("Dataset is empty.")
        if frame.columns.duplicated().any():
            raise ValueError("Dataset contains duplicate column names.")

        schema = {column: str(dtype) for column, dtype in frame.dtypes.items()}
        missing = {column: int(value) for column, value in frame.isna().sum().items()}
        duplicates = int(frame.duplicated().sum())
        column_profiles = [
            self._profile_column(frame=frame, column=column, index=index)
            for index, column in enumerate(frame.columns.tolist())
        ]
        target_candidates = [
            {
                "column": profile["column"],
                "score": profile["target_score"],
                "reason": profile["target_reason"],
                "suggested_task_type": profile["suggested_task_type"],
                "distinct_values": profile["distinct_values"],
                "missing_values": profile["missing_values"],
                "kind": profile["kind"],
            }
            for profile in column_profiles
            if profile["target_eligible"]
        ]
        target_candidates.sort(key=lambda item: (-item["score"], item["column"].casefold()))
        recommended_target = target_candidates[0]["column"] if target_candidates else None

        for profile in column_profiles:
            profile["is_recommended_target"] = profile["column"] == recommended_target

        target_summaries = {
            profile["column"]: self._build_target_summary(frame=frame, profile=profile)
            for profile in column_profiles
        }

        return {
            "project_id": project_id,
            "source_name": source_name,
            "rows": int(len(frame)),
            "columns": frame.columns.tolist(),
            "schema": schema,
            "missing_values": missing,
            "duplicates": duplicates,
            "sample_rows": dataframe_to_records(frame.head(12)),
            "column_profiles": column_profiles,
            "target_candidates": target_candidates,
            "recommended_target": recommended_target,
            "column_type_summary": self._build_column_type_summary(column_profiles),
            "temporal_context": self._build_temporal_context(frame, column_profiles),
            "target_summaries": target_summaries,
            "project_context": self._build_project_context(project_id),
        }

    @staticmethod
    def _build_column_type_summary(column_profiles: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate semantic column kinds for the upload UI."""
        kind_labels = {
            "numeric": "Числовые",
            "categorical": "Категориальные",
            "datetime": "Дата и время",
            "time": "Время",
            "boolean": "Логические",
            "text": "Текстовые",
            "empty": "Пустые",
        }
        kind_order = ("numeric", "categorical", "datetime", "time", "boolean", "text", "empty")
        counts = {kind: 0 for kind in kind_order}

        for profile in column_profiles:
            kind = profile["kind"]
            counts[kind] = counts.get(kind, 0) + 1

        return {
            "total_columns": len(column_profiles),
            "items": [
                {
                    "kind": kind,
                    "label": kind_labels.get(kind, kind.title()),
                    "count": counts[kind],
                }
                for kind in kind_order
                if counts.get(kind, 0) > 0
            ],
        }

    def _build_temporal_context(
        self,
        frame: pd.DataFrame,
        column_profiles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Describe the primary datetime column if the dataset looks temporal."""
        datetime_profile = next((profile for profile in column_profiles if profile["kind"] == "datetime"), None)
        if datetime_profile is None:
            return {
                "available": False,
                "column": None,
                "message": "Временная колонка не найдена.",
            }

        column = datetime_profile["column"]
        parsed = pd.to_datetime(frame[column], errors="coerce")
        valid = parsed.dropna().sort_values()
        if valid.empty:
            return {
                "available": False,
                "column": column,
                "message": "Не удалось распознать временную шкалу.",
            }

        unique_timestamps = valid.drop_duplicates()
        frequency = None
        has_gaps = False
        if len(unique_timestamps) >= 2:
            deltas = unique_timestamps.diff().dropna()
            positive_deltas = deltas[deltas > pd.Timedelta(0)]
            if not positive_deltas.empty:
                mode = positive_deltas.mode()
                frequency = mode.iloc[0] if not mode.empty else positive_deltas.iloc[0]
                expected_points = int(
                    round((unique_timestamps.iloc[-1] - unique_timestamps.iloc[0]) / frequency)
                ) + 1
                has_gaps = expected_points > len(unique_timestamps)

        time_span = (
            unique_timestamps.iloc[-1] - unique_timestamps.iloc[0]
            if len(unique_timestamps) >= 2
            else pd.Timedelta(0)
        )
        frequency_minutes = None
        if frequency is not None:
            frequency_minutes = round(float(frequency.total_seconds()) / 60.0, 6)

        return {
            "available": True,
            "column": column,
            "rows_with_timestamp": int(parsed.notna().sum()),
            "unique_timestamps": int(len(unique_timestamps)),
            "start": unique_timestamps.iloc[0].isoformat(),
            "end": unique_timestamps.iloc[-1].isoformat(),
            "range_label": self._format_timedelta(time_span) or "Одна временная точка",
            "frequency_minutes": frequency_minutes,
            "frequency_label": self._format_timedelta(frequency) if frequency is not None else None,
            "has_gaps": has_gaps,
            "message": None,
        }

    def _build_target_summary(
        self,
        frame: pd.DataFrame,
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        """Summarize one selected target candidate for the upload UI."""
        column = profile["column"]
        series = frame[column]
        cleaned = series.dropna()
        distinct_values = int(cleaned.nunique(dropna=True))
        kind = profile["kind"]
        task_type = profile["suggested_task_type"] if kind not in {"datetime", "time"} and distinct_values >= 2 else None

        summary: dict[str, Any] = {
            "column": column,
            "kind": kind,
            "task_type": task_type,
            "missing_values": int(series.isna().sum()),
            "non_null_values": int(len(cleaned)),
            "distinct_values": distinct_values,
            "target_eligible": profile["target_eligible"],
            "target_reason": profile["target_reason"],
            "looks_like_identifier": profile["looks_like_identifier"],
        }

        if cleaned.empty:
            summary["message"] = "Нет данных для анализа выбранной колонки."
            return summary

        if task_type == "regression":
            numeric = pd.to_numeric(cleaned, errors="coerce").dropna()
            if not numeric.empty:
                summary["numeric_stats"] = {
                    "min": round(float(numeric.min()), 6),
                    "max": round(float(numeric.max()), 6),
                    "mean": round(float(numeric.mean()), 6),
                    "median": round(float(numeric.median()), 6),
                }
        else:
            value_counts = cleaned.astype(str).value_counts(dropna=True)
            total = max(int(len(cleaned)), 1)
            summary["class_count"] = distinct_values
            summary["top_values"] = [
                {
                    "label": label,
                    "count": int(count),
                    "share": round(int(count) / total, 4),
                }
                for label, count in value_counts.head(3).items()
            ]
            summary["majority_share"] = round(float(value_counts.iloc[0]) / total, 4) if not value_counts.empty else None

        return summary

    def _build_project_context(self, project_id: str) -> dict[str, Any]:
        """Describe how the uploaded dataset relates to the current project history."""
        summary = self.registry_service.get_project_summary(project_id)
        return {
            "project_id": summary["project_id"],
            "project_name": summary["name"],
            "dataset_versions": summary["dataset_versions"],
            "model_versions": summary["model_versions"],
            "latest_dataset_version_id": summary["latest_dataset_version_id"],
            "latest_dataset_source_name": summary["latest_dataset_source_name"],
            "latest_dataset_created_at": summary["latest_dataset_created_at"],
            "latest_model_version_id": summary["latest_model_version_id"],
            "latest_model_created_at": summary["latest_model_created_at"],
            "latest_model_target": summary["target"],
            "latest_model_task_type": summary["task_type"],
            "champion_model_version_id": summary["champion_model_version_id"],
            "has_datasets": summary["has_datasets"],
            "has_models": summary["has_models"],
            "has_champion_model": summary["has_champion_model"],
            "is_new_project": not summary["has_datasets"] and not summary["has_models"],
        }

    @staticmethod
    def _format_timedelta(value: pd.Timedelta | None) -> str | None:
        """Format a timedelta into a compact human-readable label."""
        if value is None or pd.isna(value):
            return None
        total_seconds = int(round(value.total_seconds()))
        if total_seconds <= 0:
            return "0 мин"
        if total_seconds % 86400 == 0:
            return f"{total_seconds // 86400} дн"
        if total_seconds % 3600 == 0:
            return f"{total_seconds // 3600} ч"
        if total_seconds % 60 == 0:
            return f"{total_seconds // 60} мин"
        return f"{total_seconds} сек"

    @staticmethod
    def _validate_training_frame(frame: pd.DataFrame, target: str) -> None:
        """Enforce the minimum schema required for training."""
        if frame.empty:
            raise ValueError("Dataset is empty.")
        if target not in frame.columns:
            raise ValueError(f"Target column '{target}' is missing.")
        if len(frame.columns) < 2:
            raise ValueError("Dataset must contain at least one feature column and target.")
        if frame.columns.duplicated().any():
            raise ValueError("Dataset contains duplicate column names.")
        if frame[target].isna().any():
            raise ValueError("Target column contains missing values.")
        if frame[target].nunique(dropna=True) < 2:
            raise ValueError("Target column must contain at least two distinct values.")

    @staticmethod
    def validate_prediction_frame(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        """Align inference data with the model feature schema."""
        if frame.empty:
            raise ValueError("Prediction payload is empty.")

        missing = [feature for feature in feature_names if feature not in frame.columns]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")

        aligned = frame.copy()
        extra = [column for column in aligned.columns if column not in feature_names]
        if extra:
            aligned = aligned.drop(columns=extra)

        return aligned[feature_names]

    @staticmethod
    def infer_task_type(target: pd.Series) -> str:
        """Infer binary, multiclass or regression from the target column."""
        cleaned = target.dropna()
        unique_count = cleaned.nunique()

        if pd.api.types.is_numeric_dtype(cleaned):
            relative_cardinality = unique_count / max(len(cleaned), 1)
            if unique_count == 2:
                return "binary"
            if unique_count <= 10 and relative_cardinality < 0.2:
                return "multiclass"
            return "regression"

        return "binary" if unique_count == 2 else "multiclass"

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        """Trim empty rows/columns and normalize column names after parsing."""
        normalized = frame.copy()
        normalized = normalized.dropna(axis=0, how="all").dropna(axis=1, how="all")
        renamed_columns: list[str] = []
        for index, column in enumerate(normalized.columns.tolist(), start=1):
            name = str(column).replace("\ufeff", "").strip()
            renamed_columns.append(name or f"column_{index}")
        normalized.columns = renamed_columns
        return normalized.reset_index(drop=True)

    @staticmethod
    def _read_csv_bytes(content: bytes) -> pd.DataFrame:
        """Read CSV bytes using delimiter and encoding fallbacks common in local datasets."""
        sample_text = content[:8192].decode("utf-8", errors="ignore")
        sniffed_delimiter = None
        try:
            sniffed = csv.Sniffer().sniff(sample_text, delimiters=";,|\t,")
            sniffed_delimiter = sniffed.delimiter
        except csv.Error:
            sniffed_delimiter = None

        ordered_delimiters = [sniffed_delimiter, *CSV_DELIMITERS]
        last_error: Exception | None = None

        for encoding in CSV_ENCODINGS:
            seen_delimiters: set[str | None] = set()
            for delimiter in ordered_delimiters:
                if delimiter in seen_delimiters:
                    continue
                seen_delimiters.add(delimiter)
                try:
                    if delimiter is None:
                        frame = pd.read_csv(BytesIO(content), sep=None, engine="python", encoding=encoding)
                    else:
                        frame = pd.read_csv(BytesIO(content), sep=delimiter, encoding=encoding)
                except (UnicodeDecodeError, csv.Error, pd.errors.ParserError, ValueError) as exc:
                    last_error = exc
                    continue

                if not DatasetService._looks_like_valid_parse(frame):
                    continue

                return frame

        if last_error is not None:
            raise ValueError(f"Could not parse CSV file: {last_error}") from last_error
        raise ValueError("Could not parse CSV file.")

    @staticmethod
    def _looks_like_valid_parse(frame: pd.DataFrame) -> bool:
        """Reject obviously broken parse results while keeping valid one-column files."""
        if frame is None:
            return False
        if len(frame.columns) == 0:
            return False
        unnamed_columns = sum(str(column).startswith("Unnamed:") for column in frame.columns)
        return unnamed_columns < len(frame.columns)

    def _profile_column(self, frame: pd.DataFrame, column: str, index: int) -> dict[str, Any]:
        """Build one column profile and rank it as a possible target."""
        series = frame[column]
        cleaned = series.dropna()
        distinct_values = int(cleaned.nunique(dropna=True))
        non_null_values = int(len(cleaned))
        missing_values = int(series.isna().sum())
        distinct_ratio = distinct_values / max(non_null_values, 1)
        kind = self._infer_column_kind(series)
        suggested_task_type = None
        if non_null_values > 0 and distinct_values >= 2:
            suggested_task_type = self.infer_task_type(cleaned)

        looks_like_identifier = self._looks_like_identifier(
            column=column,
            series=cleaned,
            kind=kind,
            distinct_ratio=distinct_ratio,
        )
        target_score, target_reason = self._score_target_candidate(
            column=column,
            index=index,
            total_columns=len(frame.columns),
            kind=kind,
            series=cleaned,
            distinct_values=distinct_values,
            missing_values=missing_values,
            looks_like_identifier=looks_like_identifier,
            suggested_task_type=suggested_task_type,
        )
        target_eligible = distinct_values >= 2 and not looks_like_identifier and kind not in {"datetime", "time"}

        return {
            "column": column,
            "dtype": str(series.dtype),
            "kind": kind,
            "missing_values": missing_values,
            "non_null_values": non_null_values,
            "distinct_values": distinct_values,
            "distinct_ratio": round(distinct_ratio, 6),
            "example_values": [str(value) for value in cleaned.head(4).tolist()],
            "suggested_task_type": suggested_task_type,
            "target_eligible": target_eligible,
            "target_score": target_score,
            "target_reason": target_reason,
            "looks_like_identifier": looks_like_identifier,
        }

    def _score_target_candidate(
        self,
        column: str,
        index: int,
        total_columns: int,
        kind: str,
        series: pd.Series,
        distinct_values: int,
        missing_values: int,
        looks_like_identifier: bool,
        suggested_task_type: str | None,
    ) -> tuple[int, str]:
        """Rank one column as a possible target with a compact explanation."""
        score = 0
        reasons: list[str] = []
        normalized_name = column.casefold()
        row_count = int(len(series))
        distinct_ratio = distinct_values / max(row_count, 1)

        if any(hint in normalized_name for hint in TARGET_NAME_HINTS):
            score += 70
            reasons.append("имя похоже на target")
        if index == total_columns - 1:
            score += 18
            reasons.append("стоит последней колонкой")
        if suggested_task_type == "regression":
            score += 18
            reasons.append("похоже на числовую целевую переменную")
        elif suggested_task_type in {"binary", "multiclass"}:
            score += 16
            reasons.append("подходит для классификации")
        if kind in {"numeric", "boolean", "categorical"}:
            score += 10
        if missing_values == 0:
            score += 6
        if distinct_values < 2:
            score -= 100
            reasons.append("мало уникальных значений")
        if looks_like_identifier:
            score -= 50
            reasons.append("похоже на идентификатор")
        if kind in {"datetime", "time"}:
            score -= 35
            reasons.append("похоже на временную метку")
        if distinct_ratio > 0.98 and kind == "text":
            score -= 20
            reasons.append("почти все значения уникальны")

        if not reasons:
            reasons.append("может быть выбрано как target вручную")
        return score, "; ".join(reasons)

    @staticmethod
    def _infer_column_kind(series: pd.Series) -> str:
        """Infer a user-facing semantic kind for one column."""
        cleaned = series.dropna()
        if cleaned.empty:
            return "empty"
        if pd.api.types.is_bool_dtype(cleaned):
            return "boolean"
        if pd.api.types.is_numeric_dtype(cleaned):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(cleaned):
            return "datetime"

        as_text = cleaned.astype(str).str.strip()
        parsed_datetime = pd.to_datetime(as_text, errors="coerce")
        if parsed_datetime.notna().mean() >= 0.8:
            return "datetime"

        time_pattern = r"^\d{1,2}:\d{2}(:\d{2})?$"
        if as_text.str.match(time_pattern).mean() >= 0.8:
            return "time"

        unique_count = int(as_text.nunique(dropna=True))
        if unique_count <= max(20, int(len(as_text) * 0.2)):
            return "categorical"
        return "text"

    @staticmethod
    def _looks_like_identifier(
        column: str,
        series: pd.Series,
        kind: str,
        distinct_ratio: float,
    ) -> bool:
        """Detect columns that are likely identifiers rather than targets."""
        normalized_name = column.casefold()
        if any(
            normalized_name == hint
            or normalized_name.endswith(f"_{hint}")
            or normalized_name.startswith(f"{hint}_")
            for hint in IDENTIFIER_NAME_HINTS
        ):
            return True
        if kind == "datetime":
            return True
        if distinct_ratio < 0.98:
            return False
        return kind in {"text", "categorical"}
