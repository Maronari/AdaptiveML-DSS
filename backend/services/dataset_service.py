from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import UploadFile

from backend.utils.io import dataframe_from_records


class DatasetService:
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
            return pd.read_csv(file_path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)
        raise ValueError("Only CSV and Excel files are supported.")

    @staticmethod
    def read_tabular_bytes(content: bytes, suffix: str) -> pd.DataFrame:
        """Read CSV or Excel payload from raw bytes."""
        if suffix == ".csv":
            return pd.read_csv(BytesIO(content))
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(BytesIO(content))
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
        }

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
