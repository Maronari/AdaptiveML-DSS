from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from automl.evaluation.metrics import evaluate_predictions


LAG_CANDIDATES = (1, 2, 3, 6, 12, 24, 48)
ROLLING_CANDIDATES = (3, 6, 12, 24, 48)
HISTORY_TAIL_ROWS = 240


@dataclass(slots=True)
class TimestampDescriptor:
    """Describe how timestamps are reconstructed from the source frame."""

    strategy: str
    timestamp_column: str | None = None
    date_column: str | None = None
    hour_column: str | None = None
    hour_offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the descriptor into a plain dictionary."""
        return asdict(self)


def build_forecasting_bundle(
    frame: pd.DataFrame,
    target: str,
    random_state: int,
) -> dict[str, Any] | None:
    """Train a short-horizon autoregressive forecaster when the dataset is temporal."""
    descriptor = infer_timestamp_descriptor(frame=frame, target=target)
    if descriptor is None:
        return None

    series_frame = build_time_series_frame(frame=frame, target=target, descriptor=descriptor)
    if series_frame is None or len(series_frame) < 12:
        return None

    lag_steps = [step for step in LAG_CANDIDATES if step < len(series_frame)]
    rolling_windows = [window for window in ROLLING_CANDIDATES if window < len(series_frame)]
    if not lag_steps:
        return None

    required_history = max([2] + lag_steps + rolling_windows)
    if len(series_frame) <= required_history + 4:
        return None

    feature_frame = build_forecasting_feature_frame(
        series_frame=series_frame,
        target=target,
        lag_steps=lag_steps,
        rolling_windows=rolling_windows,
    )
    if len(feature_frame) < 8:
        return None

    test_size = max(1, int(round(len(feature_frame) * 0.2)))
    split_index = len(feature_frame) - test_size
    if split_index < 6:
        return None

    feature_columns = [column for column in feature_frame.columns if column not in {"timestamp", target}]
    x_train = feature_frame.iloc[:split_index][feature_columns]
    y_train = feature_frame.iloc[:split_index][target]
    x_test = feature_frame.iloc[split_index:][feature_columns]
    y_test = feature_frame.iloc[split_index:][target]

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = evaluate_predictions(task_type="regression", y_true=y_test, y_pred=predictions)

    timestamp_series = series_frame["timestamp"]
    frequency_minutes = infer_frequency_minutes(timestamp_series)
    history_window = max(required_history, HISTORY_TAIL_ROWS)

    return {
        "enabled": True,
        "model": model,
        "target": target,
        "timestamp_descriptor": descriptor.to_dict(),
        "feature_columns": feature_columns,
        "lag_steps": lag_steps,
        "rolling_windows": rolling_windows,
        "required_history": required_history,
        "base_frequency_minutes": frequency_minutes,
        "default_horizon_minutes": min(30, frequency_minutes) if frequency_minutes > 0 else 30,
        "metrics": metrics,
        "history_targets": [float(value) for value in series_frame[target].tail(history_window).tolist()],
        "recent_history": [
            {
                "timestamp": row["timestamp"].isoformat(),
                "target": float(row[target]),
            }
            for _, row in series_frame.tail(HISTORY_TAIL_ROWS).iterrows()
        ],
        "last_timestamp": timestamp_series.iloc[-1].isoformat(),
        "training_rows": int(len(feature_frame)),
        "forecasting_model": "random-forest-autoregressive",
    }


def run_forecast_from_bundle(
    bundle: dict[str, Any],
    steps: int,
    horizon_minutes: int,
) -> dict[str, Any]:
    """Produce recursive future predictions from a stored forecasting bundle."""
    forecasting = bundle.get("forecasting")
    if not forecasting or not forecasting.get("enabled"):
        raise ValueError("Forecasting is unavailable for this model.")

    if steps < 1:
        raise ValueError("Forecast steps must be at least 1.")
    if horizon_minutes < 1:
        raise ValueError("Forecast horizon must be at least 1 minute.")

    history_targets = list(forecasting["history_targets"])
    required_history = int(forecasting["required_history"])
    if len(history_targets) < required_history:
        raise ValueError("Stored history is too short for forecasting.")

    lag_steps = list(forecasting["lag_steps"])
    rolling_windows = list(forecasting["rolling_windows"])
    feature_columns = list(forecasting["feature_columns"])
    current_timestamp = pd.Timestamp(forecasting["last_timestamp"])
    model = forecasting["model"]

    future_points = []
    for step_index in range(1, steps + 1):
        current_timestamp = current_timestamp + pd.Timedelta(minutes=horizon_minutes)
        feature_row = build_forecast_feature_row(
            timestamp=current_timestamp,
            history_targets=history_targets,
            lag_steps=lag_steps,
            rolling_windows=rolling_windows,
        )
        aligned = pd.DataFrame([[feature_row[column] for column in feature_columns]], columns=feature_columns)
        prediction = float(model.predict(aligned)[0])
        future_points.append(
            {
                "step": step_index,
                "timestamp": current_timestamp.isoformat(),
                "prediction": round(prediction, 6),
            }
        )
        history_targets.append(prediction)
        if len(history_targets) > max(required_history, HISTORY_TAIL_ROWS):
            history_targets = history_targets[-max(required_history, HISTORY_TAIL_ROWS) :]

    base_frequency = int(forecasting["base_frequency_minutes"])
    warning = None
    if base_frequency > 0 and horizon_minutes != base_frequency:
        warning = (
            f"Requested horizon ({horizon_minutes} min) differs from training cadence "
            f"({base_frequency} min). Short-horizon extrapolation may be less stable."
        )

    return {
        "target": forecasting["target"],
        "steps": steps,
        "requested_horizon_minutes": horizon_minutes,
        "base_frequency_minutes": base_frequency,
        "forecasting_model": forecasting["forecasting_model"],
        "forecast_metrics": forecasting["metrics"],
        "recent_history": list(forecasting["recent_history"]),
        "forecast": future_points,
        "warning": warning,
    }


def build_time_series_frame(
    frame: pd.DataFrame,
    target: str,
    descriptor: TimestampDescriptor,
) -> pd.DataFrame | None:
    """Extract a clean timestamp-target frame from the raw dataset."""
    timestamps = build_timestamp_series(frame=frame, descriptor=descriptor)
    target_values = pd.to_numeric(frame[target], errors="coerce")
    series_frame = pd.DataFrame({"timestamp": timestamps, target: target_values})
    series_frame = series_frame.dropna(subset=["timestamp", target]).sort_values("timestamp")
    if series_frame.empty:
        return None
    series_frame = series_frame.groupby("timestamp", as_index=False).last()
    if len(series_frame) < 2:
        return None
    return series_frame.reset_index(drop=True)


def infer_timestamp_descriptor(frame: pd.DataFrame, target: str) -> TimestampDescriptor | None:
    """Detect the most plausible timestamp encoding in the dataset."""
    direct_candidate = infer_direct_timestamp_descriptor(frame=frame, target=target)
    if direct_candidate is not None:
        return direct_candidate
    return infer_date_hour_timestamp_descriptor(frame=frame, target=target)


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse heterogeneous datetime strings without noisy inference warnings."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    try:
        return pd.to_datetime(series, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce")


def infer_direct_timestamp_descriptor(frame: pd.DataFrame, target: str) -> TimestampDescriptor | None:
    """Detect a single datetime-like column."""
    best_column = None
    best_score = 0.0
    for column in frame.columns:
        if column == target:
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        parsed = parse_datetime_series(series)
        score = float(parsed.notna().mean())
        if score >= 0.8 and parsed.nunique(dropna=True) > 1 and score > best_score:
            best_column = column
            best_score = score

    if best_column is None:
        return None
    return TimestampDescriptor(strategy="datetime_column", timestamp_column=best_column)


def infer_date_hour_timestamp_descriptor(frame: pd.DataFrame, target: str) -> TimestampDescriptor | None:
    """Detect a pair of date + hour columns."""
    date_candidates: list[str] = []
    hour_candidates: list[tuple[str, int]] = []

    for column in frame.columns:
        if column == target:
            continue

        lower_name = str(column).lower()
        series = frame[column]
        if "час" in lower_name or "hour" in lower_name:
            numeric = pd.to_numeric(series, errors="coerce")
            valid = numeric.dropna()
            if valid.empty:
                continue
            if valid.between(0, 23).mean() >= 0.8:
                hour_candidates.append((column, 0))
                continue
            if valid.between(1, 24).mean() >= 0.8:
                hour_candidates.append((column, 1))
                continue

        if pd.api.types.is_numeric_dtype(series):
            continue
        parsed = parse_datetime_series(series)
        if parsed.notna().mean() >= 0.8 and parsed.nunique(dropna=True) > 1:
            date_candidates.append(column)

    for date_column in date_candidates:
        for hour_column, hour_offset in hour_candidates:
            descriptor = TimestampDescriptor(
                strategy="date_hour_columns",
                date_column=date_column,
                hour_column=hour_column,
                hour_offset=hour_offset,
            )
            timestamps = build_timestamp_series(frame=frame, descriptor=descriptor)
            if timestamps.notna().mean() >= 0.8 and timestamps.nunique(dropna=True) > 1:
                return descriptor
    return None


def build_timestamp_series(frame: pd.DataFrame, descriptor: TimestampDescriptor) -> pd.Series:
    """Construct timestamps according to the stored descriptor."""
    if descriptor.strategy == "datetime_column":
        return parse_datetime_series(frame[descriptor.timestamp_column])

    if descriptor.strategy == "date_hour_columns":
        base = parse_datetime_series(frame[descriptor.date_column]).dt.normalize()
        hour_values = pd.to_numeric(frame[descriptor.hour_column], errors="coerce")
        adjusted_hours = hour_values - descriptor.hour_offset
        adjusted_hours = adjusted_hours.where(adjusted_hours.between(0, 23))
        return base + pd.to_timedelta(adjusted_hours, unit="h")

    raise ValueError(f"Unsupported timestamp descriptor strategy: {descriptor.strategy}")


def infer_frequency_minutes(timestamp_series: pd.Series) -> int:
    """Infer the dominant cadence of the training series in whole minutes."""
    diffs = timestamp_series.sort_values().diff().dropna()
    positive = diffs[diffs > pd.Timedelta(0)]
    if positive.empty:
        return 0
    median_seconds = positive.dt.total_seconds().median()
    return max(1, int(round(float(median_seconds) / 60.0)))


def build_forecasting_feature_frame(
    series_frame: pd.DataFrame,
    target: str,
    lag_steps: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """Turn a target series into a supervised learning table for recursive forecasting."""
    timestamps = series_frame["timestamp"].tolist()
    target_values = [float(value) for value in series_frame[target].tolist()]
    required_history = max([2] + lag_steps + rolling_windows)

    rows = []
    for index in range(required_history, len(series_frame)):
        row = build_forecast_feature_row(
            timestamp=timestamps[index],
            history_targets=target_values[:index],
            lag_steps=lag_steps,
            rolling_windows=rolling_windows,
        )
        row["timestamp"] = timestamps[index]
        row[target] = target_values[index]
        rows.append(row)

    return pd.DataFrame(rows)


def build_forecast_feature_row(
    timestamp: pd.Timestamp,
    history_targets: list[float],
    lag_steps: list[int],
    rolling_windows: list[int],
) -> dict[str, float]:
    """Build one autoregressive feature row for a future timestamp."""
    row = time_features(timestamp)

    for lag in lag_steps:
        row[f"target_lag_{lag}"] = float(history_targets[-lag])

    for window in rolling_windows:
        window_values = np.asarray(history_targets[-window:], dtype=float)
        row[f"target_roll_mean_{window}"] = float(window_values.mean())
        row[f"target_roll_std_{window}"] = float(window_values.std())
        row[f"target_roll_min_{window}"] = float(window_values.min())
        row[f"target_roll_max_{window}"] = float(window_values.max())

    if len(history_targets) >= 2:
        row["target_diff_1"] = float(history_targets[-1] - history_targets[-2])
    else:
        row["target_diff_1"] = 0.0

    return row


def time_features(timestamp: pd.Timestamp) -> dict[str, float]:
    """Project a timestamp into cyclical and calendar features."""
    ts = pd.Timestamp(timestamp)
    hour_fraction = ts.hour + ts.minute / 60.0
    weekday = float(ts.dayofweek)
    day_of_year = float(ts.dayofyear)
    month = float(ts.month)

    return {
        "timestamp_unix": float(ts.value // 10**9),
        "hour": hour_fraction,
        "minute": float(ts.minute),
        "dayofweek": weekday,
        "dayofyear": day_of_year,
        "month": month,
        "is_weekend": float(ts.dayofweek >= 5),
        "hour_sin": float(np.sin(2.0 * np.pi * hour_fraction / 24.0)),
        "hour_cos": float(np.cos(2.0 * np.pi * hour_fraction / 24.0)),
        "weekday_sin": float(np.sin(2.0 * np.pi * weekday / 7.0)),
        "weekday_cos": float(np.cos(2.0 * np.pi * weekday / 7.0)),
        "month_sin": float(np.sin(2.0 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2.0 * np.pi * month / 12.0)),
    }
