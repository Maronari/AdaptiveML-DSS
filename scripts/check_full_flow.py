#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path
from uuid import uuid4

import httpx
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.dataset_service import DatasetService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end API smoke check: train -> predict -> explain -> decision."
    )
    parser.add_argument("--data", required=True, help="Path to input CSV/XLSX dataset.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running API.",
    )
    parser.add_argument(
        "--project-id",
        default="",
        help="Optional project id. Defaults to a generated smoke id.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="How many rows to use for predict/explain/decision requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ")
    if pd.isna(value):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return value


def records_to_json_ready(frame) -> list[dict]:
    records = frame.to_dict(orient="records")
    return [{key: json_ready(value) for key, value in record.items()} for record in records]


def ensure_ok(response: httpx.Response, step: str) -> dict:
    try:
        payload = response.json()
    except Exception as exc:
        raise SystemExit(f"{step} returned non-JSON response with status {response.status_code}.") from exc

    if response.status_code >= 400:
        detail = payload.get("detail", payload)
        raise SystemExit(f"{step} failed with status {response.status_code}: {detail}")
    return payload


def validate_summary(training: dict, prediction: dict, explanation: dict, decision: dict, sample_size: int) -> None:
    if "model_version" not in training or "backend" not in training:
        raise SystemExit("Training response is missing expected fields.")
    if len(prediction.get("predictions", [])) != sample_size:
        raise SystemExit("Prediction response size does not match requested sample size.")
    if len(explanation.get("items", [])) != sample_size:
        raise SystemExit("Explanation response size does not match requested sample size.")
    if len(decision.get("recommendations", [])) != sample_size:
        raise SystemExit("Decision response size does not match requested sample size.")


def main() -> int:
    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")
    if args.sample_size < 1:
        raise SystemExit("--sample-size must be greater than 0.")

    frame = DatasetService.read_tabular_file(data_path)
    DatasetService._validate_training_frame(frame, args.target)

    available_records = len(frame)
    sample_size = min(args.sample_size, available_records)
    if sample_size < args.sample_size:
        print(
            f"Requested sample_size={args.sample_size}, using {sample_size} because the dataset is smaller.",
            file=sys.stderr,
        )

    inference_records = records_to_json_ready(frame.drop(columns=[args.target]).head(sample_size))
    project_id = args.project_id or f"smoke-{data_path.stem}-{uuid4().hex[:8]}"
    content_type = mimetypes.guess_type(str(data_path))[0] or "application/octet-stream"

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
        health = ensure_ok(client.get("/health"), "Healthcheck")

        with data_path.open("rb") as source:
            training = ensure_ok(
                client.post(
                    "/training/run/file",
                    files={"file": (data_path.name, source, content_type)},
                    data={"project_id": project_id, "target": args.target},
                ),
                "Training",
            )

        prediction = ensure_ok(
            client.post(
                "/predictions/run",
                json={"project_id": project_id, "records": inference_records},
            ),
            "Prediction",
        )
        explanation = ensure_ok(
            client.post(
                "/explanations/run",
                json={"project_id": project_id, "records": inference_records},
            ),
            "Explanation",
        )
        decision = ensure_ok(
            client.post(
                "/decision/run",
                json={"project_id": project_id, "records": inference_records},
            ),
            "Decision",
        )

    validate_summary(training, prediction, explanation, decision, sample_size)

    summary = {
        "health": health,
        "project_id": project_id,
        "data": str(data_path),
        "target": args.target,
        "sample_size": sample_size,
        "training": {
            "backend": training.get("backend"),
            "task_type": training.get("task_type"),
            "model_version": training.get("model_version", {}).get("version_id"),
            "primary_metric": training.get("metrics", {}).get(training.get("primary_metric", "")),
        },
        "prediction_rows": len(prediction["predictions"]),
        "explanation_rows": len(explanation["items"]),
        "decision_rows": len(decision["recommendations"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
