#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path
from string import Template
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.dataset_service import DatasetService
from backend.services.explanation_service import ExplanationService
from backend.services.prediction_service import PredictionService
from backend.services.registry_service import RegistryService
from backend.services.settings import get_settings
from backend.utils.compat import patch_numpy_for_lightautoml
from backend.utils.io import pythonize
from dss.experta_engine.engine import DecisionEngine
from explainability.fact_builder.builder import build_facts


patch_numpy_for_lightautoml()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local HTML report generation."""
    parser = argparse.ArgumentParser(
        description="Generate an HTML report with charts for a trained project."
    )
    parser.add_argument("--project-id", required=True, help="Project id with a champion model.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=120,
        help="How many dataset rows to use for prediction/explanation charts.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional HTML output path. Defaults to storage/artifacts/<project>/visual-report.html.",
    )
    return parser.parse_args()


def figure_to_data_uri(fig) -> str:
    """Serialize a matplotlib figure into an embeddable data URI."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_metric_cards(metrics: dict[str, Any]) -> str:
    """Render metric cards for the HTML report header."""
    cards = []
    for name, value in metrics.items():
        cards.append(
            """
            <div class="metric-card">
              <div class="metric-name">{name}</div>
              <div class="metric-value">{value}</div>
            </div>
            """.format(
                name=html.escape(str(name)),
                value=html.escape(str(value)),
            )
        )
    return "".join(cards)


def render_top_factors(items: list[dict[str, Any]]) -> str:
    """Render explanation samples as HTML blocks."""
    blocks = []
    for item in items:
        factors = item.get("top_factors", [])
        if factors:
            points = "".join(
                """
                <li>
                  <strong>{feature}</strong>:
                  value={value},
                  direction={direction},
                  strength={strength},
                  impact={impact}
                </li>
                """.format(
                    feature=html.escape(str(factor["feature"])),
                    value=html.escape(str(factor.get("value"))),
                    direction=html.escape(str(factor.get("direction"))),
                    strength=html.escape(str(factor.get("strength"))),
                    impact=html.escape(str(factor.get("impact_score"))),
                )
                for factor in factors[:5]
            )
        else:
            points = "<li>Факторы не были выделены.</li>"

        blocks.append(
            """
            <div class="row-card">
              <div class="row-card-title">Row {row_index} | prediction={prediction}</div>
              <ul>{points}</ul>
            </div>
            """.format(
                row_index=html.escape(str(item["row_index"])),
                prediction=html.escape(str(item.get("prediction"))),
                points=points,
            )
        )
    return "".join(blocks)


def render_recommendations(items: list[dict[str, Any]]) -> str:
    """Render DSS recommendation samples as HTML blocks."""
    blocks = []
    for item in items:
        recommendation = item.get("recommendation", {})
        actions = recommendation.get("actions", [])
        rationale = recommendation.get("rationale", [])

        actions_html = "".join(f"<li>{html.escape(str(action))}</li>" for action in actions) or "<li>Нет действий</li>"
        rationale_html = "".join(f"<li>{html.escape(str(point))}</li>" for point in rationale) or "<li>Нет пояснений</li>"

        blocks.append(
            """
            <div class="row-card">
              <div class="row-card-title">Row {row_index} | risk={risk}</div>
              <p>{summary}</p>
              <div class="row-card-subtitle">Actions</div>
              <ul>{actions}</ul>
              <div class="row-card-subtitle">Rationale</div>
              <ul>{rationale}</ul>
            </div>
            """.format(
                row_index=html.escape(str(item["row_index"])),
                risk=html.escape(str(recommendation.get("risk_level"))),
                summary=html.escape(str(recommendation.get("summary"))),
                actions=actions_html,
                rationale=rationale_html,
            )
        )
    return "".join(blocks)


def plot_target_distribution(frame: pd.DataFrame, target: str, task_type: str) -> str:
    """Build a target-distribution chart and return it as a data URI."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    series = frame[target]

    if task_type == "regression":
        ax.hist(series, bins=30, color="#0f766e", edgecolor="white")
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
    else:
        counts = series.astype(str).value_counts().sort_index()
        ax.bar(counts.index, counts.values, color="#0f766e")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")

    ax.set_title("Target Distribution")
    fig.tight_layout()
    return figure_to_data_uri(fig)


def plot_feature_importances(bundle: dict[str, Any]) -> str:
    """Build the feature-importance chart for the report."""
    importances = bundle.get("feature_importances", {})
    ordered = sorted(importances.items(), key=lambda item: float(item[1]), reverse=True)[:12]

    fig, ax = plt.subplots(figsize=(9, 5))
    if ordered:
        features = [name for name, _ in ordered][::-1]
        values = [float(value) for _, value in ordered][::-1]
        ax.barh(features, values, color="#c2410c")
    else:
        ax.text(0.5, 0.5, "No feature importances available", ha="center", va="center")
        ax.set_axis_off()

    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    return figure_to_data_uri(fig)


def plot_predictions(sample_frame: pd.DataFrame, predictions: list[dict[str, Any]], target: str, task_type: str) -> str:
    """Build a chart that compares actual and predicted values."""
    predicted = [pythonize(item["prediction"]) for item in predictions]
    actual = sample_frame[target].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    if task_type == "regression":
        ax.scatter(actual, predicted, color="#1d4ed8", alpha=0.7)
        low = min(min(actual), min(predicted))
        high = max(max(actual), max(predicted))
        ax.plot([low, high], [low, high], linestyle="--", color="#7c3aed", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
    else:
        comparison = pd.DataFrame({"actual": actual, "predicted": predicted})
        matrix = pd.crosstab(comparison["actual"], comparison["predicted"])
        ax.imshow(matrix.values, cmap="YlGnBu")
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Prediction Matrix")

    fig.tight_layout()
    return figure_to_data_uri(fig)


def plot_first_explanation(explanations: dict[str, Any]) -> str:
    """Build a chart for the first explanation payload in the sample."""
    items = explanations.get("items", [])
    factors = items[0].get("top_factors", []) if items else []

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if factors:
        labels = [str(factor["feature"]) for factor in factors][::-1]
        values = [float(factor["impact_score"]) for factor in factors][::-1]
        colors = ["#15803d" if factor.get("direction") == "positive" else "#b91c1c" for factor in factors][::-1]
        ax.barh(labels, values, color=colors)
    else:
        ax.text(0.5, 0.5, "No explanation factors available", ha="center", va="center")
        ax.set_axis_off()

    ax.set_title("Top Factors For First Explained Row")
    fig.tight_layout()
    return figure_to_data_uri(fig)


def load_project_context(project_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Load the champion model, dataset record and source frame for a project."""
    registry_service = RegistryService()
    champion, bundle = registry_service.get_champion_bundle(project_id)
    datasets = registry_service.registry.read("datasets")

    dataset_record = None
    for item in datasets:
        if item["version_id"] == champion["dataset_version_id"]:
            dataset_record = item
            break
    if dataset_record is None:
        raise ValueError(f"Dataset version '{champion['dataset_version_id']}' was not found.")

    frame = DatasetService.read_tabular_file(dataset_record["path"])
    return champion, bundle, dataset_record, frame


def main() -> int:
    """Generate an HTML report with charts for the current champion model."""
    args = parse_args()
    champion, bundle, dataset_record, frame = load_project_context(args.project_id)
    target = champion["target"]
    task_type = champion["task_type"]

    sample_size = max(1, min(args.sample_size, len(frame)))
    sample_frame = frame.head(sample_size).copy()
    inference_records = sample_frame.drop(columns=[target]).to_dict(orient="records")

    prediction_service = PredictionService()
    explanation_service = ExplanationService()
    decision_engine = DecisionEngine()

    predictions = prediction_service.predict_with_bundle(bundle=bundle, frame=sample_frame.drop(columns=[target]))
    explanations = explanation_service.explain_with_bundle(
        bundle=bundle,
        frame=sample_frame.drop(columns=[target]),
        predictions=predictions,
    )
    decision_items = []
    for explanation in explanations["items"]:
        facts = build_facts(
            task_type=task_type,
            prediction=explanation["prediction"],
            confidence=explanation["confidence"],
            top_factors=explanation["top_factors"],
        )
        decision_items.append(
            {
                "row_index": explanation["row_index"],
                "prediction": explanation["prediction"],
                "confidence": explanation["confidence"],
                "facts": facts,
                "recommendation": decision_engine.recommend(facts),
                "top_factors": explanation["top_factors"],
            }
        )

    target_distribution_chart = plot_target_distribution(frame, target=target, task_type=task_type)
    feature_importance_chart = plot_feature_importances(bundle)
    prediction_chart = plot_predictions(sample_frame, predictions=predictions, target=target, task_type=task_type)
    explanation_chart = plot_first_explanation(explanations)

    settings = get_settings()
    default_output = settings.artifacts_dir / args.project_id / "visual-report.html"
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_payload = Template("""
    <!doctype html>
    <html lang="ru">
      <head>
        <meta charset="utf-8" />
        <title>AdaptiveML DSS Report</title>
        <style>
          :root {
            --bg: #f3efe5;
            --panel: #fffaf2;
            --ink: #1f2937;
            --muted: #6b7280;
            --line: #d6c9ad;
            --accent: #0f766e;
            --accent-2: #c2410c;
          }
          body {
            margin: 0;
            font-family: Georgia, "Times New Roman", serif;
            color: var(--ink);
            background:
              radial-gradient(circle at top right, rgba(194,65,12,.10), transparent 28%),
              radial-gradient(circle at left 20%, rgba(15,118,110,.12), transparent 35%),
              var(--bg);
          }
          .page {
            max-width: 1200px;
            margin: 0 auto;
            padding: 32px 20px 56px;
          }
          .hero {
            background: linear-gradient(135deg, rgba(255,250,242,.96), rgba(247,240,225,.96));
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 12px 32px rgba(31, 41, 55, 0.08);
          }
          .hero h1 {
            margin: 0 0 8px;
            font-size: 36px;
          }
          .hero p {
            margin: 6px 0;
            color: var(--muted);
          }
          .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-top: 20px;
          }
          .metric-card, .panel, .row-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(31, 41, 55, 0.05);
          }
          .metric-card {
            padding: 18px;
          }
          .metric-name {
            color: var(--muted);
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: .08em;
          }
          .metric-value {
            font-size: 28px;
            margin-top: 10px;
          }
          .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 18px;
            margin-top: 24px;
          }
          .panel {
            padding: 18px;
          }
          .panel h2 {
            margin: 0 0 14px;
            font-size: 20px;
          }
          .panel img {
            width: 100%;
            border-radius: 14px;
            display: block;
          }
          .stack {
            display: grid;
            gap: 16px;
            margin-top: 24px;
          }
          .row-card {
            padding: 16px 18px;
          }
          .row-card-title {
            font-weight: bold;
            margin-bottom: 8px;
          }
          .row-card-subtitle {
            margin-top: 10px;
            color: var(--muted);
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: .08em;
          }
          ul {
            margin: 8px 0 0 18px;
            padding: 0;
          }
          pre {
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
          }
        </style>
      </head>
      <body>
        <div class="page">
          <section class="hero">
            <h1>AdaptiveML DSS Visual Report</h1>
            <p>project_id=$project_id</p>
            <p>model_version=$model_version | task_type=$task_type | backend=$backend</p>
            <p>dataset=$dataset_version | source=$source_name | rows=$rows</p>
            <div class="metrics">$metric_cards</div>
          </section>

          <section class="grid">
            <div class="panel">
              <h2>Target Distribution</h2>
              <img src="$target_distribution_chart" alt="Target Distribution" />
            </div>
            <div class="panel">
              <h2>Feature Importance</h2>
              <img src="$feature_importance_chart" alt="Feature Importance" />
            </div>
            <div class="panel">
              <h2>Actual vs Predicted</h2>
              <img src="$prediction_chart" alt="Prediction Chart" />
            </div>
            <div class="panel">
              <h2>Top Factors</h2>
              <img src="$explanation_chart" alt="Explanation Chart" />
            </div>
          </section>

          <section class="stack">
            <div class="panel">
              <h2>Prediction / Explanation Sample</h2>
              $top_factors
            </div>
            <div class="panel">
              <h2>DSS Recommendations</h2>
              $recommendations
            </div>
            <div class="panel">
              <h2>Raw Metadata</h2>
              <pre>$metadata</pre>
            </div>
          </section>
        </div>
      </body>
    </html>
    """).substitute(
        project_id=html.escape(args.project_id),
        model_version=html.escape(str(champion["version_id"])),
        task_type=html.escape(str(task_type)),
        backend=html.escape(str(bundle.get("backend_name", "unknown"))),
        dataset_version=html.escape(str(dataset_record["version_id"])),
        source_name=html.escape(str(dataset_record["source_name"])),
        rows=html.escape(str(dataset_record["rows"])),
        metric_cards=render_metric_cards(champion["metrics"]),
        target_distribution_chart=target_distribution_chart,
        feature_importance_chart=feature_importance_chart,
        prediction_chart=prediction_chart,
        explanation_chart=explanation_chart,
        top_factors=render_top_factors(explanations["items"][:5]),
        recommendations=render_recommendations(decision_items[:5]),
        metadata=html.escape(
            json.dumps(
                {
                    "feature_names": champion["feature_names"],
                    "schema": dataset_record["schema"],
                    "sample_size": sample_size,
                },
                ensure_ascii=False,
                indent=2,
            )
        ),
    )

    output_path.write_text(html_payload, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
