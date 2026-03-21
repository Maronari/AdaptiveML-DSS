const state = {
  validation: null,
  training: null,
  retraining: null,
  comparison: null,
  forecast: null,
  busy: false,
};

const elements = {
  fileInput: document.getElementById("dataset-file"),
  projectIdInput: document.getElementById("project-id"),
  targetInput: document.getElementById("target-column"),
  taskTypeInput: document.getElementById("task-type"),
  backendInput: document.getElementById("training-backend"),
  presetInput: document.getElementById("training-preset"),
  cvFoldsInput: document.getElementById("cv-folds"),
  timeoutSecondsInput: document.getElementById("timeout-seconds"),
  cpuLimitInput: document.getElementById("cpu-limit"),
  testSizeInput: document.getElementById("test-size"),
  enableForecastInput: document.getElementById("enable-forecast"),
  retrainHistoryScopeInput: document.getElementById("retrain-history-scope"),
  retrainImprovementThresholdInput: document.getElementById("retrain-improvement-threshold"),
  retrainAutoActivateInput: document.getElementById("retrain-auto-activate"),
  algoInputs: Array.from(document.querySelectorAll('input[name="algo-option"]')),
  trainingOptionsNote: document.getElementById("training-options-note"),
  forecastMinutesInput: document.getElementById("forecast-minutes"),
  forecastStepsInput: document.getElementById("forecast-steps"),
  validateButton: document.getElementById("validate-button"),
  trainButton: document.getElementById("train-button"),
  retrainButton: document.getElementById("retrain-button"),
  compareButton: document.getElementById("compare-button"),
  forecastButton: document.getElementById("forecast-button"),
  trainCompareButton: document.getElementById("train-compare-button"),
  statusBanner: document.getElementById("status-banner"),
  summaryCards: document.getElementById("summary-cards"),
  datasetPreview: document.getElementById("dataset-preview"),
  comparisonTable: document.getElementById("comparison-table"),
  comparisonCanvas: document.getElementById("prediction-chart"),
  comparisonChartEmptyState: document.getElementById("chart-empty-state"),
  comparisonChartTitle: document.getElementById("chart-title"),
  forecastCanvas: document.getElementById("forecast-chart"),
  forecastChartEmptyState: document.getElementById("forecast-empty-state"),
  forecastChartTitle: document.getElementById("forecast-chart-title"),
  forecastTable: document.getElementById("forecast-table"),
  forecastWarning: document.getElementById("forecast-warning"),
};

function selectedAlgorithms() {
  return elements.algoInputs.filter((input) => input.checked).map((input) => input.value);
}

function getWorkflowState() {
  return {
    file: elements.fileInput.files?.[0] ?? null,
    projectId: elements.projectIdInput.value.trim() || "demo",
    target: elements.targetInput.value.trim(),
    trainingOptions: {
      task_type: elements.taskTypeInput.value,
      backend: elements.backendInput.value,
      preset: elements.presetInput.value,
      algos: selectedAlgorithms(),
      timeout_seconds: Number.parseInt(elements.timeoutSecondsInput.value || "30", 10),
      cpu_limit: Number.parseInt(elements.cpuLimitInput.value || "1", 10),
      test_size: Number.parseFloat(elements.testSizeInput.value || "0.2"),
      cv_folds: Number.parseInt(elements.cvFoldsInput.value || "0", 10),
      enable_forecast: elements.enableForecastInput.checked,
    },
    retrainingOptions: {
      history_scope: elements.retrainHistoryScopeInput.value,
      minimum_relative_improvement:
        Number.parseFloat(elements.retrainImprovementThresholdInput.value || "3") / 100,
      auto_activate: elements.retrainAutoActivateInput.checked,
    },
    horizonMinutes: Number.parseInt(elements.forecastMinutesInput.value || "30", 10),
    forecastSteps: Number.parseInt(elements.forecastStepsInput.value || "1", 10),
  };
}

function setBusy(isBusy) {
  state.busy = isBusy;
  for (const button of [
    elements.validateButton,
    elements.trainButton,
    elements.retrainButton,
    elements.compareButton,
    elements.forecastButton,
    elements.trainCompareButton,
  ]) {
    button.disabled = isBusy;
  }
}

function setStatus(kind, message) {
  elements.statusBanner.className = `status-banner status-${kind}`;
  elements.statusBanner.textContent = message;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function normalizeError(error) {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error ?? "Unknown error");
}

function formatMetricValue(value) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return String(value);
}

function requireFileAndTarget() {
  const workflow = getWorkflowState();
  if (!workflow.file) {
    throw new Error("Select a CSV or XLSX file.");
  }
  if (!workflow.target) {
    throw new Error("Provide the target column name.");
  }
  return workflow;
}

function requireTrainingWorkflow() {
  const workflow = requireFileAndTarget();
  if (!Number.isFinite(workflow.trainingOptions.timeout_seconds) || workflow.trainingOptions.timeout_seconds < 5) {
    throw new Error("Training timeout must be at least 5 seconds.");
  }
  if (!Number.isFinite(workflow.trainingOptions.cpu_limit) || workflow.trainingOptions.cpu_limit < 1) {
    throw new Error("CPU limit must be at least 1.");
  }
  if (!Number.isFinite(workflow.trainingOptions.test_size)) {
    throw new Error("Holdout test size must be a number.");
  }
  if (workflow.trainingOptions.test_size <= 0 || workflow.trainingOptions.test_size >= 1) {
    throw new Error("Holdout test size must be greater than 0 and less than 1.");
  }
  if (!Number.isFinite(workflow.trainingOptions.cv_folds) || workflow.trainingOptions.cv_folds < 0) {
    throw new Error("CV folds must be 0 for auto or a positive number.");
  }
  if (workflow.trainingOptions.cv_folds === 1) {
    throw new Error("CV folds must be 0 for auto or at least 2.");
  }
  if (workflow.trainingOptions.backend !== "sklearn" && workflow.trainingOptions.algos.length === 0) {
    throw new Error("Select at least one LightAutoML algorithm.");
  }
  return workflow;
}

function requireRetrainingWorkflow() {
  const workflow = requireTrainingWorkflow();
  if (!Number.isFinite(workflow.retrainingOptions.minimum_relative_improvement)) {
    throw new Error("Minimum retraining profit must be a number.");
  }
  if (workflow.retrainingOptions.minimum_relative_improvement < 0) {
    throw new Error("Minimum retraining profit must be greater than or equal to 0.");
  }
  return workflow;
}

function requireForecastParams() {
  const workflow = getWorkflowState();
  if (!workflow.projectId) {
    throw new Error("Provide a project id.");
  }
  if (!Number.isFinite(workflow.horizonMinutes) || workflow.horizonMinutes < 1) {
    throw new Error("Forecast horizon must be at least 1 minute.");
  }
  if (!Number.isFinite(workflow.forecastSteps) || workflow.forecastSteps < 1) {
    throw new Error("Forecast steps must be at least 1.");
  }
  return workflow;
}

async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function extractErrorMessage(payload, response) {
  if (payload && typeof payload === "object" && "detail" in payload) {
    return payload.detail;
  }
  if (typeof payload === "string" && payload.trim()) {
    return payload;
  }
  return response.statusText || "Request failed.";
}

async function postFile(
  path,
  { file, projectId, target, includeTarget = false, trainingOptions = null, retrainingOptions = null },
) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("project_id", projectId);
  if (includeTarget) {
    formData.append("target", target);
  }
  if (trainingOptions) {
    formData.append("task_type", trainingOptions.task_type);
    formData.append("backend", trainingOptions.backend);
    formData.append("preset", trainingOptions.preset);
    formData.append("algos", trainingOptions.algos.join(","));
    formData.append("timeout_seconds", String(trainingOptions.timeout_seconds));
    formData.append("cpu_limit", String(trainingOptions.cpu_limit));
    formData.append("test_size", String(trainingOptions.test_size));
    formData.append("cv_folds", String(trainingOptions.cv_folds));
    formData.append("enable_forecast", String(trainingOptions.enable_forecast));
  }
  if (retrainingOptions) {
    formData.append("history_scope", retrainingOptions.history_scope);
    formData.append(
      "minimum_relative_improvement",
      String(retrainingOptions.minimum_relative_improvement),
    );
    formData.append("auto_activate", String(retrainingOptions.auto_activate));
  }

  const response = await fetch(path, {
    method: "POST",
    body: formData,
  });
  const payload = await readJsonResponse(response);

  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, response));
  }
  return payload;
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  const body = await readJsonResponse(response);

  if (!response.ok) {
    throw new Error(extractErrorMessage(body, response));
  }
  return body;
}

function buildSummaryCards() {
  const cards = [];

  if (state.validation) {
    cards.push({ label: "Rows", value: state.validation.rows });
    cards.push({ label: "Columns", value: state.validation.columns.length });
    cards.push({ label: "Task Type", value: state.validation.task_type });
  }

  if (state.training) {
    const effectiveOptions = state.training.training_options?.effective ?? null;
    cards.push({ label: "Backend", value: state.training.backend });
    cards.push({ label: "Model Version", value: state.training.model_version.version_id });
    if (effectiveOptions) {
      cards.push({ label: "Task", value: effectiveOptions.task_type });
      cards.push({ label: "Preset", value: effectiveOptions.preset });
      cards.push({
        label: "Algos",
        value: effectiveOptions.algos?.length ? effectiveOptions.algos.join(", ") : "n/a",
      });
    }
    cards.push({
      label: "Forecast Head",
      value: state.training.forecasting?.available ? "available" : "not available",
    });
    if (state.training.forecasting?.base_frequency_minutes) {
      cards.push({
        label: "Cadence",
        value: `${state.training.forecasting.base_frequency_minutes} min`,
      });
    }
    if (state.training.warnings?.length) {
      cards.push({
        label: "Warnings",
        value: state.training.warnings.length,
      });
    }
  }

  if (state.retraining) {
    const profit = state.retraining.evaluation?.profit ?? null;
    cards.push({ label: "Current Model", value: state.retraining.current_model_version });
    cards.push({
      label: "Candidate Model",
      value: state.retraining.candidate_model_version.version_id,
    });
    cards.push({
      label: "History Scope",
      value: state.retraining.selection_summary.history_scope,
    });
    cards.push({
      label: "Eval Rows",
      value: state.retraining.evaluation.rows,
    });
    cards.push({
      label: "Activated",
      value: state.retraining.activated ? "yes" : "no",
    });
    if (profit) {
      cards.push({
        label: `Current ${profit.primary_metric}`,
        value: formatMetricValue(profit.current_value),
      });
      cards.push({
        label: `Candidate ${profit.primary_metric}`,
        value: formatMetricValue(profit.candidate_value),
      });
      cards.push({
        label: "Profit",
        value: `${formatMetricValue(profit.relative_gain_percent)}%`,
      });
    }
  }

  if (state.comparison) {
    cards.push({ label: "Compared Rows", value: state.comparison.rows });
    for (const [name, value] of Object.entries(state.comparison.metrics ?? {})) {
      cards.push({ label: name, value: formatMetricValue(value) });
    }
  }

  if (state.forecast) {
    cards.push({
      label: "Forecast Horizon",
      value: `${state.forecast.requested_horizon_minutes} min`,
    });
    cards.push({
      label: "Forecast Steps",
      value: state.forecast.steps,
    });
    const firstPoint = state.forecast.forecast?.[0];
    if (firstPoint) {
      cards.push({
        label: "Next Value",
        value: formatMetricValue(firstPoint.prediction),
      });
    }
  }

  if (cards.length === 0) {
    elements.summaryCards.innerHTML = `
      <article class="summary-card summary-card-empty">
        <span>Run validate, train, compare, or forecast to populate this panel.</span>
      </article>
    `;
    return;
  }

  elements.summaryCards.innerHTML = cards
    .map(
      (card) => `
        <article class="summary-card">
          <div class="summary-card-label">${escapeHtml(card.label)}</div>
          <div class="summary-card-value">${escapeHtml(card.value)}</div>
        </article>
      `,
    )
    .join("");
}

function renderTable(container, rows, columns, options = {}) {
  const visibleRows = rows.slice(0, options.limit ?? rows.length);
  if (!visibleRows.length || !columns.length) {
    container.className = "table-shell table-placeholder";
    container.textContent = options.emptyText ?? "No data to display.";
    return;
  }

  const headerHtml = columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("");
  const bodyHtml = visibleRows
    .map((row) => {
      const cells = columns
        .map((column) => {
          const value = row[column];
          const rendered =
            column === "confidence" && value !== null && value !== undefined
              ? `<span class="confidence-pill">${escapeHtml(formatMetricValue(value))}</span>`
              : escapeHtml(value ?? "-");
          return `<td>${rendered}</td>`;
        })
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  container.className = "table-shell";
  container.innerHTML = `
    <table>
      <thead>
        <tr>${headerHtml}</tr>
      </thead>
      <tbody>
        ${bodyHtml}
      </tbody>
    </table>
  `;
}

function renderPreview() {
  if (state.comparison?.items?.length) {
    const rows = state.comparison.items.map((item) => ({
      row_index: item.row_index,
      ...item.record,
      prediction: item.prediction,
      confidence: item.confidence,
    }));
    const columns = ["row_index", ...state.comparison.columns, "prediction", "confidence"];
    renderTable(elements.datasetPreview, rows, columns, {
      limit: 12,
      emptyText: "No rows available for preview.",
    });
    return;
  }

  if (state.validation?.sample_rows?.length) {
    renderTable(elements.datasetPreview, state.validation.sample_rows, state.validation.columns, {
      limit: 12,
      emptyText: "Validation did not return sample rows.",
    });
    return;
  }

  elements.datasetPreview.className = "table-shell table-placeholder";
  elements.datasetPreview.textContent = "Upload a dataset to inspect sample rows.";
}

function renderComparisonTable() {
  if (!state.comparison?.items?.length) {
    elements.comparisonTable.className = "table-shell table-placeholder";
    elements.comparisonTable.textContent = "The merged dataset and prediction table will appear here.";
    return;
  }

  const rows = state.comparison.items.map((item) => ({
    row_index: item.row_index,
    ...item.record,
    prediction: item.prediction,
    confidence: item.confidence,
  }));
  const columns = ["row_index", ...state.comparison.columns, "prediction", "confidence"];
  renderTable(elements.comparisonTable, rows, columns, {
    limit: 40,
    emptyText: "Comparison results are empty.",
  });
}

function renderForecastTable() {
  if (!state.forecast?.forecast?.length) {
    elements.forecastTable.className = "table-shell table-placeholder";
    elements.forecastTable.textContent = "Future forecast rows will appear here after the forecast call.";
    return;
  }

  renderTable(elements.forecastTable, state.forecast.forecast, ["step", "timestamp", "prediction"], {
    limit: 24,
    emptyText: "Forecast response did not contain future rows.",
  });
}

function prepareCanvas(canvas, height) {
  const parentWidth = canvas.parentElement.clientWidth;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(parentWidth * ratio);
  canvas.height = Math.floor(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {
    context,
    width: parentWidth,
    height,
  };
}

function clearCanvas(canvas, height) {
  const { context, width, height: actualHeight } = prepareCanvas(canvas, height);
  context.clearRect(0, 0, width, actualHeight);
}

function downsample(items, maxPoints) {
  if (items.length <= maxPoints) {
    return items;
  }

  const sampled = [];
  const step = (items.length - 1) / (maxPoints - 1);
  for (let index = 0; index < maxPoints; index += 1) {
    sampled.push(items[Math.round(index * step)]);
  }
  return sampled;
}

function drawAxes(context, width, height, bounds, xLabel) {
  context.strokeStyle = "rgba(23, 32, 51, 0.24)";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(bounds.left, bounds.top);
  context.lineTo(bounds.left, bounds.bottom);
  context.lineTo(bounds.right, bounds.bottom);
  context.stroke();

  context.fillStyle = "#5e6778";
  context.font = '12px "Segoe UI Variable Text", "Trebuchet MS", sans-serif';
  context.fillText(xLabel, bounds.right - context.measureText(xLabel).width, bounds.bottom + 28);
}

function drawLegend(context, entries) {
  let x = 28;
  const y = 22;
  context.font = '12px "Segoe UI Variable Text", "Trebuchet MS", sans-serif';
  for (const entry of entries) {
    context.fillStyle = entry.color;
    context.fillRect(x, y - 10, 16, 8);
    x += 22;
    context.fillStyle = "#172033";
    context.fillText(entry.label, x, y);
    x += context.measureText(entry.label).width + 24;
  }
}

function drawComparisonRegressionChart(items) {
  const { context, width, height } = prepareCanvas(elements.comparisonCanvas, 420);
  const bounds = { top: 48, right: width - 26, bottom: height - 40, left: 58 };
  const sampled = downsample(items, 160);
  const actualValues = sampled.map((item) => Number(item.actual));
  const predictedValues = sampled.map((item) => Number(item.prediction));
  const values = actualValues.concat(predictedValues).filter((value) => Number.isFinite(value));

  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fbfaf7";
  context.fillRect(0, 0, width, height);

  if (!values.length) {
    context.fillStyle = "#5e6778";
    context.fillText("No numeric values available for the regression chart.", 24, 48);
    return;
  }

  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const span = maxValue - minValue || 1;
  const paddedMin = minValue - span * 0.08;
  const paddedMax = maxValue + span * 0.08;

  drawAxes(context, width, height, bounds, "Rows");
  drawLegend(context, [
    { label: "Actual", color: "#0f766e" },
    { label: "Predicted", color: "#c2410c" },
  ]);

  for (let tick = 0; tick <= 4; tick += 1) {
    const ratio = tick / 4;
    const y = bounds.bottom - (bounds.bottom - bounds.top) * ratio;
    const value = paddedMin + (paddedMax - paddedMin) * ratio;
    context.strokeStyle = "rgba(23, 32, 51, 0.08)";
    context.beginPath();
    context.moveTo(bounds.left, y);
    context.lineTo(bounds.right, y);
    context.stroke();
    context.fillStyle = "#5e6778";
    context.font = '12px "Segoe UI Variable Text", "Trebuchet MS", sans-serif';
    context.fillText(formatMetricValue(value), 8, y + 4);
  }

  function drawSeries(valuesToDraw, color) {
    context.strokeStyle = color;
    context.lineWidth = 2.5;
    context.beginPath();
    valuesToDraw.forEach((value, index) => {
      const x =
        bounds.left +
        ((bounds.right - bounds.left) * index) / Math.max(valuesToDraw.length - 1, 1);
      const y =
        bounds.bottom -
        ((value - paddedMin) / (paddedMax - paddedMin || 1)) * (bounds.bottom - bounds.top);
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
  }

  drawSeries(actualValues, "#0f766e");
  drawSeries(predictedValues, "#c2410c");
}

function drawClassificationMatrix(items) {
  const { context, width, height } = prepareCanvas(elements.comparisonCanvas, 420);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fbfaf7";
  context.fillRect(0, 0, width, height);

  const labels = Array.from(
    new Set(items.flatMap((item) => [String(item.actual), String(item.prediction)])),
  ).sort((left, right) => left.localeCompare(right, "ru"));

  const matrix = labels.map(() => labels.map(() => 0));
  for (const item of items) {
    const actualIndex = labels.indexOf(String(item.actual));
    const predictedIndex = labels.indexOf(String(item.prediction));
    matrix[actualIndex][predictedIndex] += 1;
  }

  const maxValue = Math.max(...matrix.flat(), 1);
  const gridSize = Math.min(width - 180, height - 120);
  const cellSize = gridSize / Math.max(labels.length, 1);
  const originX = 120;
  const originY = 70;

  context.font = '12px "Segoe UI Variable Text", "Trebuchet MS", sans-serif';
  context.fillStyle = "#172033";
  context.fillText("Predicted", originX + gridSize / 2 - 24, 34);
  context.save();
  context.translate(30, originY + gridSize / 2 + 20);
  context.rotate(-Math.PI / 2);
  context.fillText("Actual", 0, 0);
  context.restore();

  labels.forEach((label, index) => {
    const position = originX + index * cellSize + cellSize / 2;
    context.fillText(label, position - context.measureText(label).width / 2, originY - 14);
    context.fillText(
      label,
      originX - 36 - context.measureText(label).width,
      originY + index * cellSize + cellSize / 2 + 4,
    );
  });

  for (let rowIndex = 0; rowIndex < labels.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < labels.length; columnIndex += 1) {
      const value = matrix[rowIndex][columnIndex];
      const alpha = 0.12 + (value / maxValue) * 0.78;
      context.fillStyle = `rgba(29, 78, 216, ${alpha})`;
      context.fillRect(
        originX + columnIndex * cellSize,
        originY + rowIndex * cellSize,
        cellSize - 2,
        cellSize - 2,
      );

      context.fillStyle = value > maxValue * 0.45 ? "#ffffff" : "#172033";
      const text = String(value);
      context.fillText(
        text,
        originX + columnIndex * cellSize + cellSize / 2 - context.measureText(text).width / 2,
        originY + rowIndex * cellSize + cellSize / 2 + 4,
      );
    }
  }
}

function renderComparisonChart() {
  const comparison = state.comparison;
  if (!comparison?.items?.length) {
    elements.comparisonChartEmptyState.hidden = false;
    clearCanvas(elements.comparisonCanvas, 420);
    return;
  }

  elements.comparisonChartEmptyState.hidden = true;
  if (comparison.task_type === "regression") {
    elements.comparisonChartTitle.textContent = "Actual vs Predicted by Row";
    drawComparisonRegressionChart(comparison.items);
    return;
  }

  elements.comparisonChartTitle.textContent = "Prediction Matrix";
  drawClassificationMatrix(comparison.items);
}

function drawForecastTimeline() {
  const { context, width, height } = prepareCanvas(elements.forecastCanvas, 360);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fbfaf7";
  context.fillRect(0, 0, width, height);

  const payload = state.forecast;
  if (!payload?.forecast?.length) {
    return;
  }

  const history = downsample(payload.recent_history ?? [], 36).map((item, index) => ({
    x: index,
    y: Number(item.target),
  }));
  const forecastStartX = history.length > 0 ? history[history.length - 1].x + 1 : 0;
  const future = payload.forecast.map((item, index) => ({
    x: forecastStartX + index,
    y: Number(item.prediction),
  }));
  const allPoints = history.concat(future).filter((point) => Number.isFinite(point.y));
  if (!allPoints.length) {
    return;
  }

  const bounds = { top: 48, right: width - 26, bottom: height - 40, left: 58 };
  const minValue = Math.min(...allPoints.map((point) => point.y));
  const maxValue = Math.max(...allPoints.map((point) => point.y));
  const span = maxValue - minValue || 1;
  const paddedMin = minValue - span * 0.08;
  const paddedMax = maxValue + span * 0.08;

  drawAxes(context, width, height, bounds, "Forecast steps");
  drawLegend(context, [
    { label: "Recent history", color: "#0f766e" },
    { label: "Forecast", color: "#c2410c" },
  ]);

  for (let tick = 0; tick <= 4; tick += 1) {
    const ratio = tick / 4;
    const y = bounds.bottom - (bounds.bottom - bounds.top) * ratio;
    const value = paddedMin + (paddedMax - paddedMin) * ratio;
    context.strokeStyle = "rgba(23, 32, 51, 0.08)";
    context.beginPath();
    context.moveTo(bounds.left, y);
    context.lineTo(bounds.right, y);
    context.stroke();
    context.fillStyle = "#5e6778";
    context.font = '12px "Segoe UI Variable Text", "Trebuchet MS", sans-serif';
    context.fillText(formatMetricValue(value), 8, y + 4);
  }

  function mapX(point) {
    const maxX = allPoints[allPoints.length - 1].x || 1;
    return bounds.left + (point.x / Math.max(maxX, 1)) * (bounds.right - bounds.left);
  }

  function mapY(point) {
    return (
      bounds.bottom -
      ((point.y - paddedMin) / (paddedMax - paddedMin || 1)) * (bounds.bottom - bounds.top)
    );
  }

  function drawSeries(points, color, dashed = false) {
    if (!points.length) {
      return;
    }
    context.save();
    context.strokeStyle = color;
    context.lineWidth = 2.5;
    if (dashed) {
      context.setLineDash([8, 6]);
    }
    context.beginPath();
    points.forEach((point, index) => {
      const x = mapX(point);
      const y = mapY(point);
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
    context.restore();
  }

  drawSeries(history, "#0f766e", false);

  if (history.length && future.length) {
    drawSeries([history[history.length - 1], ...future], "#c2410c", true);
  } else {
    drawSeries(future, "#c2410c", true);
  }

  for (const point of future) {
    context.fillStyle = "#c2410c";
    context.beginPath();
    context.arc(mapX(point), mapY(point), 4.5, 0, Math.PI * 2);
    context.fill();
  }
}

function renderForecastChart() {
  if (!state.forecast?.forecast?.length) {
    elements.forecastChartEmptyState.hidden = false;
    clearCanvas(elements.forecastCanvas, 360);
    elements.forecastWarning.className = "status-banner status-idle";
    elements.forecastWarning.textContent = "Forecast notes will appear here.";
    return;
  }

  elements.forecastChartEmptyState.hidden = true;
  elements.forecastChartTitle.textContent = "Recent History + Future Points";
  drawForecastTimeline();

  if (state.forecast.warning) {
    elements.forecastWarning.className = "status-banner status-busy";
    elements.forecastWarning.textContent = state.forecast.warning;
  } else {
    elements.forecastWarning.className = "status-banner status-success";
    elements.forecastWarning.textContent =
      `Forecast cadence: ${state.forecast.base_frequency_minutes} min.`;
  }
}

function renderAll() {
  buildSummaryCards();
  renderPreview();
  renderComparisonTable();
  renderForecastTable();
  renderComparisonChart();
  renderForecastChart();
}

function trainingStatusMessage(training) {
  const effectiveOptions = training.training_options?.effective ?? {};
  const presetSuffix = effectiveOptions.preset ? `/${effectiveOptions.preset}` : "";
  let message =
    `Training finished: backend=${training.backend}${presetSuffix}, ` +
    `model=${training.model_version.version_id}.`;
  if (training.warnings?.length) {
    message += ` Warning: ${training.warnings[0]}`;
  }
  return message;
}

function retrainingStatusMessage(retraining) {
  const profit = retraining.evaluation?.profit ?? null;
  let message =
    `Retraining finished: candidate=${retraining.candidate_model_version.version_id}, ` +
    `scope=${retraining.selection_summary.history_scope}.`;
  if (profit) {
    message +=
      ` ${profit.primary_metric}: ${formatMetricValue(profit.current_value)} -> ` +
      `${formatMetricValue(profit.candidate_value)} `;
    message += `(${formatMetricValue(profit.relative_gain_percent)}% gain).`;
  }
  if (retraining.activated) {
    message += " Candidate activated.";
  } else {
    message += ` ${retraining.activation_reason}`;
  }
  return message;
}

function syncTrainingControls() {
  const lightAutoMLSelected = elements.backendInput.value !== "sklearn";
  elements.presetInput.disabled = !lightAutoMLSelected;
  elements.cvFoldsInput.disabled = !lightAutoMLSelected;
  elements.timeoutSecondsInput.disabled = !lightAutoMLSelected;
  for (const input of elements.algoInputs) {
    input.disabled = !lightAutoMLSelected;
  }

  if (lightAutoMLSelected) {
    elements.trainingOptionsNote.textContent =
      "Preset, algorithms, CV and timeout apply only to the LightAutoML backend.";
  } else {
    elements.trainingOptionsNote.textContent =
      "sklearn uses the local random-forest pipeline. Preset, algorithms, CV and timeout are ignored.";
  }
}

async function runAction(actionName, action) {
  try {
    setBusy(true);
    setStatus("busy", actionName);
    await action();
  } catch (error) {
    setStatus("error", normalizeError(error));
  } finally {
    setBusy(false);
  }
}

async function handleValidate() {
  const workflow = requireFileAndTarget();
  state.validation = await postFile("/datasets/validate/file", {
    ...workflow,
    includeTarget: true,
  });
  setStatus("success", `Dataset validated: ${state.validation.rows} rows, task=${state.validation.task_type}.`);
  renderAll();
}

async function handleTrain() {
  const workflow = requireTrainingWorkflow();
  state.training = await postFile("/training/run/file", {
    ...workflow,
    includeTarget: true,
    trainingOptions: workflow.trainingOptions,
  });
  state.retraining = null;
  state.comparison = null;
  state.forecast = null;
  setStatus("success", trainingStatusMessage(state.training));
  renderAll();
}

async function handleCompare() {
  const workflow = requireFileAndTarget();
  state.comparison = await postFile("/predictions/compare/file", {
    ...workflow,
    includeTarget: true,
  });
  setStatus(
    "success",
    `Comparison ready: ${state.comparison.rows} rows, model=${state.comparison.model_version}.`,
  );
  renderAll();
}

async function handleRetrain() {
  const workflow = requireRetrainingWorkflow();
  state.retraining = await postFile("/retraining/run/file", {
    ...workflow,
    includeTarget: true,
    trainingOptions: workflow.trainingOptions,
    retrainingOptions: workflow.retrainingOptions,
  });
  state.training = null;
  state.comparison = null;
  state.forecast = null;
  setStatus("success", retrainingStatusMessage(state.retraining));
  renderAll();
}

async function handleForecast() {
  const workflow = requireForecastParams();
  state.forecast = await postJson("/forecast/run", {
    project_id: workflow.projectId,
    horizon_minutes: workflow.horizonMinutes,
    steps: workflow.forecastSteps,
  });
  setStatus(
    "success",
    `Forecast ready: ${state.forecast.steps} step(s), horizon ${state.forecast.requested_horizon_minutes} min.`,
  );
  renderAll();
}

async function handleTrainAndCompare() {
  await handleTrain();
  await handleCompare();
}

elements.validateButton.addEventListener("click", () => {
  runAction("Validating dataset...", handleValidate);
});

elements.trainButton.addEventListener("click", () => {
  runAction("Training champion model...", handleTrain);
});

elements.retrainButton.addEventListener("click", () => {
  runAction("Retraining challenger model and comparing profit...", handleRetrain);
});

elements.compareButton.addEventListener("click", () => {
  runAction("Comparing actual vs predicted...", handleCompare);
});

elements.forecastButton.addEventListener("click", () => {
  runAction("Generating short-horizon forecast...", handleForecast);
});

elements.trainCompareButton.addEventListener("click", () => {
  runAction("Training model and building comparison...", handleTrainAndCompare);
});

elements.backendInput.addEventListener("change", () => {
  syncTrainingControls();
});

window.addEventListener("resize", () => {
  renderComparisonChart();
  renderForecastChart();
});

syncTrainingControls();
renderAll();
