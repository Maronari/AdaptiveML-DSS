const EMPTY_VALUE = "—";
const DEFAULT_CHART_HEIGHT = 520;
const DEFAULT_INTERVAL_MINUTES = 120;
const MIN_INTERVAL_MINUTES = 10;
const MAX_INTERVAL_MINUTES = 1440;
const INTERVAL_STEP_MINUTES = 10;
const MIN_VISIBLE_POINTS = 8;
const DEFAULT_VISIBLE_POINTS = 96;
const POINTS_PER_MINUTE = 12;
const MAX_TOTAL_POINTS = 720;

const state = {
  model: null,
  forecast: null,
  forecastComparison: null,
  displayForecast: null,
  timelineRows: null,
  error: null,
  selectedIntervalMinutes: null,
  selectedPointCount: null,
  availablePointCounts: [],
  request: null,
  isLoadingForecast: false,
  forecastToken: 0,
};

const chartState = {
  viewStart: 0,
  viewEnd: 1,
  pointerId: null,
  dragStartX: 0,
  dragOriginStart: 0,
  dragWindow: 1,
  isDragging: false,
  hoverPointId: null,
  tooltipPointId: null,
  renderedPoints: [],
  renderFrame: 0,
  theme: null,
  canvasWidth: 0,
  canvasHeight: 0,
  canvasPixelWidth: 0,
  canvasPixelHeight: 0,
};

const elements = {
  chartCanvas: document.getElementById("prediction-chart"),
  chartEmptyState: document.getElementById("chart-empty-state"),
  chartTooltip: document.getElementById("chart-tooltip"),
  chartTitle: document.getElementById("chart-title"),
  chartRangeNote: document.getElementById("chart-range-note"),
  chartResetButton: document.getElementById("chart-reset-button"),
  selectedModelSummary: document.getElementById("selected-model-summary"),
  selectedModelVersion: document.getElementById("selected-model-version"),
  selectedModelStatus: document.getElementById("selected-model-status"),
  selectedModelTarget: document.getElementById("selected-model-target"),
  selectedModelMetric: document.getElementById("selected-model-metric"),
  graphMetricsSummary: document.getElementById("graph-metrics-summary"),
  graphModelMetricsGrid: document.getElementById("graph-model-metrics-grid"),
  graphForecastMetricsGrid: document.getElementById("graph-forecast-metrics-grid"),
  forecastPanelTitle: document.getElementById("forecast-panel-title"),
  forecastPanelNote: document.getElementById("forecast-panel-note"),
  predictionTableBody: document.getElementById("prediction-table-body"),
  predictionTableEmpty: document.getElementById("prediction-table-empty"),
  predictionTableAbsErrorHeader: document.getElementById("prediction-table-abs-error-header"),
  intervalSlider: document.getElementById("interval-slider"),
  intervalValue: document.getElementById("interval-value"),
  intervalMin: document.getElementById("interval-min"),
  intervalMax: document.getElementById("interval-max"),
  pointCountSlider: document.getElementById("point-count-slider"),
  pointCountValue: document.getElementById("point-count-value"),
  pointCountMin: document.getElementById("point-count-min"),
  pointCountMax: document.getElementById("point-count-max"),
  uploadLinks: Array.from(document.querySelectorAll("[data-nav-upload]")),
  trainingLinks: Array.from(document.querySelectorAll("[data-nav-training]")),
  retrainingLinks: Array.from(document.querySelectorAll("[data-nav-retraining]")),
  modelsLinks: Array.from(document.querySelectorAll("[data-nav-models]")),
  reportLinks: Array.from(document.querySelectorAll("[data-nav-report]")),
  graphLinks: Array.from(document.querySelectorAll("[data-nav-graph]")),
  dssLinks: Array.from(document.querySelectorAll("[data-nav-dss]")),
};

const QUALITY_METRICS = [
  { key: "r", label: "R" },
  { key: "r2", label: "R²" },
  { key: "mse", label: "MSE" },
  { key: "rmse", label: "RMSE" },
  { key: "aic", label: "AIC" },
  { key: "aicc", label: "AICc" },
  { key: "bic", label: "BIC" },
];

function getRequestedContext() {
  const searchParams = new URLSearchParams(window.location.search);
  const requestedHorizon = Number.parseFloat(searchParams.get("horizon_minutes") || "");
  const requestedInterval = Number.parseInt(searchParams.get("interval_minutes") || "", 10);
  const requestedPointCount = Number.parseInt(searchParams.get("point_count") || searchParams.get("steps") || "", 10);

  let intervalMinutes = null;
  if (Number.isFinite(requestedInterval) && requestedInterval > 0) {
    intervalMinutes = requestedInterval;
  } else if (Number.isFinite(requestedHorizon) && requestedHorizon > 0) {
    intervalMinutes = requestedHorizon;
  }

  return {
    projectId: searchParams.get("project_id")?.trim() || "",
    versionId: searchParams.get("model_version")?.trim() || "",
    intervalMinutes,
    pointCount: Number.isFinite(requestedPointCount) && requestedPointCount > 0 ? requestedPointCount : null,
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function getDefaultViewportRange(totalPoints) {
  if (!Number.isFinite(totalPoints) || totalPoints <= DEFAULT_VISIBLE_POINTS) {
    return { start: 0, end: 1 };
  }

  const windowSize = clamp(DEFAULT_VISIBLE_POINTS / totalPoints, MIN_VISIBLE_POINTS / totalPoints, 1);
  return {
    start: 1 - windowSize,
    end: 1,
  };
}

function getThemeValue(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function getChartTheme() {
  if (chartState.theme) {
    return chartState.theme;
  }

  chartState.theme = {
    background: getThemeValue("--chart-bg", "#ffffff"),
    axis: getThemeValue("--chart-axis", "rgba(50, 70, 97, 0.24)"),
    grid: getThemeValue("--chart-grid", "rgba(50, 70, 97, 0.1)"),
    label: getThemeValue("--chart-label", "#7d8ca1"),
    text: getThemeValue("--chart-text", "#243754"),
    actual: getThemeValue("--chart-actual", "#324661"),
    predicted: getThemeValue("--chart-predicted", "#3f7cff"),
    model: getThemeValue("--chart-model", "#da8b2b"),
    postfactum: getThemeValue("--chart-postfactum", "#c0392b"),
    divider: getThemeValue("--line-strong", "rgba(50, 70, 97, 0.28)"),
  };
  return chartState.theme;
}

function setChartFont(context, size = 12, weight = 400) {
  context.font = `${weight} ${size}px "Aptos", "Segoe UI Variable Text", "Trebuchet MS", sans-serif`;
}

function formatNumber(value, maximumFractionDigits = 3) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return "—";
  }

  return numericValue.toLocaleString("ru-RU", {
    minimumFractionDigits: 0,
    maximumFractionDigits,
  });
}

function formatPrimaryMetric(model) {
  const metricKey = String(model?.primary_metric || "").trim();
  const metricValue = model?.metrics?.[metricKey];
  if (!metricKey) {
    return "—";
  }

  return `${metricKey.toUpperCase()}: ${formatNumber(metricValue, 4)}`;
}

function formatTimestamp(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  const includeSeconds = date.getSeconds() !== 0 || date.getMilliseconds() !== 0;

  return date.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: includeSeconds ? "2-digit" : undefined,
  });
}

function formatAxisTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  return date.toLocaleTimeString("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatAxisDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }

  return date.toLocaleDateString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
  });
}

function formatDuration(totalMinutes) {
  const totalSeconds = Math.max(1, Math.round((Number(totalMinutes) || 0) * 60));
  const days = Math.floor(totalSeconds / 86400);
  const hours = Math.floor((totalSeconds % 86400) / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (days > 0 && hours === 0 && minutes === 0) {
    return `${days} дн`;
  }
  if (days > 0 && minutes === 0 && seconds === 0) {
    return `${days} дн ${hours} ч`;
  }
  if (hours > 0 && minutes === 0 && seconds === 0) {
    return `${hours} ч`;
  }
  if (hours > 0 && seconds === 0) {
    return `${hours} ч ${minutes} мин`;
  }
  if (minutes > 0 && seconds === 0) {
    return `${minutes} мин`;
  }
  if (minutes > 0) {
    return `${minutes} мин ${seconds} сек`;
  }
  return `${seconds} сек`;
}

function updateSelectedModelSummary() {
  if (!state.model) {
    elements.selectedModelSummary.hidden = true;
    elements.selectedModelVersion.textContent = "—";
    elements.selectedModelStatus.textContent = "—";
    elements.selectedModelTarget.textContent = "—";
    elements.selectedModelMetric.textContent = "—";
    return;
  }

  elements.selectedModelVersion.textContent = state.model.name
    ? `${state.model.name} (${state.model.version_id})`
    : state.model.version_id || "—";
  elements.selectedModelStatus.textContent = state.model.status || "—";
  elements.selectedModelTarget.textContent = state.model.target || "—";
  elements.selectedModelMetric.textContent = formatPrimaryMetric(state.model);
  elements.selectedModelSummary.hidden = false;
}

function renderMetricGrid(container, metrics) {
  container.replaceChildren();

  const safeMetrics = metrics && typeof metrics === "object" ? metrics : {};
  const fragment = document.createDocumentFragment();
  QUALITY_METRICS.forEach(({ key, label }) => {
    const card = document.createElement("div");
    card.className = "graph-metric-chip";

    const metricLabel = document.createElement("span");
    metricLabel.className = "graph-metric-chip-label";
    metricLabel.textContent = label;
    card.append(metricLabel);

    const metricValue = document.createElement("strong");
    metricValue.textContent = formatNumber(safeMetrics[key], 4);
    card.append(metricValue);

    fragment.append(card);
  });

  container.append(fragment);
}

function updateMetricsSummary() {
  if (!state.model) {
    elements.graphMetricsSummary.hidden = true;
    elements.graphModelMetricsGrid.replaceChildren();
    elements.graphForecastMetricsGrid.replaceChildren();
    return;
  }

  const forecastMetrics = state.model.forecasting?.metrics || state.model.training_artifacts?.forecasting?.metrics || {};
  renderMetricGrid(elements.graphModelMetricsGrid, state.model.metrics || {});
  renderMetricGrid(elements.graphForecastMetricsGrid, forecastMetrics);
  elements.graphMetricsSummary.hidden = false;
}

function getInitialIntervalMinutes(context, model) {
  if (Number.isFinite(context.intervalMinutes) && context.intervalMinutes > 0) {
    return normalizeIntervalMinutes(context.intervalMinutes);
  }

  const defaultHorizon = Number(model?.forecasting?.default_horizon_minutes) || 30;
  return normalizeIntervalMinutes(defaultHorizon);
}

function normalizeIntervalMinutes(minutes) {
  const numericMinutes = Number(minutes);
  if (!Number.isFinite(numericMinutes) || numericMinutes <= 0) {
    return DEFAULT_INTERVAL_MINUTES;
  }

  const rounded = Math.round(numericMinutes / INTERVAL_STEP_MINUTES) * INTERVAL_STEP_MINUTES;
  return clamp(rounded, MIN_INTERVAL_MINUTES, MAX_INTERVAL_MINUTES);
}

function getAvailablePointCounts(totalMinutes, baseFrequencyMinutes = DEFAULT_INTERVAL_MINUTES) {
  const normalizedInterval = Math.max(1, Math.round(Number(totalMinutes) || DEFAULT_INTERVAL_MINUTES));
  const normalizedBaseFrequency = Math.max(1, Math.round(Number(baseFrequencyMinutes) || DEFAULT_INTERVAL_MINUTES));
  const maxPointCount = Math.max(1, Math.min(MAX_TOTAL_POINTS, Math.ceil(normalizedInterval / normalizedBaseFrequency)));
  const counts = [];

  for (let pointCount = 1; pointCount <= maxPointCount; pointCount += 1) {
    counts.push(pointCount);
  }

  return counts.length ? counts : [1];
}

function getNearestPointCount(availableCounts, preferredCount) {
  if (!availableCounts.length) {
    return 1;
  }
  if (!Number.isFinite(preferredCount) || preferredCount <= 0) {
    return availableCounts[availableCounts.length - 1];
  }

  return availableCounts.reduce((closest, pointCount) => {
    const closestDelta = Math.abs(closest - preferredCount);
    const currentDelta = Math.abs(pointCount - preferredCount);
    return currentDelta < closestDelta ? pointCount : closest;
  }, availableCounts[0]);
}

function updatePointCountControls() {
  const availableCounts = state.availablePointCounts;
  const hasOptions = availableCounts.length > 0;
  const selectedCount = hasOptions
    ? getNearestPointCount(availableCounts, state.selectedPointCount)
    : null;

  elements.pointCountSlider.min = "0";
  elements.pointCountSlider.max = String(Math.max(availableCounts.length - 1, 0));
  elements.pointCountSlider.value = hasOptions ? String(availableCounts.indexOf(selectedCount)) : "0";
  elements.pointCountValue.textContent = hasOptions ? String(selectedCount) : "—";
  elements.pointCountMin.textContent = hasOptions ? String(availableCounts[0]) : "—";
  elements.pointCountMax.textContent = hasOptions ? String(availableCounts[availableCounts.length - 1]) : "—";
}

function updateIntervalControls() {
  const selectedInterval = normalizeIntervalMinutes(state.selectedIntervalMinutes);
  elements.intervalSlider.min = String(MIN_INTERVAL_MINUTES);
  elements.intervalSlider.max = String(MAX_INTERVAL_MINUTES);
  elements.intervalSlider.step = String(INTERVAL_STEP_MINUTES);
  elements.intervalSlider.value = String(selectedInterval);
  elements.intervalValue.textContent = formatDuration(selectedInterval);
  elements.intervalMin.textContent = formatDuration(MIN_INTERVAL_MINUTES);
  elements.intervalMax.textContent = formatDuration(MAX_INTERVAL_MINUTES);
}

function getSliderIntervalMinutes() {
  return normalizeIntervalMinutes(Number.parseInt(elements.intervalSlider.value || String(DEFAULT_INTERVAL_MINUTES), 10));
}

function syncPointCountControls(intervalMinutes, preferredCount = null) {
  if (!Number.isFinite(intervalMinutes) || intervalMinutes <= 0) {
    state.availablePointCounts = [];
    state.selectedPointCount = null;
    updatePointCountControls();
    return;
  }

  const baseFrequencyMinutes = Math.max(
    1,
    Number(state.model?.forecasting?.base_frequency_minutes) || Number(intervalMinutes) || DEFAULT_INTERVAL_MINUTES,
  );
  const totalMinutes = Math.max(intervalMinutes, Math.ceil(intervalMinutes / baseFrequencyMinutes) * baseFrequencyMinutes);
  state.availablePointCounts = getAvailablePointCounts(totalMinutes, baseFrequencyMinutes);
  state.selectedPointCount = getNearestPointCount(state.availablePointCounts, preferredCount);
  updatePointCountControls();
}

function getSliderPointCount() {
  const availableCounts = state.availablePointCounts;
  if (!availableCounts.length) {
    return null;
  }

  const sliderIndex = clamp(
    Number.parseInt(elements.pointCountSlider.value || "0", 10),
    0,
    availableCounts.length - 1,
  );
  return availableCounts[sliderIndex];
}

function buildForecastRequest(model, intervalMinutes, pointCount) {
  const normalizedInterval = normalizeIntervalMinutes(intervalMinutes);
  const baseFrequency = Math.max(
    1,
    Number(model?.forecasting?.base_frequency_minutes) || Number(model?.forecasting?.default_horizon_minutes) || 30,
  );
  const backendSteps = Math.max(1, Math.ceil(normalizedInterval / baseFrequency));
  const totalMinutes = backendSteps * baseFrequency;
  const availablePointCounts = getAvailablePointCounts(totalMinutes, baseFrequency);
  const resolvedPointCount = getNearestPointCount(availablePointCounts, pointCount);
  const displayStepMinutes = Number((totalMinutes / resolvedPointCount).toFixed(6));

  return {
    intervalMinutes: normalizedInterval,
    baseFrequencyMinutes: baseFrequency,
    displayStepMinutes,
    backendHorizonMinutes: baseFrequency,
    backendSteps,
    totalMinutes,
    pointCount: resolvedPointCount,
  };
}

function isSameForecastRequest(left, right) {
  if (!left || !right) {
    return false;
  }

  return (
    left.intervalMinutes === right.intervalMinutes &&
    left.displayStepMinutes === right.displayStepMinutes &&
    left.backendHorizonMinutes === right.backendHorizonMinutes &&
    left.backendSteps === right.backendSteps &&
    left.pointCount === right.pointCount &&
    left.baseFrequencyMinutes === right.baseFrequencyMinutes
  );
}

function updateLocation() {
  if (!state.model) {
    return;
  }

  const currentUrl = new URL(window.location.href);
  currentUrl.searchParams.set("project_id", state.model.project_id);
  currentUrl.searchParams.set("model_version", state.model.version_id);

  if (state.request) {
    currentUrl.searchParams.set("interval_minutes", String(state.request.intervalMinutes));
    currentUrl.searchParams.set("point_count", String(state.request.pointCount));
    currentUrl.searchParams.delete("horizon_minutes");
    currentUrl.searchParams.delete("steps");
  }

  window.history.replaceState({}, "", currentUrl);
}

function syncNavigation(projectId) {
  for (const link of elements.uploadLinks) {
    const targetUrl = new URL("./upload.html", window.location.href);
    if (projectId) {
      targetUrl.searchParams.set("project_id", projectId);
    }
    link.href = targetUrl.toString();
  }

  for (const link of elements.trainingLinks) {
    const targetUrl = new URL("./training.html", window.location.href);
    if (projectId) {
      targetUrl.searchParams.set("project_id", projectId);
    }
    link.href = targetUrl.toString();
  }

  for (const link of elements.retrainingLinks) {
    const targetUrl = new URL("./retraining.html", window.location.href);
    if (projectId) {
      targetUrl.searchParams.set("project_id", projectId);
    }
    link.href = targetUrl.toString();
  }

  for (const link of elements.modelsLinks) {
    const targetUrl = new URL("./models.html", window.location.href);
    if (projectId) {
      targetUrl.searchParams.set("project_id", projectId);
    }
    link.href = targetUrl.toString();
  }

  for (const link of elements.reportLinks) {
    const targetUrl = new URL("./report.html", window.location.href);
    if (projectId) {
      targetUrl.searchParams.set("project_id", projectId);
    }
    if (state.model?.version_id) {
      targetUrl.searchParams.set("model_version", state.model.version_id);
    }
    link.href = targetUrl.toString();
  }

  for (const link of elements.graphLinks) {
    const targetUrl = new URL("./graph.html", window.location.href);
    if (projectId) {
      targetUrl.searchParams.set("project_id", projectId);
    }
    if (state.model?.version_id) {
      targetUrl.searchParams.set("model_version", state.model.version_id);
    }
    link.href = targetUrl.toString();
  }

  for (const link of elements.dssLinks) {
    const targetUrl = new URL("./dss.html", window.location.href);
    if (projectId) {
      targetUrl.searchParams.set("project_id", projectId);
    }
    if (state.model?.version_id) {
      targetUrl.searchParams.set("model_version", state.model.version_id);
    }
    if (state.request) {
      targetUrl.searchParams.set("interval_minutes", String(state.request.intervalMinutes));
      targetUrl.searchParams.set("point_count", String(state.request.pointCount));
    }
    link.href = targetUrl.toString();
  }
}

function resetViewport(totalPoints = getTimelineRows().length) {
  const viewport = getDefaultViewportRange(totalPoints);
  chartState.viewStart = viewport.start;
  chartState.viewEnd = viewport.end;
  chartState.pointerId = null;
  chartState.dragStartX = 0;
  chartState.dragOriginStart = 0;
  chartState.dragWindow = viewport.end - viewport.start;
  chartState.isDragging = false;
  chartState.hoverPointId = null;
}

function isViewportReset() {
  const viewport = getDefaultViewportRange(getTimelineRows().length);
  return (
    Math.abs(chartState.viewStart - viewport.start) <= 0.0001 &&
    Math.abs(chartState.viewEnd - viewport.end) <= 0.0001
  );
}

function getPlotBounds(width, height) {
  return { top: 52, right: width - 26, bottom: height - 68, left: 72 };
}

function getPointId(point) {
  return `${point.series}:${point.timestamp}`;
}

function hideTooltip() {
  elements.chartTooltip.hidden = true;
  chartState.tooltipPointId = null;
}

function showTooltip(pointLayout) {
  if (!pointLayout) {
    hideTooltip();
    return;
  }

  const seriesLabels = {
    history: "История",
    model: "Модель",
    forecast: "Прогноз",
    postfactum_actual: "Факт (после прогноза)",
  };
  const tooltip = elements.chartTooltip;
  tooltip.innerHTML = `
    <strong>${formatNumber(pointLayout.value)}</strong>
    <span>${seriesLabels[pointLayout.series] || "Точка"} • ${formatTimestamp(pointLayout.timestamp)}</span>
  `;
  tooltip.hidden = false;

  const frame = elements.chartCanvas.parentElement;
  const frameRect = frame.getBoundingClientRect();
  const tooltipRect = tooltip.getBoundingClientRect();
  const left = clamp(pointLayout.x + 16, 12, frameRect.width - tooltipRect.width - 12);
  const top = clamp(pointLayout.y - tooltipRect.height - 14, 12, frameRect.height - tooltipRect.height - 12);

  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

function clearHoverPoint() {
  if (chartState.hoverPointId === null && elements.chartTooltip.hidden) {
    hideTooltip();
    return;
  }

  chartState.hoverPointId = null;
  hideTooltip();
}

function invalidateTimelineRows() {
  state.timelineRows = null;
}

function scheduleChartRender() {
  if (chartState.renderFrame) {
    return;
  }

  chartState.renderFrame = window.requestAnimationFrame(() => {
    chartState.renderFrame = 0;
    renderChart();
  });
}

function toTimestampMs(value) {
  const timestamp = new Date(value).getTime();
  return Number.isFinite(timestamp) ? timestamp : null;
}

function normalizeTimestamp(value) {
  const timestamp = toTimestampMs(value);
  return timestamp === null ? null : new Date(timestamp).toISOString();
}

function extractRecordTimestamp(record) {
  if (!record || typeof record !== "object") {
    return null;
  }

  const directKeys = ["timestamp", "datetime", "date", "Дата", "Дата и время"];
  for (const key of directKeys) {
    if (key in record) {
      const timestamp = normalizeTimestamp(record[key]);
      if (timestamp) {
        return timestamp;
      }
    }
  }

  for (const [key, value] of Object.entries(record)) {
    if (key.endsWith("__ts")) {
      const numericValue = Number(value);
      if (Number.isFinite(numericValue)) {
        return normalizeTimestamp(numericValue * 1000);
      }
    }
  }

  return null;
}

function buildDisplayForecast(rawForecast, request) {
  const rawForecastRows = Array.isArray(rawForecast?.forecast) ? rawForecast.forecast : [];
  if (!request || !rawForecastRows.length) {
    return [];
  }

  const sanitizedRows = rawForecastRows
    .map((item, index) => ({
      step: Number.isFinite(Number(item.step)) ? Number(item.step) : index + 1,
      timestamp: item.timestamp,
      prediction: Number(item.prediction),
    }))
    .filter((item) => item.timestamp && Number.isFinite(item.prediction));

  if (request.pointCount >= sanitizedRows.length) {
    return sanitizedRows;
  }

  const lastIndex = sanitizedRows.length - 1;
  const selectedIndexes = new Set();
  for (let position = 0; position < request.pointCount; position += 1) {
    const ratio = request.pointCount === 1 ? 1 : position / (request.pointCount - 1);
    selectedIndexes.add(Math.round(ratio * lastIndex));
  }

  return Array.from(selectedIndexes)
    .sort((left, right) => left - right)
    .map((index) => sanitizedRows[index]);
}

function getDisplayForecastRows() {
  if (Array.isArray(state.displayForecast) && state.displayForecast.length) {
    return state.displayForecast;
  }
  return Array.isArray(state.forecast?.forecast) ? state.forecast.forecast : [];
}

function getModelHistoryRows() {
  const historicalFitRows = Array.isArray(state.forecast?.historical_fit) ? state.forecast.historical_fit : [];
  if (historicalFitRows.length) {
    return historicalFitRows
      .map((item) => ({
        timestamp: item.timestamp,
        value: Number(item.prediction),
      }))
      .filter((item) => item.timestamp && Number.isFinite(item.value))
      .sort((left, right) => toTimestampMs(left.timestamp) - toTimestampMs(right.timestamp));
  }

  const rows = Array.isArray(state.model?.holdout_predictions) ? state.model.holdout_predictions : [];
  return rows
    .map((item) => ({
      timestamp: extractRecordTimestamp(item.record),
      value: Number(item.prediction),
    }))
    .filter((item) => item.timestamp && Number.isFinite(item.value))
    .sort((left, right) => toTimestampMs(left.timestamp) - toTimestampMs(right.timestamp));
}

function getTimelineRows() {
  if (Array.isArray(state.timelineRows)) {
    return state.timelineRows;
  }

  const rowsByTimestamp = new Map();

  const ensureRow = (timestamp) => {
    if (!timestamp) {
      return null;
    }

    const normalizedTimestamp = normalizeTimestamp(timestamp);
    if (!normalizedTimestamp) {
      return null;
    }

    let row = rowsByTimestamp.get(normalizedTimestamp);
    if (!row) {
      row = {
        timestamp: normalizedTimestamp,
        actual: null,
        model: null,
        forecast: null,
        postfactum_actual: null,
      };
      rowsByTimestamp.set(normalizedTimestamp, row);
    }
    return row;
  };

  const historyRows = Array.isArray(state.forecast?.recent_history) ? state.forecast.recent_history : [];
  const historicalFitRows = Array.isArray(state.forecast?.historical_fit) ? state.forecast.historical_fit : [];
  const sourceHistoryRows = historicalFitRows.length ? historicalFitRows : historyRows;
  sourceHistoryRows.forEach((item) => {
    const row = ensureRow(item.timestamp);
    const value = Number("target" in item ? item.target : item.actual);
    if (row && Number.isFinite(value)) {
      row.actual = value;
    }
  });

  getModelHistoryRows().forEach((item) => {
    const row = ensureRow(item.timestamp);
    if (row) {
      row.model = item.value;
    }
  });

  getDisplayForecastRows().forEach((item) => {
    const row = ensureRow(item.timestamp);
    const value = Number(item.prediction);
    if (row && Number.isFinite(value)) {
      row.forecast = value;
    }
  });

  const comparisonPoints = Array.isArray(state.forecastComparison?.points) ? state.forecastComparison.points : [];
  comparisonPoints.forEach((item) => {
    if (item.actual === null || item.actual === undefined) {
      return;
    }
    const row = ensureRow(item.timestamp);
    const value = Number(item.actual);
    if (row && Number.isFinite(value)) {
      row.postfactum_actual = value;
    }
  });

  state.timelineRows = Array.from(rowsByTimestamp.values()).sort(
    (left, right) => toTimestampMs(left.timestamp) - toTimestampMs(right.timestamp),
  );
  return state.timelineRows;
}

function applyDisplayPointCount(pointCount, { resetZoom = false } = {}) {
  if (!state.request) {
    return;
  }

  const resolvedPointCount = getNearestPointCount(state.availablePointCounts, pointCount);
  state.selectedPointCount = resolvedPointCount;
  state.request = {
    ...state.request,
    pointCount: resolvedPointCount,
    displayStepMinutes: Number((state.request.totalMinutes / resolvedPointCount).toFixed(6)),
  };
  state.displayForecast = buildDisplayForecast(state.forecast, state.request);
  invalidateTimelineRows();

  if (resetZoom) {
    resetViewport();
  }

  updatePointCountControls();
  updateLocation();
  renderAll();
}

async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: {
      Accept: "application/json",
    },
  });

  const payload = await readJsonResponse(response);
  if (!response.ok) {
    const message =
      payload && typeof payload === "object" && "detail" in payload ? payload.detail : response.statusText;
    throw new Error(message || "Ошибка запроса.");
  }

  return payload;
}

async function resolveDefaultProjectId() {
  try {
    const payload = await fetchJson("/projects");
    const projects = Array.isArray(payload.items) ? payload.items : [];
    return (
      projects.find((project) => project.has_champion_model)?.project_id ||
      projects.find((project) => project.has_models)?.project_id ||
      projects[0]?.project_id ||
      ""
    );
  } catch {
    return "";
  }
}

function setInteractiveState() {
  const disabled = state.isLoadingForecast || !state.model?.forecasting?.available;
  elements.intervalSlider.disabled = disabled;
  elements.pointCountSlider.disabled = disabled || !state.availablePointCounts.length;
  updateIntervalControls();
  updatePointCountControls();
}

function clearPredictionTable(message = "После загрузки прогноза здесь появятся точки предсказания.") {
  elements.predictionTableBody.innerHTML = "";
  elements.predictionTableEmpty.hidden = false;
  elements.predictionTableEmpty.textContent = message;
}

function getComparisonPointsByTimestamp() {
  const points = Array.isArray(state.forecastComparison?.points) ? state.forecastComparison.points : [];
  const map = new Map();
  points.forEach((point) => {
    const key = normalizeTimestamp(point.timestamp);
    if (key) {
      map.set(key, point);
    }
  });
  return map;
}

function renderPredictionTable() {
  const rows = Array.isArray(state.displayForecast) ? state.displayForecast : [];
  elements.predictionTableBody.innerHTML = "";

  const unit = state.forecast?.unit;
  elements.predictionTableAbsErrorHeader.textContent = unit ? `Абс. ошибка, ${unit}` : "Абс. ошибка";

  if (!rows.length) {
    clearPredictionTable(
      state.error || "Для выбранной модели нет точек прогноза. Выберите другой интервал или обучите модель заново.",
    );
    return;
  }

  const comparisonByTimestamp = getComparisonPointsByTimestamp();

  const bodyRows = rows.map((item) => {
    const comparisonPoint = comparisonByTimestamp.get(normalizeTimestamp(item.timestamp));
    const hasActual = comparisonPoint && comparisonPoint.actual !== null && comparisonPoint.actual !== undefined;
    const actualCell = hasActual ? formatNumber(comparisonPoint.actual) : EMPTY_VALUE;
    const absErrorCell =
      comparisonPoint && comparisonPoint.abs_error !== null && comparisonPoint.abs_error !== undefined
        ? formatNumber(comparisonPoint.abs_error)
        : EMPTY_VALUE;
    const mapeCell =
      comparisonPoint && comparisonPoint.mape_percent !== null && comparisonPoint.mape_percent !== undefined
        ? formatNumber(comparisonPoint.mape_percent, 2)
        : EMPTY_VALUE;
    return (
      `<tr><td>${item.step}</td><td>${formatTimestamp(item.timestamp)}</td><td>${formatNumber(item.prediction)}</td>` +
      `<td>${actualCell}</td><td>${absErrorCell}</td><td>${mapeCell}</td></tr>`
    );
  });

  const aggregate = state.forecastComparison?.aggregate;
  if (aggregate && aggregate.matched_points > 0) {
    const meanAbsErrorCell =
      aggregate.mean_abs_error !== null && aggregate.mean_abs_error !== undefined
        ? formatNumber(aggregate.mean_abs_error)
        : EMPTY_VALUE;
    const meanMapeCell =
      aggregate.mean_mape_percent !== null && aggregate.mean_mape_percent !== undefined
        ? formatNumber(aggregate.mean_mape_percent, 2)
        : EMPTY_VALUE;
    bodyRows.push(
      `<tr class="is-aggregate-row"><td>Итого / среднее</td><td>${EMPTY_VALUE}</td><td>${EMPTY_VALUE}</td>` +
        `<td>${EMPTY_VALUE}</td><td>${meanAbsErrorCell}</td><td>${meanMapeCell}</td></tr>`,
    );
  }

  elements.predictionTableBody.innerHTML = bodyRows.join("");
  elements.predictionTableEmpty.hidden = true;
}

function updateSidebarInfo() {
  if (state.error) {
    elements.forecastPanelTitle.textContent = "Точки предсказания";
    elements.forecastPanelNote.textContent = state.error;
    return;
  }

  if (!state.model) {
    elements.forecastPanelTitle.textContent = "Точки предсказания";
    elements.forecastPanelNote.textContent = "Сначала обучите модель, чтобы открыть прогноз и таблицу точек.";
    return;
  }

  if (!state.model.forecasting?.available || !state.request || !state.forecast) {
    elements.forecastPanelTitle.textContent = "Прогноз недоступен";
    elements.forecastPanelNote.textContent =
      "Для этой версии нет временного прогноза. Нужна регрессионная модель с временным рядом.";
    return;
  }

  const recentHistory = Array.isArray(state.forecast?.recent_history) ? state.forecast.recent_history : [];
  const anchorTimestamp = recentHistory.length ? recentHistory[recentHistory.length - 1]?.timestamp : null;
  const anchorNote = anchorTimestamp
    ? ` Прогноз считается от последней точки обучающих данных модели — ${formatTimestamp(anchorTimestamp)} — а не от сегодняшней даты.`
    : "";

  const autoNote =
    `Интервал вперёд ${formatDuration(state.request.totalMinutes)}. ` +
    `Точек ${state.request.pointCount}, шаг отображения ${formatDuration(state.request.displayStepMinutes)}, ` +
    `нативный шаг модели ${formatDuration(state.request.baseFrequencyMinutes)}.` +
    anchorNote;
  elements.forecastPanelTitle.textContent = `Точки прогноза ${state.model.version_id}`;
  elements.forecastPanelNote.textContent = autoNote;
}

function getCanvasHeight(canvas, fallbackHeight = DEFAULT_CHART_HEIGHT) {
  const actualHeight = Math.round(canvas.clientHeight || 0);
  return actualHeight > 0 ? actualHeight : fallbackHeight;
}

function prepareCanvas(canvas, fallbackHeight = DEFAULT_CHART_HEIGHT) {
  const parentWidth = canvas.parentElement.clientWidth;
  const height = getCanvasHeight(canvas, fallbackHeight);
  const ratio = window.devicePixelRatio || 1;
  const pixelWidth = Math.floor(parentWidth * ratio);
  const pixelHeight = Math.floor(height * ratio);

  if (
    chartState.canvasWidth !== parentWidth ||
    chartState.canvasHeight !== height ||
    chartState.canvasPixelWidth !== pixelWidth ||
    chartState.canvasPixelHeight !== pixelHeight
  ) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
    chartState.canvasWidth = parentWidth;
    chartState.canvasHeight = height;
    chartState.canvasPixelWidth = pixelWidth;
    chartState.canvasPixelHeight = pixelHeight;
  }

  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {
    context,
    width: parentWidth,
    height,
  };
}

function clearCanvas() {
  const { context, width, height } = prepareCanvas(elements.chartCanvas, DEFAULT_CHART_HEIGHT);
  context.clearRect(0, 0, width, height);
}

function drawAxes(context, bounds, theme) {
  context.strokeStyle = theme.axis;
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(bounds.left, bounds.top);
  context.lineTo(bounds.left, bounds.bottom);
  context.lineTo(bounds.right, bounds.bottom);
  context.stroke();
}

function drawLegend(context, entries, theme) {
  let x = 28;
  const y = 24;
  setChartFont(context, 12, 500);

  for (const entry of entries) {
    context.fillStyle = entry.color;
    context.fillRect(x, y - 10, 16, 8);
    x += 22;
    context.fillStyle = theme.text;
    context.fillText(entry.label, x, y);
    x += context.measureText(entry.label).width + 24;
  }
}

function drawPointMarkers(context, points, color, radius) {
  if (!points.length) {
    return;
  }

  context.beginPath();
  points.forEach((point) => {
    context.moveTo(point.x + radius, point.y);
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  });
  context.fillStyle = color;
  context.fill();

  context.beginPath();
  points.forEach((point) => {
    context.moveTo(point.x + radius, point.y);
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  });
  context.strokeStyle = "#ffffff";
  context.lineWidth = 1.5;
  context.stroke();
}

function buildXAxisTicks(items, maxTickCount = 6) {
  if (!items.length) {
    return [];
  }

  const tickCount = Math.max(2, Math.min(maxTickCount, items.length));
  const lastIndex = items.length - 1;
  const indexes = new Set();

  for (let position = 0; position < tickCount; position += 1) {
    const ratio = tickCount === 1 ? 0 : position / (tickCount - 1);
    indexes.add(Math.round(ratio * lastIndex));
  }

  return Array.from(indexes)
    .sort((left, right) => left - right)
    .map((index) => ({
      index,
      item: items[index],
    }));
}

function getVisibleSlice(rows) {
  const total = rows.length;
  if (!total) {
    return {
      total: 0,
      startIndex: 0,
      endIndex: 0,
      zoomLevel: 1,
      rows: [],
    };
  }

  const minWindow = Math.min(1, MIN_VISIBLE_POINTS / total);
  const currentWindow = clamp(chartState.viewEnd - chartState.viewStart || 1, minWindow, 1);
  const viewStart = clamp(chartState.viewStart, 0, 1 - currentWindow);
  const viewEnd = viewStart + currentWindow;
  const startIndex = Math.floor(viewStart * Math.max(total - 1, 0));
  const endIndex = Math.min(total - 1, Math.ceil(viewEnd * Math.max(total - 1, 0)));

  chartState.viewStart = viewStart;
  chartState.viewEnd = viewEnd;

  return {
    total,
    startIndex,
    endIndex,
    zoomLevel: Number((1 / currentWindow).toFixed(2)),
    rows: rows.slice(startIndex, endIndex + 1),
  };
}

function updateChartControls(message, canReset = false, draggable = false) {
  elements.chartRangeNote.textContent = message;
  elements.chartResetButton.disabled = !canReset;
  elements.chartCanvas.classList.toggle("is-draggable", draggable);
  elements.chartCanvas.classList.toggle("is-dragging", draggable && chartState.isDragging);
}

function renderEmptyState(message, title = "Прогноз модели") {
  chartState.renderedPoints = [];
  chartState.hoverPointId = null;
  hideTooltip();
  elements.chartEmptyState.hidden = false;
  elements.chartEmptyState.textContent = message;
  elements.chartTitle.textContent = title;
  updateChartControls(message, false, false);
  clearCanvas();
}

function drawForecastChart() {
  const timelineRows = getTimelineRows();
  if (!timelineRows.length) {
    renderEmptyState("Нет данных для построения графика.", "Прогноз модели");
    return;
  }

  const { context, width, height } = prepareCanvas(elements.chartCanvas, DEFAULT_CHART_HEIGHT);
  const theme = getChartTheme();
  const bounds = getPlotBounds(width, height);
  const visibleSlice = getVisibleSlice(timelineRows);
  const values = [];
  visibleSlice.rows.forEach((item) => {
    if (Number.isFinite(item.actual)) {
      values.push(item.actual);
    }
    if (Number.isFinite(item.model)) {
      values.push(item.model);
    }
    if (Number.isFinite(item.forecast)) {
      values.push(item.forecast);
    }
    if (Number.isFinite(item.postfactum_actual)) {
      values.push(item.postfactum_actual);
    }
  });

  context.clearRect(0, 0, width, height);
  context.fillStyle = theme.background;
  context.fillRect(0, 0, width, height);

  if (!values.length) {
    return;
  }

  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const span = maxValue - minValue || 1;
  const paddedMin = minValue - span * 0.08;
  const paddedMax = maxValue + span * 0.08;
  const plotHeight = bounds.bottom - bounds.top;
  const plotWidth = bounds.right - bounds.left;

  const hasPostfactumSeries = visibleSlice.rows.some((item) => Number.isFinite(item.postfactum_actual));
  const unit = state.forecast?.unit;

  drawAxes(context, bounds, theme);
  drawLegend(
    context,
    [
      { label: "История", color: theme.actual },
      { label: "Модель", color: theme.model },
      { label: "Прогноз", color: theme.predicted },
      ...(hasPostfactumSeries ? [{ label: "Факт (после прогноза)", color: theme.model }] : []),
    ],
    theme,
  );

  if (unit) {
    context.fillStyle = theme.label;
    setChartFont(context, 10, 500);
    context.fillText(unit, 10, bounds.top - 6);
  }

  for (let tick = 0; tick <= 4; tick += 1) {
    const ratio = tick / 4;
    const y = bounds.bottom - plotHeight * ratio;
    const value = paddedMin + (paddedMax - paddedMin) * ratio;
    context.strokeStyle = theme.grid;
    context.beginPath();
    context.moveTo(bounds.left, y);
    context.lineTo(bounds.right, y);
    context.stroke();

    context.fillStyle = theme.label;
    setChartFont(context, 12, 500);
    context.fillText(formatNumber(value), 10, y + 4);
  }

  const xTicks = buildXAxisTicks(visibleSlice.rows, width < 900 ? 4 : 6);
  xTicks.forEach(({ index, item }) => {
    const x = bounds.left + (plotWidth * index) / Math.max(visibleSlice.rows.length - 1, 1);

    context.strokeStyle = theme.grid;
    context.beginPath();
    context.moveTo(x, bounds.top);
    context.lineTo(x, bounds.bottom);
    context.stroke();

    context.strokeStyle = theme.axis;
    context.beginPath();
    context.moveTo(x, bounds.bottom);
    context.lineTo(x, bounds.bottom + 6);
    context.stroke();

    context.fillStyle = theme.label;
    setChartFont(context, 11, 500);
    const timeLabel = formatAxisTime(item.timestamp);
    const timeWidth = context.measureText(timeLabel).width;
    context.fillText(timeLabel, x - timeWidth / 2, bounds.bottom + 22);

    setChartFont(context, 10, 400);
    const dateLabel = formatAxisDate(item.timestamp);
    const dateWidth = context.measureText(dateLabel).width;
    context.fillText(dateLabel, x - dateWidth / 2, bounds.bottom + 38);
  });

  const getPointX = (index) =>
    bounds.left + (plotWidth * index) / Math.max(visibleSlice.rows.length - 1, 1);
  const getPointY = (value) =>
    bounds.bottom - ((value - paddedMin) / (paddedMax - paddedMin || 1)) * plotHeight;

  const pointLayouts = [];
  const historyPoints = [];
  const modelPoints = [];
  const forecastPoints = [];
  const postfactumPoints = [];
  let lastHistoryIndex = -1;

  visibleSlice.rows.forEach((item, index) => {
    const x = getPointX(index);
    if (Number.isFinite(item.actual)) {
      const point = {
        timestamp: item.timestamp,
        value: item.actual,
        series: "history",
        x,
        y: getPointY(item.actual),
      };
      point.pointId = getPointId(point);
      historyPoints.push(point);
      pointLayouts.push(point);
      lastHistoryIndex = index;
    }

    if (Number.isFinite(item.model)) {
      const point = {
        timestamp: item.timestamp,
        value: item.model,
        series: "model",
        x,
        y: getPointY(item.model),
      };
      point.pointId = getPointId(point);
      modelPoints.push(point);
      pointLayouts.push(point);
    }

    if (Number.isFinite(item.forecast)) {
      const point = {
        timestamp: item.timestamp,
        value: item.forecast,
        series: "forecast",
        x,
        y: getPointY(item.forecast),
      };
      point.pointId = getPointId(point);
      forecastPoints.push(point);
      pointLayouts.push(point);
    }

    if (Number.isFinite(item.postfactum_actual)) {
      const point = {
        timestamp: item.timestamp,
        value: item.postfactum_actual,
        series: "postfactum_actual",
        x,
        y: getPointY(item.postfactum_actual),
      };
      point.pointId = getPointId(point);
      postfactumPoints.push(point);
      pointLayouts.push(point);
    }
  });

  if (lastHistoryIndex >= 0) {
    const dividerX = getPointX(lastHistoryIndex);
    context.strokeStyle = theme.divider;
    context.setLineDash([6, 6]);
    context.beginPath();
    context.moveTo(dividerX, bounds.top);
    context.lineTo(dividerX, bounds.bottom);
    context.stroke();
    context.setLineDash([]);
  }

  const drawSeries = (points, color, dashed = false, prependPoint = null) => {
    if (!points.length && !prependPoint) {
      return;
    }

    context.strokeStyle = color;
    context.lineWidth = 2.5;
    context.setLineDash(dashed ? [10, 6] : []);
    context.beginPath();

    const sourcePoints = prependPoint ? [prependPoint, ...points] : points;
    sourcePoints.forEach((point, index) => {
      if (index === 0) {
        context.moveTo(point.x, point.y);
      } else {
        context.lineTo(point.x, point.y);
      }
    });

    context.stroke();
    context.setLineDash([]);
  };

  drawSeries(historyPoints, theme.actual);
  drawSeries(modelPoints, theme.model);
  const bridgePoint =
    historyPoints.length && forecastPoints.length
      ? historyPoints[historyPoints.length - 1]
      : modelPoints.length && forecastPoints.length
        ? modelPoints[modelPoints.length - 1]
        : null;
  drawSeries(forecastPoints, theme.predicted, true, bridgePoint);
  drawSeries(postfactumPoints, theme.postfactum);

  chartState.renderedPoints = pointLayouts;
  const hoveredPoint = pointLayouts.find((point) => point.pointId === chartState.hoverPointId) || null;
  if (!hoveredPoint) {
    chartState.hoverPointId = null;
  }

  drawPointMarkers(context, historyPoints, theme.actual, 2.75);
  drawPointMarkers(context, modelPoints, theme.model, 2.75);
  drawPointMarkers(context, forecastPoints, theme.predicted, 3);
  drawPointMarkers(context, postfactumPoints, theme.postfactum, 3);

  context.fillStyle = theme.text;
  setChartFont(context, 11, 600);
  const zoomText = `Масштаб: ${visibleSlice.zoomLevel}x`;
  context.fillText(zoomText, bounds.right - context.measureText(zoomText).width, bounds.top - 16);

  const zoomEnabled = timelineRows.length > MIN_VISIBLE_POINTS;
  const modelLabel = state.model.name ? `${state.model.name} (${state.model.version_id})` : state.model.version_id;
  const baseText = `Проект ${state.model.project_id}, версия ${modelLabel}, цель ${state.model.target}.`;
  const rangeText =
    ` Интервал ${formatDuration(state.request.totalMinutes)}, точек ${state.request.pointCount}, ` +
    `шаг отображения ${formatDuration(state.request.displayStepMinutes)}, нативный шаг ${formatDuration(state.request.baseFrequencyMinutes)}.`;
  const modelText = modelPoints.length
    ? ``
    : ` Исторические точки модели недоступны для этой версии, поэтому на графике показан прогнозный хвост.`;
  const interactionText = zoomEnabled
    ? " Колесо мыши масштабирует график, перетаскивание двигает окно влево и вправо, наведение показывает значение точки."
    : " Наведите на точку, чтобы увидеть значение.";

  elements.chartTitle.textContent = unit ? `Прогноз модели ${modelLabel}, ${unit}` : `Прогноз модели ${modelLabel}`;
  updateChartControls(
    `${baseText}${rangeText}${modelText}${interactionText}`,
    zoomEnabled && !isViewportReset(),
    zoomEnabled,
  );

  if (hoveredPoint) {
    showTooltip(hoveredPoint);
  } else {
    hideTooltip();
  }
}

function renderChart() {
  if (state.error) {
    renderEmptyState(state.error, "Не удалось загрузить модель");
    return;
  }

  if (!state.model) {
    renderEmptyState("Обучите модель на странице запуска, чтобы открыть её график.", "Прогноз модели");
    return;
  }

  if (!state.model.forecasting?.available) {
    renderEmptyState(
      `Для модели ${state.model.version_id} временной прогноз недоступен. Нужна регрессионная модель с временным рядом.`,
      "Прогноз недоступен",
    );
    return;
  }

  if (!state.forecast) {
    renderEmptyState("Прогноз ещё не загружен.", "Прогноз модели");
    return;
  }

  elements.chartEmptyState.hidden = true;
  drawForecastChart();
}

function renderAll() {
  updateSelectedModelSummary();
  updateMetricsSummary();
  setInteractiveState();
  updateSidebarInfo();
  renderPredictionTable();
  renderChart();
}

function getCanvasMetrics() {
  const rect = elements.chartCanvas.getBoundingClientRect();
  const bounds = getPlotBounds(rect.width, rect.height);
  return {
    rect,
    bounds,
    plotWidth: bounds.right - bounds.left,
  };
}

function canZoomChart() {
  return getTimelineRows().length > MIN_VISIBLE_POINTS;
}

function isInsidePlot(clientX, clientY) {
  const { rect, bounds } = getCanvasMetrics();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  return x >= bounds.left && x <= bounds.right && y >= bounds.top && y <= bounds.bottom;
}

function getNearestRenderedPoint(clientX, clientY) {
  if (!chartState.renderedPoints.length || !isInsidePlot(clientX, clientY)) {
    return null;
  }

  const { rect } = getCanvasMetrics();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  const maxDistance = 18;
  const maxDistanceSq = maxDistance * maxDistance;

  let nearestPoint = null;
  let nearestDistanceSq = Number.POSITIVE_INFINITY;

  for (const point of chartState.renderedPoints) {
    const distanceSq = (point.x - x) ** 2 + (point.y - y) ** 2;
    if (distanceSq < nearestDistanceSq) {
      nearestDistanceSq = distanceSq;
      nearestPoint = point;
    }
  }

  return nearestDistanceSq <= maxDistanceSq ? nearestPoint : null;
}

function handleChartHover(event) {
  if (chartState.isDragging) {
    return;
  }

  const nearestPoint = getNearestRenderedPoint(event.clientX, event.clientY);
  if (!nearestPoint) {
    clearHoverPoint();
    return;
  }

  if (nearestPoint.pointId === chartState.hoverPointId) {
    showTooltip(nearestPoint);
    return;
  }

  chartState.hoverPointId = nearestPoint.pointId;
  showTooltip(nearestPoint);
}

function handleChartWheel(event) {
  if (!canZoomChart()) {
    return;
  }

  event.preventDefault();
  const total = getTimelineRows().length;
  const minWindow = Math.min(1, MIN_VISIBLE_POINTS / total);
  const currentWindow = clamp(chartState.viewEnd - chartState.viewStart, minWindow, 1);
  const zoomFactor = event.deltaY < 0 ? 0.86 : 1.16;
  const nextWindow = clamp(currentWindow * zoomFactor, minWindow, 1);

  if (Math.abs(nextWindow - currentWindow) < 0.0001) {
    return;
  }

  const { rect, bounds, plotWidth } = getCanvasMetrics();
  const relativeX = clamp((event.clientX - rect.left - bounds.left) / Math.max(plotWidth, 1), 0, 1);
  const focusPoint = chartState.viewStart + currentWindow * relativeX;
  const nextStart = clamp(focusPoint - nextWindow * relativeX, 0, 1 - nextWindow);

  chartState.viewStart = nextStart;
  chartState.viewEnd = nextStart + nextWindow;
  scheduleChartRender();
}

function handlePointerDown(event) {
  if (event.button !== 0 || !canZoomChart() || !isInsidePlot(event.clientX, event.clientY)) {
    return;
  }

  clearHoverPoint();
  chartState.isDragging = true;
  chartState.pointerId = event.pointerId;
  chartState.dragStartX = event.clientX;
  chartState.dragOriginStart = chartState.viewStart;
  chartState.dragWindow = chartState.viewEnd - chartState.viewStart;

  elements.chartCanvas.setPointerCapture(event.pointerId);
  scheduleChartRender();
  event.preventDefault();
}

function handlePointerMove(event) {
  if (!chartState.isDragging) {
    handleChartHover(event);
    return;
  }

  if (!chartState.isDragging || chartState.pointerId !== event.pointerId) {
    return;
  }

  const { plotWidth } = getCanvasMetrics();
  const deltaRatio = (event.clientX - chartState.dragStartX) / Math.max(plotWidth, 1);
  const nextStart = clamp(
    chartState.dragOriginStart - deltaRatio * chartState.dragWindow,
    0,
    1 - chartState.dragWindow,
  );

  chartState.viewStart = nextStart;
  chartState.viewEnd = nextStart + chartState.dragWindow;
  scheduleChartRender();
}

function stopDrag(pointerId = null) {
  if (!chartState.isDragging) {
    return;
  }
  if (pointerId !== null && chartState.pointerId !== pointerId) {
    return;
  }

  chartState.isDragging = false;
  chartState.pointerId = null;
  hideTooltip();
  scheduleChartRender();
}

async function loadForecastComparison(runId, token) {
  try {
    const comparison = await fetchJson(`/forecast/${encodeURIComponent(runId)}/comparison`);
    if (token !== state.forecastToken) {
      return;
    }
    state.forecastComparison = comparison;
    invalidateTimelineRows();
    renderAll();
  } catch {
    if (token !== state.forecastToken) {
      return;
    }
    state.forecastComparison = null;
  }
}

async function loadForecast(intervalMinutes, pointCount = state.selectedPointCount, { resetZoom = true } = {}) {
  if (!state.model?.forecasting?.available) {
    state.forecast = null;
    state.forecastComparison = null;
    state.displayForecast = null;
    invalidateTimelineRows();
    state.request = null;
    syncPointCountControls(null);
    renderAll();
    return;
  }

  const request = buildForecastRequest(state.model, intervalMinutes, pointCount);
  const token = ++state.forecastToken;

  state.isLoadingForecast = true;
  state.error = null;
  state.selectedIntervalMinutes = normalizeIntervalMinutes(request.intervalMinutes);
  state.selectedPointCount = request.pointCount;
  state.availablePointCounts = getAvailablePointCounts(request.totalMinutes, request.baseFrequencyMinutes);
  setInteractiveState();
  elements.forecastPanelNote.textContent = "Загружаю прогноз на нативной частоте модели...";

  try {
    const searchParams = new URLSearchParams({
      steps: String(request.backendSteps),
      horizon_minutes: String(request.backendHorizonMinutes),
    });
    const forecast = await fetchJson(`/models/${encodeURIComponent(state.model.version_id)}/forecast?${searchParams.toString()}`);
    if (token !== state.forecastToken) {
      return;
    }

    state.error = null;
    state.forecast = forecast;
    state.forecastComparison = null;
    state.request = request;
    state.displayForecast = buildDisplayForecast(forecast, request);
    invalidateTimelineRows();
    if (resetZoom) {
      resetViewport();
    }
    updateLocation();
    if (forecast.run_id) {
      loadForecastComparison(forecast.run_id, token);
    }
  } catch (error) {
    if (token !== state.forecastToken) {
      return;
    }

    state.forecast = null;
    state.forecastComparison = null;
    state.displayForecast = null;
    invalidateTimelineRows();
    state.request = request;
    state.error = error instanceof Error ? error.message : String(error);
  } finally {
    if (token === state.forecastToken) {
      state.isLoadingForecast = false;
      renderAll();
    }
  }
}

async function loadModelAndGraph() {
  const context = getRequestedContext();
  if (!context.projectId && !context.versionId) {
    context.projectId = await resolveDefaultProjectId();
  }
  syncNavigation(context.projectId);
  state.error = null;
  state.forecast = null;
  state.forecastComparison = null;
  state.displayForecast = null;
  invalidateTimelineRows();
  state.request = null;
  state.availablePointCounts = [];
  state.selectedPointCount = null;

  if (!context.projectId && !context.versionId) {
    state.model = null;
    state.error = "В реестре пока нет проектов. Создайте проект на стартовой странице.";
    renderAll();
    return;
  }

  try {
    const model = context.versionId
      ? await fetchJson(`/models/${encodeURIComponent(context.versionId)}`)
      : await fetchJson(`/models/latest?project_id=${encodeURIComponent(context.projectId)}`);
    state.model = model;
    syncNavigation(model.project_id);
  } catch (error) {
    state.model = null;
    state.error = error instanceof Error ? error.message : String(error);
    renderAll();
    return;
  }

  if (!state.model.forecasting?.available) {
    state.selectedIntervalMinutes = null;
    syncPointCountControls(null);
    renderAll();
    return;
  }

  state.selectedIntervalMinutes = getInitialIntervalMinutes(context, state.model);
  updateIntervalControls();
  syncPointCountControls(state.selectedIntervalMinutes, context.pointCount);
  await loadForecast(state.selectedIntervalMinutes, state.selectedPointCount);
}

function handleResize() {
  scheduleChartRender();
}

elements.chartResetButton.addEventListener("click", () => {
  resetViewport();
  scheduleChartRender();
});

elements.intervalSlider.addEventListener("input", () => {
  state.selectedIntervalMinutes = getSliderIntervalMinutes();
  updateIntervalControls();
});

elements.intervalSlider.addEventListener("change", async () => {
  const intervalMinutes = getSliderIntervalMinutes();
  if (state.isLoadingForecast || !state.model?.forecasting?.available) {
    return;
  }

  state.selectedIntervalMinutes = intervalMinutes;
  syncPointCountControls(intervalMinutes);
  await loadForecast(intervalMinutes, state.selectedPointCount);
});

elements.pointCountSlider.addEventListener("input", () => {
  const pointCount = getSliderPointCount();
  if (!pointCount) {
    return;
  }

  state.selectedPointCount = pointCount;
  updatePointCountControls();
});

elements.pointCountSlider.addEventListener("change", () => {
  const pointCount = getSliderPointCount();
  if (!pointCount || state.isLoadingForecast || !state.selectedIntervalMinutes) {
    return;
  }

  applyDisplayPointCount(pointCount);
});

elements.chartCanvas.addEventListener("wheel", handleChartWheel, { passive: false });
elements.chartCanvas.addEventListener("pointerdown", handlePointerDown);
elements.chartCanvas.addEventListener("pointermove", handlePointerMove);
elements.chartCanvas.addEventListener("pointerup", (event) => {
  stopDrag(event.pointerId);
});
elements.chartCanvas.addEventListener("pointercancel", (event) => {
  stopDrag(event.pointerId);
});
elements.chartCanvas.addEventListener("lostpointercapture", () => {
  stopDrag();
});
elements.chartCanvas.addEventListener("pointerleave", () => {
  clearHoverPoint();
});

window.addEventListener("resize", handleResize);

renderAll();
loadModelAndGraph();
