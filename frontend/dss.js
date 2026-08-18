const DEFAULT_INTERVAL_MINUTES = 120;
const MIN_INTERVAL_MINUTES = 10;
const MAX_INTERVAL_MINUTES = 1440;
const INTERVAL_STEP_MINUTES = 10;
const DEFAULT_POINT_COUNT = 12;
const MAX_TOTAL_POINTS = 720;

const state = {
  projectId: "",
  requestedVersionId: "",
  datasetSummary: null,
  modelSummary: null,
  forecastPreview: null,
  forecastRows: [],
  recommendations: [],
  intervalMinutes: DEFAULT_INTERVAL_MINUTES,
  pointCount: DEFAULT_POINT_COUNT,
  request: null,
  ruleConfig: null,
  busy: false,
  error: null,
};

const elements = {
  projectTitle: document.getElementById("dss-project-title"),
  projectNote: document.getElementById("dss-project-note"),
  projectId: document.getElementById("dss-project-id"),
  datasetId: document.getElementById("dss-dataset-id"),
  modelId: document.getElementById("dss-model-id"),
  target: document.getElementById("dss-target"),
  actionNote: document.getElementById("dss-action-note"),
  horizonInput: document.getElementById("dss-horizon-input"),
  stepsInput: document.getElementById("dss-steps-input"),
  previewButton: document.getElementById("dss-preview-button"),
  runButton: document.getElementById("dss-run-button"),
  sampleHead: document.getElementById("dss-sample-head"),
  sampleBody: document.getElementById("dss-sample-body"),
  sampleEmpty: document.getElementById("dss-sample-empty"),
  resultsTitle: document.getElementById("dss-results-title"),
  resultsNote: document.getElementById("dss-results-note"),
  resultsEmpty: document.getElementById("dss-results-empty"),
  resultsList: document.getElementById("dss-results-list"),
  statusBanner: document.getElementById("dss-status-banner"),
  rulesNote: document.getElementById("dss-rules-note"),
  rulesReloadButton: document.getElementById("dss-rules-reload"),
  rulesSaveButton: document.getElementById("dss-rules-save"),
  rulesEditor: document.getElementById("dss-rules-editor"),
  uploadLinks: Array.from(document.querySelectorAll("[data-nav-upload]")),
  trainingLinks: Array.from(document.querySelectorAll("[data-nav-training]")),
  retrainingLinks: Array.from(document.querySelectorAll("[data-nav-retraining]")),
  modelsLinks: Array.from(document.querySelectorAll("[data-nav-models]")),
  reportLinks: Array.from(document.querySelectorAll("[data-nav-report]")),
  graphLinks: Array.from(document.querySelectorAll("[data-nav-graph]")),
  dssLinks: Array.from(document.querySelectorAll("[data-nav-dss]")),
};

function getPageContext() {
  const searchParams = new URLSearchParams(window.location.search);
  const requestedInterval = Number.parseInt(searchParams.get("interval_minutes") || searchParams.get("horizon_minutes") || "", 10);
  const requestedPointCount = Number.parseInt(searchParams.get("point_count") || searchParams.get("steps") || "", 10);
  return {
    projectId: searchParams.get("project_id")?.trim() || "",
    versionId: searchParams.get("model_version")?.trim() || "",
    intervalMinutes: Number.isFinite(requestedInterval) && requestedInterval > 0 ? requestedInterval : null,
    pointCount: Number.isFinite(requestedPointCount) && requestedPointCount > 0 ? requestedPointCount : null,
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
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

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => {
    const replacements = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return replacements[character];
  });
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
  return response.statusText || "Ошибка запроса.";
}

async function fetchJson(path, options = undefined) {
  const response = await fetch(path, options);
  const payload = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, response));
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

async function putJson(path, payload) {
  const response = await fetch(path, {
    method: "PUT",
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

function setStatus(kind, message) {
  elements.statusBanner.className = `status-banner status-${kind}`;
  elements.statusBanner.textContent = message;
}

function formatDateTime(value) {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
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

function formatNumber(value, maximumFractionDigits = 4) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return "—";
  }
  return numericValue.toLocaleString("ru-RU", {
    minimumFractionDigits: 0,
    maximumFractionDigits,
  });
}

function updateLocation() {
  if (!state.modelSummary) {
    return;
  }

  const currentUrl = new URL(window.location.href);
  currentUrl.searchParams.set("project_id", state.projectId);
  currentUrl.searchParams.set("model_version", state.modelSummary.version_id);
  if (state.request) {
    currentUrl.searchParams.set("interval_minutes", String(state.request.intervalMinutes));
    currentUrl.searchParams.set("point_count", String(state.request.pointCount));
    currentUrl.searchParams.delete("horizon_minutes");
    currentUrl.searchParams.delete("steps");
  }
  window.history.replaceState({}, "", currentUrl);
}

function syncNavigation(projectId) {
  const targets = [
    [elements.uploadLinks, "./upload.html"],
    [elements.trainingLinks, "./training.html"],
    [elements.retrainingLinks, "./retraining.html"],
    [elements.modelsLinks, "./models.html"],
    [elements.reportLinks, "./report.html"],
    [elements.graphLinks, "./graph.html"],
    [elements.dssLinks, "./dss.html"],
  ];

  const versionedPaths = new Set(["./graph.html", "./dss.html", "./report.html"]);

  targets.forEach(([links, relativePath]) => {
    for (const link of links) {
      const targetUrl = new URL(relativePath, window.location.href);
      if (projectId) {
        targetUrl.searchParams.set("project_id", projectId);
      }
      if (versionedPaths.has(relativePath) && state.modelSummary?.version_id) {
        targetUrl.searchParams.set("model_version", state.modelSummary.version_id);
      }
      if ((relativePath === "./graph.html" || relativePath === "./dss.html") && state.request) {
        targetUrl.searchParams.set("interval_minutes", String(state.request.intervalMinutes));
        targetUrl.searchParams.set("point_count", String(state.request.pointCount));
      }
      link.href = targetUrl.toString();
    }
  });
}

function syncRequestInputs() {
  elements.horizonInput.value = String(state.intervalMinutes);
  elements.stepsInput.value = String(state.pointCount);
}

function readRequestedForecastSettings() {
  const intervalValue = Number.parseInt(elements.horizonInput?.value ?? "", 10);
  const pointCountValue = Number.parseInt(elements.stepsInput?.value ?? "", 10);
  state.intervalMinutes = normalizeIntervalMinutes(intervalValue);
  state.pointCount = Number.isFinite(pointCountValue) && pointCountValue > 0 ? pointCountValue : DEFAULT_POINT_COUNT;
  syncRequestInputs();
}

function renderForecastTable() {
  elements.sampleHead.replaceChildren();
  elements.sampleBody.replaceChildren();

  if (!state.forecastRows.length) {
    elements.sampleEmpty.hidden = false;
    elements.sampleEmpty.textContent = "Для проекта нет будущих точек прогноза, доступных для СППР.";
    return;
  }

  const columns = ["step", "timestamp", "prediction"];
  const labels = {
    step: "Шаг",
    timestamp: "Время прогноза",
    prediction: "Прогноз",
  };
  const headRow = document.createElement("tr");
  columns.forEach((column) => {
    const cell = document.createElement("th");
    cell.textContent = labels[column] || column;
    headRow.append(cell);
  });
  elements.sampleHead.append(headRow);

  const fragment = document.createDocumentFragment();
  state.forecastRows.forEach((record) => {
    const row = document.createElement("tr");

    const stepCell = document.createElement("td");
    stepCell.textContent = String(record.step ?? "—");
    row.append(stepCell);

    const timestampCell = document.createElement("td");
    timestampCell.textContent = formatDateTime(record.timestamp);
    row.append(timestampCell);

    const predictionCell = document.createElement("td");
    predictionCell.textContent = formatNumber(record.prediction, 6);
    row.append(predictionCell);

    fragment.append(row);
  });

  elements.sampleBody.append(fragment);
  elements.sampleEmpty.hidden = true;
}

function renderResults() {
  elements.resultsList.replaceChildren();

  if (!state.recommendations.length) {
    elements.resultsList.hidden = true;
    elements.resultsEmpty.hidden = false;
    return;
  }

  const fragment = document.createDocumentFragment();
  state.recommendations.forEach((item) => {
    const card = document.createElement("article");
    card.className = "dss-result-card";

    const recommendation = item.recommendation || {};
    const factors = Array.isArray(item.top_factors) ? item.top_factors : [];

    card.innerHTML = `
      <div class="dss-result-head">
        <div>
          <strong>Шаг ${escapeHtml(item.step ?? item.row_index ?? "—")} · ${escapeHtml(formatDateTime(item.timestamp))}</strong>
          <p>${escapeHtml(recommendation.summary || "Рекомендация сформирована.")}</p>
        </div>
        <span class="dss-risk-pill risk-${escapeHtml(recommendation.risk_level || "unknown")}">${escapeHtml(recommendation.risk_level || "unknown")}</span>
      </div>
      <div class="dss-result-metrics">
        <span><strong>Prediction:</strong> ${formatNumber(item.prediction, 6)}</span>
        <span><strong>Confidence:</strong> ${formatNumber(item.confidence, 4)}</span>
      </div>
      <div class="dss-result-block">
        <strong>Действия</strong>
        <ul class="dss-result-list">
          ${(recommendation.actions || []).map((action) => `<li>${escapeHtml(action)}</li>`).join("") || "<li>—</li>"}
        </ul>
      </div>
      <div class="dss-result-block">
        <strong>Обоснование</strong>
        <ul class="dss-result-list">
          ${(recommendation.rationale || []).map((row) => `<li>${escapeHtml(row)}</li>`).join("") || "<li>—</li>"}
        </ul>
      </div>
      <div class="dss-result-block">
        <strong>Топ-факторы прогноза</strong>
        <div class="dss-factor-list">
          ${factors.map((factor) => `
            <span class="dss-factor-chip">${escapeHtml(factor.feature || "factor")}: ${formatNumber(factor.impact_score, 4)}</span>
          `).join("") || '<span class="dss-factor-chip">—</span>'}
        </div>
      </div>
    `;

    fragment.append(card);
  });

  elements.resultsList.append(fragment);
  elements.resultsList.hidden = false;
  elements.resultsEmpty.hidden = true;
}

function renderPage() {
  const title = state.projectId ? `СППР проекта ${state.projectId}` : "СППР проекта";
  elements.projectTitle.textContent = title;
  elements.projectId.textContent = state.projectId || "—";
  elements.datasetId.textContent = state.datasetSummary?.dataset_version?.version_id || "—";
  elements.modelId.textContent = state.modelSummary
    ? state.modelSummary.name
      ? `${state.modelSummary.name} (${state.modelSummary.version_id})`
      : state.modelSummary.version_id
    : "—";
  elements.target.textContent = state.datasetSummary?.target || state.modelSummary?.target || "—";
  syncRequestInputs();

  if (state.error) {
    elements.projectNote.textContent = state.error;
    elements.actionNote.textContent = state.error;
  } else if (!state.datasetSummary || !state.modelSummary) {
    elements.projectNote.textContent = "Для запуска СППР нужен последний датасет и хотя бы одна обученная модель.";
    elements.actionNote.textContent = "Подготовьте данные и обучите champion-модель проекта.";
  } else if (!state.modelSummary.forecasting?.available) {
    elements.projectNote.textContent = "Для выбранной модели недоступен forecasting head.";
    elements.actionNote.textContent = "Выберите другую модель или переобучите проект с прогнозированием.";
  } else if (!state.request) {
    elements.projectNote.textContent = `Модель ${state.modelSummary.version_id} готова к анализу прогноза.`;
    elements.actionNote.textContent = "Сформируйте прогноз для выбранного интервала и числа точек.";
  } else {
    const warning = state.forecastPreview?.warning;
    elements.projectNote.textContent =
      `СППР использует ту же версию модели, что и график: ${state.modelSummary.version_id}.`;
    elements.actionNote.textContent =
      `Интервал вперёд ${formatDuration(state.request.totalMinutes)}. Точек ${state.request.pointCount}, шаг отображения ${formatDuration(state.request.displayStepMinutes)}, нативный шаг модели ${formatDuration(state.request.baseFrequencyMinutes)}.${warning ? ` ${warning}` : ""}`;
  }

  const disabled = state.busy || !state.modelSummary?.forecasting?.available;
  elements.horizonInput.disabled = disabled;
  elements.stepsInput.disabled = disabled;
  elements.previewButton.disabled = disabled;
  elements.runButton.disabled = disabled || !state.forecastRows.length;
  elements.rulesReloadButton.disabled = state.busy;
  elements.rulesSaveButton.disabled = state.busy;
  elements.rulesEditor.disabled = state.busy;
  elements.rulesEditor.value = state.ruleConfig ? JSON.stringify(state.ruleConfig, null, 2) : "";
  elements.rulesNote.textContent = state.ruleConfig
    ? `Default rule set: ${state.ruleConfig.default_rule_set}. Можно редактировать сценарии и правила прямо здесь.`
    : "JSON-конфиг правил будет загружен автоматически.";

  renderForecastTable();
  renderResults();
}

async function loadForecastPreview() {
  if (!state.modelSummary?.version_id) {
    state.forecastPreview = null;
    state.forecastRows = [];
    state.request = null;
    renderPage();
    return null;
  }

  readRequestedForecastSettings();
  const request = buildForecastRequest(state.modelSummary, state.intervalMinutes, state.pointCount);
  const path = `/models/${encodeURIComponent(state.modelSummary.version_id)}/forecast?horizon_minutes=${encodeURIComponent(request.backendHorizonMinutes)}&steps=${encodeURIComponent(request.backendSteps)}`;
  const payload = await fetchJson(path);
  state.request = request;
  state.forecastPreview = payload;
  state.forecastRows = buildDisplayForecast(payload, request);
  updateLocation();
  syncNavigation(state.projectId);
  return payload;
}

async function loadContext() {
  const context = getPageContext();
  state.projectId = context.projectId;
  state.requestedVersionId = context.versionId;
  state.intervalMinutes = normalizeIntervalMinutes(context.intervalMinutes || DEFAULT_INTERVAL_MINUTES);
  state.pointCount = context.pointCount || DEFAULT_POINT_COUNT;
  syncRequestInputs();
  syncNavigation(state.projectId);
  state.error = null;
  state.recommendations = [];

  if (!state.projectId && !state.requestedVersionId) {
    state.projectId = await resolveDefaultProjectId();
    syncNavigation(state.projectId);
  }

  if (!state.projectId && !state.requestedVersionId) {
    state.datasetSummary = null;
    state.modelSummary = null;
    state.error = "В реестре пока нет проектов. Создайте проект на стартовой странице.";
    setStatus("error", state.error);
    renderPage();
    return;
  }

  try {
    const modelSummary = state.requestedVersionId
      ? await fetchJson(`/models/${encodeURIComponent(state.requestedVersionId)}`)
      : await fetchJson(`/models/latest?project_id=${encodeURIComponent(state.projectId)}`);
    state.modelSummary = modelSummary;
    state.projectId = modelSummary.project_id || state.projectId;
    const [datasetSummary, ruleConfig] = await Promise.all([
      fetchJson(`/projects/${encodeURIComponent(state.projectId)}/datasets/latest`),
      fetchJson("/dss/rulesets"),
    ]);
    state.datasetSummary = datasetSummary;
    state.ruleConfig = ruleConfig;

    if (modelSummary?.forecasting?.available) {
      await loadForecastPreview();
    } else {
      state.forecastPreview = null;
      state.forecastRows = [];
      state.request = null;
    }
    syncNavigation(state.projectId);
    setStatus("success", "Контекст СППР загружен.");
  } catch (error) {
    state.datasetSummary = null;
    state.modelSummary = null;
    state.forecastPreview = null;
    state.forecastRows = [];
    state.request = null;
    state.error = error instanceof Error ? error.message : String(error);
    setStatus("error", state.error);
  }

  renderPage();
}

async function reloadRuleConfig() {
  state.busy = true;
  state.error = null;
  renderPage();
  setStatus("busy", "Перезагружаю DSS rules...");
  try {
    state.ruleConfig = await fetchJson("/dss/rulesets");
    setStatus("success", "DSS rules обновлены из registry.");
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
    setStatus("error", state.error);
  } finally {
    state.busy = false;
    renderPage();
  }
}

async function saveRuleConfig() {
  state.busy = true;
  state.error = null;
  renderPage();
  setStatus("busy", "Сохраняю DSS rules...");
  try {
    const parsed = JSON.parse(elements.rulesEditor.value || "{}");
    state.ruleConfig = await putJson("/dss/rulesets", parsed);
    setStatus("success", "DSS rules сохранены.");
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
    setStatus("error", state.error);
  } finally {
    state.busy = false;
    renderPage();
  }
}

async function refreshForecastPreview() {
  if (!state.modelSummary?.forecasting?.available) {
    return;
  }

  readRequestedForecastSettings();
  state.busy = true;
  state.error = null;
  renderPage();
  setStatus("busy", "Обновляю прогноз для СППР...");

  try {
    await loadForecastPreview();
    state.recommendations = [];
    elements.resultsTitle.textContent = "Результаты СППР";
    elements.resultsNote.textContent = `Подготовлено будущих точек прогноза: ${state.forecastRows.length}.`;
    setStatus("success", `Прогноз обновлён: ${state.forecastRows.length} точек.`);
  } catch (error) {
    state.forecastPreview = null;
    state.forecastRows = [];
    state.request = null;
    state.recommendations = [];
    state.error = error instanceof Error ? error.message : String(error);
    elements.resultsNote.textContent = state.error;
    setStatus("error", state.error);
  } finally {
    state.busy = false;
    renderPage();
  }
}

async function runDecision() {
  if (!state.modelSummary?.forecasting?.available) {
    return;
  }

  readRequestedForecastSettings();
  state.busy = true;
  state.error = null;
  renderPage();
  setStatus("busy", "Запускаю СППР по прогнозу...");
  elements.resultsNote.textContent = "Использую ту же модель и те же точки прогноза, что и страница графика.";

  try {
    await loadForecastPreview();
    const payload = await fetchJson("/decision/forecast", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        project_id: state.projectId,
        version_id: state.modelSummary.version_id,
        horizon_minutes: state.request.backendHorizonMinutes,
        steps: state.request.backendSteps,
        point_count: state.request.pointCount,
      }),
    });
    state.recommendations = Array.isArray(payload.recommendations) ? payload.recommendations : [];
    elements.resultsTitle.textContent = `Результаты СППР ${payload.model_version || ""}`.trim();
    elements.resultsNote.textContent =
      `Получено рекомендаций: ${state.recommendations.length}. Последний факт: ${formatNumber(payload.baseline?.last_actual, 6)}, P90 истории: ${formatNumber(payload.baseline?.recent_p90, 6)}.`;
    setStatus("success", `СППР завершён: ${state.recommendations.length} рекомендаций по прогнозу.`);
  } catch (error) {
    state.recommendations = [];
    state.error = error instanceof Error ? error.message : String(error);
    elements.resultsNote.textContent = state.error;
    setStatus("error", state.error);
  } finally {
    state.busy = false;
    renderPage();
  }
}

elements.horizonInput.addEventListener("change", () => {
  readRequestedForecastSettings();
});

elements.stepsInput.addEventListener("change", () => {
  readRequestedForecastSettings();
});

elements.previewButton.addEventListener("click", () => {
  refreshForecastPreview();
});

elements.runButton.addEventListener("click", () => {
  runDecision();
});

elements.rulesReloadButton.addEventListener("click", () => {
  reloadRuleConfig();
});

elements.rulesSaveButton.addEventListener("click", () => {
  saveRuleConfig();
});

renderPage();
loadContext();
