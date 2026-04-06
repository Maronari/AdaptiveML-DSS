const DEFAULT_PROJECT_ID = "demo";
const EMPTY_VALUE = "—";
const DEFAULT_PREVIEW_MESSAGE = "Загрузите CSV или Excel, и страница сразу проверит совместимость нового датасета с champion-моделью.";
const LOADING_PREVIEW_MESSAGE = "Загрузка..";

const state = {
  inspection: null,
  activeJob: null,
  pollingTimer: null,
  busy: false,
  projectId: DEFAULT_PROJECT_ID,
};

const elements = {
  fileInput: document.getElementById("retraining-file"),
  targetInput: document.getElementById("retraining-target"),
  taskTypeInput: document.getElementById("retraining-task-type"),
  backendInput: document.getElementById("retraining-backend"),
  historyScopeInput: document.getElementById("retraining-history-scope"),
  improvementThresholdInput: document.getElementById("retraining-improvement-threshold"),
  autoActivateInput: document.getElementById("retraining-auto-activate"),
  submitButton: document.getElementById("retraining-submit-button"),
  summaryRows: document.getElementById("retraining-summary-rows"),
  summaryColumns: document.getElementById("retraining-summary-columns"),
  summaryDuplicates: document.getElementById("retraining-summary-duplicates"),
  summarySource: document.getElementById("retraining-summary-source"),
  championBody: document.getElementById("retraining-champion-body"),
  compatibilityBody: document.getElementById("retraining-compatibility-body"),
  temporalBody: document.getElementById("retraining-temporal-body"),
  projectBody: document.getElementById("retraining-project-body"),
  previewHead: document.getElementById("retraining-preview-head"),
  previewBody: document.getElementById("retraining-preview-body"),
  previewEmpty: document.getElementById("retraining-preview-empty"),
  statusBanner: document.getElementById("retraining-status-banner"),
  jobStatusPill: document.getElementById("retraining-job-status-pill"),
  jobStatusNote: document.getElementById("retraining-job-status-note"),
  log: document.getElementById("retraining-log"),
  uploadLinks: Array.from(document.querySelectorAll("[data-nav-upload]")),
  trainingLinks: Array.from(document.querySelectorAll("[data-nav-training]")),
  retrainingLinks: Array.from(document.querySelectorAll("[data-nav-retraining]")),
  modelsLinks: Array.from(document.querySelectorAll("[data-nav-models]")),
  graphLinks: Array.from(document.querySelectorAll("[data-nav-graph]")),
  dssLinks: Array.from(document.querySelectorAll("[data-nav-dss]")),
};

function getPageContext() {
  const searchParams = new URLSearchParams(window.location.search);
  return {
    projectId: searchParams.get("project_id")?.trim() || DEFAULT_PROJECT_ID,
  };
}

function selectedFiles() {
  return Array.from(elements.fileInput.files || []);
}

function formatTaskType(taskType) {
  if (taskType === "regression") {
    return "Регрессия";
  }
  if (taskType === "binary") {
    return "Бинарная классификация";
  }
  if (taskType === "multiclass") {
    return "Мультиклассовая классификация";
  }
  return taskType || EMPTY_VALUE;
}

function formatNumber(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return EMPTY_VALUE;
  }
  const hasFraction = Math.abs(value % 1) > 1e-9;
  return new Intl.NumberFormat("ru-RU", {
    minimumFractionDigits: 0,
    maximumFractionDigits: hasFraction ? 3 : 0,
  }).format(value);
}

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return EMPTY_VALUE;
  }
  return `${formatNumber(value * 100)}%`;
}

function formatDateTime(value) {
  if (!value) {
    return EMPTY_VALUE;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function syncNavigation(projectId) {
  const normalizedProjectId = projectId.trim();
  const linkGroups = [
    [elements.uploadLinks, "./upload.html"],
    [elements.trainingLinks, "./training.html"],
    [elements.retrainingLinks, "./retraining.html"],
    [elements.modelsLinks, "./models.html"],
    [elements.graphLinks, "./graph.html"],
    [elements.dssLinks, "./dss.html"],
  ];

  for (const [links, path] of linkGroups) {
    for (const link of links) {
      const targetUrl = new URL(path, window.location.href);
      if (normalizedProjectId) {
        targetUrl.searchParams.set("project_id", normalizedProjectId);
      }
      link.href = targetUrl.toString();
    }
  }
}

function preferredForecastSteps() {
  return Math.min(24, Math.max(8, Math.round(window.innerWidth / 120)));
}

function redirectToGraph(projectId, versionId, forecasting = null) {
  const targetUrl = new URL("./graph.html", window.location.href);
  targetUrl.searchParams.set("project_id", projectId);
  targetUrl.searchParams.set("model_version", versionId);
  if (forecasting?.available) {
    targetUrl.searchParams.set("steps", String(preferredForecastSteps()));
    if (forecasting.default_horizon_minutes) {
      targetUrl.searchParams.set("horizon_minutes", String(forecasting.default_horizon_minutes));
    }
  }
  window.location.assign(targetUrl.toString());
}

function updateSubmitState() {
  elements.submitButton.disabled = state.busy || !state.inspection?.compatibility?.ready;
}

function setBusy(isBusy) {
  state.busy = isBusy;
  elements.fileInput.disabled = isBusy;
  elements.backendInput.disabled = isBusy;
  elements.historyScopeInput.disabled = isBusy;
  elements.improvementThresholdInput.disabled = isBusy;
  elements.autoActivateInput.disabled = isBusy;
  updateSubmitState();
}

function setStatus(kind, message) {
  elements.statusBanner.className = `status-banner status-${kind}`;
  elements.statusBanner.textContent = message;
}

function timestampLabel() {
  return new Date().toLocaleTimeString("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function appendLog(message, level = "info", timeText = timestampLabel()) {
  const entry = document.createElement("div");
  entry.className = "training-log-entry";
  const time = document.createElement("strong");
  time.textContent = timeText;
  const text = document.createElement("span");
  text.textContent = message;
  entry.append(time, text);
  elements.log.prepend(entry);

  while (elements.log.children.length > 16) {
    elements.log.removeChild(elements.log.lastChild);
  }

  if (level === "error") {
    console.error(`[retraining] ${message}`);
  } else {
    console.info(`[retraining] ${message}`);
  }
}

function normalizeError(error) {
  return error instanceof Error ? error.message : String(error ?? "Неизвестная ошибка.");
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

async function fetchJson(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  const payload = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, response));
  }
  return payload;
}

async function postUploads(path, { files, projectId, target = "" }) {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append(files.length > 1 ? "files" : "file", file);
  });
  formData.append("project_id", projectId);
  if (target) {
    formData.append("target", target);
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

async function postForm(path, formData) {
  const response = await fetch(path, { method: "POST", body: formData });
  const payload = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, response));
  }
  return payload;
}

function setDetailEmpty(container, message) {
  container.replaceChildren();
  const paragraph = document.createElement("p");
  paragraph.className = "upload-detail-empty";
  paragraph.textContent = message;
  container.append(paragraph);
}

function createDetailLine(label, value, accent = false) {
  const line = document.createElement("div");
  line.className = "upload-detail-line";

  const labelNode = document.createElement("span");
  labelNode.className = "upload-detail-line-label";
  labelNode.textContent = label;

  const valueNode = document.createElement("strong");
  valueNode.className = accent ? "upload-detail-line-value is-accent" : "upload-detail-line-value";
  valueNode.textContent = value;

  line.append(labelNode, valueNode);
  return line;
}

function renderDetailLines(container, rows, emptyMessage) {
  container.replaceChildren();
  if (!rows.length) {
    setDetailEmpty(container, emptyMessage);
    return;
  }
  rows.forEach((row) => {
    container.append(createDetailLine(row.label, row.value, row.accent === true));
  });
}

function setPreviewPlaceholder(message, isLoading = false) {
  elements.previewEmpty.textContent = message;
  elements.previewEmpty.hidden = false;
  elements.previewEmpty.classList.toggle("table-placeholder-loading", isLoading);
}

function clearPreviewTable() {
  elements.previewHead.replaceChildren();
  elements.previewBody.replaceChildren();
}

function clearInspection() {
  state.inspection = null;
  elements.targetInput.value = "";
  elements.taskTypeInput.value = "";
  elements.summaryRows.textContent = EMPTY_VALUE;
  elements.summaryColumns.textContent = EMPTY_VALUE;
  elements.summaryDuplicates.textContent = EMPTY_VALUE;
  elements.summarySource.textContent = EMPTY_VALUE;
  setDetailEmpty(elements.championBody, "После разбора файла здесь появится активная модель проекта.");
  setDetailEmpty(elements.compatibilityBody, "Загрузите файл, чтобы проверить схему и обязательные поля.");
  setDetailEmpty(elements.temporalBody, "Временная структура появится после разбора файла.");
  setDetailEmpty(elements.projectBody, "Контекст проекта появится после разбора файла.");
  clearPreviewTable();
  setPreviewPlaceholder(DEFAULT_PREVIEW_MESSAGE, false);
  updateSubmitState();
}

function renderPreviewTable() {
  const columns = state.inspection?.columns ?? [];
  const rows = state.inspection?.sample_rows ?? [];
  clearPreviewTable();

  if (!columns.length || !rows.length) {
    setPreviewPlaceholder(DEFAULT_PREVIEW_MESSAGE, false);
    return;
  }

  elements.previewEmpty.hidden = true;
  elements.previewEmpty.classList.remove("table-placeholder-loading");

  const headRow = document.createElement("tr");
  columns.forEach((column) => {
    const cell = document.createElement("th");
    cell.textContent = column;
    headRow.append(cell);
  });
  elements.previewHead.append(headRow);

  rows.forEach((record) => {
    const row = document.createElement("tr");
    columns.forEach((column) => {
      const cell = document.createElement("td");
      const value = record?.[column];
      cell.textContent = value == null || value === "" ? EMPTY_VALUE : String(value);
      row.append(cell);
    });
    elements.previewBody.append(row);
  });
}

function renderChampionCard() {
  const champion = state.inspection?.champion_model;
  const rows = champion
    ? [
        { label: "Версия", value: champion.version_id || EMPTY_VALUE, accent: true },
        { label: "Статус", value: champion.status || EMPTY_VALUE },
        { label: "Target", value: champion.target || EMPTY_VALUE },
        { label: "Задача", value: formatTaskType(champion.task_type) },
        { label: "Датасет", value: champion.dataset_version_id || EMPTY_VALUE },
        { label: "Создана", value: formatDateTime(champion.created_at) },
      ]
    : [];
  renderDetailLines(elements.championBody, rows, "Champion-модель пока не найдена.");
}

function renderCompatibilityCard() {
  const compatibility = state.inspection?.compatibility;
  if (!compatibility) {
    setDetailEmpty(elements.compatibilityBody, "Загрузите файл, чтобы проверить схему и обязательные поля.");
    return;
  }

  const missingInputs = compatibility.missing_inputs?.length
    ? compatibility.missing_inputs.join(", ")
    : "нет";
  const extraFeatures = compatibility.extra_features?.length
    ? compatibility.extra_features.join(", ")
    : "нет";
  const rows = [
    {
      label: "Статус",
      value: compatibility.ready ? "Готов к дообучению" : "Нужна корректировка",
      accent: compatibility.ready,
    },
    { label: "Target найден", value: compatibility.target_present ? "да" : "нет" },
    { label: "Недостающие поля", value: missingInputs },
    { label: "Лишние колонки", value: extraFeatures },
    {
      label: "Строк достаточно",
      value: compatibility.rows_check_passed
        ? `да, минимум ${compatibility.minimum_rows_required}`
        : `нет, минимум ${compatibility.minimum_rows_required}`,
    },
  ];
  if (compatibility.validation_error) {
    rows.push({ label: "Ошибка проверки", value: compatibility.validation_error });
  }
  renderDetailLines(elements.compatibilityBody, rows, "Нет данных о совместимости.");
}

function renderTemporalCard() {
  const temporal = state.inspection?.temporal_context;
  if (!temporal || !temporal.available) {
    setDetailEmpty(
      elements.temporalBody,
      temporal?.message || "Временная структура не распознана.",
    );
    return;
  }

  const rows = [
    { label: "Колонка", value: temporal.column || EMPTY_VALUE },
    { label: "Начало", value: formatDateTime(temporal.start) },
    { label: "Конец", value: formatDateTime(temporal.end) },
    { label: "Диапазон", value: temporal.range_label || EMPTY_VALUE },
    { label: "Частота", value: temporal.frequency_label || EMPTY_VALUE },
    { label: "Разрывы", value: temporal.has_gaps ? "есть" : "нет" },
  ];
  renderDetailLines(elements.temporalBody, rows, "Временная структура не распознана.");
}

function renderProjectCard() {
  const projectContext = state.inspection?.project_context;
  if (!projectContext) {
    setDetailEmpty(elements.projectBody, "Контекст проекта появится после разбора файла.");
    return;
  }

  const rows = [
    { label: "Проект", value: projectContext.project_name || projectContext.project_id || EMPTY_VALUE, accent: true },
    { label: "Версий датасета", value: formatNumber(projectContext.dataset_versions) },
    { label: "Версий модели", value: formatNumber(projectContext.model_versions) },
    { label: "Последний датасет", value: projectContext.latest_dataset_version_id || EMPTY_VALUE },
    { label: "Последняя модель", value: projectContext.latest_model_version_id || EMPTY_VALUE },
    { label: "Champion", value: projectContext.champion_model_version_id || EMPTY_VALUE },
  ];
  renderDetailLines(elements.projectBody, rows, "Контекст проекта недоступен.");
}

function applyInspection(inspection) {
  state.inspection = inspection;
  elements.targetInput.value = inspection.expected_target || "";
  elements.taskTypeInput.value = formatTaskType(inspection.champion_model?.task_type || "");
  elements.summaryRows.textContent = formatNumber(inspection.rows);
  elements.summaryColumns.textContent = formatNumber(inspection.columns?.length ?? 0);
  elements.summaryDuplicates.textContent = formatNumber(inspection.duplicates ?? 0);
  elements.summarySource.textContent = inspection.source_name || EMPTY_VALUE;
  renderChampionCard();
  renderCompatibilityCard();
  renderTemporalCard();
  renderProjectCard();
  renderPreviewTable();
  updateSubmitState();
}

function buildInspectPath(files) {
  return files.length > 1 ? "/retraining/inspect/files" : "/retraining/inspect/file";
}

function buildRegisterPath(files) {
  return files.length > 1 ? "/datasets/register/files" : "/datasets/register/file";
}

async function inspectFiles() {
  const files = selectedFiles();
  if (!files.length) {
    clearInspection();
    setStatus("idle", "Выберите новый датасет для дообучения.");
    return;
  }

  setBusy(true);
  setStatus("busy", "Проверяю новый пакет данных и совместимость с champion-моделью...");
  setPreviewPlaceholder(LOADING_PREVIEW_MESSAGE, true);
  appendLog(`Разбираю ${files.length > 1 ? "файлы" : "файл"} для дообучения.`);

  try {
    const inspection = await postUploads(buildInspectPath(files), {
      files,
      projectId: state.projectId,
    });
    applyInspection(inspection);
    const compatibility = inspection.compatibility;
    if (compatibility?.ready) {
      setStatus("success", `Файл подходит для дообучения модели ${inspection.champion_model.version_id}.`);
      appendLog(`Схема подтверждена. Champion=${inspection.champion_model.version_id}.`);
    } else {
      setStatus("error", compatibility?.validation_error || "Новый датасет не подходит для дообучения.");
      appendLog("Проверка совместимости завершилась с замечаниями.", "error");
    }
  } catch (error) {
    clearInspection();
    setStatus("error", normalizeError(error));
    appendLog(normalizeError(error), "error");
  } finally {
    setBusy(false);
  }
}

function buildRetrainingJobFormData(datasetVersionId) {
  const championTaskType = state.inspection?.champion_model?.task_type || "auto";
  const improvement = Number.parseFloat(elements.improvementThresholdInput.value || "3");
  if (!Number.isFinite(improvement) || improvement < 0) {
    throw new Error("Минимальное улучшение должно быть больше или равно 0.");
  }

  const formData = new FormData();
  formData.append("project_id", state.projectId);
  formData.append("dataset_version_id", datasetVersionId);
  formData.append("task_type", championTaskType);
  formData.append("backend", elements.backendInput.value);
  formData.append("preset", "tabular");
  formData.append("algos", "lgb,linear_l2");
  formData.append("timeout_seconds", "30");
  formData.append("cpu_limit", "1");
  formData.append("test_size", "0.2");
  formData.append("cv_folds", "0");
  formData.append("enable_forecast", "true");
  formData.append("history_scope", elements.historyScopeInput.value);
  formData.append("minimum_relative_improvement", String(improvement / 100));
  formData.append("evaluation_fraction", "0.2");
  formData.append("auto_activate", String(elements.autoActivateInput.checked));
  return formData;
}

function formatMetricValue(value) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return String(value);
}

function retrainingStatusMessage(retraining) {
  const profit = retraining.evaluation?.profit ?? null;
  let message =
    `Дообучение завершено: кандидат=${retraining.candidate_model_version.version_id}, окно=${retraining.selection_summary.history_scope}.`;
  if (profit) {
    message += ` ${profit.primary_metric}: ${formatMetricValue(profit.current_value)} -> ${formatMetricValue(profit.candidate_value)} (${formatMetricValue(profit.relative_gain_percent)}% прироста).`;
  }
  if (retraining.activated) {
    message += " Кандидат активирован.";
  } else if (retraining.activation_reason) {
    message += ` ${retraining.activation_reason}`;
  }
  return message;
}

function summarizeJob(job) {
  return `job=${job.job_id}, type=${job.job_type}, status=${job.status}`;
}

function setJobStatus(status, note = "") {
  const normalizedStatus = status || "idle";
  elements.jobStatusPill.className = `job-status-pill job-status-${normalizedStatus}`;
  elements.jobStatusPill.textContent = normalizedStatus;
  elements.jobStatusNote.textContent = note || "Очередь пуста.";
}

function clearPolling() {
  if (state.pollingTimer) {
    window.clearTimeout(state.pollingTimer);
    state.pollingTimer = null;
  }
}

function formatLogTimestamp(value) {
  if (!value) {
    return timestampLabel();
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return timestampLabel();
  }
  return date.toLocaleTimeString("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function renderJobLogs(job) {
  const entries = Array.isArray(job.logs) ? job.logs : [];
  elements.log.replaceChildren();
  entries.slice(-16).reverse().forEach((entry) => {
    appendLog(
      `${entry.source || "worker"}: ${entry.message}`,
      "info",
      formatLogTimestamp(entry.timestamp),
    );
  });
}

async function pollJob(jobId) {
  clearPolling();
  try {
    const job = await fetchJson(`/jobs/${encodeURIComponent(jobId)}`);
    state.activeJob = job;
    const jobStatus = String(job.status || "");
    renderJobLogs(job);

    if (jobStatus === "queued") {
      setJobStatus("queued", `Задача ${job.job_id} ожидает свободный worker.`);
      setStatus("busy", `Задача в очереди: ${summarizeJob(job)}`);
      state.pollingTimer = window.setTimeout(() => pollJob(jobId), 2000);
      return;
    }

    if (jobStatus === "running") {
      setJobStatus("running", `Задача ${job.job_id} выполняется worker ${job.worker_name || "—"}.`);
      setStatus("busy", `Задача выполняется: ${summarizeJob(job)}`);
      state.pollingTimer = window.setTimeout(() => pollJob(jobId), 2000);
      return;
    }

    setBusy(false);
    if (jobStatus === "done") {
      const result = job.result || {};
      setJobStatus("done", `Задача ${job.job_id} завершена.`);
      setStatus("success", retrainingStatusMessage(result));
      redirectToGraph(result.project_id, result.candidate_model_version.version_id, result.forecasting);
      return;
    }

    const errorMessage = job.error || "Фоновая задача завершилась ошибкой.";
    setJobStatus("failed", `Задача ${job.job_id} завершилась ошибкой.`);
    setStatus("error", errorMessage);
    appendLog(errorMessage, "error");
  } catch (error) {
    setBusy(false);
    setJobStatus("failed", "Не удалось получить статус фоновой задачи.");
    setStatus("error", normalizeError(error));
    appendLog(normalizeError(error), "error");
  }
}

async function enqueueAndPoll(actionLabel, path, formData) {
  appendLog(actionLabel);
  setBusy(true);
  setJobStatus("queued", "Отправляю задачу в очередь.");
  setStatus("busy", "Ставлю задачу в очередь...");
  const job = await postForm(path, formData);
  state.activeJob = job;
  setJobStatus("queued", `Задача ${job.job_id} поставлена в очередь.`);
  appendLog(`Задача поставлена в очередь: ${summarizeJob(job)}.`);
  await pollJob(job.job_id);
}

async function handleSubmit() {
  const files = selectedFiles();
  if (!files.length) {
    throw new Error("Сначала выберите новый датасет для дообучения.");
  }
  if (!state.inspection?.compatibility?.ready) {
    throw new Error("Новый датасет еще не прошел проверку совместимости.");
  }

  appendLog("Сохраняю новый пакет как версию датасета.");
  setBusy(true);
  setStatus("busy", "Сохраняю новую версию датасета...");

  try {
    const registeredDataset = await postUploads(buildRegisterPath(files), {
      files,
      projectId: state.projectId,
      target: state.inspection.expected_target,
    });
    const datasetVersionId = registeredDataset.dataset_version.version_id;
    appendLog(`Создана версия датасета ${datasetVersionId}.`);
    const formData = buildRetrainingJobFormData(datasetVersionId);
    await enqueueAndPoll(
      `Ставлю дообучение в очередь для проекта "${state.projectId}" по датасету "${datasetVersionId}".`,
      "/jobs/retraining/dataset",
      formData,
    );
  } catch (error) {
    setBusy(false);
    throw error;
  }
}

async function restoreActiveJob(projectId) {
  const payload = await fetchJson(
    `/jobs?project_id=${encodeURIComponent(projectId)}&job_type=retraining_dataset&limit=5`,
  );
  const candidate = (payload.items || []).find((job) => ["queued", "running"].includes(job.status));
  if (!candidate) {
    return;
  }
  state.activeJob = candidate;
  appendLog(`Найдена активная задача ${candidate.job_id}, возобновляю polling.`);
  setBusy(true);
  await pollJob(candidate.job_id);
}

elements.fileInput.addEventListener("change", () => {
  inspectFiles();
});

elements.submitButton.addEventListener("click", async () => {
  try {
    await handleSubmit();
  } catch (error) {
    setStatus("error", normalizeError(error));
    appendLog(normalizeError(error), "error");
  }
});

async function initializePage() {
  const pageContext = getPageContext();
  state.projectId = pageContext.projectId;
  syncNavigation(state.projectId);
  clearInspection();
  setBusy(false);
  setJobStatus("idle", "Очередь пуста.");
  setStatus("idle", "Загрузите новый датасет для проверки и дообучения.");
  appendLog(`Страница дообучения открыта для проекта "${state.projectId}".`);

  try {
    await restoreActiveJob(state.projectId);
  } catch (error) {
    setStatus("error", normalizeError(error));
    appendLog(normalizeError(error), "error");
  }
}

window.addEventListener("beforeunload", () => {
  clearPolling();
});

initializePage();
