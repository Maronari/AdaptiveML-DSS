const state = {
  datasetSummary: null,
  activeJob: null,
  pollingTimer: null,
  busy: false,
};

const elements = {
  datasetVersionInput: document.getElementById("dataset-version-id"),
  datasetSelectionNote: document.getElementById("dataset-selection-note"),
  projectIdInput: document.getElementById("project-id"),
  uploadLinks: Array.from(document.querySelectorAll("[data-nav-upload]")),
  trainingLinks: Array.from(document.querySelectorAll("[data-nav-training]")),
  retrainingLinks: Array.from(document.querySelectorAll("[data-nav-retraining]")),
  modelsLinks: Array.from(document.querySelectorAll("[data-nav-models]")),
  graphLinks: Array.from(document.querySelectorAll("[data-nav-graph]")),
  dssLinks: Array.from(document.querySelectorAll("[data-nav-dss]")),
  targetInput: document.getElementById("target-column"),
  taskTypeInput: document.getElementById("task-type"),
  backendInput: document.getElementById("training-backend"),
  presetInput: document.getElementById("training-preset"),
  cvFoldsInput: document.getElementById("cv-folds"),
  timeoutSecondsInput: document.getElementById("timeout-seconds"),
  cpuLimitInput: document.getElementById("cpu-limit"),
  testSizeInput: document.getElementById("test-size"),
  algoInputs: Array.from(document.querySelectorAll('input[name="algo-option"]')),
  trainingOptionsNote: document.getElementById("training-options-note"),
  validateButton: document.getElementById("validate-button"),
  trainButton: document.getElementById("train-button"),
  monitorButton: document.getElementById("monitor-button"),
  statusBanner: document.getElementById("status-banner"),
  jobStatusPill: document.getElementById("job-status-pill"),
  jobStatusNote: document.getElementById("job-status-note"),
  trainingLog: document.getElementById("training-log"),
};

function getPageContext() {
  const searchParams = new URLSearchParams(window.location.search);
  return {
    projectId: searchParams.get("project_id")?.trim() || "demo",
    datasetVersionId: searchParams.get("dataset_version_id")?.trim() || "",
  };
}

function actionButtons() {
  return [elements.validateButton, elements.trainButton, elements.monitorButton].filter(Boolean);
}

function selectedAlgorithms() {
  return elements.algoInputs.filter((input) => input.checked).map((input) => input.value);
}

function getWorkflowState() {
  return {
    datasetVersionId: elements.datasetVersionInput.value.trim(),
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
      enable_forecast: true,
    },
  };
}

function setBusy(isBusy) {
  state.busy = isBusy;
  for (const button of actionButtons()) {
    button.disabled = isBusy || !state.datasetSummary;
  }
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
  elements.trainingLog.prepend(entry);

  while (elements.trainingLog.children.length > 16) {
    elements.trainingLog.removeChild(elements.trainingLog.lastChild);
  }

  if (level === "error") {
    console.error(`[training] ${message}`);
  } else {
    console.info(`[training] ${message}`);
  }
}

function normalizeError(error) {
  return error instanceof Error ? error.message : String(error ?? "Неизвестная ошибка.");
}

function formatMetricValue(value) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return String(value);
}

function requireDatasetWorkflow() {
  const workflow = getWorkflowState();
  if (!workflow.datasetVersionId || !state.datasetSummary) {
    throw new Error("Сначала загрузите датасет на странице «Данные».");
  }
  if (!workflow.target) {
    throw new Error("Не удалось определить target для выбранного датасета.");
  }
  return workflow;
}

function requireTrainingWorkflow() {
  const workflow = requireDatasetWorkflow();
  if (!Number.isFinite(workflow.trainingOptions.timeout_seconds) || workflow.trainingOptions.timeout_seconds < 5) {
    throw new Error("Таймаут обучения должен быть не меньше 5 секунд.");
  }
  if (!Number.isFinite(workflow.trainingOptions.cpu_limit) || workflow.trainingOptions.cpu_limit < 1) {
    throw new Error("Лимит CPU должен быть не меньше 1.");
  }
  if (!Number.isFinite(workflow.trainingOptions.test_size) || workflow.trainingOptions.test_size <= 0 || workflow.trainingOptions.test_size >= 1) {
    throw new Error("Размер holdout должен быть больше 0 и меньше 1.");
  }
  if (!Number.isFinite(workflow.trainingOptions.cv_folds) || workflow.trainingOptions.cv_folds < 0 || workflow.trainingOptions.cv_folds === 1) {
    throw new Error("Количество фолдов CV должно быть 0 для авто или не меньше 2.");
  }
  if (workflow.trainingOptions.backend !== "sklearn" && workflow.trainingOptions.algos.length === 0) {
    throw new Error("Выберите хотя бы один алгоритм LightAutoML.");
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

async function postForm(path, formData) {
  const response = await fetch(path, { method: "POST", body: formData });
  const payload = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, response));
  }
  return payload;
}

function buildTrainingFormData({ projectId, datasetVersionId, trainingOptions }) {
  const formData = new FormData();
  formData.append("project_id", projectId);
  formData.append("dataset_version_id", datasetVersionId);
  formData.append("task_type", trainingOptions.task_type);
  formData.append("backend", trainingOptions.backend);
  formData.append("preset", trainingOptions.preset);
  formData.append("algos", trainingOptions.algos.join(","));
  formData.append("timeout_seconds", String(trainingOptions.timeout_seconds));
  formData.append("cpu_limit", String(trainingOptions.cpu_limit));
  formData.append("test_size", String(trainingOptions.test_size));
  formData.append("cv_folds", String(trainingOptions.cv_folds));
  formData.append("enable_forecast", String(trainingOptions.enable_forecast));
  return formData;
}

function trainingStatusMessage(training) {
  const effectiveOptions = training.training_options?.effective ?? {};
  const presetSuffix = effectiveOptions.preset ? `/${effectiveOptions.preset}` : "";
  let message =
    `Обучение завершено: backend=${training.backend}${presetSuffix}, модель=${training.model_version.version_id}.`;
  if (training.mlflow?.run_id) {
    message += ` MLflow run=${training.mlflow.run_id}.`;
  }
  if (training.warnings?.length) {
    message += ` Предупреждение: ${training.warnings[0]}`;
  }
  return message;
}

function monitoringStatusMessage(monitoring) {
  const metrics = monitoring.metrics ?? {};
  let message =
    `Мониторинг завершен: drifted=${formatMetricValue(metrics.drifted_columns_count ?? 0)} колонок, share=${formatMetricValue(metrics.drifted_columns_share ?? 0)}.`;
  if (monitoring.mlflow?.run_id) {
    message += ` MLflow run=${monitoring.mlflow.run_id}.`;
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

  elements.trainingOptionsNote.textContent = lightAutoMLSelected
    ? "Пресет, алгоритмы, CV и таймаут применяются только к LightAutoML."
    : "Для sklearn используется локальный pipeline на random forest. Пресет, алгоритмы, CV и таймаут игнорируются.";
}

function preferredForecastSteps() {
  return Math.min(24, Math.max(8, Math.round(window.innerWidth / 120)));
}

function syncProjectNavigation(projectId) {
  const normalizedProjectId = projectId.trim();
  const datasetVersionId = elements.datasetVersionInput.value.trim();
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
      if (path === "./training.html" && datasetVersionId) {
        targetUrl.searchParams.set("dataset_version_id", datasetVersionId);
      }
      link.href = targetUrl.toString();
    }
  }
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

function applyDatasetSummary(summary) {
  state.datasetSummary = summary;
  elements.datasetVersionInput.value = summary.dataset_version.version_id;
  elements.projectIdInput.value = summary.project_id;
  elements.targetInput.value = summary.target;
  elements.datasetSelectionNote.textContent =
    `Подключен датасет ${summary.dataset_version.version_id} (${summary.source_name}, ${summary.rows} строк, ${summary.columns.length} колонок).`;
  syncProjectNavigation(summary.project_id);
  setBusy(false);
}

function clearDatasetSummary(message) {
  state.datasetSummary = null;
  elements.datasetVersionInput.value = "";
  elements.targetInput.value = "";
  elements.datasetSelectionNote.textContent = message;
  setBusy(false);
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

function summarizeJob(job) {
  return `job=${job.job_id}, type=${job.job_type}, status=${job.status}`;
}

function setJobStatus(status, note = "") {
  const normalizedStatus = status || "idle";
  elements.jobStatusPill.className = `job-status-pill job-status-${normalizedStatus}`;
  elements.jobStatusPill.textContent = normalizedStatus;
  elements.jobStatusNote.textContent = note || "Очередь пуста.";
}

function renderJobLogs(job) {
  const entries = Array.isArray(job.logs) ? job.logs : [];
  elements.trainingLog.replaceChildren();
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
      setJobStatus("done", `Задача ${job.job_id} завершена.`);
      const result = job.result || {};
      if (job.job_type === "training_dataset") {
        setStatus("success", trainingStatusMessage(result));
        redirectToGraph(result.project_id, result.model_version.version_id, result.forecasting);
        return;
      }
      if (job.job_type === "monitoring_project") {
        setStatus("success", monitoringStatusMessage(result));
        appendLog(`Отчет drift: ${result.artifacts?.html_report || "—"}`);
        return;
      }
      setStatus("success", `Задача завершена: ${summarizeJob(job)}`);
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

async function handleValidate() {
  const workflow = requireDatasetWorkflow();
  setStatus("success", `Датасет готов: ${workflow.target}, ${state.datasetSummary.rows} строк.`);
}

async function handleTrain() {
  const workflow = requireTrainingWorkflow();
  const formData = buildTrainingFormData({
    projectId: workflow.projectId,
    datasetVersionId: workflow.datasetVersionId,
    trainingOptions: workflow.trainingOptions,
  });
  await enqueueAndPoll(
    `Ставлю обучение в очередь для проекта "${workflow.projectId}" по датасету "${workflow.datasetVersionId}".`,
    "/jobs/training/dataset",
    formData,
  );
}

async function handleMonitoring() {
  const workflow = requireDatasetWorkflow();
  const formData = new FormData();
  formData.append("project_id", workflow.projectId);
  await enqueueAndPoll(
    `Ставлю мониторинг drift в очередь для проекта "${workflow.projectId}".`,
    "/jobs/monitoring/project",
    formData,
  );
}

async function resolveDatasetSummary(pageContext) {
  if (pageContext.datasetVersionId) {
    return fetchJson(`/datasets/${encodeURIComponent(pageContext.datasetVersionId)}`);
  }
  return fetchJson(`/projects/${encodeURIComponent(pageContext.projectId)}/datasets/latest`);
}

async function restoreActiveJob(projectId) {
  const payload = await fetchJson(`/jobs?project_id=${encodeURIComponent(projectId)}&limit=5`);
  const candidate = (payload.items || []).find((job) => ["queued", "running"].includes(job.status));
  if (!candidate) {
    return;
  }
  state.activeJob = candidate;
  appendLog(`Найдена активная задача ${candidate.job_id}, возобновляю polling.`);
  setBusy(true);
  await pollJob(candidate.job_id);
}

elements.validateButton.addEventListener("click", async () => {
  try {
    await handleValidate();
  } catch (error) {
    setStatus("error", normalizeError(error));
    appendLog(normalizeError(error), "error");
  }
});

elements.trainButton.addEventListener("click", async () => {
  try {
    await handleTrain();
  } catch (error) {
    setBusy(false);
    setStatus("error", normalizeError(error));
    appendLog(normalizeError(error), "error");
  }
});

elements.monitorButton.addEventListener("click", async () => {
  try {
    await handleMonitoring();
  } catch (error) {
    setBusy(false);
    setStatus("error", normalizeError(error));
    appendLog(normalizeError(error), "error");
  }
});

elements.backendInput.addEventListener("change", () => {
  syncTrainingControls();
});

async function initializePage() {
  const pageContext = getPageContext();
  elements.projectIdInput.value = pageContext.projectId;
  syncTrainingControls();
  syncProjectNavigation(pageContext.projectId);
  setBusy(false);
  setJobStatus("idle", "Очередь пуста.");
  setStatus("idle", "Готово.");
  clearDatasetSummary("Загрузка выбранного датасета...");

  appendLog("Загружаю датасет для страницы обучения.");
  try {
    const summary = await resolveDatasetSummary(pageContext);
    applyDatasetSummary(summary);
    setJobStatus("idle", `Проект ${summary.project_id} готов к запуску задач.`);
    setStatus("success", `Подключен датасет ${summary.dataset_version.version_id}: ${summary.rows} строк, target=${summary.target}.`);
    appendLog(`Датасет ${summary.dataset_version.version_id} подключен к странице обучения.`);
    await restoreActiveJob(summary.project_id);
  } catch (error) {
    clearDatasetSummary("Сначала загрузите датасет на странице «Данные», затем вернитесь к обучению.");
    setJobStatus("failed", "Не удалось загрузить контекст страницы обучения.");
    setStatus("error", normalizeError(error));
    appendLog(normalizeError(error), "error");
    for (const button of actionButtons()) {
      button.disabled = true;
    }
  }

  appendLog("Страница обучения готова.");
}

window.addEventListener("beforeunload", () => {
  clearPolling();
});

initializePage();
