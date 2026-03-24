const state = {
  validation: null,
  training: null,
  retraining: null,
  datasetSummary: null,
  busy: false,
};

const elements = {
  datasetVersionInput: document.getElementById("dataset-version-id"),
  datasetSelectionNote: document.getElementById("dataset-selection-note"),
  projectIdInput: document.getElementById("project-id"),
  uploadLinks: Array.from(document.querySelectorAll("[data-nav-upload]")),
  trainingLinks: Array.from(document.querySelectorAll("[data-nav-training]")),
  modelsLinks: Array.from(document.querySelectorAll("[data-nav-models]")),
  graphLinks: Array.from(document.querySelectorAll("[data-nav-graph]")),
  targetInput: document.getElementById("target-column"),
  taskTypeInput: document.getElementById("task-type"),
  backendInput: document.getElementById("training-backend"),
  presetInput: document.getElementById("training-preset"),
  cvFoldsInput: document.getElementById("cv-folds"),
  timeoutSecondsInput: document.getElementById("timeout-seconds"),
  cpuLimitInput: document.getElementById("cpu-limit"),
  testSizeInput: document.getElementById("test-size"),
  retrainHistoryScopeInput: document.getElementById("retrain-history-scope"),
  retrainImprovementThresholdInput: document.getElementById("retrain-improvement-threshold"),
  retrainAutoActivateInput: document.getElementById("retrain-auto-activate"),
  algoInputs: Array.from(document.querySelectorAll('input[name="algo-option"]')),
  trainingOptionsNote: document.getElementById("training-options-note"),
  validateButton: document.getElementById("validate-button"),
  trainButton: document.getElementById("train-button"),
  retrainButton: document.getElementById("retrain-button"),
  statusBanner: document.getElementById("status-banner"),
  trainingLog: document.getElementById("training-log"),
};

function getPageContext() {
  const searchParams = new URLSearchParams(window.location.search);
  return {
    projectId: searchParams.get("project_id")?.trim() || "demo",
    datasetVersionId: searchParams.get("dataset_version_id")?.trim() || "",
  };
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
    retrainingOptions: {
      history_scope: elements.retrainHistoryScopeInput.value,
      minimum_relative_improvement:
        Number.parseFloat(elements.retrainImprovementThresholdInput.value || "3") / 100,
      auto_activate: elements.retrainAutoActivateInput.checked,
    },
  };
}

function setBusy(isBusy) {
  state.busy = isBusy;
  for (const button of [elements.validateButton, elements.trainButton, elements.retrainButton]) {
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

function appendLog(message, level = "info") {
  if (!elements.trainingLog) {
    return;
  }

  const entry = document.createElement("div");
  entry.className = "training-log-entry";
  const time = document.createElement("strong");
  time.textContent = timestampLabel();
  const text = document.createElement("span");
  text.textContent = message;
  entry.append(time, text);
  elements.trainingLog.prepend(entry);

  while (elements.trainingLog.children.length > 12) {
    elements.trainingLog.removeChild(elements.trainingLog.lastChild);
  }

  if (level === "error") {
    console.error(`[training] ${message}`);
  } else {
    console.info(`[training] ${message}`);
  }
}

function normalizeError(error) {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error ?? "Неизвестная ошибка.");
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
  if (!Number.isFinite(workflow.trainingOptions.test_size)) {
    throw new Error("Размер holdout должен быть числом.");
  }
  if (workflow.trainingOptions.test_size <= 0 || workflow.trainingOptions.test_size >= 1) {
    throw new Error("Размер holdout должен быть больше 0 и меньше 1.");
  }
  if (!Number.isFinite(workflow.trainingOptions.cv_folds) || workflow.trainingOptions.cv_folds < 0) {
    throw new Error("Количество фолдов CV должно быть 0 для авто или положительным числом.");
  }
  if (workflow.trainingOptions.cv_folds === 1) {
    throw new Error("Количество фолдов CV должно быть 0 для авто или не меньше 2.");
  }
  if (workflow.trainingOptions.backend !== "sklearn" && workflow.trainingOptions.algos.length === 0) {
    throw new Error("Выберите хотя бы один алгоритм LightAutoML.");
  }
  return workflow;
}

function requireRetrainingWorkflow() {
  const workflow = requireTrainingWorkflow();
  if (!Number.isFinite(workflow.retrainingOptions.minimum_relative_improvement)) {
    throw new Error("Минимальная выгода переобучения должна быть числом.");
  }
  if (workflow.retrainingOptions.minimum_relative_improvement < 0) {
    throw new Error("Минимальная выгода переобучения должна быть больше или равна 0.");
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
  const response = await fetch(path, {
    headers: {
      Accept: "application/json",
    },
  });
  const payload = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, response));
  }
  return payload;
}

async function postDatasetVersion(
  path,
  { projectId, datasetVersionId, trainingOptions = null, retrainingOptions = null },
) {
  appendLog(`Отправляю запрос ${path} для проекта "${projectId}" и датасета "${datasetVersionId}".`);
  const formData = new FormData();
  formData.append("project_id", projectId);
  formData.append("dataset_version_id", datasetVersionId);
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
    appendLog(`Запрос ${path} завершился ошибкой: ${extractErrorMessage(payload, response)}`, "error");
    throw new Error(extractErrorMessage(payload, response));
  }
  appendLog(`Запрос ${path} выполнен успешно.`);
  return payload;
}

function trainingStatusMessage(training) {
  const effectiveOptions = training.training_options?.effective ?? {};
  const presetSuffix = effectiveOptions.preset ? `/${effectiveOptions.preset}` : "";
  let message =
    `Обучение завершено: backend=${training.backend}${presetSuffix}, ` +
    `модель=${training.model_version.version_id}.`;
  if (training.warnings?.length) {
    message += ` Предупреждение: ${training.warnings[0]}`;
  }
  return message;
}

function retrainingStatusMessage(retraining) {
  const profit = retraining.evaluation?.profit ?? null;
  let message =
    `Переобучение завершено: кандидат=${retraining.candidate_model_version.version_id}, ` +
    `окно=${retraining.selection_summary.history_scope}.`;
  if (profit) {
    message +=
      ` ${profit.primary_metric}: ${formatMetricValue(profit.current_value)} -> ` +
      `${formatMetricValue(profit.candidate_value)} `;
    message += `(${formatMetricValue(profit.relative_gain_percent)}% прироста).`;
  }
  if (retraining.activated) {
    message += " Кандидат активирован.";
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
      "Пресет, алгоритмы, CV и таймаут применяются только к LightAutoML.";
  } else {
    elements.trainingOptionsNote.textContent =
      "Для sklearn используется локальный пайплайн на random forest. Пресет, алгоритмы, CV и таймаут игнорируются.";
  }
}

function preferredForecastSteps() {
  return Math.min(24, Math.max(8, Math.round(window.innerWidth / 120)));
}

function syncProjectNavigation(projectId) {
  const normalizedProjectId = projectId.trim();
  const datasetVersionId = elements.datasetVersionInput.value.trim();

  for (const link of elements.uploadLinks) {
    const targetUrl = new URL("./upload.html", window.location.href);
    if (normalizedProjectId) {
      targetUrl.searchParams.set("project_id", normalizedProjectId);
    }
    link.href = targetUrl.toString();
  }

  for (const link of elements.trainingLinks) {
    const targetUrl = new URL("./training.html", window.location.href);
    if (normalizedProjectId) {
      targetUrl.searchParams.set("project_id", normalizedProjectId);
    }
    if (datasetVersionId) {
      targetUrl.searchParams.set("dataset_version_id", datasetVersionId);
    }
    link.href = targetUrl.toString();
  }

  for (const link of elements.modelsLinks) {
    const targetUrl = new URL("./models.html", window.location.href);
    if (normalizedProjectId) {
      targetUrl.searchParams.set("project_id", normalizedProjectId);
    }
    link.href = targetUrl.toString();
  }

  for (const link of elements.graphLinks) {
    const targetUrl = new URL("./graph.html", window.location.href);
    if (normalizedProjectId) {
      targetUrl.searchParams.set("project_id", normalizedProjectId);
    }
    link.href = targetUrl.toString();
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
    `Подключён датасет ${summary.dataset_version.version_id} (${summary.source_name}, ${summary.rows} строк, ${summary.columns.length} колонок).`;
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

async function runAction(actionName, action) {
  try {
    setBusy(true);
    setStatus("busy", actionName);
    appendLog(actionName);
    await action();
  } catch (error) {
    appendLog(normalizeError(error), "error");
    setStatus("error", normalizeError(error));
    setBusy(false);
  }
}

async function handleValidate() {
  const workflow = requireDatasetWorkflow();
  state.validation = state.datasetSummary;
  setStatus("success", `Датасет готов: ${workflow.target}, ${state.validation.rows} строк.`);
}

async function handleTrain() {
  const workflow = requireTrainingWorkflow();
  appendLog(`Запускаю обучение для проекта "${workflow.projectId}" по датасету "${workflow.datasetVersionId}".`);
  state.training = await postDatasetVersion("/training/run/dataset", {
    ...workflow,
    trainingOptions: workflow.trainingOptions,
  });
  state.retraining = null;
  setStatus("success", trainingStatusMessage(state.training));
  redirectToGraph(
    workflow.projectId,
    state.training.model_version.version_id,
    state.training.forecasting,
  );
}

async function handleRetrain() {
  const workflow = requireRetrainingWorkflow();
  appendLog(`Запускаю переобучение для проекта "${workflow.projectId}" по датасету "${workflow.datasetVersionId}".`);
  state.retraining = await postDatasetVersion("/retraining/run/dataset", {
    ...workflow,
    trainingOptions: workflow.trainingOptions,
    retrainingOptions: workflow.retrainingOptions,
  });
  state.training = null;
  setStatus("success", retrainingStatusMessage(state.retraining));
  redirectToGraph(
    workflow.projectId,
    state.retraining.candidate_model_version.version_id,
    state.retraining.forecasting,
  );
}

async function resolveDatasetSummary(pageContext) {
  if (pageContext.datasetVersionId) {
    return fetchJson(`/datasets/${encodeURIComponent(pageContext.datasetVersionId)}`);
  }
  return fetchJson(`/projects/${encodeURIComponent(pageContext.projectId)}/datasets/latest`);
}

elements.validateButton.addEventListener("click", () => {
  runAction("Проверка датасета...", handleValidate);
});

elements.trainButton.addEventListener("click", () => {
  runAction("Обучение основной модели...", handleTrain);
});

elements.retrainButton.addEventListener("click", () => {
  runAction("Переобучение кандидата...", handleRetrain);
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
  setStatus("idle", "Готово.");
  clearDatasetSummary("Загрузка выбранного датасета...");

  appendLog("Загружаю датасет для страницы обучения.");
  try {
    const summary = await resolveDatasetSummary(pageContext);
    applyDatasetSummary(summary);
    setStatus(
      "success",
      `Подключён датасет ${summary.dataset_version.version_id}: ${summary.rows} строк, target=${summary.target}.`,
    );
    appendLog(`Датасет ${summary.dataset_version.version_id} подключён к странице обучения.`);
  } catch (error) {
    clearDatasetSummary("Сначала загрузите датасет на странице «Данные», затем вернитесь к обучению.");
    setStatus("error", normalizeError(error));
    appendLog(normalizeError(error), "error");
    for (const button of [elements.validateButton, elements.trainButton, elements.retrainButton]) {
      button.disabled = true;
    }
  }

  appendLog("Страница обучения готова.");
}

initializePage();
