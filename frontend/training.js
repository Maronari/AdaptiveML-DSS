const state = {
  validation: null,
  training: null,
  retraining: null,
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
  retrainHistoryScopeInput: document.getElementById("retrain-history-scope"),
  retrainImprovementThresholdInput: document.getElementById("retrain-improvement-threshold"),
  retrainAutoActivateInput: document.getElementById("retrain-auto-activate"),
  algoInputs: Array.from(document.querySelectorAll('input[name="algo-option"]')),
  trainingOptionsNote: document.getElementById("training-options-note"),
  validateButton: document.getElementById("validate-button"),
  trainButton: document.getElementById("train-button"),
  retrainButton: document.getElementById("retrain-button"),
  statusBanner: document.getElementById("status-banner"),
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
    button.disabled = isBusy;
  }
}

function setStatus(kind, message) {
  elements.statusBanner.className = `status-banner status-${kind}`;
  elements.statusBanner.textContent = message;
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

function requireFileAndTarget() {
  const workflow = getWorkflowState();
  if (!workflow.file) {
    throw new Error("Выберите файл CSV или XLSX.");
  }
  if (!workflow.target) {
    throw new Error("Укажите имя целевой колонки.");
  }
  return workflow;
}

function requireTrainingWorkflow() {
  const workflow = requireFileAndTarget();
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

function redirectToGraph(projectId, versionId, forecasting = null) {
  const targetUrl = new URL("./index.html", window.location.href);
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
  setStatus("success", `Датасет проверен: ${state.validation.rows} строк, тип=${state.validation.task_type}.`);
}

async function handleTrain() {
  const workflow = requireTrainingWorkflow();
  state.training = await postFile("/training/run/file", {
    ...workflow,
    includeTarget: true,
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
  state.retraining = await postFile("/retraining/run/file", {
    ...workflow,
    includeTarget: true,
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

syncTrainingControls();
