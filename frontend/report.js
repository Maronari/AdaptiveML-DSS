const state = {
  projects: [],
  projectId: "",
  projectName: "",
  items: [],
  selectedVersionId: "",
  selectedItem: null,
  modelDetail: null,
  error: null,
  loading: false,
};

const ALGO_LABELS = {
  lgb: "LightGBM (градиентный бустинг)",
  cb: "CatBoost (градиентный бустинг)",
  xgb: "XGBoost (градиентный бустинг)",
  rf: "Random Forest",
  linear_l2: "линейная модель (L2-регуляризация)",
  nn: "нейронная сеть",
};

const BACKEND_LABELS = {
  lightautoml: "LightAutoML (LAMA)",
  "sklearn-fallback": "scikit-learn (резервный режим)",
};

const PRESET_LABELS = {
  tabular: "TabularAutoML",
  utilized: "TabularUtilizedAutoML",
  "sklearn-random-forest": "RandomForest pipeline",
};

const METRIC_LABELS = {
  r: "Коэффициент корреляции (r)",
  r2: "Коэффициент детерминации (R²)",
  mse: "MSE — среднеквадратичная ошибка",
  rmse: "RMSE — корень из среднеквадратичной ошибки",
  mae: "MAE — средняя абсолютная ошибка прогноза",
  aic: "Информационный критерий Акаике (AIC)",
  bic: "Байесовский информационный критерий (BIC)",
  aicc: "Скорректированный AIC (AICc)",
  accuracy: "Accuracy — доля верных предсказаний",
  f1_weighted: "F1-score (взвешенный по классам)",
};

const HYPERPARAMETER_LABELS = {
  task_type: "Тип задачи",
  backend: "AutoML-бэкенд",
  preset: "Пресет",
  algos: "Алгоритмы",
  timeout_seconds: "Лимит времени обучения, с",
  cpu_limit: "Лимит CPU",
  test_size: "Доля holdout-выборки",
  cv_folds: "Число фолдов кросс-валидации",
  enable_forecast: "Прогнозирующая надстройка",
};

const STATUS_LABELS = {
  champion: "Champion",
  latest: "Latest",
  candidate: "Candidate",
  archived: "Archived",
};

const LOWER_IS_BETTER_METRICS = new Set(["rmse"]);

const elements = {
  projectTitle: document.getElementById("report-project-title"),
  projectNote: document.getElementById("report-project-note"),
  projectSelect: document.getElementById("report-project-select"),
  versionSelect: document.getElementById("report-version-select"),
  modelsLink: document.getElementById("report-models-link"),
  summaryVersion: document.getElementById("report-summary-version"),
  summaryStatus: document.getElementById("report-summary-status"),
  summaryTask: document.getElementById("report-summary-task"),
  summaryCreated: document.getElementById("report-summary-created"),
  descriptionBody: document.getElementById("report-description-body"),
  metricsBody: document.getElementById("report-metrics-body"),
  metricsEmpty: document.getElementById("report-metrics-empty"),
  hyperparametersBody: document.getElementById("report-hyperparameters-body"),
  hyperparametersEmpty: document.getElementById("report-hyperparameters-empty"),
  leaderboardBody: document.getElementById("report-leaderboard-body"),
  leaderboardEmpty: document.getElementById("report-leaderboard-empty"),
  historySummary: document.getElementById("report-history-summary"),
  historyChart: document.getElementById("report-history-chart"),
  historyBody: document.getElementById("report-history-body"),
  historyEmpty: document.getElementById("report-history-empty"),
  uploadLinks: Array.from(document.querySelectorAll("[data-nav-upload]")),
  trainingLinks: Array.from(document.querySelectorAll("[data-nav-training]")),
  retrainingLinks: Array.from(document.querySelectorAll("[data-nav-retraining]")),
  modelsLinks: Array.from(document.querySelectorAll("[data-nav-models]")),
  reportLinks: Array.from(document.querySelectorAll("[data-nav-report]")),
  graphLinks: Array.from(document.querySelectorAll("[data-nav-graph]")),
  dssLinks: Array.from(document.querySelectorAll("[data-nav-dss]")),
};

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

function getPageContext() {
  const searchParams = new URLSearchParams(window.location.search);
  return {
    projectId: searchParams.get("project_id")?.trim() || "",
    versionId: searchParams.get("model_version")?.trim() || "",
  };
}

function formatDateTime(dateString) {
  if (!dateString) {
    return "—";
  }

  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) {
    return "—";
  }

  return date.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) {
    return null;
  }

  return JSON.parse(text);
}

function extractErrorMessage(payload, response) {
  if (payload && typeof payload === "object" && "detail" in payload) {
    return payload.detail;
  }
  return response.statusText || "Не удалось загрузить отчёт по модели.";
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

function buildModelDisplayName(item, detail) {
  const artifacts = detail?.training_artifacts || {};
  const backend = artifacts.backend || "unknown";
  const preset = artifacts.preset || "";
  const composition = artifacts.model_composition || {};
  const fittedAlgos =
    composition.source === "fitted" && Array.isArray(composition.algos) && composition.algos.length
      ? composition.algos
      : null;
  const algos = fittedAlgos || artifacts.training_options?.effective?.algos || [];
  const isFitted = Boolean(fittedAlgos);

  const backendLabel = BACKEND_LABELS[backend] || backend;
  const presetLabel = PRESET_LABELS[preset] || preset;
  const algoLabels = algos.map((algo) => ALGO_LABELS[algo] || algo);

  if (algoLabels.length && backend === "lightautoml") {
    const algoText = isFitted
      ? `фактический состав ансамбля: ${algoLabels.join(", ")}`
      : algoLabels.join(", ");
    return `${backendLabel} / ${presetLabel}: ${algoText}`;
  }
  return presetLabel ? `${backendLabel} / ${presetLabel}` : backendLabel;
}

function renderDescription(item, detail, modelName) {
  const artifacts = detail?.training_artifacts || {};
  const backend = artifacts.backend || "unknown";
  const taskType = detail?.task_type || item?.task_type || "";
  const target = detail?.target || item?.target || "";
  const featureCount = (detail?.feature_names || item?.feature_names || []).length;

  const libraryNote =
    backend === "lightautoml"
      ? "Модель обучена библиотекой автоматизированного машинного обучения LightAutoML " +
        "(LAMA, Sber AI Lab): библиотека сама подбирает и комбинирует перечисленные ниже " +
        "алгоритмы по результатам кросс-валидации и строит итоговый ансамбль (blending)."
      : "LightAutoML была недоступна либо тренировка на ней завершилась ошибкой, поэтому " +
        "использован резервный конвейер на scikit-learn.";

  const paragraphs = [
    item?.name
      ? `<p><strong>Название:</strong> ${escapeHtml(item.name)}</p>`
      : "",
    `<p><strong>Алгоритм:</strong> ${escapeHtml(modelName)}</p>`,
    `<p><strong>Версия:</strong> ${escapeHtml(item?.version_id || "—")} ` +
      `(статус: ${escapeHtml(item?.status || "—")})</p>`,
    `<p><strong>Задача:</strong> ${escapeHtml(taskType)}, целевая переменная — ` +
      `<code>${escapeHtml(target)}</code>, признаков использовано: ${escapeHtml(String(featureCount))}.</p>`,
    `<p><strong>Обучающие данные:</strong> версия датасета ` +
      `${escapeHtml(item?.dataset_version_id || "—")} ` +
      `(${escapeHtml(item?.dataset_source_name || "источник не указан")}), ` +
      `строк: ${escapeHtml(String(item?.dataset_rows ?? "—"))}.</p>`,
    `<p>${libraryNote}</p>`,
  ];
  return paragraphs.join("");
}

function renderMetricsTable(metrics, primaryMetric) {
  const entries = Object.entries(metrics || {});
  elements.metricsEmpty.hidden = entries.length > 0;
  elements.metricsBody.replaceChildren();

  const fragment = document.createDocumentFragment();
  entries.forEach(([key, value]) => {
    const row = document.createElement("tr");
    if (key === primaryMetric) {
      row.className = "primary-row";
    }
    row.innerHTML = `
      <td>${escapeHtml(METRIC_LABELS[key] || key)}</td>
      <td><code>${escapeHtml(key)}</code></td>
      <td>${escapeHtml(String(value))}</td>
    `;
    fragment.append(row);
  });
  elements.metricsBody.append(fragment);
}

function renderHyperparametersTable(detail) {
  const effective = detail?.training_artifacts?.training_options?.effective || {};
  const entries = Object.entries(effective);
  elements.hyperparametersEmpty.hidden = entries.length > 0;
  elements.hyperparametersBody.replaceChildren();

  const fragment = document.createDocumentFragment();
  entries.forEach(([key, value]) => {
    const row = document.createElement("tr");
    let displayValue;
    if (Array.isArray(value)) {
      displayValue = value.length ? value.join(", ") : "—";
    } else if (value === null || value === undefined) {
      displayValue = "—";
    } else {
      displayValue = String(value);
    }
    row.innerHTML = `
      <td>${escapeHtml(HYPERPARAMETER_LABELS[key] || key)}</td>
      <td>${escapeHtml(displayValue)}</td>
    `;
    fragment.append(row);
  });
  elements.hyperparametersBody.append(fragment);
}

function formatWeightPercent(weight) {
  const numericValue = Number(weight);
  if (!Number.isFinite(numericValue)) {
    return "—";
  }
  return `${(numericValue * 100).toLocaleString("ru-RU", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}%`;
}

function renderLeaderboardTable(detail) {
  const leaderboard = detail?.training_artifacts?.model_leaderboard;
  const entries = Array.isArray(leaderboard) ? leaderboard : [];
  elements.leaderboardEmpty.hidden = entries.length > 0;
  elements.leaderboardBody.replaceChildren();

  const fragment = document.createDocumentFragment();
  entries.forEach((entry, index) => {
    const row = document.createElement("tr");
    if (index === 0) {
      row.className = "primary-row";
    }
    const algoLabel = ALGO_LABELS[entry.algo] || entry.algo || entry.model_name || "—";
    const winnerBadge = index === 0 ? ' <span class="report-status-pill champion">победитель</span>' : "";
    row.innerHTML = `
      <td>${escapeHtml(algoLabel)}${winnerBadge}</td>
      <td>${escapeHtml(String(entry.folds ?? "—"))}</td>
      <td>${escapeHtml(formatWeightPercent(entry.weight))}</td>
    `;
    fragment.append(row);
  });
  elements.leaderboardBody.append(fragment);
}

function formatMetricNumber(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return "—";
  }
  return numericValue.toLocaleString("ru-RU", { minimumFractionDigits: 0, maximumFractionDigits: 4 });
}

function isImprovement(primaryMetric, delta) {
  if (!Number.isFinite(delta) || delta === 0) {
    return "flat";
  }
  const lowerIsBetter = LOWER_IS_BETTER_METRICS.has(primaryMetric);
  const improved = lowerIsBetter ? delta < 0 : delta > 0;
  return improved ? "up" : "down";
}

function buildHistoryPoints(items) {
  return [...items]
    .filter((item) => Number.isFinite(Number(item.metric_value)))
    .sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
    .map((item) => ({
      versionId: item.version_id,
      createdAt: item.created_at,
      status: item.is_champion ? "champion" : item.is_latest ? "latest" : item.status,
      statusLabel: item.is_champion ? "Champion" : item.is_latest ? "Latest" : STATUS_LABELS[item.status] || item.status,
      value: Number(item.metric_value),
      primaryMetric: item.primary_metric,
    }));
}

function buildHistorySvg(points) {
  if (points.length < 2) {
    return '<p class="field-note">Нужно минимум две версии модели, чтобы построить график динамики.</p>';
  }

  const width = 760;
  const height = 220;
  const paddingX = 20;
  const paddingTop = 26;
  const paddingBottom = 30;
  const values = points.map((point) => point.value);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const valueRange = maxValue - minValue || Math.abs(maxValue) || 1;

  const plotted = points.map((point, index) => {
    const x = paddingX + (index / (points.length - 1)) * (width - paddingX * 2);
    const y =
      height -
      paddingBottom -
      ((point.value - minValue) / valueRange) * (height - paddingTop - paddingBottom);
    return { ...point, x, y };
  });

  const linePoints = plotted.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");

  const circles = plotted
    .map((point) => {
      const radius = point.status === "champion" ? 7 : point.status === "latest" ? 5.5 : 4;
      const fill = point.status === "champion" ? "var(--accent)" : "var(--surface)";
      const stroke = point.status === "champion" ? "var(--accent-strong)" : "var(--muted)";
      return `
        <circle cx="${point.x.toFixed(1)}" cy="${point.y.toFixed(1)}" r="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="2">
          <title>${escapeHtml(point.versionId)} · ${escapeHtml(point.statusLabel)} · ${escapeHtml(formatDateTime(point.createdAt))} · ${escapeHtml(formatMetricNumber(point.value))}</title>
        </circle>
      `;
    })
    .join("");

  const firstLabel = plotted[0];
  const lastLabel = plotted[plotted.length - 1];

  return `
    <svg viewBox="0 0 ${width} ${height}" class="report-history-svg" role="img" aria-label="Динамика метрики по версиям модели проекта">
      <line x1="${paddingX}" y1="${height - paddingBottom}" x2="${width - paddingX}" y2="${height - paddingBottom}" class="report-history-axis" />
      <text x="${firstLabel.x.toFixed(1)}" y="${height - 6}" class="report-history-point-label" text-anchor="start">${escapeHtml(formatDateTime(firstLabel.createdAt))}</text>
      <text x="${lastLabel.x.toFixed(1)}" y="${height - 6}" class="report-history-point-label" text-anchor="end">${escapeHtml(formatDateTime(lastLabel.createdAt))}</text>
      <text x="${paddingX}" y="16" class="report-history-value-label">${escapeHtml(formatMetricNumber(maxValue))}</text>
      <text x="${paddingX}" y="${height - paddingBottom - 6}" class="report-history-value-label">${escapeHtml(formatMetricNumber(minValue))}</text>
      <polyline points="${linePoints}" class="report-history-line" fill="none" />
      ${circles}
    </svg>
  `;
}

function renderHistoryTable(points) {
  elements.historyBody.replaceChildren();
  elements.historyEmpty.hidden = points.length > 1;

  const fragment = document.createDocumentFragment();
  points.forEach((point, index) => {
    const previous = index > 0 ? points[index - 1] : null;
    const delta = previous ? point.value - previous.value : null;
    const direction = previous ? isImprovement(point.primaryMetric, delta) : "flat";
    const deltaLabel =
      previous && Number.isFinite(delta)
        ? `${delta > 0 ? "+" : ""}${formatMetricNumber(delta)}`
        : "—";
    const arrow = direction === "up" ? "↑" : direction === "down" ? "↓" : "→";

    const row = document.createElement("tr");
    row.innerHTML = `
      <td><code>${escapeHtml(point.versionId)}</code></td>
      <td>${escapeHtml(formatDateTime(point.createdAt))}</td>
      <td><span class="report-status-pill ${escapeHtml(point.status)}">${escapeHtml(point.statusLabel)}</span></td>
      <td>${escapeHtml(METRIC_LABELS[point.primaryMetric] || point.primaryMetric)}: ${escapeHtml(formatMetricNumber(point.value))}</td>
      <td><span class="report-delta is-${direction}">${arrow} ${escapeHtml(deltaLabel)}</span></td>
    `;
    fragment.append(row);
  });
  elements.historyBody.append(fragment);
}

function renderHistorySection() {
  const points = buildHistoryPoints(state.items);

  if (!state.projectId) {
    elements.historySummary.textContent = "Данные появятся после выбора проекта.";
    elements.historyChart.innerHTML = "";
    elements.historyBody.replaceChildren();
    elements.historyEmpty.hidden = false;
    return;
  }

  if (points.length < 2) {
    elements.historySummary.textContent =
      "У проекта пока только одна обученная версия модели — динамику показывать не по чему. Обучите ещё одну версию, чтобы увидеть изменение качества.";
    elements.historyChart.innerHTML = "";
    renderHistoryTable(points);
    return;
  }

  const first = points[0];
  const last = points[points.length - 1];
  const totalDelta = last.value - first.value;
  const direction = isImprovement(last.primaryMetric, totalDelta);
  const trendWord = direction === "up" ? "улучшилось" : direction === "down" ? "ухудшилось" : "не изменилось";
  const metricLabel = METRIC_LABELS[last.primaryMetric] || last.primaryMetric;

  elements.historySummary.textContent =
    `Версий с метриками: ${points.length}. С первой версии (${formatDateTime(first.createdAt)}) до последней ` +
    `(${formatDateTime(last.createdAt)}) качество по метрике «${metricLabel}» ${trendWord}: ` +
    `${formatMetricNumber(first.value)} → ${formatMetricNumber(last.value)} ` +
    `(${totalDelta > 0 ? "+" : ""}${formatMetricNumber(totalDelta)}).`;

  elements.historyChart.innerHTML = buildHistorySvg(points);
  renderHistoryTable(points);
}

function renderProjectSelect() {
  elements.projectSelect.replaceChildren();

  if (!state.projects.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Нет доступных проектов";
    elements.projectSelect.append(option);
    elements.projectSelect.disabled = true;
    return;
  }

  elements.projectSelect.disabled = false;
  state.projects.forEach((project) => {
    const option = document.createElement("option");
    option.value = project.project_id;
    option.textContent =
      project.name && project.name !== project.project_id
        ? `${project.name} (${project.project_id})`
        : project.project_id;
    option.selected = project.project_id === state.projectId;
    elements.projectSelect.append(option);
  });
}

function renderVersionSelect() {
  elements.versionSelect.replaceChildren();
  state.items.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.version_id;
    const badge = item.is_champion ? "Champion" : item.is_latest ? "Latest" : STATUS_LABELS[item.status] || item.status;
    const label = item.name ? `${item.name} (${item.version_id})` : item.version_id;
    option.textContent = `${label} · ${badge} · ${formatDateTime(item.created_at)}`;
    option.selected = item.version_id === state.selectedVersionId;
    elements.versionSelect.append(option);
  });
  elements.versionSelect.disabled = state.items.length === 0;
}

function renderSummary() {
  const item = state.selectedItem;
  if (!item) {
    elements.summaryVersion.textContent = "—";
    elements.summaryStatus.textContent = "—";
    elements.summaryTask.textContent = "—";
    elements.summaryCreated.textContent = "—";
    return;
  }

  const statusKey = item.is_champion ? "champion" : item.is_latest ? "latest" : item.status;
  const statusLabel = item.is_champion ? "Champion" : item.is_latest ? "Latest" : STATUS_LABELS[item.status] || item.status;

  elements.summaryVersion.textContent = item.name ? `${item.name} (${item.version_id})` : item.version_id;
  elements.summaryStatus.innerHTML = `<span class="report-status-pill ${escapeHtml(statusKey)}">${escapeHtml(statusLabel)}</span>`;
  elements.summaryTask.textContent = `${item.task_type || "—"} / ${item.target || "—"}`;
  elements.summaryCreated.textContent = formatDateTime(item.created_at);
}

function renderPage() {
  const title = state.projectId
    ? `Отчёт по модели проекта ${state.projectName || state.projectId}`
    : "Отчёт по модели";
  elements.projectTitle.textContent = title;

  const modelsUrl = new URL("./models.html", window.location.href);
  if (state.projectId) {
    modelsUrl.searchParams.set("project_id", state.projectId);
  }
  elements.modelsLink.href = modelsUrl.toString();

  if (state.error) {
    elements.projectNote.textContent = state.error;
  } else if (!state.projects.length) {
    elements.projectNote.textContent = "В реестре пока нет проектов. Создайте проект на стартовой странице.";
  } else if (!state.projectId) {
    elements.projectNote.textContent = "Выберите проект в списке выше.";
  } else if (!state.items.length) {
    elements.projectNote.textContent = "У проекта ещё нет обученных версий модели.";
  } else {
    elements.projectNote.textContent =
      `Версий модели в проекте: ${state.items.length}. Отчёт собран для выбранной версии ниже.`;
  }

  renderProjectSelect();
  renderVersionSelect();
  renderSummary();

  if (state.selectedItem && state.modelDetail) {
    const modelName = buildModelDisplayName(state.selectedItem, state.modelDetail);
    elements.descriptionBody.innerHTML = renderDescription(state.selectedItem, state.modelDetail, modelName);
    renderMetricsTable(state.modelDetail.metrics, state.modelDetail.primary_metric);
    renderHyperparametersTable(state.modelDetail);
    renderLeaderboardTable(state.modelDetail);
  } else {
    elements.descriptionBody.innerHTML =
      '<p class="field-note">Данные появятся после выбора проекта и версии модели.</p>';
    elements.metricsBody.replaceChildren();
    elements.hyperparametersBody.replaceChildren();
    elements.leaderboardBody.replaceChildren();
    elements.metricsEmpty.hidden = false;
    elements.hyperparametersEmpty.hidden = false;
    elements.leaderboardEmpty.hidden = false;
  }

  renderHistorySection();
}

function syncNavigation(projectId, versionId = "") {
  const normalizedProjectId = projectId.trim();
  const linkGroups = [
    [elements.uploadLinks, "./upload.html"],
    [elements.trainingLinks, "./training.html"],
    [elements.retrainingLinks, "./retraining.html"],
    [elements.modelsLinks, "./models.html"],
    [elements.reportLinks, "./report.html"],
    [elements.graphLinks, "./graph.html"],
    [elements.dssLinks, "./dss.html"],
  ];

  for (const [links, path] of linkGroups) {
    for (const link of links) {
      const targetUrl = new URL(path, window.location.href);
      if (normalizedProjectId) {
        targetUrl.searchParams.set("project_id", normalizedProjectId);
      }
      if (versionId && (path === "./graph.html" || path === "./dss.html" || path === "./report.html")) {
        targetUrl.searchParams.set("model_version", versionId);
      }
      link.href = targetUrl.toString();
    }
  }
}

function updateUrl() {
  const currentUrl = new URL(window.location.href);
  if (state.projectId) {
    currentUrl.searchParams.set("project_id", state.projectId);
  }
  if (state.selectedVersionId) {
    currentUrl.searchParams.set("model_version", state.selectedVersionId);
  }
  window.history.replaceState({}, "", currentUrl);
}

async function loadModelDetail(versionId) {
  state.modelDetail = await fetchJson(`/models/${encodeURIComponent(versionId)}`);
}

async function loadProjectList() {
  const payload = await fetchJson("/projects");
  state.projects = Array.isArray(payload.items) ? payload.items : [];
}

function pickDefaultProjectId() {
  return (
    state.projects.find((project) => project.has_champion_model)?.project_id ||
    state.projects.find((project) => project.has_models)?.project_id ||
    state.projects[0]?.project_id ||
    ""
  );
}

async function loadProjectModels(requestedVersionId = "") {
  state.error = null;
  state.projectName = "";
  state.items = [];
  state.selectedVersionId = "";
  state.selectedItem = null;
  state.modelDetail = null;

  if (!state.projectId) {
    return;
  }

  const payload = await fetchJson(`/projects/${encodeURIComponent(state.projectId)}/models`);
  state.projectName = payload.project_name || state.projectId;
  state.items = Array.isArray(payload.items) ? payload.items : [];

  const fallbackVersionId = payload.champion_model_version_id || payload.latest_model_version_id || "";
  const resolvedVersionId =
    state.items.find((item) => item.version_id === requestedVersionId)?.version_id ||
    state.items.find((item) => item.version_id === fallbackVersionId)?.version_id ||
    state.items[0]?.version_id ||
    "";

  state.selectedVersionId = resolvedVersionId;
  state.selectedItem = state.items.find((item) => item.version_id === resolvedVersionId) || null;

  if (resolvedVersionId) {
    await loadModelDetail(resolvedVersionId);
  }
}

async function loadReport() {
  const context = getPageContext();
  state.error = null;

  try {
    await loadProjectList();
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
    state.projects = [];
  }

  state.projectId =
    state.projects.find((project) => project.project_id === context.projectId)?.project_id ||
    context.projectId ||
    pickDefaultProjectId();

  try {
    await loadProjectModels(context.versionId);
    syncNavigation(state.projectId, state.selectedVersionId);
    updateUrl();
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  }

  renderPage();
}

async function onProjectChange(projectId) {
  if (!projectId || projectId === state.projectId) {
    return;
  }

  state.projectId = projectId;
  state.error = null;

  try {
    await loadProjectModels();
    syncNavigation(state.projectId, state.selectedVersionId);
    updateUrl();
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  }

  renderPage();
}

async function onVersionChange(versionId) {
  if (!versionId || versionId === state.selectedVersionId) {
    return;
  }

  state.selectedVersionId = versionId;
  state.selectedItem = state.items.find((item) => item.version_id === versionId) || null;
  state.error = null;
  state.loading = true;

  try {
    await loadModelDetail(versionId);
    syncNavigation(state.projectId, state.selectedVersionId);
    updateUrl();
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
    state.modelDetail = null;
  } finally {
    state.loading = false;
  }

  renderPage();
}

elements.projectSelect.addEventListener("change", (event) => {
  onProjectChange(event.target.value);
});

elements.versionSelect.addEventListener("change", (event) => {
  onVersionChange(event.target.value);
});

renderPage();
loadReport();
