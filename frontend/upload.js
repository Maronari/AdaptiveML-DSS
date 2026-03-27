const DEFAULT_PROJECT_ID = "demo";
const EMPTY_VALUE = "—";
const DEFAULT_PREVIEW_MESSAGE = "Загрузите CSV или Excel, и страница сразу выполнит разбор файла.";
const LOADING_PREVIEW_MESSAGE = "Загрузка..";

const state = {
  inspection: null,
  busy: false,
  projectId: DEFAULT_PROJECT_ID,
};

const elements = {
  fileInput: document.getElementById("dataset-file"),
  targetSelect: document.getElementById("target-column"),
  continueButton: document.getElementById("continue-button"),
  summaryRows: document.getElementById("summary-rows"),
  summaryColumns: document.getElementById("summary-columns"),
  summaryDuplicates: document.getElementById("summary-duplicates"),
  summarySource: document.getElementById("summary-source"),
  columnTypesBody: document.getElementById("column-types-body"),
  temporalContextBody: document.getElementById("temporal-context-body"),
  targetSummaryBody: document.getElementById("target-summary-body"),
  projectContextBody: document.getElementById("project-context-body"),
  previewHead: document.getElementById("upload-preview-head"),
  previewBody: document.getElementById("upload-preview-body"),
  previewEmpty: document.getElementById("upload-preview-empty"),
  statusBanner: document.getElementById("status-banner"),
  uploadLinks: Array.from(document.querySelectorAll("[data-nav-upload]")),
  trainingLinks: Array.from(document.querySelectorAll("[data-nav-training]")),
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

function setBusy(isBusy) {
  state.busy = isBusy;
  elements.continueButton.disabled = isBusy || !state.inspection || !elements.targetSelect.value.trim();
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
    if (response.status === 404) {
      throw new Error(
        `Маршрут ${path} недоступен на сервере. Перезапустите backend, чтобы подхватить новые API загрузки данных.`,
      );
    }
    throw new Error(extractErrorMessage(payload, response));
  }
  return payload;
}

function syncNavigation(projectId) {
  const normalizedProjectId = projectId.trim();

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

  for (const link of elements.dssLinks) {
    const targetUrl = new URL("./dss.html", window.location.href);
    if (normalizedProjectId) {
      targetUrl.searchParams.set("project_id", normalizedProjectId);
    }
    link.href = targetUrl.toString();
  }
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

function formatPercent(share) {
  if (typeof share !== "number" || Number.isNaN(share)) {
    return EMPTY_VALUE;
  }
  return `${formatNumber(share * 100)}%`;
}

function formatTimestamp(value) {
  if (!value) {
    return EMPTY_VALUE;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
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
  return EMPTY_VALUE;
}

function formatColumnKind(kind) {
  const labels = {
    numeric: "Числовая",
    categorical: "Категориальная",
    datetime: "Дата и время",
    time: "Время",
    boolean: "Логическая",
    text: "Текстовая",
    empty: "Пустая",
  };
  return labels[kind] || kind || EMPTY_VALUE;
}

function formatDuration(totalMinutes) {
  if (!Number.isFinite(totalMinutes) || totalMinutes <= 0) {
    return null;
  }
  const roundedMinutes = Math.round(totalMinutes * 1000) / 1000;
  if (Number.isInteger(roundedMinutes / 1440)) {
    return `${formatNumber(roundedMinutes / 1440)} дн`;
  }
  if (Number.isInteger(roundedMinutes / 60)) {
    return `${formatNumber(roundedMinutes / 60)} ч`;
  }
  if (roundedMinutes >= 1) {
    return `${formatNumber(roundedMinutes)} мин`;
  }
  return `${formatNumber(roundedMinutes * 60)} сек`;
}

function parseDateValue(value) {
  if (value == null) {
    return null;
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function inferColumnKindFromDtype(dtype) {
  const normalized = String(dtype ?? "").toLowerCase();
  if (normalized.includes("datetime")) {
    return "datetime";
  }
  if (normalized.includes("bool")) {
    return "boolean";
  }
  if (
    normalized.includes("int")
    || normalized.includes("float")
    || normalized.includes("double")
    || normalized.includes("number")
  ) {
    return "numeric";
  }
  if (normalized.includes("date") || normalized.includes("time")) {
    return "datetime";
  }
  return "text";
}

function inferTaskTypeFromValues(values) {
  if (!values.length) {
    return null;
  }

  const distinctValues = new Set(values.map((value) => String(value))).size;
  const numericValues = values
    .map((value) => (typeof value === "number" ? value : Number(value)))
    .filter((value) => Number.isFinite(value));

  if (numericValues.length === values.length) {
    const relativeCardinality = distinctValues / Math.max(values.length, 1);
    if (distinctValues === 2) {
      return "binary";
    }
    if (distinctValues <= 10 && relativeCardinality < 0.2) {
      return "multiclass";
    }
    return "regression";
  }

  return distinctValues === 2 ? "binary" : "multiclass";
}

function getColumnProfiles(inspection) {
  if (Array.isArray(inspection.column_profiles) && inspection.column_profiles.length) {
    return inspection.column_profiles;
  }

  const columns = inspection.columns ?? [];
  const schema = inspection.schema ?? {};
  const missingValues = inspection.missing_values ?? {};
  const targetCandidates = Array.isArray(inspection.target_candidates) ? inspection.target_candidates : [];
  const candidateByColumn = new Map(targetCandidates.map((item) => [item.column, item]));

  return columns.map((column) => {
    const candidate = candidateByColumn.get(column);
    return {
      column,
      dtype: schema[column] ?? "object",
      kind: inferColumnKindFromDtype(schema[column]),
      missing_values: Number(missingValues[column] ?? 0),
      non_null_values: Number(inspection.rows ?? 0) - Number(missingValues[column] ?? 0),
      distinct_values: candidate?.distinct_values ?? 0,
      distinct_ratio: 0,
      example_values: [],
      suggested_task_type: candidate?.suggested_task_type ?? null,
      target_eligible: Boolean(candidate),
      target_score: candidate?.score ?? 0,
      target_reason: candidate?.reason ?? "",
      looks_like_identifier: false,
      is_recommended_target: inspection.recommended_target === column,
    };
  });
}

function buildColumnTypeSummaryFallback(inspection) {
  const profiles = getColumnProfiles(inspection);
  const labels = {
    numeric: "Числовые",
    categorical: "Категориальные",
    datetime: "Дата и время",
    time: "Время",
    boolean: "Логические",
    text: "Текстовые",
    empty: "Пустые",
  };
  const counts = new Map();
  profiles.forEach((profile) => {
    counts.set(profile.kind, (counts.get(profile.kind) ?? 0) + 1);
  });

  return {
    total_columns: inspection.columns?.length ?? profiles.length,
    items: Array.from(counts.entries()).map(([kind, count]) => ({
      kind,
      label: labels[kind] ?? kind,
      count,
    })),
  };
}

function buildTemporalContextFallback(inspection) {
  const profiles = getColumnProfiles(inspection);
  const temporalProfile = profiles.find((profile) => profile.kind === "datetime");
  if (!temporalProfile) {
    return {
      available: false,
      column: null,
      message: "Временная колонка не найдена.",
    };
  }

  const column = temporalProfile.column;
  const parsed = (inspection.sample_rows ?? [])
    .map((row) => parseDateValue(row?.[column]))
    .filter((value) => value !== null)
    .sort((left, right) => left.getTime() - right.getTime());

  if (!parsed.length) {
    return {
      available: false,
      column,
      message: "Не удалось распознать временную шкалу.",
    };
  }

  const uniqueMs = [...new Set(parsed.map((value) => value.getTime()))].sort((left, right) => left - right);
  let frequencyMinutes = null;
  let hasGaps = false;
  if (uniqueMs.length >= 2) {
    const deltas = [];
    for (let index = 1; index < uniqueMs.length; index += 1) {
      const deltaMinutes = (uniqueMs[index] - uniqueMs[index - 1]) / 60000;
      if (deltaMinutes > 0) {
        deltas.push(deltaMinutes);
      }
    }
    if (deltas.length) {
      const frequencyMap = new Map();
      deltas.forEach((delta) => {
        const key = delta.toFixed(6);
        frequencyMap.set(key, (frequencyMap.get(key) ?? 0) + 1);
      });
      const [modeKey] = [...frequencyMap.entries()].sort((left, right) => right[1] - left[1])[0];
      frequencyMinutes = Number(modeKey);
      hasGaps = deltas.some((delta) => delta > frequencyMinutes * 1.5);
    }
  }

  const first = new Date(uniqueMs[0]);
  const last = new Date(uniqueMs[uniqueMs.length - 1]);

  return {
    available: true,
    column,
    rows_with_timestamp: parsed.length,
    unique_timestamps: uniqueMs.length,
    start: first.toISOString(),
    end: last.toISOString(),
    range_label: formatDuration((last.getTime() - first.getTime()) / 60000) || "Одна временная точка",
    frequency_minutes: frequencyMinutes,
    frequency_label: formatDuration(frequencyMinutes),
    has_gaps: hasGaps,
    message: null,
    based_on_preview: true,
  };
}

function buildTargetSummariesFallback(inspection) {
  const profiles = getColumnProfiles(inspection);
  const sampleRows = inspection.sample_rows ?? [];
  const summaries = {};

  profiles.forEach((profile) => {
    const values = sampleRows
      .map((row) => row?.[profile.column])
      .filter((value) => value != null && String(value).trim() !== "");

    const taskType = profile.suggested_task_type ?? inferTaskTypeFromValues(values);
    const summary = {
      column: profile.column,
      kind: profile.kind,
      task_type: taskType,
      missing_values: Number(profile.missing_values ?? 0),
      non_null_values: Number(profile.non_null_values ?? values.length),
      distinct_values: Number(profile.distinct_values ?? new Set(values.map(String)).size),
      target_eligible: Boolean(profile.target_eligible),
      target_reason: profile.target_reason ?? "",
      looks_like_identifier: Boolean(profile.looks_like_identifier),
      example_values: Array.isArray(profile.example_values) ? profile.example_values : values.slice(0, 4).map(String),
      summary_basis: "preview",
    };

    const numericValues = values
      .map((value) => (typeof value === "number" ? value : Number(value)))
      .filter((value) => Number.isFinite(value));

    if (taskType === "regression" && numericValues.length) {
      const sorted = [...numericValues].sort((left, right) => left - right);
      const middleIndex = Math.floor(sorted.length / 2);
      summary.numeric_stats = {
        min: Math.min(...numericValues),
        max: Math.max(...numericValues),
        mean: numericValues.reduce((sum, value) => sum + value, 0) / numericValues.length,
        median:
          sorted.length % 2 === 0
            ? (sorted[middleIndex - 1] + sorted[middleIndex]) / 2
            : sorted[middleIndex],
      };
    } else if (values.length) {
      const counts = new Map();
      values.forEach((value) => {
        const key = String(value);
        counts.set(key, (counts.get(key) ?? 0) + 1);
      });
      const ordered = [...counts.entries()].sort((left, right) => right[1] - left[1]);
      summary.class_count = counts.size;
      summary.top_values = ordered.slice(0, 3).map(([label, count]) => ({
        label,
        count,
        share: count / values.length,
      }));
      summary.majority_share = ordered.length ? ordered[0][1] / values.length : null;
    }

    summaries[profile.column] = summary;
  });

  return summaries;
}

async function buildProjectContextFallback(projectId) {
  const context = {
    project_id: projectId,
    project_name: projectId,
    dataset_versions: 0,
    model_versions: 0,
    latest_dataset_version_id: null,
    latest_dataset_source_name: null,
    latest_dataset_created_at: null,
    latest_model_version_id: null,
    latest_model_created_at: null,
    latest_model_target: null,
    latest_model_task_type: null,
    champion_model_version_id: null,
    has_datasets: false,
    has_models: false,
    has_champion_model: false,
    is_new_project: true,
  };

  try {
    const payload = await fetchJson("/projects");
    const item = (payload.items ?? []).find((project) => project.project_id === projectId);
    if (item) {
      context.project_name = item.name || projectId;
      context.dataset_versions = Number(item.dataset_versions ?? 0);
      context.model_versions = Number(item.model_versions ?? 0);
      context.latest_model_version_id = item.latest_model_version_id ?? null;
      context.champion_model_version_id = item.champion_model_version_id ?? null;
      context.latest_model_target = item.target ?? null;
      context.latest_model_task_type = item.task_type ?? null;
      context.has_datasets = context.dataset_versions > 0;
      context.has_models = Boolean(item.has_models) || context.model_versions > 0;
      context.has_champion_model = Boolean(item.has_champion_model) || Boolean(context.champion_model_version_id);
      context.is_new_project = !context.has_datasets && !context.has_models;
    }
  } catch {
    // Older backend may not expose project summary yet.
  }

  try {
    const latestDataset = await fetchJson(`/projects/${encodeURIComponent(projectId)}/datasets/latest`);
    context.latest_dataset_version_id = latestDataset.dataset_version?.version_id ?? null;
    context.latest_dataset_source_name = latestDataset.source_name ?? null;
    context.latest_dataset_created_at = latestDataset.dataset_version?.created_at ?? null;
    context.has_datasets = true;
    context.is_new_project = false;
  } catch {
    // Older backend may not expose latest-dataset route yet.
  }

  return context;
}

async function hydrateInspection(rawInspection) {
  const inspection = {
    ...rawInspection,
  };

  inspection.column_profiles = getColumnProfiles(inspection);
  inspection.column_type_summary = inspection.column_type_summary ?? buildColumnTypeSummaryFallback(inspection);
  inspection.temporal_context = inspection.temporal_context ?? buildTemporalContextFallback(inspection);
  inspection.target_summaries = {
    ...buildTargetSummariesFallback(inspection),
    ...(inspection.target_summaries ?? {}),
  };
  inspection.project_context = {
    ...(await buildProjectContextFallback(inspection.project_id || state.projectId)),
    ...(inspection.project_context ?? {}),
  };

  return inspection;
}

function clearInspection() {
  state.inspection = null;
  elements.targetSelect.innerHTML = '<option value=""></option>';
  elements.summaryRows.textContent = EMPTY_VALUE;
  elements.summaryColumns.textContent = EMPTY_VALUE;
  elements.summaryDuplicates.textContent = EMPTY_VALUE;
  elements.summarySource.textContent = EMPTY_VALUE;
  setDetailEmpty(elements.columnTypesBody, "Загрузите файл, чтобы увидеть состав схемы.");
  setDetailEmpty(elements.temporalContextBody, "Временная шкала появится после разбора файла.");
  setDetailEmpty(elements.targetSummaryBody, "Выберите целевую колонку после автозагрузки.");
  setDetailEmpty(elements.projectContextBody, "Контекст проекта появится после разбора файла.");
  elements.previewHead.replaceChildren();
  elements.previewBody.replaceChildren();
  setPreviewPlaceholder(DEFAULT_PREVIEW_MESSAGE, false);
  setBusy(false);
}

function renderTargetOptions() {
  const columns = state.inspection?.columns ?? [];
  const selectedValue = state.inspection?.recommended_target || "";
  elements.targetSelect.replaceChildren();

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Выберите колонку";
  elements.targetSelect.append(placeholder);

  columns.forEach((column) => {
    const option = document.createElement("option");
    option.value = column;
    option.textContent = column;
    elements.targetSelect.append(option);
  });

  elements.targetSelect.value = columns.includes(selectedValue) ? selectedValue : "";
}

function renderPreviewTable() {
  const columns = state.inspection?.columns ?? [];
  const rows = state.inspection?.sample_rows ?? [];
  elements.previewHead.replaceChildren();
  elements.previewBody.replaceChildren();

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
      const value = record[column];
      cell.textContent = value == null ? EMPTY_VALUE : String(value);
      row.append(cell);
    });
    elements.previewBody.append(row);
  });
}

function renderColumnTypes() {
  const summary = state.inspection?.column_type_summary;
  const items = summary?.items ?? [];
  const rows = [
    { label: "Всего колонок", value: formatNumber(summary?.total_columns ?? 0) },
    ...items.map((item) => ({
      label: item.label,
      value: formatNumber(item.count),
    })),
  ];
  renderDetailLines(elements.columnTypesBody, rows, "Типы колонок будут показаны после разбора файла.");
}

function renderTemporalContext() {
  const temporalContext = state.inspection?.temporal_context;
  if (!temporalContext) {
    setDetailEmpty(elements.temporalContextBody, "Временная шкала появится после разбора файла.");
    return;
  }

  if (!temporalContext.available) {
    setDetailEmpty(elements.temporalContextBody, temporalContext.message || "Временная колонка не найдена.");
    return;
  }

  const rows = [
    { label: "Колонка", value: temporalContext.column || EMPTY_VALUE },
    { label: "Период", value: `${formatTimestamp(temporalContext.start)} - ${formatTimestamp(temporalContext.end)}` },
    { label: "Охват", value: temporalContext.range_label || EMPTY_VALUE },
    { label: "Частота", value: temporalContext.frequency_label || "Определить не удалось" },
    { label: "Уникальных точек", value: formatNumber(temporalContext.unique_timestamps) },
    {
      label: "Пропуски по времени",
      value: temporalContext.has_gaps ? "Есть" : "Нет",
      accent: temporalContext.has_gaps,
    },
  ];

  if (temporalContext.based_on_preview) {
    rows.push({
      label: "Источник",
      value: "Оценка по preview",
    });
  }

  renderDetailLines(elements.temporalContextBody, rows, "Временная шкала не найдена.");
}

function renderTargetSummary() {
  const selectedTarget = elements.targetSelect.value.trim();
  if (!state.inspection || !selectedTarget) {
    setDetailEmpty(elements.targetSummaryBody, "Выберите целевую колонку после автозагрузки.");
    return;
  }

  const summary = state.inspection.target_summaries?.[selectedTarget];
  if (!summary) {
    setDetailEmpty(elements.targetSummaryBody, "Для выбранной колонки нет сводки.");
    return;
  }

  const rows = [
    { label: "Колонка", value: summary.column || selectedTarget },
    { label: "Тип", value: formatColumnKind(summary.kind) },
    { label: "Задача", value: formatTaskType(summary.task_type) },
    { label: "Заполнено", value: formatNumber(summary.non_null_values) },
    { label: "Уникальных", value: formatNumber(summary.distinct_values) },
  ];

  if (summary.numeric_stats) {
    rows.push(
      { label: "Минимум", value: formatNumber(summary.numeric_stats.min) },
      { label: "Максимум", value: formatNumber(summary.numeric_stats.max) },
      { label: "Среднее", value: formatNumber(summary.numeric_stats.mean) },
      { label: "Медиана", value: formatNumber(summary.numeric_stats.median) },
    );
  } else if (summary.top_values?.length) {
    rows.push(
      { label: "Классов", value: formatNumber(summary.class_count) },
      { label: "Доля лидера", value: formatPercent(summary.majority_share) },
    );
    summary.top_values.forEach((item, index) => {
      rows.push({
        label: `Топ ${index + 1}`,
        value: `${item.label} (${formatPercent(item.share)})`,
      });
    });
  } else if (summary.example_values?.length) {
    rows.push({
      label: "Примеры",
      value: summary.example_values.join(", "),
    });
  }

  if (summary.summary_basis === "preview") {
    rows.push({
      label: "Источник",
      value: "Оценка по preview",
    });
  }

  if (!summary.target_eligible || summary.looks_like_identifier) {
    rows.push({
      label: "Замечание",
      value: "Колонка выглядит не лучшим выбором для target",
      accent: true,
    });
  }

  renderDetailLines(elements.targetSummaryBody, rows, "Сводка по target недоступна.");
}

function renderProjectContext() {
  const projectContext = state.inspection?.project_context;
  if (!projectContext) {
    setDetailEmpty(elements.projectContextBody, "Контекст проекта появится после разбора файла.");
    return;
  }

  const rows = [
    { label: "Проект", value: projectContext.project_name || projectContext.project_id || EMPTY_VALUE },
    { label: "Версий датасета", value: formatNumber(projectContext.dataset_versions) },
    { label: "Версий модели", value: formatNumber(projectContext.model_versions) },
  ];

  if (projectContext.latest_dataset_source_name) {
    rows.push(
      { label: "Последний датасет", value: projectContext.latest_dataset_source_name },
      { label: "Последняя загрузка", value: formatTimestamp(projectContext.latest_dataset_created_at) },
    );
  } else if (projectContext.is_new_project) {
    rows.push({
      label: "История данных",
      value: "Для проекта это первая загрузка",
      accent: true,
    });
  }

  if (projectContext.latest_model_version_id) {
    rows.push(
      { label: "Последняя модель", value: projectContext.latest_model_version_id },
      { label: "Target модели", value: projectContext.latest_model_target || EMPTY_VALUE },
      { label: "Тип задачи", value: formatTaskType(projectContext.latest_model_task_type) },
      { label: "Обучена", value: formatTimestamp(projectContext.latest_model_created_at) },
    );
  } else {
    rows.push({
      label: "История моделей",
      value: "Модели ещё не обучались",
    });
  }

  if (projectContext.champion_model_version_id) {
    rows.push({
      label: "Champion",
      value: projectContext.champion_model_version_id,
    });
  }

  renderDetailLines(elements.projectContextBody, rows, "Связь с проектом недоступна.");
}

function renderInspection(inspection) {
  state.inspection = inspection;
  elements.summaryRows.textContent = String(inspection.rows ?? EMPTY_VALUE);
  elements.summaryColumns.textContent = String(inspection.columns?.length ?? EMPTY_VALUE);
  elements.summaryDuplicates.textContent = String(inspection.duplicates ?? EMPTY_VALUE);
  elements.summarySource.textContent = inspection.source_name || EMPTY_VALUE;
  renderTargetOptions();
  renderColumnTypes();
  renderTemporalContext();
  renderProjectContext();
  renderTargetSummary();
  renderPreviewTable();
  setBusy(false);
}

function requireFiles() {
  const files = Array.from(elements.fileInput.files ?? []);
  if (!files.length) {
    throw new Error("Выберите хотя бы один CSV или XLSX файл.");
  }
  return files;
}

function requireSelectedTarget() {
  const target = elements.targetSelect.value.trim();
  if (!target) {
    throw new Error("Выберите целевую колонку.");
  }
  return target;
}

function trainingUrl(projectId, datasetVersionId) {
  const targetUrl = new URL("./training.html", window.location.href);
  targetUrl.searchParams.set("project_id", projectId);
  targetUrl.searchParams.set("dataset_version_id", datasetVersionId);
  return targetUrl.toString();
}

async function runAction(actionName, action, options = {}) {
  const { showPreviewLoading = false } = options;
  try {
    setBusy(true);
    if (showPreviewLoading) {
      setPreviewPlaceholder(LOADING_PREVIEW_MESSAGE, true);
    }
    setStatus("busy", actionName);
    await action();
  } catch (error) {
    if (showPreviewLoading && !state.inspection) {
      setPreviewPlaceholder(DEFAULT_PREVIEW_MESSAGE, false);
    }
    setStatus("error", normalizeError(error));
    setBusy(false);
  }
}

async function handleInspect() {
  const files = requireFiles();
  const inspection = await postUploads(files.length > 1 ? "/datasets/inspect/files" : "/datasets/inspect/file", {
    files,
    projectId: state.projectId,
  });
  const hydratedInspection = await hydrateInspection(inspection);
  renderInspection(hydratedInspection);
  setStatus(
    "success",
    `${files.length > 1 ? "Пакет файлов" : "Файл"} разобран: ${hydratedInspection.rows} строк, ${hydratedInspection.columns.length} колонок.`,
  );
}

async function handleContinue() {
  const files = requireFiles();
  const target = requireSelectedTarget();
  const payload = await postUploads(files.length > 1 ? "/datasets/register/files" : "/datasets/register/file", {
    files,
    projectId: state.projectId,
    target,
  });
  setStatus("success", `Датасет сохранён: ${payload.dataset_version.version_id}.`);
  window.location.assign(trainingUrl(state.projectId, payload.dataset_version.version_id));
}

elements.continueButton.addEventListener("click", () => {
  runAction("Сохраняю датасет...", handleContinue);
});

elements.targetSelect.addEventListener("change", () => {
  setBusy(false);
  renderTargetSummary();
});

elements.fileInput.addEventListener("change", () => {
  clearInspection();
  if (elements.fileInput.files?.length) {
    runAction(LOADING_PREVIEW_MESSAGE, handleInspect, { showPreviewLoading: true });
  }
});

const pageContext = getPageContext();
state.projectId = pageContext.projectId;
syncNavigation(state.projectId);
clearInspection();
setStatus("idle", "Готово.");
