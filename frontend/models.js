const state = {
  projectId: "",
  projectName: "",
  latestModelVersionId: "",
  championModelVersionId: "",
  items: [],
  error: null,
  busyVersionId: "",
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

const elements = {
  projectTitle: document.getElementById("models-project-title"),
  projectNote: document.getElementById("models-project-note"),
  totalCount: document.getElementById("models-total-count"),
  championId: document.getElementById("models-champion-id"),
  latestId: document.getElementById("models-latest-id"),
  tableBody: document.getElementById("models-table-body"),
  tableEmpty: document.getElementById("models-table-empty"),
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

function formatMetricValue(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) {
    return "—";
  }

  return numericValue.toLocaleString("ru-RU", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 4,
  });
}

function renderQualityMetrics(metrics, primaryMetric) {
  const safeMetrics = metrics && typeof metrics === "object" ? metrics : {};
  const primaryMetricKey = String(primaryMetric || "").trim().toLowerCase();

  return `
    <div class="model-metrics-grid">
      ${QUALITY_METRICS.map(({ key, label }) => {
        const metricClass = primaryMetricKey === key ? "metric-chip metric-chip-primary" : "metric-chip";
        return `
          <div class="${metricClass}">
            <span class="metric-chip-label">${escapeHtml(label)}</span>
            <strong>${formatMetricValue(safeMetrics[key])}</strong>
          </div>
        `;
      }).join("")}
    </div>
  `;
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
  return response.statusText || "Не удалось загрузить историю моделей.";
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

async function postJson(path) {
  const response = await fetch(path, {
    method: "POST",
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

async function patchJson(path, body) {
  const response = await fetch(path, {
    method: "PATCH",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  const payload = await readJsonResponse(response);

  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, response));
  }

  return payload;
}

function syncNavigation(projectId, versionId = "") {
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

  for (const link of elements.retrainingLinks) {
    const targetUrl = new URL("./retraining.html", window.location.href);
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

  for (const link of elements.reportLinks) {
    const targetUrl = new URL("./report.html", window.location.href);
    if (normalizedProjectId) {
      targetUrl.searchParams.set("project_id", normalizedProjectId);
    }
    if (versionId) {
      targetUrl.searchParams.set("model_version", versionId);
    }
    link.href = targetUrl.toString();
  }

  for (const link of elements.graphLinks) {
    const targetUrl = new URL("./graph.html", window.location.href);
    if (normalizedProjectId) {
      targetUrl.searchParams.set("project_id", normalizedProjectId);
    }
    if (versionId) {
      targetUrl.searchParams.set("model_version", versionId);
    }
    link.href = targetUrl.toString();
  }

  for (const link of elements.dssLinks) {
    const targetUrl = new URL("./dss.html", window.location.href);
    if (normalizedProjectId) {
      targetUrl.searchParams.set("project_id", normalizedProjectId);
    }
    if (versionId) {
      targetUrl.searchParams.set("model_version", versionId);
    }
    link.href = targetUrl.toString();
  }
}

function graphUrl(projectId, versionId) {
  const targetUrl = new URL("./graph.html", window.location.href);
  targetUrl.searchParams.set("project_id", projectId);
  targetUrl.searchParams.set("model_version", versionId);
  return targetUrl.toString();
}

function reportUrl(projectId, versionId) {
  const targetUrl = new URL("./report.html", window.location.href);
  targetUrl.searchParams.set("project_id", projectId);
  targetUrl.searchParams.set("model_version", versionId);
  return targetUrl.toString();
}

function renderSummary() {
  elements.totalCount.textContent = String(state.items.length);
  elements.championId.textContent = state.championModelVersionId || "—";
  elements.latestId.textContent = state.latestModelVersionId || "—";
}

function renderTable() {
  elements.tableBody.replaceChildren();

  if (state.error) {
    elements.tableEmpty.hidden = false;
    elements.tableEmpty.textContent = state.error;
    return;
  }

  if (!state.items.length) {
    elements.tableEmpty.hidden = false;
    elements.tableEmpty.textContent = state.projectId
      ? "Для этого проекта пока нет версий модели. Обучите модель на вкладке обучения."
      : "Сначала выберите проект на стартовой странице.";
    return;
  }

  const fragment = document.createDocumentFragment();
  state.items.forEach((item) => {
    const row = document.createElement("tr");
    const effectiveStatus = item.is_champion ? "champion" : item.is_latest ? "latest" : item.status;
    const statusLabel = item.is_champion ? "Champion" : item.is_latest ? "Latest" : item.status;

    row.innerHTML = `
      <td>
        <div class="model-name-cell">
          <strong>${escapeHtml(item.name || item.version_id)}</strong>
          ${item.name ? `<small class="model-version-note">${escapeHtml(item.version_id)}</small>` : ""}
        </div>
      </td>
      <td><span class="model-status ${escapeHtml(effectiveStatus)}">${escapeHtml(statusLabel)}</span></td>
      <td>${formatDateTime(item.created_at)}</td>
      <td>${escapeHtml(item.task_type || "—")}</td>
      <td>${escapeHtml(item.target || "—")}</td>
      <td>${renderQualityMetrics(item.metrics, item.primary_metric)}</td>
      <td>
        <div class="model-metric">
          <strong>${escapeHtml(item.dataset_version_id || "—")}</strong>
          <small class="model-dataset-note">${escapeHtml(item.dataset_source_name || "Источник не указан")}</small>
        </div>
      </td>
      <td>
        <div class="model-actions">
          <a href="${reportUrl(state.projectId, item.version_id)}" class="button button-secondary">Отчёт</a>
          <a href="${graphUrl(state.projectId, item.version_id)}" class="button button-secondary">График</a>
          <button type="button" class="button button-secondary" data-action="rename-model" data-version-id="${escapeHtml(item.version_id)}" data-current-name="${escapeHtml(item.name || "")}">Переименовать</button>
          ${
            item.is_champion
              ? '<button type="button" class="button button-primary" disabled>Активна</button>'
              : `<button type="button" class="button button-secondary" data-action="activate-model" data-version-id="${escapeHtml(item.version_id)}" ${
                  state.busyVersionId === item.version_id ? "disabled" : ""
                }>${state.busyVersionId === item.version_id ? "Переключаю..." : "Сделать активной"}</button>`
          }
        </div>
      </td>
    `;
    fragment.append(row);
  });

  elements.tableBody.append(fragment);
  elements.tableEmpty.hidden = true;
}

function renderPage() {
  const title = state.projectId
    ? `История моделей проекта ${state.projectName || state.projectId}`
    : "История моделей";
  elements.projectTitle.textContent = title;

  if (state.error) {
    elements.projectNote.textContent = state.error;
  } else if (!state.projectId) {
    elements.projectNote.textContent = "Откройте проект со стартовой страницы или передайте project_id в адресе.";
  } else if (!state.items.length) {
    elements.projectNote.textContent = "У проекта ещё нет обученных версий модели.";
  } else {
    elements.projectNote.textContent =
      `Найдено версий: ${state.items.length}. Champion: ${state.championModelVersionId || "нет"}, ` +
      `последняя версия: ${state.latestModelVersionId || "нет"}.`;
  }

  renderSummary();
  renderTable();
}

async function loadModelHistory() {
  const context = getPageContext();
  state.projectId = context.projectId;
  state.error = null;
  state.projectName = "";
  state.latestModelVersionId = "";
  state.championModelVersionId = "";
  state.items = [];
  syncNavigation(state.projectId);

  if (!state.projectId) {
    renderPage();
    return;
  }

  try {
    const payload = await fetchJson(`/projects/${encodeURIComponent(state.projectId)}/models`);
    state.projectName = payload.project_name || state.projectId;
    state.latestModelVersionId = payload.latest_model_version_id || "";
    state.championModelVersionId = payload.champion_model_version_id || "";
    state.items = Array.isArray(payload.items) ? payload.items : [];
    syncNavigation(state.projectId, state.championModelVersionId || state.latestModelVersionId);
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
  }

  renderPage();
}

async function activateModel(versionId) {
  if (!state.projectId || !versionId || state.busyVersionId) {
    return;
  }

  state.busyVersionId = versionId;
  state.error = null;
  elements.projectNote.textContent = `Переключаю активную модель на ${versionId}...`;
  renderTable();

  try {
    await postJson(`/projects/${encodeURIComponent(state.projectId)}/models/${encodeURIComponent(versionId)}/activate`);
    await loadModelHistory();
    elements.projectNote.textContent = `Активная модель переключена на ${versionId}.`;
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
    renderPage();
  } finally {
    state.busyVersionId = "";
    renderPage();
  }
}

async function renameModel(versionId, currentName) {
  const nextName = window.prompt("Название модели:", currentName || "");
  if (nextName === null) {
    return;
  }
  const trimmed = nextName.trim();
  if (!trimmed) {
    return;
  }

  try {
    await patchJson(`/models/${encodeURIComponent(versionId)}`, { name: trimmed });
    await loadModelHistory();
  } catch (error) {
    state.error = error instanceof Error ? error.message : String(error);
    renderPage();
  }
}

elements.tableBody.addEventListener("click", (event) => {
  const target = event.target instanceof HTMLElement ? event.target.closest("[data-action]") : null;
  if (!target) {
    return;
  }

  if (target.dataset.action === "activate-model" && target.dataset.versionId) {
    activateModel(target.dataset.versionId);
  }

  if (target.dataset.action === "rename-model" && target.dataset.versionId) {
    renameModel(target.dataset.versionId, target.dataset.currentName || "");
  }
});

renderPage();
loadModelHistory();
