const container = document.getElementById("container");
const emptyState = document.getElementById("projectsEmpty");
const projectsStatus = document.getElementById("projectsStatus");
const modal = document.getElementById("newProjectModal");
const projectNameInput = document.getElementById("projectName");
const projectModalStatus = document.getElementById("projectModalStatus");
const createProjectButton = document.getElementById("createProjectButton");
const cancelProjectButton = document.getElementById("cancelProjectButton");

const state = {
  projects: [],
  isCreating: false,
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

async function readJson(response) {
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
  return response.statusText || "Не удалось выполнить запрос.";
}

async function fetchJson(path, options = undefined) {
  const response = await fetch(path, options);
  const payload = await readJson(response);

  if (!response.ok) {
    throw new Error(extractErrorMessage(payload, response));
  }

  return payload;
}

function formatDate(dateString) {
  if (!dateString) {
    return "-";
  }

  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) {
    return "-";
  }

  return date.toLocaleDateString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "2-digit",
  });
}

function setPageStatus(message, kind = "idle") {
  projectsStatus.textContent = message;
  projectsStatus.dataset.kind = kind;
}

function setModalStatus(message, kind = "idle") {
  projectModalStatus.textContent = message;
  projectModalStatus.dataset.kind = kind;
}

function trainingUrl(projectId) {
  const targetUrl = new URL("./training.html", window.location.href);
  targetUrl.searchParams.set("project_id", projectId);
  return targetUrl.toString();
}

function graphUrl(project) {
  const targetUrl = new URL("./graph.html", window.location.href);
  targetUrl.searchParams.set("project_id", project.project_id);
  if (project.latest_model_version_id) {
    targetUrl.searchParams.set("model_version", project.latest_model_version_id);
  }
  return targetUrl.toString();
}

function openProject(projectId) {
  const project = state.projects.find((item) => item.project_id === projectId);
  if (!project) {
    return;
  }

  if (project.has_models) {
    window.location.assign(graphUrl(project));
    return;
  }

  window.location.assign(trainingUrl(project.project_id));
}

function renderCards() {
  emptyState.hidden = state.projects.length > 0;
  container.innerHTML = `
    <div class="card new" data-action="new">
      <div>
        <h3>+ Новый проект</h3>
        <p>Укажите только имя. Данные можно будет загрузить позднее</p>
      </div>
    </div>
    ${state.projects.map((project) => `
      <div class="card existing" data-action="open" data-project-id="${escapeHtml(project.project_id)}">
        <div>
          <div class="project-topline">
            <span class="project-id">${escapeHtml(project.project_id)}</span>
            ${project.has_champion_model ? '<span class="project-state trained">Champion</span>' : '<span class="project-state">Без модели</span>'}
          </div>
          <h3>${escapeHtml(project.name)}</h3>
          <p>
            Проект создан - ${formatDate(project.created_at)}<br>
            Модель обучена - ${formatDate(project.last_trained_at)}<br>
            Версий модели - ${project.model_versions}
          </p>
        </div>
        <div class="card-actions">
          <button class="btn" type="button" data-action="open" data-project-id="${escapeHtml(project.project_id)}">Открыть</button>
          <button class="btn danger" type="button" data-action="delete" data-project-id="${escapeHtml(project.project_id)}">Удалить</button>
        </div>
      </div>
    `).join("")}
  `;
}

function openModal() {
  modal.classList.add("active");
  projectNameInput.focus();
  setModalStatus("Укажите название проекта. Идентификатор будет создан автоматически.");
}

function closeModal(force = false) {
  if (state.isCreating && !force) {
    return;
  }
  modal.classList.remove("active");
  projectNameInput.value = "";
  setModalStatus("Укажите название проекта. Идентификатор будет создан автоматически.");
}

function setCreating(isCreating) {
  state.isCreating = isCreating;
  createProjectButton.disabled = isCreating;
  cancelProjectButton.disabled = isCreating;
}

async function loadProjects() {
  setPageStatus("Загружаю проекты...", "loading");

  try {
    const payload = await fetchJson("/projects");
    state.projects = Array.isArray(payload?.items) ? payload.items : [];
    renderCards();
    if (state.projects.length) {
      setPageStatus(`Найдено проектов: ${state.projects.length}.`, "success");
    } else {
      setPageStatus("Проектов пока нет.", "idle");
    }
  } catch (error) {
    renderCards();
    setPageStatus(error instanceof Error ? error.message : String(error), "error");
  }
}

async function createProject() {
  const name = projectNameInput.value.trim();
  if (!name) {
    setModalStatus("Введите имя проекта.", "error");
    projectNameInput.focus();
    return;
  }

  setCreating(true);
  setModalStatus("Создаю проект...", "loading");

  try {
    const project = await fetchJson("/projects", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ name }),
    });
    closeModal(true);
    await loadProjects();
    window.location.assign(trainingUrl(project.project_id));
  } catch (error) {
    setModalStatus(error instanceof Error ? error.message : String(error), "error");
  } finally {
    setCreating(false);
  }
}

async function deleteProject(projectId) {
  const project = state.projects.find((item) => item.project_id === projectId);
  if (!project) {
    return;
  }

  const confirmed = window.confirm(
    `Удалить проект "${project.name}"?\n\nБудут удалены связанные датасеты, модели и артефакты.`,
  );
  if (!confirmed) {
    return;
  }

  setPageStatus(`Удаляю проект ${project.name}...`, "loading");

  try {
    await fetchJson(`/projects/${encodeURIComponent(projectId)}`, {
      method: "DELETE",
    });
    await loadProjects();
  } catch (error) {
    setPageStatus(error instanceof Error ? error.message : String(error), "error");
  }
}

container.addEventListener("click", (event) => {
  const target = event.target instanceof HTMLElement ? event.target.closest("[data-action]") : null;
  if (!target) {
    return;
  }

  const { action, projectId } = target.dataset;
  if (action === "new") {
    openModal();
    return;
  }

  if (action === "open" && projectId) {
    openProject(projectId);
    return;
  }

  if (action === "delete" && projectId) {
    event.stopPropagation();
    deleteProject(projectId);
  }
});

cancelProjectButton.addEventListener("click", () => {
  closeModal();
});

createProjectButton.addEventListener("click", () => {
  createProject();
});

modal.addEventListener("click", (event) => {
  if (event.target === modal) {
    closeModal();
  }
});

projectNameInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    createProject();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeModal();
  }
});

renderCards();
loadProjects();
