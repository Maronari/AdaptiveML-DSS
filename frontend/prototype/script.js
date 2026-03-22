const container = document.getElementById('container');
const modal = document.getElementById('newProjectModal');
const projectNameInput = document.getElementById('projectName');

// Исходные проекты
const projects = [
    { name: 'Электроэнергия Апрель', created: '02.02.26', trained: '29.02.26' },
    { name: 'Электроэнергия Март', created: '15.01.26', trained: '20.02.26' },
    { name: 'Вода Февраль', created: '10.01.26', trained: '25.02.26' },
    { name: 'Газ Январь', created: '01.01.26', trained: '15.02.26' }
];

function renderCards() {
    container.innerHTML = `
        <div class="card new" onclick="openModal()">
            <h3>+ Новый проект</h3>
            <p>Укажите только имя. Данные можно будет загрузить позднее</p>
        </div>
        ${projects.map(project => `
            <div class="card existing" onclick="openProject('${project.name}')">
                <h3>${project.name}</h3>
                <p>Проект создан - ${project.created}<br>Модель обучена - ${project.trained}</p>
                <button class="btn" onclick="event.stopPropagation(); openProject('${project.name}')">Открыть</button>
            </div>
        `).join('')}
    `;
}

function openModal() {
    modal.classList.add('active');
    projectNameInput.focus();
}

function closeModal() {
    modal.classList.remove('active');
    projectNameInput.value = '';
}

function createProject() {
    const name = projectNameInput.value.trim();
    if (name) {
        projects.unshift({ name, created: '22.03.26', trained: '-' });
        renderCards();
        closeModal();
    } else {
        alert('Введите имя проекта');
    }
}

function openProject(name) {
    console.log(`Редирект на проект - ${name}`);
}

// Закрытие модалки по клику вне
modal.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
});

// ESC
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
});

// Инициализация
renderCards();
