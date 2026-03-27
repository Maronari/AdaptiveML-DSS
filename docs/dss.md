# СППР

## Назначение

СППР в `AdaptiveML DSS` не заменяет ML-модель, а интерпретирует её результат в прикладных терминах:

`модель -> объяснение / эвристики -> факты -> правила -> рекомендации`

Итоговая цель слоя СППР:

- перевести числовой прогноз или класс в понятный уровень риска;
- показать, какие факторы сильнее всего повлияли на решение;
- вернуть не только оценку, но и набор действий.

## Где находится логика

Основные файлы:

- `backend/services/decision_service.py` — orchestration DSS-потока;
- `backend/services/dss_config_service.py` — чтение и сохранение rule-set конфигурации;
- `explainability/fact_builder/builder.py` — сборка фактов из prediction/explanation;
- `dss/experta_engine/engine.py` — rule engine на `Experta`;
- `dss/rules/loader.py` — загрузка и валидация JSON-конфига;
- `dss/rules/default_rule_sets.json` — встроенный baseline rule sets;
- `storage/registry/dss_rule_sets.json` — editable override, который меняется из UI;
- `dss/recommendations/formatter.py` — сборка финального payload;
- `frontend/dss.html`
- `frontend/dss.js`

## Два режима работы

Сейчас в проекте есть два разных DSS-сценария.

### 1. Inline DSS

Эндпоинт:

- `POST /decision/run`

Этот режим работает по произвольным входным строкам.

Поток:

1. backend получает `records`;
2. активная champion-модель проекта делает prediction;
3. `ExplanationService` строит `top_factors`;
4. `build_facts(...)` превращает prediction и top factors в факты;
5. `DecisionEngine` выбирает правило;
6. formatter возвращает `summary`, `actions`, `rationale`.

Этот режим нужен для обычного inference по входным данным пользователя.

### 2. Forecast DSS

Эндпоинт:

- `POST /decision/forecast`

Это основной режим для страницы СППР в UI.

Он работает не по историческим строкам датасета, а по будущим точкам прогноза.

Поток:

1. выбирается конкретная версия модели `model_version` или champion проекта;
2. строится forecast из forecasting bundle;
3. forecast downsampling делается тем же способом, что и на `graph.html`;
4. по каждой будущей точке собираются DSS-факты;
5. правила возвращают рекомендации уже для будущих значений target.

Именно поэтому текущая страница СППР должна совпадать со страницей графика по:

- `model_version`;
- интервалу прогноза;
- числу отображаемых точек.

## Как формируются факты

### Для inline DSS

Факты собираются в `explainability/fact_builder/builder.py`.

На вход идут:

- `task_type`
- `prediction`
- `confidence`
- `top_factors`

На выходе:

- `risk_level`
- `prediction`
- `confidence`
- `strong_positive_factors`
- `medium_positive_factors`
- `negative_factors`

Логика простая:

- сначала из `confidence` выводится базовый `risk_level`;
- затем из `top_factors` выделяются сильные положительные, средние положительные и отрицательные факторы.

### Для forecast DSS

Факты собираются в `backend/services/decision_service.py`.

Так как для future forecast нет SHAP по сырым входным строкам, используется эвристический контекст на основе недавней истории:

- средний уровень ряда за последнее окно;
- стандартное отклонение;
- квантиль `P75`;
- квантиль `P90`;
- последний фактический target.

На основе этого для каждой будущей точки оцениваются:

- отклонение от среднего уровня;
- отклонение от последнего факта или предыдущего forecast-шага;
- выход выше `P90` недавней истории;
- попадание в пиковые часы.

Из этих признаков строятся `top_factors`, а затем вычисляется `risk_level`.

## Как работают правила

Теперь правила не зашиты в Python и редактируются как JSON-конфиг.

Источник конфигурации:

- builtin fallback: `dss/rules/default_rule_sets.json`
- editable override: `storage/registry/dss_rule_sets.json`

В конфиге есть:

- `default_rule_set`
- несколько независимых `rule_sets`
- внутри каждого rule set: `scenarios` и `rules`

Каждое правило задаёт:

- `rule_id`
- `priority`
- `scenario_id`
- `conditions`

Сейчас в проекте уже есть отдельные rule sets для разных режимов:

- `inline_default`
- `forecast_default`
- `forecast_conservative`

И расширенные сценарии, например:

- `manual_review`
- `targeted_diagnostics`
- `observe`
- `load_shedding`
- `capacity_reallocation`
- `preventive_maintenance`
- `watch_peak_window`
- `stabilization_review`

### Как матчится правило

`DecisionEngine` проходит правила выбранного `rule_set` по `priority` и ищет первое совпадение по `conditions`.

Поддерживаются условия:

- точное сравнение значения;
- `_min`
- `_max`
- `_in`
- `_equals`

Итог ответа теперь включает:

- `rule_id`
- `rule_set`
- `scenario_id`

Поэтому в UI и в логах можно видеть не только текст рекомендации, но и конкретное сработавшее правило.

## Как формируется рекомендация

Финальный ответ СППР состоит из четырёх частей:

- `risk_level` — итоговый уровень риска;
- `summary` — короткая текстовая формулировка;
- `actions` — список рекомендуемых действий;
- `rationale` — объяснение, какие факторы сработали.

`actions` берутся из `dss/scenarios/defaults.py`.

Примеры:

- `manual_review` — снижение нагрузки, ручная проверка, внеплановая диагностика;
- `targeted_diagnostics` — проверка ведущих факторов и повторный расчёт;
- `observe` — продолжить наблюдение без немедленного вмешательства.

`rationale` собирается из списков факторов:

- сильные положительные факторы;
- сдерживающие факторы.

## Как это выглядит в UI

Страница:

- `/app/dss.html`

Текущий UI работает так:

1. получает `project_id` и, при наличии, `model_version` из query string;
2. загружает summary выбранной модели;
3. строит тот же forecast request, что и `graph.html`;
4. запрашивает `GET /models/{version_id}/forecast`;
5. показывает sampled forecast points;
6. вызывает `POST /decision/forecast` с той же моделью и теми же параметрами;
7. показывает рекомендации по тем же точкам, что видны на графике.

Это сделано специально, чтобы прогноз на графике и прогноз внутри СППР совпадали.

## Редактирование правил прямо в UI

На `/app/dss.html` есть встроенный JSON editor.

Он работает так:

1. при загрузке страницы запрашивается `GET /dss/rulesets`;
2. текущий JSON показывается в textarea;
3. после правки frontend отправляет `PUT /dss/rulesets`;
4. backend валидирует конфиг и сохраняет override в `storage/registry/dss_rule_sets.json`.

Это позволяет менять:

- наборы правил;
- сценарии;
- приоритеты;
- условия срабатывания

без изменения Python-кода и без пересборки rule engine.

## Ограничения текущей реализации

Текущее СППР функционально, но ещё не является полноценной предметной экспертной системой.

Основные ограничения:

- сценарии уже разделены по rule sets, но пока не привязаны к отдельным проектам;
- для forecast DSS используются эвристические факторы, а не полноценный SHAP по будущим точкам;
- пороги `low/medium/high` заданы общими правилами, а не бизнес-нормативами предметной области;
- правила редактируются через JSON, но пока нет отдельного визуального form-builder поверх JSON;
- часть факторов модели может быть технической, если в обучение попали слабые признаки.

## Когда стоит дорабатывать СППР дальше

Следующий практический шаг для развития DSS-слоя:

- вынести пороги риска в конфигурацию проекта;
- добавить предметные правила вместо общих `observe/manual_review`;
- разделить рекомендации по типам объектов или режимам работы;
- привязать DSS к доменным ограничениям, а не только к статистике ряда;
- сделать визуальный слой, где прогноз и рекомендации показываются на одном графике.

## Краткая схема

```text
dataset/model_version
    -> forecast or prediction
    -> explanation or heuristic factors
    -> facts
    -> Experta rules
    -> scenario
    -> recommendation payload
```
