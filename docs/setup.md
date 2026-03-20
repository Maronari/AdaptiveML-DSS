# Setup Notes

Проект запускается на `FastAPI` и использует локальные файловые registry/directories как стартовую реализацию `Dataset Registry` и `Model Registry`.

## Полный стек

`requirements.txt` описывает полный стек из `README.md` и runtime-зависимости `LightAutoML`.

Для полной установки используйте `Python 3.13.x`.

`torch` ставится через официальный CPU-only индекс PyTorch, а `LightAutoML` ставится в bootstrap-скрипте отдельным шагом с `--no-deps`.

Причина: upstream pin `statsmodels<=0.14.0` ломает установку через resolver, а стандартный `torch` на Linux тянет тяжёлые CUDA-пакеты. Поэтому скрипт:
1. ставит CPU-only `torch`;
2. ставит runtime-зависимости с `statsmodels>=0.14.6` и более лёгким `xgboost==2.0.3`;
3. затем ставит сам `LightAutoML`.

Рекомендуемая локальная установка:

```bash
scripts/setup_full_env.sh python3.13
```

Если `python3.13` уже активен:

```bash
python -m pip install --upgrade "pip<26" "setuptools<81" wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.10.0+cpu"
python -m pip install -r requirements.txt
python -m pip install --no-deps "LightAutoML==0.4.1"
```

## Core-only режим

`requirements-core.txt` оставлен только для текущего fallback/scaffold-режима, если нужно запускать backend на Python 3.14 без полного ML-стека.

В этом режиме каркас использует:
- sklearn fallback вместо `LightAutoML`;
- proxy explainability вместо жёсткой SHAP-зависимости;
- rule-based fallback вместо жёсткой `Experta`-зависимости.

API при этом уже собран так, чтобы заменить внутренние адаптеры без смены внешнего контракта.
