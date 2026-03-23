# Установка и окружение

Проект запускается на `FastAPI` и использует локальные файловые каталоги как стартовую реализацию реестра датасетов и реестра моделей.

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

Для Windows:

```powershell
scripts\setup_full_env.cmd
```

Если `python3.13` уже активен:

```bash
python -m pip install --upgrade "pip<26" "setuptools<81" wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.10.0+cpu"
python -m pip install -r requirements.txt
python -m pip install --no-deps "LightAutoML==0.4.1"
```

## Режим core-only

`requirements-core.txt` оставлен только для текущего fallback/scaffold-режима, если нужно запускать backend на Python 3.14 без полного ML-стека.

В этом режиме каркас использует:
- sklearn fallback вместо `LightAutoML`;
- proxy explainability вместо жёсткой SHAP-зависимости;
- fallback на правилах вместо жёсткой `Experta`-зависимости.

API при этом уже собран так, чтобы заменить внутренние адаптеры без смены внешнего контракта.

## Документация

Для сайта документации добавлены:

- `mkdocs.yml`
- `requirements-docs.txt`

Установка зависимостей для документации:

```bash
python -m pip install -r requirements-docs.txt
```

Локальный запуск:

```bash
mkdocs serve
```

Сборка статической версии:

```bash
mkdocs build
```

## PostgreSQL и миграции

Для локальной базы используется сервис `postgres` из `docker-compose.yml`.

Запуск базы:

```bash
docker compose up -d postgres
```

Применение миграций:

```bash
python -m alembic upgrade head
```

По умолчанию Alembic подключается к:

- `postgresql+psycopg2://adaptiveml:adaptiveml@localhost:5432/adaptiveml`

Подключение можно переопределить через:

- `ADAPTIVEML_DATABASE_URL`
- или переменные `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`

Где смотреть схему:

- `storage/postgresql/schema.sql`
- `docs/database.md`
- `docs/database-single-diagram.md`

## Запуск API и UI

Linux/macOS:

```bash
uvicorn backend.main:app --reload
```

Windows:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

После запуска:

- API: `http://localhost:8000`
- UI: `http://localhost:8000/app/`

## Storage в Docker

При запуске через `docker compose` runtime-storage разделён на два слоя:

- metadata и `registry.sqlite3` живут в docker volume `api_storage`;
- snapshot-ы датасетов и model artifacts пишутся в MinIO как S3-объекты.

Это значит:

- контейнер `api` больше не зависит от локального `./storage` для рабочих данных;
- dataset/model artifacts переживают пересоздание `api`, пока жив volume `minio_data`;
- registry metadata переживает пересоздание `api`, пока жив volume `api_storage`.

Текущая конфигурация внутри контейнера:

- `ADAPTIVEML_STORAGE_ROOT=/var/lib/adaptiveml/storage`
- `ADAPTIVEML_OBJECT_STORAGE_BACKEND=minio`
- `ADAPTIVEML_OBJECT_STORAGE_ENDPOINT=http://minio:9000`
- bucket `adaptiveml-datasets` для dataset snapshot-ов
- bucket `adaptiveml-artifacts` для model bundle-ов

Полезные команды:

```bash
docker compose up -d api minio
docker volume inspect adaptiveml-dss_api_storage
docker volume inspect adaptiveml-dss_minio_data
```

Если нужно полностью сбросить runtime-storage контейнеров:

```bash
docker compose down
docker volume rm adaptiveml-dss_api_storage adaptiveml-dss_minio_data
```
