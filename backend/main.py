from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.routes import router
from backend.services.settings import get_settings


settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_directories()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "AdaptiveML DSS API: ingestion, training, prediction, explanation and "
        "decision support for tabular datasets."
    ),
    lifespan=lifespan,
)
app.include_router(router)
