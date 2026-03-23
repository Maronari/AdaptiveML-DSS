from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.services.settings import get_settings


settings = get_settings()
frontend_dir = Path(__file__).resolve().parents[1] / "frontend"


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


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Return an empty favicon response to keep logs clean."""
    return Response(status_code=204)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the bare app root to the frontend shell."""
    return RedirectResponse(url="/app/")


app.include_router(router)
app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")
