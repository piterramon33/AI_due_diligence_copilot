from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __doc__ as description
from app.api.copilot import router as copilot_router
from app.api.documents import router as documents_router
from app.config import get_settings, load_dotenv
from app.demo import preload, preload_enabled
from app.storage import get_store

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    load_dotenv()
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if preload_enabled():  # the bundled Apple 10-K, so there is something to ask about right away
            preload(app.dependency_overrides.get(get_store, get_store)())
        yield

    app = FastAPI(title=get_settings().app_name, version="0.1.0", description=description or "", lifespan=lifespan)
    app.include_router(documents_router)
    app.include_router(copilot_router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
