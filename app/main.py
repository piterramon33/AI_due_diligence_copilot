from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __doc__ as description
from app.api.copilot import router as copilot_router
from app.api.documents import router as documents_router
from app.config import get_settings, load_dotenv

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    load_dotenv()
    app = FastAPI(title=get_settings().app_name, version="0.1.0", description=description or "")
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
