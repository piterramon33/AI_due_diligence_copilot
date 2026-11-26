from fastapi import FastAPI

from app import __doc__ as description
from app.api.documents import router as documents_router
from app.config import get_settings


def create_app() -> FastAPI:
    app = FastAPI(title=get_settings().app_name, version="0.1.0", description=description or "")
    app.include_router(documents_router)

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
