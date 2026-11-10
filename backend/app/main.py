from contextlib import asynccontextmanager

from fastapi import FastAPI

""" from app.core.logging_config import configure_logging
from app.core.llm_config import APP_NAME """
from app.routers.rag_router import router as rag_router
from app.config.settings import settings
logger = settings.get_logger()

#configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lazy initialization: model/vector store are initialized on first request.
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.include_router(rag_router)


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}
