"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env file into os.environ BEFORE importing other modules
# This is required for LangSmith tracing to work
load_dotenv()

from app.api.routes import deep_search, deep_graph, health, workflow
from app.config import get_settings
from app.models.database import init_db

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Validate configuration
    errors = settings.validate()
    if errors:
        from app.core import get_logger
        logger = get_logger(__name__)
        for error in errors:
            logger.error(f"Configuration error: {error}")
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    # Startup
    await init_db()
    yield
    # Shutdown
    pass


app = FastAPI(
    title="Tech News Aggregator",
    description="An AI-powered tech news aggregation system built with LangGraph",
    version="0.1.0",
    lifespan=lifespan,
    root_path="/ai",  # For proxy support - enables correct Swagger UI URLs
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods_list,
    allow_headers=settings.cors_allow_headers_list,
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(workflow.router, prefix="/api/v1/workflow", tags=["workflow"])
app.include_router(deep_search.router, prefix="/api/v1", tags=["deep_search"])
app.include_router(deep_graph.router, prefix="/api/v1", tags=["deep_graph"])


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "name": "Tech News Aggregator",
        "version": "0.1.0",
        "docs": "/docs",
    }