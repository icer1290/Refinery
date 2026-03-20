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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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