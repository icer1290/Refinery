"""Health check endpoints."""

from datetime import datetime

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/ready")
async def readiness_check() -> dict:
    """Readiness check for container orchestration."""
    # TODO: Add database connectivity check
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/live")
async def liveness_check() -> dict:
    """Liveness check for container orchestration."""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
    }