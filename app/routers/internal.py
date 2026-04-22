from datetime import datetime, timezone, timedelta

from fastapi import APIRouter
from pydantic import BaseModel

from app.schemas.health import DetailedHealthResponse
from app.services.health import get_detailed_health

router = APIRouter(prefix="/internal", tags=["internal"])

KST = timezone(timedelta(hours=9))


class HealthResponse(BaseModel):
    status: str
    timestamp: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="UP",
        timestamp=datetime.now(KST).isoformat(timespec="seconds"),
    )


@router.get("/ai/health/detailed", response_model=DetailedHealthResponse)
async def health_detailed():
    return await get_detailed_health()
