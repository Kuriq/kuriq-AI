from datetime import datetime, timezone, timedelta

from fastapi import APIRouter
from pydantic import BaseModel

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
