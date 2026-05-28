from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.schemas.health import DetailedHealthResponse
from app.schemas.course import CourseSearchRequest, CourseSearchResponse
from app.services.health import get_detailed_health
from app.routers.courses import search_courses as chroma_search

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


@router.post("/ai/courses/search", response_model=CourseSearchResponse)
async def search_courses(
    request: CourseSearchRequest,
    x_internal_key: str = Header(..., alias="X-Internal-Key"),
):
    """내부 API: chromaDB 에서 강좌 검색 (백엔드용 프록시)"""
    INTERNAL_API_KEY = "dev-secret"
    if x_internal_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    return await chroma_search(request, x_internal_key=x_internal_key)
