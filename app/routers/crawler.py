from fastapi import APIRouter, HTTPException, status

from app.schemas.crawler import (
    CrawlerStatusResponse,
    CrawlerTriggerRequest,
    CrawlerTriggerResponse,
)
from app.services.crawler import (
    CrawlerDataClientError,
    fetch_crawler_job,
    trigger_crawler_job,
)

router = APIRouter()


@router.get("/ai/crawler/status/{jobId}", response_model=CrawlerStatusResponse)
async def get_crawler_status(jobId: str):
    try:
        return await fetch_crawler_job(jobId)
    except CrawlerDataClientError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@router.post(
    "/ai/crawler/trigger",
    response_model=CrawlerTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_crawler(request: CrawlerTriggerRequest):
    try:
        return await trigger_crawler_job(request.platform, request.incremental)
    except CrawlerDataClientError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
