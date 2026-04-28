import logging
from urllib.parse import urljoin

import httpx

from app.core.config import settings
from app.schemas.crawler import (
    CrawlerPlatform,
    CrawlerStatusResponse,
    CrawlerTriggerResponse,
)

logger = logging.getLogger(__name__)


class CrawlerDataClientError(Exception):
    def __init__(self, status_code: int, detail):
        self.status_code = status_code
        self.detail = detail


def _data_url(path: str) -> str:
    base_url = settings.crawler_data_base_url.rstrip("/") + "/"
    return urljoin(base_url, path.lstrip("/"))


async def trigger_crawler_job(platform: CrawlerPlatform, incremental: bool) -> CrawlerTriggerResponse:
    payload = {"platform": platform, "incremental": incremental}
    try:
        async with httpx.AsyncClient(timeout=settings.crawler_data_timeout) as client:
            response = await client.post(
                _data_url("/internal/ai/crawler/trigger"),
                json=payload,
            )
    except httpx.HTTPError as e:
        logger.error("[Crawler] kuriq-data trigger 요청 실패: %s", e)
        raise CrawlerDataClientError(503, {
            "code": "CRAWLER_DATA_UNAVAILABLE",
            "message": "크롤링 데이터 서버에 연결할 수 없습니다.",
        })

    if response.status_code >= 400:
        raise CrawlerDataClientError(response.status_code, response.json())

    return CrawlerTriggerResponse.model_validate(response.json())


async def fetch_crawler_job(job_id: str) -> CrawlerStatusResponse:
    try:
        async with httpx.AsyncClient(timeout=settings.crawler_data_timeout) as client:
            response = await client.get(_data_url(f"/internal/ai/crawler/status/{job_id}"))
    except httpx.HTTPError as e:
        logger.error("[Crawler] kuriq-data status 요청 실패: %s", e)
        raise CrawlerDataClientError(503, {
            "code": "CRAWLER_DATA_UNAVAILABLE",
            "message": "크롤링 데이터 서버에 연결할 수 없습니다.",
        })

    if response.status_code >= 400:
        raise CrawlerDataClientError(response.status_code, response.json())

    return CrawlerStatusResponse.model_validate(response.json())
