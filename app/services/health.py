import asyncio
import time
import logging
from datetime import datetime, timezone, timedelta

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.chroma import get_collection
from app.core.scheduler_state import get_state
from app.schemas.health import (
    OpenAIHealth, ChromaDBHealth, SchedulerHealth,
    HealthComponents, DetailedHealthResponse,
)

logger = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))


async def _check_openai() -> OpenAIHealth:
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    start = time.monotonic()
    try:
        await client.models.retrieve(settings.llm_model)
        latency = round((time.monotonic() - start) * 1000)
        return OpenAIHealth(status="UP", latencyMs=latency)
    except Exception as e:
        latency = round((time.monotonic() - start) * 1000)
        logger.warning(f"[Health] OpenAI DOWN: {e}")
        return OpenAIHealth(status="DOWN", latencyMs=latency)


async def _check_chromadb() -> ChromaDBHealth:
    start = time.monotonic()
    try:
        collection = await asyncio.to_thread(get_collection)
        count = await asyncio.to_thread(collection.count)
        latency = round((time.monotonic() - start) * 1000)
        return ChromaDBHealth(status="UP", vectorCount=count, latencyMs=latency)
    except Exception as e:
        latency = round((time.monotonic() - start) * 1000)
        logger.warning(f"[Health] ChromaDB DOWN: {e}")
        return ChromaDBHealth(status="DOWN", vectorCount=0, latencyMs=latency)


async def _check_scheduler() -> SchedulerHealth:
    state = get_state()
    return SchedulerHealth(
        status=state["status"],
        nextCrawlAt=state.get("nextCrawlAt"),
        lastCrawlCompletedAt=state.get("lastCrawlCompletedAt"),
    )


async def get_detailed_health() -> DetailedHealthResponse:
    openai_health, chroma_health, scheduler_health = await asyncio.gather(
        _check_openai(),
        _check_chromadb(),
        _check_scheduler(),
    )

    statuses = {openai_health.status, chroma_health.status, scheduler_health.status}
    overall = "DEGRADED" if statuses & {"DOWN", "DEGRADED"} else "UP"

    return DetailedHealthResponse(
        status=overall,
        components=HealthComponents(
            openai=openai_health,
            chromadb=chroma_health,
            scheduler=scheduler_health,
        ),
        timestamp=datetime.now(KST).isoformat(timespec="seconds"),
    )
