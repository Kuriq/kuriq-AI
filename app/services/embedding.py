import asyncio
import time
import logging

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.chroma import get_collection
from app.schemas.course import (
    CourseEmbeddingItem, EmbeddingRequest,
    EmbeddingMetadata, EmbeddingResponse,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
EMBEDDING_MODEL = "text-embedding-3-small"


def _build_text(course: CourseEmbeddingItem) -> str:
    parts = [course.title]
    if course.description:
        parts.append(course.description)
    if course.keywords:
        parts.append(" ".join(course.keywords))
    return "\n".join(parts)


def _build_metadata(course: CourseEmbeddingItem) -> dict:
    m = course.metadata
    return {
        "courseId": course.courseId,
        "title": course.title,
        "platform": m.platform,
        "difficulty": m.difficulty,
        "category": m.category,
        "durationWeeks": m.durationWeeks,
        "hasCertificate": m.hasCertificate,
        "isActive": m.isActive,
    }


async def embed_courses(request: EmbeddingRequest) -> EmbeddingResponse:
    start = time.monotonic()
    courses = request.courses
    collection = await asyncio.to_thread(get_collection)
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    succeeded = 0
    failed_ids: list[str] = []
    call_count = 0

    for offset in range(0, len(courses), BATCH_SIZE):
        batch = courses[offset: offset + BATCH_SIZE]
        texts = [_build_text(c) for c in batch]

        try:
            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            call_count += 1
            vectors = [e.embedding for e in response.data]

            await asyncio.to_thread(
                collection.upsert,
                ids=[c.courseId for c in batch],
                embeddings=vectors,
                metadatas=[_build_metadata(c) for c in batch],
                documents=texts,
            )
            succeeded += len(batch)
            logger.info(f"[Embedding] 배치 upsert 완료 — offset={offset}, count={len(batch)}")

        except Exception as e:
            logger.error(f"[Embedding] 배치 실패 — offset={offset}: {e}")
            failed_ids.extend(c.courseId for c in batch)

    elapsed = round((time.monotonic() - start) * 1000)

    return EmbeddingResponse(
        processed=len(courses),
        succeeded=succeeded,
        failed=len(failed_ids),
        failedCourseIds=failed_ids,
        metadata=EmbeddingMetadata(
            embeddingCallCount=call_count,
            processingTimeMs=elapsed,
        ),
    )
