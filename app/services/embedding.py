import asyncio
import time
import logging

from app.core.chroma import get_collection
from app.services.auto_embedder import embed_texts
from app.schemas.course import (
    CourseEmbeddingItem, EmbeddingRequest,
    EmbeddingMetadata, EmbeddingResponse,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


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
        "courseId": course.courseId,  # 백엔드 DB UUID
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

    succeeded = 0
    failed_ids: list[str] = []

    for offset in range(0, len(courses), BATCH_SIZE):
        batch = courses[offset: offset + BATCH_SIZE]
        texts = [_build_text(c) for c in batch]

        try:
            # 자동 임베딩 (ChromaDB 차원에 맞춤)
            embeddings = embed_texts(texts, batch_size=BATCH_SIZE)

            await asyncio.to_thread(
                collection.upsert,
                ids=[c.courseId for c in batch],
                embeddings=embeddings,
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
            embeddingCallCount=len(courses) // BATCH_SIZE + 1,
            processingTimeMs=elapsed,
        ),
    )
