import logging
from openai import OpenAI

from app.core.chroma import get_collection
from app.core.config import settings
from app.schemas.course import CourseResult

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def _embed(text: str) -> list[float]:
    response = _get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def search_courses(
    query: str,
    category: str | None = None,
    top_k: int = 20,
) -> list[CourseResult]:
    collection = get_collection()

    where = {"category": category} if category else None

    vector = _embed(query)

    query_kwargs = dict(
        query_embeddings=[vector],
        n_results=top_k,
        include=["metadatas"],
    )
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    metadatas: list[dict] = results.get("metadatas", [[]])[0]
    ids: list[str] = results.get("ids", [[]])[0]

    courses = []
    for course_id, m in zip(ids, metadatas):
        # chromaDB ID 를 platform_courseId 형식으로 변환
        # 백엔드에서 platform + platformCourseId 로 강좌를 찾음
        platform = m.get("platform", "UNKNOWN")
        platform_prefix = platform.replace(" ", "_").replace("-", "_").upper()
        formatted_course_id = f"{platform_prefix}_{course_id}"
        
        courses.append(CourseResult(
            course_id=formatted_course_id,  # LLL_PORTAL_3379164 형식
            title=m.get("title", ""),
            institution=m.get("institution", m.get("platform", "")),
            category=m.get("category", ""),
            duration=m.get("duration", ""),
        ))

    logger.info(f"[RAG] query='{query[:30]}' category={category} results={len(courses)}")
    logger.info(f"[RAG] 반환된 course_id 목록: {[c.course_id for c in courses[:5]]}...")  # 최대 5 개만
    return courses
