import logging
from openai import OpenAI

from app.core.config import settings
from app.core.chroma import get_collection
from app.schemas.course import CourseResult

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"

_openai_client = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _embed(text: str) -> list[float]:
    response = _get_client().embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def search_courses(
    query: str,
    category: str | None = None,
    top_k: int = 20,
) -> list[CourseResult]:
    collection = get_collection()

    where = {"isActive": True}
    if category:
        where["category"] = category

    vector = _embed(query)

    results = collection.query(
        query_embeddings=[vector],
        n_results=top_k,
        where=where,
        include=["metadatas"],
    )

    metadatas: list[dict] = results.get("metadatas", [[]])[0]

    courses = []
    for m in metadatas:
        courses.append(CourseResult(
            course_id=m["courseId"],
            title=m["title"],
            institution=m.get("platform", ""),
            category=m.get("category", ""),
            duration=f"{m.get('durationWeeks', 0)}주",
        ))

    logger.info(f"[RAG] query='{query[:30]}' category={category} results={len(courses)}")
    return courses
