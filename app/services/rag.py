import logging
from sentence_transformers import SentenceTransformer

from app.core.chroma import get_collection
from app.schemas.course import CourseResult

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_st_model = None


def _get_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(EMBEDDING_MODEL)
    return _st_model


def _embed(text: str) -> list[float]:
    return _get_model().encode(text).tolist()


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
        courses.append(CourseResult(
            course_id=course_id,
            title=m.get("title", ""),
            institution=m.get("institution", m.get("platform", "")),
            category=m.get("category", ""),
            duration=m.get("duration", ""),
        ))

    logger.info(f"[RAG] query='{query[:30]}' category={category} results={len(courses)}")
    return courses
