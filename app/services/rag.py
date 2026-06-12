import logging
import re

from app.core.chroma import get_collection
from app.services.auto_embedder import embed_text
from app.schemas.course import CourseResult

logger = logging.getLogger(__name__)


def _format_course_id(chroma_id: str, metadata: dict) -> str:
    backend_course_id = (metadata.get("courseId") or "").strip()
    if backend_course_id:
        return backend_course_id

    raw_platform = metadata.get("platform", "UNKNOWN")
    platform_en = {
        "온국민평생배움터": "LLL_PORTAL",
        "에버러닝": "EVERLEARNING",
        "K-MOOC": "K_MOOC",
        "KOCW": "KOCW",
        "전국평생학습": "EVERLEARNING",
    }.get(raw_platform, raw_platform.replace("-", "_").replace(" ", "_").upper())

    if "_" in chroma_id:
        platform_course_id = chroma_id.split("_", 1)[1]
    else:
        platform_course_id = chroma_id

    return f"{platform_en}_{platform_course_id}"


def _to_course_result(chroma_id: str, metadata: dict) -> CourseResult:
    raw_platform = metadata.get("platform", "UNKNOWN")
    return CourseResult(
        course_id=_format_course_id(chroma_id, metadata),
        title=metadata.get("title", ""),
        institution=metadata.get("institution", raw_platform),
        category=metadata.get("category", ""),
        duration=metadata.get("duration", ""),
    )


def fallback_search_courses(query: str, top_k: int = 20) -> list[CourseResult]:
    collection = get_collection()
    fallback_limit = max(top_k * 10, 200)
    results = collection.get(include=["metadatas"], limit=fallback_limit)

    ids: list[str] = results.get("ids", [])
    metadatas: list[dict] = results.get("metadatas", [])
    courses = [_to_course_result(chroma_id, metadata or {}) for chroma_id, metadata in zip(ids, metadatas)]

    tokens = [token for token in re.findall(r"[0-9A-Za-z가-힣]+", query.lower()) if len(token) > 1]
    if not tokens:
        return courses[:top_k]

    scored: list[tuple[int, CourseResult]] = []
    for course in courses:
        haystack = f"{course.title} {course.category} {course.institution}".lower()
        score = sum(token in haystack for token in tokens)
        if score > 0:
            scored.append((score, course))

    if not scored:
        return courses[:top_k]

    scored.sort(key=lambda item: (-item[0], item[1].title))
    ranked = [course for _, course in scored]

    if len(ranked) < top_k:
        seen = {course.course_id for course in ranked}
        ranked.extend(course for course in courses if course.course_id not in seen)

    return ranked[:top_k]


def search_courses(
    query: str,
    category: str | None = None,
    top_k: int = 20,
) -> list[CourseResult]:
    collection = get_collection()

    where = {"category": category} if category else None

    # 자동 임베딩 (ChromaDB 차원에 맞춤)
    vector = embed_text(query)

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
    for chroma_id, m in zip(ids, metadatas):
        courses.append(_to_course_result(chroma_id, m or {}))

    logger.info(f"[RAG] query='{query[:30]}' category={category} results={len(courses)}")
    logger.info(f"[RAG] 반환된 course_id 목록: {[c.course_id for c in courses[:5]]}...")  # 최대 5 개만
    return courses
