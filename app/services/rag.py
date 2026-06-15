import logging
import re
from app.core.chroma import get_collection
from app.services.auto_embedder import embed_text
from app.schemas.course import CourseResult

logger = logging.getLogger(__name__)


def _format_course_id(chroma_id: str, metadata: dict) -> str:
    """ChromaDB ID를 백엔드 course_id로 변환"""
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
    """ChromaDB 메타데이터 → CourseResult 변환"""
    raw_platform = metadata.get("platform", "UNKNOWN")
    return CourseResult(
        course_id=_format_course_id(chroma_id, metadata),
        title=metadata.get("title", ""),
        institution=metadata.get("institution", raw_platform),
        category=metadata.get("category", ""),
        duration=metadata.get("duration", ""),
        difficulty=metadata.get("difficulty") or metadata.get("level", ""),  # 난이도
        duration_weeks=metadata.get("durationWeeks") or 0,                   # 수강 기간 (주)
    )


def _deduplicate(courses: list[CourseResult]) -> list[CourseResult]:
    """같은 제목의 강좌 중복 제거 — 첫 번째만 유지 (DB 데이터는 유지됨)"""
    seen_titles = set()
    result = []
    for course in courses:
        if course.title not in seen_titles:
            seen_titles.add(course.title)
            result.append(course)
    return result


def _deduplicate_with_score(courses_with_scores: list[tuple[CourseResult, float]]) -> list[tuple[CourseResult, float]]:
    """점수 포함 결과에서 같은 제목의 강좌 중복 제거 — 첫 번째만 유지"""
    seen_titles = set()
    result = []
    for course, score in courses_with_scores:
        if course.title not in seen_titles:
            seen_titles.add(course.title)
            result.append((course, score))
    return result


def fallback_search_courses(query: str, top_k: int = 20) -> list[CourseResult]:
    """벡터 검색 실패 시 키워드 기반 fallback 검색"""
    collection = get_collection()
    # 중복 제거 후 top_k개를 확보하기 위해 넉넉하게 가져옴
    fallback_limit = max(top_k * 20, 500)
    results = collection.get(include=["metadatas"], limit=fallback_limit)

    ids: list[str] = results.get("ids", [])
    metadatas: list[dict] = results.get("metadatas", [])
    courses = [_to_course_result(chroma_id, metadata or {}) for chroma_id, metadata in zip(ids, metadatas)]

    # 제목 기준 중복 제거
    courses = _deduplicate(courses)

    # 쿼리 토큰으로 점수 계산
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

    # 상위 결과가 top_k보다 적으면 나머지로 채움
    if len(ranked) < top_k:
        seen = {course.course_id for course in ranked}
        ranked.extend(course for course in courses if course.course_id not in seen)

    return ranked[:top_k]


def search_courses(
    query: str,
    category: str | None = None,
    top_k: int = 20,
) -> list[CourseResult]:
    """ChromaDB 벡터 검색 — 중복 제거 후 top_k개 반환"""
    collection = get_collection()
    where = {"category": category} if category else None

    # 중복 제거 후 top_k개를 확보하기 위해 넉넉하게 요청
    fetch_k = top_k * 5

    # 쿼리 텍스트를 임베딩 벡터로 변환
    vector = embed_text(query)

    query_kwargs = dict(
        query_embeddings=[vector],
        n_results=fetch_k,
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

    # 제목 기준 중복 제거 후 top_k개만 반환
    courses = _deduplicate(courses)[:top_k]

    logger.info(f"[RAG] query='{query[:30]}' category={category} results={len(courses)} (중복 제거 후)")
    logger.info(f"[RAG] 반환된 course_id 목록: {[c.course_id for c in courses[:5]]}...")

    return courses


def search_courses_with_score(
    query: str,
    category: str | None = None,
    top_k: int = 20,
) -> list[tuple[CourseResult, float]]:
    """ChromaDB 벡터 검색 — 유사도 점수(distance) 포함하여 반환

    distance: cosine distance (0에 가까울수록 유사, 1에 가까울수록 무관)
    추천 필터링 시 distance가 낮은 강좌만 반환하도록 사용
    """
    collection = get_collection()
    where = {"category": category} if category else None

    # 중복 제거 후 top_k개를 확보하기 위해 넉넉하게 요청
    fetch_k = top_k * 5

    # 쿼리 텍스트를 임베딩 벡터로 변환
    vector = embed_text(query)

    query_kwargs = dict(
        query_embeddings=[vector],
        n_results=fetch_k,
        include=["metadatas", "distances"],  # distances 포함
    )
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    metadatas: list[dict] = results.get("metadatas", [[]])[0]
    ids: list[str] = results.get("ids", [[]])[0]
    distances: list[float] = results.get("distances", [[]])[0]  # cosine distance

    courses_with_scores = []
    for chroma_id, m, distance in zip(ids, metadatas, distances):
        course = _to_course_result(chroma_id, m or {})
        courses_with_scores.append((course, distance))

    # 제목 기준 중복 제거 후 top_k개만 반환
    courses_with_scores = _deduplicate_with_score(courses_with_scores)[:top_k]

    logger.info(f"[RAG] query='{query[:30]}' category={category} results={len(courses_with_scores)} (점수 포함, 중복 제거 후)")

    return courses_with_scores
