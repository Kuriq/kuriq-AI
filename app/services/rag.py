import logging

from app.core.chroma import get_collection
from app.services.auto_embedder import embed_text
from app.schemas.course import CourseResult

logger = logging.getLogger(__name__)


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
        # ChromaDB 에 저장된 ID 가 이미 "platform_courseId" 형식
        # 예: "K-MOOC_6695", "KOCW_ac910cc25f4c5099"
        # 백엔드 format: "K_MOOC_6695", "KOCW_ac910cc25f4c5099" (대시 → 언더스코어)
        raw_platform = m.get("platform", "UNKNOWN")
        
        # 플랫폼명을 영어로 변환 (백엔드 Platform Enum 과 일치)
        platform_en = {
            "온국민평생배움터": "LLL_PORTAL",
            "에버러닝": "EVERLEARNING",
            "K-MOOC": "K_MOOC",
            "KOCW": "KOCW",
            "전국평생학습": "LLL_PORTAL",
        }.get(raw_platform, raw_platform.replace("-", "_").replace(" ", "_").upper())
        
        # ChromaDB ID 에서 플랫폼 프리픽스 제거 후 다시 포맷팅
        # "K-MOOC_6695" → "6695" → "K_MOOC_6695"
        if "_" in chroma_id:
            platform_course_id = chroma_id.split("_", 1)[1]
        else:
            platform_course_id = chroma_id
        
        formatted_course_id = f"{platform_en}_{platform_course_id}"
        
        courses.append(CourseResult(
            course_id=formatted_course_id,
            title=m.get("title", ""),
            institution=m.get("institution", raw_platform),
            category=m.get("category", ""),
            duration=m.get("duration", ""),
        ))

    logger.info(f"[RAG] query='{query[:30]}' category={category} results={len(courses)}")
    logger.info(f"[RAG] 반환된 course_id 목록: {[c.course_id for c in courses[:5]]}...")  # 최대 5 개만
    return courses
