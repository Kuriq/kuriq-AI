import logging
from fastapi import APIRouter, HTTPException
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse, RecommendationCourse
from app.services.rag import search_courses_with_score
from app.utils.course_metadata import normalize_category

logger = logging.getLogger(__name__)
router = APIRouter()

# 유사도 임계값 — 이 값 이하면 관련 없는 강좌로 판단하여 제외
# ChromaDB cosine distance 기준: 0에 가까울수록 유사, 1에 가까울수록 무관
SIMILARITY_THRESHOLD = 4.0

@router.post("/ai/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    # 1. 강좌 제목 기반으로 ChromaDB 벡터 유사도 검색 (점수 포함)
    try:
        candidates = search_courses_with_score(
            query=request.courseTitle,
            category=None,  # 카테고리 필터 제거, 유사도만으로 검색
            top_k=request.top_k * 3,  # 필터링 후 충분한 결과를 위해 넉넉하게 요청
        )
        logger.info(f"[추천] courseId={request.courseId} 기준 후보 {len(candidates)}개 검색됨")
    except Exception as e:
        logger.error(f"[추천] 벡터 검색 실패: {e}")
        raise HTTPException(status_code=500, detail={
            "code": "SEARCH_ERROR",
            "message": "강좌 검색 중 오류가 발생했습니다.",
        })

    # 2. 본인 강좌 제외 + 유사도 임계값 이하 제외
    # distance가 낮을수록 유사도가 높음 (cosine distance)
    filtered = [
        c for c, distance in candidates
        if c.course_id != request.courseId
        and distance <= SIMILARITY_THRESHOLD
    ]

    logger.info(f"[추천] 유사도 필터링 후 {len(filtered)}개 남음 (threshold={SIMILARITY_THRESHOLD})")

    # 3. 응답 반환 (top_k개만)
    return RecommendationResponse(
        courses=[
            RecommendationCourse(
                course_id=c.course_id,
                title=c.title,
                institution=c.institution,
                category=c.category,
                duration=c.duration,
            )
            for c in filtered[:request.top_k]
        ]
    )
