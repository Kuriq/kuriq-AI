import logging
from fastapi import APIRouter, HTTPException
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse, RecommendationCourse
from app.services.rag import search_courses

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/ai/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):

    # 1. 카테고리 기반으로 ChromaDB 벡터 유사도 검색
    try:
        query = f"{request.courseTitle} 심화 다음단계 응용" # 강좌 제목에 "심화", "다음단계", "응용" 넣어서 다음 추천 검색
        candidates = search_courses(
            query=request.courseTitle,  # 카테고리 -> 강좌 제목으로 변경
            category=None,  # 카테고리 필터 제거, 유사도만으로 검색
            top_k=request.top_k,
        )
        logger.info(f"[추천] courseId={request.courseId} 기준 후보 {len(candidates)}개 검색됨")
    except Exception as e:
        logger.error(f"[추천] 벡터 검색 실패: {e}")
        raise HTTPException(status_code=500, detail={
            "code": "SEARCH_ERROR",
            "message": "강좌 검색 중 오류가 발생했습니다.",
        })

    # 2. 방금 이수한 강좌 본인 제외
    filtered = [c for c in candidates if c.course_id != request.courseId]

    # 3. 응답 반환
    return RecommendationResponse(
        courses=[
            RecommendationCourse(
                course_id=c.course_id,
                title=c.title,
                institution=c.institution,
                category=c.category,
                duration=c.duration,
            )
            for c in filtered
        ]
    )