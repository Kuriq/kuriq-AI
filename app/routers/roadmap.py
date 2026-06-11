import logging
import time
from fastapi import APIRouter, HTTPException
from app.schemas.roadmap import RoadmapRequest, RoadmapResponse, RescheduleRequest, RescheduleResponse
from app.services.rag import fallback_search_courses, search_courses
from app.services.llm import extract_intent, generate_roadmap
from app.services.reschedule import reschedule_roadmap
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

MIN_CANDIDATE_POOL = 1


@router.post("/ai/roadmap/generate", response_model=RoadmapResponse)
async def generate(request: RoadmapRequest):
    start = time.time()

    # 의도 추출
    try:
        intent = extract_intent(request.prompt)
        logger.info(f"[로드맵] 의도 추출 완료 — {intent.interestArea} / {intent.currentLevel}")
    except Exception as e:
        logger.error(f"[로드맵] 의도 추출 실패: {e}")
        raise HTTPException(status_code=502, detail={
            "code": "LLM_API_ERROR",
            "message": "LLM API 호출에 실패했습니다.",
        })

    # 벡터 검색
    try:
        query = f"{intent.interestArea} {intent.goal} {intent.currentLevel}"
        candidates = search_courses(
            query=query,
            category=None,
            top_k=settings.rag_top_k,
        )
        if len(candidates) < MIN_CANDIDATE_POOL:
            logger.warning(f"[로드맵] 벡터 검색 결과 부족 — fallback 검색 시도 ({len(candidates)}개)")
            candidates = fallback_search_courses(query, top_k=max(settings.rag_top_k, 20))
        logger.info(f"[로드맵] 후보 강좌 {len(candidates)}개 검색됨")
    except Exception as e:
        logger.error(f"[로드맵] 벡터 검색 실패: {e}")
        raise HTTPException(status_code=500, detail={
            "code": "SEARCH_ERROR",
            "message": "강좌 검색 중 오류가 발생했습니다.",
        })

    if len(candidates) < MIN_CANDIDATE_POOL:
        raise HTTPException(status_code=404, detail={
            "code": "NO_COURSES_FOUND",
            "message": "관련 강좌를 충분히 찾을 수 없습니다.",
        })

    # 로드맵 생성
    try:
        roadmap = generate_roadmap(request, intent, candidates)
        logger.info(f"[로드맵] 생성 완료 — {roadmap.totalWeeks}주 / userId={request.userId}")
        return roadmap
    except TimeoutError:
        raise HTTPException(status_code=504, detail={
            "code": "LLM_TIMEOUT",
            "message": "LLM 응답 시간이 초과되었습니다.",
        })
    except Exception as e:
        logger.error(f"[로드맵] 생성 실패: {e}")
        raise HTTPException(status_code=502, detail={
            "code": "LLM_API_ERROR",
            "message": "로드맵 생성 중 오류가 발생했습니다.",
        })


@router.post("/ai/roadmap/reschedule", response_model=RescheduleResponse)
async def reschedule(request: RescheduleRequest):
    try:
        result = reschedule_roadmap(request)
        logger.info(f"[로드맵] 재조정 완료 — {len(result.weeks)}주 / userId={request.userId}")
        return result
    except TimeoutError:
        raise HTTPException(status_code=504, detail={
            "code": "LLM_TIMEOUT",
            "message": "LLM 응답 시간이 초과되었습니다.",
        })
    except Exception as e:
        logger.error(f"[로드맵] 재조정 실패: {e}")
        raise HTTPException(status_code=502, detail={
            "code": "LLM_API_ERROR",
            "message": "로드맵 재조정 중 오류가 발생했습니다.",
        })
