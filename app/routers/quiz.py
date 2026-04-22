import logging
from fastapi import APIRouter, HTTPException
from app.schemas.quiz import QuizGradeRequest, QuizGradeResponse, QuizGenerateRequest, QuizGenerateResponse
from app.services.quiz import grade_quiz, generate_quiz

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ai/quiz/grade", response_model=QuizGradeResponse)
async def grade(request: QuizGradeRequest):
    try:
        return grade_quiz(request)
    except TimeoutError:
        raise HTTPException(status_code=504, detail={
            "code": "LLM_TIMEOUT",
            "message": "LLM 응답 시간이 초과되었습니다.",
        })
    except Exception as e:
        logger.error(f"[Quiz] 처리 실패: {e}")
        raise HTTPException(status_code=502, detail={
            "code": "LLM_API_ERROR",
            "message": "퀴즈 채점 중 오류가 발생했습니다.",
        })


@router.post("/ai/quiz/generate", response_model=QuizGenerateResponse)
async def generate(request: QuizGenerateRequest):
    try:
        return generate_quiz(request)
    except TimeoutError:
        raise HTTPException(status_code=504, detail={
            "code": "LLM_TIMEOUT",
            "message": "LLM 응답 시간이 초과되었습니다.",
        })
    except Exception as e:
        logger.error(f"[Quiz] 생성 실패: {e}")
        raise HTTPException(status_code=502, detail={
            "code": "LLM_API_ERROR",
            "message": "퀴즈 생성 중 오류가 발생했습니다.",
        })
