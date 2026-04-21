import logging
from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import chat

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ai/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        return chat(request)
    except TimeoutError:
        raise HTTPException(status_code=504, detail={
            "code": "LLM_TIMEOUT",
            "message": "LLM 응답 시간이 초과되었습니다.",
        })
    except Exception as e:
        logger.error(f"[Chat] 처리 실패: {e}")
        raise HTTPException(status_code=502, detail={
            "code": "LLM_API_ERROR",
            "message": "AI 응답 생성 중 오류가 발생했습니다.",
        })