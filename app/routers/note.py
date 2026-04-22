import logging
from fastapi import APIRouter, HTTPException
from app.schemas.note import NoteOrganizeRequest, NoteOrganizeResponse
from app.services.note import organize_note

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ai/note/organize", response_model=NoteOrganizeResponse)
async def note_organize(request: NoteOrganizeRequest):
    try:
        return organize_note(request)
    except TimeoutError:
        raise HTTPException(status_code=504, detail={
            "code": "LLM_TIMEOUT",
            "message": "LLM 응답 시간이 초과되었습니다.",
        })
    except Exception as e:
        logger.error(f"[Note] 처리 실패: {e}")
        raise HTTPException(status_code=502, detail={
            "code": "LLM_API_ERROR",
            "message": "노트 정리 중 오류가 발생했습니다.",
        })
