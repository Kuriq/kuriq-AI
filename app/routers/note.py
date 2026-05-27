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
        logger.error(f"[Note] Request data: {request}")
        raise HTTPException(status_code=502, detail={
            "code": "LLM_API_ERROR",
            "message": "노트 정리 중 오류가 발생했습니다.",
        })


# Health check for validation errors
@router.post("/ai/note/organize/debug")
async def note_organize_debug(request: dict):
    logger.info(f"[Debug] Received: {request}")
    try:
        validated = NoteOrganizeRequest(
            noteContent=request.get("noteContent", ""),
            courseTitle=request.get("courseTitle", ""),
            courseCategory=request.get("courseCategory", ""),
            userId=request.get("userId", ""),
        )
        logger.info(f"[Debug] Validated: {validated}")
        return {"status": "ok", "validated": validated.model_dump()}
    except Exception as e:
        logger.error(f"[Debug] Validation error: {e}")
        return {"status": "error", "message": str(e)}
