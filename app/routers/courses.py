import logging
from fastapi import APIRouter, HTTPException
from app.schemas.course import (
    DeleteEmbeddingRequest, DeleteEmbeddingResponse,
    EmbeddingRequest, EmbeddingResponse,
)
from app.core.chroma import get_collection
from app.services.embedding import embed_courses

logger = logging.getLogger(__name__)
router = APIRouter()


@router.delete("/ai/embedding/course", response_model=DeleteEmbeddingResponse)
async def delete_course_embeddings(request: DeleteEmbeddingRequest):
    try:
        collection = get_collection()
        existing = collection.get(ids=request.courseIds, include=[])
        found_ids = existing["ids"]
    except Exception as e:
        logger.error(f"[Embedding] ChromaDB 조회 실패: {e}")
        raise HTTPException(status_code=500, detail={
            "code": "CHROMA_ERROR",
            "message": "ChromaDB 조회 중 오류가 발생했습니다.",
        })

    not_found = len(request.courseIds) - len(found_ids)

    if found_ids:
        try:
            collection.delete(ids=found_ids)
            logger.info(f"[Embedding] 삭제 완료 — {len(found_ids)}건, notFound={not_found}")
        except Exception as e:
            logger.error(f"[Embedding] ChromaDB 삭제 실패: {e}")
            raise HTTPException(status_code=500, detail={
                "code": "CHROMA_ERROR",
                "message": "ChromaDB 삭제 중 오류가 발생했습니다.",
            })

    return DeleteEmbeddingResponse(deleted=len(found_ids), notFound=not_found)


@router.post("/ai/embedding/course", response_model=EmbeddingResponse)
async def create_course_embeddings(request: EmbeddingRequest):
    try:
        result = await embed_courses(request)
        logger.info(
            f"[Embedding] 생성 완료 — processed={result.processed}, "
            f"succeeded={result.succeeded}, failed={result.failed}"
        )
        return result
    except Exception as e:
        logger.error(f"[Embedding] 처리 실패: {e}")
        raise HTTPException(status_code=500, detail={
            "code": "EMBEDDING_ERROR",
            "message": "임베딩 생성 중 오류가 발생했습니다.",
        })
