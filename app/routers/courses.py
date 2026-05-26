import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Header
from app.schemas.course import (
    DeleteEmbeddingRequest, DeleteEmbeddingResponse,
    EmbeddingRequest, EmbeddingResponse,
    CourseSearchRequest, CourseSearchResponse,
)
from app.core.chroma import get_collection
from app.services.embedding import embed_courses

logger = logging.getLogger(__name__)
router = APIRouter()

INTERNAL_API_KEY = "dev-secret"  # 환경 변수로 관리 권장


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


@router.post("/ai/courses/search", response_model=CourseSearchResponse)
async def search_courses(
    request: CourseSearchRequest,
    x_internal_key: Optional[str] = Header(None, alias="X-Internal-Key"),
):
    """
    chromaDB 에서 강좌 검색
    - keyword: 벡터 검색에 사용
    - platform, category, difficulty: 메타데이터 필터
    """
    # 내부 API 키 검증
    if x_internal_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        collection = get_collection()
        
        # 메타데이터 필터 빌드 (실제 chromaDB 필드명 사용)
        where_filter = None
        if request.platform or request.category or request.difficulty:
            where_conditions = []
            if request.platform:
                where_conditions.append({"platform": request.platform})
            if request.category:
                where_conditions.append({"category": request.category})
            if request.difficulty:
                # chromaDB 에는 'level' 필드로 저장됨
                where_conditions.append({"level": request.difficulty})
            
            if len(where_conditions) == 1:
                where_filter = where_conditions[0]
            elif len(where_conditions) > 1:
                where_filter = {"$and": where_conditions}
        
        # 벡터 검색 (keyword 가 있으면 임베딩, 없으면 전체)
        if request.keyword and request.keyword.strip():
            # keyword 로 임베딩 생성해서 검색
            from openai import OpenAI
            from app.core.config import settings
            
            client = OpenAI(api_key=settings.openai_api_key)
            embedding_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=request.keyword,
            )
            query_embedding = embedding_response.data[0].embedding
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=100,  # 일단 많이 가져와서 필터링
                where=where_filter,
                include=["metadatas", "documents"],
            )
        else:
            # keyword 없으면 메타데이터 필터만 사용
            results = collection.get(
                where=where_filter,
                include=["metadatas"],
                limit=100,  # 기본 제한
            )
        
        # 결과 가공
        courses = []
        if request.keyword and request.keyword.strip():
            # query 결과 처리
            if results["ids"] and len(results["ids"]) > 0:
                for i, course_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    courses.append({
                        "id": course_id,
                        "title": metadata.get("title", ""),
                        "platform": metadata.get("platform", ""),
                        "institution": metadata.get("institution", metadata.get("platform", "")),
                        "category": metadata.get("category", ""),
                        "difficulty": metadata.get("level", ""),
                        "durationWeeks": 0,  # chromaDB 에 없음
                        "estimatedHours": 0,  # chromaDB 에 없음
                        "hasCertificate": False,  # chromaDB 에 없음
                        "url": metadata.get("url", ""),
                    })
        else:
            # get 결과 처리
            if results["ids"]:
                for i, course_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    courses.append({
                        "id": course_id,
                        "title": metadata.get("title", ""),
                        "platform": metadata.get("platform", ""),
                        "institution": metadata.get("institution", metadata.get("platform", "")),
                        "category": metadata.get("category", ""),
                        "difficulty": metadata.get("level", ""),
                        "durationWeeks": 0,
                        "estimatedHours": 0,
                        "hasCertificate": False,
                        "url": metadata.get("url", ""),
                    })
        
        # 페이징
        page = request.page or 0
        size = request.size or 20
        total_elements = len(courses)
        start_idx = page * size
        end_idx = start_idx + size
        paginated_courses = courses[start_idx:end_idx]
        
        return CourseSearchResponse(
            content=paginated_courses,
            totalElements=total_elements,
            totalPages=(total_elements + size - 1) // size,
            currentPage=page,
            size=size,
            hasNext=end_idx < total_elements,
        )
        
    except Exception as e:
        logger.error(f"[Search] chromaDB 검색 실패: {e}")
        raise HTTPException(status_code=500, detail={
            "code": "CHROMA_SEARCH_ERROR",
            "message": "chromaDB 검색 중 오류가 발생했습니다.",
        })
