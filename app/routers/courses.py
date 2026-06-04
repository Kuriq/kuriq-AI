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


def _platform_aliases(platform: str) -> list[str]:
    normalized = (platform or "").strip()
    if not normalized:
        return []

    alias_map = {
        "K_MOOC": ["K_MOOC", "K-MOOC"],
        "K-MOOC": ["K_MOOC", "K-MOOC"],
        "KOCW": ["KOCW"],
        "LLL_PORTAL": ["LLL_PORTAL", "ALLGO", "온국민평생배움터"],
        "ALLGO": ["LLL_PORTAL", "ALLGO", "온국민평생배움터"],
        "온국민평생배움터": ["LLL_PORTAL", "ALLGO", "온국민평생배움터"],
        "EVERLEARNING": ["EVERLEARNING", "에버러닝", "전국평생학습"],
        "에버러닝": ["EVERLEARNING", "에버러닝", "전국평생학습"],
        "전국평생학습": ["EVERLEARNING", "에버러닝", "전국평생학습"],
        "SEOUL_LLL": ["SEOUL_LLL", "서울시평생학습포털", "서울시 평생학습포털"],
        "서울시평생학습포털": ["SEOUL_LLL", "서울시평생학습포털", "서울시 평생학습포털"],
        "서울시 평생학습포털": ["SEOUL_LLL", "서울시평생학습포털", "서울시 평생학습포털"],
    }

    return alias_map.get(normalized, [normalized])


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
    - keyword: title 에서 검색 (문자열 포함 여부)
    - category: metadata category 필터
    - platform: metadata platform 필터 (기관)
    """
    # 내부 API 키 검증
    if x_internal_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        collection = get_collection()
        
        # 메타데이터 필터 빌드
        where_filter = None
        where_conditions = []
        
        # category 필터
        if request.category:
            where_conditions.append({"category": request.category})
        
        # platform 필터 (기관/플랫폼 alias 허용)
        if request.platform:
            aliases = _platform_aliases(request.platform)
            if len(aliases) == 1:
                where_conditions.append({"platform": aliases[0]})
            elif aliases:
                where_conditions.append({"$or": [{"platform": alias} for alias in aliases]})
        
        # difficulty 필터
        if request.difficulty:
            where_conditions.append({"level": request.difficulty})
        
        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
        elif len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}
        
        # 페이징 설정
        page = request.page or 0
        size = request.size or 20
        offset = page * size
        
        # ChromaDB 는 한 번에 많은 데이터를 가져올 수 없으므로 배치로 조회
        BATCH_SIZE = 5000
        all_courses = []
        keyword = request.keyword.strip() if request.keyword else ""
        
        for batch_offset in range(0, 100000, BATCH_SIZE):  # 최대 10 만개까지
            results = collection.get(
                where=where_filter,
                include=["metadatas"],
                limit=BATCH_SIZE,
                offset=batch_offset,
            )
            
            if not results["ids"]:
                break
            
            metadatas_list = results["metadatas"] if isinstance(results["metadatas"], list) else []
            
            for i, course_id in enumerate(results["ids"]):
                metadata = metadatas_list[i] if i < len(metadatas_list) else {}
                
                if isinstance(metadata, dict):
                    title = metadata.get("title", "")
                    backend_course_id = metadata.get("courseId") or course_id
                    
                    # keyword 필터링: title 에 포함 여부
                    if keyword and keyword not in title:
                        continue
                    
                    all_courses.append({
                        "id": backend_course_id,
                        "title": title,
                        "platform": metadata.get("platform", ""),
                        "institution": metadata.get("institution", metadata.get("platform", "")),
                        "category": metadata.get("category", ""),
                        "difficulty": metadata.get("level", ""),
                        "durationWeeks": 0,
                        "estimatedHours": 0,
                        "hasCertificate": False,
                        "url": metadata.get("url", ""),
                    })
            
            # 다음 배치가 필요 없는 경우 (현재 페이지의 데이터를 모두 찾음)
            if len(all_courses) > offset + size:
                # 하지만 필터/키워드가 있으면 전체 개수를 알아야 하므로 계속 조회
                if where_filter or keyword:
                    continue
                else:
                    break
        
        # 페이징 적용
        paginated_courses = all_courses[offset:offset + size]
        
        # 총 개수: 필터/키워드가 없으면 전체 count 사용, 있으면 실제 필터링된 수
        if not where_filter and not keyword:
            total_elements = collection.count()
        else:
            total_elements = len(all_courses)
        
        return CourseSearchResponse(
            content=paginated_courses,
            totalElements=total_elements,
            totalPages=(total_elements + size - 1) // size,
            currentPage=page,
            size=size,
            hasNext=len(paginated_courses) == size,
        )
        
    except Exception as e:
        logger.error(f"[Search] chromaDB 검색 실패: {e}")
        raise HTTPException(status_code=500, detail={
            "code": "CHROMA_SEARCH_ERROR",
            "message": "chromaDB 검색 중 오류가 발생했습니다.",
        })
