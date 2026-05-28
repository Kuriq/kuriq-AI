import logging
from typing import List
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from app.core.chroma import get_collection

logger = logging.getLogger(__name__)
router = APIRouter()

INTERNAL_API_KEY = "dev-secret"


class CourseMetadataRequest(BaseModel):
    courseIds: List[str]


class CourseMetadataResponse(BaseModel):
    courses: List[dict]


@router.post("/ai/courses/metadata", response_model=CourseMetadataResponse)
async def get_course_metadata(
    request: CourseMetadataRequest,
    x_internal_key: str = Header(None, alias="X-Internal-Key"),
):
    """
    chromaDB 에서 courseId 로 메타데이터 조회
    백엔드가 로드맵 생성 시 사용
    courseId 형식: PLATFORM_COURSEID (예: LLL_PORTAL_3379164)
    """
    if x_internal_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        collection = get_collection()
        
        # courseId 에서 platform prefix 제거 (LLL_PORTAL_3379164 → 3379164)
        chroma_ids = []
        for course_id in request.courseIds:
            if "_" in course_id:
                # PLATFORM_COURSEID 형식 → 실제 ChromaDB ID 추출
                chroma_id = course_id.split("_", 1)[1]
                chroma_ids.append(chroma_id)
            else:
                chroma_ids.append(course_id)
        
        # chromaDB 에서 메타데이터 조회
        results = collection.get(
            ids=chroma_ids,
            include=["metadatas"],
        )
        
        # 결과 가공
        courses = []
        for i, course_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i] if results["metadatas"] else {}
            
            # 원래 요청된 courseId 형식으로 반환 (PLATFORM_COURSEID)
            original_course_id = request.courseIds[i] if i < len(request.courseIds) else course_id
            
            courses.append({
                "courseId": original_course_id,
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
        
        logger.info(f"[Metadata] 조회 완료 — {len(courses)}개")
        return CourseMetadataResponse(courses=courses)
        
    except Exception as e:
        logger.error(f"[Metadata] chromaDB 조회 실패: {e}")
        raise HTTPException(status_code=500, detail={
            "code": "CHROMA_ERROR",
            "message": "chromaDB 조회 중 오류가 발생했습니다.",
        })
