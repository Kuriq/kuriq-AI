import logging
from typing import List
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from app.core.chroma import get_collection
from app.utils.course_metadata import canonicalize_platform, normalize_institution

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
        
        # courseId 를 ChromaDB 저장 형식으로 변환
        # 백엔드 형식: K_MOOC_12287, KOCW_abc123, 온국민평생배움터_12345
        # ChromaDB 형식: K-MOOC_12287, KOCW_abc123, 온국민평생배움터_12345
        chroma_ids = []
        requested_by_chroma_id = {}
        for course_id in request.courseIds:
            chroma_id = course_id

            if course_id.startswith("K_MOOC_"):
                chroma_id = f"K-MOOC_{course_id[len('K_MOOC_'):]}"
            elif course_id.startswith("EVERLEARNING_"):
                chroma_id = f"에버러닝_{course_id[len('EVERLEARNING_'):]}"
            elif course_id.startswith("LLL_PORTAL_"):
                chroma_id = f"온국민평생배움터_{course_id[len('LLL_PORTAL_'):]}"
            elif course_id.startswith("KOCW_"):
                chroma_id = course_id

            chroma_ids.append(chroma_id)
            requested_by_chroma_id[chroma_id] = course_id
        
        logger.info(f"[Metadata] 백엔드 ID → ChromaDB ID 변환: {request.courseIds[:3]} → {chroma_ids[:3]}")
        
        # chromaDB 에서 메타데이터 조회
        results = collection.get(
            ids=chroma_ids,
            include=["metadatas"],
        )
        
        # 결과 가공
        courses = []
        result_ids = results.get("ids", [])
        result_metadatas = results.get("metadatas", []) or []

        for i, course_id in enumerate(result_ids):
            metadata = result_metadatas[i] if i < len(result_metadatas) else {}
            
            # 원래 요청된 courseId 형식으로 반환 (PLATFORM_COURSEID)
            original_course_id = requested_by_chroma_id.get(course_id, course_id)
            
            raw_platform = metadata.get("platform", "")
            raw_institution = metadata.get("institution", "")
            platform_en = canonicalize_platform(raw_platform, raw_institution)
            
            courses.append({
                "courseId": original_course_id,
                "title": metadata.get("title", ""),
                "platform": platform_en,
                "institution": normalize_institution(raw_institution, platform_en),
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
