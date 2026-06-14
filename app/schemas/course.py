from typing import Optional
from pydantic import BaseModel, field_validator


# ── 강좌 검색 결과 (RAG 파이프라인 내부용) ────────────────────────────────────

class CourseResult(BaseModel):
    course_id: str
    title: str
    institution: str
    category: str
    duration: str
    difficulty: Optional[str] = None      # 난이도 (입문/초급/중급/심화)
    duration_weeks: Optional[int] = None  # 수강 기간 (주 단위)


# ── 임베딩 생성 ────────────────────────────────────────────────────────────────

class CourseMetadata(BaseModel):
    platform: str
    difficulty: str
    category: str
    durationWeeks: int
    hasCertificate: bool
    isActive: bool


class CourseEmbeddingItem(BaseModel):
    courseId: str
    title: str
    description: Optional[str] = None
    keywords: Optional[list[str]] = None
    metadata: CourseMetadata


class EmbeddingRequest(BaseModel):
    courses: list[CourseEmbeddingItem]

    @field_validator("courses")
    @classmethod
    def must_not_be_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("courses must not be empty")
        return v


class EmbeddingMetadata(BaseModel):
    embeddingCallCount: int
    processingTimeMs: int


class EmbeddingResponse(BaseModel):
    processed: int
    succeeded: int
    failed: int
    failedCourseIds: list[str]
    metadata: EmbeddingMetadata


# ── 임베딩 삭제 ────────────────────────────────────────────────────────────────

class DeleteEmbeddingRequest(BaseModel):
    courseIds: list[str]

    @field_validator("courseIds")
    @classmethod
    def ids_must_not_be_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("courseIds must not be empty")
        return v


class DeleteEmbeddingResponse(BaseModel):
    deleted: int
    notFound: int


# ── 강좌 검색 (검색 API용) ─────────────────────────────────────────────────────

class CourseSearchRequest(BaseModel):
    keyword: Optional[str] = None
    platform: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    page: Optional[int] = 0
    size: Optional[int] = 20


class CourseSearchResult(BaseModel):
    id: str
    title: str
    platform: str
    institution: str
    category: str
    difficulty: str
    durationWeeks: int
    estimatedHours: float
    hasCertificate: bool
    url: str


class CourseSearchResponse(BaseModel):
    content: list[CourseSearchResult]
    totalElements: int
    totalPages: int
    currentPage: int
    size: int
    hasNext: bool
