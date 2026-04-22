from typing import Optional
from pydantic import BaseModel, field_validator


class CourseResult(BaseModel):
    course_id: str
    title: str
    institution: str
    category: str
    duration: str


# 임베딩 생성 

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


# ── 임베딩 삭제 ────────────────────────────────────────────────

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
