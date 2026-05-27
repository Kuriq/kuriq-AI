from typing import Literal
from pydantic import BaseModel, field_validator

VALID_DIFFICULTIES = {"입문", "초급", "중급", "심화"}


class NoteOrganizeRequest(BaseModel):
    noteContent: str
    courseTitle: str
    courseCategory: str
    userId: str

    @field_validator("noteContent")
    @classmethod
    def min_length(cls, v: str) -> str:
        if len(v.strip()) < 50:
            raise ValueError("noteContent 는 최소 50 자 이상이어야 합니다.")
        return v

    @field_validator("courseTitle", "courseCategory", "userId")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("값이 비어 있을 수 없습니다.")
        return v

    @field_validator("courseTitle", "courseCategory", "userId")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("값이 비어 있을 수 없습니다.")
        return v


class NoteOrganizeMetadata(BaseModel):
    llmCallCount: int
    processingTimeMs: int


class NoteOrganizeResponse(BaseModel):
    keywords: list[str]
    structuredSummary: str
    suggestions: list[str]
    metadata: NoteOrganizeMetadata
