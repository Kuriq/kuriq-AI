from pydantic import BaseModel, field_validator
from typing import List, Optional

# Request 

class RoadmapRequest(BaseModel):
    prompt: str
    userId: str
    regeneration: Optional[bool] = False
    excludeCourseIds: Optional[List[str]] = []

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError("prompt는 10자 이상이어야 합니다.")
        if len(v) > 500:
            raise ValueError("prompt는 500자 이하이어야 합니다.")
        return v


# 의도 추출 결과

class ExtractedIntent(BaseModel):
    interestArea: str
    currentLevel: str       # 입문 / 초급 / 중급 / 심화
    goal: str
    weeklyHours: float
    durationPreference: str  # short / medium / long


#  Response 

class CourseInWeek(BaseModel):
    courseId: str
    orderInWeek: int


class WeekPlan(BaseModel):
    weekNumber: int
    title: str
    description: str
    totalHours: float
    courses: List[CourseInWeek]


class RoadmapMetadata(BaseModel):
    extractedIntent: ExtractedIntent
    candidatePoolSize: int
    llmCallCount: int
    processingTimeMs: int


class RoadmapResponse(BaseModel):
    goal: str
    totalWeeks: int
    weeklyHours: float
    weeks: List[WeekPlan]
    metadata: RoadmapMetadata


# 에러 응답

class ErrorResponse(BaseModel):
    code: str
    message: str


# Reschedule

class RemainingCourse(BaseModel):
    courseId: str
    estimatedHours: float
    difficulty: str


class RescheduleRequest(BaseModel):
    remainingCourses: List[RemainingCourse]
    newWeeklyHours: Optional[int] = None
    newTotalWeeks: Optional[int] = None
    userId: str

    @field_validator("remainingCourses")
    @classmethod
    def courses_not_empty(cls, v: List[RemainingCourse]) -> List[RemainingCourse]:
        if not v:
            raise ValueError("remainingCourses는 비어 있을 수 없습니다.")
        return v

    @field_validator("userId")
    @classmethod
    def user_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("userId는 비어 있을 수 없습니다.")
        return v


class RescheduleMetadata(BaseModel):
    llmCallCount: int
    processingTimeMs: int


class RescheduleResponse(BaseModel):
    weeks: List[WeekPlan]
    metadata: RescheduleMetadata