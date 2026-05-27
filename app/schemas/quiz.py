from typing import Optional, Union
from pydantic import BaseModel, field_validator


class QuizGradeRequest(BaseModel):
    question: str
    correctAnswer: str
    acceptableKeywords: Optional[list[str]] = None
    userAnswer: str
    userId: str

    @field_validator("userAnswer", "question", "correctAnswer")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("값이 비어 있을 수 없습니다.")
        return v


class QuizGradeMetadata(BaseModel):
    matchedKeyword: Optional[str]
    llmCallCount: int
    processingTimeMs: int


class QuizGradeResponse(BaseModel):
    result: str  # CORRECT | PARTIAL | WRONG
    feedback: str
    correctAnswer: str
    metadata: QuizGradeMetadata


class QuizGenerateRequest(BaseModel):
    noteContent: str
    courseTitle: str
    courseDifficulty: str = "초급"
    excludeQuestions: Optional[list[str]] = None
    questionCount: int = 5
    userId: str

    @field_validator("noteContent")
    @classmethod
    def note_min_length(cls, v: str) -> str:
        if len(v.strip()) < 50:
            raise ValueError("noteContent 는 최소 50 자 이상이어야 합니다.")
        return v

    @field_validator("courseTitle", "userId")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("값이 비어 있을 수 없습니다.")
        return v


class QuizOption(BaseModel):
    id: str
    text: str


class QuizQuestion(BaseModel):
    questionId: str
    type: str  # MULTIPLE_CHOICE | TRUE_FALSE | SHORT_ANSWER
    question: str
    options: Optional[list[QuizOption]]
    correctAnswer: Union[str, bool]
    acceptableKeywords: Optional[list[str]] = None
    explanation: str
    noteReference: str
    topic: str


class QuizGenerateMetadata(BaseModel):
    questionsGenerated: int
    llmCallCount: int
    processingTimeMs: int


class QuizGenerateResponse(BaseModel):
    questions: list[QuizQuestion]
    metadata: QuizGenerateMetadata
