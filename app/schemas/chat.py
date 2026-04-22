from pydantic import BaseModel, field_validator
from typing import List, Optional

# Request 

class ChatMessage(BaseModel):
    role: str    # "user" | "assistant"
    message: str


class ChatRequest(BaseModel):
    message: str
    noteContent: Optional[str] = None
    courseTitle: str
    courseCategory: str
    courseDifficulty: str   # 입문 / 초급 / 중급 / 심화
    chatHistory: Optional[List[ChatMessage]] = []
    userId: str

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if len(v) < 1:
            raise ValueError("message는 1자 이상이어야 합니다.")
        if len(v) > 1000:
            raise ValueError("message는 1,000자 이하이어야 합니다.")
        return v

    @field_validator("chatHistory")
    @classmethod
    def limit_chat_history(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        # 최근 10턴만 유지
        return v[-10:] if v else []


# Response 

class ChatMetadata(BaseModel):
    llmCallCount: int
    processingTimeMs: int
    inputTokens: Optional[int] = None
    outputTokens: Optional[int] = None


class ChatResponse(BaseModel):
    message: str
    noteReferences: List[str]
    isOffTopic: bool
    metadata: ChatMetadata