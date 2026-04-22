from typing import Optional
from pydantic import BaseModel


class OpenAIHealth(BaseModel):
    status: str
    latencyMs: int


class ChromaDBHealth(BaseModel):
    status: str
    vectorCount: int
    latencyMs: int


class SchedulerHealth(BaseModel):
    status: str
    nextCrawlAt: Optional[str] = None
    lastCrawlCompletedAt: Optional[str] = None


class HealthComponents(BaseModel):
    openai: OpenAIHealth
    chromadb: ChromaDBHealth
    scheduler: SchedulerHealth


class DetailedHealthResponse(BaseModel):
    status: str
    components: HealthComponents
    timestamp: str
