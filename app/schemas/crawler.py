from typing import Literal, Optional

from pydantic import BaseModel


CrawlerPlatform = Literal["K-MOOC", "KOCW", "LLL_PORTAL", "SEOUL_LLL", "ALL"]
CrawlerJobStatus = Literal["IN_PROGRESS", "COMPLETED", "FAILED"]


class CrawlerTriggerRequest(BaseModel):
    platform: CrawlerPlatform = "ALL"
    incremental: bool = True


class CrawlerTriggerResponse(BaseModel):
    jobId: str
    platform: CrawlerPlatform
    status: Literal["STARTED"]
    startedAt: str


class CrawlerProgress(BaseModel):
    totalExpected: int
    crawled: int
    newCourses: int
    updatedCourses: int
    failed: int
    percentComplete: float


class CrawlerStatusResponse(BaseModel):
    jobId: str
    platform: CrawlerPlatform
    status: CrawlerJobStatus
    progress: CrawlerProgress
    startedAt: str
    estimatedCompletionAt: Optional[str] = None
