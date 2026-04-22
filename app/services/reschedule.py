import time
import json
import logging

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from app.core.config import settings
from app.schemas.roadmap import (
    RescheduleRequest, RescheduleResponse, RescheduleMetadata,
    WeekPlan, CourseInWeek,
)

logger = logging.getLogger(__name__)

RESCHEDULE_SYSTEM = """당신은 학습 로드맵 일정 재조정 전문가입니다.
사용자가 제공한 남은 강좌 목록을 새 일정 조건에 맞게 주차별로 재배분하세요.

규칙:
1. 모든 강좌는 반드시 한 주차에 배정되어야 합니다. 누락 없이 전부 포함하세요.
2. 각 주차의 totalHours는 해당 주에 배정된 강좌들의 estimatedHours 합계여야 합니다.
3. 난이도가 낮은 강좌를 앞 주차에 배치하고 점진적으로 난이도를 높이세요.
4. 각 주차의 title과 description은 해당 주 강좌 내용을 반영해 한국어로 작성하세요.
5. 반드시 아래 JSON 배열만 출력하세요. 설명이나 마크다운 없이.

[
  {
    "weekNumber": 1,
    "title": "주차 제목",
    "description": "이 주차에서 학습할 내용 요약 (1~2문장)",
    "totalHours": 3.0,
    "courses": [
      {"courseId": "uuid", "orderInWeek": 1}
    ]
  }
]"""


def reschedule_roadmap(request: RescheduleRequest) -> RescheduleResponse:
    start = time.monotonic()

    schedule_constraints = _build_constraint_text(request)
    courses_text = "\n".join(
        f"- courseId: {c.courseId}, estimatedHours: {c.estimatedHours}, difficulty: {c.difficulty}"
        for c in request.remainingCourses
    )

    user_msg = (
        f"일정 조건:\n{schedule_constraints}\n\n"
        f"남은 강좌 목록 ({len(request.remainingCourses)}개):\n{courses_text}"
    )

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout,
        temperature=0.3,
    )
    response = llm.invoke([
        SystemMessage(content=RESCHEDULE_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    raw_weeks: list[dict] = json.loads(response.content.strip())

    weeks = [
        WeekPlan(
            weekNumber=w["weekNumber"],
            title=w["title"],
            description=w["description"],
            totalHours=w["totalHours"],
            courses=[CourseInWeek(**c) for c in w["courses"]],
        )
        for w in raw_weeks
    ]

    elapsed = round((time.monotonic() - start) * 1000)
    logger.info(
        f"[Reschedule] userId={request.userId} weeks={len(weeks)} {elapsed}ms"
    )

    return RescheduleResponse(
        weeks=weeks,
        metadata=RescheduleMetadata(llmCallCount=1, processingTimeMs=elapsed),
    )


def _build_constraint_text(request: RescheduleRequest) -> str:
    parts = []
    if request.newWeeklyHours is not None:
        parts.append(f"주당 학습 시간: {request.newWeeklyHours}시간")
    if request.newTotalWeeks is not None:
        parts.append(f"전체 기간: {request.newTotalWeeks}주")
    if not parts:
        total_hours = sum(c.estimatedHours for c in request.remainingCourses)
        parts.append(f"총 학습 시간 {total_hours}시간을 적절히 주차별로 배분하세요.")
    return "\n".join(parts)
