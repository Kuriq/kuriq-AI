import json
import time
import logging
from typing import List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from app.core.config import settings
from app.schemas.roadmap import (
    RoadmapRequest, RoadmapResponse,
    ExtractedIntent, WeekPlan, CourseInWeek, RoadmapMetadata,
)
from app.schemas.course import CourseResult

logger = logging.getLogger(__name__)

# 1단계: 의도 추출 프롬프트 

INTENT_SYSTEM = """당신은 학습자의 목표를 분석하는 전문가입니다.
사용자의 자연어 입력에서 다음 정보를 JSON으로 추출하세요.
반드시 JSON만 출력하세요. 설명이나 마크다운 없이.

출력 형식:
{
  "interestArea": "관심 분야 (한두 단어)",
  "currentLevel": "입문 | 초급 | 중급 | 심화",
  "goal": "학습 목표 (한 문장)",
  "weeklyHours": 숫자,
  "durationPreference": "short | medium | long"
}

durationPreference 기준:
- short: ~4주
- medium: 4~12주
- long: 12주+
"""

# 로드맵 생성 프롬프트 

ROADMAP_SYSTEM = """당신은 공공 교육 커리큘럼 설계 전문가입니다.
사용자 프로필과 후보 강좌 목록을 바탕으로 주간 학습 로드맵을 JSON으로 설계하세요.
반드시 JSON만 출력하세요. 설명이나 마크다운 없이.

규칙:
1. 반드시 제공된 후보 강좌 목록 내에서만 추천하세요. 없는 강좌를 만들지 마세요.
2. 난이도 순서로 배열하세요 (입문 → 초급 → 중급 → 심화).
3. 선수 지식이 필요한 강좌는 기초 강좌 이후에 배치하세요.
4. 주당 총 학습 시간이 weeklyHours를 크게 벗어나지 않도록 조정하세요.
5. 각 주차마다 학습 흐름 설명을 한 줄로 작성하세요.

출력 형식:
{
  "goal": "목표 요약 (한 문장)",
  "totalWeeks": 숫자,
  "weeks": [
    {
      "weekNumber": 1,
      "title": "주차 제목",
      "description": "이번 주 학습 흐름 설명",
      "totalHours": 5.0,
      "courses": [
        {"courseId": "실제 강좌 ID", "orderInWeek": 1},
        {"courseId": "실제 강좌 ID", "orderInWeek": 2}
      ]
    }
  ]
}
"""

ROADMAP_USER_TEMPLATE = """사용자 프로필:
- 관심 분야: {interestArea}
- 현재 수준: {currentLevel}
- 목표: {goal}
- 주당 학습 시간: {weeklyHours}시간

후보 강좌 목록 (이 목록 내에서만 추천):
{candidate_courses}

위 정보를 바탕으로 주간 학습 로드맵을 JSON으로 생성해주세요."""


def _get_llm(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout,
        temperature=temperature,
    )


def _call_llm(messages: list, llm: ChatOpenAI, retries: int = 2) -> str:
    """LLM 호출 — 실패 시 retries 횟수만큼 재시도"""
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            last_error = e
            logger.warning(f"LLM 호출 실패 (시도 {attempt + 1}/{retries + 1}): {e}")
    raise last_error


def extract_intent(prompt: str) -> ExtractedIntent:
    """1단계: 사용자 입력 → 의도 추출"""
    llm = _get_llm(temperature=0.1)
    messages = [
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=f'사용자 입력: "{prompt}"'),
    ]
    raw = _call_llm(messages, llm)
    data = json.loads(raw)
    return ExtractedIntent(**data)


def generate_roadmap(
    request: RoadmapRequest,
    intent: ExtractedIntent,
    candidates: List[CourseResult],
) -> RoadmapResponse:
    """2단계: 의도 + 후보 강좌 → 로드맵 생성"""
    start = time.time()

    # 재생성 시 temperature 높이고 제외 강좌 필터링
    temperature = 0.1
    if request.regeneration:
        temperature += 0.3
        exclude = set(request.excludeCourseIds or [])
        candidates = [c for c in candidates if c.course_id not in exclude]

    candidate_text = json.dumps([
        {
            "courseId": c.course_id,
            "title": c.title,
            "institution": c.institution,
            "category": c.category,
            "duration": c.duration,
        }
        for c in candidates
    ], ensure_ascii=False, indent=2)

    llm = _get_llm(temperature=temperature)
    user_msg = ROADMAP_USER_TEMPLATE.format(
        interestArea=intent.interestArea,
        currentLevel=intent.currentLevel,
        goal=intent.goal,
        weeklyHours=intent.weeklyHours,
        candidate_courses=candidate_text,
    )
    messages = [
        SystemMessage(content=ROADMAP_SYSTEM),
        HumanMessage(content=user_msg),
    ]

    raw = _call_llm(messages, llm)
    data = json.loads(raw)

    # courseId 실존 여부 검증
    valid_ids = {c.course_id for c in candidates}
    weeks = []
    for w in data.get("weeks", []):
        valid_courses = [
            CourseInWeek(courseId=c["courseId"], orderInWeek=c["orderInWeek"])
            for c in w.get("courses", [])
            if c["courseId"] in valid_ids
        ]
        if valid_courses:
            weeks.append(WeekPlan(
                weekNumber=w["weekNumber"],
                title=w["title"],
                description=w["description"],
                totalHours=w.get("totalHours", intent.weeklyHours),
                courses=valid_courses,
            ))

    elapsed = round((time.time() - start) * 1000)

    return RoadmapResponse(
        goal=data.get("goal", intent.goal),
        totalWeeks=len(weeks),
        weeklyHours=intent.weeklyHours,
        weeks=weeks,
        metadata=RoadmapMetadata(
            extractedIntent=intent,
            candidatePoolSize=len(candidates),
            llmCallCount=2,
            processingTimeMs=elapsed,
        ),
    )