import json
import time
import logging
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.core.config import settings
from app.schemas.roadmap import (
    RoadmapRequest, RoadmapResponse,
    ExtractedIntent, WeekPlan, CourseInWeek, RoadmapMetadata,
)
from app.schemas.course import CourseResult

logger = logging.getLogger(__name__)


# ── 1단계: 의도 추출 프롬프트 ────────────────────────────────────────────────────

INTENT_SYSTEM = """당신은 학습자의 목표를 분석하는 전문가입니다.
사용자의 자연어 입력에서 다음 정보를 JSON으로 추출하세요.
입력이 짧거나 불명확해도 최대한 합리적으로 추론하세요.
반드시 JSON만 출력하세요. 설명이나 마크다운 없이.

출력 형식:
{
  "interestArea": "관심 분야 (한두 단어, 예: 파이썬, 영어회화, 데이터분석)",
  "currentLevel": "입문 | 초급 | 중급 | 심화",
  "goal": "학습 목표 (구체적인 한 문장, 예: 파이썬 기초 문법을 익혀 간단한 프로그램을 만들 수 있다)",
  "weeklyHours": 숫자,
  "durationPreference": "short | medium | long",
  "targetAudience": "학습자 특성 요약 (예: 직장인, 50대 중장년, 대학생, 취업준비생 등. 언급 없으면 일반 성인)"
}

추론 기준:
- 언급이 없으면 weeklyHours는 5로 설정
- 입문/처음/기초/모른다 언급 시 currentLevel은 "입문"
- durationPreference: 단기/빠르게 → short, 언급 없으면 → medium, 깊게/전문가 → long
- goal은 반드시 구체적으로 작성 (단순히 "배우고 싶다" → "X를 활용해 Y를 할 수 있다"로 구체화)
- 연령/직업/상황이 언급되면 targetAudience에 반영
"""


# ── 2단계: 로드맵 생성 프롬프트 ────────────────────────────────────────────────

ROADMAP_SYSTEM = """당신은 대한민국 공공 교육 플랫폼(K-MOOC, KOCW, 온국민평생배움터 등) 전문 커리큘럼 설계자입니다.
사용자 프로필과 후보 강좌 목록을 바탕으로 체계적이고 풍부한 주간 학습 로드맵을 JSON으로 설계하세요.
반드시 JSON만 출력하세요. 설명이나 마크다운 없이.

[필수 규칙]
1. totalWeeks는 반드시 1 이상이어야 합니다.
2. 반드시 제공된 후보 강좌 목록의 [인덱스]만 사용하세요. 목록에 없는 인덱스를 절대 만들지 마세요.
3. courseId를 직접 입력하지 말고, 인덱스 번호만 사용하세요.
4. 사용 가능한 인덱스 범위: 0부터 {max_index}까지. 이 범위를 절대 벗어나지 마세요.
5. 후보 강좌가 1개 이상이면 무조건 1주 이상 생성하세요.

[퀄리티 규칙]
6. 난이도 순서로 강좌를 배열하세요 (입문 → 초급 → 중급 → 심화).
7. 주당 총 학습 시간(totalHours)이 weeklyHours에 맞도록 강좌 수를 조절하세요.
8. 주차당 강좌는 2~4개로 구성하세요 (너무 많거나 적으면 안 됨).
9. 각 주차 title은 학습 흐름을 잘 나타내는 구체적인 제목으로 작성하세요.
   예: "파이썬 기초 문법 다지기", "데이터 시각화 실습", "머신러닝 개념 이해"
10. 각 주차 description은 반드시 3문장으로 작성하세요:
    ① 이번 주 핵심 학습 목표
    ② 이번 주에 수강할 강좌들의 특징 및 연결고리
    ③ 이번 주를 마치면 할 수 있게 되는 것 (구체적 능력)
11. 로드맵 goal은 전체 수료 후 달성 가능한 능력을 2~3문장으로 구체적으로 서술하세요.
12. 학습자 특성(나이, 직업, 상황)을 반영해 강좌 선택과 설명 톤을 조절하세요.
13. 로드맵 title은 학습자의 목표와 수준을 반영한 구체적인 제목으로 작성하세요.
    예: "50대를 위한 디지털 역량 강화 로드맵", "비전공자를 위한 파이썬 입문 로드맵"

출력 형식:
{{
  "title": "구체적인 로드맵 제목",
  "goal": "이 로드맵을 완료하면 [구체적 능력]을 갖출 수 있습니다. 2~3문장으로.",
  "totalWeeks": 숫자,
  "weeks": [
    {{
      "weekNumber": 1,
      "title": "구체적인 주차 제목",
      "description": "① 이번 주 목표 문장. ② 강좌 소개 문장. ③ 학습 후 얻을 능력 문장.",
      "totalHours": 5.0,
      "courses": [
        {{"courseIndex": 0, "orderInWeek": 1}},
        {{"courseIndex": 3, "orderInWeek": 2}}
      ]
    }}
  ]
}}
"""

ROADMAP_USER_TEMPLATE = """사용자 프로필:
- 관심 분야: {interestArea}
- 현재 수준: {currentLevel}
- 학습 목표: {goal}
- 주당 학습 시간: {weeklyHours}시간
- 학습자 특성: {targetAudience}
- 원본 입력: "{original_prompt}"

후보 강좌 목록 (이 목록 내에서만 추천):
{candidate_courses}

위 정보를 바탕으로 체계적이고 풍부한 주간 학습 로드맵을 JSON으로 생성해주세요.
학습자 특성과 원본 입력을 반드시 반영하고, 각 주차 description은 반드시 3문장으로 작성하세요."""


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


def _clean_json(raw: str) -> str:
    """LLM이 마크다운 코드블록으로 감싸서 반환할 경우 제거"""
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def extract_intent(prompt: str) -> ExtractedIntent:
    """1단계: 사용자 입력 → 의도 추출"""
    llm = _get_llm(temperature=0.1)
    messages = [
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=f'사용자 입력: "{prompt}"'),
    ]
    raw = _call_llm(messages, llm)
    data = json.loads(_clean_json(raw))

    # targetAudience가 없는 경우 기본값 설정 (기존 스키마 호환)
    data.setdefault("targetAudience", "일반 성인")

    return ExtractedIntent(**data)


def generate_roadmap(
    request: RoadmapRequest,
    intent: ExtractedIntent,
    candidates: List[CourseResult],
) -> RoadmapResponse:
    """2단계: 의도 + 후보 강좌 → 로드맵 생성"""
    start = time.time()

    # weeklyHours가 0이면 기본값 5시간으로 설정
    weekly_hours = intent.weeklyHours if intent.weeklyHours and intent.weeklyHours > 0 else 5.0

    # 재생성 시 temperature 높이고 제외 강좌 필터링
    temperature = 0.3
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            current_temp = temperature + (attempt * 0.2)

            # 재생성 요청 시: temperature 높이고 이전에 사용된 강좌 제외
            if request.regeneration:
                current_temp += 0.2
                exclude = set(request.excludeCourseIds or [])
                candidates = [c for c in candidates if c.course_id not in exclude]

            # 후보 강좌 텍스트 — 난이도·기간 포함해서 AI가 더 잘 판단하도록
            candidate_text = "\n".join([
                f"[{i}] {c.title} | {c.institution} | {c.category} | 난이도: {c.difficulty or '미정'} | 기간: {c.duration_weeks or 0}주"
                for i, c in enumerate(candidates)
            ])

            llm = _get_llm(temperature=current_temp)

            # targetAudience: 기존 intent에 없을 경우 기본값
            target_audience = getattr(intent, "targetAudience", "일반 성인") or "일반 성인"

            user_msg = ROADMAP_USER_TEMPLATE.format(
                interestArea=intent.interestArea,
                currentLevel=intent.currentLevel,
                goal=intent.goal,
                weeklyHours=weekly_hours,
                targetAudience=target_audience,
                original_prompt=request.prompt,  # 원본 입력 그대로 전달
                candidate_courses=candidate_text,
            )
            roadmap_system = ROADMAP_SYSTEM.format(max_index=len(candidates) - 1)
            messages = [
                SystemMessage(content=roadmap_system),
                HumanMessage(content=user_msg),
            ]

            raw = _call_llm(messages, llm)
            logger.info(f"[LLM] raw 응답 (시도 {attempt + 1}/{max_retries + 1}): {raw[:500]}...")

            try:
                data = json.loads(_clean_json(raw))
            except json.JSONDecodeError as e:
                logger.error(f"[LLM] JSON 파싱 실패: {e}")
                if attempt < max_retries:
                    continue
                raise

            logger.info(f"[LLM] 파싱된 data keys: {list(data.keys())}")

            # courseIndex → courseId 변환, 범위 벗어난 인덱스 제거
            valid_indices = set(range(len(candidates)))
            weeks = []
            total_valid_courses = 0
            all_requested_indices = []

            for w in data.get("weeks", []):
                valid_courses = []
                for c in w.get("courses", []):
                    idx = c.get("courseIndex")
                    all_requested_indices.append(idx)
                    if idx is not None and idx in valid_indices:
                        valid_courses.append(CourseInWeek(
                            courseId=candidates[idx].course_id,
                            orderInWeek=c["orderInWeek"]
                        ))
                    elif idx is not None:
                        logger.warning(f"[LLM] 범위 벗어난 courseIndex={idx} 무시 (유효: 0-{len(candidates)-1})")

                if valid_courses:
                    weeks.append(WeekPlan(
                        weekNumber=w["weekNumber"],
                        title=w["title"],
                        description=w["description"],
                        totalHours=w.get("totalHours", weekly_hours),
                        courses=valid_courses,
                    ))
                    total_valid_courses += len(valid_courses)
                else:
                    logger.warning(f"[LLM] {w.get('weekNumber')}주차에 유효한 강좌 없음 — 해당 주차 제외")

            # 유효한 강좌가 하나도 없으면 재시도
            if total_valid_courses == 0:
                if attempt < max_retries:
                    logger.warning(f"[LLM] 유효한 강좌 없음 — 재시도 ({attempt + 1}/{max_retries})")
                    continue

                # 최대 재시도 후에도 실패 시 첫 번째 강좌로 강제 배정 (fallback)
                logger.error(f"[LLM] hallucination 발생 — 강제 fallback. 요청 인덱스: {all_requested_indices}")
                if candidates:
                    weeks = [WeekPlan(
                        weekNumber=1,
                        title="학습 시작",
                        description="선택된 강좌로 학습을 시작합니다. 기초부터 차근차근 진행해보세요. 첫 번째 강좌를 완료하면 다음 단계로 나아갈 준비가 됩니다.",
                        totalHours=weekly_hours,
                        courses=[CourseInWeek(courseId=candidates[0].course_id, orderInWeek=1)],
                    )]
                    total_valid_courses = 1
                else:
                    raise ValueError("후보 강좌가 없어 로드맵을 생성할 수 없습니다.")

            if total_valid_courses < len(candidates) * 0.3:
                logger.warning(f"[LLM] 사용된 강좌 {total_valid_courses}개 / 후보 {len(candidates)}개 — 30% 미만")

            elapsed = round((time.time() - start) * 1000)

            # title이 없으면 fallback 제목 생성
            roadmap_title = data.get("title") or _build_fallback_title(intent, data.get("goal", intent.goal))

            return RoadmapResponse(
                title=roadmap_title,
                goal=data.get("goal", intent.goal),
                totalWeeks=len(weeks),
                weeklyHours=weekly_hours,
                weeks=weeks,
                metadata=RoadmapMetadata(
                    extractedIntent=intent,
                    candidatePoolSize=len(candidates),
                    llmCallCount=2,
                    processingTimeMs=elapsed,
                ),
            )

        except Exception as e:
            if attempt >= max_retries:
                raise
            logger.warning(f"[LLM] 로드맵 생성 실패 (시도 {attempt + 1}/{max_retries + 1}): {e}")
            continue

    raise ValueError("LLM 로드맵 생성 최대 재시도 횟수 초과")


def _build_fallback_title(intent: ExtractedIntent, goal: str) -> str:
    """title이 없을 때 intent 정보로 fallback 제목 생성"""
    source = f"{intent.interestArea} {goal}".lower()
    target = getattr(intent, "targetAudience", "") or ""

    subject = "맞춤 학습"
    if any(k in source for k in ["ai", "인공지능", "머신러닝", "딥러닝"]):
        subject = "AI·머신러닝"
    elif any(k in source for k in ["데이터", "분석", "통계"]):
        subject = "데이터 분석"
    elif "영어" in source:
        subject = "영어"
    elif "일본어" in source:
        subject = "일본어"
    elif "중국어" in source:
        subject = "중국어"
    elif "파이썬" in source:
        subject = "파이썬"
    elif any(k in source for k in ["프로그래밍", "코딩", "개발"]):
        subject = "프로그래밍"
    elif any(k in source for k in ["디자인", "ui", "ux"]):
        subject = "디자인"
    elif "마케팅" in source:
        subject = "마케팅"
    elif any(k in source for k in ["경영", "비즈니스"]):
        subject = "경영"

    level = intent.currentLevel.strip() if intent.currentLevel else ""

    # 학습자 특성이 있으면 제목에 반영
    if target and target != "일반 성인":
        return f"{target}을 위한 {subject} {level} 로드맵".strip()
    return f"{subject} {level} 로드맵".strip()
