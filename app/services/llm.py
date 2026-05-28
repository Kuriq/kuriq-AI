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
사용자 프로필과 후보 강좌 목록을 바탕으로 주간 학습 로드맵을 JSON 으로 설계하세요.
반드시 JSON 만 출력하세요. 설명이나 마크다운 없이.

중요 규칙:
1. **반드시 1 주 이상 생성하세요. totalWeeks 는 최소 1 이상이어야 합니다.**
2. **반드시 제공된 후보 강좌 목록의 [인덱스] 만 사용하세요.**
3. **목록에 없는 인덱스를 절대 만들지 마세요.**
4. **courseId 를 직접 입력하지 말고, 인덱스 번호만 사용하세요.**
5. 난이도 순서로 배열하세요 (입문 → 초급 → 중급 → 심화).
6. 주당 총 학습 시간이 weeklyHours 를 크게 벗어나지 않도록 조정하세요.
7. **사용 가능한 인덱스 범위: 0 부터 {max_index} 까지입니다. 이 범위를 절대 벗어나지 마세요.**
8. **후보 강좌가 1 개 이상이면 무조건 1 주 이상 생성하세요. 빈 로드맵은 허용되지 않습니다.**

출력 형식:
{{
  "goal": "목표 요약 (한 문장)",
  "totalWeeks": 숫자,  // 최소 1 이상
  "weeks": [
    {{
      "weekNumber": 1,
      "title": "주차 제목",
      "description": "이번 주 학습 흐름 설명",
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
    """2 단계: 의도 + 후보 강좌 → 로드맵 생성"""
    start = time.time()

    # 재생성 시 temperature 높이고 제외 강좌 필터링
    temperature = 0.3  # 기본 temperature 상향 (0.1 → 0.3)
    max_retries = 2
    
    for attempt in range(max_retries + 1):
        try:
            current_temp = temperature + (attempt * 0.2)  # 재시도할 때마다 temperature 증가
            
            if request.regeneration:
                current_temp += 0.2
                exclude = set(request.excludeCourseIds or [])
                candidates = [c for c in candidates if c.course_id not in exclude]

            candidate_text = "\n".join([
                f"[{i}] {c.course_id} | {c.title} | {c.institution} | {c.category}"
                for i, c in enumerate(candidates)
            ])

            llm = _get_llm(temperature=current_temp)
            user_msg = ROADMAP_USER_TEMPLATE.format(
                interestArea=intent.interestArea,
                currentLevel=intent.currentLevel,
                goal=intent.goal,
                weeklyHours=intent.weeklyHours,
                candidate_courses=candidate_text,
            )
            # 프롬프트에 인덱스 범위 명시
            roadmap_system = ROADMAP_SYSTEM.format(max_index=len(candidates) - 1)
            messages = [
                SystemMessage(content=roadmap_system),
                HumanMessage(content=user_msg),
            ]

            raw = _call_llm(messages, llm)
            logger.info(f"[LLM] raw 응답 (시도 {attempt + 1}/{max_retries + 1}): {raw[:500]}...")
            
            # 마크다운 코드 블록 제거
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.error(f"[LLM] JSON 파싱 실패: {e}")
                if attempt < max_retries:
                    continue
                raise
            
            logger.info(f"[LLM] 파싱된 data: {data}")

            # courseIndex 를 courseId 로 변환
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
                        logger.warning(f"[LLM] 범위를 벗어난 courseIndex={idx} 무시 (유효 범위: 0-{len(candidates)-1})")
                
                if valid_courses:
                    weeks.append(WeekPlan(
                        weekNumber=w["weekNumber"],
                        title=w["title"],
                        description=w["description"],
                        totalHours=w.get("totalHours", intent.weeklyHours),
                        courses=valid_courses,
                    ))
                    total_valid_courses += len(valid_courses)
                else:
                    logger.warning(f"[LLM] 주차 {w.get('weekNumber')} 에 유효한 강좌가 없음")
            
            # 유효한 강좌가 없으면 재시도
            if total_valid_courses == 0:
                if attempt < max_retries:
                    logger.warning(f"[LLM] 유효한 강좌 없음 — {attempt + 1}번째 시도 실패, 재시도...")
                    continue
                    
                logger.error(f"[LLM] 모든 주차에 유효한 강좌가 없음 — LLM hallucination 발생")
                logger.error(f"[LLM] LLM 이 반환한 courseIndex: {all_requested_indices}")
                logger.error(f"[LLM] 실제 유효한 인덱스: 0-{len(candidates)-1}")
                raise ValueError("LLM 이 존재하지 않는 강좌 인덱스를 반환했습니다.")
            
            if total_valid_courses < len(candidates) * 0.3:
                logger.warning(f"[LLM] 유효한 강좌가 {total_valid_courses}개로 적음 ({len(candidates)}개 중)")

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
            
        except Exception as e:
            if attempt >= max_retries:
                raise
            logger.warning(f"[LLM] 로드맵 생성 실패 (시도 {attempt + 1}/{max_retries + 1}): {e}")
            continue
    
    # 모든 재시도 실패
    raise ValueError("LLM 로드맵 생성 최대 재시도 횟수 초과")
    # 프롬프트에 인덱스 범위 명시
    roadmap_system = ROADMAP_SYSTEM.format(max_index=len(candidates) - 1)
    messages = [
        SystemMessage(content=roadmap_system),
        HumanMessage(content=user_msg),
    ]

    raw = _call_llm(messages, llm)
    logger.info(f"[LLM] raw 응답: {raw[:500]}...")  # 앞 500 자만 로그
    
    # 마크다운 코드 블록 제거 (LLM 이 ```json ... ``` 으로 반환할 수 있음)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"[LLM] JSON 파싱 실패: {e}")
        logger.error(f"[LLM] raw 응답: {raw}")
        raise
    
    logger.info(f"[LLM] 파싱된 data: {data}")

    # courseIndex 를 courseId 로 변환
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
                # 범위를 벗어난 인덱스는 로그만 하고 무시
                logger.warning(f"[LLM] 범위를 벗어난 courseIndex={idx} 무시 (유효 범위: 0-{len(candidates)-1})")
        
        if valid_courses:
            weeks.append(WeekPlan(
                weekNumber=w["weekNumber"],
                title=w["title"],
                description=w["description"],
                totalHours=w.get("totalHours", intent.weeklyHours),
                courses=valid_courses,
            ))
            total_valid_courses += len(valid_courses)
        else:
            logger.warning(f"[LLM] 주차 {w.get('weekNumber')} 에 유효한 강좌가 없음")
    
    # 유효한 강좌가 너무 적으면 에러
    if total_valid_courses == 0:
        logger.error(f"[LLM] 모든 주차에 유효한 강좌가 없음 — LLM hallucination 발생")
        logger.error(f"[LLM] LLM 이 반환한 courseIndex: {all_requested_indices}")
        logger.error(f"[LLM] 실제 유효한 인덱스: 0-{len(candidates)-1}")
        logger.error(f"[LLM] 후보 강좌 목록: {[c.course_id for c in candidates]}")
        raise ValueError("LLM 이 존재하지 않는 강좌 인덱스를 반환했습니다.")
    
    if total_valid_courses < len(candidates) * 0.3:  # 30% 미만이면 경고
        logger.warning(f"[LLM] 유효한 강좌가 {total_valid_courses}개로 적음 ({len(candidates)}개 중)")

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