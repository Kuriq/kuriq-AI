import time
import json
import logging

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from app.core.config import settings
from app.schemas.note import NoteOrganizeRequest, NoteOrganizeMetadata, NoteOrganizeResponse

logger = logging.getLogger(__name__)

ORGANIZE_SYSTEM = """당신은 학습 노트 정리 전문가입니다.
사용자의 학습 노트를 분석하여 아래 세 가지를 JSON으로 반환하세요.
반드시 JSON만 출력하세요. 설명이나 마크다운 없이.

출력 형식:
{
  "keywords": ["핵심 키워드 목록 (5~10개)"],
  "structuredSummary": "마크다운 형식의 구조화 요약",
  "suggestions": ["누락된 내용이나 추가 학습 제안 (2~4개)"]
}

작성 지침:
- keywords: 노트에서 핵심 개념어만 추출
- structuredSummary: 헤더(###), 굵게(**), 목록(-) 을 활용해 계층적으로 정리
- suggestions: "~해 보면 좋을 것 같아요" 형식의 친절한 어투로 작성
- 강좌 난이도에 맞게 설명 깊이를 조절"""

ORGANIZE_USER_TEMPLATE = """강좌 정보:
- 강좌명: {courseTitle}
- 분야: {courseCategory}
- 난이도: {courseDifficulty}

학습 노트:
{noteContent}"""


def organize_note(request: NoteOrganizeRequest) -> NoteOrganizeResponse:
    start = time.monotonic()

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout,
        temperature=0.3,
    )
    user_msg = ORGANIZE_USER_TEMPLATE.format(
        courseTitle=request.courseTitle,
        courseCategory=request.courseCategory,
        courseDifficulty=request.courseDifficulty,
        noteContent=request.noteContent,
    )
    response = llm.invoke([
        SystemMessage(content=ORGANIZE_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    data = json.loads(response.content.strip())
    elapsed = round((time.monotonic() - start) * 1000)

    logger.info(f"[Note] 정리 완료 — userId={request.userId} {elapsed}ms")

    return NoteOrganizeResponse(
        keywords=data["keywords"],
        structuredSummary=data["structuredSummary"],
        suggestions=data["suggestions"],
        metadata=NoteOrganizeMetadata(
            llmCallCount=1,
            processingTimeMs=elapsed,
        ),
    )
