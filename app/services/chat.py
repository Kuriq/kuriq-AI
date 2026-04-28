import time
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse, ChatMetadata

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """당신은 '{courseTitle}' 강좌의 AI 학습 튜터입니다.

강좌 정보:
- 강좌명: {courseTitle}
- 분야: {courseCategory}
- 난이도: {courseDifficulty}

{note_section}

지침:
1. 이 강좌 및 관련 학습 내용에 대한 질문에만 답변하세요.
2. 날씨, 일상 대화 등 학습 도메인과 무관한 질문은 정중히 거절하세요.
3. 강좌 난이도({courseDifficulty})에 맞게 답변 깊이를 조절하세요.
4. 노트 내용을 인용할 때는 정확하게 인용하세요.
5. 답변은 한국어로 하세요.

도메인 외 질문 응답 형식 (반드시 이 문장으로 시작):
"학습 관련 질문에 답변드리고 있어요. 이 강좌와 관련된 내용을 질문해 주세요!"
"""

NOTE_WITH_CONTENT = """학습자 노트:
{noteContent}

이 사용자는 위 노트 수준까지 학습한 상태이며, 노트에 있는 내용은 이미 알고 있다고 가정하고 답변하세요.
가능하면 노트의 특정 부분을 인용하여 답변에 연결해 주세요."""

NOTE_WITHOUT_CONTENT = """학습자 노트: 없음
노트가 아직 없으므로 강좌 정보만 참고하여 답변하세요."""

OFF_TOPIC_MARKER = "학습 관련 질문에 답변드리고 있어요."
NO_NOTE_PREFIX = "노트가 아직 없어서, 강좌 정보만 참고하여 답변할게요. 노트를 작성하면 더 정확한 답변을 드릴 수 있어요!\n\n"


def _build_note_references(response_text: str, note_content: str) -> list[str]:
    """응답에서 노트 내용을 인용한 부분 추출"""
    if not note_content:
        return []

    references = []
    note_lines = [
        line.strip()
        for line in note_content.split("\n")
        if line.strip() and not line.startswith("#")
    ]
    for line in note_lines:
        if len(line) > 10 and line in response_text:
            references.append(line)

    return references[:5]  # 최대 5개


def chat(request: ChatRequest) -> ChatResponse:
    start = time.time()

    note_section = (
        NOTE_WITH_CONTENT.format(noteContent=request.noteContent)
        if request.noteContent
        else NOTE_WITHOUT_CONTENT
    )

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        courseTitle=request.courseTitle,
        courseCategory=request.courseCategory,
        courseDifficulty=request.courseDifficulty,
        note_section=note_section,
    )

    # 대화 이력 구성
    messages = [SystemMessage(content=system_prompt)]
    for turn in (request.chatHistory or []):
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.message))
        elif turn.role == "assistant":
            messages.append(AIMessage(content=turn.message))
    messages.append(HumanMessage(content=request.message))

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout,
        temperature=0.3,
    )

    response = llm.invoke(messages)
    answer = response.content.strip()

    # 도메인 외 질문 여부 판단
    is_off_topic = answer.startswith(OFF_TOPIC_MARKER)

    # 노트 없음 안내
    if not request.noteContent and not is_off_topic:
        answer = NO_NOTE_PREFIX + answer

    # 노트 인용 추출
    note_refs = (
        _build_note_references(answer, request.noteContent)
        if not is_off_topic
        else []
    )

    elapsed = round((time.time() - start) * 1000)
    usage = response.response_metadata.get("token_usage", {})

    logger.info(
        f"[Chat] userId={request.userId} "
        f"isOffTopic={is_off_topic} "
        f"{elapsed}ms"
    )

    return ChatResponse(
        message=answer,
        noteReferences=note_refs,
        isOffTopic=is_off_topic,
        metadata=ChatMetadata(
            llmCallCount=1,
            processingTimeMs=elapsed,
            inputTokens=usage.get("prompt_tokens"),
            outputTokens=usage.get("completion_tokens"),
        ),
    )