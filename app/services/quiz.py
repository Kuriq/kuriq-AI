import time
import json
import logging
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.schemas.quiz import (
    QuizGradeRequest, QuizGradeMetadata, QuizGradeResponse,
    QuizGenerateRequest, QuizGenerateResponse, QuizGenerateMetadata,
    QuizQuestion, QuizOption,
)

logger = logging.getLogger(__name__)

GRADE_SYSTEM = """당신은 단답형 퀴즈 채점관입니다.
사용자 답변이 정답과 의미적으로 얼마나 일치하는지 판단하세요.

판단 기준:
- PARTIAL: 핵심 개념은 맞지만 표현이 다르거나 부분적으로 맞는 경우
- WRONG: 개념이 틀렸거나 관계없는 답변인 경우

반드시 아래 JSON만 출력하세요. 설명이나 마크다운 없이.
{
  "result": "PARTIAL" | "WRONG",
  "feedback": "학습자에게 전달할 피드백 (1~3문장, 한국어)"
}"""

CORRECT_FEEDBACK = "정확합니다! 잘 이해하고 계시네요."


def _normalize(text: str) -> str:
    return text.lower().replace(" ", "")


def _exact_match(user: str, correct: str, keywords: list[str] | None) -> str | None:
    """정확 일치하는 키워드 반환. 없으면 None."""
    targets = [correct] + (keywords or [])
    for target in targets:
        if _normalize(user) == _normalize(target):
            return target
    return None


def _call_llm(request: QuizGradeRequest) -> tuple[str, str]:
    """LLM으로 채점 → (result, feedback) 반환"""
    keywords_text = (
        f"\n허용 키워드: {', '.join(request.acceptableKeywords)}"
        if request.acceptableKeywords
        else ""
    )
    user_msg = (
        f"문제: {request.question}\n"
        f"정답: {request.correctAnswer}{keywords_text}\n"
        f"사용자 답변: {request.userAnswer}"
    )
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout,
        temperature=0.1,
    )
    response = llm.invoke([
        SystemMessage(content=GRADE_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    data = json.loads(response.content.strip())
    return data["result"], data["feedback"]


def grade_quiz(request: QuizGradeRequest) -> QuizGradeResponse:
    start = time.monotonic()
    llm_call_count = 0

    matched = _exact_match(request.userAnswer, request.correctAnswer, request.acceptableKeywords)

    if matched:
        result = "CORRECT"
        feedback = CORRECT_FEEDBACK
    else:
        try:
            result, feedback = _call_llm(request)
            llm_call_count = 1
        except Exception as e:
            logger.error(f"[Quiz] LLM 채점 실패 userId={request.userId}: {e}")
            raise

    elapsed = round((time.monotonic() - start) * 1000)
    logger.info(f"[Quiz] userId={request.userId} result={result} {elapsed}ms")

    return QuizGradeResponse(
        result=result,
        feedback=feedback,
        correctAnswer=request.correctAnswer,
        metadata=QuizGradeMetadata(
            matchedKeyword=matched,
            llmCallCount=llm_call_count,
            processingTimeMs=elapsed,
        ),
    )


GENERATE_SYSTEM = """당신은 학습 퀴즈 출제 전문가입니다.
반드시 아래 규칙을 따르세요:
1. 오직 사용자가 제공한 노트 내용에서만 문제를 출제하세요. 노트에 없는 내용은 절대 출제하지 마세요.
2. 문제 구성: MULTIPLE_CHOICE {mc_count}개, TRUE_FALSE {tf_count}개, SHORT_ANSWER {sa_count}개
3. 각 문제에는 노트에서 직접 인용한 noteReference를 포함하세요.
4. 강좌 난이도({difficulty})에 맞는 수준으로 출제하세요.
5. 반드시 아래 JSON 배열만 출력하세요. 설명이나 마크다운 없이.

[
  {{
    "type": "MULTIPLE_CHOICE",
    "question": "문제 텍스트",
    "options": [{{"id": "A", "text": "..."}}, {{"id": "B", "text": "..."}}, {{"id": "C", "text": "..."}}, {{"id": "D", "text": "..."}}],
    "correctAnswer": "A",
    "acceptableKeywords": null,
    "explanation": "해설 텍스트",
    "noteReference": "노트에서 인용한 문장",
    "topic": "주제"
  }},
  {{
    "type": "TRUE_FALSE",
    "question": "문제 텍스트",
    "options": null,
    "correctAnswer": true,
    "acceptableKeywords": null,
    "explanation": "해설 텍스트",
    "noteReference": "노트에서 인용한 문장",
    "topic": "주제"
  }},
  {{
    "type": "SHORT_ANSWER",
    "question": "문제 텍스트",
    "options": null,
    "correctAnswer": "정답",
    "acceptableKeywords": ["키워드1", "키워드2"],
    "explanation": "해설 텍스트",
    "noteReference": "노트에서 인용한 문장",
    "topic": "주제"
  }}
]"""


def _build_counts(question_count: int) -> tuple[int, int, int]:
    """questionCount에 따라 (mc, tf, sa) 비율 결정."""
    if question_count <= 3:
        return question_count - 1, 0, 1
    if question_count == 4:
        return 2, 1, 1
    # 5 이상: 나머지는 객관식으로 채움
    return question_count - 2, 1, 1


def generate_quiz(request: QuizGenerateRequest) -> QuizGenerateResponse:
    start = time.monotonic()

    mc_count, tf_count, sa_count = _build_counts(request.questionCount)
    system_prompt = GENERATE_SYSTEM.format(
        mc_count=mc_count,
        tf_count=tf_count,
        sa_count=sa_count,
        difficulty=request.courseDifficulty,
    )

    exclude_note = ""
    if request.excludeQuestions:
        previous_questions = "\n".join(f"- {question}" for question in request.excludeQuestions)
        exclude_note = (
            f"\n이전에 출제된 문제가 {len(request.excludeQuestions)}개 있습니다. 아래 문제들과 중복되지 않도록 새로운 문제를 출제하세요."
            f"\n이전 문제 목록:\n{previous_questions}"
        )

    user_msg = (
        f"강좌명: {request.courseTitle}\n"
        f"난이도: {request.courseDifficulty}\n"
        f"{exclude_note}\n"
        f"노트 내용:\n{request.noteContent}"
    )

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout,
        temperature=0.4,
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ])

    raw_questions: list[dict] = json.loads(response.content.strip())

    questions = [
        QuizQuestion(
            questionId=str(uuid.uuid4()),
            type=q["type"],
            question=q["question"],
            options=[QuizOption(**o) for o in q["options"]] if q.get("options") else None,
            correctAnswer=q["correctAnswer"],
            acceptableKeywords=q.get("acceptableKeywords"),
            explanation=q["explanation"],
            noteReference=q["noteReference"],
            topic=q["topic"],
        )
        for q in raw_questions
    ]

    elapsed = round((time.monotonic() - start) * 1000)
    logger.info(f"[Quiz] generate userId={request.userId} count={len(questions)} {elapsed}ms")

    return QuizGenerateResponse(
        questions=questions,
        metadata=QuizGenerateMetadata(
            questionsGenerated=len(questions),
            llmCallCount=1,
            processingTimeMs=elapsed,
        ),
    )
