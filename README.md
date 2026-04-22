# 큐릭 (Kuriq) — FastAPI Server

> AI 기반 평생교육 커리큘럼 추천 웹 서비스


<br/>

## 서비스 소개

큐릭은 사용자가 자연어로 학습 목표를 입력하면 K-MOOC, KOCW, 서울시 평생학습포털 등 공공 교육 플랫폼의 강의 데이터를 분석하여 개인 맞춤형 주차별 학습 로드맵을 자동으로 생성해주는 웹 서비스입니다.

- **자연어 기반 로드맵 생성** — "3개월 안에 데이터 분석 자격증을 따고 싶어"처럼 자유롭게 입력
- **공공 교육 데이터 통합** — 여러 플랫폼의 무료 강의를 한곳에서
- **학습 진도 트래킹** — 주차별 체크리스트와 영양소 갭 분석
- **리마인드 알림** — 이메일 / 카카오톡으로 학습 일정 알림
  

## 기술 스택

| 분류 | 기술 |
|------|------|
| 프레임워크 | FastAPI, Uvicorn |
| LLM | OpenAI GPT (LangChain, LangGraph) |
| 벡터 DB | ChromaDB |
| 임베딩 | Sentence-Transformers |
| 데이터 검증 | Pydantic v2 |
| ML | PyTorch, Transformers, scikit-learn |


## 프로젝트 구조

```text
kuriq-AI/
├── app/
│   ├── main.py               # 앱 진입점, 미들웨어, 라우터 등록
│   ├── core/
│   │   ├── config.py         # 환경 변수 기반 설정
│   │   ├── chroma.py         # ChromaDB 싱글턴 클라이언트
│   │   └── scheduler_state.py
│   ├── routers/
│   │   ├── internal.py       # 헬스 체크
│   │   ├── roadmap.py        # 학습 로드맵 생성/재조정
│   │   ├── courses.py        # 코스 임베딩 관리
│   │   ├── chat.py           # AI 튜터링 채팅
│   │   ├── quiz.py           # 퀴즈 생성 및 채점
│   │   └── note.py           # 노트 정리
│   ├── services/             # 비즈니스 로직
│   └── schemas/              # Pydantic 요청/응답 스키마
├── requirements.txt
└── .env
```



## 시작하기

### 사전 요구사항

- Python 3.13+
- OpenAI API Key

### 설치

```bash
# 1. 가상 환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
cp .env.example .env
# .env 파일에 필요한 값을 입력합니다
```

### 환경 변수

| 변수 | 설명 | 예시 |
|------|------|------|
| `OPENAI_API_KEY` | OpenAI API 인증 키 | `sk-proj-...` |
| `INTERNAL_SECRET_KEY` | 내부 API 인증 키 | `dev-secret` |
| `LLM_MODEL` | 사용할 LLM 모델 | `gpt-4o-mini` |
| `LLM_TIMEOUT` | LLM 요청 타임아웃 (초) | `30` |
| `RAG_TOP_K` | RAG 검색 시 반환할 코스 수 | `20` |
| `CHROMA_MODE` | ChromaDB 실행 모드 | `local` |
| `CHROMA_PATH` | ChromaDB 저장 경로 | `../kuriq-data/chroma_db` |

### 서버 실행

```bash
# 개발 모드
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 프로덕션 모드
uvicorn app.main:app --host 0.0.0.0 --port 8000
```



## API 문서

서버 실행 후 아래 주소에서 인터랙티브 API 문서를 확인할 수 있습니다.

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc



## 인증

대부분의 엔드포인트는 요청 헤더에 `X-Internal-Key`가 필요합니다.

```bash
curl -H "X-Internal-Key: <your-secret-key>" ...
```

**인증 불필요 경로**: `/internal/health`, `/internal/ai/health/detailed`, `/docs`, `/redoc`, `/openapi.json`