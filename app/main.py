import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from app.routers import roadmap, courses, chat
from app.core.config import settings
from app.core.chroma import get_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

app = FastAPI(
    title="Kuriq AI Server",
    description="RAG 기반 커리큘럼 추천 내부 API",
    version="1.0.0",
    docs_url="/docs",       # 개발 환경에서만 사용
    redoc_url="/redoc",
)


# 내부 인증 미들웨어
@app.middleware("http")
async def verify_internal_key(request: Request, call_next):
    # 헬스체크는 인증 제외
    if request.url.path in ["/internal/health", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)

    key = request.headers.get("X-Internal-Key")
    if key != settings.internal_secret_key:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    return await call_next(request)


# 라우터 등록
app.include_router(roadmap.router, prefix="/internal")
app.include_router(courses.router, prefix="/internal")
app.include_router(chat.router, prefix="/internal")


# 헬스체크
@app.get("/internal/health")
async def health():
    try:
        collection = get_collection()
        count = collection.count()
        return {
            "status": "ok",
            "chromadb": "connected",
            "course_count": count,
            "llm_model": settings.llm_model,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(e)},
        )