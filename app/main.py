import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from app.routers import roadmap, courses, chat, internal, quiz, note
from app.core.config import settings

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


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["components"]["securitySchemes"] = {
        "InternalKey": {
            "type": "apiKey",
            "in": "header",
            "name": "X-Internal-Key",
        }
    }
    for path in schema.get("paths", {}).values():
        for operation in path.values():
            operation["security"] = [{"InternalKey": []}]
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


# 내부 인증 미들웨어
@app.middleware("http")
async def verify_internal_key(request: Request, call_next):
    # 헬스체크는 인증 제외
    if request.url.path in ["/internal/health", "/internal/ai/health/detailed", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)

    key = request.headers.get("X-Internal-Key")
    if key != settings.internal_secret_key:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})

    return await call_next(request)


# 라우터 등록
app.include_router(internal.router)
app.include_router(roadmap.router, prefix="/internal")
app.include_router(courses.router, prefix="/internal")
app.include_router(chat.router, prefix="/internal")
app.include_router(quiz.router, prefix="/internal")
app.include_router(note.router, prefix="/internal")