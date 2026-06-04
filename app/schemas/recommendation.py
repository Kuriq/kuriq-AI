from pydantic import BaseModel

# Spring Boot에서 AI 서버로 보내는 요청
class RecommendationRequest(BaseModel):
    courseId: str   # 가장 최근 이수한 강좌 ID (본인 강좌 제외용)
    courseTitle: str # 가장 최근 이수한 강좌 제목 (벡터 검색 쿼리용)
    category: str   # 해당 강좌의 카테고리 (벡터 검색 필터)
    top_k: int = 5  # 추천 강좌 후보 수

# 추천 강좌 1개
class RecommendationCourse(BaseModel):
    course_id: str    # 강좌 ID
    title: str        # 강좌 제목
    institution: str  # 운영 기관
    category: str     # 카테고리
    duration: str     # 수강 기간

# AI 서버 → Spring Boot 응답
class RecommendationResponse(BaseModel):
    courses: list[RecommendationCourse]  # 추천 강좌 목록