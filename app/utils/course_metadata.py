PLATFORM_DISPLAY_LABELS = {
    "K_MOOC": "K-MOOC",
    "KOCW": "KOCW",
    "LLL_PORTAL": "온국민평생배움터",
    "EVERLEARNING": "전국평생학습",
    "SEOUL_LLL": "서울시평생학습포털",
}


CATEGORY_LABELS = {
    "IT·데이터": ["디지털", "컴퓨터", "소프트웨어", "데이터", "인공지능", "정보통신", "IT", "코딩", "프로그래밍", "AI", "머신러닝", "딥러닝", "빅데이터", "알고리즘", "웹", "앱", "모바일", "네트워크", "보안", "클라우드"],
    "경영·경제": ["경제/경영", "경영", "경제", "회계", "금융", "마케팅", "창업", "비즈니스", "관리", "리더십", "전략", "조직", "인사", "재무", "투자", "주식", "부동산"],
    "인문·교양": ["인문/교양", "인문", "철학", "역사", "문학", "심리", "교양", "고전", "사상", "종교", "신화", "인간", "사회사상", "윤리"],
    "외국어": ["영어", "중국어", "일본어", "외국어", "어학", "회화", "문법", "작문", "독해", "통역", "번역", "TESOL"],
    "자연과학·공학": ["수학", "물리", "화학", "생물", "공학", "기계", "전기", "건축", "과학", "환경", "에너지", "소재", "나노", "로봇", "항공", "우주", "지질", "천문"],
    "의료·보건": ["가족/건강/운동", "의학", "간호", "보건", "의료", "약학", "복지", "건강", "질병", "치료", "예방", "영양", "운동", "재활", "노인", "아동"],
    "예술·문화": ["예술", "음악", "미술", "디자인", "문화", "영상", "사진", "공연", "영화", "만화", "애니메이션", "패션", "공예", "무용", "연극"],
    "사회·법": ["사회", "법", "행정", "정치", "교육", "복지", "인권", "민주주의", "시민", "가족", "아동", "여성", "장애인", "다문화", "통일", "국제"],
    "취미·생활": ["취미", "생활", "요리", "여행", "스포츠", "원예", "반려동물", "레저", "게임", "독서", "글쓰기", "학습법", "자기계발"],
}


PLATFORM_ALIASES = {
    "K_MOOC": "K_MOOC",
    "K-MOOC": "K_MOOC",
    "KMOOC": "K_MOOC",
    "KOCW": "KOCW",
    "LLL_PORTAL": "LLL_PORTAL",
    "ALLGO": "LLL_PORTAL",
    "온국민평생배움터": "LLL_PORTAL",
    "EVERLEARNING": "EVERLEARNING",
    "에버러닝": "EVERLEARNING",
    "전국평생학습": "EVERLEARNING",
    "SEOUL_LLL": "SEOUL_LLL",
    "서울시평생학습포털": "SEOUL_LLL",
    "서울시 평생학습포털": "SEOUL_LLL",
}


def _exact_platform_alias(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""

    if raw in PLATFORM_ALIASES:
        return PLATFORM_ALIASES[raw]

    normalized = raw.replace("-", "_").replace(" ", "_").upper()
    return PLATFORM_ALIASES.get(normalized, "")


def _canonical_from_value(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""

    exact = _exact_platform_alias(raw)
    if exact:
        return exact

    normalized = raw.replace("-", "_").replace(" ", "_").upper()

    if raw.startswith("KOCW") or normalized.startswith("KOCW"):
        return "KOCW"
    if raw.startswith("K-MOOC") or normalized.startswith("K_MOOC"):
        return "K_MOOC"
    if "서울시평생학습포털" in raw or "서울시 평생학습포털" in raw:
        return "SEOUL_LLL"

    return ""


def canonicalize_platform(raw_platform: str, raw_institution: str = "") -> str:
    return (
        _canonical_from_value(raw_institution)
        or _canonical_from_value(raw_platform)
        or (raw_platform or "").strip().replace("-", "_").replace(" ", "_").upper()
    )


def normalize_institution(raw_institution: str, canonical_platform: str) -> str:
    institution = (raw_institution or "").strip()
    if not institution:
        return PLATFORM_DISPLAY_LABELS.get(canonical_platform, canonical_platform)

    institution_platform = _exact_platform_alias(institution)
    if institution_platform:
        return PLATFORM_DISPLAY_LABELS.get(institution_platform, institution)

    return institution


def matches_platform(raw_platform: str, raw_institution: str, requested_platform: str | None) -> bool:
    if not requested_platform:
        return True
    return canonicalize_platform(raw_platform, raw_institution) == canonicalize_platform(requested_platform)


def normalize_category(raw_category: str) -> str:
    value = (raw_category or "").strip()
    if not value:
        return "기타"

    for canonical, aliases in CATEGORY_LABELS.items():
        if value == canonical:
            return canonical
        if any(alias in value for alias in aliases):
            return canonical

    return "기타" if "기타" in value else value
