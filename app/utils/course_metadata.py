PLATFORM_DISPLAY_LABELS = {
    "K_MOOC": "K-MOOC",
    "KOCW": "KOCW",
    "LLL_PORTAL": "온국민평생배움터",
    "EVERLEARNING": "전국평생학습",
    "SEOUL_LLL": "서울시평생학습포털",
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
