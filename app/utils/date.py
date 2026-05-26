"""
날짜 유틸리티 함수
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

KOREA_TZ = ZoneInfo("Asia/Seoul")


def format_date(dt: datetime) -> str:
    """한국 시간대로 날짜 포맷팅"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    korea_dt = dt.astimezone(KOREA_TZ)
    return korea_dt.strftime("%Y 년 %m 월 %d 일")


def format_time(dt: datetime) -> str:
    """한국 시간대로 시간 포맷팅"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    korea_dt = dt.astimezone(KOREA_TZ)
    return korea_dt.strftime("%H:%M")


def format_datetime(dt: datetime) -> str:
    """한국 시간대로 날짜 + 시간 포맷팅"""
    return f"{format_date(dt)} {format_time(dt)}"


def to_korea_time(dt: datetime) -> datetime:
    """UTC 시간을 한국 시간으로 변환"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(KOREA_TZ)
