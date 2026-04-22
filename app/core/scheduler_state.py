from datetime import datetime
from typing import Optional

_state: dict = {
    "status": "UP",
    "nextCrawlAt": None,
    "lastCrawlCompletedAt": None,
}


def get_state() -> dict:
    return _state.copy()


def update_state(
    status: str = "UP",
    next_crawl_at: Optional[datetime] = None,
    last_crawl_completed_at: Optional[datetime] = None,
) -> None:
    _state["status"] = status
    if next_crawl_at:
        _state["nextCrawlAt"] = next_crawl_at.isoformat(timespec="seconds")
    if last_crawl_completed_at:
        _state["lastCrawlCompletedAt"] = last_crawl_completed_at.isoformat(timespec="seconds")
