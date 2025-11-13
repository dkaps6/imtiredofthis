from __future__ import annotations

from datetime import datetime, timedelta
import os

import pytz


CHI_TZ = pytz.timezone(os.getenv("LOCAL_TZ", "America/Chicago"))


def parse_iso_dt(s: str) -> datetime:
    # expects ISO8601 or 'YYYY-MM-DD'
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.strptime(s, "%Y-%m-%d")
        dt = CHI_TZ.localize(dt)
    return dt.astimezone(CHI_TZ)


def slate_window_from_anchor(anchor_local: datetime) -> tuple[datetime, datetime]:
    """Thu 00:00 through Tue 03:59 (covers MNF overtime) of the anchor week."""

    # normalize to local midnight
    anchor_mid = anchor_local.replace(hour=0, minute=0, second=0, microsecond=0)
    # find Thursday of that week (Mon=0..Sun=6)
    dow = anchor_mid.weekday()
    days_to_thu = (3 - dow) % 7
    thu = anchor_mid + timedelta(days=days_to_thu)
    tue = thu + timedelta(days=5, hours=3, minutes=59)  # through early Tue
    return thu, tue


def compute_slate_window() -> tuple[datetime, datetime, datetime]:
    """Returns (anchor_local, start_local, end_local). Priority:
       1) env SLATE_DATE (YYYY-MM-DD or ISO),
       2) first event date (caller can pass),
       3) today local."""

    from datetime import datetime as dt

    env_date = os.getenv("SLATE_DATE")
    if env_date:
        anchor = parse_iso_dt(env_date)
    else:
        anchor = CHI_TZ.localize(dt.now().replace(microsecond=0))
    start, end = slate_window_from_anchor(anchor)
    return anchor, start, end
