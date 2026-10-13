from datetime import datetime, timezone

import pandas as pd
import pytest

from scripts.research.capture_fantasyalarm_wr_cb_pregame_v1 import (
    _validate_url_and_title, assess_capture_rows,
)


def dt(ymd):
    return datetime.fromisoformat(ymd.replace("Z", "+00:00"))


def test_exact_url_title_and_week_are_required():
    html = "<html><head><meta property='og:title' content='2026 WR-CB Week 4 report'></head></html>"
    _validate_url_and_title(
        "https://www.fantasyalarm.com/articles/nfl/wide-receivers/example", html, 2026, 4,
    )
    with pytest.raises(ValueError):
        _validate_url_and_title(
            "https://www.fantasyalarm.com.evil.test/articles/nfl/wide-receivers/example",
            html, 2026, 4,
        )
    with pytest.raises(ValueError):
        _validate_url_and_title(
            "https://www.fantasyalarm.com/articles/nfl/wide-receivers/example", html, 2026, 14,
        )


def test_rowwise_pregame_snapshot_quarantines_after_kickoff_and_bad_opponents():
    rows = pd.DataFrame([
        {"season": 2026, "week": 4, "wr_team": "IND", "opponent": "TEN",
         "published_at_utc": "2026-10-01T12:00:00Z",
         "modified_at_utc": "2026-10-01T12:00:00Z",
         "modification_metadata_status": "UNAMBIGUOUS_MODIFICATION_METADATA"},
        {"season": 2026, "week": 4, "wr_team": "KC", "opponent": "SEA",
         "published_at_utc": "2026-10-01T12:00:00Z",
         "modified_at_utc": "2026-10-01T12:00:00Z",
         "modification_metadata_status": "UNAMBIGUOUS_MODIFICATION_METADATA"},
        {"season": 2026, "week": 4, "wr_team": "NYJ", "opponent": "DAL",
         "published_at_utc": "2026-10-01T12:00:00Z", "modified_at_utc": "",
         "modification_metadata_status": "MISSING_MODIFICATION_METADATA"},
    ])
    # Synthetic schedule only; no assertion about 2026 real opponents.
    sched = pd.DataFrame([
        {"season": 2026, "week": 4, "team": "IND",
         "scheduled_opponent": "TEN", "kickoff_utc": "2026-10-04T17:00:00Z"},
        {"season": 2026, "week": 4, "team": "KC",
         "scheduled_opponent": "SEA", "kickoff_utc": "2026-10-01T17:00:00Z"},
        {"season": 2026, "week": 4, "team": "NYJ",
         "scheduled_opponent": "BUF", "kickoff_utc": "2026-10-04T17:00:00Z"},
    ])
    out = assess_capture_rows(
        rows, sched,
        capture_started=dt("2026-10-02T12:00:00Z"),
        capture_complete=dt("2026-10-02T12:00:05Z"),
    )
    assert out["capture_timing_status"].tolist() == [
        "PRE_KICKOFF_EXACT_HTML_CAPTURE",
        "QUARANTINE_CAPTURE_NOT_PREGAME",
        "QUARANTINE_SCHEDULE",
    ]
    assert out["pregame_fact_snapshot_candidate"].tolist() == [True, False, False]
    assert not out["model_feature_eligible"].any()


def test_modified_metadata_future_of_capture_cannot_certify_snapshot():
    row = pd.DataFrame([{
        "season": 2026, "week": 4, "wr_team": "IND", "opponent": "TEN",
        "published_at_utc": "2026-10-01T12:00:00Z",
        "modified_at_utc": "2026-10-03T12:00:00Z",
        "modification_metadata_status": "UNAMBIGUOUS_MODIFICATION_METADATA",
    }])
    sched = pd.DataFrame([{
        "season": 2026, "week": 4, "team": "IND",
        "scheduled_opponent": "TEN", "kickoff_utc": "2026-10-04T17:00:00Z",
    }])
    out = assess_capture_rows(
        row, sched, capture_started=dt("2026-10-02T12:00:00Z"),
        capture_complete=dt("2026-10-02T12:00:05Z"),
    )
    assert out.iloc[0]["capture_timing_status"] == "QUARANTINE_METADATA_FUTURE_OF_CAPTURE"
    assert not bool(out.iloc[0]["pregame_fact_snapshot_candidate"])
