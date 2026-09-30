from datetime import datetime, timezone

from scripts.research.discover_wr_cb_archive_snapshots_v1 import (
    choose_cc_indices, classify_index_ts, same_source_url, wayback_index,
)

def d(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))

def test_exact_url_identity_never_confuses_another_article():
    src = "https://www.fantasyalarm.com/articles/nfl/wide-receivers/foo/180232"
    assert same_source_url(src, "http://fantasyalarm.com/articles/nfl/wide-receivers/foo/180232/")
    assert not same_source_url(src, src.replace("180232", "180233"))

def test_capture_classification_never_promotes_metadata_to_content_proof():
    first, last = d("2025-09-25T00:00:00Z"), d("2025-09-30T00:00:00Z")
    assert classify_index_ts("20250924235959", first, last) == "INDEX_PRE_FIRST_KICKOFF_CANDIDATE_ONLY"
    assert classify_index_ts("20250926000000", first, last) == "INDEX_BETWEEN_WEEK_GAMES_REQUIRES_PER_WR_CHECK"
    assert classify_index_ts("20250930120000", first, last) == "INDEX_AFTER_FINAL_WEEK_KICKOFF_NOT_PREGAME"
    assert classify_index_ts("invalid", first, last) == "INVALID_INDEX_TIMESTAMP"

def test_choose_index_is_bounded_and_not_exhaustive():
    coll = [{"id":f"CC-MAIN-2025-{n:02d}"} for n in (30, 34, 37, 39, 40, 42, 50)]
    selected = choose_cc_indices(coll, d("2025-09-25T00:00:00Z"))
    assert 0 < len(selected) <= 3
    assert all(x.startswith("CC-MAIN-2025-") for x in selected)
    assert "CC-MAIN-2025-50" not in selected

def test_wayback_http_error_is_not_misreported_as_no_snapshot(monkeypatch):
    import scripts.research.discover_wr_cb_archive_snapshots_v1 as s
    monkeypatch.setattr(s, "request_json", lambda *a, **kw: (None, "HTTP_429"))
    rows, status = wayback_index("https://www.fantasyalarm.com/a/1", 2025)
    assert rows == []
    assert status == "HTTP_429"
