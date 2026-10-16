from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

from scripts.research.audit_qb_team_pass_intent_common_crawl_archive_transport_v1i import (
    ArchiveRow,
    Collection,
    COMBINED_GATE_ROWS,
    EXPECTED_ARCHIVE_NEEDED,
    EXPECTED_DIRECT_READY,
    EXPECTED_UNRESOLVED,
    _capture_from_record,
    _evaluate,
    _load_v1h,
    _overlapping_collections,
    _query_url,
)


def test_load_v1h_requires_exact_repaired_counts_and_map(tmp_path: Path):
    rows = []
    for i in range(EXPECTED_DIRECT_READY):
        rows.append({
            "season": 2023,
            "week": 2,
            "team": "ARI",
            "opponent": "NYG",
            "kickoff": "2023-09-17T20:05:00+00:00",
            "mapped_publishers": ["azcentral.com", "arizonasports.com"],
            "has_reachable_publisher": True,
        })
    for i in range(EXPECTED_ARCHIVE_NEEDED):
        rows.append({
            "season": 2023,
            "week": 2,
            "team": "DAL",
            "opponent": "NYJ",
            "kickoff": "2023-09-17T20:25:00+00:00",
            "mapped_publishers": ["dallasnews.com", "star-telegram.com"],
            "has_reachable_publisher": False,
        })
    assert len(rows) == EXPECTED_UNRESOLVED
    p = tmp_path / "rows.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    loaded = _load_v1h(p)
    assert len(loaded) == EXPECTED_UNRESOLVED


def test_overlapping_collections_uses_capture_window():
    collections = [
        Collection("A", "https://index.commoncrawl.org/A-index",
                   "2023-09-01T00:00:00", "2023-09-10T23:59:59"),
        Collection("B", "https://index.commoncrawl.org/B-index",
                   "2023-09-15T00:00:00", "2023-09-30T23:59:59"),
        Collection("C", "https://index.commoncrawl.org/C-index",
                   "2024-01-01T00:00:00", "2024-01-10T23:59:59"),
    ]
    start = datetime(2023, 9, 8, tzinfo=timezone.utc)
    end = datetime(2023, 12, 16, tzinfo=timezone.utc)
    got = _overlapping_collections(collections, start, end)
    assert [x.id for x in got] == ["A", "B"]


def test_query_url_keeps_frozen_domain_time_and_status_contract():
    coll = Collection(
        "CC-MAIN-2023-40",
        "https://index.commoncrawl.org/CC-MAIN-2023-40-index",
        "2023-09-20T00:00:00",
        "2023-10-03T00:00:00",
    )
    start = datetime(2023, 9, 8, tzinfo=timezone.utc)
    end = datetime(2023, 12, 16, tzinfo=timezone.utc)
    url = _query_url(coll, "dallasnews.com", start, end, "DAL", "NYJ")
    assert "matchType=domain" in url
    assert "output=json" in url
    assert "from=20230908000000" in url
    assert "to=20231216000000" in url
    assert "limit=200" in url
    assert "filter=%3Dstatus%3A200" in url


def test_capture_ready_requires_frozen_domain_html_and_warc_pointer():
    record = {
        "timestamp": "20230914120000",
        "url": "https://www.dallasnews.com/sports/cowboys/2023/09/14/cowboys-jets-plan/",
        "mime": "text/html",
        "status": "200",
        "filename": "crawl-data/test.warc.gz",
        "offset": "123",
        "length": "456",
    }
    start = datetime(2023, 9, 8, tzinfo=timezone.utc)
    end = datetime(2023, 12, 16, tzinfo=timezone.utc)
    cap = _capture_from_record(
        record,
        collection="CC-MAIN-X",
        domain="dallasnews.com",
        team="DAL",
        opponent="NYJ",
        start=start,
        end=end,
    )
    assert cap is not None
    assert cap.offset == 123
    assert cap.length == 456


def test_evaluate_clears_unchanged_gate_with_four_archive_rescues():
    rows = []
    for i in range(EXPECTED_DIRECT_READY):
        rows.append(ArchiveRow(
            season=2023, week=2, team="ARI", opponent="NYG",
            kickoff="2023-09-17T20:05:00+00:00",
            direct_ready=True,
            mapped_publishers=["azcentral.com", "arizonasports.com"],
            archive_ready=False, rescued_by_archive=False,
            capture_count=0, captures=[], query_errors=[],
        ))
    for i in range(EXPECTED_ARCHIVE_NEEDED):
        rescued = i < (COMBINED_GATE_ROWS - EXPECTED_DIRECT_READY)
        rows.append(ArchiveRow(
            season=2023, week=2, team="DAL", opponent="NYJ",
            kickoff="2023-09-17T20:25:00+00:00",
            direct_ready=False,
            mapped_publishers=["dallasnews.com", "star-telegram.com"],
            archive_ready=rescued, rescued_by_archive=rescued,
            capture_count=1 if rescued else 0,
            captures=[{
                "url": "https://www.dallasnews.com/sports/cowboys/example"
            }] if rescued else [],
            query_errors=[],
        ))
    summary = _evaluate(rows, http_calls=10)
    assert summary["combined_transport_ready_rows"] == COMBINED_GATE_ROWS
    assert summary["disposition"] == "COMMON_CRAWL_ARCHIVE_TRANSPORT_RESCUES_V1H_GATE"
