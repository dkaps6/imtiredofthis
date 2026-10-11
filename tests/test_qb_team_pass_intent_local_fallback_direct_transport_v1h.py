from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.audit_qb_team_pass_intent_local_fallback_direct_transport_v1h import (
    EXPECTED_OFFICIAL_READY,
    EXPECTED_UNRESOLVED,
    EXPECTED_V1G_ROWS,
    LOCAL_SOURCES,
    PublisherResult,
    SitemapEntry,
    _candidate_urls,
    _evaluate,
    _load_v1g,
    _parse_sitemap,
)
from scripts.research.audit_qb_team_pass_intent_gold_semantic_extractor_v1e import TEAM_SYNONYMS


def test_local_source_map_is_exact_32_by_2_and_has_no_search_engines():
    assert set(LOCAL_SOURCES) == set(TEAM_SYNONYMS)
    assert len(LOCAL_SOURCES) == 32
    all_domains = []
    for team, domains in LOCAL_SOURCES.items():
        assert len(domains) == 2, team
        assert len(set(domains)) == 2, team
        for domain in domains:
            assert domain not in {"google.com", "bing.com", "duckduckgo.com"}
            all_domains.append(domain)
    assert len(all_domains) == 64


def test_candidate_urls_requires_team_or_opponent_plus_window_or_year_week():
    row = {
        "season": 2024,
        "week": 5,
        "team": "DAL",
        "opponent": "PIT",
        "kickoff": "2024-10-07T00:20:00+00:00",
    }
    entries = [
        SitemapEntry(
            "https://dallasnews.com/sports/cowboys/2024/10/03/cowboys-steelers-week-5-offense-plan",
            "2024-10-03T12:00:00+00:00",
        ),
        SitemapEntry(
            "https://dallasnews.com/news/local/2024/10/03/city-council",
            "2024-10-03T12:00:00+00:00",
        ),
        SitemapEntry(
            "https://dallasnews.com/sports/cowboys/2022/old-story",
            "2022-10-03T12:00:00+00:00",
        ),
    ]
    ranked = _candidate_urls(row, entries)
    assert len(ranked) == 1
    assert "cowboys-steelers-week-5" in ranked[0][2]


def test_load_v1g_requires_exact_state_counts(tmp_path: Path):
    rows = []
    for i in range(EXPECTED_OFFICIAL_READY):
        rows.append({"screen_state": "OFFICIAL_AUTO_REVIEW_READY", "i": i})
    for i in range(EXPECTED_UNRESOLVED):
        rows.append({
            "screen_state": "OFFICIAL_UNRESOLVED_REQUIRES_FALLBACK_OR_REVIEW",
            "i": i,
        })
    assert len(rows) == EXPECTED_V1G_ROWS
    p = tmp_path / "rows.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    ready, unresolved = _load_v1g(p)
    assert len(ready) == EXPECTED_OFFICIAL_READY
    assert len(unresolved) == EXPECTED_UNRESOLVED


def test_evaluate_ready_when_frozen_transport_gates_clear():
    unresolved = [
        {
            "season": 2023,
            "week": 2,
            "team": "ARI",
            "opponent": "NYG",
            "kickoff": "2023-09-17T20:05:00+00:00",
        }
        for _ in range(EXPECTED_UNRESOLVED)
    ]
    probes = {}
    publisher_results = {}
    for domains in LOCAL_SOURCES.values():
        for domain in domains:
            probes[domain] = True
            publisher_results[domain] = PublisherResult(
                domain=domain,
                probe_reachable=True,
                transport_reachable=True,
                parsed_sitemaps=1,
                enumerated_urls=10,
                errors=[],
            )
    row_results = []
    from scripts.research.audit_qb_team_pass_intent_local_fallback_direct_transport_v1h import RowResult
    for _ in range(EXPECTED_UNRESOLVED):
        row_results.append(RowResult(
            season=2023,
            week=2,
            team="ARI",
            opponent="NYG",
            kickoff="2023-09-17T20:05:00+00:00",
            mapped_publishers=list(LOCAL_SOURCES["ARI"]),
            reachable_publishers=[LOCAL_SOURCES["ARI"][0]],
            candidate_counts={LOCAL_SOURCES["ARI"][0]: 1, LOCAL_SOURCES["ARI"][1]: 0},
            candidate_examples={LOCAL_SOURCES["ARI"][0]: ["https://example.test"], LOCAL_SOURCES["ARI"][1]: []},
            has_reachable_publisher=True,
            has_local_candidate=True,
        ))
    summary = _evaluate(unresolved, probes, publisher_results, row_results)
    assert summary["disposition"] == "LOCAL_DIRECT_TRANSPORT_READY"
    assert summary["gates"]["exact_unresolved_input_158"] is True
    assert summary["gates"]["frozen_map_exactly_32x2"] is True


def test_malformed_sitemap_xml_uses_generic_tolerant_fallback():
    raw = b"""<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url>
        <loc>https://www.dallasnews.com/sports/cowboys/2024/10/03/week-5-steelers/</loc>
        <lastmod>2024-10-03T12:00:00+00:00</lastmod>
        <news>A & B</news>
      </url>
    </urlset>"""
    kind, rows = _parse_sitemap(raw)
    assert kind == "urlset"
    assert rows == [(
        "https://www.dallasnews.com/sports/cowboys/2024/10/03/week-5-steelers/",
        "2024-10-03T12:00:00+00:00",
    )]
