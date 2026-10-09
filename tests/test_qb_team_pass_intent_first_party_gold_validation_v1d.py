import json
from datetime import datetime, timezone

from scripts.research.audit_qb_team_pass_intent_first_party_gold_validation_v1d import (
    GoldRowResult,
    _extract_safe_publication,
    _norm_url,
    _parse_absolute,
    evaluate,
)


def _gold(team, source_class):
    return {
        "team": team,
        "source_class": source_class,
        "timestamp_safe": "true",
        "locator": f"https://www.{team.lower()}.example.com/news/x",
        "kickoff": "2023-09-17T17:00:00+00:00",
    }


def _audit(team, reacquired=True, fetch=True, source=True, timestamp=True):
    return GoldRowResult(
        team=team,
        source_class="OFFICIAL",
        canonical_url=f"https://www.{team.lower()}.example.com/news/x",
        expected_timestamp_safe=True,
        route="OFFICIAL_FIRST_PARTY",
        sitemap_transport_reachable=True,
        parsed_sitemaps=1,
        enumerated_urls=100,
        canonical_reacquired=reacquired,
        direct_fetch_success=fetch,
        final_url=f"https://www.{team.lower()}.example.com/news/x",
        official_source_class_correct=source,
        timestamp_method="JSON_LD_datePublished",
        timestamp_value="2023-09-15T12:00:00-04:00",
        timestamp_parseable_absolute=True,
        timestamp_pre_kickoff=timestamp,
        timestamp_safe_correct=timestamp,
        errors=[],
    )


def test_url_normalization_ignores_scheme_www_query_fragment_and_trailing_slash():
    a = _norm_url("https://www.example.com/news/thing/?a=1#x")
    b = _norm_url("http://example.com/news/thing")
    assert a == b == "example.com/news/thing"


def test_absolute_timestamp_requires_timezone():
    assert _parse_absolute("2023-09-15T12:00:00-04:00") is not None
    assert _parse_absolute("2023-09-15T16:00:00Z") is not None
    assert _parse_absolute("2023-09-15") is None
    assert _parse_absolute("2023-09-15T12:00:00") is None


def test_jsonld_timestamp_is_accepted_before_kickoff():
    raw = b"""<html><head><script type="application/ld+json">
    {"@type":"NewsArticle","datePublished":"2023-09-15T12:00:00-04:00"}
    </script></head></html>"""
    kickoff = datetime(2023, 9, 17, 17, tzinfo=timezone.utc)
    method, value, parseable, pre = _extract_safe_publication(raw, kickoff)
    assert method.startswith("JSON_LD")
    assert value == "2023-09-15T12:00:00-04:00"
    assert parseable is True
    assert pre is True


def test_sitemap_lastmod_or_visible_date_is_not_timestamp_evidence():
    raw = b"""<html><body><time>September 15, 2023</time>
    <div>Published September 15, 2023</div></body></html>"""
    kickoff = datetime(2023, 9, 17, 17, tzinfo=timezone.utc)
    method, value, parseable, pre = _extract_safe_publication(raw, kickoff)
    assert (method, value, parseable, pre) == ("", "", False, False)


def test_frozen_gates_pass_with_11_of_12_reacquired_but_all_pages_timestamp_safe():
    official = ["ARI","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU"]
    rows = [_gold(t, "OFFICIAL") for t in official] + [_gold("ATL", "LOCAL_ATTRIBUTABLE")]
    audited = [_audit(t, reacquired=(i != 0)) for i, t in enumerate(official)]
    out = evaluate(rows, audited)
    assert out["official_canonical_reacquired"] == 11
    assert out["official_direct_fetch_success"] == 12
    assert out["timestamp_safe_correct"] == 12
    assert out["official_source_class_correct"] == 12
    assert out["local_fallback_teams"] == ["ATL"]
    assert out["disposition"] == "FIRST_PARTY_GOLD_VALIDATION_QUALIFIED_FOR_SEMANTIC_COVERAGE_AUDIT"
    assert out["semantic_coverage_audit_authorized"] is True
    assert out["predictive_candidate_authorized"] is False


def test_timestamp_gate_cannot_be_rescued_at_11_of_12():
    official = ["ARI","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU"]
    rows = [_gold(t, "OFFICIAL") for t in official] + [_gold("ATL", "LOCAL_ATTRIBUTABLE")]
    audited = [_audit(t, timestamp=(i != 0)) for i, t in enumerate(official)]
    out = evaluate(rows, audited)
    assert out["timestamp_safe_correct"] == 11
    assert out["disposition"] == "FIRST_PARTY_GOLD_VALIDATION_NOT_QUALIFIED"


def test_governance_is_source_only():
    official = ["ARI","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU"]
    rows = [_gold(t, "OFFICIAL") for t in official] + [_gold("ATL", "LOCAL_ATTRIBUTABLE")]
    out = evaluate(rows, [_audit(t) for t in official])
    assert out["external_search_engine_html_contacted"] is False
    assert out["football_outcomes_read"] == 0
    assert out["model_residuals_read"] == 0
    assert out["sportsbook_inputs_used"] is False
    assert out["paid_oddsapi_calls"] == 0
    assert out["week5_2026_outcomes_read"] == 0
    assert out["predictive_models_fit"] == 0
    assert out["production_changed"] is False
