from scripts.research.audit_qb_team_pass_intent_first_party_archive_transport_v1c import (
    TeamResult,
    _candidate,
    _discovery_year,
    _first_party,
    evaluate,
)


def _row(team, *, transport=True, candidates=1, years=(2023, 2024, 2025)):
    return TeamResult(
        team=team,
        domain=f"{team.lower()}.example.com",
        robots_status="OK",
        robots_sitemaps=[],
        sitemap_transport_reachable=transport,
        parsed_sitemaps=1 if transport else 0,
        enumerated_urls=candidates,
        candidate_urls=candidates,
        candidate_years=list(years) if candidates else [],
        candidate_examples=[],
        errors=[],
    )


def test_first_party_boundary():
    assert _first_party("https://www.patriots.com/news/x", "patriots.com")
    assert _first_party("https://media.patriots.com/x", "patriots.com")
    assert not _first_party("https://evilpatriots.com/x", "patriots.com")
    assert not _first_party("https://example.com/patriots.com/x", "patriots.com")


def test_candidate_vocabulary_is_fixed_path_only():
    assert _candidate("https://www.team.com/news/transcripts-head-coach")
    assert _candidate("https://www.team.com/news/press-conference-week-5")
    assert _candidate("https://www.team.com/news/media-availability-october-2")
    assert _candidate("https://www.team.com/news/what-they-said-week-8")
    assert not _candidate("https://www.team.com/news/game-preview-week-5")
    assert not _candidate("https://www.team.com/search?q=transcript")


def test_discovery_year_prefers_lastmod_and_is_bounded():
    assert _discovery_year("https://x.com/no-year", "2024-10-02T12:00:00Z") == 2024
    assert _discovery_year("https://x.com/news/transcript-2023-week-5", "") == 2023
    assert _discovery_year("https://x.com/news/transcript-2022", "2022-01-01") is None
    assert _discovery_year("https://x.com/news/transcript", "") is None


def test_exact_frozen_transport_gates_pass():
    teams=[f"T{i:02d}" for i in range(32)]
    rows=[]
    for i,team in enumerate(teams):
        if i < 12:
            rows.append(_row(team, years=(2023, 2024, 2025)))
        elif i < 20:
            rows.append(_row(team, years=(2023, 2024)))
        elif i < 24:
            rows.append(_row(team, years=(2024,)))
        else:
            rows.append(_row(team, transport=False, candidates=0, years=()))
    out=evaluate(rows)
    assert out["disposition"]=="FIRST_PARTY_ARCHIVE_TRANSPORT_QUALIFIED_FOR_SEMANTIC_AUDIT"
    assert out["transport_reachable_clubs"]==24
    assert out["candidate_url_clubs"]==24
    assert out["candidate_two_of_three_year_clubs"]==20
    assert out["candidate_all_three_year_clubs"]==12
    assert out["semantic_audit_authorized"] is True
    assert out["predictive_candidate_authorized"] is False


def test_one_gate_short_fails_without_threshold_rescue():
    teams=[f"T{i:02d}" for i in range(32)]
    rows=[]
    for i,team in enumerate(teams):
        if i < 11:
            rows.append(_row(team, years=(2023, 2024, 2025)))
        elif i < 20:
            rows.append(_row(team, years=(2023, 2024)))
        elif i < 24:
            rows.append(_row(team, years=(2024,)))
        else:
            rows.append(_row(team, transport=False, candidates=0, years=()))
    out=evaluate(rows)
    assert out["candidate_all_three_year_clubs"]==11
    assert out["disposition"]=="FIRST_PARTY_ARCHIVE_TRANSPORT_NOT_QUALIFIED"
    assert out["semantic_audit_authorized"] is False


def test_governance_is_source_only():
    rows=[_row(f"T{i:02d}") for i in range(32)]
    out=evaluate(rows)
    assert out["sportsbook_inputs_used"] is False
    assert out["paid_oddsapi_calls"]==0
    assert out["football_outcomes_read"]==0
    assert out["model_residuals_read"]==0
    assert out["week5_2026_outcomes_read"]==0
    assert out["predictive_models_fit"]==0
    assert out["production_changed"] is False
