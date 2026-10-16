from scripts.research.audit_qb_team_pass_intent_full_official_first_source_screen_v1g import (
    EXPECTED_UNIVERSE,
    Candidate,
    ScreenRow,
    SitemapEntry,
    TeamWeek,
    _candidate_rows,
    _evaluate,
)


def _tw(season=2024, week=5, team="DAL", opponent="PIT"):
    return TeamWeek(
        season=season,
        week=week,
        team=team,
        opponent=opponent,
        kickoff="2024-10-06T20:25:00+00:00",
    )


def test_candidate_ranking_prefers_opponent_then_week_then_source_window():
    tw = _tw()
    entries = [
        SitemapEntry(
            "https://www.dallascowboys.com/news/general-story",
            "2024-10-04T12:00:00Z",
        ),
        SitemapEntry(
            "https://www.dallascowboys.com/news/press-conference-week-5",
            "2024-10-03T12:00:00Z",
        ),
        SitemapEntry(
            "https://www.dallascowboys.com/news/cowboys-prepare-for-steelers",
            "2024-09-01T12:00:00Z",
        ),
    ]
    out = _candidate_rows(tw, entries)
    assert len(out) == 3
    assert out[0].week_url_match is True
    assert out[0].source_token_match is True
    assert out[0].date_window_match is True
    assert out[1].opponent_url_match is True
    assert out[2].date_window_match is True


def test_old_unrelated_source_page_is_not_candidate():
    tw = _tw()
    entries = [
        SitemapEntry(
            "https://www.dallascowboys.com/news/transcript-from-2023",
            "2023-10-01T12:00:00Z",
        )
    ]
    assert _candidate_rows(tw, entries) == []


def test_target_year_source_token_can_enter_strong_route_without_lastmod():
    tw = _tw()
    entries = [
        SitemapEntry(
            "https://www.dallascowboys.com/news/2024/press-conference-offense",
            "",
        )
    ]
    out = _candidate_rows(tw, entries)
    assert len(out) == 1
    assert out[0].source_token_match is True
    assert out[0].year_url_match is True


def _row(i, accepted=True):
    seasons = (2023, 2024, 2025)
    weeks = (2, 5, 8, 11, 14, 17)
    season = seasons[i % 3]
    week = weeks[i % 6]
    team = f"T{i % 32:02d}"
    return ScreenRow(
        season=season,
        week=week,
        team=team,
        opponent="OPP",
        kickoff="2024-10-06T20:25:00+00:00",
        screen_state=(
            "OFFICIAL_AUTO_REVIEW_READY"
            if accepted else "OFFICIAL_UNRESOLVED_REQUIRES_FALLBACK_OR_REVIEW"
        ),
        candidate_count=3,
        candidates_attempted=1,
        accepted_rank=1 if accepted else 0,
        locator="https://team.example/news/x" if accepted else "",
        publication_time="2024-10-04T12:00:00Z" if accepted else "",
        timestamp_method="JSON_LD_datePublished" if accepted else "",
        candidate_tag="PASS_EMPHASIS" if accepted else "",
        positive_score=0.5 if accepted else 0.0,
        negative_score=0.2 if accepted else 0.0,
        semantic_margin=0.3 if accepted else 0.0,
        evidence="We plan to throw the ball more this week." if accepted else "",
    )


def test_evaluate_never_finalizes_unresolved_as_no_source():
    # Build the exact-sized structural population. All accepted gives a clean pass
    # on density, independent of any network/model mechanics.
    rows = [_row(i, accepted=True) for i in range(EXPECTED_UNIVERSE)]
    out = _evaluate(rows)
    assert out["team_weeks"] == EXPECTED_UNIVERSE
    assert out["official_rows_finalized_as_no_source"] == 0
    assert out["disposition"] == "OFFICIAL_AUTO_READY_LOWER_BOUND_CLEARS_PARENT_DENSITY_GATES"
    assert out["final_source_qualification_claimed"] is False
    assert out["predictive_candidate_authorized"] is False


def test_density_failure_is_routing_result_not_source_rejection():
    rows = [_row(i, accepted=(i % 2 == 0)) for i in range(EXPECTED_UNIVERSE)]
    out = _evaluate(rows)
    assert out["disposition"] == "OFFICIAL_AUTO_READY_LOWER_BOUND_DOES_NOT_CLEAR_PARENT_DENSITY_GATES"
    assert out["official_rows_finalized_as_no_source"] == 0
    assert out["final_source_qualification_claimed"] is False


def test_governance_is_source_only():
    rows = [_row(i, accepted=True) for i in range(EXPECTED_UNIVERSE)]
    out = _evaluate(rows)
    assert out["external_search_engine_html_contacted"] is False
    assert out["football_outcomes_read"] == 0
    assert out["model_residuals_read"] == 0
    assert out["sportsbook_inputs_used"] is False
    assert out["paid_oddsapi_calls"] == 0
    assert out["week5_2026_outcomes_read"] == 0
    assert out["supervised_fit_or_finetune_used"] is False
    assert out["football_predictive_models_fit"] == 0
    assert out["production_changed"] is False
