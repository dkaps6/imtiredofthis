from scripts.research.collect_qb_team_pass_intent_prospective_structured_v1l import (
    DiscoveryCandidate,
    _excluded_path,
    _review_excerpts,
    _source_family,
)


def test_allowed_source_family_is_narrow():
    assert _source_family("/news/transcripts-week-6-vs-ravens") == "transcript"
    assert _source_family("/video/press-conference-quarterback-week-6") == "press-conference"
    assert _source_family("/news/game-preview-week-6") == ""


def test_excluded_generic_article_families():
    assert _excluded_path("/news/injury-report-week-6") is True
    assert _excluded_path("/news/game-preview-week-6") is True
    assert _excluded_path("/news/storylines-week-6") is True
    assert _excluded_path("/news/transcripts-week-6") is False


def test_review_excerpts_prioritizes_action_language():
    text = (
        "The opponent allows many passing yards. "
        "We want to establish the run and use play action against the Ravens this week. "
        "The quarterback threw for three touchdowns after the game."
    )
    rows = _review_excerpts(text, "BAL")
    assert rows
    assert "establish the run" in rows[0].lower()
    assert all("after the game" not in x.lower() for x in rows)
