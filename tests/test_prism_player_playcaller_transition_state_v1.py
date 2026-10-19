from scripts.research.audit_prism_player_playcaller_transition_state_v1 import (
    TEAMS,
    historical_mapping_audit,
    prospective_2026_probe,
)


def test_historical_playcaller_mapping_is_complete_for_2024_2025():
    for season in (2024, 2025):
        r = historical_mapping_audit(season)
        assert r["opening_team_count"] == 32
        assert r["opening_missing_teams"] == []
        assert r["opening_extra_teams"] == []
        assert r["bad_overrides"] == []
        assert r["deterministic_blank_team_weeks"] == 0
        assert r["documented_inseason_transition_count"] > 0


def test_team_universe_is_exactly_32():
    assert len(TEAMS) == 32
    assert "WAS" in TEAMS
    assert "WSH" not in TEAMS


def test_2026_prospective_parity_fails_closed_without_complete_mapping():
    r = prospective_2026_probe()
    assert r["season"] == 2026
    assert r["repository_season_opening_team_count"] < 32
    assert r["repository_season_opening_complete_32"] is False
    assert r["same_caller_for_semantics_available"] is False
    assert r["prospective_parity_ready"] is False
