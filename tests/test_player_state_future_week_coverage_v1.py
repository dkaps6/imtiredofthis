from __future__ import annotations

import pandas as pd

from scripts.research.audit_player_state_future_week_coverage_v1 import (
    prior_week_scheduled_target_teams,
    source_complete_for_prior_week,
    target_schedule,
    target_week_roster_source_ready,
    GAME_ROSTER_STATUS,
)


def test_target_schedule_builds_two_team_rows_per_game():
    pairs = [
        ("IND","HOU"),("DAL","NYG"),("BUF","MIA"),("BAL","CLE"),("PIT","CIN"),
        ("DEN","LV"),("LAC","KC"),("SEA","SF"),("TB","ATL"),("CHI","GB"),
    ]
    schedule = pd.DataFrame(
        [{"week":6,"home":home,"away":away} for home,away in pairs]
    )
    out = target_schedule(schedule, 6)
    assert len(out) == 20
    assert dict(zip(out["team"], out["opponent"]))["IND"] == "HOU"
    assert dict(zip(out["team"], out["opponent"]))["HOU"] == "IND"


def test_prior_week_required_teams_respects_target_byes():
    schedule = pd.DataFrame([
        {"week": 5, "home": "IND", "away": "HOU"},
        {"week": 5, "home": "DAL", "away": "NYG"},
        {"week": 6, "home": "IND", "away": "DAL"},
    ])
    required = prior_week_scheduled_target_teams(
        schedule, {"IND", "DAL"}, 6
    )
    assert required == {"IND", "DAL"}


def test_prior_week_source_completion_fails_closed_on_missing_team():
    current = pd.DataFrame([
        {"week": 5, "team": "IND"},
    ])
    snaps = pd.DataFrame([
        {"week": 5, "team": "IND"},
        {"week": 5, "team": "DAL"},
    ])
    got = source_complete_for_prior_week(
        current, snaps, {"IND", "DAL"}, 6
    )
    assert got["complete"] is False
    assert got["missing_stat_teams"] == ["DAL"]
    assert got["missing_snap_teams"] == []


def test_prior_week_source_completion_passes_only_when_both_sources_cover_all():
    current = pd.DataFrame([
        {"week": 5, "team": "IND"},
        {"week": 5, "team": "DAL"},
    ])
    snaps = pd.DataFrame([
        {"week": 5, "team": "IND"},
        {"week": 5, "team": "DAL"},
    ])
    got = source_complete_for_prior_week(
        current, snaps, {"IND", "DAL"}, 6
    )
    assert got["complete"] is True
    assert got["missing_stat_teams"] == []
    assert got["missing_snap_teams"] == []


def test_target_week_roster_source_must_be_exact():
    exact = pd.DataFrame([
        {"roster_source_week": 6},
        {"roster_source_week": 6},
    ])
    prior = pd.DataFrame([
        {"roster_source_week": 5},
        {"roster_source_week": 5},
    ])
    mixed = pd.DataFrame([
        {"roster_source_week": 5},
        {"roster_source_week": 6},
    ])
    assert target_week_roster_source_ready(exact, 6) is True
    assert target_week_roster_source_ready(prior, 6) is False
    assert target_week_roster_source_ready(mixed, 6) is False


def test_future_player_state_game_roster_status_contract_matches_pregame_universe():
    assert GAME_ROSTER_STATUS == {"ACT", "INA"}
    statuses = pd.Series(["ACT", "INA", "RES", "DEV", "CUT", "RET"])
    keep = statuses.isin(GAME_ROSTER_STATUS)
    assert statuses.loc[keep].tolist() == ["ACT", "INA"]
