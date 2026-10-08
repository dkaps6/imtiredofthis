import pandas as pd
import pytest

from scripts.validate_full_slate_data_quality_v1 import (
    _derive_positive_row_injury_scope,
    _validate_current_roster_scope,
    _validate_team_coverage_scope,
)


ALL_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}
PLAYED_TEAMS = {"LAR", "NE", "SEA", "SF"}
LIVE_TEAMS = ALL_TEAMS - PLAYED_TEAMS


def _roles(teams: set[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"team": team, "player": f"{team} Example"} for team in sorted(teams)]
    )


def _game_odds(teams: set[str]) -> pd.DataFrame:
    ordered = sorted(teams)
    assert len(ordered) % 2 == 0
    return pd.DataFrame(
        [
            {
                "event_id": f"event-{idx // 2}",
                "home_team": ordered[idx],
                "away_team": ordered[idx + 1],
            }
            for idx in range(0, len(ordered), 2)
        ]
    )


def _injuries(teams: set[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"team": team, "player": f"{team} Injured Player"} for team in sorted(teams)]
    )


def test_current_roster_scope_accepts_28_team_remaining_slate():
    role_teams, live_event_teams = _validate_current_roster_scope(
        ALL_TEAMS,
        _roles(LIVE_TEAMS),
        _game_odds(LIVE_TEAMS),
    )

    assert role_teams == LIVE_TEAMS
    assert live_event_teams == LIVE_TEAMS


def test_current_roster_scope_accepts_full_32_team_roster():
    role_teams, live_event_teams = _validate_current_roster_scope(
        ALL_TEAMS,
        _roles(ALL_TEAMS),
        _game_odds(LIVE_TEAMS),
    )

    assert role_teams == ALL_TEAMS
    assert live_event_teams == LIVE_TEAMS


def test_current_roster_scope_rejects_missing_live_event_team():
    missing_team = sorted(LIVE_TEAMS)[0]
    with pytest.raises(RuntimeError, match="missing live-event teams"):
        _validate_current_roster_scope(
            ALL_TEAMS,
            _roles(LIVE_TEAMS - {missing_team}),
            _game_odds(LIVE_TEAMS),
        )


def test_positive_row_injury_scope_certifies_exact_32_team_coverage():
    scope = _derive_positive_row_injury_scope(
        _injuries(ALL_TEAMS),
        ALL_TEAMS,
        source="nflverse",
    )

    assert scope is not None
    assert set(scope["team"]) == ALL_TEAMS
    assert set(scope["scope_state"]) == {"OFFICIAL_REPORT_ROWS"}
    assert scope["injury_rows"].eq(1).all()


def test_positive_row_injury_scope_keeps_31_team_coverage_unproven():
    missing_team = sorted(ALL_TEAMS)[0]
    scope = _derive_positive_row_injury_scope(
        _injuries(ALL_TEAMS - {missing_team}),
        ALL_TEAMS,
        source="nflverse",
    )

    assert scope is None


def test_positive_row_injury_scope_rejects_off_schedule_team():
    bad = _injuries(ALL_TEAMS)
    bad.loc[len(bad)] = {"team": "XYZ", "player": "Bad Team Player"}
    with pytest.raises(RuntimeError, match="outside active schedule"):
        _derive_positive_row_injury_scope(
            bad,
            ALL_TEAMS,
            source="nflverse",
        )


def test_week5_raw_32_roster_allows_only_certified_carolina_kc_byes():
    byes={"CAR","KC"}
    scheduled=ALL_TEAMS-byes
    role_teams,live_teams=_validate_current_roster_scope(
        scheduled,_roles(ALL_TEAMS),_game_odds(scheduled),off_week_teams=byes)
    assert role_teams==ALL_TEAMS
    assert live_teams==scheduled


def test_week5_rejects_outside_schedule_without_verified_bye_provenance():
    byes={"CAR","KC"}
    scheduled=ALL_TEAMS-byes
    with pytest.raises(RuntimeError,match="unauthorized off-schedule"):
        _validate_current_roster_scope(
            scheduled,_roles(ALL_TEAMS),_game_odds(scheduled),off_week_teams=set())


def test_week5_rejects_unknown_raw_roster_team_even_when_valid_byes():
    byes={"CAR","KC"}
    scheduled=ALL_TEAMS-byes
    with pytest.raises(RuntimeError,match="unauthorized off-schedule"):
        _validate_current_roster_scope(
            scheduled,_roles(ALL_TEAMS|{"XYZ"}),_game_odds(scheduled),off_week_teams=byes)


def _coverage(teams):
    return pd.DataFrame([{"team":t,"coverage_available":1} for t in sorted(teams)])


def test_coverage_scope_certifies_all_30_week5_teams():
    scheduled=ALL_TEAMS-{"CAR","KC"}
    assert _validate_team_coverage_scope(_coverage(scheduled),scheduled)==30


def test_coverage_scope_certifies_all_32_no_bye_teams():
    assert _validate_team_coverage_scope(_coverage(ALL_TEAMS),ALL_TEAMS)==32


def test_coverage_scope_rejects_missing_week5_team():
    scheduled=ALL_TEAMS-{"CAR","KC"}
    missing=next(iter(sorted(scheduled)))
    with pytest.raises(RuntimeError,match="schedule mismatch"):
        _validate_team_coverage_scope(_coverage(scheduled-{missing}),scheduled)


def test_coverage_scope_rejects_extra_bye_team():
    scheduled=ALL_TEAMS-{"CAR","KC"}
    with pytest.raises(RuntimeError,match="schedule mismatch"):
        _validate_team_coverage_scope(_coverage(scheduled|{"CAR"}),scheduled)


def test_coverage_scope_rejects_unavailable_scheduled_team():
    scheduled=ALL_TEAMS-{"CAR","KC"}
    frame=_coverage(scheduled)
    frame.loc[frame.team.eq("ARI"),"coverage_available"]=0
    with pytest.raises(RuntimeError,match="unavailable"):
        _validate_team_coverage_scope(frame,scheduled)
