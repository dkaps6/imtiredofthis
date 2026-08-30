import pandas as pd
import pytest

from scripts._opponent_map import CANON_TEAM_CODES
from scripts.validate_2026_provider_artifacts import (
    validate_coverage,
    validate_identity_summary,
    validate_injuries,
    validate_ourlads,
    validate_schedule,
    validate_weather,
)


def _week1_schedule():
    teams = sorted(CANON_TEAM_CODES)
    rows = []
    for i in range(0, len(teams), 2):
        a, b = teams[i], teams[i + 1]
        rows.append({"season": 2026, "week": 1, "team": a, "opponent": b})
        rows.append({"season": 2026, "week": 1, "team": b, "opponent": a})
    return pd.DataFrame(rows)


def test_schedule_requires_symmetric_active_matchups():
    good = _week1_schedule()
    result, teams = validate_schedule(good, 2026, 1)
    assert result["status"] == "ready"
    assert len(teams) == 32

    bad = good.copy()
    bad.loc[0, "opponent"] = "KC"
    with pytest.raises(RuntimeError, match="not symmetric"):
        validate_schedule(bad, 2026, 1)


def test_ourlads_requires_all_32_teams_but_not_specific_player_count():
    roles = pd.DataFrame([
        {
            "team": team,
            "player": f"Player {i}",
            "player_clean_key": f"player{i}",
            "position": "QB",
            "role": "QB1",
        }
        for i, team in enumerate(sorted(CANON_TEAM_CODES))
    ])
    assert validate_ourlads(roles)["status"] == "ready"
    with pytest.raises(RuntimeError, match="all 32 teams"):
        validate_ourlads(roles.iloc[:-1].copy())


def test_weather_retains_schedule_when_forecasts_are_pending():
    schedule = _week1_schedule()
    _, teams = validate_schedule(schedule, 2026, 1)
    ordered = sorted(teams)
    games = []
    for i in range(0, len(ordered), 2):
        games.append({
            "season": 2026,
            "week": 1,
            "home": ordered[i],
            "away": ordered[i + 1],
            "forecast_ok": 0,
        })
    result = validate_weather(pd.DataFrame(games), teams, 2026, 1)
    assert result["status"] == "schedule_ready_forecast_pending"


def test_injury_provider_outage_is_fatal_but_valid_empty_report_state_is_allowed():
    empty = pd.DataFrame()
    valid_empty = {
        "season": 2026,
        "week": 1,
        "state": "no_official_report",
        "source": "no_official_report",
        "provider_errors": ["nfl.com blocked"],
    }
    result = validate_injuries(empty, valid_empty, 2026, 1)
    assert result["status"] == "no_official_report"

    outage = {
        "season": 2026,
        "week": 1,
        "state": "provider_outage",
        "source": "provider_outage",
        "provider_errors": ["nflverse down", "nfl.com down"],
    }
    with pytest.raises(RuntimeError, match="provider outage"):
        validate_injuries(empty, outage, 2026, 1)


def test_coverage_structure_can_pass_even_when_optional_scheme_source_is_unavailable():
    schedule = _week1_schedule()
    _, teams = validate_schedule(schedule, 2026, 1)
    tc = pd.DataFrame({
        "team": sorted(teams),
        "coverage_available": [0] * len(teams),
    })
    first_team = sorted(teams)[0]
    opp = schedule.loc[schedule["team"].eq(first_team), "opponent"].iloc[0]
    exposure = pd.DataFrame([
        {
            "player": "Example WR",
            "team": first_team,
            "opponent": opp,
            "matchup_available": 0,
        }
    ])
    result = validate_coverage(tc, exposure, teams)
    assert result["status"] == "structurally_ready_scheme_optional_unavailable"


def test_identity_gate_surfaces_high_temporary_share_and_fails_mapping_collapse():
    summary = pd.DataFrame([
        {"metric": "slate_players", "value": 100},
        {"metric": "stable_gsis", "value": 70},
        {"metric": "temporary_new_or_unmapped", "value": 20},
    ])
    result = validate_identity_summary(summary)
    assert result["status"] == "ready_high_temporary_identity_share"

    collapsed = pd.DataFrame([
        {"metric": "slate_players", "value": 100},
        {"metric": "stable_gsis", "value": 30},
        {"metric": "temporary_new_or_unmapped", "value": 60},
    ])
    with pytest.raises(RuntimeError, match="temporary identity rate"):
        validate_identity_summary(collapsed)
