import pandas as pd

from scripts.backtest.historical_inputs import (
    build_pregame_universe_for_week,
    build_schedule_history,
)


def test_pregame_universe_comes_from_roster_and_depth_only():
    schedule = pd.DataFrame([
        {"season": 2025, "week": 8, "team": "BUF", "opponent": "MIA", "game_id": "g1"},
        {"season": 2025, "week": 8, "team": "MIA", "opponent": "BUF", "game_id": "g1"},
    ])
    rosters = pd.DataFrame([
        {"season": 2025, "week": 8, "team": "BUF", "position": "WR", "status": "ACT", "full_name": "Alpha WR"},
        {"season": 2025, "week": 8, "team": "BUF", "position": "LB", "status": "ACT", "full_name": "Defender"},
        {"season": 2025, "week": 8, "team": "MIA", "position": "QB", "status": "ACT", "full_name": "Beta QB"},
        {"season": 2025, "week": 8, "team": "BUF", "position": "RB", "status": "CUT", "full_name": "Cut RB"},
    ])
    depth = pd.DataFrame([
        {"season": 2025, "week": 8, "club_code": "BUF", "full_name": "Alpha WR", "depth_position": "LWR", "depth_team": 1},
        {"season": 2025, "week": 8, "club_code": "MIA", "full_name": "Beta QB", "depth_position": "QB", "depth_team": 1},
    ])
    out = build_pregame_universe_for_week(
        season=2025, week=8, schedule_history=schedule,
        rosters_weekly=rosters, depth_charts=depth,
    )
    assert set(out["player"]) == {"Alpha WR", "Beta QB"}
    assert out.loc[out["player"].eq("Alpha WR"), "opponent"].iloc[0] == "MIA"
    assert out.loc[out["player"].eq("Alpha WR"), "role"].iloc[0] == "LWR1"
    assert out["pregame_source"].eq("nflverse_weekly_roster+week_tagged_depth_chart").all()


def test_pregame_universe_never_needs_target_week_results():
    schedule = pd.DataFrame([
        {"season": 2025, "week": 3, "team": "IND", "opponent": "TEN", "game_id": "g3"},
        {"season": 2025, "week": 3, "team": "TEN", "opponent": "IND", "game_id": "g3"},
    ])
    rosters = pd.DataFrame([
        {"season": 2025, "week": 3, "team": "IND", "position": "RB", "status": "ACT", "full_name": "Runner One"},
    ])
    out = build_pregame_universe_for_week(
        season=2025, week=3, schedule_history=schedule,
        rosters_weekly=rosters, depth_charts=pd.DataFrame(),
    )
    assert out.iloc[0]["player"] == "Runner One"
    assert out.iloc[0]["season"] == 2025
    assert out.iloc[0]["week"] == 3


def test_schedule_history_expands_games_to_team_rows(monkeypatch):
    raw = pd.DataFrame([
        {"season": 2025, "week": 1, "home_team": "BUF", "away_team": "MIA", "game_id": "x"}
    ])
    monkeypatch.setattr("scripts.backtest.historical_inputs._load_schedule", lambda season: raw.copy())
    out = build_schedule_history([2025])
    assert len(out) == 2
    assert set(out["team"]) == {"BUF", "MIA"}
    assert set(out["opponent"]) == {"BUF", "MIA"}
    assert not out.duplicated(["season", "week", "team"]).any()
