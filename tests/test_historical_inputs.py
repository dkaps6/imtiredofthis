import pandas as pd

from scripts.backtest.historical_inputs import (
    build_pregame_universe_for_week,
    build_schedule_history,
    build_team_weekly_from_pbp,
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


def test_team_weekly_tracks_official_attempts_per_dropback(monkeypatch):
    # BUF has four dropbacks: three official attempts plus one sack. The
    # conversion must therefore be 0.75 rather than treating all dropbacks as
    # pass attempts. Add one rush play so the offensive-play denominator exists.
    pbp = pd.DataFrame([
        {"season_type": "REG", "week": 1, "posteam": "BUF", "defteam": "MIA", "qb_dropback": 1, "rush_attempt": 0, "pass_attempt": 1, "sack": 0, "qb_hit": 0, "success": 1, "epa": .2, "yards_gained": 8},
        {"season_type": "REG", "week": 1, "posteam": "BUF", "defteam": "MIA", "qb_dropback": 1, "rush_attempt": 0, "pass_attempt": 1, "sack": 0, "qb_hit": 0, "success": 1, "epa": .1, "yards_gained": 5},
        {"season_type": "REG", "week": 1, "posteam": "BUF", "defteam": "MIA", "qb_dropback": 1, "rush_attempt": 0, "pass_attempt": 1, "sack": 0, "qb_hit": 0, "success": 0, "epa": -.1, "yards_gained": 0},
        {"season_type": "REG", "week": 1, "posteam": "BUF", "defteam": "MIA", "qb_dropback": 1, "rush_attempt": 0, "pass_attempt": 0, "sack": 1, "qb_hit": 1, "success": 0, "epa": -.5, "yards_gained": -7},
        {"season_type": "REG", "week": 1, "posteam": "BUF", "defteam": "MIA", "qb_dropback": 0, "rush_attempt": 1, "pass_attempt": 0, "sack": 0, "qb_hit": 0, "success": 1, "epa": .1, "yards_gained": 4},
        {"season_type": "REG", "week": 1, "posteam": "MIA", "defteam": "BUF", "qb_dropback": 1, "rush_attempt": 0, "pass_attempt": 1, "sack": 0, "qb_hit": 0, "success": 1, "epa": .2, "yards_gained": 6},
        {"season_type": "REG", "week": 1, "posteam": "MIA", "defteam": "BUF", "qb_dropback": 0, "rush_attempt": 1, "pass_attempt": 0, "sack": 0, "qb_hit": 0, "success": 1, "epa": .1, "yards_gained": 3},
    ])
    monkeypatch.setattr("scripts.backtest.historical_inputs.get_pbp", lambda season, min_rows=1: pbp.copy())
    out = build_team_weekly_from_pbp([2025])
    buf = out.loc[out["team"].eq("BUF")].iloc[0]
    assert buf["dropback_rate"] == 0.8
    assert buf["pass_attempts_per_dropback"] == 0.75
