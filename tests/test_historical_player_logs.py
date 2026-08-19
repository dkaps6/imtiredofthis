import pandas as pd

from scripts.backtest.historical_player_logs import build_historical_player_logs


def test_historical_player_logs_attach_historical_opponent(monkeypatch):
    raw = pd.DataFrame([
        {
            "season": 2025,
            "week": 3,
            "recent_team": "IND",
            "position": "RB",
            "player_display_name": "Runner One",
            "player_id": "p1",
            "rushing_attempts": 12,
            "rushing_yards": 60,
            "targets": 2,
            "receptions": 1,
            "receiving_yards": 8,
            "attempts": 0,
            "passing_yards": 0,
        }
    ])
    schedule = pd.DataFrame([
        {"season": 2025, "week": 3, "team": "IND", "opponent": "TEN", "game_id": "g3"},
        {"season": 2025, "week": 3, "team": "TEN", "opponent": "IND", "game_id": "g3"},
    ])
    monkeypatch.setattr("scripts.backtest.historical_player_logs._load_weekly", lambda season: raw.copy())
    out = build_historical_player_logs(seasons=[2025], schedule_history=schedule)
    assert len(out) == 1
    assert out.iloc[0]["player"] == "Runner One"
    assert out.iloc[0]["opponent"] == "TEN"
    assert out.iloc[0]["game_id"] == "g3"
    assert out.iloc[0]["rush_yards"] == 60


def test_historical_player_logs_fail_closed_without_schedule_match(monkeypatch):
    raw = pd.DataFrame([
        {
            "season": 2025,
            "week": 3,
            "recent_team": "IND",
            "position": "RB",
            "player_display_name": "Runner One",
            "player_id": "p1",
            "rushing_attempts": 12,
            "rushing_yards": 60,
            "targets": 0,
            "receptions": 0,
            "receiving_yards": 0,
            "attempts": 0,
            "passing_yards": 0,
        }
    ])
    schedule = pd.DataFrame([
        {"season": 2025, "week": 3, "team": "BUF", "opponent": "MIA", "game_id": "g3"},
    ])
    monkeypatch.setattr("scripts.backtest.historical_player_logs._load_weekly", lambda season: raw.copy())
    try:
        build_historical_player_logs(seasons=[2025], schedule_history=schedule)
    except RuntimeError as exc:
        assert "could not resolve opponent" in str(exc)
    else:
        raise AssertionError("expected missing historical schedule match to fail closed")
