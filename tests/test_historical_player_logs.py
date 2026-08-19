import sys
import types

import pandas as pd

from scripts.backtest.historical_player_logs import (
    _load_historical_weekly,
    build_historical_player_logs,
)


def test_historical_loader_uses_current_nflreadpy_weekly_api(monkeypatch):
    calls = {}

    class FakeFrame:
        def to_pandas(self):
            return pd.DataFrame([{"season": 2025, "week": 1}])

    fake = types.SimpleNamespace()

    def load_player_stats(*, seasons, summary_level):
        calls["seasons"] = seasons
        calls["summary_level"] = summary_level
        return FakeFrame()

    fake.load_player_stats = load_player_stats
    monkeypatch.setitem(sys.modules, "nflreadpy", fake)

    out = _load_historical_weekly(2025)
    assert len(out) == 1
    assert calls == {"seasons": [2025], "summary_level": "week"}


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
    monkeypatch.setattr("scripts.backtest.historical_player_logs._load_historical_weekly", lambda season: raw.copy())
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
    monkeypatch.setattr("scripts.backtest.historical_player_logs._load_historical_weekly", lambda season: raw.copy())
    try:
        build_historical_player_logs(seasons=[2025], schedule_history=schedule)
    except RuntimeError as exc:
        assert "could not resolve opponent" in str(exc)
    else:
        raise AssertionError("expected missing historical schedule match to fail closed")


def test_historical_player_logs_exclude_postseason_before_schedule_join(monkeypatch):
    raw = pd.DataFrame([
        {
            "season": 2024,
            "week": 18,
            "recent_team": "LAR",
            "position": "QB",
            "player_display_name": "Regular QB",
            "player_id": "p1",
            "attempts": 30,
            "passing_yards": 250,
            "rushing_attempts": 1,
            "rushing_yards": 3,
            "targets": 0,
            "receptions": 0,
            "receiving_yards": 0,
        },
        {
            "season": 2024,
            "week": 19,
            "recent_team": "LAR",
            "position": "QB",
            "player_display_name": "Regular QB",
            "player_id": "p1",
            "attempts": 28,
            "passing_yards": 240,
            "rushing_attempts": 2,
            "rushing_yards": 8,
            "targets": 0,
            "receptions": 0,
            "receiving_yards": 0,
        },
    ])
    schedule = pd.DataFrame([
        {"season": 2024, "week": 18, "team": "LAR", "opponent": "SEA", "game_id": "g18"},
        {"season": 2024, "week": 18, "team": "SEA", "opponent": "LAR", "game_id": "g18"},
    ])
    monkeypatch.setattr("scripts.backtest.historical_player_logs._load_historical_weekly", lambda season: raw.copy())
    out = build_historical_player_logs(seasons=[2024], schedule_history=schedule)
    assert len(out) == 1
    assert int(out.iloc[0]["week"]) == 18
    assert out.iloc[0]["opponent"] == "SEA"
