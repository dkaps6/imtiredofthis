import pandas as pd

from scripts.build.build_injuries_weekly import SCHEMA, build_injuries, normalize_nflverse_injuries
from scripts.build.build_weather_week_v2 import build_weather_slate


def test_nflverse_injury_normalization_current_week():
    raw = pd.DataFrame({
        "season": [2026, 2026],
        "week": [1, 2],
        "full_name": ["Example Player", "Other Player"],
        "team": ["LA", "SF"],
        "report_status": ["Questionable", "Out"],
        "practice_status": ["Limited", "DNP"],
        "report_primary_injury": ["Hamstring", "Knee"],
    })
    out = normalize_nflverse_injuries(raw, 2026, 1)
    assert len(out) == 1
    assert out.loc[0, "player"] == "Example Player"
    assert out.loc[0, "team"] == "LAR"
    assert out.loc[0, "week"] == 1
    assert out.loc[0, "source"] == "nflverse"
    assert list(out.columns) == SCHEMA


def test_nflverse_injury_no_active_week_is_valid_empty():
    raw = pd.DataFrame({
        "season": [2026],
        "week": [2],
        "full_name": ["Example Player"],
        "team": ["SF"],
    })
    out = normalize_nflverse_injuries(raw, 2026, 1)
    assert out.empty
    assert list(out.columns) == SCHEMA


def test_weather_canonicalizes_la_to_lar():
    schedule = pd.DataFrame({
        "season": [2026],
        "week": [1],
        "game_id": ["2026_01_SF_LAR"],
        "home": ["LA"],
        "away": ["SF"],
        "kickoff_utc": ["2026-09-10T20:35:00Z"],
    })
    out = build_weather_slate(2026, 1, schedule=schedule)
    assert out.loc[0, "home"] == "LAR"
    assert out.loc[0, "away"] == "SF"
